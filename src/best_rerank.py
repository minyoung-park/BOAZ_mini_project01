"""중간점검 베스트 추천.

실험(B0→M3→C1→X1→P1, N=500)에서 순위 성능이 가장 높았던 설정만 고정한다.

    점수 = z(BPR) + λ_ctx * z(X1 프로필)
    λ_ctx = 1.0
    X1 프로필 = C1 연속 맞춤
               + Highly niche는 인기곡 항 유지
               + Discovery는 인기곡 일괄 하향 금지

넣지 않는 것:
    M1  인기 하드 규칙 — validation이 λ=0을 고름
    M2  아티스트 하드 규칙 — 팬형은 올리지만 탐색형을 깨뜨림
    P1  제목 Artist Lift — 목적 칸 NDCG를 내림 (구성만 갈라짐)

새 학습은 하지 않는다. 기존 BPR-MF 체크포인트의 후보를 재랭킹한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .bpr_mf import BPRMF, load_mpd
from .context_rerank import (
    build_profiles,
    combine_scores,
    evaluate_model,
    format_table,
    load_playlist_context,
    ndcg_of,
    profile_fit_x1,
    rank_targets,
    resolve_device,
    same_artist_matrix,
    slice_masks,
    top1_mask,
    topn_candidates,
    train_item_popularity,
    zscore_1d,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEST_LAMBDA_CTX = 1.0
BEST_CANDIDATE_N = 500
BEST_TOP_K = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description="실험에서 이긴 X1 재랭킹만 고정해 중간점검 베스트 추천을 만듭니다."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--candidate-n", type=int, default=BEST_CANDIDATE_N)
    parser.add_argument("--top-k", type=int, default=BEST_TOP_K)
    parser.add_argument("--lambda-ctx", type=float, default=BEST_LAMBDA_CTX)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="결과 JSON. 기본은 체크포인트 옆 {stem}_best_rerank_n{N}.json",
    )
    parser.add_argument(
        "--write-topk",
        action="store_true",
        help="플리별 top-K track_uri를 같은 폴더의 JSONL로 저장합니다.",
    )
    return parser.parse_args()


def x1_scores(bpr: np.ndarray, ctx: np.ndarray, lam_ctx: float) -> np.ndarray:
    zeros = np.zeros_like(bpr)
    return combine_scores(bpr, zeros, zeros, 0.0, 0.0, ctx, lam_ctx)


def metric_delta(baseline: dict, best: dict, k: int) -> dict:
    rows = {}
    for name in ("All", "Q1", "Q5", "highly_niche", "discovery", "centric", "diverse"):
        b0 = baseline.get(name) or {}
        b1 = best.get(name) or {}
        if not b0 or not b1 or b0.get("n", 0) == 0:
            continue
        ndcg_key = f"ndcg@{k}"
        hit_key = f"hit@{k}"
        rows[name] = {
            "n": int(b1["n"]),
            "ndcg_b0": b0.get(ndcg_key),
            "ndcg_best": b1.get(ndcg_key),
            "ndcg_ratio": (
                None
                if not b0.get(ndcg_key)
                else float(b1[ndcg_key]) / float(b0[ndcg_key])
            ),
            "hit_b0": b0.get(hit_key),
            "hit_best": b1.get(hit_key),
            "rec_top1_b0": b0.get("rec_top1_ratio"),
            "rec_top1_best": b1.get("rec_top1_ratio"),
        }
    return rows


def write_topk_jsonl(
    path: Path,
    playlist_ids: list[int],
    track_uris: list[str],
    ranked: np.ndarray,
    scores: np.ndarray,
    k: int,
):
    order = np.argsort(-scores, axis=1, kind="mergesort")
    ranked_scores = np.take_along_axis(scores, order, axis=1)
    with path.open("w", encoding="utf-8") as file:
        for user, pid in enumerate(playlist_ids):
            recs = [
                {
                    "rank": rank,
                    "track_uri": track_uris[int(item)],
                    "score": float(ranked_scores[user, rank - 1]),
                }
                for rank, item in enumerate(ranked[user, :k], start=1)
            ]
            file.write(json.dumps({"pid": int(pid), "items": recs}, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    playlist_ids = list(checkpoint["playlist_ids"])
    track_uris = list(checkpoint["track_uris"])
    config = checkpoint.get("config", {})
    max_slices = config.get("max_slices") or None
    factors = int(
        config.get("factors", checkpoint["model_state_dict"]["user_embedding.weight"].shape[1])
    )
    print(
        f"BEST-X1 λ_ctx={args.lambda_ctx} candidate_n={args.candidate_n} "
        f"users={len(playlist_ids):,} items={len(track_uris):,} device={device}"
    )

    train_history, validation, test, loaded_ids, loaded_uris = load_mpd(
        args.data_dir, max_slices, None
    )
    if loaded_ids != playlist_ids:
        raise ValueError("체크포인트의 playlist_ids와 현재 데이터가 다릅니다.")
    if loaded_uris != track_uris:
        raise ValueError("체크포인트의 track_uris와 현재 데이터가 다릅니다.")

    _names, track_artist = load_playlist_context(args.data_dir, playlist_ids, max_slices)
    artist_to_id: dict[str, int] = {"": -1}
    item_artists = np.full(len(track_uris), -1, dtype=np.int32)
    for item_id, uri in enumerate(track_uris):
        artist_uri = track_artist.get(uri, "")
        if artist_uri not in artist_to_id:
            artist_to_id[artist_uri] = len(artist_to_id)
        item_artists[item_id] = artist_to_id[artist_uri]

    n_users = len(train_history)
    n_items = len(track_uris)
    pop = train_item_popularity(train_history, n_items)
    is_top1 = top1_mask(pop)
    profiles = build_profiles(train_history, item_artists, pop, is_top1)
    masks, type_thresholds = slice_masks(profiles, None, None)
    for name, mask in masks.items():
        print(f"slice {name}: n={int(mask.sum()):,}")

    model = BPRMF(n_users, n_items, factors)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    user_weight = model.user_embedding.weight.detach()
    item_weight = model.item_embedding.weight.detach()

    test_seen = [
        np.concatenate([items, np.asarray([val], dtype=np.int64)])
        for items, val in zip(train_history, validation)
    ]
    print("scoring test candidates...")
    test_cand, test_bpr = topn_candidates(
        user_weight, item_weight, test_seen, args.candidate_n, device, args.score_batch_size
    )
    test_same = same_artist_matrix(test_cand, item_artists, profiles["seen_artists"])
    test_x1 = profile_fit_x1(
        test_cand,
        profiles,
        test_same,
        masks["highly_niche"],
        masks["discovery"],
    )
    spread_z = zscore_1d(profiles["popularity_spread"])
    spread_gate = 1.0 - 1.0 / (1.0 + np.exp(-np.clip(spread_z, -20.0, 20.0)))
    print(
        "X1 spread-gate mean: "
        f"highly_niche={float(spread_gate[masks['highly_niche']].mean()):.3f}  "
        f"discovery={float(spread_gate[masks['discovery']].mean()):.3f}  "
        "(override: niche=1, discovery=0)"
    )

    k = args.top_k
    dummy = np.zeros_like(test_bpr)
    common = dict(
        item_artists=item_artists,
        is_top1=is_top1,
        logpop=profiles["logpop"],
        seen_artists=profiles["seen_artists"],
        masks=masks,
        k=k,
    )
    b0_test = evaluate_model(test_cand, test_bpr, dummy, dummy, test, 0.0, 0.0, **common)
    best_test = evaluate_model(
        test_cand,
        test_bpr,
        dummy,
        dummy,
        test,
        0.0,
        0.0,
        ctx=test_x1,
        lam_ctx=args.lambda_ctx,
        **common,
    )
    models = {"B0": b0_test, "BEST": best_test}
    table = format_table(models, k)
    print("\n" + table)

    delta = metric_delta(b0_test, best_test, k)
    all_block = delta.get("All") or {}
    print(
        f"\nAll NDCG@{k}: {all_block.get('ndcg_b0')} → {all_block.get('ndcg_best')}  "
        f"ratio={all_block.get('ndcg_ratio')}"
    )

    best_scores_mat = x1_scores(test_bpr, test_x1, args.lambda_ctx)
    ranked, _, _ = rank_targets(test_cand, best_scores_mat, test)

    output = {
        "model": "BEST-X1",
        "why": (
            "1차 실험에서 All NDCG가 가장 높았던 설정. "
            "C1 연속 프로필 + Discovery에 인기곡 일괄↓ 금지. "
            "P1은 순위가 떨어져 제외."
        ),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "candidate_n": args.candidate_n,
        "top_k": k,
        "lambda_ctx": args.lambda_ctx,
        "discovery_thresholds": type_thresholds,
        "sample_eval": checkpoint.get("test_metrics"),
        "delta": delta,
        "test": models,
        "table": table,
    }
    output_path = args.output or args.checkpoint.with_name(
        f"{args.checkpoint.stem}_best_rerank_n{args.candidate_n}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={output_path}")

    if args.write_topk:
        topk_path = output_path.with_name(output_path.stem + "_topk.jsonl")
        write_topk_jsonl(topk_path, playlist_ids, track_uris, ranked, best_scores_mat, k)
        print(f"topk={topk_path}")


if __name__ == "__main__":
    main()
