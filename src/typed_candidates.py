"""유형별로 1단계 후보를 다르게 뽑은 뒤 X1로 재랭킹한다.

BEST-X1은 BPR 상위 N만 줄 세운다. 이 실험은 N을 전 플리에 늘리지 않고,
슬롯 일부를 유형에 맞는 곡으로 갈아 끼운다.

    팬형(centric): 아는 가수의 아직 안 넣은 곡
    탐색형(diverse): 이미 있는 가수 여러 명을 나눠 커버
    그 외 / Discovery만: BPR 상위 N 유지

    --centric-only 이면 탐색형은 넣지 않고 팬형만 후보를 바꾼다.

학습은 하지 않는다. 기존 BPR 체크포인트 + X1(λ=1).
정답·채점은 BEST와 같다 (다음 1곡, NDCG@10).
"""

from __future__ import annotations

import argparse
import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

try:
    from .bpr_mf import BPRMF, load_mpd
except ImportError:
    from .bpr_mf import BPRMF as BPRMF, load_mpd as load_mpd

try:
    from .context_rerank import (
        build_profiles,
        combine_scores,
        evaluate_model,
        format_table,
        load_playlist_context,
        profile_fit_x1,
        rank_targets,
        resolve_device,
        same_artist_matrix,
        slice_masks,
        top1_mask,
        topn_candidates,
        train_item_popularity,
    )
except ImportError:
    from .context_rerank import (
        build_profiles as build_profiles,
        combine_scores as combine_scores,
        evaluate_model as evaluate_model,
        format_table as format_table,
        load_playlist_context as load_playlist_context,
        profile_fit_x1 as profile_fit_x1,
        rank_targets as rank_targets,
        resolve_device as resolve_device,
        same_artist_matrix as same_artist_matrix,
        slice_masks as slice_masks,
        top1_mask as top1_mask,
        topn_candidates as topn_candidates,
        train_item_popularity as train_item_popularity,
    )

BPRMF = BPRMF
load_mpd = load_mpd
resolve_device = resolve_device
load_playlist_context = load_playlist_context
train_item_popularity = train_item_popularity
top1_mask = top1_mask
build_profiles = build_profiles
slice_masks = slice_masks
topn_candidates = topn_candidates
same_artist_matrix = same_artist_matrix
profile_fit_x1 = profile_fit_x1
evaluate_model = evaluate_model
format_table = format_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEST_LAMBDA_CTX = 1.0
BEST_CANDIDATE_N = 500
BEST_TOP_K = 10
DEFAULT_INJECT = 80
DIVERSE_ARTISTS = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description="팬형·탐색형만 후보 소스를 바꿔 X1 재랭킹합니다."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", "--data-dir", dest="data_dir", type=Path, required=True)
    parser.add_argument("--candidate-n", dest="candidate_n", type=int, default=BEST_CANDIDATE_N)
    parser.add_argument("--top-k", dest="top_k", type=int, default=BEST_TOP_K)
    parser.add_argument("--lambda-ctx", dest="lambda_ctx", type=float, default=BEST_LAMBDA_CTX)
    parser.add_argument(
        "--inject-k",
        dest="inject_k",
        type=int,
        default=DEFAULT_INJECT,
        help="유형 플리에서 BPR 하위 몇 칸을 유형 후보로 바꿀지",
    )
    parser.add_argument(
        "--score-batch-size",
        "--score-batch-size",
        dest="score_batch_size",
        type=int,
        default=256,
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--centric-only",
        dest="centric_only",
        action="store_true",
        help="탐색형은 그대로 두고 팬형만 후보를 바꾼다",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def pick(mapping, *keys):
    for key in keys:
        if isinstance(mapping, dict) and key in mapping:
            return mapping[key]
    raise KeyError(keys)


def artist_catalog(item_artists: np.ndarray) -> list[np.ndarray]:
    n_artists = int(item_artists.max()) + 1 if item_artists.size else 1
    buckets: list[list[int]] = [[] for _ in range(n_artists)]
    for item, artist in enumerate(item_artists.tolist()):
        if artist >= 0:
            buckets[artist].append(item)
    return [np.asarray(rows, dtype=np.int64) for rows in buckets]


def ranked_artists(history: np.ndarray, item_artists: np.ndarray) -> list[int]:
    if history.size == 0:
        return []
    counts = Counter(int(artist) for artist in item_artists[history] if int(artist) >= 0)
    return [artist for artist, _ in counts.most_common()]


def score_user_items(
    user_vec: np.ndarray,
    item_weight: np.ndarray,
    items: np.ndarray,
) -> np.ndarray:
    if items.size == 0:
        return np.zeros(0, dtype=np.float32)
    return (item_weight[items] @ user_vec).astype(np.float32)


def pick_from_artists(
    user_vec: np.ndarray,
    item_weight: np.ndarray,
    catalog: list[np.ndarray],
    artists: list[int],
    blocked: set[int],
    take: int,
    cover: bool,
) -> np.ndarray:
    if take <= 0 or not artists:
        return np.zeros(0, dtype=np.int64)
    if not cover:
        pool = catalog[artists[0]] if artists[0] < len(catalog) else np.zeros(0, dtype=np.int64)
        pool = np.asarray([item for item in pool.tolist() if item not in blocked], dtype=np.int64)
        scores = score_user_items(user_vec, item_weight, pool)
        if scores.size == 0:
            return pool
        order = np.argsort(-scores, kind="mergesort")
        return pool[order[:take]]

    per = max(1, take // max(len(artists), 1))
    chosen: list[int] = []
    for artist in artists:
        if artist < 0 or artist >= len(catalog):
            continue
        pool = np.asarray(
            [item for item in catalog[artist].tolist() if item not in blocked],
            dtype=np.int64,
        )
        scores = score_user_items(user_vec, item_weight, pool)
        if scores.size == 0:
            continue
        order = np.argsort(-scores, kind="mergesort")
        for item in pool[order[:per]].tolist():
            if item not in blocked:
                chosen.append(int(item))
                blocked.add(int(item))
            if len(chosen) >= take:
                return np.asarray(chosen[:take], dtype=np.int64)
    return np.asarray(chosen[:take], dtype=np.int64)


def mix_pool(
    bpr_cand: np.ndarray,
    bpr_scores: np.ndarray,
    extras: list[np.ndarray],
    user_weight: np.ndarray,
    item_weight: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    n_users, width = bpr_cand.shape
    take = min(n, width)
    out_c = np.zeros((n_users, take), dtype=np.int32)
    out_s = np.zeros((n_users, take), dtype=np.float32)
    n_swapped = 0
    extra_slots = 0
    for user in range(n_users):
        extra = extras[user]
        if extra.size == 0:
            out_c[user] = bpr_cand[user, :take]
            out_s[user] = bpr_scores[user, :take]
            continue
        n_swapped += 1
        keep = max(take - min(len(extra), take), 0)
        head = bpr_cand[user, :keep].tolist()
        head_s = bpr_scores[user, :keep].tolist()
        have = set(int(item) for item in head)
        add = [int(item) for item in extra.tolist() if int(item) not in have]
        extra_slots += len(add)
        add_s = score_user_items(
            user_weight[user], item_weight, np.asarray(add, dtype=np.int64)
        ).tolist()
        merged_i = head + add
        merged_s = head_s + add_s
        if len(merged_i) < take:
            for item, score in zip(
                bpr_cand[user, keep:].tolist(), bpr_scores[user, keep:].tolist()
            ):
                if int(item) in have:
                    continue
                merged_i.append(int(item))
                merged_s.append(float(score))
                have.add(int(item))
                if len(merged_i) >= take:
                    break
        out_c[user, : len(merged_i[:take])] = np.asarray(merged_i[:take], dtype=np.int32)
        out_s[user, : len(merged_s[:take])] = np.asarray(merged_s[:take], dtype=np.float32)
    return (
        out_c,
        out_s,
        {
            "typed_playlists": n_swapped,
            "mean_extra_kept": extra_slots / max(n_swapped, 1),
            "mean_extra_kept": extra_slots / max(n_swapped, 1),
        },
    )


artist_catalog = artist_catalog
ranked_artists = ranked_artists
mix_pool = mix_pool


def x1_scores(bpr: np.ndarray, ctx: np.ndarray, lam_ctx: float) -> np.ndarray:
    zeros = np.zeros_like(bpr)
    return combine_scores(bpr, zeros, zeros, 0.0, 0.0, ctx, lam_ctx)


def recall_row(result: dict, name: str) -> float | None:
    block = result.get(name) or {}
    if not block or block.get("n", 0) == 0:
        return None
    if "candidate_recall" in block:
        return float(block["candidate_recall"])
    if "candidate_recall" in block:
        return float(block["candidate_recall"])
    return None


recall_row = recall_row


def eval_model(candidates, bpr, dummy, dummy2, test, common, ctx=None, lam_ctx=0.0):
    params = inspect.signature(evaluate_model).parameters
    kwargs = dict(common)
    if "ctx" in params:
        kwargs["ctx"] = ctx
        kwargs["lam_ctx"] = lam_ctx
    elif "ctx" in params:
        kwargs["ctx"] = ctx
        kwargs["lam_ctx"] = lam_ctx
    return evaluate_model(candidates, bpr, dummy, dummy2, test, 0.0, 0.0, **kwargs)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    playlist_ids = list(pick(checkpoint, "playlist_ids", "playlist_ids"))
    track_uris = list(pick(checkpoint, "track_uris", "track_uris"))
    config = checkpoint.get("config") or checkpoint.get("config") or {}
    max_slices = config.get("max_slices") or config.get("max_slices") or None
    state = pick(checkpoint, "model_state_dict", "model_state_dict")
    weight_key = (
        "user_embedding.weight"
        if "user_embedding.weight" in state
        else "user_embedding.weight"
    )
    factors = int(config.get("factors") or config.get("factors") or state[weight_key].shape[1])

    train_history, validation, test, loaded_ids, loaded_uris = load_mpd(
        args.data_dir, max_slices, None
    )
    if loaded_ids != playlist_ids or loaded_uris != track_uris:
        raise ValueError("체크포인트와 data-dir이 맞지 않습니다.")

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
    centric_mask = pick(masks, "centric", "centric")
    diverse_mask = pick(masks, "diverse", "diverse")
    niche_mask = pick(masks, "highly_niche", "highly_niche")
    discovery_mask = pick(masks, "discovery", "discovery")

    model = BPRMF(n_users, n_items, factors)
    model.load_state_dict(state)
    model.eval()
    user_emb = model.user_embedding if hasattr(model, "user_embedding") else model.user_embedding
    item_emb = model.item_embedding if hasattr(model, "item_embedding") else model.item_embedding
    user_w = user_emb.weight.detach()
    item_w = item_emb.weight.detach()
    user_np = user_w.cpu().numpy()
    item_np = item_w.cpu().numpy()

    test_seen = [
        np.concatenate([items, np.asarray([val], dtype=np.int64)])
        for items, val in zip(train_history, validation)
    ]
    n = args.candidate_n
    inject = min(max(args.inject_k, 0), n)
    mode = "centric-only" if args.centric_only else "centric+diverse"
    print(
        f"TYPED-CAND n={n} inject_k={inject} mode={mode} "
        f"users={n_users:,} items={n_items:,} device={device}"
    )
    print("scoring BPR top-N...")
    bpr_cand, bpr_scores = topn_candidates(
        user_w, item_w, test_seen, n, device, args.score_batch_size
    )

    catalog = artist_catalog(item_artists)
    extras: list[np.ndarray] = []
    n_centric = n_diverse = 0
    empty = np.zeros(0, dtype=np.int64)
    for user in range(n_users):
        blocked = set(int(x) for x in test_seen[user].tolist())
        artists = ranked_artists(train_history[user], item_artists)
        if bool(centric_mask[user]):
            n_centric += 1
            extras.append(
                pick_from_artists(
                    user_np[user], item_np, catalog, artists[:1], blocked, inject, False
                )
            )
        elif bool(diverse_mask[user]) and not args.centric_only:
            n_diverse += 1
            extras.append(
                pick_from_artists(
                    user_np[user],
                    item_np,
                    catalog,
                    artists[:DIVERSE_ARTISTS],
                    blocked,
                    inject,
                    True,
                )
            )
        else:
            extras.append(empty)

    typed_cand, typed_bpr, mix_stats = mix_pool(
        bpr_cand, bpr_scores, extras, user_np, item_np, n
    )
    print(
        f"centric inject={n_centric:,}  diverse inject={n_diverse:,}  "
        f"mean extra kept={mix_stats['mean_extra_kept']:.1f}"
    )

    def x1_on(cand, scores):
        same = same_artist_matrix(
            cand, item_artists, pick(profiles, "seen_artists", "seen_artists")
        )
        return profile_fit_x1(cand, profiles, same, niche_mask, discovery_mask)

    k = args.top_k
    dummy = np.zeros_like(bpr_scores)
    logpop = pick(profiles, "logpop", "log_pop", "logpop")
    seen_artists = pick(profiles, "seen_artists", "seen_artists")
    eval_params = inspect.signature(evaluate_model).parameters
    common = dict(item_artists=item_artists, masks=masks, k=k)
    if "is_top1" in eval_params:
        common["is_top1"] = is_top1
    else:
        common["is_top1"] = is_top1
    if "logpop" in eval_params:
        common["logpop"] = logpop
    elif "log_pop" in eval_params:
        common["log_pop"] = logpop
    else:
        common["logpop"] = logpop
    if "seen_artists" in eval_params:
        common["seen_artists"] = seen_artists
    else:
        common["seen_artists"] = seen_artists

    b0 = eval_model(bpr_cand, bpr_scores, dummy, dummy, test, common)
    best = eval_model(
        bpr_cand, bpr_scores, dummy, dummy, test, common, ctx=x1_on(bpr_cand, bpr_scores), lam_ctx=args.lambda_ctx
    )
    dummy_t = np.zeros_like(typed_bpr)
    typed_b0 = eval_model(typed_cand, typed_bpr, dummy_t, dummy_t, test, common)
    typed_x1 = eval_model(
        typed_cand,
        typed_bpr,
        dummy_t,
        dummy_t,
        test,
        common,
        ctx=x1_on(typed_cand, typed_bpr),
        lam_ctx=args.lambda_ctx,
    )
    if args.centric_only:
        models = {
            "B0": b0,
            "BEST-X1": best,
            "FAN-B0": typed_b0,
            "FAN-X1": typed_x1,
        }
        model_name = "FAN-X1"
        why = (
            "BPR 상위 N을 전 플리에 늘리지 않는다. "
            "팬형만 아는 가수 미수록곡으로 하위 칸을 교체한 뒤 X1. 탐색형은 BEST-X1과 같은 BPR 500."
        )
        suffix = f"_typed_cand_n{n}_inj{inject}_centric.json"
    else:
        models = {
            "B0": b0,
            "BEST-X1": best,
            "TYPED-B0": typed_b0,
            "TYPED-X1": typed_x1,
        }
        model_name = "TYPED-X1"
        why = (
            "BPR 상위 N을 전 플리에 늘리지 않는다. "
            "팬형은 아는 가수 미수록곡, 탐색형은 기존 가수 커버로 하위 칸을 교체한 뒤 X1."
        )
        suffix = f"_typed_cand_n{n}_inj{inject}.json"

    table = format_table(models, k)
    print("\n" + table)
    print("\ncandidate_recall (정답이 후보 N 안에 있는지)")
    slice_names = (
        "All",
        "Q1",
        "centric",
        "diverse",
        "highly_niche",
        "discovery",
        "All",
        "Q1",
        "centric",
        "diverse",
        "highly_niche",
        "discovery",
    )
    seen_names = []
    for name in slice_names:
        if name in seen_names:
            continue
        seen_names.append(name)
        bits = []
        for model_label, result in models.items():
            rec = recall_row(result, name)
            if rec is None:
                continue
            bits.append(f"{model_label}={rec:.3f}")
        if bits:
            print(f"  {name}: " + "  ".join(bits))

    output = {
        "model": model_name,
        "why": why,
        "checkpoint": str(args.checkpoint),
        "candidate_n": n,
        "inject_k": inject,
        "lambda_ctx": args.lambda_ctx,
        "centric_only": bool(args.centric_only),
        "mix": {**mix_stats, "n_centric": n_centric, "n_diverse": n_diverse},
        "discovery_thresholds": type_thresholds,
        "test": models,
        "table": table,
        "candidate_recall": {
            name: {m: recall_row(r, name) for m, r in models.items()}
            for name in (
                "All",
                "Q1",
                "Q5",
                "centric",
                "diverse",
                "highly_niche",
                "discovery",
            )
        },
    }
    output_path = args.output or args.checkpoint.with_name(f"{args.checkpoint.stem}{suffix}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
