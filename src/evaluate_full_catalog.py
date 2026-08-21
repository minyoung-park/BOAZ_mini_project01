"""학습된 BPR-MF를 전체 트랙 후보와 RecSys Challenge 지표로 평가한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def sorted_slice_files(data_dir: Path):
    """MPD slice 파일을 플레이리스트 ID 순서로 반환한다."""
    return sorted(
        data_dir.glob("mpd.slice.*.json"),
        key=lambda path: int(path.name.split(".")[2].split("-")[0]),
    )


def unique_tracks(playlist: dict):
    """플레이리스트 순서를 유지하면서 중복 트랙을 제거한다."""
    result = []
    seen = set()
    for track in sorted(playlist["tracks"], key=lambda value: value["pos"]):
        uri = track["track_uri"]
        if uri not in seen:
            seen.add(uri)
            result.append(track)
    return result


def choose_user_indices(n_users: int, count: int, selection: str, seed: int):
    """평가할 체크포인트 내부 사용자 인덱스를 선택한다."""
    count = min(count, n_users)
    if selection == "first":
        return np.arange(count, dtype=np.int64)
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(n_users, size=count, replace=False))


def load_eval_cases(
    data_dir: Path,
    selected_indices: np.ndarray,
    playlist_ids: list[int],
    track_to_id: dict[str, int],
    withheld: int,
):
    """선택된 플레이리스트의 관찰 트랙과 마지막 정답 트랙을 복원한다."""
    pid_to_user = {int(playlist_ids[index]): int(index) for index in selected_indices}
    cases: dict[int, dict] = {}
    known_artists: dict[str, str] = {}

    for path in sorted_slice_files(data_dir):
        with path.open(encoding="utf-8") as file:
            playlists = json.load(file)["playlists"]
        for playlist in playlists:
            pid = int(playlist["pid"])
            if pid not in pid_to_user:
                continue

            tracks = unique_tracks(playlist)
            mapped = [track for track in tracks if track["track_uri"] in track_to_id]
            if len(mapped) <= withheld:
                continue
            for track in mapped:
                known_artists[track["track_uri"]] = track["artist_uri"]

            seed_tracks = mapped[:-withheld]
            target_tracks = mapped[-withheld:]
            cases[pid_to_user[pid]] = {
                "pid": pid,
                "seen": np.asarray(
                    [track_to_id[track["track_uri"]] for track in seed_tracks],
                    dtype=np.int64,
                ),
                "targets": np.asarray(
                    [track_to_id[track["track_uri"]] for track in target_tracks],
                    dtype=np.int64,
                ),
                "target_artists": {
                    track["artist_uri"] for track in target_tracks
                },
            }
        if len(cases) == len(pid_to_user):
            break

    missing = set(map(int, selected_indices)) - set(cases)
    if missing:
        raise ValueError(f"평가용 플레이리스트 {len(missing)}개를 복원하지 못했습니다.")
    return [cases[int(index)] for index in selected_indices], known_artists


@torch.no_grad()
def full_catalog_topk(
    user_weights: torch.Tensor,
    item_weights: torch.Tensor,
    user_indices: np.ndarray,
    cases: list[dict],
    device: torch.device,
    top_k: int,
    user_batch_size: int,
    item_chunk_size: int,
):
    """전체 아이템을 블록 단위로 점수화해 사용자별 top-k를 찾는다."""
    item_factors = item_weights.to(device)
    recommendations = []

    for user_start in range(0, len(user_indices), user_batch_size):
        user_end = min(user_start + user_batch_size, len(user_indices))
        batch_indices = torch.from_numpy(user_indices[user_start:user_end]).long()
        user_factors = user_weights[batch_indices].to(device)
        batch_cases = cases[user_start:user_end]
        batch_size = len(batch_cases)
        best_scores = torch.full((batch_size, top_k), -torch.inf, device=device)
        best_items = torch.full(
            (batch_size, top_k), -1, dtype=torch.long, device=device
        )

        for item_start in range(0, len(item_factors), item_chunk_size):
            item_end = min(item_start + item_chunk_size, len(item_factors))
            scores = user_factors @ item_factors[item_start:item_end].T

            # 이미 관찰한 트랙은 추천 후보에서 제외한다.
            for row, case in enumerate(batch_cases):
                seen = case["seen"]
                local = seen[(seen >= item_start) & (seen < item_end)] - item_start
                if len(local):
                    scores[row, torch.from_numpy(local).to(device)] = -torch.inf

            chunk_k = min(top_k, item_end - item_start)
            chunk_scores, chunk_items = torch.topk(scores, chunk_k, dim=1)
            chunk_items += item_start
            merged_scores = torch.cat((best_scores, chunk_scores), dim=1)
            merged_items = torch.cat((best_items, chunk_items), dim=1)
            best_scores, positions = torch.topk(merged_scores, top_k, dim=1)
            best_items = torch.gather(merged_items, 1, positions)

        recommendations.append(best_items.cpu().numpy())
        print(f"full-ranking {user_end:,}/{len(user_indices):,}")

    return np.concatenate(recommendations)


def resolve_recommended_artists(
    data_dir: Path,
    needed_uris: set[str],
    known_artists: dict[str, str],
):
    """R-precision의 아티스트 부분 점수에 필요한 아티스트 URI를 찾는다."""
    unresolved = needed_uris - set(known_artists)
    if not unresolved:
        return known_artists

    for path in sorted_slice_files(data_dir):
        with path.open(encoding="utf-8") as file:
            playlists = json.load(file)["playlists"]
        for playlist in playlists:
            for track in playlist["tracks"]:
                uri = track["track_uri"]
                if uri in unresolved:
                    known_artists[uri] = track["artist_uri"]
                    unresolved.remove(uri)
            if not unresolved:
                return known_artists
    return known_artists


def playlist_metrics(
    recommended: np.ndarray,
    targets: np.ndarray,
    target_artists: set[str],
    id_to_track: list[str],
    track_artists: dict[str, str],
):
    """논문의 R-precision, NDCG, clicks와 보조 지표를 계산한다."""
    target_set = set(map(int, targets))
    r = len(target_set)

    # 정확한 트랙은 1점, 트랙이 틀렸지만 정답 아티스트면 0.25점을 준다.
    r_score = 0.0
    for item in recommended[:r]:
        item = int(item)
        if item in target_set:
            r_score += 1.0
        elif track_artists.get(id_to_track[item]) in target_artists:
            r_score += 0.25
    r_precision = r_score / r

    relevance = np.fromiter(
        (int(int(item) in target_set) for item in recommended), dtype=np.float64
    )
    discounts = 1.0 / np.log2(np.arange(2, len(recommended) + 2))
    dcg = float(np.sum(relevance * discounts))
    ideal_hits = min(r, len(recommended))
    idcg = float(np.sum(discounts[:ideal_hits]))
    ndcg = dcg / idcg if idcg else 0.0

    relevant_positions = np.flatnonzero(relevance)
    clicks = int(relevant_positions[0] // 10) if len(relevant_positions) else 51
    recall = float(relevance.sum() / r)
    mrr = float(1.0 / (relevant_positions[0] + 1)) if len(relevant_positions) else 0.0
    return r_precision, ndcg, clicks, recall, mrr


def parse_args():
    parser = argparse.ArgumentParser(
        description="전체 카탈로그에서 RecSys Challenge 방식으로 BPR-MF를 평가합니다."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/full_catalog_metrics.json")
    )
    parser.add_argument("--num-playlists", type=int, default=1000)
    parser.add_argument("--selection", choices=["first", "random"], default="random")
    parser.add_argument("--withheld", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--user-batch-size", type=int, default=32)
    parser.add_argument("--item-chunk-size", type=int, default=100000)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.withheld != 2:
        raise ValueError(
            "현재 체크포인트는 마지막 2곡만 학습에서 제외했습니다. "
            "누수 없는 평가를 위해 --withheld 2를 사용하세요."
        )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 장치를 찾지 못했습니다.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    user_weights = state["user_embedding.weight"]
    item_weights = state["item_embedding.weight"]
    playlist_ids = checkpoint["playlist_ids"]
    id_to_track = checkpoint["track_uris"]
    track_to_id = {uri: index for index, uri in enumerate(id_to_track)}
    selected = choose_user_indices(
        len(playlist_ids), args.num_playlists, args.selection, args.seed
    )
    print(
        f"device={device} playlists={len(selected):,} items={len(id_to_track):,} "
        f"top_k={args.top_k}"
    )

    cases, known_artists = load_eval_cases(
        args.data_dir, selected, playlist_ids, track_to_id, args.withheld
    )
    recommendations = full_catalog_topk(
        user_weights,
        item_weights,
        selected,
        cases,
        device,
        args.top_k,
        args.user_batch_size,
        args.item_chunk_size,
    )

    r_items = {
        int(item)
        for row, case in zip(recommendations, cases)
        for item in row[: len(case["targets"])]
    }
    needed_uris = {id_to_track[item] for item in r_items}
    track_artists = resolve_recommended_artists(
        args.data_dir, needed_uris, known_artists
    )

    values = [
        playlist_metrics(
            row,
            case["targets"],
            case["target_artists"],
            id_to_track,
            track_artists,
        )
        for row, case in zip(recommendations, cases)
    ]
    array = np.asarray(values)
    metrics = {
        "r_precision": float(array[:, 0].mean()),
        f"ndcg@{args.top_k}": float(array[:, 1].mean()),
        "recommended_songs_clicks": float(array[:, 2].mean()),
        f"recall@{args.top_k}": float(array[:, 3].mean()),
        f"mrr@{args.top_k}": float(array[:, 4].mean()),
        "playlists": len(cases),
        "catalog_items": len(id_to_track),
        "withheld_per_playlist": args.withheld,
        "protocol": "full_catalog_leave_two_out",
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
