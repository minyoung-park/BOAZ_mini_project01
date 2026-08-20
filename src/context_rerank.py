"""BPR-MF 후보를 Popularity / Artist / Purpose 프로필로 재랭킹한다.

`gpt.md`의 B0 → M3 → T0 → C1 → X1 → P1.
체크포인트가 있으면 재학습하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .bpr_mf import BPRMF, load_mpd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAMBDA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
MIN_OVERALL_RATIO = 0.98
CENTRIC_SHARE = 0.50
DIVERSE_SHARE = 0.30
DIVERSE_UNIQUENESS_Q = 0.60
MIN_SLICE_N = 200
DISCOVERY_WEIGHTS = {
    "track_novelty": 0.35,
    "artist_novelty": 0.25,
    "unique_artist_ratio": 0.25,
    "popularity_spread": 0.15,
}
# 우현 통합안에 가깝게, 민서처럼 chill→sleep / work→study 로 과확장하지 않는다.
PURPOSE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("exercise", ("workout", "exercise", "gym", "running", "fitness", "cardio", "for exercising")),
    ("sleep", ("sleep", "bedtime", "for sleeping")),
    ("party", ("party", "club", "pregame", "for social")),
    ("driving", ("driving", "road trip", "roadtrip", "for driving")),
    ("holiday", ("christmas", "holiday", "halloween", "for the holiday")),
    ("summer", ("summer", "beach", "for the summer")),
    ("study", ("study", "studying", "homework", "for studying", "for focus")),
    ("worship", ("worship", "gospel", "for worship")),
    ("romance", ("wedding", "romance", "for a wedding")),
    ("gaming", ("gaming", "game day")),
]


def parse_args():
    parser = argparse.ArgumentParser(description="BPR 후보를 context로 재랭킹합니다.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--purpose-labels",
        type=Path,
        default=PROJECT_ROOT / "purpose_titles" / "outputs" / "all_unique_purpose_labels.csv",
    )
    parser.add_argument("--candidate-n", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="결과 JSON 경로. 기본은 체크포인트 옆 context_rerank_n{N}_x1p1.json",
    )
    return parser.parse_args()


def resolve_device(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 요청했지만 GPU를 찾지 못했습니다.")
    return torch.device(name)


def name_key(value: str) -> str:
    return str(value or "").strip().lower()


def assign_purpose_category(title: str, note: str) -> str | None:
    text = f"{name_key(title)} {name_key(note)}"
    for category, keys in PURPOSE_RULES:
        if any(key in text for key in keys):
            return category
    return None


def load_purpose_map(path: Path) -> dict[str, dict]:
    if not path.is_file():
        print(f"purpose labels missing: {path}")
        return {}
    mapping: dict[str, dict] = {}
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = row.get("name_key") or name_key(row.get("name", ""))
            has_purpose = row.get("has_purpose", "")
            confidence = row.get("confidence", "")
            note = row.get("purpose_note") or ""
            title = row.get("name") or key
            if has_purpose == "yes" and confidence == "high":
                mapping[key] = {
                    "flag": "purpose",
                    "category": assign_purpose_category(title, note),
                }
            elif has_purpose == "no":
                mapping[key] = {"flag": "non_purpose", "category": None}
    print(f"purpose labels: {path} keys={len(mapping):,}")
    return mapping


def load_playlist_context(data_dir: Path, playlist_ids: list[int], max_slices: int | None):
    """학습에 쓰인 플리의 제목과 track→artist 매핑을 읽는다."""
    files = sorted(
        data_dir.glob("mpd.slice.*.json"),
        key=lambda path: int(path.name.split(".")[2].split("-")[0]),
    )
    if max_slices is not None:
        files = files[:max_slices]

    wanted = set(playlist_ids)
    names: dict[int, str] = {}
    track_artist: dict[str, str] = {}
    for path in files:
        with path.open(encoding="utf-8") as file:
            playlists = json.load(file)["playlists"]
        for playlist in playlists:
            pid = int(playlist["pid"])
            if pid not in wanted:
                continue
            names[pid] = playlist.get("name") or ""
            for track in playlist.get("tracks", []):
                uri = track.get("track_uri")
                artist = track.get("artist_uri") or ""
                if uri and uri not in track_artist:
                    track_artist[uri] = artist
            if len(names) >= len(wanted):
                return names, track_artist
    return names, track_artist


def train_item_popularity(histories: list[np.ndarray], n_items: int) -> np.ndarray:
    pop = np.zeros(n_items, dtype=np.int32)
    for items in histories:
        if len(items) == 0:
            continue
        pop[np.unique(items)] += 1
    return pop


def top1_mask(pop: np.ndarray) -> np.ndarray:
    n_items = len(pop)
    order = np.argsort(-pop, kind="mergesort")
    rank = np.empty(n_items, dtype=np.int32)
    rank[order] = np.arange(1, n_items + 1)
    return (rank / n_items) <= 0.01


def zscore_1d(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    mean = float(finite.mean()) if finite.size else 0.0
    std = float(finite.std()) if finite.size else 1.0
    if std < 1e-8:
        std = 1.0
    out = (values - mean) / std
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = (np.arange(len(values)) + 0.5) / len(values)
    return ranks


def build_profiles(
    histories: list[np.ndarray],
    item_artists: np.ndarray,
    pop: np.ndarray,
    is_top1: np.ndarray,
):
    n_users = len(histories)
    n_items = len(pop)
    logpop = np.log1p(pop.astype(np.float32))
    n_playlists = max(n_users, 2)
    track_novelty = np.log(n_playlists / np.maximum(pop, 1)) / np.log(n_playlists)
    track_novelty = np.clip(track_novelty, 0.0, 1.0).astype(np.float32)

    n_artists = int(item_artists.max()) + 1 if item_artists.size else 1
    artist_pl = np.zeros(max(n_artists, 1), dtype=np.int32)
    for items in histories:
        if len(items) == 0:
            continue
        artists = np.unique(item_artists[np.unique(items)])
        artists = artists[artists >= 0]
        if artists.size:
            artist_pl[artists] += 1
    artist_novelty = np.log(n_playlists / np.maximum(artist_pl, 1)) / np.log(n_playlists)
    artist_novelty = np.clip(artist_novelty, 0.0, 1.0).astype(np.float32)

    top1_ratio = np.zeros(n_users, dtype=np.float32)
    avg_logpop = np.zeros(n_users, dtype=np.float32)
    avg_pop_frac = np.zeros(n_users, dtype=np.float32)
    unique_artist_ratio = np.zeros(n_users, dtype=np.float32)
    top_artist_share = np.zeros(n_users, dtype=np.float32)
    popularity_spread = np.zeros(n_users, dtype=np.float32)
    mean_track_novelty = np.zeros(n_users, dtype=np.float32)
    mean_artist_novelty = np.zeros(n_users, dtype=np.float32)
    seen_artists: list[set[int]] = []
    pop_frac = pop.astype(np.float32) / n_playlists

    for user, items in enumerate(histories):
        if len(items) == 0:
            seen_artists.append(set())
            continue
        unique_items = np.unique(items)
        top1_ratio[user] = float(is_top1[unique_items].mean())
        avg_logpop[user] = float(logpop[unique_items].mean())
        avg_pop_frac[user] = float(pop_frac[unique_items].mean())
        popularity_spread[user] = (
            float(pop_frac[unique_items].std()) if unique_items.size > 1 else 0.0
        )
        mean_track_novelty[user] = float(track_novelty[unique_items].mean())
        artists = item_artists[unique_items]
        valid = artists >= 0
        artist_ids = artists[valid]
        if artist_ids.size == 0:
            seen_artists.append(set())
            continue
        counts = Counter(int(a) for a in artist_ids)
        n = len(artist_ids)
        unique_artist_ratio[user] = len(counts) / n
        top_artist_share[user] = max(counts.values()) / n
        mean_artist_novelty[user] = float(artist_novelty[artist_ids].mean())
        seen_artists.append(set(counts))

    discovery = (
        DISCOVERY_WEIGHTS["track_novelty"] * percentile_rank(mean_track_novelty)
        + DISCOVERY_WEIGHTS["artist_novelty"] * percentile_rank(mean_artist_novelty)
        + DISCOVERY_WEIGHTS["unique_artist_ratio"] * percentile_rank(unique_artist_ratio)
        + DISCOVERY_WEIGHTS["popularity_spread"] * percentile_rank(popularity_spread)
    )
    return {
        "top1_ratio": top1_ratio,
        "avg_logpop": avg_logpop,
        "avg_pop_frac": avg_pop_frac,
        "unique_artist_ratio": unique_artist_ratio,
        "top_artist_share": top_artist_share,
        "popularity_spread": popularity_spread,
        "discovery_score": discovery.astype(np.float32),
        "seen_artists": seen_artists,
        "logpop": logpop,
        "n_items": n_items,
    }


def discovery_type_masks(profiles: dict):
    """현우 노트북과 같은 33/67 분위 규칙. 성분은 현재 실험 train 곡으로 다시 계산한다."""
    discovery = profiles["discovery_score"]
    avg_pop = profiles["avg_pop_frac"]
    d_low, d_high = np.quantile(discovery, [1 / 3, 2 / 3])
    p_low, p_high = np.quantile(avg_pop, [1 / 3, 2 / 3])
    mainstream = (avg_pop >= p_high) & (discovery <= d_low)
    highly_niche = (avg_pop <= p_low) & (discovery >= d_high)
    discovery_oriented = (discovery >= d_high) & (~highly_niche)
    balanced = ~(mainstream | highly_niche | discovery_oriented)
    return {
        "highly_niche": highly_niche,
        "discovery": discovery_oriented,
        "mainstream": mainstream,
        "balanced": balanced,
        "thresholds": {
            "discovery_33": float(d_low),
            "discovery_67": float(d_high),
            "popularity_33": float(p_low),
            "popularity_67": float(p_high),
        },
    }


def quintile_masks(values: np.ndarray):
    finite = values[np.isfinite(values)]
    edges = np.quantile(finite, [0.2, 0.4, 0.6, 0.8])
    q1 = values <= edges[0]
    q5 = values > edges[3]
    return q1, q5


def slice_masks(
    profiles: dict,
    purpose_flags: np.ndarray | None,
    purpose_cats: list[str | None] | None = None,
):
    top1 = profiles["top1_ratio"]
    share = profiles["top_artist_share"]
    uniq = profiles["unique_artist_ratio"]
    q1, q5 = quintile_masks(top1)
    centric = share >= CENTRIC_SHARE
    uniq_cut = np.quantile(uniq[np.isfinite(uniq)], DIVERSE_UNIQUENESS_Q)
    diverse = (share < DIVERSE_SHARE) & (uniq >= uniq_cut)
    types = discovery_type_masks(profiles)
    masks = {
        "All": np.ones(len(top1), dtype=bool),
        "Q1": q1,
        "Q5": q5,
        "highly_niche": types["highly_niche"],
        "discovery": types["discovery"],
        "centric": centric,
        "diverse": diverse,
        "Q1_centric": q1 & centric,
        "Q1_diverse": q1 & diverse,
        "mainstream": types["mainstream"],
        "balanced": types["balanced"],
    }
    if purpose_flags is not None:
        masks["purpose"] = purpose_flags == 1
        masks["non_purpose"] = purpose_flags == 0
    if purpose_cats is not None:
        cats = np.asarray(purpose_cats, dtype=object)
        for category in ("exercise", "sleep"):
            masks[category] = cats == category
    return masks, types["thresholds"]


def topn_candidates(
    user_weight: torch.Tensor,
    item_weight: torch.Tensor,
    seen: list[np.ndarray],
    n: int,
    device: torch.device,
    batch_size: int,
):
    n_users = user_weight.size(0)
    n_items = item_weight.size(0)
    take = min(n, max(n_items - 1, 1))
    item_weight = item_weight.to(device)
    candidates = np.zeros((n_users, take), dtype=np.int32)
    scores = np.zeros((n_users, take), dtype=np.float32)
    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        users = user_weight[start:end].to(device)
        batch_scores = users @ item_weight.T
        for row, user in enumerate(range(start, end)):
            observed = seen[user]
            if len(observed):
                batch_scores[row, torch.as_tensor(observed, device=device, dtype=torch.long)] = -1e9
        top_scores, top_index = torch.topk(batch_scores, take, dim=1)
        candidates[start:end] = top_index.cpu().numpy()
        scores[start:end] = top_scores.cpu().numpy()
        del batch_scores
    return candidates, scores


def zscore_rows(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (values - mean) / std


def pop_fit(
    candidates: np.ndarray,
    q1: np.ndarray,
    q5: np.ndarray,
    logpop: np.ndarray,
) -> np.ndarray:
    fit = np.zeros(candidates.shape, dtype=np.float32)
    cand_logpop = logpop[candidates]
    fit[q1] = -cand_logpop[q1]
    fit[q5] = cand_logpop[q5]
    return fit


def art_fit(
    candidates: np.ndarray,
    item_artists: np.ndarray,
    seen_artists: list[set[int]],
    centric: np.ndarray,
    diverse: np.ndarray,
) -> np.ndarray:
    fit = np.zeros(candidates.shape, dtype=np.float32)
    cand_artists = item_artists[candidates]
    for user in np.flatnonzero(centric | diverse):
        known = seen_artists[user]
        artists = cand_artists[user]
        if centric[user]:
            fit[user] = np.isin(artists, list(known) if known else [-1]).astype(np.float32)
        elif diverse[user]:
            if not known:
                fit[user] = 1.0
            else:
                fit[user] = (~np.isin(artists, list(known))).astype(np.float32)
    return fit


def same_artist_matrix(
    candidates: np.ndarray,
    item_artists: np.ndarray,
    seen_artists: list[set[int]],
) -> np.ndarray:
    fit = np.zeros(candidates.shape, dtype=np.float32)
    cand_artists = item_artists[candidates]
    for user, known in enumerate(seen_artists):
        if not known:
            continue
        fit[user] = np.isin(cand_artists[user], list(known)).astype(np.float32)
    return fit


def profile_fit(
    candidates: np.ndarray,
    profiles: dict,
    same_artist: np.ndarray,
) -> np.ndarray:
    """C1: 연속 niche·centric 프로필과 곡 자질의 맞음."""
    niche = zscore_1d(-profiles["avg_logpop"])
    centric = zscore_1d(profiles["top_artist_share"])
    cand_logpop = profiles["logpop"][candidates]
    return (
        -niche[:, None] * cand_logpop + centric[:, None] * same_artist
    ).astype(np.float32)


def profile_fit_x1(
    candidates: np.ndarray,
    profiles: dict,
    same_artist: np.ndarray,
) -> np.ndarray:
    """X1: spread가 높으면(혼합형) 인기곡 일괄 하향을 줄인다. discovery_score는 곱하지 않는다."""
    niche = zscore_1d(-profiles["avg_logpop"])
    centric = zscore_1d(profiles["top_artist_share"])
    spread_z = zscore_1d(profiles["popularity_spread"])
    gate = 1.0 - 1.0 / (1.0 + np.exp(-np.clip(spread_z, -20.0, 20.0)))
    cand_logpop = profiles["logpop"][candidates]
    pop_term = -niche[:, None] * cand_logpop * gate[:, None]
    return (pop_term + centric[:, None] * same_artist).astype(np.float32)


def purpose_fit(
    candidates: np.ndarray,
    item_artists: np.ndarray,
    histories: list[np.ndarray],
    purpose_cats: list[str | None],
) -> np.ndarray:
    """P1: 다른 플리 train 곡만으로 artist-purpose lift. 대상 플리는 집계에서 뺀다."""
    n_artists = int(item_artists.max()) + 1 if item_artists.size else 1
    cat_ids = {cat: i for i, cat in enumerate(sorted({c for c in purpose_cats if c}))}
    n_cat = len(cat_ids)
    fit = np.zeros(candidates.shape, dtype=np.float32)
    if n_cat == 0:
        return fit

    count_ac = np.zeros((n_cat, n_artists), dtype=np.float64)
    count_c = np.zeros(n_cat, dtype=np.float64)
    count_a = np.zeros(n_artists, dtype=np.float64)
    total = 0.0
    user_counts: list[Counter] = []
    user_n: list[int] = []
    for user, items in enumerate(histories):
        unique_items = np.unique(items) if len(items) else np.array([], dtype=np.int64)
        artists = item_artists[unique_items]
        artists = artists[artists >= 0]
        counts = Counter(int(a) for a in artists)
        user_counts.append(counts)
        n_tracks = int(artists.size)
        user_n.append(n_tracks)
        for artist, n in counts.items():
            count_a[artist] += n
        total += n_tracks
        cat = purpose_cats[user]
        if cat is None:
            continue
        cid = cat_ids[cat]
        for artist, n in counts.items():
            count_ac[cid, artist] += n
        count_c[cid] += n_tracks

    cand_artists = item_artists[candidates]
    for user, cat in enumerate(purpose_cats):
        if cat is None:
            continue
        cid = cat_ids[cat]
        mine = user_counts[user]
        my_n = user_n[user]
        tot_c = count_c[cid] - my_n
        tot_all = total - my_n
        if tot_c <= 0 or tot_all <= 0:
            continue
        ac_row = count_ac[cid].copy()
        a_row = count_a.copy()
        for artist, n in mine.items():
            ac_row[artist] -= n
            a_row[artist] -= n
        artists = cand_artists[user]
        valid = artists >= 0
        if not valid.any():
            continue
        ids = artists[valid]
        pa_c = np.maximum(ac_row[ids], 0.0) / tot_c
        pa = np.maximum(a_row[ids], 0.0) / tot_all
        lift = np.ones(ids.size, dtype=np.float64)
        seen = pa > 0
        lift[seen] = pa_c[seen] / pa[seen]
        fit[user, valid] = np.log(np.clip(lift, 0.25, 8.0)).astype(np.float32)
    return fit


def combine_scores(
    bpr: np.ndarray,
    pop: np.ndarray,
    art: np.ndarray,
    lam_pop: float,
    lam_art: float,
    ctx: np.ndarray | None = None,
    lam_ctx: float = 0.0,
    pur: np.ndarray | None = None,
    lam_pur: float = 0.0,
):
    total = zscore_rows(bpr)
    if lam_pop:
        total = total + lam_pop * zscore_rows(pop)
    if lam_art:
        total = total + lam_art * zscore_rows(art)
    if lam_ctx and ctx is not None:
        total = total + lam_ctx * zscore_rows(ctx)
    if lam_pur and pur is not None:
        total = total + lam_pur * zscore_rows(pur)
    return total


def rank_targets(candidates: np.ndarray, scores: np.ndarray, targets: np.ndarray):
    order = np.argsort(-scores, axis=1, kind="mergesort")
    ranked = np.take_along_axis(candidates, order, axis=1)
    match = ranked == targets[:, None]
    found = match.any(axis=1)
    rank = np.where(found, match.argmax(axis=1) + 1, 0)
    return ranked, rank, found


def metric_block(
    ranked: np.ndarray,
    rank: np.ndarray,
    found: np.ndarray,
    targets: np.ndarray,
    item_artists: np.ndarray,
    is_top1: np.ndarray,
    logpop: np.ndarray,
    seen_artists: list[set[int]],
    mask: np.ndarray,
    k: int,
):
    users = np.flatnonzero(mask)
    n = len(users)
    if n == 0:
        return {"n": 0}
    hit = (found[users] & (rank[users] <= k)).mean()
    ndcg_vals = np.zeros(n, dtype=np.float64)
    in_k = found[users] & (rank[users] <= k)
    ndcg_vals[in_k] = 1.0 / np.log2(rank[users][in_k] + 1)
    top = ranked[users, :k]
    rec_top1 = is_top1[top].mean(axis=1)
    rec_logpop = logpop[top].mean(axis=1)
    rec_spread = logpop[top].std(axis=1)
    rec_new = np.zeros(n, dtype=np.float32)
    artist_hit = np.zeros(n, dtype=np.float32)
    target_artists = item_artists[targets[users]]
    for i, user in enumerate(users):
        rec_artists = item_artists[top[i]]
        known = seen_artists[user]
        rec_new[i] = np.mean([a not in known for a in rec_artists]) if known else 1.0
        artist_hit[i] = float(target_artists[i] >= 0 and target_artists[i] in set(rec_artists))
    return {
        "n": int(n),
        f"hit@{k}": float(hit),
        f"ndcg@{k}": float(ndcg_vals.mean()),
        "candidate_recall": float(found[users].mean()),
        "rec_top1_ratio": float(rec_top1.mean()),
        "rec_avg_logpop": float(rec_logpop.mean()),
        "rec_spread": float(rec_spread.mean()),
        "rec_new_artist_ratio": float(rec_new.mean()),
        "novelty": float((-rec_logpop).mean()),
        f"artist_hit@{k}": float(artist_hit.mean()),
    }


def evaluate_model(
    candidates: np.ndarray,
    bpr_scores: np.ndarray,
    pop: np.ndarray,
    art: np.ndarray,
    targets: np.ndarray,
    lam_pop: float,
    lam_art: float,
    item_artists: np.ndarray,
    is_top1: np.ndarray,
    logpop: np.ndarray,
    seen_artists: list[set[int]],
    masks: dict[str, np.ndarray],
    k: int,
    ctx: np.ndarray | None = None,
    lam_ctx: float = 0.0,
    pur: np.ndarray | None = None,
    lam_pur: float = 0.0,
):
    scores = combine_scores(
        bpr_scores, pop, art, lam_pop, lam_art, ctx, lam_ctx, pur, lam_pur
    )
    ranked, rank, found = rank_targets(candidates, scores, targets)
    return {
        name: metric_block(
            ranked, rank, found, targets, item_artists, is_top1, logpop, seen_artists, mask, k
        )
        for name, mask in masks.items()
    }


def ndcg_of(result: dict, slice_name: str, k: int) -> float:
    block = result.get(slice_name) or {}
    if block.get("n", 0) == 0:
        return float("nan")
    return float(block[f"ndcg@{k}"])


def select_lambda(
    grid_results: list[tuple[float, float, dict]],
    target_slices: list[str],
    k: int,
    b0_overall: float,
    pair_guard: tuple[str, str] | None = None,
    b0_result: dict | None = None,
):
    kept = []
    for lam_pop, lam_art, result in grid_results:
        overall = ndcg_of(result, "All", k)
        if overall < b0_overall * MIN_OVERALL_RATIO:
            continue
        if pair_guard and b0_result is not None:
            left, right = pair_guard
            d_left = ndcg_of(result, left, k) - ndcg_of(b0_result, left, k)
            d_right = ndcg_of(result, right, k) - ndcg_of(b0_result, right, k)
            base_left = max(abs(ndcg_of(b0_result, left, k)), 1e-8)
            base_right = max(abs(ndcg_of(b0_result, right, k)), 1e-8)
            broken = (d_left > 0 and d_right < -0.05 * base_right) or (
                d_right > 0 and d_left < -0.05 * base_left
            )
            if broken:
                continue
        usable = [name for name in target_slices if (result.get(name) or {}).get("n", 0) > 0]
        if not usable:
            usable = ["All"]
        target = float(np.nanmean([ndcg_of(result, name, k) for name in usable]))
        novelty = result["All"].get("novelty", 0.0)
        kept.append(((target, novelty, overall), lam_pop, lam_art, result))
    if not kept:
        best = max(grid_results, key=lambda row: ndcg_of(row[2], "All", k))
        return best[0], best[1], best[2], True
    kept.sort(key=lambda row: row[0], reverse=True)
    _, lam_pop, lam_art, result = kept[0]
    return lam_pop, lam_art, result, False


def _table_block(models: dict[str, dict], slices: list[str], k: int) -> list[str]:
    header = f"{'model':<8}" + "".join(f"{name:>22}" for name in slices)
    lines = [header]
    for model_name, result in models.items():
        cells = [f"{model_name:<8}"]
        for name in slices:
            block = result.get(name) or {}
            if name not in result:
                cells.append(f"{'—':>22}")
                continue
            if block.get("n", 0) < MIN_SLICE_N and name != "All":
                cells.append(f"{'n=' + str(block.get('n', 0)):>22}")
                continue
            ndcg = block.get(f"ndcg@{k}", float("nan"))
            if name == "discovery":
                extra = block.get("rec_spread", float("nan"))
                cells.append(f"{ndcg:.4f}/s{extra:.3f}".rjust(22))
            else:
                top1 = block.get("rec_top1_ratio", float("nan"))
                cells.append(f"{ndcg:.4f}/{top1:.3f}".rjust(22))
        lines.append("".join(cells))
    return lines


def format_table(models: dict[str, dict], k: int) -> str:
    slices = ["All", "Q1", "Q5", "highly_niche", "discovery", "centric", "diverse"]
    lines = _table_block(models, slices, k)
    extra = [
        name
        for name in ("purpose", "non_purpose", "exercise", "sleep")
        if any(name in result for result in models.values())
    ]
    if extra:
        lines.append("")
        lines.extend(_table_block(models, extra, k))
    lines.append("cell = NDCG@K / rec_top1_ratio  (discovery 열은 NDCG / rec_spread)")
    return "\n".join(lines)


def artist_mix_cosine(
    candidates: np.ndarray,
    scores: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    item_artists: np.ndarray,
    k: int,
) -> float:
    order = np.argsort(-scores, axis=1, kind="mergesort")
    ranked = np.take_along_axis(candidates, order, axis=1)
    n_artists = int(item_artists.max()) + 1 if item_artists.size else 1
    vec_a = np.zeros(n_artists, dtype=np.float64)
    vec_b = np.zeros(n_artists, dtype=np.float64)
    for mask, vec in ((mask_a, vec_a), (mask_b, vec_b)):
        for user in np.flatnonzero(mask):
            for artist in item_artists[ranked[user, :k]]:
                if artist >= 0:
                    vec[artist] += 1
    na = np.linalg.norm(vec_a)
    nb = np.linalg.norm(vec_b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(vec_a @ vec_b / (na * nb))


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    playlist_ids = list(checkpoint["playlist_ids"])
    track_uris = list(checkpoint["track_uris"])
    config = checkpoint.get("config", {})
    max_slices = config.get("max_slices")
    if not max_slices:
        max_slices = None
    factors = int(config.get("factors", checkpoint["model_state_dict"]["user_embedding.weight"].shape[1]))
    print(
        f"checkpoint={args.checkpoint} users={len(playlist_ids):,} "
        f"items={len(track_uris):,} device={device} max_slices={max_slices} "
        f"candidate_n={args.candidate_n}"
    )

    train_history, validation, test, loaded_ids, loaded_uris = load_mpd(
        args.data_dir, max_slices, None
    )
    if loaded_ids != playlist_ids:
        raise ValueError("체크포인트의 playlist_ids와 현재 데이터가 다릅니다. data-dir / max_slices를 확인하세요.")
    if loaded_uris != track_uris:
        raise ValueError("체크포인트의 track_uris와 현재 데이터가 다릅니다.")

    names, track_artist = load_playlist_context(args.data_dir, playlist_ids, max_slices)
    artist_to_id: dict[str, int] = {"": -1}
    item_artists = np.full(len(track_uris), -1, dtype=np.int32)
    for item_id, uri in enumerate(track_uris):
        artist_uri = track_artist.get(uri, "")
        if artist_uri not in artist_to_id:
            artist_to_id[artist_uri] = len(artist_to_id)
        item_artists[item_id] = artist_to_id[artist_uri]

    purpose_map = load_purpose_map(args.purpose_labels)
    purpose_flags = np.full(len(playlist_ids), -1, dtype=np.int8)
    purpose_cats: list[str | None] = [None] * len(playlist_ids)
    if purpose_map:
        for i, pid in enumerate(playlist_ids):
            info = purpose_map.get(name_key(names.get(pid, "")))
            if not info:
                continue
            if info["flag"] == "purpose":
                purpose_flags[i] = 1
                purpose_cats[i] = info.get("category")
            elif info["flag"] == "non_purpose":
                purpose_flags[i] = 0
        cat_counts = Counter(cat for cat in purpose_cats if cat)
        print("purpose categories:", dict(cat_counts))
        print(
            f"purpose axis on: {sum(c is not None for c in purpose_cats):,}  "
            f"(has_purpose high but ambiguous: "
            f"{int(((purpose_flags == 1) & np.array([c is None for c in purpose_cats])).sum()):,})"
        )

    n_users = len(train_history)
    n_items = len(track_uris)
    pop = train_item_popularity(train_history, n_items)
    is_top1 = top1_mask(pop)
    profiles = build_profiles(train_history, item_artists, pop, is_top1)
    masks, type_thresholds = slice_masks(
        profiles,
        purpose_flags if purpose_map else None,
        purpose_cats if purpose_map else None,
    )
    for name, mask in masks.items():
        print(f"slice {name}: n={int(mask.sum()):,}")
    print("discovery thresholds", type_thresholds)

    model = BPRMF(n_users, n_items, factors)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    user_weight = model.user_embedding.weight.detach()
    item_weight = model.item_embedding.weight.detach()

    val_seen = [np.asarray(items, dtype=np.int64) for items in train_history]
    test_seen = [
        np.concatenate([items, np.asarray([val], dtype=np.int64)])
        for items, val in zip(train_history, validation)
    ]
    print("scoring validation candidates...")
    val_cand, val_bpr = topn_candidates(
        user_weight, item_weight, val_seen, args.candidate_n, device, args.score_batch_size
    )
    print("scoring test candidates...")
    test_cand, test_bpr = topn_candidates(
        user_weight, item_weight, test_seen, args.candidate_n, device, args.score_batch_size
    )

    q1, q5 = masks["Q1"], masks["Q5"]
    val_pop = pop_fit(val_cand, q1, q5, profiles["logpop"])
    test_pop = pop_fit(test_cand, q1, q5, profiles["logpop"])
    val_art = art_fit(val_cand, item_artists, profiles["seen_artists"], masks["centric"], masks["diverse"])
    test_art = art_fit(test_cand, item_artists, profiles["seen_artists"], masks["centric"], masks["diverse"])
    val_same = same_artist_matrix(val_cand, item_artists, profiles["seen_artists"])
    test_same = same_artist_matrix(test_cand, item_artists, profiles["seen_artists"])
    val_ctx = profile_fit(val_cand, profiles, val_same)
    test_ctx = profile_fit(test_cand, profiles, test_same)
    val_x1 = profile_fit_x1(val_cand, profiles, val_same)
    test_x1 = profile_fit_x1(test_cand, profiles, test_same)
    n_purpose_axis = sum(c is not None for c in purpose_cats)
    if n_purpose_axis:
        print("building purpose lift (leave-one-out train artists)...")
        val_pur = purpose_fit(val_cand, item_artists, train_history, purpose_cats)
        test_pur = purpose_fit(test_cand, item_artists, train_history, purpose_cats)
    else:
        val_pur = test_pur = None
        print("P1 skipped: no mapped title purpose categories")

    k = args.top_k
    common = dict(
        item_artists=item_artists,
        is_top1=is_top1,
        logpop=profiles["logpop"],
        seen_artists=profiles["seen_artists"],
        masks=masks,
        k=k,
    )
    b0_val = evaluate_model(val_cand, val_bpr, val_pop, val_art, validation, 0.0, 0.0, **common)
    b0_test = evaluate_model(test_cand, test_bpr, test_pop, test_art, test, 0.0, 0.0, **common)
    b0_overall = ndcg_of(b0_val, "All", k)
    print(
        f"B0 val NDCG@{k}={b0_overall:.4f}  "
        f"candidate_recall={b0_val['All']['candidate_recall']:.4f}"
    )

    def sweep(lams_pop, lams_art):
        rows = []
        for lam_pop in lams_pop:
            for lam_art in lams_art:
                result = evaluate_model(
                    val_cand, val_bpr, val_pop, val_art, validation, lam_pop, lam_art, **common
                )
                rows.append((lam_pop, lam_art, result))
        return rows

    m1_pop, m1_art, _, m1_fallback = select_lambda(
        sweep(LAMBDA_GRID, [0.0]), ["Q1"], k, b0_overall
    )
    m2_pop, m2_art, _, m2_fallback = select_lambda(
        sweep([0.0], LAMBDA_GRID), ["centric"], k, b0_overall
    )
    m3_pop, m3_art, _, m3_fallback = select_lambda(
        sweep(LAMBDA_GRID, LAMBDA_GRID), ["Q1", "centric"], k, b0_overall
    )

    c1_rows = []
    for lam_ctx in LAMBDA_GRID:
        result = evaluate_model(
            val_cand,
            val_bpr,
            val_pop,
            val_art,
            validation,
            0.0,
            0.0,
            ctx=val_ctx,
            lam_ctx=lam_ctx,
            **common,
        )
        c1_rows.append((lam_ctx, 0.0, result))
    c1_ctx, _, _, c1_fallback = select_lambda(
        c1_rows, ["highly_niche", "discovery"], k, b0_overall
    )

    x1_rows = []
    for lam_ctx in LAMBDA_GRID:
        result = evaluate_model(
            val_cand,
            val_bpr,
            val_pop,
            val_art,
            validation,
            0.0,
            0.0,
            ctx=val_x1,
            lam_ctx=lam_ctx,
            **common,
        )
        x1_rows.append((lam_ctx, 0.0, result))
    x1_ctx, _, _, x1_fallback = select_lambda(
        x1_rows,
        ["highly_niche", "discovery"],
        k,
        b0_overall,
        pair_guard=("highly_niche", "discovery"),
        b0_result=b0_val,
    )

    p1_pur = 0.0
    p1_fallback = True
    if val_pur is not None:
        p1_target = (
            ["purpose"]
            if int(masks.get("purpose", np.zeros(1)).sum()) >= MIN_SLICE_N
            else ["All"]
        )
        p1_rows = []
        for lam_pur in LAMBDA_GRID:
            result = evaluate_model(
                val_cand,
                val_bpr,
                val_pop,
                val_art,
                validation,
                0.0,
                0.0,
                ctx=val_x1,
                lam_ctx=x1_ctx,
                pur=val_pur,
                lam_pur=lam_pur,
                **common,
            )
            p1_rows.append((lam_pur, 0.0, result))
        p1_pur, _, _, p1_fallback = select_lambda(p1_rows, p1_target, k, b0_overall)

    print(
        f"lambda M1=({m1_pop}, {m1_art}) fallback={m1_fallback}  "
        f"M2=({m2_pop}, {m2_art}) fallback={m2_fallback}  "
        f"M3=({m3_pop}, {m3_art}) fallback={m3_fallback}  "
        f"C1=({c1_ctx}) fallback={c1_fallback}  "
        f"X1=({x1_ctx}) fallback={x1_fallback}  "
        f"P1=({x1_ctx}, {p1_pur}) fallback={p1_fallback}"
    )

    models_test = {
        "B0": b0_test,
        "M1": evaluate_model(test_cand, test_bpr, test_pop, test_art, test, m1_pop, m1_art, **common),
        "M2": evaluate_model(test_cand, test_bpr, test_pop, test_art, test, m2_pop, m2_art, **common),
        "M3": evaluate_model(test_cand, test_bpr, test_pop, test_art, test, m3_pop, m3_art, **common),
        "C1": evaluate_model(
            test_cand,
            test_bpr,
            test_pop,
            test_art,
            test,
            0.0,
            0.0,
            ctx=test_ctx,
            lam_ctx=c1_ctx,
            **common,
        ),
        "X1": evaluate_model(
            test_cand,
            test_bpr,
            test_pop,
            test_art,
            test,
            0.0,
            0.0,
            ctx=test_x1,
            lam_ctx=x1_ctx,
            **common,
        ),
    }
    if test_pur is not None:
        models_test["P1"] = evaluate_model(
            test_cand,
            test_bpr,
            test_pop,
            test_art,
            test,
            0.0,
            0.0,
            ctx=test_x1,
            lam_ctx=x1_ctx,
            pur=test_pur,
            lam_pur=p1_pur,
            **common,
        )
    table = format_table(models_test, k)
    print("\n" + table)

    exercise_sleep = {}
    if "exercise" in masks and "sleep" in masks:
        n_ex = int(masks["exercise"].sum())
        n_sl = int(masks["sleep"].sum())
        if n_ex and n_sl:
            mix_models = {
                "B0": (None, 0.0, None, 0.0),
                "X1": (test_x1, x1_ctx, None, 0.0),
            }
            if test_pur is not None:
                mix_models["P1"] = (test_x1, x1_ctx, test_pur, p1_pur)
            for name, (ctx, lam_ctx, pur, lam_pur) in mix_models.items():
                scores = combine_scores(
                    test_bpr, test_pop, test_art, 0.0, 0.0, ctx, lam_ctx, pur, lam_pur
                )
                exercise_sleep[name] = {
                    "n_exercise": n_ex,
                    "n_sleep": n_sl,
                    "rec_artist_cosine": artist_mix_cosine(
                        test_cand,
                        scores,
                        masks["exercise"],
                        masks["sleep"],
                        item_artists,
                        k,
                    ),
                }
            print("exercise vs sleep rec artist cosine:", exercise_sleep)

    output = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "candidate_n": args.candidate_n,
        "top_k": k,
        "discovery_thresholds": type_thresholds,
        "lambda": {
            "M1": {"pop": m1_pop, "art": m1_art, "fallback": m1_fallback},
            "M2": {"pop": m2_pop, "art": m2_art, "fallback": m2_fallback},
            "M3": {"pop": m3_pop, "art": m3_art, "fallback": m3_fallback},
            "C1": {"ctx": c1_ctx, "fallback": c1_fallback},
            "X1": {"ctx": x1_ctx, "fallback": x1_fallback},
            "P1": {"ctx": x1_ctx, "pur": p1_pur, "fallback": p1_fallback},
        },
        "exercise_sleep_artist_cosine": exercise_sleep,
        "sample_eval": checkpoint.get("test_metrics"),
        "test": models_test,
        "validation_b0": b0_val,
        "table": table,
    }
    output_path = args.output or args.checkpoint.with_name(
        f"{args.checkpoint.stem}_context_rerank_n{args.candidate_n}_x1p1.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
