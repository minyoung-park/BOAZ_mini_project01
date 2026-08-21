"""BPR-MF 추천 전후의 인기도·다양성 분포를 비교하는 도우미 함수."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ItemMetadata:
    """체크포인트 내부 item ID 순서에 맞춘 트랙 메타데이터."""

    popularity: np.ndarray
    novelty: np.ndarray
    top_1pct: np.ndarray
    artist_code: np.ndarray


def build_item_metadata(
    track_uris: list[str],
    track_frequency: pd.DataFrame,
    n_playlists: int,
) -> ItemMetadata:
    """전처리 결과를 체크포인트 item ID 순서로 정렬한다."""
    required = {"track_uri", "n_playlists", "pct_rank", "artist_name"}
    missing = required - set(track_frequency.columns)
    if missing:
        raise ValueError(f"track_freq.parquet에 필요한 컬럼이 없습니다: {sorted(missing)}")

    ordered = track_frequency.set_index("track_uri").reindex(track_uris)
    if ordered["n_playlists"].isna().any():
        count = int(ordered["n_playlists"].isna().sum())
        raise ValueError(f"체크포인트 트랙 {count:,}개가 빈도 집계에 없습니다.")

    playlist_count = ordered["n_playlists"].to_numpy(dtype=np.float64)
    popularity = playlist_count / n_playlists
    novelty = np.log(n_playlists / playlist_count) / np.log(n_playlists)
    artist_code, _ = pd.factorize(ordered["artist_name"], sort=False)
    return ItemMetadata(
        popularity=popularity.astype(np.float32),
        novelty=novelty.astype(np.float32),
        top_1pct=(ordered["pct_rank"].to_numpy() <= 0.01),
        artist_code=artist_code.astype(np.int32),
    )


def _entropy(codes: np.ndarray) -> float:
    if len(codes) == 0:
        return np.nan
    _, counts = np.unique(codes, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def metrics_for_lists(
    item_lists: list[np.ndarray] | np.ndarray,
    metadata: ItemMetadata,
) -> pd.DataFrame:
    """각 트랙 목록의 인기도와 아티스트 다양성 지표를 계산한다."""
    rows = []
    for items in item_lists:
        items = np.asarray(items, dtype=np.int64)
        artists = metadata.artist_code[items]
        unique_artists, artist_counts = np.unique(artists, return_counts=True)
        entropy = _entropy(artists)
        rows.append({
            "n_items": len(items),
            "avg_track_popularity": float(metadata.popularity[items].mean()),
            "track_novelty": float(metadata.novelty[items].mean()),
            "top_1pct_ratio": float(metadata.top_1pct[items].mean()),
            "n_unique_artists": len(unique_artists),
            "unique_artist_ratio": len(unique_artists) / max(len(items), 1),
            "artist_entropy": entropy,
            "normalized_artist_entropy": (
                entropy / np.log(len(items)) if len(items) > 1 else 0.0
            ),
            "top_artist_share": float(artist_counts.max() / max(len(items), 1)),
        })
    return pd.DataFrame(rows)


def recommendation_novelty_metrics(
    train_lists: list[np.ndarray],
    recommendation_lists: np.ndarray,
    metadata: ItemMetadata,
) -> pd.DataFrame:
    """추천이 기존 아티스트 범위를 얼마나 확장하는지 계산한다."""
    rows = []
    for train, recommended in zip(train_lists, recommendation_lists):
        train_artists = set(map(int, metadata.artist_code[np.asarray(train)]))
        recommended_artists = metadata.artist_code[np.asarray(recommended)]
        new_mask = np.fromiter(
            (int(artist) not in train_artists for artist in recommended_artists),
            dtype=bool,
        )
        new_unique = set(map(int, recommended_artists)) - train_artists
        rows.append({
            "new_artist_track_ratio": float(new_mask.mean()),
            "new_unique_artists": len(new_unique),
            "seen_artist_track_ratio": float(1.0 - new_mask.mean()),
        })
    return pd.DataFrame(rows)


def paired_delta_summary(
    before: pd.DataFrame,
    after: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """추천 전후 변화량의 평균·분위수와 표준화 효과크기를 반환한다."""
    rows = []
    for column in columns:
        delta = after[column].to_numpy() - before[column].to_numpy()
        before_std = before[column].std(ddof=1)
        rows.append({
            "metric": column,
            "train_mean": before[column].mean(),
            "after_mean": after[column].mean(),
            "mean_delta": delta.mean(),
            "median_delta": np.median(delta),
            "p10_delta": np.quantile(delta, 0.10),
            "p90_delta": np.quantile(delta, 0.90),
            "share_increase": np.mean(delta > 0),
            "paired_effect_size": delta.mean() / before_std if before_std else np.nan,
        })
    return pd.DataFrame(rows).set_index("metric")


def ranking_metrics(
    recommendations: np.ndarray,
    targets: list[np.ndarray],
) -> dict[str, float]:
    """전체 카탈로그 추천에서 exact-track Recall, NDCG, MRR을 계산한다."""
    recall = ndcg = mrr = hit = 0.0
    for recommended, target in zip(recommendations, targets):
        target_set = set(map(int, target))
        relevance = np.fromiter(
            (int(int(item) in target_set) for item in recommended), dtype=np.float64
        )
        positions = np.flatnonzero(relevance)
        recall += relevance.sum() / max(len(target_set), 1)
        hit += bool(len(positions))
        mrr += 1.0 / (positions[0] + 1) if len(positions) else 0.0
        discounts = 1.0 / np.log2(np.arange(2, len(recommended) + 2))
        ideal = discounts[: min(len(target_set), len(recommended))].sum()
        ndcg += float((relevance * discounts).sum() / ideal) if ideal else 0.0
    count = len(recommendations)
    return {
        f"hit_rate@{recommendations.shape[1]}": hit / count,
        f"recall@{recommendations.shape[1]}": recall / count,
        f"ndcg@{recommendations.shape[1]}": ndcg / count,
        f"mrr@{recommendations.shape[1]}": mrr / count,
        "playlists": count,
    }


def exposure_summary(recommendations: np.ndarray, n_items: int) -> dict[str, float]:
    """추천 목록 전체의 카탈로그 커버리지와 노출 집중도를 계산한다."""
    counts = np.bincount(recommendations.ravel(), minlength=n_items)
    exposed = counts[counts > 0].astype(np.float64)
    if len(exposed) == 0:
        return {"catalog_coverage": 0.0, "exposed_items": 0, "exposure_gini": np.nan}
    exposed.sort()
    index = np.arange(1, len(exposed) + 1, dtype=np.float64)
    gini = (
        2 * np.sum(index * exposed) / (len(exposed) * exposed.sum())
        - (len(exposed) + 1) / len(exposed)
    )
    return {
        "catalog_coverage": len(exposed) / n_items,
        "exposed_items": int(len(exposed)),
        "exposure_gini": float(gini),
    }
