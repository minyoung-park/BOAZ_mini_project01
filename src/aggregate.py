"""MPD 전역 frequency + 플레이리스트 구성 집계.

JSON을 두 번 순회한다.
1) Track/Artist 등장 횟수, 플레이리스트 메타
2) 전역 순위(n_playlists)를 붙인 뒤 플레이리스트별 concentration

Popularity 기본 정의: unique track이 몇 개의 Playlist에 포함됐는가 (`n_playlists`).
`n_occurrences`는 별도 occurrence concentration으로만 보고한다.

Legacy Head/Mid/Tail (참고용):
- Head: 상위 10% / Mid: 다음 40% / Tail: 하위 50%

Sensitivity thresholds: Top 1% / 5% / 10% (pct_rank 기준).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .download import PROJECT_ROOT
from .load_data import PLAYLIST_META_COLS, find_slice_files, _iter_playlists_from_slices

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

HEAD_FRAC = 0.10
TAIL_FRAC = 0.50
TOP_FRACS = (0.01, 0.05, 0.10)

TRACK_FREQ_PATH = PROCESSED_DIR / "track_freq.parquet"
ARTIST_FREQ_PATH = PROCESSED_DIR / "artist_freq.parquet"
PLAYLIST_META_PATH = PROCESSED_DIR / "playlist_meta.parquet"
PLAYLIST_COMP_PATH = PROCESSED_DIR / "playlist_composition.parquet"
SUMMARY_PATH = PROCESSED_DIR / "frequency_summary.json"


def _slice_files(n_slices: int | None) -> list[Path]:
    slices = find_slice_files()
    if n_slices is not None:
        slices = slices[:n_slices]
    return slices


def _gini(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=np.float64))
    total = x.sum()
    if x.size == 0 or total <= 0:
        return float("nan")
    n = x.size
    return float((2.0 * np.sum(np.arange(1, n + 1) * x) / (n * total)) - (n + 1) / n)


def _share_of_top(counts: np.ndarray, frac: float) -> float:
    """상위 frac 아이템이 전체 count에서 차지하는 비율."""
    if counts.size == 0:
        return float("nan")
    k = max(1, int(math.ceil(counts.size * frac)))
    ordered = np.sort(counts)[::-1]
    total = ordered.sum()
    if total <= 0:
        return float("nan")
    return float(ordered[:k].sum() / total)


def _ratio_label(frac: float) -> str:
    pct = int(round(frac * 100))
    return f"top_{pct}pct_ratio"


def _empty_composition_row(pid) -> dict:
    row = {
        "pid": pid,
        "n_items": 0,
        "n_unique_tracks": 0,
        "n_unique_artists": 0,
        "avg_n_playlists": np.nan,
        "median_n_playlists": np.nan,
        "avg_log_n_playlists": np.nan,
        "avg_pct_rank": np.nan,
        "head_ratio": np.nan,
        "mid_ratio": np.nan,
        "tail_ratio": np.nan,
        "unknown_ratio": np.nan,
        "unique_artist_ratio": np.nan,
        "top_artist_share": np.nan,
        "artist_entropy": np.nan,
    }
    for frac in TOP_FRACS:
        row[_ratio_label(frac)] = np.nan
    return row


def _assign_tiers(freq: pd.DataFrame, count_col: str) -> pd.DataFrame:
    out = freq.sort_values(count_col, ascending=False, kind="mergesort").reset_index(drop=True)
    n = len(out)
    out["rank"] = np.arange(1, n + 1)
    out["pct_rank"] = out["rank"] / n
    out["tier"] = np.where(
        out["pct_rank"] <= HEAD_FRAC,
        "head",
        np.where(out["pct_rank"] > TAIL_FRAC, "tail", "mid"),
    )
    return out


def build_frequencies(n_slices: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    slices = _slice_files(n_slices)

    track_occ: Counter[str] = Counter()
    track_pl: Counter[str] = Counter()
    artist_occ: Counter[str] = Counter()
    artist_pl: Counter[str] = Counter()
    track_meta: dict[str, tuple[str, str, str]] = {}
    artist_name: dict[str, str] = {}
    playlist_rows: list[dict] = []

    for playlist in _iter_playlists_from_slices(slices, show_progress=True):
        row = {col: playlist.get(col) for col in PLAYLIST_META_COLS}
        playlist_rows.append(row)

        seen_tracks: set[str] = set()
        seen_artists: set[str] = set()
        for track in playlist.get("tracks", []):
            t_uri = track.get("track_uri")
            a_uri = track.get("artist_uri")
            if not t_uri:
                continue

            track_occ[t_uri] += 1
            if t_uri not in seen_tracks:
                track_pl[t_uri] += 1
                seen_tracks.add(t_uri)
            if t_uri not in track_meta:
                track_meta[t_uri] = (
                    track.get("track_name") or "",
                    a_uri or "",
                    track.get("artist_name") or "",
                )

            if not a_uri:
                continue
            artist_occ[a_uri] += 1
            if a_uri not in seen_artists:
                artist_pl[a_uri] += 1
                seen_artists.add(a_uri)
            if a_uri not in artist_name:
                artist_name[a_uri] = track.get("artist_name") or ""

    track_rows = [
        {
            "track_uri": uri,
            "track_name": track_meta[uri][0],
            "artist_uri": track_meta[uri][1],
            "artist_name": track_meta[uri][2],
            "n_playlists": track_pl[uri],
            "n_occurrences": track_occ[uri],
        }
        for uri in track_occ
    ]
    artist_rows = [
        {
            "artist_uri": uri,
            "artist_name": artist_name.get(uri, ""),
            "n_playlists": artist_pl[uri],
            "n_occurrences": artist_occ[uri],
        }
        for uri in artist_occ
    ]

    tracks = _assign_tiers(pd.DataFrame(track_rows), "n_playlists")
    artists = _assign_tiers(pd.DataFrame(artist_rows), "n_playlists")
    n_tracks_by_artist = tracks.groupby("artist_uri").size().rename("n_tracks")
    artists = artists.merge(n_tracks_by_artist, on="artist_uri", how="left")
    artists["n_tracks"] = artists["n_tracks"].fillna(0).astype(int)

    playlists = pd.DataFrame(playlist_rows, columns=PLAYLIST_META_COLS)
    if not playlists.empty:
        playlists["collaborative"] = playlists["collaborative"].map(
            lambda x: str(x).lower() == "true" if x is not None else False
        )
        playlists["modified_at"] = pd.to_datetime(playlists["modified_at"], unit="s", utc=True)

    return tracks, artists, playlists


def build_playlist_composition(
    track_freq: pd.DataFrame,
    n_slices: int | None = None,
) -> pd.DataFrame:
    """Popularity = n_playlists. Top-k ratio는 pct_rank <= k 인 곡 비율."""
    popularity = dict(zip(track_freq["track_uri"], track_freq["n_playlists"]))
    pct_rank = dict(zip(track_freq["track_uri"], track_freq["pct_rank"]))
    tier = dict(zip(track_freq["track_uri"], track_freq["tier"]))
    slices = _slice_files(n_slices)

    rows: list[dict] = []
    for playlist in _iter_playlists_from_slices(slices, show_progress=True):
        pid = playlist.get("pid")
        tracks = playlist.get("tracks") or []
        if not tracks:
            rows.append(_empty_composition_row(pid))
            continue

        pops: list[int] = []
        ranks: list[float] = []
        tiers: Counter[str] = Counter()
        top_hits = {frac: 0 for frac in TOP_FRACS}
        artist_counts: Counter[str] = Counter()
        unique_tracks: set[str] = set()
        n_unknown = 0

        for track in tracks:
            t_uri = track.get("track_uri")
            a_uri = track.get("artist_uri") or ""
            if t_uri:
                unique_tracks.add(t_uri)
                pop = popularity.get(t_uri)
                if pop is None:
                    n_unknown += 1
                else:
                    pops.append(int(pop))
                    pr = float(pct_rank[t_uri])
                    ranks.append(pr)
                    tiers[tier[t_uri]] += 1
                    for frac in TOP_FRACS:
                        if pr <= frac:
                            top_hits[frac] += 1
            if a_uri:
                artist_counts[a_uri] += 1

        n_items = len(tracks)
        n_known = n_items - n_unknown
        pop_arr = np.asarray(pops, dtype=np.float64) if pops else np.array([], dtype=np.float64)
        rank_arr = np.asarray(ranks, dtype=np.float64) if ranks else np.array([], dtype=np.float64)

        total_artists = sum(artist_counts.values())
        if total_artists > 0:
            entropy = 0.0
            for c in artist_counts.values():
                p = c / total_artists
                entropy -= p * math.log2(p)
            top_artist_share = max(artist_counts.values()) / total_artists
        else:
            entropy = float("nan")
            top_artist_share = float("nan")

        row = {
            "pid": pid,
            "n_items": n_items,
            "n_unique_tracks": len(unique_tracks),
            "n_unique_artists": len(artist_counts),
            "avg_n_playlists": float(pop_arr.mean()) if pop_arr.size else np.nan,
            "median_n_playlists": float(np.median(pop_arr)) if pop_arr.size else np.nan,
            "avg_log_n_playlists": float(np.log1p(pop_arr).mean()) if pop_arr.size else np.nan,
            "avg_pct_rank": float(rank_arr.mean()) if rank_arr.size else np.nan,
            "head_ratio": tiers["head"] / n_known if n_known else np.nan,
            "mid_ratio": tiers["mid"] / n_known if n_known else np.nan,
            "tail_ratio": tiers["tail"] / n_known if n_known else np.nan,
            "unknown_ratio": n_unknown / n_items if n_items else np.nan,
            "unique_artist_ratio": (
                len(artist_counts) / n_items if n_items else float("nan")
            ),
            "top_artist_share": top_artist_share,
            "artist_entropy": entropy,
        }
        for frac in TOP_FRACS:
            row[_ratio_label(frac)] = top_hits[frac] / n_known if n_known else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def _tier_share(freq: pd.DataFrame, weight_col: str) -> dict[str, float]:
    total = float(freq[weight_col].sum())
    if total <= 0:
        return {"head": float("nan"), "mid": float("nan"), "tail": float("nan")}
    shares = freq.groupby("tier")[weight_col].sum() / total
    return {k: float(shares.get(k, 0.0)) for k in ("head", "mid", "tail")}


def _describe_series(s: pd.Series) -> dict[str, float]:
    s = s.dropna()
    if s.empty:
        keys = ("mean", "std", "p10", "p25", "p50", "p75", "p90")
        return {k: float("nan") for k in keys}
    return {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
    }


def build_summary(
    tracks: pd.DataFrame,
    artists: pd.DataFrame,
    playlists: pd.DataFrame,
    composition: pd.DataFrame,
) -> dict:
    track_counts = tracks["n_playlists"].to_numpy()
    track_occ = tracks["n_occurrences"].to_numpy()
    artist_counts = artists["n_playlists"].to_numpy()
    artist_occ = artists["n_occurrences"].to_numpy()

    summary = {
        "n_playlists": int(len(playlists)),
        "n_unique_tracks": int(len(tracks)),
        "n_unique_artists": int(len(artists)),
        "n_interactions": int(track_occ.sum()),
        "popularity_definition": "n_playlists",
        "head_frac_tracks": HEAD_FRAC,
        "tail_frac_tracks": TAIL_FRAC,
        "top_fracs": list(TOP_FRACS),
        "track": {
            "gini_n_playlists": _gini(track_counts),
            "top_1pct_playlist_mass_share": _share_of_top(track_counts, 0.01),
            "top_5pct_playlist_mass_share": _share_of_top(track_counts, 0.05),
            "top_10pct_playlist_mass_share": _share_of_top(track_counts, 0.10),
            "tier_track_share": tracks["tier"].value_counts(normalize=True).to_dict(),
            "tier_playlist_mass_share": _tier_share(tracks, "n_playlists"),
            "occurrence_concentration": {
                "top_1pct_share": _share_of_top(track_occ, 0.01),
                "top_10pct_share": _share_of_top(track_occ, 0.10),
                "tier_share": _tier_share(tracks, "n_occurrences"),
            },
            "median_n_playlists": float(np.median(track_counts)),
            "mean_n_playlists": float(np.mean(track_counts)),
        },
        "artist": {
            "gini_n_playlists": _gini(artist_counts),
            "top_1pct_playlist_mass_share": _share_of_top(artist_counts, 0.01),
            "top_10pct_playlist_mass_share": _share_of_top(artist_counts, 0.10),
            "occurrence_concentration": {
                "top_1pct_share": _share_of_top(artist_occ, 0.01),
                "top_10pct_share": _share_of_top(artist_occ, 0.10),
            },
            "median_n_playlists": float(np.median(artist_counts)),
            "mean_n_playlists": float(np.mean(artist_counts)),
        },
        "playlist_composition": {
            "legacy_head_ratio": _describe_series(composition["head_ratio"]),
            "legacy_tail_ratio": _describe_series(composition["tail_ratio"]),
            "avg_log_n_playlists": _describe_series(composition["avg_log_n_playlists"]),
            "avg_pct_rank": _describe_series(composition["avg_pct_rank"]),
        },
    }
    for frac in TOP_FRACS:
        col = _ratio_label(frac)
        summary["playlist_composition"][col] = _describe_series(composition[col])
    return summary


def print_summary(summary: dict) -> None:
    t = summary["track"]
    a = summary["artist"]
    p = summary["playlist_composition"]
    print("\n=== Frequency summary ===")
    print(f"playlists={summary['n_playlists']:,}  tracks={summary['n_unique_tracks']:,}  "
          f"artists={summary['n_unique_artists']:,}  interactions={summary['n_interactions']:,}")
    print(f"Popularity definition: {summary['popularity_definition']}")
    print(
        f"Track Gini(n_playlists)={t['gini_n_playlists']:.3f}  "
        f"top1% playlist-mass={t['top_1pct_playlist_mass_share']:.3f}  "
        f"top10% playlist-mass={t['top_10pct_playlist_mass_share']:.3f}"
    )
    occ = t["occurrence_concentration"]
    print(
        f"Track occurrence concentration (별도): "
        f"top1%={occ['top_1pct_share']:.3f}  top10%={occ['top_10pct_share']:.3f}"
    )
    print(
        f"Artist Gini(n_playlists)={a['gini_n_playlists']:.3f}  "
        f"top1% playlist-mass={a['top_1pct_playlist_mass_share']:.3f}  "
        f"top10% playlist-mass={a['top_10pct_playlist_mass_share']:.3f}"
    )
    for frac in TOP_FRACS:
        col = _ratio_label(frac)
        d = p[col]
        print(
            f"Playlist {col}: "
            f"mean={d['mean']:.3f}  p10={d['p10']:.3f}  p25={d['p25']:.3f}  "
            f"p50={d['p50']:.3f}  p75={d['p75']:.3f}  p90={d['p90']:.3f}"
        )
    logd = p["avg_log_n_playlists"]
    print(
        "Playlist avg_log_n_playlists: "
        f"mean={logd['mean']:.3f}  p10={logd['p10']:.3f}  p50={logd['p50']:.3f}  p90={logd['p90']:.3f}"
    )


def save_tables(
    tracks: pd.DataFrame,
    artists: pd.DataFrame,
    playlists: pd.DataFrame,
    composition: pd.DataFrame,
    summary: dict,
    *,
    write_freq_tables: bool = True,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if write_freq_tables:
        tracks.to_parquet(TRACK_FREQ_PATH, index=False)
        artists.to_parquet(ARTIST_FREQ_PATH, index=False)
        playlists.to_parquet(PLAYLIST_META_PATH, index=False)
        print(f"Wrote {TRACK_FREQ_PATH}")
        print(f"Wrote {ARTIST_FREQ_PATH}")
        print(f"Wrote {PLAYLIST_META_PATH}")
    composition.to_parquet(PLAYLIST_COMP_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {PLAYLIST_COMP_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


def run(n_slices: int | None = None, composition_only: bool = False) -> dict:
    if composition_only:
        if n_slices is not None:
            raise ValueError(
                "--composition-only 는 전체 playlist_meta를 기준으로만 동작합니다. "
                "--n-slices 와 함께 쓰지 마세요."
            )
        missing = [p for p in (TRACK_FREQ_PATH, ARTIST_FREQ_PATH, PLAYLIST_META_PATH) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "composition-only 모드인데 기존 집계가 없습니다: "
                + ", ".join(str(p) for p in missing)
            )
        print("Reuse existing track/artist/playlist tables", flush=True)
        tracks = pd.read_parquet(TRACK_FREQ_PATH)
        artists = pd.read_parquet(ARTIST_FREQ_PATH)
        playlists = pd.read_parquet(PLAYLIST_META_PATH)
    else:
        print("Pass 1/2: track/artist frequency + playlist meta", flush=True)
        tracks, artists, playlists = build_frequencies(n_slices=n_slices)

    print("Pass 2/2: playlist Top-k / continuous composition", flush=True)
    composition = build_playlist_composition(tracks, n_slices=n_slices)
    summary = build_summary(tracks, artists, playlists, composition)
    save_tables(
        tracks,
        artists,
        playlists,
        composition,
        summary,
        write_freq_tables=not composition_only,
    )
    print_summary(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="MPD frequency / composition 집계")
    parser.add_argument(
        "--n-slices",
        type=int,
        default=None,
        help="앞에서부터 사용할 slice 수. 생략하면 전체.",
    )
    parser.add_argument(
        "--composition-only",
        action="store_true",
        help="기존 track_freq/playlist_meta를 재사용하고 composition만 다시 계산.",
    )
    args = parser.parse_args()
    run(n_slices=args.n_slices, composition_only=args.composition_only)


if __name__ == "__main__":
    main()
