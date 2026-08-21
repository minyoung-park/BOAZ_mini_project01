"""MPD 분석 노트북(01~04)에 필요한 중간 집계 파일을 만든다."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd


OUTPUT_FILES = (
    "frequency_summary.json",
    "track_freq.parquet",
    "artist_freq.parquet",
    "playlist_meta.parquet",
    "playlist_composition.parquet",
)
AGGREGATION_VERSION = 2


def _slice_start(path: Path) -> int:
    match = re.search(r"mpd\.slice\.(\d+)-", path.name)
    return int(match.group(1)) if match else 10**12


def _gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or values.sum() == 0:
        return 0.0
    values.sort()
    index = np.arange(1, values.size + 1, dtype=np.float64)
    return float((2 * np.sum(index * values) / (values.size * values.sum())) - (values.size + 1) / values.size)


def _top_share(frame: pd.DataFrame, fraction: float, column: str) -> float:
    count = max(1, math.ceil(len(frame) * fraction))
    total = frame[column].sum()
    return float(frame.iloc[:count][column].sum() / total) if total else 0.0


def _frequency_frame(
    playlist_count: Counter,
    occurrence_count: Counter,
    names: dict[str, tuple[str, str] | str],
    kind: str,
) -> pd.DataFrame:
    rows = []
    for uri, n_playlists in playlist_count.items():
        row = {
            f"{kind}_uri": uri,
            "n_playlists": n_playlists,
            "n_occurrences": occurrence_count[uri],
        }
        if kind == "track":
            track_name, artist_name = names[uri]
            row.update(track_name=track_name, artist_name=artist_name)
        else:
            row["artist_name"] = names[uri]
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values(
        ["n_playlists", "n_occurrences", f"{kind}_uri"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    frame["pct_rank"] = frame["rank"] / max(len(frame), 1)
    frame["tier"] = np.select(
        [frame["pct_rank"] <= 0.10, frame["pct_rank"] <= 0.50],
        ["head", "mid"],
        default="tail",
    )
    return frame


def _summary_for(frame: pd.DataFrame) -> dict:
    interaction_total = frame["n_occurrences"].sum()
    tier_share = (
        frame.groupby("tier", observed=True)["n_occurrences"].sum() / interaction_total
    ).reindex(["head", "mid", "tail"], fill_value=0.0)
    return {
        "gini_n_playlists": _gini(frame["n_playlists"].to_numpy()),
        "median_n_playlists": float(frame["n_playlists"].median()),
        "mean_n_playlists": float(frame["n_playlists"].mean()),
        "top_1pct_interaction_share": _top_share(frame, 0.01, "n_occurrences"),
        "top_5pct_interaction_share": _top_share(frame, 0.05, "n_occurrences"),
        "top_10pct_interaction_share": _top_share(frame, 0.10, "n_occurrences"),
        "top_1pct_playlist_mass_share": _top_share(frame, 0.01, "n_playlists"),
        "top_5pct_playlist_mass_share": _top_share(frame, 0.05, "n_playlists"),
        "top_10pct_playlist_mass_share": _top_share(frame, 0.10, "n_playlists"),
        "tier_interaction_share": {key: float(value) for key, value in tier_share.items()},
    }


def _load_slice(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)["playlists"]


def prepare_notebook_data(
    data_dir: str | Path,
    processed_dir: str | Path,
    max_slices: int | None = 10,
    force: bool = False,
) -> Path:
    """원본 MPD JSON에서 네 분석 노트북이 공유하는 parquet 파일을 만든다."""
    data_dir = Path(data_dir).resolve()
    processed_dir = Path(processed_dir).resolve()
    slice_files = sorted(data_dir.glob("mpd.slice.*.json"), key=_slice_start)
    if not slice_files:
        raise FileNotFoundError(f"MPD 슬라이스를 찾을 수 없습니다: {data_dir}")
    if max_slices is not None:
        if max_slices < 1:
            raise ValueError("max_slices는 1 이상 또는 None이어야 합니다.")
        slice_files = slice_files[:max_slices]

    processed_dir.mkdir(parents=True, exist_ok=True)
    summary_path = processed_dir / "frequency_summary.json"
    if not force and all((processed_dir / name).exists() for name in OUTPUT_FILES):
        cached = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            cached.get("source_n_slices") == len(slice_files)
            and cached.get("aggregation_version") == AGGREGATION_VERSION
        ):
            print(f"기존 집계 사용: {processed_dir} ({len(slice_files)}개 슬라이스)")
            return processed_dir

    track_playlists: Counter = Counter()
    track_occurrences: Counter = Counter()
    artist_playlists: Counter = Counter()
    artist_occurrences: Counter = Counter()
    track_names: dict[str, tuple[str, str]] = {}
    artist_names: dict[str, str] = {}
    playlist_rows = []

    print(f"1/2 빈도 집계: {len(slice_files)}개 슬라이스")
    for index, path in enumerate(slice_files, 1):
        for playlist in _load_slice(path):
            playlist_rows.append({
                "pid": playlist["pid"],
                "name": playlist.get("name", ""),
                "num_tracks": playlist.get("num_tracks", len(playlist["tracks"])),
                "num_followers": playlist.get("num_followers", 0),
                "num_edits": playlist.get("num_edits", 0),
                "num_artists": playlist.get("num_artists", 0),
                "num_albums": playlist.get("num_albums", 0),
                "modified_at": playlist.get("modified_at"),
                "collaborative": playlist.get("collaborative", "false"),
                "duration_ms": playlist.get("duration_ms", 0),
            })
            seen_tracks = set()
            seen_artists = set()
            for track in playlist["tracks"]:
                track_uri = track["track_uri"]
                artist_uri = track["artist_uri"]
                track_occurrences[track_uri] += 1
                artist_occurrences[artist_uri] += 1
                seen_tracks.add(track_uri)
                seen_artists.add(artist_uri)
                track_names.setdefault(track_uri, (track["track_name"], track["artist_name"]))
                artist_names.setdefault(artist_uri, track["artist_name"])
            track_playlists.update(seen_tracks)
            artist_playlists.update(seen_artists)
        if index % 10 == 0 or index == len(slice_files):
            print(f"  {index}/{len(slice_files)}")

    tracks = _frequency_frame(track_playlists, track_occurrences, track_names, "track")
    artists = _frequency_frame(artist_playlists, artist_occurrences, artist_names, "artist")
    playlists = pd.DataFrame(playlist_rows)
    n_playlists = len(playlists)
    n_interactions = int(tracks["n_occurrences"].sum())
    track_summary = _summary_for(tracks)
    artist_summary = _summary_for(artists)

    # 전체 MPD에서도 메모리를 아끼기 위해 큰 DataFrame은 먼저 저장한 뒤
    # 두 번째 순회에 필요한 최소 lookup만 남긴다.
    tracks.to_parquet(processed_dir / "track_freq.parquet", index=False)
    artists.to_parquet(processed_dir / "artist_freq.parquet", index=False)
    playlists.to_parquet(processed_dir / "playlist_meta.parquet", index=False)
    track_lookup = dict(zip(
        tracks["track_uri"],
        zip(tracks["n_playlists"], tracks["pct_rank"], tracks["tier"]),
    ))
    artist_playlist_count = dict(zip(artists["artist_uri"], artists["n_playlists"]))
    del (
        track_playlists, track_occurrences, artist_playlists, artist_occurrences,
        track_names, artist_names, playlist_rows, tracks, artists, playlists,
    )

    composition_rows = []
    print("2/2 플레이리스트 구성 집계")
    for index, path in enumerate(slice_files, 1):
        for playlist in _load_slice(path):
            playlist_tracks = playlist["tracks"]
            n_items = len(playlist_tracks)
            track_info = [track_lookup[track["track_uri"]] for track in playlist_tracks]
            tiers = Counter(info[2] for info in track_info)
            artist_counts = Counter(track["artist_uri"] for track in playlist_tracks)
            album_counts = Counter(track["album_uri"] for track in playlist_tracks)
            probabilities = np.fromiter(artist_counts.values(), dtype=np.float64) / max(n_items, 1)
            album_probabilities = np.fromiter(album_counts.values(), dtype=np.float64) / max(n_items, 1)
            popularity = np.fromiter((info[0] / n_playlists for info in track_info), dtype=np.float64)
            log_n_playlists = math.log(max(n_playlists, 2))
            composition_rows.append({
                "pid": playlist["pid"],
                "n_items": n_items,
                "actual_num_tracks": n_items,
                "n_unique_artists": len(artist_counts),
                "unique_artists": len(artist_counts),
                "unique_albums": len(album_counts),
                "head_ratio": tiers["head"] / max(n_items, 1),
                "mid_ratio": tiers["mid"] / max(n_items, 1),
                "tail_ratio": tiers["tail"] / max(n_items, 1),
                "top_1pct_ratio": np.mean([info[1] <= 0.01 for info in track_info]),
                "top_5pct_ratio": np.mean([info[1] <= 0.05 for info in track_info]),
                "top_10pct_ratio": np.mean([info[1] <= 0.10 for info in track_info]),
                "avg_log_n_playlists": np.mean([math.log1p(info[0]) for info in track_info]),
                "avg_pct_rank": np.mean([info[1] for info in track_info]),
                "avg_track_popularity": float(popularity.mean()),
                "popularity_variance": float(popularity.var(ddof=1)) if n_items > 1 else 0.0,
                "popularity_spread": float(popularity.std(ddof=1)) if n_items > 1 else 0.0,
                "track_novelty": float(np.mean([
                    math.log(n_playlists / info[0]) / log_n_playlists for info in track_info
                ])),
                "artist_novelty": float(np.mean([
                    math.log(n_playlists / artist_playlist_count[track["artist_uri"]]) / log_n_playlists
                    for track in playlist_tracks
                ])),
                "niche_track_ratio": np.mean([info[0] <= 2 for info in track_info]),
                "niche_artist_ratio": np.mean([
                    artist_playlist_count[track["artist_uri"]] <= 2 for track in playlist_tracks
                ]),
                "unique_artist_ratio": len(artist_counts) / max(n_items, 1),
                "artist_diversity": len(artist_counts) / max(n_items, 1),
                "album_diversity": len(album_counts) / max(n_items, 1),
                "top_artist_share": max(artist_counts.values(), default=0) / max(n_items, 1),
                "artist_entropy": float(-np.sum(probabilities * np.log(probabilities))) if n_items else 0.0,
                "album_entropy": float(-np.sum(album_probabilities * np.log(album_probabilities))) if n_items else 0.0,
            })
        if index % 10 == 0 or index == len(slice_files):
            print(f"  {index}/{len(slice_files)}")

    composition = pd.DataFrame(composition_rows)
    comp_summary = {}
    for column in ("head_ratio", "tail_ratio"):
        values = composition[column]
        prefix = column.removesuffix("_ratio")
        comp_summary.update({
            f"{column}_mean": float(values.mean()),
            f"{column}_std": float(values.std()),
            f"{column}_p10": float(values.quantile(0.10)),
            f"{column}_p50": float(values.quantile(0.50)),
            f"{column}_p90": float(values.quantile(0.90)),
        })

    summary = {
        "aggregation_version": AGGREGATION_VERSION,
        "source_data_dir": str(data_dir),
        "source_n_slices": len(slice_files),
        "n_playlists": n_playlists,
        "n_unique_tracks": len(track_lookup),
        "n_unique_artists": len(artist_playlist_count),
        "n_interactions": n_interactions,
        "popularity_definition": "MPD 내 트랙이 등장한 플레이리스트 수(n_playlists)",
        "track": track_summary,
        "artist": artist_summary,
        "playlist_composition": comp_summary,
    }

    composition.to_parquet(processed_dir / "playlist_composition.parquet", index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"집계 완료: {processed_dir}")
    return processed_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-slices", type=int, default=10, help="기본 10, 전체는 0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare_notebook_data(
        args.data_dir,
        args.processed_dir,
        max_slices=None if args.max_slices == 0 else args.max_slices,
        force=args.force,
    )


if __name__ == "__main__":
    main()
