
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from .download import DEFAULT_RAW_DIR, _resolve_data_dir

PLAYLIST_META_COLS = [
    "pid",
    "name",
    "collaborative",
    "modified_at",
    "num_tracks",
    "num_albums",
    "num_followers",
    "num_edits",
    "duration_ms",
    "num_artists",
    "description",
]

TRACK_COLS = [
    "pid",
    "pos",
    "track_name",
    "track_uri",
    "artist_name",
    "artist_uri",
    "album_name",
    "album_uri",
    "duration_ms",
]


def find_slice_files(data_dir: Path | str | None = None) -> list[Path]:
    root = Path(data_dir) if data_dir else DEFAULT_RAW_DIR
    resolved = _resolve_data_dir(root)
    if resolved is None:
        raise FileNotFoundError(
            f"슬라이스 JSON을 찾을 수 없습니다: {root}\n"
            "먼저 `python -m src.download` 로 데이터를 받으세요."
        )
    return sorted(resolved.glob("mpd.slice*.json"))


def _iter_playlists_from_slices(
    slice_files: Iterable[Path],
    show_progress: bool = True,
) -> Iterable[dict]:
    files = list(slice_files)
    iterator = tqdm(files, desc="Reading slices") if show_progress else files
    for path in iterator:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        for playlist in payload.get("playlists", []):
            yield playlist


def load_playlists(
    data_dir: Path | str | None = None,
    n_slices: int | None = 1,
    show_progress: bool = True,
) -> pd.DataFrame:

    slices = find_slice_files(data_dir)
    if n_slices is not None:
        slices = slices[:n_slices]

    rows: list[dict] = []
    for playlist in _iter_playlists_from_slices(slices, show_progress=show_progress):
        row = {col: playlist.get(col) for col in PLAYLIST_META_COLS}
        rows.append(row)

    df = pd.DataFrame(rows, columns=PLAYLIST_META_COLS)
    if not df.empty:
        df["collaborative"] = df["collaborative"].map(
            lambda x: str(x).lower() == "true" if x is not None else False
        )
        df["modified_at"] = pd.to_datetime(df["modified_at"], unit="s", utc=True)
    return df


def load_tracks(
    data_dir: Path | str | None = None,
    n_slices: int | None = 1,
    show_progress: bool = True,
) -> pd.DataFrame:

    slices = find_slice_files(data_dir)
    if n_slices is not None:
        slices = slices[:n_slices]

    rows: list[dict] = []
    for playlist in _iter_playlists_from_slices(slices, show_progress=show_progress):
        pid = playlist.get("pid")
        for track in playlist.get("tracks", []):
            row = {"pid": pid}
            for col in TRACK_COLS:
                if col == "pid":
                    continue
                row[col] = track.get(col)
            rows.append(row)

    return pd.DataFrame(rows, columns=TRACK_COLS)


def playlist_summary(playlists: pd.DataFrame) -> pd.DataFrame:

    numeric = [
        "num_tracks",
        "num_albums",
        "num_followers",
        "num_edits",
        "duration_ms",
        "num_artists",
    ]
    cols = [c for c in numeric if c in playlists.columns]
    return playlists[cols].describe()
