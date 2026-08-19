from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.load_data import PLAYLIST_META_COLS, find_slice_files  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs"
LABELS = OUT_DIR / "all_unique_purpose_labels.csv"


def name_key(s: str) -> str:
    return str(s or "").strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-slices", type=int, default=None, help="테스트용 slice 수 제한")
    parser.add_argument(
        "--labels",
        type=Path,
        default=LABELS,
        help="라벨 CSV 경로",
    )
    args = parser.parse_args()

    labels = pd.read_csv(args.labels)
    yes = labels[labels["has_purpose"] == "yes"][["name_key", "purpose_note", "confidence"]]
    yes_keys = set(yes["name_key"].astype(str))
    note_map = yes.set_index("name_key")["purpose_note"].fillna("").to_dict()
    conf_map = yes.set_index("name_key")["confidence"].fillna("").to_dict()

    slices = find_slice_files(PROJECT_ROOT / "data" / "raw" / "spotify-million")
    if args.n_slices is not None:
        slices = slices[: args.n_slices]

    rows: list[dict] = []
    for path in tqdm(slices, desc="Scanning MPD"):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        for pl in payload.get("playlists", []):
            key = name_key(pl.get("name"))
            if key not in yes_keys:
                continue
            row = {col: pl.get(col) for col in PLAYLIST_META_COLS}
            row["name_key"] = key
            row["purpose_note"] = note_map.get(key, "")
            row["confidence"] = conf_map.get(key, "")
            rows.append(row)

    out = OUT_DIR / "purpose_playlists.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    print(f"purpose playlists: {len(df)}")
    print(f"unique titles: {df['name_key'].nunique() if len(df) else 0}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
