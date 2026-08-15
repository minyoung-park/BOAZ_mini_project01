"""전체 MPD에서 고유 플레이리스트 제목 추출."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.load_data import find_slice_files  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "outputs"


def name_key(s: str) -> str:
    return str(s or "").strip().lower()


def main() -> None:
    slices = find_slice_files(PROJECT_ROOT / "data" / "raw" / "spotify-million")
    # name_key -> {display_name, count, example_pids}
    stats: dict[str, dict] = {}

    for path in tqdm(slices, desc="Scanning slices"):
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        for pl in payload.get("playlists", []):
            raw = pl.get("name") or ""
            key = name_key(raw)
            if key not in stats:
                stats[key] = {
                    "name_key": key,
                    "name": raw.strip(),
                    "n_playlists": 0,
                    "example_pids": [],
                }
            stats[key]["n_playlists"] += 1
            if len(stats[key]["example_pids"]) < 5:
                stats[key]["example_pids"].append(pl.get("pid"))

    rows = sorted(stats.values(), key=lambda r: (-r["n_playlists"], r["name_key"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "all_unique_titles.csv"

    import pandas as pd

    df = pd.DataFrame(rows)
    df.insert(0, "sample_id", range(len(df)))
    df["example_pids"] = df["example_pids"].map(lambda xs: "|".join(str(x) for x in xs))
    df.to_csv(out, index=False)

    print(f"unique titles: {len(df)}")
    print(f"total playlist mass: {df['n_playlists'].sum()}")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
