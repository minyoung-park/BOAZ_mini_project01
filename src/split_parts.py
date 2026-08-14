
from __future__ import annotations

import os
from pathlib import Path

from .download import DEFAULT_RAW_DIR, PROJECT_ROOT, _resolve_data_dir

N_PARTS = 4
PARTS_DIR = PROJECT_ROOT / "data" / "parts"


def split_into_parts(n_parts: int = N_PARTS) -> None:
    data_dir = _resolve_data_dir(DEFAULT_RAW_DIR)
    if data_dir is None:
        raise FileNotFoundError("원본 데이터가 없습니다. 먼저 python -m src.download")

    slices = sorted(data_dir.glob("mpd.slice*.json"))
    size = len(slices) // n_parts

    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(n_parts):
        part_dir = PARTS_DIR / f"part{i + 1}"
        part_dir.mkdir(exist_ok=True)

        start = i * size
        end = (i + 1) * size if i < n_parts - 1 else len(slices)
        part_slices = slices[start:end]

        for src in part_slices:
            dst = part_dir / src.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            os.symlink(src.resolve(), dst)

        print(f"part{i + 1}: {len(part_slices)} slices ({start}~{end - 1})")


if __name__ == "__main__":
    split_into_parts()
