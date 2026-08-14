
from __future__ import annotations

import os
import shutil
from pathlib import Path

DATASET_SLUG = "himanshuwagh/spotify-million"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "spotify-million"


def _load_env() -> None:

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path)


def ensure_dataset(raw_dir: Path | str | None = None, link: bool = True) -> Path:

    _load_env()
    raw_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    existing = _resolve_data_dir(raw_dir)
    if existing is not None:
        return existing

    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "kagglehub가 없습니다. `pip install -r requirements.txt` 후 다시 실행하세요."
        ) from exc

    print(f"Downloading {DATASET_SLUG} via kagglehub (수 GB, 시간이 걸릴 수 있음)...")
    cache_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    data_root = _resolve_data_dir(cache_path)
    if data_root is None:
        raise FileNotFoundError(
            f"다운로드는 됐지만 mpd.slice*.json을 찾지 못했습니다: {cache_path}"
        )

    if link:
        raw_dir.parent.mkdir(parents=True, exist_ok=True)
        if raw_dir.exists() or raw_dir.is_symlink():
            if raw_dir.is_symlink() or raw_dir.is_file():
                raw_dir.unlink()
            else:
                shutil.rmtree(raw_dir)
        try:
            os.symlink(data_root, raw_dir, target_is_directory=True)
            print(f"Linked {raw_dir} -> {data_root}")
        except OSError:
            # WSL 등에서 symlink 실패 시 경로 안내만 남기고 캐시 경로 사용
            print(f"심볼릭 링크 실패. 캐시 경로를 그대로 사용합니다: {data_root}")
            return data_root

        return raw_dir

    return data_root


def _resolve_data_dir(root: Path) -> Path | None:

    root = Path(root)
    if not root.exists():
        return None

    direct = sorted(root.glob("mpd.slice*.json"))
    if direct:
        return root

    nested = sorted(root.rglob("mpd.slice*.json"))
    if nested:
        return nested[0].parent

    return None


if __name__ == "__main__":
    path = ensure_dataset()
    slices = sorted(path.glob("mpd.slice*.json"))
    print(f"Dataset ready: {path}")
    print(f"Slice files: {len(slices)}")
    if slices:
        print(f"First slice: {slices[0].name}")
