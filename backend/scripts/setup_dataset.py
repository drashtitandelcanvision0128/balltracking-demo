"""
Download and validate the cricket ball YOLO dataset.

Dataset source (Kaggle):
  https://www.kaggle.com/datasets/kushagra3204/cricket-ball-dataset-for-yolo

Usage:
  python scripts/setup_dataset.py
  python scripts/setup_dataset.py --kaggle   # install kaggle + download (needs API key)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
DATASET_ROOT = _BACKEND / "dataset" / "cricket_ball_data"
KAGGLE_DATASET = "kushagra3204/cricket-ball-dataset-for-yolo"


def _ensure_layout() -> None:
    for split in ("train", "valid"):
        (DATASET_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)


def _count_images() -> tuple[int, int]:
    train = len(list((DATASET_ROOT / "train" / "images").glob("*.*")))
    val = len(list((DATASET_ROOT / "valid" / "images").glob("*.*")))
    return train, val


def _normalize_kaggle_extract(download_dir: Path) -> None:
    """Move images/labels from common Kaggle zip layouts into dataset/cricket_ball_data."""
    candidates = [
        download_dir,
        download_dir / "cricket_ball_data",
        download_dir / "Cricket Ball Dataset for YOLO",
    ]
    for root in candidates:
        if not root.exists():
            continue
        for split in ("train", "valid"):
            for sub in ("images", "labels"):
                src = root / split / sub
                if src.is_dir() and any(src.iterdir()):
                    dst = DATASET_ROOT / split / sub
                    for f in src.iterdir():
                        target = dst / f.name
                        if not target.exists():
                            shutil.copy2(f, target)
        # Some zips use flat images + labels folders
        for flat in ("images", "labels"):
            flat_src = root / flat
            if flat_src.is_dir():
                dst = DATASET_ROOT / "train" / flat
                for f in flat_src.iterdir():
                    target = dst / f.name
                    if not target.exists():
                        shutil.copy2(f, target)


def download_kaggle() -> bool:
    try:
        from kaggle import KaggleApi
    except ImportError:
        print("Installing kaggle package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])
        from kaggle import KaggleApi

    dl_dir = _BACKEND / "dataset" / "_kaggle_download"
    dl_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} ...")

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(dl_dir),
        unzip=True,
        quiet=False,
    )

    zips = list(dl_dir.glob("*.zip"))
    for z in zips:
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(dl_dir)
    _normalize_kaggle_extract(dl_dir)
    return True


def validate() -> bool:
    train_n, val_n = _count_images()
    ok = train_n > 0 and val_n > 0
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"  train/images: {train_n}")
    print(f"  valid/images: {val_n}")
    if ok:
        print("OK - ready for training.")
    else:
        print("\nDataset missing or empty. Download it first:\n")
        print("  Option A - Kaggle CLI (recommended):")
        print("    1. Create API token: https://www.kaggle.com/settings -> Create New Token")
        print("    2. Save kaggle.json to %USERPROFILE%\\.kaggle\\kaggle.json")
        print("    3. Run: python scripts/setup_dataset.py --kaggle")
        print("\n  Option B - Manual:")
        print("    1. Download zip from:")
        print("       https://www.kaggle.com/datasets/kushagra3204/cricket-ball-dataset-for-yolo")
        print(f"    2. Extract so you have:")
        print(f"       {DATASET_ROOT / 'train' / 'images'}")
        print(f"       {DATASET_ROOT / 'valid' / 'images'}")
        print("    3. Run: python scripts/setup_dataset.py --check")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Setup cricket ball YOLO dataset")
    parser.add_argument("--kaggle", action="store_true", help="Download from Kaggle")
    parser.add_argument("--check", action="store_true", help="Only validate paths")
    args = parser.parse_args()

    _ensure_layout()

    if args.kaggle:
        try:
            download_kaggle()
        except Exception as exc:
            print(f"Kaggle download failed: {exc}")
            print("Use manual download instructions below.")
            validate()
            sys.exit(1)

    if not validate():
        sys.exit(1)


if __name__ == "__main__":
    main()
