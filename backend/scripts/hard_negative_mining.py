"""
Hard-negative mining for cricket ball detector.

Usage (from backend folder):
  python scripts/hard_negative_mining.py --source uploads
  python scripts/hard_negative_mining.py --source C:/path/to/practice/videos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from ultralytics import YOLO

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

DEFAULT_SOURCES = [
    _BACKEND / "uploads",
    _BACKEND.parent / "videos",
    _BACKEND / "videos",
]


def _iter_frames(source: Path, stride: int):
    if source.suffix.lower() in IMAGE_EXTS:
        img = cv2.imread(str(source))
        if img is not None:
            yield 0, img
        return
    if source.suffix.lower() not in VIDEO_EXTS:
        return
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            yield idx, frame
        idx += 1
    cap.release()


def _collect_files(source_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for source_dir in source_dirs:
        if not source_dir.is_dir():
            continue
        files.extend(
            p for p in source_dir.rglob("*")
            if p.suffix.lower() in VIDEO_EXTS | IMAGE_EXTS
        )
    return files


def mine(source_dirs: list[Path], model_path: str, out_dir: Path, conf: float, stride: int, max_per_file: int):
    model_file = Path(model_path)
    if not model_file.is_file():
        model_file = _BACKEND / model_path
    if not model_file.is_file():
        # Fall back to pretrained checkpoint (auto-downloaded by Ultralytics)
        print(f"Note: {model_path} not found, using yolov8m.pt")
        model_file = Path("yolov8m.pt")

    files = _collect_files(source_dirs)
    if not files:
        print("ERROR: No videos/images found in:")
        for d in source_dirs:
            print(f"  {d}")
        print("\nPut practice/match videos in one of these folders, or pass --source:")
        print("  python scripts/hard_negative_mining.py --source C:/your/videos/folder")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    model = YOLO(str(model_file))
    saved = 0
    print(f"Scanning {len(files)} file(s)...")

    for fpath in files:
        count = 0
        for frame_idx, frame in _iter_frames(fpath, stride):
            results = model.predict(frame, conf=conf, imgsz=1280, verbose=False, max_det=5)
            for box in results[0].boxes:
                if int(box.cls[0].item()) != 0:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                pad = 8
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                name = f"{fpath.stem}_f{frame_idx}_{count}.jpg"
                cv2.imwrite(str(images_dir / name), crop)
                saved += 1
                count += 1
                if count >= max_per_file:
                    break
        if count:
            print(f"[{fpath.name}] saved {count} hard negatives")

    labels_dir = out_dir / "labels"
    labels_dir.mkdir(exist_ok=True)
    for img in images_dir.glob("*.jpg"):
        (labels_dir / f"{img.stem}.txt").touch()

    print(f"Done. {saved} hard-negative crops in {images_dir}")
    if saved == 0:
        print("No false positives found (good model, or videos have no tricky backgrounds).")
    else:
        print("Copy images into dataset/cricket_ball_data/train/images (empty .txt labels = no ball).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        help="Folder with videos (repeatable). Default: uploads, videos/",
    )
    parser.add_argument("--model", default="yolov8m.pt", help="YOLO weights (auto-downloads if missing)")
    parser.add_argument("--out", default="dataset/hard_negatives")
    parser.add_argument("--conf", type=float, default=0.12)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-per-file", type=int, default=30)
    args = parser.parse_args()

    if args.source:
        sources = [Path(s).resolve() for s in args.source]
    else:
        sources = DEFAULT_SOURCES

    out = Path(args.out)
    if not out.is_absolute():
        out = _BACKEND / out

    mine(sources, args.model, out, args.conf, args.stride, args.max_per_file)


if __name__ == "__main__":
    main()
