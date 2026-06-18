"""
Improved YOLOv8 training for cricket ball detection.

Usage (from backend folder — you are already here, do NOT cd backend again):
  python scripts/setup_dataset.py --kaggle
  python scripts/train_improved.py --data data.yaml --epochs 100 --model yolov8m.pt --imgsz 1280
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import yaml
from ultralytics import YOLO


def _resolve_dataset(data_arg: str) -> Path:
    data_path = Path(data_arg)
    if not data_path.is_absolute():
        data_path = (_BACKEND / data_path).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")
    return data_path


def _validate_dataset(data_yaml: Path) -> None:
    with open(data_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base = cfg.get("path", "")
    if not os.path.isabs(base):
        base = str((data_yaml.parent / base).resolve())
    else:
        base = str(Path(base).resolve())

    missing = []
    for key in ("train", "val"):
        rel = cfg.get(key) or cfg.get("valid")
        if not rel:
            continue
        full = Path(base) / rel
        if not full.is_dir() or not any(full.glob("*")):
            missing.append(str(full))

    if missing:
        print("ERROR: Dataset images not found.\n")
        for m in missing:
            print(f"  Missing: {m}")
        print("\nFix:")
        print("  python scripts/setup_dataset.py --kaggle")
        print("  — or download manually and extract to backend/dataset/cricket_ball_data/")
        sys.exit(1)

    train_n = len(list((Path(base) / cfg["train"]).glob("*.*")))
    val_key = cfg.get("val") or cfg.get("valid")
    val_n = len(list((Path(base) / val_key).glob("*.*"))) if val_key else 0
    print(f"Dataset OK — train: {train_n} images, val: {val_n} images")


def main():
    os.chdir(_BACKEND)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--model", default="yolov8m.pt", help="yolov8m.pt or yolov8l.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4, help="Use 4 on CPU, 8+ on GPU")
    parser.add_argument("--name", default="train_improved")
    parser.add_argument("--device", default="", help="cuda, cpu, or 0 for GPU")
    parser.add_argument("--cache", action="store_true", help="Cache images for faster epochs")
    args = parser.parse_args()

    data_yaml = _resolve_dataset(args.data)
    _validate_dataset(data_yaml)

    device = args.device
    if not device:
        try:
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    print(f"Device: {device}")

    project = str(_BACKEND / "runs" / "detect")
    model = YOLO(args.model)

    train_kw = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=25,
        project=project,
        name=args.name,
        exist_ok=True,
        mosaic=1.0,
        copy_paste=0.15,
        degrees=5.0,
        translate=0.1,
        scale=0.6,
        shear=2.0,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        close_mosaic=15,
        amp=True,
        cache="disk" if args.cache else False,
        workers=4,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        device=device,
    )

    print(f"Training -> {project}/{args.name}")
    results = model.train(**train_kw)
    best = Path(results.save_dir) / "weights" / "best.pt"
    rel = f"runs/detect/{args.name}/weights/best.pt"
    print(f"\nTraining complete.")
    print(f"  Best weights: {best}")

    if best.is_file():
        cfg_path = _BACKEND / "config.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg.setdefault("model", {})["path"] = rel
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"  config.yaml updated -> model.path: {rel}")

    return results


if __name__ == "__main__":
    main()
