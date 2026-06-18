"""
Wait for YOLO training to finish, then update config.yaml automatically.

Usage (run in parallel while training):
  python scripts/wait_and_deploy.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import yaml

_BACKEND = Path(__file__).resolve().parent.parent
RUN_DIR = _BACKEND / "runs" / "detect" / "train_improved"
RESULTS = RUN_DIR / "results.csv"
BEST_PT = RUN_DIR / "weights" / "best.pt"
CONFIG = _BACKEND / "config.yaml"
TARGET_EPOCHS = 100
POLL_SEC = 60


def _epoch_count() -> int:
    if not RESULTS.is_file():
        return 0
    lines = RESULTS.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    return max(0, len(lines) - 1)  # minus header


def _deploy() -> None:
    rel = "runs/detect/train_improved/weights/best.pt"
    if not BEST_PT.is_file():
        print(f"ERROR: {BEST_PT} not found")
        sys.exit(1)

    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("model", {})["path"] = rel
    with open(CONFIG, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"Deployed: config.yaml model.path -> {rel}")
    print(f"Weights: {BEST_PT}")


def main():
    print(f"Watching training: {RUN_DIR}")
    print(f"Target: {TARGET_EPOCHS} epochs. Polling every {POLL_SEC}s...")
    print("Do not close this window or the training terminal.\n")

    last_epoch = -1
    while True:
        ep = _epoch_count()
        if ep != last_epoch and ep > 0:
            pct = round(100 * ep / TARGET_EPOCHS, 1)
            print(f"Progress: epoch {ep}/{TARGET_EPOCHS} ({pct}%)")
            last_epoch = ep

        if ep >= TARGET_EPOCHS and BEST_PT.is_file():
            print("\nTraining finished!")
            _deploy()
            return

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
