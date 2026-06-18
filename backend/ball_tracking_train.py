"""
YOLOv8 cricket ball training — delegates to improved training script.

Usage:
  python ball_tracking_train.py
  python ball_tracking_train.py --epochs 120 --model yolov8l.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.train_improved import main

if __name__ == "__main__":
    main()
