"""Train all DL models (synthetic bootstrap) and save to runs/dl/."""

from __future__ import annotations

from core.dl.train_synthetic import ensure_dl_weights

if __name__ == "__main__":
    root = ensure_dl_weights("runs/dl")
    print(f"DL weights ready in {root}")
