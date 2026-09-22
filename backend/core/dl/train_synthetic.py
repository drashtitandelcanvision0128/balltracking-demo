"""Train DL models on synthetic cricket trajectories (bootstrap when no weights exist)."""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from core.classifier import LENGTH_BOUNDARIES_M
from core.dl.models import BounceTemporalNet, HitClassifierNet, LENGTH_CLASSES, LengthClassifierNet

SEQ_LEN = 48
FEAT_DIM = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _synthetic_bounce_track(
    height: int = 1080,
    width: int = 1080,
    n_frames: int = 32,
) -> tuple[np.ndarray, int]:
    """Parabolic fall + bounce rise; returns features and bounce index."""
    start_x = random.uniform(width * 0.35, width * 0.65)
    start_y = random.uniform(height * 0.15, height * 0.35)
    bounce_i = random.randint(max(8, n_frames // 4), min(n_frames - 6, int(n_frames * 0.72)))
    pts = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        x = start_x + random.uniform(-0.5, 0.5) * width * 0.02 * t
        if i <= bounce_i:
            prog = i / max(bounce_i, 1)
            y = start_y + (height * random.uniform(0.35, 0.55) - start_y) * (prog ** 1.1)
        else:
            prog = (i - bounce_i) / max(n_frames - bounce_i - 1, 1)
            y_peak = pts[-1][1] if pts else start_y
            y = y_peak - height * random.uniform(0.04, 0.14) * prog
        x += random.gauss(0, 2.5)
        y += random.gauss(0, 2.5)
        pts.append((x, y))
    feats = _track_to_features(pts, height, width)
    return feats, bounce_i


def _track_to_features(pts: list[tuple[float, float]], height: int, width: int) -> np.ndarray:
    arr = np.zeros((SEQ_LEN, FEAT_DIM), dtype=np.float32)
    n = min(len(pts), SEQ_LEN)
    for i in range(n):
        x, y = pts[i]
        dx = pts[i][0] - pts[i - 1][0] if i > 0 else 0.0
        dy = pts[i][1] - pts[i - 1][1] if i > 0 else 0.0
        arr[i] = [
            x / max(width, 1),
            y / max(height, 1),
            dx / max(height * 0.02, 1),
            dy / max(height * 0.02, 1),
            math.hypot(dx, dy) / max(height * 0.02, 1),
        ]
    return arr


def _synthetic_hit_features(hit: bool) -> np.ndarray:
    """24-dim feature vector mimicking trajectory + pose cues."""
    v = np.random.randn(24).astype(np.float32) * 0.3
    if hit:
        v[0:4] = np.abs(v[0:4]) + np.random.uniform(0.4, 1.2, 4)
        v[8] = random.uniform(0.5, 1.0)
    else:
        v[0:4] *= 0.2
        v[8] = random.uniform(0.0, 0.25)
    return v


def _length_class_for_ym(y_m: float) -> int:
    for i, name in enumerate(LENGTH_CLASSES):
        if name == "FULL TOSS":
            continue
        lo, hi = LENGTH_BOUNDARIES_M[name]
        if lo <= y_m < hi:
            return i
    return len(LENGTH_CLASSES) - 1


def train_bounce_model(epochs: int = 40, n_samples: int = 2000) -> BounceTemporalNet:
    model = BounceTemporalNet(SEQ_LEN, FEAT_DIM).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    xs, ys = [], []
    for _ in range(n_samples):
        h = random.choice([720, 1080, 1920])
        w = random.choice([720, 1080, 1920])
        n_f = random.randint(20, SEQ_LEN)
        feats, bi = _synthetic_bounce_track(h, w, n_f)
        label = np.zeros(SEQ_LEN, dtype=np.float32)
        label[bi] = 1.0
        xs.append(feats)
        ys.append(label)

    x_t = torch.tensor(np.stack(xs), device=DEVICE)
    y_t = torch.tensor(np.stack(ys), device=DEVICE)
    loader = DataLoader(TensorDataset(x_t, y_t), batch_size=64, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


def train_hit_model(epochs: int = 30, n_samples: int = 3000) -> HitClassifierNet:
    model = HitClassifierNet(24).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    xs, ys = [], []
    for _ in range(n_samples):
        hit = random.random() < 0.45
        xs.append(_synthetic_hit_features(hit))
        ys.append(1.0 if hit else 0.0)
    x_t = torch.tensor(np.stack(xs), dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(ys, dtype=torch.float32, device=DEVICE)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()
    model.eval()
    return model


def train_length_model(epochs: int = 30, n_samples: int = 4000) -> LengthClassifierNet:
    model = LengthClassifierNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    xs, ys = [], []
    for _ in range(n_samples):
        y_m = random.uniform(0.2, 18.0)
        x_m = random.uniform(-1.2, 1.2)
        xs.append([x_m / 1.5, y_m / 20.12])
        ys.append(_length_class_for_ym(y_m))
    x_t = torch.tensor(np.stack(xs), dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(ys, dtype=torch.long, device=DEVICE)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()
    model.eval()
    return model


def ensure_dl_weights(weights_dir: str | Path) -> Path:
    """Train and save bootstrap weights if missing."""
    root = Path(weights_dir)
    root.mkdir(parents=True, exist_ok=True)
    bounce_p = root / "bounce_temporal.pt"
    hit_p = root / "hit_classifier.pt"
    length_p = root / "length_classifier.pt"

    if not bounce_p.exists():
        print("[DL] Training bounce temporal net (synthetic bootstrap)...", flush=True)
        m = train_bounce_model()
        torch.save({"state_dict": m.state_dict(), "seq_len": SEQ_LEN, "feat_dim": FEAT_DIM}, bounce_p)

    if not hit_p.exists():
        print("[DL] Training hit classifier (synthetic bootstrap)...", flush=True)
        m = train_hit_model()
        torch.save({"state_dict": m.state_dict()}, hit_p)

    if not length_p.exists():
        print("[DL] Training length classifier (synthetic bootstrap)...", flush=True)
        m = train_length_model()
        torch.save({"state_dict": m.state_dict()}, length_p)

    return root
