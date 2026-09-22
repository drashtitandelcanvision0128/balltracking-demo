"""PyTorch models — temporal bounce, hit, and length classifiers."""

from __future__ import annotations

import torch
import torch.nn as nn

LENGTH_CLASSES = (
    "FULL TOSS",
    "YORKER",
    "FULL LENGTH",
    "GOOD LENGTH",
    "BACK OF LENGTH",
    "SHORT BALL",
    "BOUNCER",
)


class BounceTemporalNet(nn.Module):
    """1D-CNN + BiLSTM — per-frame bounce probability from ball track."""

    def __init__(self, seq_len: int = 48, feat_dim: int = 5) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.encoder = nn.Sequential(
            nn.Conv1d(feat_dim, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        h = self.encoder(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(h)
        return torch.sigmoid(self.head(h).squeeze(-1))


class HitClassifierNet(nn.Module):
    """MLP on trajectory + pose features for bat-ball contact."""

    def __init__(self, in_dim: int = 24) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x).squeeze(-1))


class LengthClassifierNet(nn.Module):
    """MLP on pitch-world coordinates → length zone."""

    def __init__(self, n_classes: int = len(LENGTH_CLASSES)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
