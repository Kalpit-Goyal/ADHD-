"""EEG models: a raw-signal CNN baseline and a CTE-Net-style graph transformer."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import ModelConfig


class EEGNet(nn.Module):
    """Compact EEGNet-style convolutional network on raw epochs.

    Input shape: (batch, 1, n_channels, n_times).
    """

    def __init__(
        self,
        n_channels: int = 19,
        n_times: int = 640,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times

        self.block1 = nn.Sequential(
            nn.Conv2d(
                1,
                f1,
                kernel_size=(1, kernel_length),
                padding=(0, kernel_length // 2),
                bias=False,
            ),
            nn.BatchNorm2d(f1),
            nn.Conv2d(
                f1,
                f1 * depth_multiplier,
                kernel_size=(n_channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                f1 * depth_multiplier,
                f1 * depth_multiplier,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=f1 * depth_multiplier,
                bias=False,
            ),
            nn.Conv2d(
                f1 * depth_multiplier,
                f2,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(f2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"EEGNet expects (B, 1, C, T), got {tuple(x.shape)}")
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        return self.head(x).squeeze(-1)


class GraphAttentionLayer(nn.Module):
    """Multi-head attention across EEG channels with a connectivity bias."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.25):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.edge_bias = nn.Linear(1, n_heads)
        self.output = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, channels, d_model = x.shape

        query = self.query(x).view(batch, channels, self.n_heads, self.head_dim)
        key = self.key(x).view(batch, channels, self.n_heads, self.head_dim)
        value = self.value(x).view(batch, channels, self.n_heads, self.head_dim)

        query = query.transpose(1, 2)  # (B, H, C, Dh)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        attention = torch.matmul(query, key.transpose(-1, -2))
        attention = attention / math.sqrt(self.head_dim)

        bias = self.edge_bias(adjacency.unsqueeze(-1))  # (B, C, C, H)
        bias = bias.permute(0, 3, 1, 2)                 # (B, H, C, C)
        attention = attention + bias

        attention = torch.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        context = torch.matmul(attention, value)
        context = context.transpose(1, 2).reshape(batch, channels, d_model)

        x = self.norm1(x + self.dropout(self.output(context)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class CTENet(nn.Module):
    """Transformer over channel-wise spectral features + directed connectivity.

    Inputs
    ------
    x : (batch, n_channels, n_features)
        Per-channel band-power features.
    adjacency : (batch, n_channels, n_channels)
        Directed connectivity; ``adjacency[target, source]``.

    This mirrors the modeling idea in CTE-Net (Gomez-Rivera et al., 2026):
    treat electrodes as graph nodes and use a transformer with connectivity
    bias. It is a clean reimplementation, not the authors' code.
    """

    def __init__(
        self,
        n_features: int,
        n_channels: int = 19,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.node_projection = nn.Linear(n_features, d_model)
        self.layers = nn.ModuleList(
            [
                GraphAttentionLayer(d_model, n_heads, dropout)
                for _ in range(n_layers)
            ]
        )
        self.pool_score = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"CTENet expects (B, C, F), got {tuple(x.shape)}")

        hidden = self.node_projection(x)
        for layer in self.layers:
            hidden = layer(hidden, adjacency)

        scores = self.pool_score(hidden).squeeze(-1)  # (B, C)
        weights = torch.softmax(scores, dim=1)
        pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(pooled).squeeze(-1)


def build_model(
    config: ModelConfig,
    n_channels: int,
    n_features: int,
    n_times: int,
) -> nn.Module:
    """Construct the requested model."""
    name = config.name.lower()

    if name == "eegnet":
        return EEGNet(
            n_channels=n_channels,
            n_times=n_times,
            f1=config.eegnet_f1,
            depth_multiplier=config.eegnet_d,
            f2=config.eegnet_f2,
            kernel_length=config.eegnet_kernel,
            dropout=config.dropout,
        )

    if name in {"cte_net", "ctenet", "graph"}:
        return CTENet(
            n_features=n_features,
            n_channels=n_channels,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout,
        )

    raise ValueError(f"Unknown model: {config.name}")
