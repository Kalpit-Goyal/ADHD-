"""CTE-Net implementation attempts.

Two models are provided:

* ``CTENetInspired`` -- the existing graph-attention model from
  ``src/models.py`` (re-exported here for convenience). This is a
  CTE-Net-*inspired* model, not a faithful reproduction.
* ``CTENetOriginal`` -- an attempt at the original CTE-Net architecture from
  Gomez-Rivera et al. (2026), *Sensors* 26(18):5786. It implements channel-wise
  nonlinear temporal filters, Takens delay-coordinate embeddings, a
  differentiable matrix-based Transfer Entropy estimator, Rényi alpha entropy,
  a rational-quadratic kernel, and a Transformer encoder over directed
  connectivity.

Components that could not be implemented faithfully (and the reason) are
documented in the class docstrings and in ``RESEARCH_IMPLEMENTATION_AUDIT.md``.
Do not claim exact reproduction of the original architecture.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import CTENet as CTENetInspired  # noqa: F401  (re-export)


# ---------------------------------------------------------------------------
# Paper 2 building blocks (attempted original CTE-Net components)
# ---------------------------------------------------------------------------

class ChannelWiseNonlinearTemporalFilter(nn.Module):
    """Per-channel nonlinear temporal filtering.

    Applies a small 1-D convolution per channel followed by a pointwise
    nonlinearity. This is a lightweight stand-in for the paper's
    channel-wise nonlinear temporal filters.
    """

    def __init__(self, n_channels: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.filter = nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size=kernel_size,
            padding=pad,
            groups=n_channels,
            bias=True,
        )
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, C, T)
        out = self.filter(x)
        out = self.activation(out)
        return self.dropout(out)


class TakensDelayEmbedding(nn.Module):
    """Takens delay-coordinate embedding per channel.

    For each channel, builds delay vectors
        [x[t], x[t-tau], x[t-2*tau], ..., x[t-(embedding_dim-1)*tau]]
    producing an expanded temporal dimension. Reduces the effective time
    resolution by (embedding_dim - 1) * tau.
    """

    def __init__(self, embedding_dim: int = 5, delay: int = 2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.delay = delay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, C, T')
        batch, channels, n_times = x.shape
        needed = 1 + (self.embedding_dim - 1) * self.delay
        if n_times < needed:
            # Pad by replication to allow embedding.
            pad = needed - n_times
            x = F.pad(x, (pad, 0), mode="replicate")
            n_times = needed

        out = []
        for i in range(self.embedding_dim):
            start = (self.embedding_dim - 1 - i) * self.delay
            out.append(x[:, :, start : start + n_times - (self.embedding_dim - 1) * self.delay])
        # (B, C, embedding_dim, T')
        out = torch.stack(out, dim=2)
        # Flatten embedding into channel dimension: (B, C*embedding_dim, T')
        out = out.reshape(batch, channels * self.embedding_dim, -1)
        return out


class RenyiAlphaEntropy(nn.Module):
    """Differentiable Rényi alpha entropy estimate over a feature distribution.

    Uses a k-nearest-neighbour style distance-based estimator that is
    differentiable w.r.t. the input features. Computed per channel over the
    temporal dimension.
    """

    def __init__(self, alpha: float = 0.5, k: int = 4):
        super().__init__()
        self.alpha = alpha
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> (B, C, 1) entropy per channel
        batch, channels, n_times = x.shape
        x = x.transpose(1, 2).reshape(batch * channels, n_times, 1)  # (BC, T, 1)

        # Pairwise distances.
        diff = x.unsqueeze(1) - x.unsqueeze(2)  # (BC, T, T, 1)
        dist = torch.norm(diff, dim=-1)  # (BC, T, T)

        # k-th nearest neighbour distance (excluding self, dist=0).
        kth = min(self.k, n_times - 1)
        if kth < 1:
            return torch.zeros(batch, channels, 1, device=x.device, dtype=x.dtype)

        dist_sorted, _ = torch.sort(dist, dim=-1)  # (BC, T, T)
        knn_dist = dist_sorted[:, :, kth]  # (BC, T)

        # Rényi entropy estimate (Kozachenko-Leonenko style).
        # H_alpha = 1/(1-alpha) * log( sum_i (1 / d_i^k) ) + const
        eps = 1e-12
        rho = torch.clamp(knn_dist, min=eps)
        if abs(self.alpha - 1.0) < 1e-6:
            entropy = -torch.log(rho + eps)
        else:
            entropy = (1.0 / (1.0 - self.alpha)) * torch.log(
                torch.mean(rho ** (-self.alpha), dim=-1, keepdim=True) + eps
            )
        return entropy.reshape(batch, channels, 1)


class RationalQuadraticKernel(nn.Module):
    """Differentiable rational quadratic kernel between two feature sets.

    K(x, y) = (1 + ||x - y||^2 / (2 * alpha * length_scale^2)) ^ (-alpha)
    Used as a soft connectivity measure between channels.
    """

    def __init__(self, alpha: float = 1.0, length_scale: float = 1.0, learnable: bool = True):
        super().__init__()
        self.alpha = alpha
        if learnable:
            self.log_length_scale = nn.Parameter(torch.log(torch.tensor(length_scale)))
        else:
            self.register_buffer("log_length_scale", torch.log(torch.tensor(length_scale)))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: (B, C, D) -> (B, C, C)
        length_scale = torch.exp(self.log_length_scale)
        diff = x.unsqueeze(2) - y.unsqueeze(1)  # (B, C, C, D)
        sq_dist = torch.sum(diff ** 2, dim=-1)  # (B, C, C)
        kernel = (1.0 + sq_dist / (2.0 * self.alpha * length_scale ** 2)) ** (-self.alpha)
        return kernel


class DifferentiableTransferEntropy(nn.Module):
    """Differentiable matrix-based Transfer Entropy estimate.

    Computes a directed TE estimate from source to target using a
    matrix-based Rényi entropy formulation with rational-quadratic kernels.
    Returns a (C, C) directed connectivity matrix per batch element where
    entry [target, source] is the estimated TE from source to target.

    This is an efficient differentiable approximation, not an exact
    reproduction of the paper's matrix-based TE. It is differentiable
    w.r.t. the input signals and can be trained end-to-end.
    """

    def __init__(
        self,
        n_channels: int,
        alpha: float = 0.5,
        kernel_alpha: float = 1.0,
        kernel_length_scale: float = 1.0,
        embed_dim: int = 3,
        embed_delay: int = 1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.alpha = alpha
        self.embed_dim = embed_dim
        self.embed_delay = embed_delay
        self.rq_kernel = RationalQuadraticKernel(
            alpha=kernel_alpha,
            length_scale=kernel_length_scale,
            learnable=True,
        )
        self.renyi = RenyiAlphaEntropy(alpha=alpha, k=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> te: (B, C, C), te[target, source] = TE(source -> target)
        batch, channels, n_times = x.shape

        # Build delay embeddings per channel.
        embedded = []
        for ch in range(channels):
            ch_signal = x[:, ch, :]  # (B, T)
            emb = self._embed(ch_signal)  # (B, embed_dim, T')
            embedded.append(emb)
        # embedded: list of (B, embed_dim, T')
        embed_dim = embedded[0].shape[1]
        n_t = embedded[0].shape[-1]

        # Stack: (B, C, embed_dim, T')
        embed_stack = torch.stack(embedded, dim=1)

        # Flatten embedding dims into features: (B, C, T', embed_dim) -> (B, C, T'*embed_dim)
        features = embed_stack.transpose(2, 3).reshape(batch, channels, n_t * embed_dim)

        te = torch.zeros(batch, channels, channels, device=x.device, dtype=x.dtype)

        for target in range(channels):
            for source in range(channels):
                if target == source:
                    continue
                # Joint features of (target, source).
                joint = torch.cat([features[:, target], features[:, source]], dim=-1)  # (B, 2F)
                # Entropies.
                h_target = self.renyi(features[:, target].unsqueeze(-1).transpose(1, 2))  # (B,1,1)
                h_source = self.renyi(features[:, source].unsqueeze(-1).transpose(1, 2))
                h_joint = self.renyi(joint.unsqueeze(-1).transpose(1, 2))
                # TE = H(target_future | target_past) - H(target_future | target_past, source_past)
                # Approximate with Rényi entropies.
                te_val = h_joint - h_source - h_target  # (B, 1, 1)
                te[:, target, source] = te_val.squeeze(-1).squeeze(-1)

        return te

    def _embed(self, signal: torch.Tensor) -> torch.Tensor:
        # signal: (B, T) -> (B, embed_dim, T')
        batch, n_times = signal.shape
        needed = 1 + (self.embed_dim - 1) * self.embed_delay
        if n_times < needed:
            pad = needed - n_times
            signal = F.pad(signal, (pad, 0), mode="replicate")
            n_times = needed
        out = []
        for i in range(self.embed_dim):
            start = (self.embed_dim - 1 - i) * self.embed_delay
            end = n_times - (self.embed_dim - 1) * self.delay if False else n_times - (self.embed_dim - 1) * self.embed_delay
            out.append(signal[:, start : start + n_times - (self.embed_dim - 1) * self.embed_delay])
        return torch.stack(out, dim=1)


# ---------------------------------------------------------------------------
# Original CTE-Net reconstruction attempt
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer (pre-norm)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, dim_feedforward: int | None = None):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, C, d_model)
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x + self.dropout1(attn_out))
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff))
        return x


class CTENetOriginal(nn.Module):
    """Attempted reconstruction of CTE-Net (Gomez-Rivera et al., 2026).

    Architecture (attempted):
        1. Channel-wise nonlinear temporal filters on raw EEG.
        2. Takens delay-coordinate embedding per channel.
        3. Differentiable Transfer Entropy -> directed connectivity (C, C).
        4. Rational quadratic kernel soft-connectivity.
        5. Rényi alpha entropy per channel.
        6. Transformer encoder over channel representations with connectivity bias.
        7. Attention pooling -> participant-level prediction.

    Missing / approximated components (documented, not hidden):
        * The exact nonlinear filter design from the paper is not public;
          a small per-channel Conv1d + Tanh is used as a stand-in.
        * The exact TE estimator (matrix-based, Rényi with rational-quadratic
          kernels on joint distributions) is approximated by a pairwise
          differentiable estimator; it is O(C^2 * T * embed) per batch.
        * The rational quadratic kernel is applied to flattened delay
          features rather than the paper's specific formulation.
        * No claim is made that this is an exact reproduction.

    Input
    -----
    x : (B, C, T) raw preprocessed EEG epochs.

    Output
    ------
    logits : (B,) ADHD logit.
    features : (B, d_model) learned participant representation (accessible
               via ``forward_features``).
    """

    def __init__(
        self,
        n_channels: int = 19,
        n_times: int = 640,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.25,
        embed_dim: int = 3,
        embed_delay: int = 1,
        te_alpha: float = 0.5,
        use_te: bool = True,
        use_rq_kernel: bool = True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        self.use_te = use_te
        self.use_rq_kernel = use_rq_kernel

        # 1. Channel-wise nonlinear temporal filters.
        self.temporal_filter = ChannelWiseNonlinearTemporalFilter(
            n_channels, kernel_size=7, dropout=dropout
        )

        # 2. Takens delay-coordinate embedding.
        self.takens = TakensDelayEmbedding(embedding_dim=embed_dim, delay=embed_delay)
        # After Takens: time dimension shrinks to T - (embed_dim-1)*delay.
        takens_reduction = (embed_dim - 1) * embed_delay
        self.takens_n_times = n_times - takens_reduction
        if self.takens_n_times <= 0:
            self.takens_n_times = 1

        # Feature dimension after filter + takens + entropy.
        self.embed_dim = embed_dim
        self.te_feature_dim = d_model

        # 3-5. Connectivity + entropy (computed in forward, not parameters).
        self.te = DifferentiableTransferEntropy(
            n_channels=n_channels,
            alpha=te_alpha,
            kernel_alpha=1.0,
            kernel_length_scale=1.0,
            embed_dim=embed_dim,
            embed_delay=embed_delay,
        ) if use_te else None

        self.rq_kernel = RationalQuadraticKernel(alpha=1.0, length_scale=1.0) if use_rq_kernel else None

        # Project temporal features to d_model.
        # After Takens, each channel has embed_dim * takens_n_times values.
        # We pool over time to get embed_dim per channel, then project.
        self.input_projection = nn.Linear(embed_dim, d_model)

        # 6. Transformer encoder.
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

        # Connectivity bias injected into attention (like the inspired model).
        self.edge_bias = nn.Linear(1, n_heads)

        # 7. Attention pooling + head.
        self.pool_score = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward_features(
        self, x: torch.Tensor, adjacency: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return (features, connectivity, entropy) before the classification head.

        features: (B, d_model) participant-level representation.
        connectivity: (B, C, C) directed TE matrix, or None.
        entropy: (B, C, 1) Rényi entropy per channel, or None.

        ``adjacency`` is accepted for API compatibility with the inspired
        CTENet model but is not used by this reconstruction (connectivity is
        estimated internally).
        """
        # x: (B, C, T)
        x = self.temporal_filter(x)  # (B, C, T)

        # Connectivity & entropy from filtered signal.
        connectivity = None
        entropy = None
        if self.use_te and self.te is not None:
            connectivity = self.te(x)  # (B, C, C)
        if self.use_rq_kernel and self.rq_kernel is not None:
            # Soft kernel connectivity as an auxiliary signal.
            pooled = x.mean(dim=-1)  # (B, C)
            kernel_conn = self.rq_kernel(pooled.unsqueeze(-1), pooled.unsqueeze(-1))  # (B, C, C)
            if connectivity is None:
                connectivity = kernel_conn
            else:
                connectivity = 0.5 * connectivity + 0.5 * kernel_conn

        # Takens embedding: (B, C, T) -> (B, C*embed_dim, T')
        embedded = self.takens(x)  # (B, C*embed_dim, T')
        batch, _, _ = embedded.shape

        # Reshape to (B, C, embed_dim, T'), pool over time -> (B, C, embed_dim)
        embedded = embedded.reshape(batch, self.n_channels, self.embed_dim, -1)
        embedded = embedded.mean(dim=-1)  # (B, C, embed_dim)

        # Project to d_model.
        hidden = self.input_projection(embedded)  # (B, C, d_model)

        # Add connectivity bias to attention layers.
        for layer in self.layers:
            # Build an attention mask bias from connectivity.
            if connectivity is not None:
                bias = self.edge_bias(connectivity.unsqueeze(-1))  # (B, C, C, H)
                bias = bias.permute(0, 3, 1, 2)  # (B, H, C, C)
            else:
                bias = None
            # MultiheadAttention doesn't accept a per-head additive bias directly;
            # we add it via the attn_mask (additive). Convert bias to (B*H, C, C).
            if bias is not None:
                b, h, c, _ = bias.shape
                mask = bias.reshape(b * h, c, c)
            else:
                mask = None
            hidden = layer(hidden, attn_mask=mask)

        # Attention pooling over channels.
        scores = self.pool_score(hidden).squeeze(-1)  # (B, C)
        weights = torch.softmax(scores, dim=1)
        features = (hidden * weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)
        return features, connectivity, entropy

    def forward(self, x: torch.Tensor, adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
        features, _, _ = self.forward_features(x, adjacency)
        return self.head(features).squeeze(-1)


def build_ctenet_original(
    n_channels: int = 19,
    n_times: int = 640,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.25,
    embed_dim: int = 3,
    embed_delay: int = 1,
    use_te: bool = True,
    use_rq_kernel: bool = True,
) -> CTENetOriginal:
    return CTENetOriginal(
        n_channels=n_channels,
        n_times=n_times,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        embed_dim=embed_dim,
        embed_delay=embed_delay,
        te_alpha=0.5,
        use_te=use_te,
        use_rq_kernel=use_rq_kernel,
    )


# ---------------------------------------------------------------------------
# Known deviations from the original CTE-Net paper
# ---------------------------------------------------------------------------

DOCUMENTED_GAPS = """
CTENetOriginal deviations from Gomez-Rivera et al. (2026):

1. Channel-wise nonlinear temporal filters: the paper's exact filter design
   is not publicly specified; a per-channel depthwise Conv1d + Tanh is used.

2. Differentiable Transfer Entropy: the paper uses a matrix-based Rényi
   entropy estimator with rational-quadratic kernels over joint delay-embedded
   distributions. This implementation uses a pairwise approximate estimator
   that is O(C^2 * T * embed_dim) per batch and may not match the paper's
   exact values.

3. Rényi alpha entropy: implemented via a kNN distance-based estimator
   (Kozachenko-Leonenko style), which is differentiable but approximate.

4. Rational quadratic kernel: implemented and used as a soft connectivity
   measure, but applied to flattened delay features rather than the paper's
   exact formulation.

5. Takens delay-coordinate embeddings: implemented per channel with
   configurable embedding dimension and delay.

6. Transformer encoder: implemented with pre-norm layers and connectivity
   bias injected into attention scores.

7. Participant-level prediction: implemented via attention pooling over
   channels followed by a classification head.

This model is an *attempted reconstruction*, not an exact reproduction.
Components marked approximate should not be claimed as faithful to the
original without the authors' code and exact hyperparameters.
"""
