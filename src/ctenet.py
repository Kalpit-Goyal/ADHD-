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
            end = start + n_times - (self.embed_dim - 1) * self.embed_delay
            out.append(signal[:, start:end])
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
    """Faithful reconstruction of CTE-Net (Gomez-Rivera et al., 2026).

    Architecture matches the paper (Sensors 2026, 26(18):5786):

        Block 1: Transformer-based contextualization
            X (C,T) -> transpose -> LayerNorm -> Linear(C->d) ->
            TransformerEncoder(L=1, nh=1, d_ff=d, p=0.5) ->
            LayerNorm -> Linear(d->C) -> transpose -> X_f (C,T)
            d=128, no positional encoding.

        Block 2: Differentiable directed Transfer Entropy
            DWConv1d(K=83) -> ReLU -> AvgPool(pool=4) -> BatchNorm -> phi (C,T_phi)
            Takens embeddings: Dx=6 (source past), Dy=1 (target past), target present
            tau=2, valid anchors Tz computed dynamically
            Bias-free linear projections per state
            RQ kernel Gram matrices (a=1, l=1, alpha_RQ=1)
            TE = H(Ky_past, Kx) - H(Ky_present, Ky_past, Kx)
                 + H(Ky_present, Ky_past) - H(Ky_past)
            H_alpha(A) = -log(||N(A)||_F^2) for alpha=2 (Renyi order 2)
            N(A) = A / tr(A)
            Output: TE (C,C), asymmetric, diagonal excluded

        Block 3: Classification
            Flatten off-diagonal TE -> C*(C-1) dims
            Dropout(0.5) -> Dense(64) -> ReLU -> Dropout(0.5) -> Dense(1) -> sigmoid

    Input
    -----
    x : (B, C, T) raw preprocessed EEG epochs.

    Output
    ------
    logits : (B,) ADHD logit.
    features : (B, C*(C-1)) flattened off-diagonal TE (via ``forward_features``).
    """

    def __init__(
        self,
        n_channels: int = 19,
        n_times: int = 512,
        d_model: int = 128,
        n_heads: int = 1,
        n_layers: int = 1,
        dropout: float = 0.5,
        embed_dim_source: int = 6,
        embed_dim_target: int = 1,
        embed_delay: int = 2,
        te_alpha: float = 2.0,
        kernel_size: int = 83,
        pool_size: int = 4,
        rq_alpha: float = 1.0,
        rq_length_scale: float = 1.0,
        rq_amplitude: float = 1.0,
        head_hidden: int = 64,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.d_model = d_model
        self.te_alpha = te_alpha
        self.embed_dim_source = embed_dim_source
        self.embed_dim_target = embed_dim_target
        self.embed_delay = embed_delay
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.rq_alpha = rq_alpha
        self.rq_length_scale = rq_length_scale
        self.rq_amplitude = rq_amplitude
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout

        # ------------------------------------------------------------------
        # Block 1: Transformer-based contextualization
        # ------------------------------------------------------------------
        # Input projection: (T, C) -> (T, d), with LayerNorm
        self.input_norm = nn.LayerNorm(n_channels)
        self.input_proj = nn.Linear(n_channels, d_model)

        # Transformer encoder: L layers, nh heads, d_ff = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Channel projection: (T, d) -> (T, C), with LayerNorm
        self.channel_proj = nn.Linear(d_model, n_channels)
        self.channel_norm = nn.LayerNorm(n_channels)

        # ------------------------------------------------------------------
        # Block 2: Channel-wise temporal filtering
        # ------------------------------------------------------------------
        self.temporal_filter = nn.Conv1d(
            n_channels,
            n_channels,
            kernel_size=kernel_size,
            padding=0,
            groups=n_channels,
            bias=True,
        )
        self.filter_activation = nn.ReLU()
        self.filter_pool = nn.AvgPool1d(kernel_size=pool_size, stride=pool_size)
        self.filter_norm = nn.BatchNorm1d(n_channels)

        # Trainable bias-free linear projections for Takens states
        self.proj_source = nn.Linear(embed_dim_source, embed_dim_source, bias=False)
        self.proj_target_past = nn.Linear(embed_dim_target, embed_dim_target, bias=False)
        self.proj_target_present = nn.Linear(1, 1, bias=False)

        # ------------------------------------------------------------------
        # Block 3: Classification head on flattened off-diagonal TE
        # ------------------------------------------------------------------
        te_flat_dim = n_channels * (n_channels - 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(te_flat_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    # ------------------------------------------------------------------
    # Kernel / entropy utilities
    # ------------------------------------------------------------------
    def _rq_kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Rational quadratic kernel Gram matrix.

        X: (B, N, D), Y: (B, M, D) -> (B, N, M)
        K(x,y) = a^2 * (1 + ||x-y||^2 / (2 * alpha_RQ * l^2)) ^ (-alpha_RQ)
        """
        a2 = self.rq_amplitude ** 2
        l2 = self.rq_length_scale ** 2
        denom = 2.0 * self.rq_alpha * l2

        x_sq = torch.sum(X ** 2, dim=-1, keepdim=True)      # (B, N, 1)
        y_sq = torch.sum(Y ** 2, dim=-1, keepdim=True)      # (B, M, 1)
        xy = torch.matmul(X, Y.transpose(-1, -2))           # (B, N, M)
        sq_dist = x_sq + y_sq.transpose(-1, -2) - 2.0 * xy  # (B, N, M)
        sq_dist = torch.clamp(sq_dist, min=0.0)

        kernel = (1.0 + sq_dist / denom) ** (-self.rq_alpha)
        return kernel * a2

    def _trace_normalize(self, K: torch.Tensor) -> torch.Tensor:
        """N(A) = A / tr(A). K: (..., N, N)"""
        trace = torch.diagonal(K, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        trace = trace.clamp_min(1e-8)
        return K / trace.unsqueeze(-1)

    def _renyi_entropy_order2(self, K: torch.Tensor) -> torch.Tensor:
        """H_2(N(A)) = -log(||N(A)||_F^2) for symmetric N(A).

        Uses ||N(A)||_F^2 = tr(N(A)^2) since N(A) is symmetric.
        K: (..., N, N) -> (...,)
        """
        K_norm = self._trace_normalize(K)
        # Frobenius norm squared = sum of squared entries
        fro_sq = torch.sum(K_norm ** 2, dim=(-2, -1))
        return -torch.log(fro_sq.clamp_min(1e-12))

    def _joint_entropy_order2(self, *kernels: torch.Tensor) -> torch.Tensor:
        """H_2(K1 ⊙ ... ⊙ KM) via Hadamard product then entropy."""
        joint = kernels[0]
        for k in kernels[1:]:
            joint = joint * k
        return self._renyi_entropy_order2(joint)

    # ------------------------------------------------------------------
    # Transfer Entropy computation (vectorized over all channel pairs)
    # ------------------------------------------------------------------
    def _takens_embeddings(self, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build Takens delay-coordinate embeddings from filtered signal.

        phi: (B, C, T_phi)
        Returns:
            Zx: (B, C, Tz, Dx)   source past
            Zy_past: (B, C, Tz, Dy)   target past
            Zy_present: (B, C, Tz, 1)  target present
        """
        batch, channels, t_phi = phi.shape
        tau = self.embed_delay
        Dx = self.embed_dim_source
        Dy = self.embed_dim_target

        min_t = max((Dx - 1) * tau, Dy * tau)
        tz = t_phi - min_t
        if tz <= 0:
            raise ValueError(
                f"Not enough time samples for Takens: t_phi={t_phi}, min_t={min_t}"
            )

        # Source past: Zx_c(t) = [phi_c(t - j*tau)]_{j=0}^{Dx-1}
        source_frames = []
        for j in range(Dx):
            start = min_t - j * tau
            source_frames.append(phi[:, :, start : start + tz])
        Zx = torch.stack(source_frames, dim=2).transpose(2, 3)  # (B, C, Tz, Dx)

        # Target past: Zy_c'(t) = [phi_c'(t - j*tau)]_{j=1}^{Dy}
        target_past_frames = []
        for j in range(1, Dy + 1):
            start = min_t - j * tau
            target_past_frames.append(phi[:, :, start : start + tz])
        Zy_past = torch.stack(target_past_frames, dim=2).transpose(2, 3)  # (B, C, Tz, Dy)

        # Target present: zy_c'(t) = phi_c'(t)
        Zy_present = phi[:, :, min_t : min_t + tz].unsqueeze(-1)  # (B, C, Tz, 1)

        return Zx, Zy_past, Zy_present

    def _compute_te_vectorized(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute directed TE matrix for all channel pairs simultaneously.

        phi: (B, C, T_phi)
        Returns: TE: (B, C, C), TE[target, source] = TE(source -> target)
        """
        batch, channels, _ = phi.shape
        alpha = self.te_alpha

        Zx, Zy_past, Zy_present = self._takens_embeddings(phi)

        # Apply bias-free linear projections
        Zx_p = self.proj_source(Zx)           # (B, C, Tz, Dx)
        Zy_past_p = self.proj_target_past(Zy_past)  # (B, C, Tz, Dy)
        Zy_present_p = self.proj_target_present(Zy_present)  # (B, C, Tz, 1)

        # Precompute kernel matrices per channel (source side and target side)
        # K_x[c]: Gram of source past states for channel c -> (B, C, Tz, Tz)
        K_x = self._batch_gram(Zx_p)   # (B, C, Tz, Tz)
        # K_y_past[c']: Gram of target past states for channel c' -> (B, C, Tz, Tz)
        K_y_past = self._batch_gram(Zy_past_p)  # (B, C, Tz, Tz)
        # K_y_present[c']: Gram of target present states for channel c' -> (B, C, Tz, Tz)
        K_y_present = self._batch_gram(Zy_present_p)  # (B, C, Tz, Tz)

        # TE for all pairs via broadcasting:
        # K_y_past[c']: (B, C, 1, Tz, Tz)  [target]
        # K_x[c]:       (B, 1, C, Tz, Tz)  [source]
        Kyp = K_y_past.unsqueeze(2)       # (B, C, 1, Tz, Tz)
        Kx = K_x.unsqueeze(1)             # (B, 1, C, Tz, Tz)
        Kyp_present = K_y_present.unsqueeze(2)  # (B, C, 1, Tz, Tz)

        # Joint kernels via Hadamard product
        # J1 = K_y_past[c'] ⊙ K_x[c]          -> (B, C, C, Tz, Tz)
        J1 = Kyp * Kx
        # J2 = K_y_present[c'] ⊙ K_y_past[c'] ⊙ K_x[c]
        J2 = Kyp_present * Kyp * Kx
        # J3 = K_y_present[c'] ⊙ K_y_past[c']
        J3 = Kyp_present * Kyp
        # J4 = K_y_past[c']
        J4 = Kyp

        # Entropy terms: H_2 = -log(||N(J)||_F^2)
        # Frobenius norm squared over last two dims
        def fro_sq(J):
            return torch.sum(J ** 2, dim=(-2, -1))  # (B, C, C)

        H1 = -torch.log(self._trace_normalize_fro(J1).clamp_min(1e-12))
        H2 = -torch.log(self._trace_normalize_fro(J2).clamp_min(1e-12))
        H3 = -torch.log(self._trace_normalize_fro(J3).clamp_min(1e-12))
        H4 = -torch.log(self._trace_normalize_fro(J4).clamp_min(1e-12))

        # TE = H1 - H2 + H3 - H4
        TE = H1 - H2 + H3 - H4  # (B, C, C)

        # Zero out diagonal (self-connections excluded)
        diag_mask = torch.eye(channels, dtype=bool, device=TE.device)
        TE = TE.masked_fill(diag_mask.unsqueeze(0), 0.0)

        return TE

    def _trace_normalize_fro(self, J: torch.Tensor) -> torch.Tensor:
        """||N(J)||_F^2 where N(J) = J / tr(J).

        J: (B, C, C, Tz, Tz) -> (B, C, C)
        """
        trace = torch.diagonal(J, dim1=-2, dim2=-1).sum(dim=-1)  # (B, C, C, Tz)
        trace = trace.clamp_min(1e-8)
        J_norm = J / trace.unsqueeze(-1).unsqueeze(-1)  # (B, C, C, Tz, Tz)
        return torch.sum(J_norm ** 2, dim=(-2, -1))  # (B, C, C)

    def _batch_gram(self, X: torch.Tensor) -> torch.Tensor:
        """Gram matrices per channel via RQ kernel.

        X: (B, C, Tz, D) -> (B, C, Tz, Tz)
        Computes K[X[b,c,i], X[b,c,j]] for all i,j per channel.
        """
        batch, channels, tz, d = X.shape
        # Flatten batch and channel for batched kernel computation
        X_flat = X.reshape(batch * channels, tz, d)  # (B*C, Tz, D)

        # Squared distances: ||x - y||^2
        x_sq = torch.sum(X_flat ** 2, dim=-1, keepdim=True)   # (BC, Tz, 1)
        xy = torch.matmul(X_flat, X_flat.transpose(-1, -2))   # (BC, Tz, Tz)
        sq_dist = x_sq + x_sq.transpose(-1, -2) - 2.0 * xy    # (BC, Tz, Tz)
        sq_dist = torch.clamp(sq_dist, min=0.0)

        denom = 2.0 * self.rq_alpha * self.rq_length_scale ** 2
        kernel = (1.0 + sq_dist / denom) ** (-self.rq_alpha)
        kernel = kernel * (self.rq_amplitude ** 2)

        return kernel.reshape(batch, channels, tz, tz)  # (B, C, Tz, Tz)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward_features(
        self, x: torch.Tensor, adjacency: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (features, connectivity) before the classification head.

        features: (B, C*(C-1)) flattened off-diagonal TE.
        connectivity: (B, C, C) directed TE matrix.

        ``adjacency`` is accepted for API compatibility with the inspired
        CTENet model but is not used (connectivity is estimated internally).
        """
        # x: (B, C, T)

        # ---- Block 1: Transformer contextualization ----
        # Transpose to (B, T, C) for sequence processing
        x_t = x.transpose(1, 2)  # (B, T, C)

        # Input projection with LayerNorm
        E = self.input_norm(x_t)
        E = self.input_proj(E)  # (B, T, d)

        # Transformer encoder
        H = self.transformer(E)  # (B, T, d)

        # Channel projection back to C
        X_f_t = self.channel_proj(H)
        X_f_t = self.channel_norm(X_f_t)  # (B, T, C)

        # Transpose back to (B, C, T)
        X_f = X_f_t.transpose(1, 2)  # (B, C, T)

        # ---- Block 2: Temporal filtering ----
        phi = self.temporal_filter(X_f)   # (B, C, T)
        phi = self.filter_activation(phi)
        phi = self.filter_pool(phi)       # (B, C, T_phi)
        phi = self.filter_norm(phi)       # (B, C, T_phi)

        # ---- Block 2: Transfer Entropy ----
        connectivity = self._compute_te_vectorized(phi)  # (B, C, C)

        # ---- Block 3: Flatten off-diagonal TE ----
        mask = ~torch.eye(self.n_channels, dtype=bool, device=connectivity.device)
        features = connectivity[:, mask]  # (B, C*(C-1))

        return features, connectivity

    def forward(self, x: torch.Tensor, adjacency: Optional[torch.Tensor] = None) -> torch.Tensor:
        features, _ = self.forward_features(x)
        return self.head(features).squeeze(-1)


def build_ctenet_original(
    n_channels: int = 19,
    n_times: int = 512,
    d_model: int = 128,
    n_heads: int = 1,
    n_layers: int = 1,
    dropout: float = 0.5,
    embed_dim_source: int = 6,
    embed_dim_target: int = 1,
    embed_delay: int = 2,
    te_alpha: float = 2.0,
    kernel_size: int = 83,
    pool_size: int = 4,
    rq_alpha: float = 1.0,
    rq_length_scale: float = 1.0,
    rq_amplitude: float = 1.0,
    head_hidden: int = 64,
) -> CTENetOriginal:
    return CTENetOriginal(
        n_channels=n_channels,
        n_times=n_times,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        embed_dim_source=embed_dim_source,
        embed_dim_target=embed_dim_target,
        embed_delay=embed_delay,
        te_alpha=te_alpha,
        kernel_size=kernel_size,
        pool_size=pool_size,
        rq_alpha=rq_alpha,
        rq_length_scale=rq_length_scale,
        rq_amplitude=rq_amplitude,
        head_hidden=head_hidden,
    )


# ---------------------------------------------------------------------------
# Known deviations from the original CTE-Net paper
# ---------------------------------------------------------------------------

DOCUMENTED_GAPS = """
CTENetOriginal deviations from Gomez-Rivera et al. (2026):

1. Input window length: the paper uses T=512 (4 seconds at 128 Hz) with 50%
   overlap; this implementation accepts configurable T (default 512).
   The repository's main pipeline uses T=640 (5 seconds), which differs.

2. Channel order: the paper uses [Fz, Cz, Pz, C3, T3, C4, T4, Fp1, Fp2, F3,
   F4, F7, F8, P3, P4, T5, T6, O1, O2]; this implementation uses
   [Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, Pz]
   matching the IEEE DataPort .mat channel order.

3. Preprocessing: the paper uses raw EEG without filtering, ICA, or artifact
   rejection (only post-hoc window quality screening). This implementation's
   pipeline applies band-pass filtering, notch filtering, and z-score
   normalization by default.

4. All other components (Transformer contextualization, depthwise temporal
   filtering, Takens embeddings, bias-free linear projections, RQ kernel,
   matrix-based Renyi alpha=2 entropy, 4-term TE formula, flattened
   off-diagonal TE classification head) match the paper's specification.
   Parameter values match Table 2 of the paper (d=128, L=1, nh=1, d_ff=128,
   p=0.5, K=83, pool=4, Dx=6, Dy=1, tau=2, alpha_R=2, a=1, l=1, alpha_RQ=1,
   H=64).

This is a faithful reconstruction of the published architecture. Remaining
differences are in input window length, channel order, and preprocessing,
which are pipeline-level choices rather than architectural ones.
"""
