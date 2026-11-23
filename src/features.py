"""Epoch-level EEG features: band power and directed connectivity."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, welch

# numpy 2.0 renamed trapz to trapezoid; support both.
_trapezoid = getattr(np, "trapezoid", np.trapz)


def _band_indices(freqs: np.ndarray, low: float, high: float) -> np.ndarray:
    idx = (freqs >= low) & (freqs <= high)
    if not np.any(idx):
        raise ValueError(f"No frequency bins in [{low}, {high}] Hz")
    return idx


def bandpower_features(
    epoch: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Log band power per channel plus a log theta/beta ratio.

    Parameters
    ----------
    epoch : (n_channels, n_times)
    fs : sampling frequency
    bands : ordered mapping name -> (low_hz, high_hz)

    Returns
    -------
    (n_channels, n_bands + 1) float32
    """
    nperseg = min(256, epoch.shape[-1])
    freqs, psd = welch(epoch, fs=fs, nperseg=nperseg, axis=-1)

    feats = []
    for low, high in bands.values():
        idx = _band_indices(freqs, low, high)
        power = _trapezoid(psd[:, idx], freqs[idx], axis=-1)
        feats.append(np.log(power + 1e-12))

    band_power = np.stack(feats, axis=-1)  # (C, n_bands)

    names = list(bands)
    if "theta" in names and "beta" in names:
        theta = band_power[:, names.index("theta")]
        beta = band_power[:, names.index("beta")]
        ratio = (theta - beta)[:, None]
        band_power = np.concatenate([band_power, ratio], axis=-1)

    return band_power.astype(np.float32)


def corr_adjacency(epoch: np.ndarray) -> np.ndarray:
    """Absolute Pearson correlation, diagonal set to zero."""
    adj = np.abs(np.corrcoef(epoch))
    adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(adj, 0.0)
    return adj.astype(np.float32)


def plv_adjacency(
    epoch: np.ndarray,
    fs: float,
    band: tuple[float, float] = (1.0, 45.0),
) -> np.ndarray:
    """Phase-locking value on a band-passed signal."""
    high = min(band[1], fs / 2.0 - 1.0)
    if high <= band[0]:
        return np.zeros((epoch.shape[0], epoch.shape[0]), dtype=np.float32)

    sos = butter(4, (band[0], high), btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, epoch, axis=-1)
    phase = np.angle(hilbert(filtered, axis=-1))

    diff = phase[:, None, :] - phase[None, :, :]
    plv = np.abs(np.mean(np.exp(1j * diff), axis=-1))
    plv = np.nan_to_num(plv, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(plv, 0.0)
    return plv.astype(np.float32)


def var_coefficient_adjacency(
    epoch: np.ndarray,
    order: int = 5,
    ridge: float = 1e-3,
) -> np.ndarray:
    """Directed connectivity proxy from ridge-regularised VAR coefficients.

    ``adj[target, source]`` is the L2 norm across lags of the VAR coefficient
    linking ``source`` to ``target``. Normalised to [0, 1] per epoch.

    Notes
    -----
    This is *not* a formal Granger-causality test. It is a directed spectral
    summary that is cheap enough to compute for every epoch. If the VAR fit is
    ill-conditioned, the function returns zeros.
    """
    x = np.asarray(epoch, dtype=np.float64)
    n_channels, n_times = x.shape

    if n_times <= order + n_channels * order + 2:
        return np.zeros((n_channels, n_channels), dtype=np.float32)

    sd = x.std(axis=1, keepdims=True)
    x = (x - x.mean(axis=1, keepdims=True)) / (sd + 1e-8)

    target = x[:, order:].T  # (T - order, C)
    lagged = [x[:, order - lag : n_times - lag].T for lag in range(1, order + 1)]
    design = np.concatenate(lagged, axis=1)
    design = np.column_stack([design, np.ones(len(design))])

    gram = design.T @ design
    gram.flat[:: gram.shape[0] + 1] += ridge

    try:
        coef = np.linalg.solve(gram, design.T @ target)
    except np.linalg.LinAlgError:
        return np.zeros((n_channels, n_channels), dtype=np.float32)

    # (lag, target, source)
    coef = coef[:-1].reshape(order, n_channels, n_channels)
    adj = np.linalg.norm(coef, axis=0)

    max_value = float(adj.max())
    if not np.isfinite(max_value) or max_value <= 1e-12:
        return np.zeros((n_channels, n_channels), dtype=np.float32)

    adj = adj / max_value
    np.fill_diagonal(adj, 0.0)
    return adj.astype(np.float32)


# Backward-compatible alias.
granger_adjacency = var_coefficient_adjacency


def compute_connectivity(
    epoch: np.ndarray,
    fs: float,
    method: str = "granger",
    var_order: int = 5,
) -> np.ndarray:
    """Dispatch to the requested connectivity estimator."""
    method = method.lower()
    if method in {"granger", "var_coefficient"}:
        return var_coefficient_adjacency(epoch, order=var_order)
    if method in {"plv", "phase"}:
        return plv_adjacency(epoch, fs)
    if method in {"corr", "correlation", "pearson"}:
        return corr_adjacency(epoch)
    raise ValueError(f"Unknown connectivity method: {method}")


def extract_epoch_features(
    epoch: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]],
    connectivity: str = "granger",
    var_order: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (node features, adjacency) for one epoch."""
    node = bandpower_features(epoch, fs, bands)
    adj = compute_connectivity(epoch, fs, connectivity, var_order)
    return node, adj
