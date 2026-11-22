"""Paper 1-style handcrafted EEG features.

The feature families follow the requirements distilled from Garcia-Ponsoda
et al. (2024): descriptive statistics, absolute/relative/log band power,
band-power ratios, Katz fractal dimension, Hjorth parameters, and spectral
shape descriptors.  Features are computed per channel.

The published paper used the ``mne-features`` library with 53 features.  This
module implements the interpretable subset required by this project, not a
byte-exact reproduction of that library.  That difference is documented in
``RESEARCH_IMPLEMENTATION_AUDIT.md``.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew

FEATURE_GROUPS = ("stats", "spectral", "bands", "ratios", "fractal", "hjorth")

# Order is fixed and shared by extraction and naming.
_STATS = (
    "mean",
    "std",
    "variance",
    "skewness",
    "kurtosis",
    "rms",
    "peak_to_peak",
)
_SPECTRAL = ("spectral_entropy", "spectral_edge_95")
_RATIOS = ("theta_beta", "theta_alpha", "delta_alpha", "delta_theta", "alpha_beta")
_FRACTAL = ("katz_fd",)
_HJORTH = ("hjorth_mobility", "hjorth_complexity")


def _as_epoch(epoch: np.ndarray) -> np.ndarray:
    array = np.asarray(epoch, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected (channels, times), got {array.shape}")
    if array.shape[1] < 8:
        raise ValueError("At least 8 samples are required")
    return array


def _resolve_groups(groups: Sequence[str] | str | None) -> tuple[str, ...]:
    if groups is None or groups == "all":
        return FEATURE_GROUPS
    if isinstance(groups, str):
        groups = [groups]
    requested = tuple(str(group).lower() for group in groups)
    unknown = set(requested) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown feature groups: {sorted(unknown)}")
    return tuple(group for group in FEATURE_GROUPS if group in requested)


def _psd(epoch: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(256, epoch.shape[-1])
    return welch(epoch, fs=fs, nperseg=nperseg, axis=-1)


def _band_matrix(
    freqs: np.ndarray,
    psd: np.ndarray,
    bands: dict[str, tuple[float, float]],
    low_total: float,
    high_total: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (absolute power, relative power), both (channels, bands)."""
    total_mask = (freqs >= low_total) & (freqs <= high_total)
    total = np.trapezoid(psd[:, total_mask], freqs[total_mask], axis=-1)
    absolute = []
    for low, high in bands.values():
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            absolute.append(np.zeros(psd.shape[0], dtype=np.float64))
        else:
            absolute.append(
                np.trapezoid(psd[:, mask], freqs[mask], axis=-1)
            )
    absolute_array = np.stack(absolute, axis=-1)
    relative_array = absolute_array / np.maximum(total[:, None], 1e-12)
    return absolute_array, relative_array


def _katz_fd(signal: np.ndarray, fs: float) -> float:
    """Katz fractal dimension of a one-dimensional waveform."""
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)
    if n < 3:
        return 0.0
    # Distance in the (time, amplitude) plane, with time in sample units.
    steps = np.sqrt(np.diff(x) ** 2 + 1.0)
    length = float(steps.sum())
    mean_step = length / max(n - 1, 1)
    start = np.sqrt((x - x[0]) ** 2 + np.arange(n, dtype=np.float64) ** 2)
    diameter = float(start.max())
    if length <= 0 or mean_step <= 0 or diameter <= mean_step:
        return 0.0
    return float(
        np.log10(length / mean_step)
        / (np.log10(diameter / mean_step) + np.log10(length / mean_step))
    )


def _hjorth(signal: np.ndarray) -> tuple[float, float]:
    x = np.asarray(signal, dtype=np.float64)
    dx = np.diff(x)
    var_x = float(np.var(x))
    var_dx = float(np.var(dx))
    if var_x <= 1e-20:
        return 0.0, 0.0
    mobility = float(np.sqrt(var_dx / var_x))
    if var_dx <= 1e-20:
        return mobility, 0.0
    ddx = np.diff(dx)
    var_ddx = float(np.var(ddx))
    complexity = float(np.sqrt(var_ddx / var_dx) / max(mobility, 1e-12))
    return mobility, complexity


def _spectral_shape(
    freqs: np.ndarray,
    psd: np.ndarray,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (freqs >= low) & (freqs <= high)
    selected = np.maximum(psd[:, mask], 0.0)
    total = selected.sum(axis=-1, keepdims=True)
    probability = selected / np.maximum(total, 1e-12)
    entropy = -np.sum(probability * np.log(probability + 1e-12), axis=-1)
    entropy = entropy / np.log(max(selected.shape[-1], 2))
    cumulative = np.cumsum(selected, axis=-1)
    threshold = 0.95 * cumulative[:, -1:]
    edge_index = np.argmax(cumulative >= threshold, axis=-1)
    edge = freqs[mask][edge_index]
    return entropy, edge


def _channel_features(
    signal: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]],
    groups: tuple[str, ...],
    psd: np.ndarray | None = None,
    freqs: np.ndarray | None = None,
) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    values: list[float] = []

    if "stats" in groups:
        std = float(np.std(x))
        values.extend(
            [
                float(np.mean(x)),
                std,
                float(np.var(x)),
                float(skew(x, bias=False)) if std > 1e-12 else 0.0,
                float(kurtosis(x, bias=False)) if std > 1e-12 else 0.0,
                float(np.sqrt(np.mean(x**2))),
                float(np.ptp(x)),
            ]
        )

    if "spectral" in groups:
        if psd is None or freqs is None:
            freqs, psd = welch(x, fs=fs, nperseg=min(256, len(x)))
        entropy, edge = _spectral_shape(
            freqs, psd[None, :], max(0.5, bands["delta"][0]), min(45.0, fs / 2 - 1)
        )
        values.extend([float(entropy[0]), float(edge[0])])

    if "bands" in groups:
        if psd is None or freqs is None:
            freqs, psd = welch(x, fs=fs, nperseg=min(256, len(x)))
        low_total = min(band[0] for band in bands.values())
        high_total = min(max(band[1] for band in bands.values()), fs / 2 - 1)
        absolute, relative = _band_matrix(
            freqs, psd[None, :], bands, low_total, high_total
        )
        for index in range(len(bands)):
            values.append(float(np.log10(absolute[0, index] + 1e-12)))
        for index in range(len(bands)):
            values.append(float(relative[0, index]))
        for index in range(len(bands)):
            values.append(float(np.log10(relative[0, index] + 1e-12)))

    if "ratios" in groups:
        if psd is None or freqs is None:
            freqs, psd = welch(x, fs=fs, nperseg=min(256, len(x)))
        low_total = min(band[0] for band in bands.values())
        high_total = min(max(band[1] for band in bands.values()), fs / 2 - 1)
        absolute, _ = _band_matrix(freqs, psd[None, :], bands, low_total, high_total)
        index = {name: i for i, name in enumerate(bands)}
        pairs = (
            ("theta", "beta"),
            ("theta", "alpha"),
            ("delta", "alpha"),
            ("delta", "theta"),
            ("alpha", "beta"),
        )
        for numerator, denominator in pairs:
            if numerator in index and denominator in index:
                ratio = (absolute[0, index[numerator]] + 1e-12) / (
                    absolute[0, index[denominator]] + 1e-12
                )
                values.append(float(np.log10(ratio)))
            else:
                values.append(0.0)

    if "fractal" in groups:
        values.append(_katz_fd(x, fs))

    if "hjorth" in groups:
        values.extend(_hjorth(x))

    return np.asarray(values, dtype=np.float64)


def _names_for_channel(
    channel: str,
    bands: dict[str, tuple[float, float]],
    groups: tuple[str, ...],
) -> list[str]:
    names: list[str] = []
    if "stats" in groups:
        names.extend(f"{channel} | {name}" for name in _STATS)
    if "spectral" in groups:
        names.extend(f"{channel} | {name}" for name in _SPECTRAL)
    if "bands" in groups:
        for name in bands:
            names.append(f"{channel} | absolute_log_power_{name}")
        for name in bands:
            names.append(f"{channel} | relative_power_{name}")
        for name in bands:
            names.append(f"{channel} | relative_log_power_{name}")
    if "ratios" in groups:
        names.extend(f"{channel} | ratio_{name}" for name in _RATIOS)
    if "fractal" in groups:
        names.extend(f"{channel} | {name}" for name in _FRACTAL)
    if "hjorth" in groups:
        names.extend(f"{channel} | {name}" for name in _HJORTH)
    return names


def feature_names(
    channels: Sequence[str],
    bands: dict[str, tuple[float, float]],
    groups: Sequence[str] | str | None = "all",
) -> list[str]:
    """Deterministic feature names matching :func:`extract_features`."""
    resolved = _resolve_groups(groups)
    names: list[str] = []
    for channel in channels:
        names.extend(_names_for_channel(str(channel), bands, resolved))
    return names


def extract_features(
    epoch: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]],
    groups: Sequence[str] | str | None = "all",
    channels: Sequence[int] | None = None,
) -> np.ndarray:
    """Extract a flattened per-channel feature vector for one segment.

    Returns shape ``(n_channels_used * n_features_per_channel,)``.
    """
    array = _as_epoch(epoch)
    resolved = _resolve_groups(groups)
    indices = range(array.shape[0]) if channels is None else [int(c) for c in channels]

    freqs, psd = _psd(array, fs)
    rows = []
    for channel in indices:
        rows.append(
            _channel_features(
                array[channel],
                fs,
                bands,
                resolved,
                psd=psd[channel],
                freqs=freqs,
            )
        )
    return np.concatenate(rows).astype(np.float32)


def extract_features_by_channel(
    epoch: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]],
    groups: Sequence[str] | str | None = "all",
) -> np.ndarray:
    """Return features as ``(channels, features_per_channel)``."""
    array = _as_epoch(epoch)
    resolved = _resolve_groups(groups)
    freqs, psd = _psd(array, fs)
    rows = [
        _channel_features(
            array[channel], fs, bands, resolved, psd=psd[channel], freqs=freqs
        )
        for channel in range(array.shape[0])
    ]
    return np.stack(rows, axis=0).astype(np.float32)


def segment_recording(
    recording: np.ndarray,
    segment: str,
) -> np.ndarray:
    """Return one temporal segment of a (times, channels) recording.

    Supported segments correspond to the Paper 1 analysis: ``whole``,
    ``first_half``, ``second_half``, ``last_fifth``, ``last_twentieth``, and
    ``segment_<i>_of_20``.
    """
    array = np.asarray(recording)
    n_times = array.shape[0]
    if n_times < 8:
        raise ValueError("Recording is too short")

    if segment in {"whole", "1/1", "full"}:
        return array
    if segment in {"first_half", "1/2"}:
        return array[: n_times // 2]
    if segment in {"second_half", "2/2"}:
        return array[n_times // 2 :]
    if segment in {"last_fifth", "5/5"}:
        return array[int(n_times * 0.8) :]
    if segment in {"last_twentieth", "20/20"}:
        return array[int(n_times * 0.95) :]
    if segment.startswith("segment_") and "_of_20" in segment:
        index = int(segment.split("_")[1]) - 1
        start = int(n_times * index / 20)
        stop = int(n_times * (index + 1) / 20)
        return array[start:stop]
    raise ValueError(f"Unknown segment: {segment}")


def extract_recording_features(
    recording: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]],
    channel_names: Sequence[str],
    groups: Sequence[str] | str | None = "all",
    channels: Sequence[int] | None = None,
    segment: str = "whole",
) -> tuple[np.ndarray, list[str]]:
    """Extract one subject-level feature vector from a full recording."""
    selected = segment_recording(recording, segment)
    features = extract_features(
        selected.T, fs=fs, bands=bands, groups=groups, channels=channels
    )
    used_names = (
        list(channel_names)
        if channels is None
        else [channel_names[int(c)] for c in channels]
    )
    return features, feature_names(used_names, bands, groups)
