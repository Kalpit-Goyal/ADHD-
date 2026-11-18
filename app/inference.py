"""Inference for the deployable ADHD screening model.

The web app accepts one EEG recording, applies the same preprocessing used in
training, extracts per-epoch spectral + directed-connectivity features, and
aggregates epoch probabilities to a subject-level screening score.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from src.config import load_config
from src.data import (
    epoch_recording,
    extract_epoch_features,
    load_recording,
    normalize_epochs,
    preprocess_recording,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "adhd_svm_select150.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
CONFIG_PATH = ROOT / "configs" / "eeg_cte_real.yaml"

_model = None
_metadata: dict | None = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model artifact not found. Run: python scripts/train_deploy_model.py"
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def get_metadata() -> dict:
    global _metadata
    if _metadata is None:
        _metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return _metadata


def _build_epoch_features(node_features: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    """Must match scripts/train_deploy_model.py exactly."""
    n = node_features.shape[0]
    node = node_features.reshape(n, -1)

    probabilities = np.exp(node_features[:, :, :5])
    relative = probabilities / probabilities.sum(axis=2, keepdims=True)
    relative = relative.reshape(n, -1)

    rows, columns = np.triu_indices(adjacency.shape[1], k=1)
    edges = adjacency[:, rows, columns]

    return np.concatenate([node, edges, relative], axis=1)


def _band_profile(node_features: np.ndarray, bands: list[str]) -> list[dict]:
    probabilities = np.exp(node_features[:, :, :5])
    relative = probabilities / probabilities.sum(axis=2, keepdims=True)
    means = relative.mean(axis=(0, 1))
    means = means / max(float(means.sum()), 1e-12) * 100.0
    return [
        {"band": band, "pct": round(float(value), 1)}
        for band, value in zip(bands, means)
    ]


def _top_edges(
    adjacency: np.ndarray,
    channels: list[str],
    top_k: int = 5,
) -> list[dict]:
    mean_adj = adjacency.mean(axis=0)
    rows, columns = np.triu_indices(mean_adj.shape[0], k=1)
    values = mean_adj[rows, columns]
    order = np.argsort(values)[::-1][:top_k]
    return [
        {
            "pair": f"{channels[rows[i]]} \u2194 {channels[columns[i]]}",
            "strength": round(float(values[i]), 3),
        }
        for i in order
    ]


def analyze_recording(
    path: str | Path,
    source_fs: int = 128,
    demo: str | None = None,
) -> dict:
    """Run the full screening pipeline on one recording."""
    config = load_config(CONFIG_PATH)
    metadata = get_metadata()
    model = get_model()

    array, source_fs_used = load_recording(
        path, n_channels=config.data.n_channels, fs=source_fs
    )
    if source_fs_used is None:
        source_fs_used = source_fs

    array = preprocess_recording(
        array,
        fs=config.data.fs,
        bandpass=config.data.bandpass,
        notch=config.data.notch,
        source_fs=source_fs_used,
    )

    duration = float(array.shape[0]) / float(config.data.fs)
    epochs = epoch_recording(
        array,
        fs=config.data.fs,
        epoch_seconds=config.data.epoch_seconds,
        overlap=config.data.overlap,
        max_abs_uv=config.data.max_abs_uv,
    )
    if epochs.shape[0] == 0:
        raise ValueError(
            "No 5-second epochs could be built. The recording must be at least "
            f"{config.data.epoch_seconds:.0f} seconds long and contain "
            f"{config.data.n_channels} channels."
        )

    total_epochs = int(epochs.shape[0])
    epochs = normalize_epochs(epochs)

    if config.data.max_abs_z is not None:
        peaks = np.abs(epochs).max(axis=(1, 2))
        keep = peaks <= float(config.data.max_abs_z)
        if not np.any(keep):
            raise ValueError(
                "Every epoch was rejected as an artifact. Check the recording "
                "scale and channel order."
            )
        epochs = epochs[keep]

    node_list = []
    adjacency_list = []
    for epoch in epochs:
        node, adjacency = extract_epoch_features(
            epoch,
            fs=config.data.fs,
            bands=config.features.bands,
            connectivity=config.data.connectivity,
            var_order=config.data.var_order,
        )
        node_list.append(node)
        adjacency_list.append(adjacency)

    node_features = np.stack(node_list, axis=0)
    adjacency = np.stack(adjacency_list, axis=0)
    features = _build_epoch_features(node_features, adjacency)

    epoch_probabilities = model.predict_proba(features)[:, 1]
    probability = float(np.mean(epoch_probabilities))
    threshold = float(metadata.get("threshold", 0.5))
    positive = probability >= threshold

    bands = list(config.features.bands.keys())
    channel_names = metadata.get("channels") or [
        f"Ch{i + 1}" for i in range(node_features.shape[1])
    ]
    band_profile = _band_profile(node_features, bands)
    theta_beta = float(node_features[:, :, 5].mean())

    metrics = metadata.get("metrics", {})
    return {
        "label": "ADHD-like pattern" if positive else "No ADHD-like pattern",
        "positive": bool(positive),
        "probability": round(probability, 4),
        "control_probability": round(1.0 - probability, 4),
        "threshold": threshold,
        "screening_score": int(round(probability * 100)),
        "n_epochs": int(features.shape[0]),
        "n_epochs_total": total_epochs,
        "duration_seconds": round(duration, 1),
        "channels": int(node_features.shape[1]),
        "source_fs": int(source_fs_used),
        "band_profile": band_profile,
        "theta_beta_ratio": round(theta_beta, 3),
        "top_edges": _top_edges(adjacency, channel_names),
        "epoch_positive_fraction": round(
            float(np.mean(epoch_probabilities >= threshold)), 3
        ),
        "selected_features": metadata.get("selected_features", [])[:8],
        "metrics": metrics,
        "demo": demo,
        "model": metadata.get("model"),
        "notes": metadata.get("notes"),
    }
