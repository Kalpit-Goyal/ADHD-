"""Configuration objects for the ADHD EEG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass
class DataConfig:
    root: str = "data/real"
    manifest: str = "manifest.csv"
    n_channels: int = 19
    fs: int = 128
    epoch_seconds: float = 5.0
    overlap: float = 0.0
    bandpass: tuple[float, float] = (0.5, 45.0)
    notch: float | None = 50.0
    max_abs_uv: float | None = None
    max_abs_z: float | None = None
    artifact_removal: str = "none"  # none | ica
    blink_channels: list[int] = field(default_factory=lambda: [0, 1])
    ica_max_components: int = 2
    ica_blink_corr: float = 0.5
    connectivity: str = "granger"
    var_order: int = 5


@dataclass
class FeatureConfig:
    bands: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(DEFAULT_BANDS)
    )


@dataclass
class ModelConfig:
    name: str = "ctenet_original"  # cte_net | ctenet_original | eegnet
    d_model: int = 128
    n_heads: int = 1
    n_layers: int = 1
    dropout: float = 0.5
    eegnet_f1: int = 8
    eegnet_d: int = 2
    eegnet_f2: int = 16
    eegnet_kernel: int = 64


@dataclass
class TrainingConfig:
    n_splits: int = 5
    val_size: float = 0.2
    batch_size: int = 32
    epochs: int = 60
    patience: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"  # auto | cpu | cuda


@dataclass
class OutputConfig:
    dir: str = "results"


@dataclass
class Config:
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    training: TrainingConfig
    output: OutputConfig
    raw: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> Config:
    """Load a YAML config and fill in defaults for missing keys."""
    text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}

    data_raw = dict(raw.get("data", {}))
    if "bandpass" in data_raw:
        data_raw["bandpass"] = tuple(float(v) for v in data_raw["bandpass"])

    features_raw = dict(raw.get("features", {}))
    bands_raw = features_raw.get("bands", DEFAULT_BANDS)
    features_raw["bands"] = {
        str(name): tuple(float(v) for v in limits)
        for name, limits in bands_raw.items()
    }

    return Config(
        data=DataConfig(**data_raw),
        features=FeatureConfig(**features_raw),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        output=OutputConfig(**raw.get("output", {})),
        raw=raw,
    )
