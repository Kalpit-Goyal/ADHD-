"""Unit tests for preprocessing, features, models, and subject-wise splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import DataConfig, FeatureConfig
from src.cv import assert_no_subject_overlap, make_repeated_folds
from src.data import (
    FeatureBundle,
    build_feature_bundle,
    epoch_recording,
    load_recording,
    parse_label,
    read_manifest,
    remove_blink_components,
    subject_folds,
)
from src.evaluate import binary_metrics
from src.features import bandpower_features, granger_adjacency, plv_adjacency
from src.hybrid_model import run_feature_fusion, run_prediction_ensemble
from src.models import CTENet, EEGNet


def _make_bundle(n_subjects: int = 12, epochs_per_subject: int = 3) -> FeatureBundle:
    n_channels, n_times, n_features = 19, 64, 6
    n_epochs = n_subjects * epochs_per_subject
    subject_ids = np.repeat(
        [f"s{index:02d}" for index in range(n_subjects)], epochs_per_subject
    )
    subject_labels = {f"s{index:02d}": index % 2 for index in range(n_subjects)}
    labels = np.asarray([subject_labels[s] for s in subject_ids], dtype=np.float32)

    return FeatureBundle(
        raw=np.random.randn(n_epochs, 1, n_channels, n_times).astype(np.float32),
        node_features=np.random.randn(n_epochs, n_channels, n_features).astype(np.float32),
        adjacency=np.random.rand(n_epochs, n_channels, n_channels).astype(np.float32),
        labels=labels,
        subject_ids=subject_ids,
        epoch_index=np.tile(np.arange(epochs_per_subject), n_subjects),
        subject_labels=subject_labels,
        fs=128,
        epoch_seconds=0.5,
    )


def test_parse_label() -> None:
    assert parse_label("ADHD") == 1
    assert parse_label("control") == 0
    assert parse_label(1) == 1
    assert parse_label("1") == 1


def test_epoch_recording_shapes() -> None:
    recording = np.random.randn(1000, 19).astype(np.float32)
    epochs = epoch_recording(recording, fs=128, epoch_seconds=1.0, overlap=0.0)
    assert epochs.shape == (7, 19, 128)

    overlapping = epoch_recording(recording, fs=128, epoch_seconds=1.0, overlap=0.5)
    assert overlapping.shape[0] > epochs.shape[0]
    assert overlapping.shape[1:] == (19, 128)


def test_feature_shapes_are_finite() -> None:
    rng = np.random.default_rng(0)
    epoch = rng.normal(size=(19, 640)).astype(np.float32)
    bands = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }

    node = bandpower_features(epoch, fs=128, bands=bands)
    assert node.shape == (19, 6)
    assert np.isfinite(node).all()

    granger = granger_adjacency(epoch, order=5)
    assert granger.shape == (19, 19)
    assert np.isfinite(granger).all()
    assert np.allclose(np.diag(granger), 0.0)

    plv = plv_adjacency(epoch, fs=128)
    assert plv.shape == (19, 19)
    assert np.isfinite(plv).all()


def test_model_forward_shapes() -> None:
    batch, channels, n_times, n_features = 2, 19, 640, 6

    graph_model = CTENet(
        n_features=n_features,
        n_channels=channels,
        d_model=16,
        n_heads=4,
        n_layers=1,
        dropout=0.1,
    )
    node = torch.randn(batch, channels, n_features)
    adjacency = torch.rand(batch, channels, channels)
    assert graph_model(node, adjacency).shape == (batch,)

    cnn = EEGNet(n_channels=channels, n_times=n_times, f1=4, f2=8, dropout=0.1)
    raw = torch.randn(batch, 1, channels, n_times)
    assert cnn(raw).shape == (batch,)


def test_subject_folds_are_disjoint() -> None:
    bundle = _make_bundle()
    folds = subject_folds(bundle, n_splits=3, val_size=0.25, seed=42)

    for train_subjects, val_subjects, test_subjects in folds:
        train_set = set(train_subjects)
        val_set = set(val_subjects)
        test_set = set(test_subjects)

        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)
        assert train_set | val_set | test_set == set(bundle.subject_labels)


def test_csv_loading_drops_time_column(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        rng.normal(size=(200, 19)),
        columns=[f"ch{index}" for index in range(19)],
    )
    frame.insert(0, "time", np.arange(len(frame)) / 128.0)
    path = tmp_path / "recording.csv"
    frame.to_csv(path, index=False)

    recording, source_fs = load_recording(path, n_channels=19, fs=128)
    assert recording.shape == (200, 19)
    assert source_fs == 128
    assert np.isfinite(recording).all()


def test_mat_loading_from_struct(tmp_path: Path) -> None:
    from scipy.io import savemat

    rng = np.random.default_rng(2)
    signal = rng.normal(size=(19, 1280)).astype(np.float32)
    path = tmp_path / "recording.mat"
    savemat(path, {"EEG": {"data": signal, "fs": 128}, "note": "test"})

    recording, _ = load_recording(path, n_channels=19)
    assert recording.shape == (1280, 19)
    assert np.isfinite(recording).all()


def test_ica_removes_frontal_oscillation() -> None:
    fs = 128
    n_samples = fs * 20
    time = np.arange(n_samples) / fs
    rng = np.random.default_rng(3)

    signal = rng.normal(scale=10.0, size=(n_samples, 19))
    blink = 500.0 * np.sin(2.0 * np.pi * 1.5 * time)
    signal[:, [0, 1]] += blink[:, None]

    cleaned = remove_blink_components(
        signal,
        fs=fs,
        blink_channels=(0, 1),
        max_components=2,
        min_corr=0.5,
    )

    assert cleaned.shape == signal.shape
    assert np.isfinite(cleaned).all()
    assert cleaned[:, 0].std() < 0.5 * signal[:, 0].std()


def test_binary_metrics_accepts_predictions() -> None:
    metrics = binary_metrics(
        [0, 1, 0, 1],
        [0.1, 0.9, 0.8, 0.2],
        y_pred=[0, 1, 0, 0],
        n_boot=50,
    )
    assert metrics["accuracy"] == 0.75
    assert metrics["auroc"] == 0.75


def test_manifest_and_feature_bundle(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    rows = []
    for index in range(4):
        subject_id = f"sub-{index:02d}"
        np.save(
            tmp_path / f"{subject_id}.npy",
            rng.normal(size=(1280, 19)).astype(np.float32),
        )
        rows.append(
            {
                "subject_id": subject_id,
                "label": index % 2,
                "site": "test",
                "file": f"{subject_id}.npy",
            }
        )

    pd.DataFrame(rows).to_csv(tmp_path / "manifest.csv", index=False)

    records = read_manifest(tmp_path)
    assert len(records) == 4

    data_config = DataConfig(
        root=str(tmp_path),
        n_channels=19,
        fs=128,
        epoch_seconds=5.0,
    )
    bundle = build_feature_bundle(
        records,
        data_config,
        FeatureConfig(),
        verbose=False,
    )

    assert len(bundle.labels) == 8
    assert bundle.raw.shape[1:] == (1, 19, 640)
    assert bundle.node_features.shape[1:] == (19, 6)
    assert len(bundle.subject_labels) == 4


def test_feature_fusion_and_ensemble() -> None:
    rng = np.random.default_rng(0)
    n_subjects = 20
    epochs_per_subject = 2
    n_epochs = n_subjects * epochs_per_subject

    subject_ids = np.repeat(
        [f"s{index:02d}" for index in range(n_subjects)], epochs_per_subject
    )
    subject_labels = {f"s{index:02d}": index % 2 for index in range(n_subjects)}
    labels = np.asarray([subject_labels[s] for s in subject_ids], dtype=np.float32)

    bundle = FeatureBundle(
        raw=np.random.randn(n_epochs, 1, 19, 64).astype(np.float32),
        node_features=np.random.randn(n_epochs, 19, 6).astype(np.float32),
        adjacency=np.random.rand(n_epochs, 19, 19).astype(np.float32),
        labels=labels,
        subject_ids=subject_ids,
        epoch_index=np.tile(np.arange(epochs_per_subject), n_subjects),
        subject_labels=subject_labels,
        fs=128,
        epoch_seconds=0.5,
    )

    folds = make_repeated_folds(subject_labels, n_splits=2, n_repeats=1, seed=42)

    # Soft voting with dummy branch probs
    branch_a_probs: dict[int, dict[str, float]] = {}
    branch_b_probs: dict[int, dict[str, float]] = {}
    for fold in folds:
        for sid in fold.test:
            branch_a_probs[fold.fold] = {sid: 0.7}
            branch_b_probs[fold.fold] = {sid: 0.3}

    result = run_prediction_ensemble(
        bundle=bundle,
        folds=folds,
        seed=42,
        branch_a_probs=branch_a_probs,
        branch_b_probs=branch_b_probs,
        method="soft_voting",
        real_data=False,
    )
    assert result.experiment == "soft_voting"
    assert 0.0 <= result.roc_auc <= 1.0 or np.isnan(result.roc_auc)


def test_no_subject_overlap_raises() -> None:
    bundle = _make_bundle()
    folds = subject_folds(bundle, n_splits=3, val_size=0.25, seed=42)

    # All folds should pass (no overlap)
    for train_subjects, val_subjects, test_subjects in folds:
        assert_no_subject_overlap(train_subjects, val_subjects, test_subjects)

    # Overlapping train/test should raise
    try:
        assert_no_subject_overlap(["s00"], ["s01"], ["s00"])
        assert False, "Expected ValueError for subject overlap"
    except ValueError:
        pass


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

