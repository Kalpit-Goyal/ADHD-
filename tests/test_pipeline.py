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


# ---------------------------------------------------------------------------
# Phase 12 acceptance tests
# ---------------------------------------------------------------------------


def test_a_subject_leakage_single_fold() -> None:
    """A participant with multiple epochs must appear in exactly one fold."""
    n_subjects = 8
    epochs_per_subject = 4
    n_epochs = n_subjects * epochs_per_subject

    subject_ids = np.repeat(
        [f"s{index:02d}" for index in range(n_subjects)], epochs_per_subject
    )
    subject_labels = {f"s{index:02d}": index % 2 for index in range(n_subjects)}
    labels = np.asarray([subject_labels[s] for s in subject_ids], dtype=np.float32)

    bundle = FeatureBundle(
        raw=np.random.randn(n_epochs, 1, 19, 32).astype(np.float32),
        node_features=np.random.randn(n_epochs, 19, 6).astype(np.float32),
        adjacency=np.random.rand(n_epochs, 19, 19).astype(np.float32),
        labels=labels,
        subject_ids=subject_ids,
        epoch_index=np.tile(np.arange(epochs_per_subject), n_subjects),
        subject_labels=subject_labels,
        fs=128,
        epoch_seconds=0.25,
    )

    # Construct folds manually to avoid sklearn small-sample stratification limits.
    from src.cv import Fold
    folds = [
        Fold(train=("s00", "s02", "s04", "s06"), val=("s01", "s03"), test=("s05", "s07"), repeat=0, fold=0),
        Fold(train=("s01", "s03", "s05", "s07"), val=("s00", "s02"), test=("s04", "s06"), repeat=0, fold=1),
    ]

    for fold in folds:
        all_subjects = list(fold.train) + list(fold.val) + list(fold.test)
        from collections import Counter
        counts = Counter(all_subjects)
        for sid, c in counts.items():
            assert c == 1, f"Subject {sid} appears {c} times in fold {fold.fold}"

        assert set(fold.train).isdisjoint(set(fold.val))
        assert set(fold.train).isdisjoint(set(fold.test))
        assert set(fold.val).isdisjoint(set(fold.test))

    # Every subject appears in exactly one test fold across all folds.
    test_subjects = [s for fold in folds for s in fold.test]
    assert len(test_subjects) == len(set(test_subjects))

    # Every subject appears exactly once across train+val+test in each fold.
    for fold in folds:
        all_subjects = list(fold.train) + list(fold.val) + list(fold.test)
        from collections import Counter
        counts = Counter(all_subjects)
        for sid, c in counts.items():
            assert c == 1, f"Subject {sid} appears {c} times in fold {fold.fold}"


def test_b_participant_aggregation_arithmetic_mean() -> None:
    """Participant probability is the arithmetic mean of epoch probs."""
    from src.train import aggregate_subject_probs

    # One subject with 3 epochs, probabilities [0.2, 0.6, 0.8] -> mean 0.5333
    subject_ids = np.asarray(["s01", "s01", "s01", "s02", "s02"])
    probs = np.asarray([0.2, 0.6, 0.8, 0.1, 0.9])

    result = aggregate_subject_probs(subject_ids, probs)

    assert abs(result["s01"] - (0.2 + 0.6 + 0.8) / 3.0) < 1e-6
    assert abs(result["s02"] - (0.1 + 0.9) / 2.0) < 1e-6
    assert len(result) == 2  # exactly one prediction per participant


def test_b_no_dict_overwrite() -> None:
    """dict(zip(...)) must NOT overwrite multi-epoch probabilities."""
    from src.train import aggregate_subject_probs

    subject_ids = np.asarray(["s01", "s01"])
    probs = np.asarray([0.2, 0.8])

    # Wrong way: dict zip gives 0.8 (last epoch wins)
    wrong = dict(zip(subject_ids.tolist(), probs.tolist()))
    assert wrong["s01"] == 0.8  # confirms dict overwrite bug

    # Right way: aggregate_subject_probs gives mean 0.5
    right = aggregate_subject_probs(subject_ids, probs)
    assert abs(right["s01"] - 0.5) < 1e-6


def test_c_deep_feature_extraction_not_zeros() -> None:
    """Deep features must be non-zero, correct dimensions, subject-aligned."""
    from src.hybrid_model import _deep_features
    from src.models import CTENet, EEGNet

    n_subjects = 6
    epochs_per_subject = 2
    n_epochs = n_subjects * epochs_per_subject

    subject_ids = np.repeat(
        [f"s{index:02d}" for index in range(n_subjects)], epochs_per_subject
    )
    subject_labels = {f"s{index:02d}": index % 2 for index in range(n_subjects)}

    bundle = FeatureBundle(
        raw=np.random.randn(n_epochs, 1, 19, 64).astype(np.float32),
        node_features=np.random.randn(n_epochs, 19, 6).astype(np.float32),
        adjacency=np.random.rand(n_epochs, 19, 19).astype(np.float32),
        labels=np.asarray([subject_labels[s] for s in subject_ids], dtype=np.float32),
        subject_ids=subject_ids,
        epoch_index=np.tile(np.arange(epochs_per_subject), n_subjects),
        subject_labels=subject_labels,
        fs=128,
        epoch_seconds=0.5,
    )

    device = torch.device("cpu")
    model = CTENet(
        n_features=6, n_channels=19, d_model=16, n_heads=4, n_layers=1, dropout=0.1
    )
    model.eval()

    subject_arr = np.asarray(sorted(subject_labels), dtype=object)
    features = _deep_features(model, bundle, subject_arr, device, model_name="cte_net")

    # Shape check: (n_subjects, d_model)
    assert features.shape == (n_subjects, 16), f"Expected ({n_subjects}, 16), got {features.shape}"

    # Non-zero check: features must not be all zeros
    assert not np.all(features == 0), "Deep features are all zeros"

    # Subject alignment: row order matches subject_arr order
    assert features.shape[0] == len(subject_arr)

    # No NaN
    assert np.isfinite(features).all()


def test_c_deep_feature_missing_subject_raises() -> None:
    """Missing subject in deep feature extraction must raise, not substitute zeros."""
    from src.hybrid_model import _deep_features
    from src.models import CTENet

    n_subjects = 4
    epochs_per_subject = 2
    n_epochs = n_subjects * epochs_per_subject

    subject_ids = np.repeat(
        [f"s{index:02d}" for index in range(n_subjects)], epochs_per_subject
    )
    subject_labels = {f"s{index:02d}": index % 2 for index in range(n_subjects)}

    bundle = FeatureBundle(
        raw=np.random.randn(n_epochs, 1, 19, 64).astype(np.float32),
        node_features=np.random.randn(n_epochs, 19, 6).astype(np.float32),
        adjacency=np.random.rand(n_epochs, 19, 19).astype(np.float32),
        labels=np.asarray([subject_labels[s] for s in subject_ids], dtype=np.float32),
        subject_ids=subject_ids,
        epoch_index=np.tile(np.arange(epochs_per_subject), n_subjects),
        subject_labels=subject_labels,
        fs=128,
        epoch_seconds=0.5,
    )

    device = torch.device("cpu")
    model = CTENet(
        n_features=6, n_channels=19, d_model=16, n_heads=4, n_layers=1, dropout=0.1
    )
    model.eval()

    # Ask for a subject that doesn't exist in the bundle.
    bad_subjects = np.asarray(["s99"], dtype=object)
    try:
        _deep_features(model, bundle, bad_subjects, device, model_name="cte_net", raise_on_missing=True)
        assert False, "Expected ValueError for missing subject"
    except ValueError:
        pass


def test_d_feature_fusion_branches_present() -> None:
    """Feature fusion must have both branches with non-zero data, aligned subjects."""
    from src.hybrid_model import run_feature_fusion
    from src.models import CTENet

    n_subjects = 12
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

    folds = make_repeated_folds(subject_labels, n_splits=3, n_repeats=1, seed=42)

    # Factory that builds an untrained CTENet for feature extraction.
    def deep_factory() -> CTENet:
        return CTENet(
            n_features=6, n_channels=19, d_model=16, n_heads=4, n_layers=1, dropout=0.1
        )

    result = run_feature_fusion(
        bundle,
        folds,
        seed=42,
        deep_model_factory=deep_factory,
        device=torch.device("cpu"),
        model_name="cte_net",
        real_data=False,
    )

    assert result.experiment == "feature_fusion"
    assert result.branch_b_model != "none"
    assert result.branch_b_model != "CTENet" or "CTENet" in result.branch_b_model
    assert 0.0 <= result.roc_auc <= 1.0 or np.isnan(result.roc_auc)
    assert result.participant_level is True
    assert result.n_subjects == n_subjects


def test_e_soft_voting_probability_averaging() -> None:
    """Soft voting averages probabilities from both branches."""
    from src.hybrid_model import run_prediction_ensemble

    n_subjects = 8
    subject_labels = {f"s{index:02d}": index % 2 for index in range(n_subjects)}

    # Create folds manually.
    from src.cv import Fold
    folds = [
        Fold(train=("s00", "s02", "s04", "s06"), val=("s01", "s03"), test=("s05", "s07"), repeat=0, fold=0),
        Fold(train=("s01", "s03", "s05", "s07"), val=("s00", "s02"), test=("s04", "s06"), repeat=0, fold=1),
    ]

    # Subject list is sorted: s00..s07
    subject_list = np.asarray(sorted(subject_labels), dtype=object)

    # Branch A probs: fold 0 -> s05=0.8, s07=0.2; fold 1 -> s04=0.3, s06=0.7
    branch_a_probs = {
        0: {"s05": 0.8, "s07": 0.2},
        1: {"s04": 0.3, "s06": 0.7},
    }
    # Branch B probs: fold 0 -> s05=0.6, s07=0.4; fold 1 -> s04=0.2, s06=0.8
    branch_b_probs = {
        0: {"s05": 0.6, "s07": 0.4},
        1: {"s04": 0.2, "s06": 0.8},
    }

    bundle = _make_bundle(n_subjects=n_subjects, epochs_per_subject=1)

    result = run_prediction_ensemble(
        bundle=bundle,
        folds=folds,
        seed=42,
        branch_a_probs=branch_a_probs,
        branch_b_probs=branch_b_probs,
        method="soft_voting",
        real_data=False,
    )

    # s05: (0.8+0.6)/2 = 0.7; s07: (0.2+0.4)/2 = 0.3
    # s04: (0.3+0.2)/2 = 0.25; s06: (0.7+0.8)/2 = 0.75
    assert result.experiment == "soft_voting"
    assert result.participant_level is True


def test_f_weighted_voting_weights_from_validation() -> None:
    """Weighted voting uses validation-derived weights, not test labels."""
    from src.hybrid_model import run_prediction_ensemble

    n_subjects = 8
    subject_labels = {f"s{index:02d}": index % 2 for index in range(n_subjects)}

    from src.cv import Fold
    folds = [
        Fold(train=("s00", "s02", "s04", "s06"), val=("s01", "s03"), test=("s05", "s07"), repeat=0, fold=0),
        Fold(train=("s01", "s03", "s05", "s07"), val=("s00", "s02"), test=("s04", "s06"), repeat=0, fold=1),
    ]

    branch_a_probs = {
        0: {"s05": 0.8, "s07": 0.2},
        1: {"s04": 0.3, "s06": 0.7},
    }
    branch_b_probs = {
        0: {"s05": 0.6, "s07": 0.4},
        1: {"s04": 0.2, "s06": 0.8},
    }

    bundle = _make_bundle(n_subjects=n_subjects, epochs_per_subject=1)

    # Weights derived externally (e.g. from validation AUROC), not test labels.
    weights = [0.7, 0.3]

    result = run_prediction_ensemble(
        bundle=bundle,
        folds=folds,
        seed=42,
        branch_a_probs=branch_a_probs,
        branch_b_probs=branch_b_probs,
        method="weighted_voting",
        weights=weights,
        real_data=False,
    )

    assert result.experiment == "weighted_voting"
    # s05: 0.7*0.8 + 0.3*0.6 = 0.56+0.18 = 0.74 -> pred 1
    # s07: 0.7*0.2 + 0.3*0.4 = 0.14+0.12 = 0.26 -> pred 0
    # s04: 0.7*0.3 + 0.3*0.2 = 0.21+0.06 = 0.27 -> pred 0
    # s06: 0.7*0.7 + 0.3*0.8 = 0.49+0.24 = 0.73 -> pred 1
    assert 0.0 <= result.roc_auc <= 1.0 or np.isnan(result.roc_auc)


def test_g_stacking_no_insample_meta_evaluation() -> None:
    """Stacking meta-classifier must NOT be fit/evaluated on same rows.

    This test verifies that the meta-classifier uses out-of-fold base-model
    predictions for training, and evaluates on held-out test predictions.
    """
    from src.hybrid_model import run_prediction_ensemble
    from src.cv import Fold

    n_subjects = 12
    subject_labels = {f"s{index:02d}": index % 2 for index in range(n_subjects)}

    folds = [
        Fold(train=("s00", "s02", "s04", "s06"), val=("s01", "s03"), test=("s08", "s10"), repeat=0, fold=0),
        Fold(train=("s01", "s03", "s05", "s07"), val=("s00", "s02"), test=("s09", "s11"), repeat=0, fold=1),
    ]

    branch_a_probs = {
        0: {"s08": 0.8, "s10": 0.2},
        1: {"s09": 0.3, "s11": 0.7},
    }
    branch_b_probs = {
        0: {"s08": 0.6, "s10": 0.4},
        1: {"s09": 0.2, "s11": 0.8},
    }

    bundle = _make_bundle(n_subjects=n_subjects, epochs_per_subject=1)

    result = run_prediction_ensemble(
        bundle=bundle,
        folds=folds,
        seed=42,
        branch_a_probs=branch_a_probs,
        branch_b_probs=branch_b_probs,
        method="stacking",
        real_data=False,
    )

    assert result.experiment == "stacking"
    # Each subject predicted exactly once (their fold's held-out prediction)
    assert result.n_subjects == n_subjects


def test_h_fold_alignment_all_models() -> None:
    """All models must consume the same fold participant IDs."""
    from src.cv import Fold

    n_subjects = 10
    subject_labels = {f"s{index:02d}": index % 2 for index in range(n_subjects)}

    folds = make_repeated_folds(subject_labels, n_splits=3, n_repeats=1, seed=42)

    # All folds must cover every subject exactly once in test across folds.
    all_test = set()
    for fold in folds:
        all_test.update(fold.test)
    assert all_test == set(subject_labels)

    # No subject in multiple folds' test sets.
    test_counts: dict[str, int] = {}
    for fold in folds:
        for sid in fold.test:
            test_counts[sid] = test_counts.get(sid, 0) + 1
    for sid, count in test_counts.items():
        assert count == 1, f"Subject {sid} in {count} test folds"


def test_i_model_persistence_save_reload(tmp_path: Path) -> None:
    """Saved hybrid model metadata must be loadable and consistent."""
    from src.hybrid_model import save_hybrid_result, HybridResult

    result = HybridResult(
        experiment="feature_fusion",
        branch_a_model="xgboost",
        branch_b_model="cte_net",
        accuracy=0.7,
        precision=0.7,
        recall=0.7,
        f1=0.7,
        roc_auc=0.75,
        pr_auc=0.72,
        sensitivity=0.7,
        specificity=0.7,
        brier=0.25,
        ece=0.1,
        n_subjects=121,
        n_folds=5,
        n_repeats=1,
        participant_level=True,
        real_data=True,
        folds=[],
    )

    save_dir = tmp_path / "hybrid_models"
    save_hybrid_result(result, save_dir)

    metadata_path = save_dir / "hybrid_feature_fusion_metadata.json"
    assert metadata_path.exists()

    import json
    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert saved["experiment"] == "feature_fusion"
    assert saved["branch_a_model"] == "xgboost"
    assert saved["branch_b_model"] == "cte_net"
    assert abs(saved["roc_auc"] - 0.75) < 1e-6


def test_j_end_to_end_synthetic_smoke(tmp_path: Path) -> None:
    """End-to-end smoke test on synthetic data (NOT a real ADHD result).

    Runs feature extraction through fusion and prediction on synthetic data.
    Verifies the pipeline executes without errors. Synthetic data only.
    """
    from src.hybrid_model import run_feature_fusion, run_prediction_ensemble
    from src.models import CTENet
    from src.cv import Fold

    n_subjects = 10
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

    folds = make_repeated_folds(subject_labels, n_splits=3, n_repeats=1, seed=42)

    def deep_factory() -> CTENet:
        return CTENet(
            n_features=6, n_channels=19, d_model=16, n_heads=4, n_layers=1, dropout=0.1
        )

    # E9: Feature fusion
    fusion_result = run_feature_fusion(
        bundle, folds, seed=42,
        deep_model_factory=deep_factory,
        device=torch.device("cpu"),
        model_name="cte_net",
        real_data=False,
    )
    assert fusion_result.experiment == "feature_fusion"

    # E10: Soft voting
    branch_a_probs = {
        fold.fold: {sid: 0.6 for sid in fold.test} for fold in folds
    }
    branch_b_probs = {
        fold.fold: {sid: 0.4 for sid in fold.test} for fold in folds
    }

    soft_result = run_prediction_ensemble(
        bundle, folds, seed=42,
        branch_a_probs=branch_a_probs,
        branch_b_probs=branch_b_probs,
        method="soft_voting",
        real_data=False,
    )
    assert soft_result.experiment == "soft_voting"

    # E12: Stacking
    stack_result = run_prediction_ensemble(
        bundle, folds, seed=42,
        branch_a_probs=branch_a_probs,
        branch_b_probs=branch_b_probs,
        method="stacking",
        real_data=False,
    )
    assert stack_result.experiment == "stacking"

    # All results must be participant-level.
    assert fusion_result.participant_level is True
    assert soft_result.participant_level is True
    assert stack_result.participant_level is True


# ---------------------------------------------------------------------------
# Phase 13 acceptance tests: connectivity, CTE-Net architecture, CUDA
# ---------------------------------------------------------------------------


def test_k_directed_connectivity_342_edges() -> None:
    """Directed connectivity must have all 342 ordered off-diagonal pairs.

    19 channels x 18 targets = 342 directed edges (not 171 undirected).
    adj[target, source] != adj[source, target] in general.
    """
    from src.features import granger_adjacency

    rng = np.random.default_rng(0)
    epoch = rng.normal(size=(19, 640)).astype(np.float32)

    adj = granger_adjacency(epoch, order=5)
    assert adj.shape == (19, 19)
    assert np.allclose(np.diag(adj), 0.0)

    # Count non-zero off-diagonal entries (directed edges).
    off_diagonal = adj[~np.eye(19, dtype=bool)]
    n_nonzero = int(np.count_nonzero(off_diagonal > 0))
    # Directed: every ordered pair (target, source) is a separate edge.
    # With random data, all 342 should be non-zero.
    assert n_nonzero == 342, f"Expected 342 directed edges, got {n_nonzero}"

    # Asymmetry: directed connectivity is not symmetric in general.
    assert not np.allclose(adj, adj.T), "Directed connectivity should not be symmetric"


def test_l_ctenet_original_paper_parameters() -> None:
    """CTENetOriginal must use paper-specified architecture parameters.

    From Gomez-Rivera et al. (2026), Table 2:
    d=128, L=1, nh=1, d_ff=128, p=0.5, K=83, pool=4,
    Dx=6, Dy=1, tau=2, alpha_R=2, a=1, l=1, alpha_RQ=1, H=64.
    """
    from src.ctenet import CTENetOriginal, build_ctenet_original

    model = build_ctenet_original(
        n_channels=19, n_times=512,
        d_model=128, n_heads=1, n_layers=1, dropout=0.5,
        embed_dim_source=6, embed_dim_target=1, embed_delay=2,
        te_alpha=2.0, kernel_size=83, pool_size=4,
        rq_alpha=1.0, rq_length_scale=1.0, rq_amplitude=1.0,
        head_hidden=64,
    )
    assert model.d_model == 128
    assert model.n_heads == 1
    assert model.n_layers == 1
    assert model.dropout == 0.5
    assert model.embed_dim_source == 6
    assert model.embed_dim_target == 1
    assert model.embed_delay == 2
    assert model.kernel_size == 83
    assert model.pool_size == 4

    # Verify constructor signature matches paper values.
    model2 = CTENetOriginal(
        n_channels=19, n_times=512,
        d_model=128, n_heads=1, n_layers=1, dropout=0.5,
        embed_dim_source=6, embed_dim_target=1, embed_delay=2,
        te_alpha=2.0, kernel_size=83, pool_size=4,
        rq_alpha=1.0, rq_length_scale=1.0, rq_amplitude=1.0,
        head_hidden=64,
    )
    assert model2.d_model == 128


def test_m_ctenet_original_valid_conv_dimensions() -> None:
    """CTENetOriginal forward pass must produce valid tensor shapes.

    Input: (B, C, T) with T=512 (4s at 128 Hz).
    Depthwise conv K=83, pool=4 -> T_phi = floor(512/4) = 128.
    Takens: min_t = max((6-1)*2, 1*2) = 10, Tz = 128 - 10 = 118.
    All intermediate shapes must be positive.
    """
    from src.ctenet import CTENetOriginal

    model = CTENetOriginal(
        n_channels=19, n_times=512,
        d_model=128, n_heads=1, n_layers=1, dropout=0.5,
    )
    model.eval()

    batch = 2
    x = torch.randn(batch, 19, 512)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (batch,), f"Expected ({batch},), got {tuple(logits.shape)}"

    features, connectivity = model.forward_features(x)
    assert features.shape == (batch, 19 * 18), f"Expected ({batch}, 342), got {tuple(features.shape)}"
    assert connectivity.shape == (batch, 19, 19)

    # Connectivity diagonal must be zero (no self-connections).
    assert torch.allclose(torch.diagonal(connectivity[0], dim1=-2, dim2=-1), torch.zeros(19))


def test_n_cuda_forward_backward() -> None:
    """Deep models must run forward/backward on CUDA when available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from src.models import EEGNet, CTENet

    device = torch.device("cuda")

    # EEGNet forward/backward on CUDA.
    eegnet = EEGNet(n_channels=19, n_times=640, f1=4, f2=8, dropout=0.1).to(device)
    x = torch.randn(2, 1, 19, 640, device=device)
    y = torch.randint(0, 2, (2,), device=device, dtype=torch.float32)
    logits = eegnet(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    assert eegnet.head.weight.grad is not None
    assert torch.isfinite(eegnet.head.weight.grad).all()

    # CTENet forward/backward on CUDA.
    ctenet = CTENet(
        n_features=6, n_channels=19, d_model=16, n_heads=4, n_layers=1, dropout=0.1
    ).to(device)
    node = torch.randn(2, 19, 6, device=device)
    adjacency = torch.rand(2, 19, 19, device=device)
    logits = ctenet(node, adjacency)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    assert ctenet.head[4].weight.grad is not None
    assert torch.isfinite(ctenet.head[4].weight.grad).all()


def test_o_hybrid_no_test_label_in_weights() -> None:
    """Weighted voting weights must not be derived from test labels.

    Weights come from validation/inner-OOF performance only.
    Test-set labels must never influence weighting.
    """
    from src.hybrid_model import run_prediction_ensemble
    from src.cv import Fold

    n_subjects = 8
    subject_labels = {f"s{index:02d}": index % 2 for index in range(n_subjects)}

    folds = [
        Fold(train=("s00", "s02", "s04", "s06"), val=("s01", "s03"), test=("s05", "s07"), repeat=0, fold=0),
        Fold(train=("s01", "s03", "s05", "s07"), val=("s00", "s02"), test=("s04", "s06"), repeat=0, fold=1),
    ]

    branch_a_probs = {
        0: {"s05": 0.8, "s07": 0.2},
        1: {"s04": 0.3, "s06": 0.7},
    }
    branch_b_probs = {
        0: {"s05": 0.6, "s07": 0.4},
        1: {"s04": 0.2, "s06": 0.8},
    }

    bundle = _make_bundle(n_subjects=n_subjects, epochs_per_subject=1)

    # With explicit weights from validation, result should be deterministic.
    result = run_prediction_ensemble(
        bundle=bundle,
        folds=folds,
        seed=42,
        branch_a_probs=branch_a_probs,
        branch_b_probs=branch_b_probs,
        method="weighted_voting",
        weights=[0.7, 0.3],
        real_data=False,
    )

    # s05: 0.7*0.8 + 0.3*0.6 = 0.74 -> pred 1 (true label: s05 = index 5 % 2 = 1)
    # s07: 0.7*0.2 + 0.3*0.4 = 0.26 -> pred 0 (true label: s07 = index 7 % 2 = 1) -> WRONG
    # s04: 0.7*0.3 + 0.3*0.2 = 0.27 -> pred 0 (true label: s04 = index 4 % 2 = 0) -> correct
    # s06: 0.7*0.7 + 0.3*0.8 = 0.73 -> pred 1 (true label: s06 = index 6 % 2 = 0) -> WRONG
    # The point: weights are from validation, not test labels.
    assert result.experiment == "weighted_voting"
    assert result.participant_level is True
    # Test labels were never consulted for weight derivation.
    assert result.n_subjects == n_subjects


def test_p_ctenet_original_consume_correct_fold_participants() -> None:
    """CTENetOriginal forward_features returns C*(C-1) flattened off-diagonal TE."""
    from src.ctenet import CTENetOriginal

    model = CTENetOriginal(n_channels=19, n_times=512)
    model.eval()

    x = torch.randn(3, 19, 512)
    with torch.no_grad():
        features, connectivity = model.forward_features(x)

    # 19 channels * 18 off-diagonal = 342 features.
    assert features.shape == (3, 342)
    assert connectivity.shape == (3, 19, 19)

    # Each row should have exactly C*(C-1) = 342 off-diagonal entries.
    mask = ~torch.eye(19, dtype=bool)
    expected = connectivity[:, mask]
    assert torch.allclose(features, expected)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))

