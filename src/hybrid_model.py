"""Hybrid ADHD detection model: feature-level fusion and prediction-level ensemble.

Branch A: Statistical/spectral feature branch (classical ML on Paper 1 features).
Branch B: Deep learning branch (CTE-Net inspired or original).
Branch C: Feature-level fusion (concat Branch-A probability + deep features,
         train a meta-classifier on training folds only).
Branch D: Prediction-level ensemble (soft voting, weighted voting, stacking).

All splitting is subject-wise. Preprocessing, feature selection, and scalers
are fitted on training subjects only. No test-set information is used for
weight selection or threshold tuning.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.classical_models import (
    BANDS,
    CHANNELS,
    ClassicalResult,
    _build_epoch_features,
    _subject_probs,
    evaluate_model,
    make_model,
)
from src.cv import Fold, assert_no_subject_overlap
from src.evaluate import binary_metrics, youden_threshold


@dataclass
class HybridResult:
    experiment: str  # e.g. "feature_fusion", "soft_voting", "weighted_voting", "stacking"
    branch_a_model: str
    branch_b_model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    sensitivity: float
    specificity: float
    brier: float
    ece: float
    n_subjects: int
    n_folds: int
    n_repeats: int
    participant_level: bool
    real_data: bool
    folds: list[dict] = field(default_factory=list)


def _deep_predict_proba(
    model: torch.nn.Module,
    bundle,
    subject_ids_subset: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
    model_name: str = "cte_net",
) -> np.ndarray:
    """Predict subject-level probabilities using a deep model on epochs.

    Averages epoch probabilities per subject. Works with both CTENetInspired
    (forward(x, adjacency)) and CTENetOriginal (forward(x, adjacency=None)).
    """
    from torch.utils.data import DataLoader

    from src.data import EpochDataset
    from src.train import resolve_device

    if device is None:
        device = resolve_device("cpu")

    indices = np.flatnonzero(np.isin(bundle.subject_ids, subject_ids_subset))
    if len(indices) == 0:
        return np.array([])

    dataset = EpochDataset(bundle, indices, model_name)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model.eval()
    model.to(device)
    probs = []
    with torch.no_grad():
        for batch in loader:
            if model_name == "eegnet" or model_name == "ctenet_original":
                features, labels = batch
                logits = model(features.to(device))
            else:
                features, adjacency, labels = batch
                logits = model(features.to(device), adjacency.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    if not probs:
        return np.array([])
    epoch_probs = np.concatenate(probs)
    epoch_subjects = bundle.subject_ids[indices]

    subject_probs = _subject_probs(epoch_probs, epoch_subjects)
    aligned = np.asarray([float(subject_probs[str(s)]) for s in subject_ids_subset])
    return aligned


def _deep_features(
    model: torch.nn.Module,
    bundle,
    subject_ids_subset: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
    model_name: str = "cte_net",
    raise_on_missing: bool = True,
) -> np.ndarray:
    """Extract deep learned features per subject.

    Returns (n_subjects, d_model). Raises if any expected subject is missing
    from the extracted features -- never substitutes zeros.

    Strict alignment check: verifies that every subject in
    ``subject_ids_subset`` has a corresponding feature vector, and that output
    row order matches ``subject_ids_subset`` order.
    """
    from torch.utils.data import DataLoader

    from src.data import EpochDataset
    from src.train import resolve_device

    if device is None:
        device = resolve_device("cpu")
    if not hasattr(model, "forward_features"):
        raise ValueError(
            f"Model {type(model).__name__} has no forward_features method; "
            "cannot extract deep representations."
        )

    indices = np.flatnonzero(np.isin(bundle.subject_ids, subject_ids_subset))
    if len(indices) == 0:
        raise ValueError(
            f"No epoch indices found for subjects {subject_ids_subset.tolist()}"
        )

    dataset = EpochDataset(bundle, indices, model_name)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model.eval()
    model.to(device)
    per_subject: dict[str, list[np.ndarray]] = {}
    d_model = getattr(model, "d_model", 64)
    found_subjects: set[str] = set()
    with torch.no_grad():
        offset = 0
        for batch in loader:
            if model_name == "eegnet" or model_name == "ctenet_original":
                features, labels = batch
                feat, _, _ = model.forward_features(features.to(device))
            else:
                features, adjacency, labels = batch
                feat, _, _ = model.forward_features(
                    features.to(device), adjacency.to(device)
                )
            batch_feat = feat.cpu().numpy()
            for i in range(len(batch_feat)):
                sid = str(bundle.subject_ids[indices[offset + i]])
                per_subject.setdefault(sid, []).append(batch_feat[i])
                found_subjects.add(sid)
            offset += len(batch_feat)

    # Strict alignment check: every expected subject must be present.
    expected = [str(s) for s in subject_ids_subset]
    missing = [s for s in expected if s not in found_subjects]
    if missing:
        if raise_on_missing:
            raise ValueError(
                f"Deep feature extraction missing {len(missing)} expected "
                f"subjects: {missing[:10]}{'...' if len(missing) > 10 else ''}"
            )

    aligned = []
    for s in expected:
        vecs = per_subject.get(s, [])
        if len(vecs) == 0:
            raise ValueError(
                f"No feature vectors extracted for subject {s}; cannot "
                "substitute zeros."
            )
        aligned.append(np.mean(np.stack(vecs), axis=0))

    result = np.stack(aligned) if aligned else np.empty((0, d_model))
    # Verify output shape matches expected.
    if result.shape[0] != len(expected):
        raise ValueError(
            f"Feature alignment failure: expected {len(expected)} rows, "
            f"got {result.shape[0]}."
        )
    return result


def _make_inner_folds(
    subjects: list[str],
    labels: dict[str, int],
    n_splits: int = 3,
    seed: int = 42,
) -> list[tuple[list[str], list[str]]]:
    """Create subject-wise inner folds for OOF prediction generation.

    Returns list of (inner_train_subjects, inner_val_subjects) tuples.
    """
    from sklearn.model_selection import StratifiedKFold

    subjects_arr = np.asarray(subjects, dtype=object)
    y = np.asarray([labels[str(s)] for s in subjects_arr], dtype=int)

    if len(np.unique(y)) < 2 or len(subjects_arr) < 2:
        # Degenerate case: return one fold with all as train, none as val.
        return [(list(subjects), [])]

    # Cap n_splits at the smallest class count to avoid StratifiedKFold error.
    class_counts = np.bincount(y)
    min_class = int(class_counts[class_counts > 0].min())
    actual_splits = min(n_splits, min_class)
    if actual_splits < 2:
        return [(list(subjects), [])]

    inner = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=seed)
    folds: list[tuple[list[str], list[str]]] = []
    for train_idx, val_idx in inner.split(subjects_arr, y):
        folds.append(
            (
                subjects_arr[train_idx].tolist(),
                subjects_arr[val_idx].tolist(),
            )
        )
    return folds


def _train_deep_model_on_subjects(
    bundle,
    train_subjects: list[str],
    val_subjects: list[str],
    seed: int,
    device: torch.device,
    model_name: str,
    model_factory: Optional[Callable[[], torch.nn.Module]] = None,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
):
    """Train a deep model on train subjects with early stopping on val subjects.

    Returns the model with best-validation weights loaded.  The model is never
    exposed to any subject outside ``train_subjects`` during training.
    """
    from torch.utils.data import DataLoader

    from src.data import EpochDataset
    from src.models import build_model
    from src.train import aggregate_subject_probs

    from src.config import ModelConfig

    val_set = set(map(str, val_subjects)) if val_subjects else set()
    # Exclude validation/development subjects from actual training to prevent
    # train/dev overlap leakage.
    actual_train_subjects = [s for s in train_subjects if str(s) not in val_set]
    assert set(map(str, actual_train_subjects)).isdisjoint(val_set), (
        "Train/dev participant overlap detected"
    )
    # Never silently restore validation subjects into training.  If there are
    # no actual training subjects after excluding validation, raise an error
    # rather than leaking dev data into the training set.
    if len(actual_train_subjects) == 0:
        raise ValueError(
            f"No actual training subjects remain after excluding validation "
            f"subjects {sorted(val_set)} from train subjects "
            f"{sorted(map(str, train_subjects))}.  Cannot train deep model "
            "without train/dev disjointness."
        )
    val_indices = (
        np.flatnonzero(np.isin(bundle.subject_ids, val_subjects))
        if val_subjects
        else np.array([], dtype=int)
    )
    train_indices = np.flatnonzero(np.isin(bundle.subject_ids, actual_train_subjects))
    if len(train_indices) == 0:
        raise ValueError(
            f"No training epochs available for subjects {actual_train_subjects}; "
            "cannot train deep model."
        )
    # Assert train/dev/test participant disjointness before every training op.
    test_subjects = getattr(bundle, "test_subjects", None)
    if test_subjects is not None:
        test_set = set(map(str, test_subjects))
        assert set(map(str, actual_train_subjects)).isdisjoint(test_set), (
            "Train/test participant overlap detected"
        )
        assert val_set.isdisjoint(test_set), (
            "Dev/test participant overlap detected"
        )

    if model_factory is not None:
        model = model_factory().to(device)
    else:
        cfg = ModelConfig(name=model_name)
        # Derive n_times from actual bundle data, not hardcoded.
        n_times = int(bundle.raw.shape[-1]) if bundle.raw.ndim >= 4 else int(bundle.fs * bundle.epoch_seconds)
        model = (
            build_model(cfg, n_channels=19, n_features=6, n_times=n_times)
            .to(device)
        )

    train_dataset = EpochDataset(bundle, train_indices, model_name)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    val_loader = None
    if len(val_indices) > 0:
        val_dataset = EpochDataset(bundle, val_indices, model_name)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    train_labels = bundle.labels[train_indices]
    n_pos = float(train_labels.sum())
    n_neg = float(len(train_labels) - n_pos)
    pos_weight = torch.tensor(
        [n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_auc = -float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if model_name == "eegnet" or model_name == "ctenet_original":
                feats, labels_batch = batch
                logits = model(feats.to(device))
            else:
                feats, adj, labels_batch = batch
                logits = model(feats.to(device), adj.to(device))
            loss = criterion(logits, labels_batch.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if val_loader is None:
            break

        model.eval()
        val_probs_list = []
        val_subject_list = []
        with torch.no_grad():
            for batch in val_loader:
                if model_name == "eegnet" or model_name == "ctenet_original":
                    feats, labels_batch = batch
                    logits = model(feats.to(device))
                else:
                    feats, adj, labels_batch = batch
                    logits = model(feats.to(device), adj.to(device))
                val_probs_list.append(torch.sigmoid(logits).cpu().numpy())
                val_subject_list.extend(
                    bundle.subject_ids[val_dataset.indices[
                        len(val_subject_list) : len(val_subject_list) + len(logits)
                    ]].tolist()
                )
        if val_probs_list:
            val_epoch_probs = np.concatenate(val_probs_list)
            val_subject_probs = aggregate_subject_probs(
                np.asarray(val_subject_list), val_epoch_probs
            )
            val_y_arr = np.asarray(
                [bundle.subject_labels[str(s)] for s in val_subject_probs.index],
                dtype=int,
            )
            val_auc = _safe_auc(val_y_arr, val_subject_probs.to_numpy())
        else:
            val_auc = float("nan")

        if np.isfinite(val_auc) and val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _train_deep_oof(
    bundle,
    train_subj: list[str],
    val_subj: list[str],
    seed: int,
    device: torch.device,
    model_name: str,
    model_factory: Optional[Callable[[], torch.nn.Module]] = None,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
) -> dict[str, float]:
    """Train a deep model on train_subj, return subject-level OOF probs for val_subj.

    Uses a separate early-stopping dev split carved from train_subj so that
    val_subj labels never influence checkpoint selection.
    """
    from src.train import resolve_device

    if device is None:
        device = resolve_device("cpu")

    n_train = len(train_subj)
    early_stopping_disabled_reason = None
    if n_train < 2:
        dev_subj = []
        early_stopping_disabled_reason = (
            f"only {n_train} training subject(s); need at least 2 for a dev split"
        )
    else:
        # Carve out 20% of inner-train as dev for early stopping (stratified).
        # Ensure at least one subject per class remains for training.
        train_labels = np.asarray(
            [bundle.subject_labels[str(s)] for s in train_subj], dtype=int
        )
        pos = [s for s, l in zip(train_subj, train_labels) if l == 1]
        neg = [s for s, l in zip(train_subj, train_labels) if l == 0]
        rng = np.random.default_rng(seed)
        pos_dev_size = min(max(1, int(round(0.2 * len(pos)))), max(0, len(pos) - 1))
        neg_dev_size = min(max(1, int(round(0.2 * len(neg)))), max(0, len(neg) - 1))
        pos_dev = (
            rng.choice(pos, size=pos_dev_size, replace=False)
            if pos and pos_dev_size > 0
            else np.array([], dtype=object)
        )
        neg_dev = (
            rng.choice(neg, size=neg_dev_size, replace=False)
            if neg and neg_dev_size > 0
            else np.array([], dtype=object)
        )
        dev_subj = list(pos_dev) + list(neg_dev)
        # If dev consumed everything from a class, early stopping cannot
        # meaningfully operate (no train subjects left for that class).
        if len(dev_subj) >= n_train:
            early_stopping_disabled_reason = (
                "dev split would consume all training subjects; "
                "training without early stopping"
            )
            dev_subj = []

    if early_stopping_disabled_reason:
        print(
            f"[deep_oof] Early stopping disabled: {early_stopping_disabled_reason}"
        )

    deep_model = _train_deep_model_on_subjects(
        bundle,
        train_subj,
        dev_subj,
        seed,
        device,
        model_name,
        model_factory=model_factory,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
    )

    # Extract participant-level probabilities for val_subj.
    val_subject_arr = np.asarray(val_subj, dtype=object)
    deep_probs = _deep_predict_proba(
        deep_model,
        bundle,
        val_subject_arr,
        device,
        batch_size=batch_size,
        model_name=model_name,
    )
    return {str(s): float(p) for s, p in zip(val_subj, deep_probs)}


def run_feature_fusion(
    bundle,
    folds: list[Fold],
    seed: int,
    deep_model_factory: Optional[Callable] = None,
    deep_config: Optional[dict] = None,
    device: Optional[torch.device] = None,
    model_name: str = "cte_net",
    real_data: bool = True,
) -> HybridResult:
    """Branch C: probability-level feature fusion with genuine OOF meta-features.

    Uses calibrated participant-level branch probabilities as meta-features
    instead of concatenating deep latent representations from independently
    trained models (which do not share a coordinate system).

    Per outer fold:
      1. Split outer train subjects into inner subject-wise folds.
      2. For each inner fold:
         a. Split inner-train into sub-train / sub-dev for early stopping.
         b. Train Branch A (XGBoost) on inner-train only.
         c. Train Branch B (deep model) on inner-train only,
            using sub-dev for early stopping/checkpoint selection.
         d. Generate OOF XGBoost participant-level probabilities for inner-val.
         e. Generate OOF deep model participant-level probabilities for inner-val.
      3. Concatenate all inner-val OOF probabilities -> OOF fused features
         [xgb_prob, deep_prob] for every outer-train subject.
      4. Train Logistic Regression meta-classifier on OOF fused features.
      5. Train final Branch A + Branch B on all outer-train subjects.
      6. Tune meta-classifier threshold on outer-val subjects using
         predictions from models trained on outer-train only.
      7. Retrain Branch A + Branch B on train+val subjects.
      8. Predict held-out outer-test subjects, apply meta-classifier,
         apply tuned threshold.

    Meta-classifier is trained on OOF training-fold outputs only -- never on
    in-sample predictions. Deep model is never trained on test subjects.
    Validation subjects used for early stopping never contribute OOF features
    to the meta-classifier.
    """
    from collections.abc import Callable
    from torch.utils.data import DataLoader

    from src.train import (
        EpochDataset,
        aggregate_subject_probs,
        indices_for_subjects,
        resolve_device,
        train_one_fold,
    )
    from src.models import build_model as _build_model_from_config

    if device is None:
        device = resolve_device("cpu")

    feature_set = "node+adj+rel"
    a_features = _build_epoch_features(bundle, feature_set)

    subject_list = np.asarray(sorted(bundle.subject_labels), dtype=object)
    all_subject_labels = dict(bundle.subject_labels)

    probs = np.zeros(len(bundle.labels), dtype=float)
    fold_thresholds: dict[int, float] = {}
    fold_of_subject: dict[str, int] = {}
    fold_rows: list[dict] = []

    n_inner = 3

    for fold, fold_spec in enumerate(folds):
        train_subjects = list(fold_spec.train)
        val_subjects = list(fold_spec.val)
        test_subjects = list(fold_spec.test)

        assert_no_subject_overlap(train_subjects, val_subjects)
        assert_no_subject_overlap(train_subjects, test_subjects)
        assert_no_subject_overlap(val_subjects, test_subjects)

        train_index = np.flatnonzero(np.isin(bundle.subject_ids, train_subjects))
        val_index = np.flatnonzero(np.isin(bundle.subject_ids, val_subjects))
        full_index = np.flatnonzero(
            np.isin(bundle.subject_ids, train_subjects + val_subjects)
        )
        test_index = np.flatnonzero(np.isin(bundle.subject_ids, test_subjects))

        # ----- Step 1: Inner CV on outer train subjects -----
        inner_folds = _make_inner_folds(
            train_subjects, all_subject_labels, n_splits=n_inner, seed=seed
        )

        oof_a_probs: dict[str, float] = {}
        oof_b_probs: dict[str, float] = {}

        for inner_train_subj, inner_val_subj in inner_folds:
            if len(inner_val_subj) == 0:
                continue

            inner_train_index = np.flatnonzero(
                np.isin(bundle.subject_ids, inner_train_subj)
            )
            inner_val_index = np.flatnonzero(
                np.isin(bundle.subject_ids, inner_val_subj)
            )

            # Branch A: XGBoost on inner-train, predict inner-val (OOF).
            inner_xgb = make_model(
                "xgboost", seed, n_features=a_features.shape[1], k=150
            )
            inner_xgb.fit(
                a_features[inner_train_index],
                bundle.labels[inner_train_index].astype(int),
            )
            inner_val_prob = inner_xgb.predict_proba(
                a_features[inner_val_index]
            )[:, 1]
            inner_val_subject_probs = _subject_probs(
                inner_val_prob, bundle.subject_ids[inner_val_index]
            )
            for sid, p in inner_val_subject_probs.items():
                oof_a_probs[str(sid)] = float(p)

            # Branch B: Deep model on inner-train, OOF probs for inner-val.
            # Early stopping uses a dev split carved from inner-train,
            # NOT the inner-val subjects (to avoid label leakage into
            # checkpoint selection).
            if deep_model_factory is not None:
                inner_deep_probs = _train_deep_oof(
                    bundle,
                    list(inner_train_subj),
                    list(inner_val_subj),
                    seed,
                    device,
                    model_name,
                    model_factory=deep_model_factory,
                    epochs=30,
                    batch_size=32,
                    lr=1e-3,
                    weight_decay=1e-4,
                    patience=10,
                )
                oof_b_probs.update(inner_deep_probs)

        # Verify every outer-train subject has OOF predictions.
        missing_a = [s for s in train_subjects if str(s) not in oof_a_probs]
        if missing_a:
            raise RuntimeError(
                f"Fold {fold}: missing OOF XGBoost predictions for {missing_a}"
            )
        if deep_model_factory is not None:
            missing_b = [s for s in train_subjects if str(s) not in oof_b_probs]
            if missing_b:
                raise RuntimeError(
                    f"Fold {fold}: missing OOF deep predictions for {missing_b}"
                )

        # ----- Step 2: Train meta-classifier on OOF fused probabilities -----
        X_oof = []
        y_oof = []
        for sid in train_subjects:
            sid_str = str(sid)
            row = [oof_a_probs[sid_str]]
            if deep_model_factory is not None:
                row.append(oof_b_probs[sid_str])
            X_oof.append(row)
            y_oof.append(all_subject_labels[sid_str])

        X_oof = np.asarray(X_oof, dtype=float)
        y_oof = np.asarray(y_oof, dtype=int)

        meta = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
        meta.fit(X_oof, y_oof)

        # ----- Step 3: Train final models on all outer-train subjects -----
        branch_a = make_model(
            "xgboost", seed, n_features=a_features.shape[1], k=150
        )
        branch_a.fit(
            a_features[train_index],
            bundle.labels[train_index].astype(int),
        )

        final_deep_model = None
        if deep_model_factory is not None:
            final_deep_model = _train_deep_model_on_subjects(
                bundle,
                train_subjects,
                val_subjects,
                seed,
                device,
                model_name,
                model_factory=deep_model_factory,
                epochs=30,
                batch_size=32,
                lr=1e-3,
                weight_decay=1e-4,
                patience=10,
            )

        # ----- Step 4: Tune meta threshold on outer-val subjects -----
        val_a_prob = branch_a.predict_proba(a_features[val_index])[:, 1]
        val_a_map = _subject_probs(val_a_prob, bundle.subject_ids[val_index])

        val_b_probs = None
        if final_deep_model is not None:
            val_subject_arr = np.asarray(val_subjects, dtype=object)
            val_b_arr = _deep_predict_proba(
                final_deep_model,
                bundle,
                val_subject_arr,
                device,
                batch_size=32,
                model_name=model_name,
            )
            val_b_probs = {str(s): float(p) for s, p in zip(val_subjects, val_b_arr)}

        val_fused = []
        for i, sid in enumerate(val_subjects):
            sid_str = str(sid)
            row = [float(val_a_map[sid_str])]
            if val_b_probs is not None:
                row.append(val_b_probs[sid_str])
            val_fused.append(row)

        val_meta_prob = meta.predict_proba(np.asarray(val_fused, dtype=float))[:, 1]
        val_y = np.asarray(
            [all_subject_labels[str(s)] for s in val_subjects], dtype=int
        )
        meta_threshold = float(youden_threshold(val_y, val_meta_prob))
        fold_thresholds[fold] = meta_threshold

        # ----- Step 5: Train final models on train+val, predict on test -----
        branch_a_full = make_model(
            "xgboost", seed, n_features=a_features.shape[1], k=150
        )
        branch_a_full.fit(
            a_features[full_index],
            bundle.labels[full_index].astype(int),
        )

        final_deep_model_full = None
        if deep_model_factory is not None:
            final_deep_model_full = _train_deep_model_on_subjects(
                bundle,
                train_subjects + val_subjects,
                [],
                seed,
                device,
                model_name,
                epochs=30,
                batch_size=32,
                lr=1e-3,
                weight_decay=1e-4,
                patience=10,
            )

        test_a_prob = branch_a_full.predict_proba(a_features[test_index])[:, 1]
        test_a_map = _subject_probs(test_a_prob, bundle.subject_ids[test_index])

        test_b_probs = None
        if final_deep_model_full is not None:
            test_subject_arr = np.asarray(test_subjects, dtype=object)
            test_b_arr = _deep_predict_proba(
                final_deep_model_full,
                bundle,
                test_subject_arr,
                device,
                batch_size=32,
                model_name=model_name,
            )
            test_b_probs = {str(s): float(p) for s, p in zip(test_subjects, test_b_arr)}

        test_fused = []
        for i, sid in enumerate(test_subjects):
            sid_str = str(sid)
            row = [float(test_a_map[sid_str])]
            if test_b_probs is not None:
                row.append(test_b_probs[sid_str])
            test_fused.append(row)

        test_meta_prob = meta.predict_proba(
            np.asarray(test_fused, dtype=float)
        )[:, 1]

        # Participant-level: apply meta threshold to test predictions.
        for i, subject in enumerate(test_subjects):
            sid_str = str(subject)
            probs[
                np.flatnonzero(bundle.subject_ids == subject)
            ] = float(test_meta_prob[i])
            fold_of_subject[sid_str] = fold

    subject_probs = _subject_probs(probs, bundle.subject_ids)
    subject_y = np.asarray(
        [all_subject_labels[str(s)] for s in subject_probs.index], dtype=int
    )
    y_pred = np.asarray(
        [
            int(subject_probs[subject] >= fold_thresholds[fold_of_subject[str(subject)]])
            for subject in subject_probs.index
        ],
        dtype=int,
    )

    metrics = binary_metrics(
        subject_y, subject_probs.to_numpy(), y_pred=y_pred, n_boot=2000, seed=seed
    )

    predictions = pd.DataFrame(
        {
            "subject_id": list(subject_probs.index),
            "y_true": subject_y,
            "y_prob": subject_probs.to_numpy(),
            "y_pred": y_pred,
            "fold": [fold_of_subject[str(s)] for s in subject_probs.index],
        }
    )

    for fold_id, group in predictions.groupby("fold"):
        fold_rows.append(
            {
                "fold": int(fold_id),
                "auroc": float(
                    roc_auc_score(group["y_true"], group["y_prob"])
                    if len(np.unique(group["y_true"])) > 1
                    else float("nan")
                ),
                "accuracy": float((group["y_pred"] == group["y_true"]).mean()),
                "sensitivity": float(
                    ((group["y_pred"] == 1) & (group["y_true"] == 1)).sum()
                    / max((group["y_true"] == 1).sum(), 1)
                ),
                "specificity": float(
                    ((group["y_pred"] == 0) & (group["y_true"] == 0)).sum()
                    / max((group["y_true"] == 0).sum(), 1)
                ),
                "n_test": int(len(group)),
            }
        )

    branch_b_model = (
        deep_model_factory.__name__
        if deep_model_factory is not None
        and hasattr(deep_model_factory, "__name__")
        else "deep_model"
    )

    return HybridResult(
        experiment="feature_fusion",
        branch_a_model="xgboost",
        branch_b_model=branch_b_model,
        accuracy=float(metrics["accuracy"]),
        precision=float(metrics["precision"]),
        recall=float(metrics["recall"]),
        f1=float(metrics["f1"]),
        roc_auc=float(metrics["auroc"]),
        pr_auc=float(metrics["auprc"]),
        sensitivity=float(metrics["sensitivity"]),
        specificity=float(metrics["specificity"]),
        brier=float(metrics["brier"]),
        ece=float(metrics["ece"]),
        n_subjects=int(len(np.unique(subject_probs.index))),
        n_folds=len(folds),
        n_repeats=1,
        participant_level=True,
        real_data=real_data,
        folds=fold_rows,
    )


def run_prediction_ensemble(
    bundle,
    folds: list[Fold],
    seed: int,
    branch_a_probs: dict[int, dict[str, float]],
    branch_a_val_probs: Optional[dict[int, dict[str, float]]] = None,
    branch_b_probs: Optional[dict[int, dict[str, float]]] = None,
    branch_b_val_probs: Optional[dict[int, dict[str, float]]] = None,
    method: str = "soft_voting",
    weights: Optional[list[float]] = None,
    deep_model_factory: Optional[Callable[[], torch.nn.Module]] = None,
    device: Optional[torch.device] = None,
    model_name: str = "cte_net",
    real_data: bool = True,
) -> HybridResult:
    """Branch D: prediction-level ensemble with leakage-free weight/threshold selection.

    branch_a_probs: {fold_index: {subject_id: prob}} for out-of-fold predictions.
    branch_b_probs: same for deep model, or None.

    method: "soft_voting", "weighted_voting", "stacking".

    For soft_voting and weighted_voting, pre-computed OOF probabilities are
    combined directly. Weights are derived from inner OOF predictions on
    outer-training participants only -- never from outer-test labels.

    For stacking, proper nested cross-validation is used:
      1. Outer training participants are split into inner subject-wise folds.
      2. Base models are trained on inner-train, predict on inner-val.
      3. OOF participant-level probabilities for every outer training participant
         are collected ([xgb_prob, deep_prob]).
      4. Meta-classifier is trained on these OOF probabilities.
      5. Base models are retrained on full outer-training participants.
      6. Meta-classifier predicts on outer-test base-model probabilities.

    The meta-classifier is never evaluated on the same predictions used to
    train it. Deep model early stopping uses a separate dev split carved from
    inner-train, never the inner-val subjects that contribute OOF features.
    """
    from sklearn.model_selection import StratifiedKFold

    subject_list = np.asarray(sorted(bundle.subject_labels), dtype=object)
    subject_labels_arr = np.asarray(
        [bundle.subject_labels[str(s)] for s in subject_list], dtype=int
    )
    index_of = {str(s): i for i, s in enumerate(subject_list)}
    n_subjects = len(subject_list)
    all_subject_labels = dict(bundle.subject_labels)

    # Track whether val probs were explicitly provided by the caller,
    # so we can distinguish "missing from caller" from "derived but empty".
    branch_a_val_explicit = branch_a_val_probs is not None
    branch_b_val_explicit = branch_b_val_probs is not None

    # Build Branch A features for stacking inner CV.
    feature_set = "node+adj+rel"
    from src.classical_models import (
        _build_epoch_features as _cla_build_features,
        make_model as _cla_make_model,
        _subject_probs as _cla_subject_probs,
    )
    from src.evaluate import youden_threshold as _cla_youden

    a_features = _cla_build_features(bundle, feature_set)

    def _derive_weights_from_inner_oof(
        outer_train_subj: list[str],
        n_inner: int = 3,
    ) -> Optional[list[float]]:
        """Derive branch weights from fresh inner OOF predictions.

        For each inner fold:
          - Train XGBoost on inner-train subjects only.
          - Train a fresh deep model on inner-train subjects only,
            with a disjoint dev split carved from inner-train for early stopping.
          - Predict on inner-held-out participants only.
        - Align predictions by participant ID.
        - Calculate weights from these genuine inner OOF predictions.

        Never reuses globally precomputed branch_b_probs, which were
        computed with a different fold structure and may leak test-fold
        information into weight derivation.
        """
        if deep_model_factory is None or device is None:
            return None

        inner_folds = _make_inner_folds(
            outer_train_subj, all_subject_labels, n_splits=n_inner, seed=seed
        )

        a_oof_p = []
        a_oof_y = []
        b_oof_p = []
        b_oof_y = []

        for inner_tr, inner_val in inner_folds:
            if len(inner_val) == 0:
                continue

            inner_tr_index = np.flatnonzero(
                np.isin(bundle.subject_ids, inner_tr)
            )
            inner_val_index = np.flatnonzero(
                np.isin(bundle.subject_ids, inner_val)
            )

            # Branch A: XGBoost on inner-train, predict inner-held-out.
            xgb_inner = _cla_make_model(
                "xgboost", seed, n_features=a_features.shape[1], k=150
            )
            xgb_inner.fit(
                a_features[inner_tr_index],
                bundle.labels[inner_tr_index].astype(int),
            )
            inner_val_prob = xgb_inner.predict_proba(
                a_features[inner_val_index]
            )[:, 1]
            inner_val_subject_probs = _cla_subject_probs(
                inner_val_prob, bundle.subject_ids[inner_val_index]
            )
            for sid, prob in inner_val_subject_probs.items():
                a_oof_p.append(float(prob))
                a_oof_y.append(all_subject_labels[str(sid)])

            # Branch B: fresh deep model on inner-train, predict inner-held-out.
            # Early stopping uses a disjoint dev split carved from inner-train.
            inner_deep_probs = _train_deep_oof(
                bundle,
                list(inner_tr),
                list(inner_val),
                seed,
                device,
                model_name,
                model_factory=deep_model_factory,
                epochs=10,
                batch_size=32,
                lr=1e-3,
                weight_decay=1e-4,
                patience=5,
            )
            for sid in inner_val:
                sid_str = str(sid)
                if sid_str not in inner_deep_probs:
                    raise RuntimeError(
                        f"E11 weight derivation: missing deep OOF prob for "
                        f"subject {sid_str} in inner fold"
                    )
                b_oof_p.append(float(inner_deep_probs[sid_str]))
                b_oof_y.append(all_subject_labels[sid_str])

        if not a_oof_p or not b_oof_p:
            return None

        a_auc = _safe_auc(np.asarray(a_oof_y), np.asarray(a_oof_p))
        b_auc = _safe_auc(np.asarray(b_oof_y), np.asarray(b_oof_p))

        a_w = max(a_auc - 0.5, 0.01)
        b_w = max(b_auc - 0.5, 0.01)
        total = a_w + b_w
        return [a_w / total, b_w / total]

    if method in {"soft_voting", "weighted_voting"}:
        # ----- Direct combination of pre-computed OOF probs -----
        a_oof = np.zeros(n_subjects, dtype=float)
        b_oof = (
            np.zeros(n_subjects, dtype=float)
            if branch_b_probs is not None
            else None
        )
        a_fold_of = np.zeros(n_subjects, dtype=int)

        for fold, fold_spec in enumerate(folds):
            test_subjects = list(fold_spec.test)
            fold_probs = branch_a_probs.get(fold, {})
            for s in test_subjects:
                sid_str = str(s)
                idx = index_of[sid_str]
                if sid_str not in fold_probs:
                    raise RuntimeError(
                        f"E10/E11: missing Branch A test probability for "
                        f"subject {sid_str} in fold {fold}"
                    )
                a_oof[idx] = float(fold_probs[sid_str])
                a_fold_of[idx] = fold
                if b_oof is not None and fold in (branch_b_probs or {}):
                    fb = (branch_b_probs or {})[fold]
                    if sid_str not in fb:
                        raise RuntimeError(
                            f"E10/E11: missing Branch B test probability for "
                            f"subject {sid_str} in fold {fold}"
                        )
                    b_oof[idx] = float(fb[sid_str])

        # ----- Derive per-fold weights (training data only) -----
        fold_weights: dict[int, list[float]] = {}
        if method == "weighted_voting":
            if weights is not None:
                # User-provided fixed weights (same for all folds).
                for fold in range(len(folds)):
                    fold_weights[fold] = list(weights)
            else:
                # Derive weights per fold from inner OOF on outer-train subjects.
                for fold, fold_spec in enumerate(folds):
                    outer_train = list(fold_spec.train) + list(fold_spec.val)
                    w = _derive_weights_from_inner_oof(outer_train)
                    fold_weights[fold] = w if w is not None else [1.0]

        # ----- Compute combined OOF probabilities for test subjects -----
        combined = np.zeros(n_subjects, dtype=float)
        if method == "soft_voting":
            if b_oof is not None:
                combined = 0.5 * a_oof + 0.5 * b_oof
            else:
                combined = a_oof
        else:  # weighted_voting
            combined = np.zeros(n_subjects, dtype=float)
            for fold, fold_spec in enumerate(folds):
                test_subjects = list(fold_spec.test)
                w = fold_weights.get(fold, [1.0])
                w_sum = sum(w)
                for s in test_subjects:
                    idx = index_of[str(s)]
                    a_p = float(a_oof[idx])
                    if b_oof is not None:
                        b_p = float(b_oof[idx])
                        # Handle single-weight case (no deep model).
                        if len(w) >= 2:
                            combined[idx] = (w[0] * a_p + w[1] * b_p) / w_sum
                        else:
                            combined[idx] = a_p
                    else:
                        combined[idx] = a_p

        # ----- Compute Branch B validation probs if not provided -----
        if branch_b_val_probs is None and deep_model_factory is not None and device is not None:
            branch_b_val_probs = {}
            for fold, fold_spec in enumerate(folds):
                train_subjects = list(fold_spec.train)
                val_subjects = list(fold_spec.val)
                val_deep_probs = _train_deep_oof(
                    bundle,
                    train_subjects,
                    val_subjects,
                    seed,
                    device,
                    model_name,
                    model_factory=deep_model_factory,
                    epochs=10,
                    batch_size=32,
                    lr=1e-3,
                    weight_decay=1e-4,
                    patience=5,
                )
                branch_b_val_probs[fold] = val_deep_probs

        # ----- Build val probs from test probs if val probs not provided -----
        # Each subject is a test subject in exactly one fold, so val subjects
        # for fold f can be looked up in branch_a_probs[branch_b_probs] of the
        # fold where they were test subjects.
        if branch_a_val_probs is None:
            branch_a_val_probs = {}
            test_fold_of_subject = {}
            for f, fspec in enumerate(folds):
                for s in fspec.test:
                    test_fold_of_subject[str(s)] = f
            for fold, fold_spec in enumerate(folds):
                val_probs: dict[str, float] = {}
                for s in fold_spec.val:
                    sid_str = str(s)
                    test_fold = test_fold_of_subject.get(sid_str)
                    if test_fold is not None and test_fold in (branch_a_probs or {}):
                        val_probs[sid_str] = float(
                            branch_a_probs[test_fold][sid_str]
                        )
                branch_a_val_probs[fold] = val_probs

        if branch_b_val_probs is None and branch_b_probs is not None:
            branch_b_val_probs = {}
            test_fold_of_subject = {}
            for f, fspec in enumerate(folds):
                for s in fspec.test:
                    test_fold_of_subject[str(s)] = f
            for fold, fold_spec in enumerate(folds):
                val_probs: dict[str, float] = {}
                for s in fold_spec.val:
                    sid_str = str(s)
                    test_fold = test_fold_of_subject.get(sid_str)
                    if test_fold is not None and test_fold in (branch_b_probs or {}):
                        fb = branch_b_probs[test_fold]
                        if sid_str in fb:
                            val_probs[sid_str] = float(fb[sid_str])
                branch_b_val_probs[fold] = val_probs

        # ----- Compute Branch A val probs on the fly for subjects never in test -----
        # Some val subjects may never appear as test subjects (e.g., with few folds),
        # so they can't be looked up from branch_a_probs. Train XGBoost on train
        # subjects and predict on val subjects for those missing entries.
        if branch_a_val_probs is not None:
            for fold, fold_spec in enumerate(folds):
                val_subjects = list(fold_spec.val)
                train_subjects = list(fold_spec.train)
                missing = [s for s in val_subjects if str(s) not in branch_a_val_probs.get(fold, {})]
                if missing:
                    train_index = np.flatnonzero(
                        np.isin(bundle.subject_ids, train_subjects)
                    )
                    val_index = np.flatnonzero(
                        np.isin(bundle.subject_ids, missing)
                    )
                    if len(train_index) == 0:
                        raise RuntimeError(
                            f"E10/E11: no training data available to derive "
                            f"Branch A validation probabilities for fold {fold}"
                        )
                    # Skip if all training labels are identical (XGBoost
                    # cannot fit a single-class logistic objective).
                    train_labels = bundle.labels[train_index].astype(int)
                    if len(np.unique(train_labels)) < 2:
                        constant = float(train_labels.mean())
                        if fold not in branch_a_val_probs:
                            branch_a_val_probs[fold] = {}
                        for sid in missing:
                            branch_a_val_probs[fold][str(sid)] = constant
                    else:
                        xgb_val = _cla_make_model(
                            "xgboost", seed, n_features=a_features.shape[1], k=150
                        )
                        xgb_val.fit(
                            a_features[train_index], train_labels
                        )
                        val_prob = xgb_val.predict_proba(
                            a_features[val_index]
                        )[:, 1]
                        val_map = _cla_subject_probs(
                            val_prob, bundle.subject_ids[val_index]
                        )
                        if fold not in branch_a_val_probs:
                            branch_a_val_probs[fold] = {}
                        for sid, p in val_map.items():
                            branch_a_val_probs[fold][str(sid)] = float(p)

        # ----- Compute Branch B val probs on the fly if deep factory available -----
        if branch_b_val_probs is not None and deep_model_factory is not None and device is not None:
            for fold, fold_spec in enumerate(folds):
                val_subjects = list(fold_spec.val)
                train_subjects = list(fold_spec.train)
                missing = [s for s in val_subjects if str(s) not in branch_b_val_probs.get(fold, {})]
                if missing:
                    val_deep_probs = _train_deep_oof(
                        bundle,
                        train_subjects,
                        missing,
                        seed,
                        device,
                        model_name,
                        model_factory=deep_model_factory,
                        epochs=10,
                        batch_size=32,
                        lr=1e-3,
                        weight_decay=1e-4,
                        patience=5,
                    )
                    if fold not in branch_b_val_probs:
                        branch_b_val_probs[fold] = {}
                    for sid in missing:
                        sid_str = str(sid)
                        if sid_str in val_deep_probs:
                            branch_b_val_probs[fold][sid_str] = float(
                                val_deep_probs[sid_str]
                            )
        # Validation thresholds are calculated from actual validation
        # predictions. Missing predictions cause an explicit error.
        fold_thresholds: dict[int, float] = {}
        final_y_pred = np.zeros(n_subjects, dtype=int)

        has_deep = branch_b_probs is not None

        for fold, fold_spec in enumerate(folds):
            val_subjects = list(fold_spec.val)
            test_subjects = list(fold_spec.test)

            val_a_map = branch_a_val_probs.get(fold, {}) if branch_a_val_probs else {}
            val_b_map = branch_b_val_probs.get(fold, {}) if branch_b_val_probs else {}

            val_a = []
            val_b = []
            val_y = []
            for s in val_subjects:
                sid_str = str(s)
                if sid_str not in val_a_map:
                    raise RuntimeError(
                        f"E10/E11: missing Branch A validation probability for "
                        f"subject {sid_str} in fold {fold}"
                    )
                val_a.append(float(val_a_map[sid_str]))
                val_y.append(all_subject_labels[sid_str])
                if has_deep:
                    if sid_str not in val_b_map:
                        # branch_b_val_probs was explicitly provided but
                        # incomplete: raise rather than silently use 0.5.
                        if branch_b_val_explicit:
                            raise RuntimeError(
                                f"E10/E11: missing Branch B validation probability for "
                                f"subject {sid_str} in fold {fold}"
                            )
                        # Not explicitly provided and no deep factory: use 0.5.
                        val_b.append(0.5)
                    else:
                        val_b.append(float(val_b_map[sid_str]))
                else:
                    val_b.append(0.5)

            if len(val_a) == 0:
                raise RuntimeError(
                    f"E10/E11: no validation predictions available for fold {fold}"
                )

            val_a_arr = np.asarray(val_a)
            val_b_arr = np.asarray(val_b)
            if method == "soft_voting":
                val_combined = 0.5 * val_a_arr + 0.5 * val_b_arr
            else:
                w = fold_weights.get(fold, [1.0])
                w_sum = sum(w)
                if len(w) >= 2:
                    val_combined = (w[0] * val_a_arr + w[1] * val_b_arr) / w_sum
                else:
                    val_combined = val_a_arr

            val_y_arr = np.asarray(val_y, dtype=int)
            fold_thresholds[fold] = float(
                _cla_youden(val_y_arr, val_combined)
            )

            # Apply threshold to test subjects.
            for s in test_subjects:
                idx = index_of[str(s)]
                c = float(combined[idx])
                final_y_pred[idx] = int(c >= fold_thresholds[fold])

        y_pred = final_y_pred

    elif method == "stacking":
        # ----- Probability-level nested stacking (no data leakage) -----
        # Meta-features: [xgb_participant_prob, deep_participant_prob].
        combined = np.zeros(n_subjects, dtype=float)
        a_fold_of = np.zeros(n_subjects, dtype=int)
        inner_n_splits = 3
        fold_thresholds: dict[int, float] = {}

        for outer_fold, fold_spec in enumerate(folds):
            outer_test = list(fold_spec.test)
            outer_train = list(fold_spec.train) + list(fold_spec.val)

            # Inner CV on outer training participants.
            outer_train_labels = np.asarray(
                [bundle.subject_labels[str(s)] for s in outer_train], dtype=int
            )
            outer_train_arr = np.asarray(outer_train, dtype=object)

            inner_cv = StratifiedKFold(
                n_splits=inner_n_splits, shuffle=True, random_state=seed
            )

            # OOF participant-level probabilities for outer training participants.
            # Also track which inner fold produced each subject's OOF prediction
            # for cross-fitted meta threshold selection.
            xgb_oof_train = np.zeros(len(outer_train), dtype=float)
            deep_oof_train = None
            if deep_model_factory is not None and device is not None:
                deep_oof_train = np.zeros(len(outer_train), dtype=float)
            inner_fold_of_subject = np.full(len(outer_train), -1, dtype=int)

            for inner_fold_idx, (inner_tr_idx, inner_val_idx) in enumerate(
                inner_cv.split(outer_train_arr, outer_train_labels)
            ):
                inner_tr_subjects = outer_train_arr[inner_tr_idx].tolist()
                inner_val_subjects = outer_train_arr[inner_val_idx].tolist()

                inner_tr_index = np.flatnonzero(
                    np.isin(bundle.subject_ids, inner_tr_subjects)
                )
                inner_val_index = np.flatnonzero(
                    np.isin(bundle.subject_ids, inner_val_subjects)
                )

                # Train XGBoost on inner-train.
                xgb_inner = _cla_make_model(
                    "xgboost", seed, n_features=a_features.shape[1], k=150
                )
                xgb_inner.fit(
                    a_features[inner_tr_index],
                    bundle.labels[inner_tr_index].astype(int),
                )
                inner_val_prob = xgb_inner.predict_proba(
                    a_features[inner_val_index]
                )[:, 1]
                inner_val_subject_probs = _cla_subject_probs(
                    inner_val_prob, bundle.subject_ids[inner_val_index]
                )
                for sid, prob in inner_val_subject_probs.items():
                    pos = int(np.where(outer_train_arr == str(sid))[0][0])
                    xgb_oof_train[pos] = float(prob)
                    inner_fold_of_subject[pos] = inner_fold_idx

                # Deep model OOF: participant-level probabilities for inner-val.
                # Early stopping uses a dev split carved from inner-train,
                # NOT inner-val subjects.
                if deep_model_factory is not None and device is not None:
                    inner_deep_probs = _train_deep_oof(
                        bundle,
                        list(inner_tr_subjects),
                        list(inner_val_subjects),
                        seed,
                        device,
                        model_name,
                        model_factory=deep_model_factory,
                        epochs=30,
                        batch_size=32,
                        lr=1e-3,
                        weight_decay=1e-4,
                        patience=10,
                    )
                    for i, sid in enumerate(inner_val_subjects):
                        pos = int(np.where(outer_train_arr == str(sid))[0][0])
                        deep_oof_train[pos] = float(
                            inner_deep_probs[str(sid)]
                        )

            # Train meta-classifier on inner OOF probabilities (train subjects only).
            if deep_oof_train is not None:
                meta_X = np.stack([xgb_oof_train, deep_oof_train], axis=1)
            else:
                meta_X = xgb_oof_train.reshape(-1, 1)
            meta_y = outer_train_labels

            # Verify every outer-train subject has an OOF prediction.
            missing_oof = inner_fold_of_subject < 0
            if np.any(missing_oof):
                missing_sids = outer_train_arr[missing_oof].tolist()
                raise RuntimeError(
                    f"E12 stacking: missing OOF predictions for outer-train "
                    f"subjects: {missing_sids}"
                )

            meta = LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=seed
            )
            meta.fit(meta_X, meta_y)

            # Cross-fitted meta predictions for threshold selection.
            # For each inner fold, retrain meta on OOF from the other folds
            # and predict on this fold's subjects. This gives truly
            # out-of-sample meta predictions (no in-sample threshold tuning).
            cross_fitted_meta_prob = np.zeros(len(outer_train), dtype=float)
            for cf_fold_idx in range(inner_n_splits):
                val_mask = inner_fold_of_subject == cf_fold_idx
                if not np.any(val_mask):
                    continue
                train_mask = inner_fold_of_subject != cf_fold_idx
                if not np.any(train_mask):
                    continue
                meta_cf = LogisticRegression(
                    max_iter=3000, class_weight="balanced", random_state=seed
                )
                meta_cf.fit(meta_X[train_mask], meta_y[train_mask])
                cross_fitted_meta_prob[val_mask] = meta_cf.predict_proba(
                    meta_X[val_mask]
                )[:, 1]

            fold_thresholds[outer_fold] = float(
                _cla_youden(meta_y, cross_fitted_meta_prob)
            )

            # Retrain base models on full outer training participants.
            outer_tr_index = np.flatnonzero(
                np.isin(bundle.subject_ids, outer_train)
            )
            xgb_full = _cla_make_model(
                "xgboost", seed, n_features=a_features.shape[1], k=150
            )
            xgb_full.fit(
                a_features[outer_tr_index],
                bundle.labels[outer_tr_index].astype(int),
            )

            # Retrain deep model on full outer training participants.
            deep_full = None
            if deep_model_factory is not None and device is not None:
                deep_full = _train_deep_model_on_subjects(
                    bundle,
                    outer_train,
                    [],
                    seed,
                    device,
                    model_name,
                    model_factory=deep_model_factory,
                    epochs=30,
                    batch_size=32,
                    lr=1e-3,
                    weight_decay=1e-4,
                    patience=10,
                )

            # Predict on outer test participants.
            outer_test_index = np.flatnonzero(
                np.isin(bundle.subject_ids, outer_test)
            )
            xgb_test_prob = xgb_full.predict_proba(
                a_features[outer_test_index]
            )[:, 1]
            xgb_test_subject_probs = _cla_subject_probs(
                xgb_test_prob, bundle.subject_ids[outer_test_index]
            )

            deep_test_probs = None
            if deep_full is not None:
                outer_test_subject_arr = np.asarray(outer_test, dtype=object)
                deep_test_arr = _deep_predict_proba(
                    deep_full,
                    bundle,
                    outer_test_subject_arr,
                    device,
                    batch_size=32,
                    model_name=model_name,
                )
                deep_test_probs = {
                    str(s): float(p)
                    for s, p in zip(outer_test, deep_test_arr)
                }

            # Apply meta-classifier to outer test probabilities.
            for i, sid in enumerate(outer_test):
                sid_str = str(sid)
                x_p = float(xgb_test_subject_probs.get(sid_str, 0.5))
                if deep_test_probs is not None:
                    d_p = float(deep_test_probs[sid_str])
                    meta_pred = float(
                        meta.predict_proba([[x_p, d_p]])[:, 1][0]
                    )
                else:
                    meta_pred = float(meta.predict_proba([[x_p]])[:, 1][0])
                idx = index_of[sid_str]
                combined[idx] = meta_pred
                a_fold_of[idx] = outer_fold

        y_pred = np.asarray(
            [
                int(combined[index_of[str(s)]] >= fold_thresholds[a_fold_of[index_of[str(s)]]])
                for s in subject_list
            ],
            dtype=int,
        )

    else:
        raise ValueError(f"Unknown ensemble method: {method}")


    metrics = binary_metrics(
        subject_labels_arr, combined, y_pred=y_pred, n_boot=2000, seed=seed
    )

    fold_rows = []
    for fold_id in range(len(folds)):
        mask = a_fold_of == fold_id
        if not np.any(mask):
            continue
        fold_y = subject_labels_arr[mask]
        fold_p = combined[mask]
        fold_pred = y_pred[mask]
        fold_rows.append(
            {
                "fold": int(fold_id),
                "auroc": float(_safe_auc(fold_y, fold_p)),
                "accuracy": float((fold_pred == fold_y).mean()),
                "sensitivity": float(
                    ((fold_pred == 1) & (fold_y == 1)).sum()
                    / max((fold_y == 1).sum(), 1)
                ),
                "specificity": float(
                    ((fold_pred == 0) & (fold_y == 0)).sum()
                    / max((fold_y == 0).sum(), 1)
                ),
                "n_test": int(mask.sum()),
            }
        )

    return HybridResult(
        experiment=method,
        branch_a_model="xgboost",
        branch_b_model="ctenet" if branch_b_probs is not None else "none",
        accuracy=float(metrics["accuracy"]),
        precision=float(metrics["precision"]),
        recall=float(metrics["recall"]),
        f1=float(metrics["f1"]),
        roc_auc=float(metrics["auroc"]),
        pr_auc=float(metrics["auprc"]),
        sensitivity=float(metrics["sensitivity"]),
        specificity=float(metrics["specificity"]),
        brier=float(metrics["brier"]),
        ece=float(metrics["ece"]),
        n_subjects=int(n_subjects),
        n_folds=len(folds),
        n_repeats=1,
        participant_level=True,
        real_data=real_data,
        folds=fold_rows,
    )


def _safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def save_hybrid_result(result: HybridResult, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = f"hybrid_{result.experiment}"
    import json
    metadata = {
        "experiment": result.experiment,
        "branch_a_model": result.branch_a_model,
        "branch_b_model": result.branch_b_model,
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "roc_auc": result.roc_auc,
        "pr_auc": result.pr_auc,
        "sensitivity": result.sensitivity,
        "specificity": result.specificity,
        "brier": result.brier,
        "ece": result.ece,
        "n_subjects": result.n_subjects,
        "n_folds": result.n_folds,
        "participant_level": result.participant_level,
        "real_data": result.real_data,
        "folds": result.folds,
        "channels": CHANNELS,
    }
    (output_dir / f"{safe}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
