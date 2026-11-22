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
    """Branch C: feature-level fusion.

    Per fold:
      1. Train Branch A (XGBoost on Paper 1 features) on train subjects;
         threshold on val subjects.
      2. Train Branch B (deep model) on train subjects only, using val for
         early stopping and best-checkpoint selection.
      3. Extract learned deep representations for train, val, and test
         subjects from the trained model (never zero-filled).
      4. Concat [Branch-A probability, deep features] -> fused feature vector.
      5. Train Logistic Regression meta-classifier on fused train features.
      6. Refit on train+val; predict held-out test subjects.

    Meta-classifier is trained on training-fold outputs only -- never on
    test labels. Deep model is never trained on test subjects.
    """
    from collections.abc import Callable
    from torch.utils.data import DataLoader

    from src.train import (
        EpochDataset,
        aggregate_subject_probs,
        build_model as _build_deep_model,
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
    subject_labels_arr = np.asarray(
        [bundle.subject_labels[str(s)] for s in subject_list], dtype=int
    )

    probs = np.zeros(len(bundle.labels), dtype=float)
    fold_thresholds: dict[int, float] = {}
    fold_of_subject: dict[str, int] = {}
    fold_rows: list[dict] = []

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

        # ----- Branch A: XGBoost -----
        branch_a = make_model("xgboost", seed, n_features=a_features.shape[1], k=150)
        branch_a.fit(
            a_features[train_index], bundle.labels[train_index].astype(int)
        )

        val_a_prob = branch_a.predict_proba(a_features[val_index])[:, 1]
        val_a_subject = _subject_probs(val_a_prob, bundle.subject_ids[val_index])
        val_y = np.asarray(
            [bundle.subject_labels[str(s)] for s in val_a_subject.index], dtype=int
        )
        threshold = float(youden_threshold(val_y, val_a_subject.to_numpy()))
        fold_thresholds[fold] = threshold

        branch_a.fit(
            a_features[full_index], bundle.labels[full_index].astype(int)
        )

        # ----- Branch B: Train deep model on train subjects only -----
        train_subject_arr = np.asarray(train_subjects)
        val_subject_arr = np.asarray(val_subjects)
        test_subject_arr = np.asarray(test_subjects)

        deep_train: Optional[np.ndarray] = None
        deep_val: Optional[np.ndarray] = None
        deep_test: Optional[np.ndarray] = None

        if deep_model_factory is not None:
            # Build and train a fresh deep model for this fold.
            fold_deep_model = deep_model_factory()
            fold_deep_model = fold_deep_model.to(device)

            # Prepare data loaders for this fold.
            train_indices = indices_for_subjects(bundle, train_subjects)
            val_indices = indices_for_subjects(bundle, val_subjects)
            test_indices = indices_for_subjects(bundle, test_subjects)

            train_dataset = EpochDataset(bundle, train_indices, model_name)
            val_dataset = EpochDataset(bundle, val_indices, model_name)
            test_dataset = EpochDataset(bundle, test_indices, model_name)

            from torch.utils.data import DataLoader as _DataLoader

            train_loader = _DataLoader(
                train_dataset,
                batch_size=32,
                shuffle=True,
                num_workers=0,
                drop_last=False,
            )
            val_loader = _DataLoader(
                val_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=0,
            )
            test_loader = _DataLoader(
                test_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=0,
            )

            # Train with early stopping on validation subjects.
            train_labels = bundle.labels[train_indices]
            n_pos = float(train_labels.sum())
            n_neg = float(len(train_labels) - n_pos)
            pos_weight = torch.tensor(
                [n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = torch.optim.AdamW(
                fold_deep_model.parameters(), lr=1e-3, weight_decay=1e-4
            )

            best_auc = -float("inf")
            best_state = None
            bad_epochs = 0

            for epoch in range(1, 31):  # 30 epochs for hybrid experiments
                fold_deep_model.train()
                running_loss = 0.0
                for batch in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    if model_name == "eegnet" or model_name == "ctenet_original":
                        feats, labels_batch = batch
                        logits = fold_deep_model(feats.to(device))
                    else:
                        feats, adj, labels_batch = batch
                        logits = fold_deep_model(
                            feats.to(device), adj.to(device)
                        )
                    loss = criterion(logits, labels_batch.to(device))
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        fold_deep_model.parameters(), max_norm=1.0
                    )
                    optimizer.step()
                    running_loss += float(loss.item()) * len(labels_batch)

                # Validation.
                fold_deep_model.eval()
                val_probs_list = []
                val_subject_list = []
                with torch.no_grad():
                    for batch in val_loader:
                        if model_name == "eegnet" or model_name == "ctenet_original":
                            feats, labels_batch = batch
                            logits = fold_deep_model(feats.to(device))
                        else:
                            feats, adj, labels_batch = batch
                            logits = fold_deep_model(
                                feats.to(device), adj.to(device)
                            )
                        val_probs_list.append(
                            torch.sigmoid(logits).cpu().numpy()
                        )
                        val_subject_list.extend(
                            bundle.subject_ids[val_dataset.indices[
                                len(val_subject_list) :
                                len(val_subject_list) + len(logits)
                            ]].tolist()
                        )
                if val_probs_list:
                    val_epoch_probs = np.concatenate(val_probs_list)
                    val_subject_probs = aggregate_subject_probs(
                        np.asarray(val_subject_list), val_epoch_probs
                    )
                    val_y_arr = np.asarray(
                        [
                            bundle.subject_labels[str(s)]
                            for s in val_subject_probs.index
                        ],
                        dtype=int,
                    )
                    val_auc = _safe_auc(val_y_arr, val_subject_probs.to_numpy())
                else:
                    val_auc = float("nan")

                if np.isfinite(val_auc) and val_auc > best_auc + 1e-4:
                    best_auc = val_auc
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in fold_deep_model.state_dict().items()
                    }
                    bad_epochs = 0
                else:
                    bad_epochs += 1

                if bad_epochs >= 10:
                    break

            if best_state is None:
                raise RuntimeError(
                    f"Fold {fold}: deep model failed to produce a checkpoint"
                )
            fold_deep_model.load_state_dict(best_state)

            # Extract deep representations from the trained model.
            deep_train = _deep_features(
                fold_deep_model,
                bundle,
                train_subject_arr,
                device,
                model_name=model_name,
                raise_on_missing=True,
            )
            deep_val = _deep_features(
                fold_deep_model,
                bundle,
                val_subject_arr,
                device,
                model_name=model_name,
                raise_on_missing=True,
            )
            deep_test = _deep_features(
                fold_deep_model,
                bundle,
                test_subject_arr,
                device,
                model_name=model_name,
                raise_on_missing=True,
            )

            # Verify no zero rows (all features extracted from trained model).
            for name, arr in [
                ("train", deep_train),
                ("val", deep_val),
                ("test", deep_test),
            ]:
                if arr is not None and np.all(arr == 0):
                    raise ValueError(
                        f"Fold {fold}: deep {name} features are all zeros -- "
                        "model not producing learned representations."
                    )

        # ----- Branch A probabilities per subject -----
        train_a_prob = branch_a.predict_proba(a_features[train_index])[:, 1]
        train_a_map = _subject_probs(train_a_prob, bundle.subject_ids[train_index])
        val_a_prob = branch_a.predict_proba(a_features[val_index])[:, 1]
        val_a_map = _subject_probs(val_a_prob, bundle.subject_ids[val_index])

        d_model = deep_train.shape[1] if deep_train is not None else 64

        def _fuse(subject_arr, deep_vecs, a_prob_map):
            rows = []
            for s in subject_arr:
                sid = str(s)
                a_p = float(a_prob_map.get(sid, 0.5))
                if deep_vecs is not None:
                    d_idx = int(np.where(subject_arr == s)[0][0])
                    d_vec = deep_vecs[d_idx]
                else:
                    raise ValueError(
                        "Deep features are None but deep_model_factory was "
                        "provided. Feature fusion requires learned deep "
                        "representations."
                    )
                rows.append(np.concatenate([[a_p], d_vec]))
            return np.stack(rows)

        X_train_fused = _fuse(train_subject_arr, deep_train, train_a_map)
        X_val_fused = _fuse(val_subject_arr, deep_val, val_a_map)

        y_train = np.asarray(
            [bundle.subject_labels[str(s)] for s in train_subject_arr], dtype=int
        )
        y_val = np.asarray(
            [bundle.subject_labels[str(s)] for s in val_subject_arr], dtype=int
        )

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
        meta.fit(X_train_fused, y_train)

        val_meta_prob = meta.predict_proba(X_val_fused)[:, 1]
        val_meta_threshold = float(youden_threshold(y_val, val_meta_prob))

        # Refit meta on train+val.
        all_subjects = np.asarray(train_subjects + val_subjects)
        if deep_train is not None and deep_val is not None:
            deep_full = np.vstack([deep_train, deep_val])
        else:
            deep_full = None
        a_map_full = {
            **{str(s): float(train_a_map[str(s)]) for s in train_subject_arr},
            **{str(s): float(val_a_map[str(s)]) for s in val_subject_arr},
        }
        X_full_fused = _fuse(all_subjects, deep_full, a_map_full)
        y_full = np.asarray(
            [bundle.subject_labels[str(s)] for s in all_subjects], dtype=int
        )
        meta.fit(X_full_fused, y_full)

        # ----- Predict on test -----
        test_a_prob = branch_a.predict_proba(a_features[test_index])[:, 1]
        test_a_map = _subject_probs(test_a_prob, bundle.subject_ids[test_index])

        X_test_fused = _fuse(test_subject_arr, deep_test, test_a_map)
        test_prob = meta.predict_proba(X_test_fused)[:, 1]

        # Participant-level: one probability per test subject.
        for i, subject in enumerate(test_subject_arr):
            probs[
                np.flatnonzero(bundle.subject_ids == subject)
            ] = test_prob[i]
            fold_of_subject[str(subject)] = fold

    subject_probs = _subject_probs(probs, bundle.subject_ids)
    subject_y = np.asarray(
        [bundle.subject_labels[str(s)] for s in subject_probs.index], dtype=int
    )
    y_pred = np.asarray(
        [
            int(
                subject_probs[subject]
                >= fold_thresholds[fold_of_subject[str(subject)]]
            )
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
    branch_b_probs: Optional[dict[int, dict[str, float]]] = None,
    method: str = "soft_voting",
    weights: Optional[list[float]] = None,
    real_data: bool = True,
) -> HybridResult:
    """Branch D: prediction-level ensemble.

    branch_a_probs: {fold_index: {subject_id: prob}} for out-of-fold predictions.
    branch_b_probs: same for deep model, or None.

    method: "soft_voting", "weighted_voting", "stacking".

    For soft_voting and weighted_voting, pre-computed OOF probabilities are
    combined directly.

    For stacking, proper nested cross-validation is used:
      1. Outer training participants are split into inner subject-wise folds.
      2. Base models are trained on inner-train, predict on inner-val.
      3. OOF predictions for every outer training participant are collected.
      4. Meta-classifier is trained on these OOF predictions.
      5. Base models are retrained on full outer-training participants.
      6. Meta-classifier predicts on outer-test base-model predictions.

    The meta-classifier is never evaluated on the same predictions used to
    train it.
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split

    subject_list = np.asarray(sorted(bundle.subject_labels), dtype=object)
    subject_labels_arr = np.asarray(
        [bundle.subject_labels[str(s)] for s in subject_list], dtype=int
    )
    index_of = {str(s): i for i, s in enumerate(subject_list)}
    n_subjects = len(subject_list)

    # Build Branch A features for stacking inner CV.
    feature_set = "node+adj+rel"
    from src.classical_models import (
        _build_epoch_features as _cla_build_features,
        make_model as _cla_make_model,
        _subject_probs as _cla_subject_probs,
    )
    from src.evaluate import youden_threshold as _cla_youden

    a_features = _cla_build_features(bundle, feature_set)

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
                idx = index_of[str(s)]
                a_oof[idx] = float(fold_probs.get(str(s), 0.5))
                a_fold_of[idx] = fold
                if (
                    b_oof is not None
                    and fold in (branch_b_probs or {})
                ):
                    b_oof[idx] = float(
                        (branch_b_probs or {})[fold].get(str(s), 0.5)
                    )

        if method == "soft_voting":
            if b_oof is not None:
                combined = 0.5 * a_oof + 0.5 * b_oof
            else:
                combined = a_oof
        else:  # weighted_voting
            if weights is None:
                weights = [0.5, 0.5] if b_oof is not None else [1.0]
            if b_oof is not None:
                combined = (
                    weights[0] * a_oof + weights[1] * b_oof
                ) / (weights[0] + weights[1])
            else:
                combined = a_oof

        y_pred = (combined >= 0.5).astype(int)

    elif method == "stacking":
        # ----- Proper nested stacking (no data leakage) -----
        combined = np.zeros(n_subjects, dtype=float)
        a_fold_of = np.zeros(n_subjects, dtype=int)
        inner_n_splits = 3

        from sklearn.model_selection import StratifiedKFold

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

            # OOF predictions for outer training participants.
            xgb_oof_train = np.zeros(len(outer_train), dtype=float)
            deep_oof_train = (
                np.zeros(len(outer_train), dtype=float)
                if branch_b_probs is not None
                else None
            )

            for inner_tr_idx, inner_val_idx in inner_cv.split(
                outer_train_arr, outer_train_labels
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

                # Deep model OOF: use pre-computed branch_b_probs.
                if deep_oof_train is not None and branch_b_probs is not None:
                    for sid in inner_val_subjects:
                        for b_fold, b_fold_spec in enumerate(folds):
                            if str(sid) in map(str, b_fold_spec.test):
                                pos = int(
                                    np.where(outer_train_arr == str(sid))[0][0]
                                )
                                deep_oof_train[pos] = float(
                                    branch_b_probs[b_fold].get(str(sid), 0.5)
                                )
                                break

            # Train meta-classifier on inner OOF predictions (train subjects only).
            if deep_oof_train is not None:
                meta_X = np.stack([xgb_oof_train, deep_oof_train], axis=1)
            else:
                meta_X = xgb_oof_train.reshape(-1, 1)
            meta_y = outer_train_labels

            meta = LogisticRegression(
                max_iter=3000, class_weight="balanced", random_state=seed
            )
            meta.fit(meta_X, meta_y)

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

            deep_test_probs: dict[str, float] = {}
            if branch_b_probs is not None:
                for sid in outer_test:
                    for b_fold, b_fold_spec in enumerate(folds):
                        if str(sid) in map(str, b_fold_spec.test):
                            deep_test_probs[str(sid)] = float(
                                branch_b_probs[b_fold].get(str(sid), 0.5)
                            )
                            break

            # Apply meta-classifier to outer test predictions.
            for sid in outer_test:
                sid_str = str(sid)
                x_p = float(xgb_test_subject_probs.get(sid_str, 0.5))
                if deep_oof_train is not None:
                    d_p = float(deep_test_probs.get(sid_str, 0.5))
                    meta_pred = float(
                        meta.predict_proba([[x_p, d_p]])[:, 1][0]
                    )
                else:
                    meta_pred = float(meta.predict_proba([[x_p]])[:, 1][0])
                idx = index_of[sid_str]
                combined[idx] = meta_pred
                a_fold_of[idx] = outer_fold

        y_pred = (combined >= 0.5).astype(int)

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
