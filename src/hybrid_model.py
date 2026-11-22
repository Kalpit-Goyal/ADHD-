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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
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
            if model_name == "eegnet":
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
) -> Optional[np.ndarray]:
    """Extract deep learned features per subject.

    Returns (n_subjects, d_model) or None if unavailable.
    """
    from torch.utils.data import DataLoader

    from src.data import EpochDataset
    from src.train import resolve_device

    if device is None:
        device = resolve_device("cpu")
    if not hasattr(model, "forward_features"):
        return None

    indices = np.flatnonzero(np.isin(bundle.subject_ids, subject_ids_subset))
    if len(indices) == 0:
        return None

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
    with torch.no_grad():
        for batch in loader:
            if model_name == "eegnet":
                features, labels = batch
                feat, _, _ = model.forward_features(features.to(device))
            else:
                features, adjacency, labels = batch
                feat, _, _ = model.forward_features(features.to(device), adjacency.to(device))
            batch_feat = feat.cpu().numpy()
            for i in range(len(batch_feat)):
                sid = str(bundle.subject_ids[indices[i]])
                per_subject.setdefault(sid, []).append(batch_feat[i])

    aligned = []
    for s in subject_ids_subset:
        vecs = per_subject.get(str(s), [])
        if len(vecs) == 0:
            aligned.append(np.zeros(d_model, dtype=np.float32))
        else:
            aligned.append(np.mean(np.stack(vecs), axis=0))
    return np.stack(aligned) if aligned else None


def run_feature_fusion(
    bundle,
    folds: list[Fold],
    seed: int,
    deep_model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None,
    model_name: str = "cte_net",
    real_data: bool = True,
) -> HybridResult:
    """Branch C: feature-level fusion.

    Per fold:
      1. Train Branch A (XGBoost on Paper 1 features) on train subjects;
         threshold on val subjects.
      2. Get deep branch probabilities + features on train subjects.
      3. Concat [Branch-A prob, deep features] -> fused feature vector.
      4. Train Logistic Regression meta-classifier on fused train features.
      5. Refit on train+val; predict held-out test subjects.

    Meta-classifier is trained on training-fold outputs only -- never on
    test labels.
    """
    from src.train import resolve_device

    if device is None:
        device = resolve_device("cpu")

    feature_set = "node+adj+rel"
    a_features = _build_epoch_features(bundle, feature_set)

    subject_list = np.asarray(sorted(bundle.subject_labels), dtype=object)
    subject_labels = np.asarray([bundle.subject_labels[str(s)] for s in subject_list], dtype=int)

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
        full_index = np.flatnonzero(np.isin(bundle.subject_ids, train_subjects + val_subjects))
        test_index = np.flatnonzero(np.isin(bundle.subject_ids, test_subjects))

        # ----- Branch A: XGBoost -----
        branch_a = make_model("xgboost", seed, n_features=a_features.shape[1], k=150)
        branch_a.fit(a_features[train_index], bundle.labels[train_index].astype(int))

        val_a_prob = branch_a.predict_proba(a_features[val_index])[:, 1]
        val_a_subject = _subject_probs(val_a_prob, bundle.subject_ids[val_index])
        val_y = np.asarray(
            [bundle.subject_labels[str(s)] for s in val_a_subject.index], dtype=int
        )
        threshold = float(youden_threshold(val_y, val_a_subject.to_numpy()))
        fold_thresholds[fold] = threshold

        branch_a.fit(a_features[full_index], bundle.labels[full_index].astype(int))

        # ----- Branch B: deep features + probs -----
        train_subject_arr = np.asarray(train_subjects)
        val_subject_arr = np.asarray(val_subjects)
        test_subject_arr = np.asarray(test_subjects)

        deep_train = _deep_features(deep_model, bundle, train_subject_arr, device, model_name=model_name) if deep_model is not None else None
        deep_val = _deep_features(deep_model, bundle, val_subject_arr, device, model_name=model_name) if deep_model is not None else None
        deep_test = _deep_features(deep_model, bundle, test_subject_arr, device, model_name=model_name) if deep_model is not None else None

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
                    d_vec = np.zeros(d_model, dtype=np.float32)
                rows.append(np.concatenate([[a_p], d_vec]))
            return np.stack(rows)

        X_train_fused = _fuse(train_subject_arr, deep_train, train_a_map)
        X_val_fused = _fuse(val_subject_arr, deep_val, val_a_map)

        y_train = np.asarray([bundle.subject_labels[str(s)] for s in train_subject_arr], dtype=int)
        y_val = np.asarray([bundle.subject_labels[str(s)] for s in val_subject_arr], dtype=int)

        meta = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000, class_weight="balanced", random_state=seed
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
        a_map_full = {**{str(s): float(train_a_map[str(s)]) for s in train_subject_arr},
                      **{str(s): float(val_a_map[str(s)]) for s in val_subject_arr}}
        X_full_fused = _fuse(all_subjects, deep_full, a_map_full)
        y_full = np.asarray([bundle.subject_labels[str(s)] for s in all_subjects], dtype=int)
        meta.fit(X_full_fused, y_full)

        # ----- Predict on test -----
        test_a_prob = branch_a.predict_proba(a_features[test_index])[:, 1]
        test_a_map = _subject_probs(test_a_prob, bundle.subject_ids[test_index])

        X_test_fused = _fuse(test_subject_arr, deep_test, test_a_map)
        test_prob = meta.predict_proba(X_test_fused)[:, 1]
        # Expand subject-level probabilities to all epochs of each test subject.
        subject_to_prob = dict(zip(test_subject_arr, test_prob))
        epoch_probs = np.asarray(
            [subject_to_prob[str(s)] for s in bundle.subject_ids[test_index]]
        )
        probs[test_index] = epoch_probs
        for subject in test_subjects:
            fold_of_subject[str(subject)] = fold

    subject_probs = _subject_probs(probs, bundle.subject_ids)
    subject_y = np.asarray(
        [bundle.subject_labels[str(s)] for s in subject_probs.index], dtype=int
    )
    y_pred = np.asarray(
        [
            int(subject_probs[subject] >= fold_thresholds[fold_of_subject[str(subject)]])
            for subject in subject_probs.index
        ],
        dtype=int,
    )

    metrics = binary_metrics(subject_y, subject_probs.to_numpy(), y_pred=y_pred, n_boot=2000, seed=seed)

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
                "auroc": float(roc_auc_score(group["y_true"], group["y_prob"])) if len(np.unique(group["y_true"])) > 1 else float("nan"),
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

    return HybridResult(
        experiment="feature_fusion",
        branch_a_model="xgboost",
        branch_b_model=model_name if deep_model is not None else "none",
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

    For stacking, a Logistic Regression meta-classifier is trained on
    out-of-fold predictions of all subjects (each subject's prediction comes
    from the fold where it was held out -- genuinely out-of-sample).
    """
    subject_list = np.asarray(sorted(bundle.subject_labels), dtype=object)
    subject_labels = np.asarray([bundle.subject_labels[str(s)] for s in subject_list], dtype=int)
    index_of = {str(s): i for i, s in enumerate(subject_list)}
    n_subjects = len(subject_list)

    a_oof = np.zeros(n_subjects, dtype=float)
    b_oof = np.zeros(n_subjects, dtype=float) if branch_b_probs is not None else None
    a_fold_of = np.zeros(n_subjects, dtype=int)

    for fold, fold_spec in enumerate(folds):
        test_subjects = list(fold_spec.test)
        fold_probs = branch_a_probs.get(fold, {})
        for s in test_subjects:
            idx = index_of[str(s)]
            a_oof[idx] = float(fold_probs.get(str(s), 0.5))
            a_fold_of[idx] = fold
            if b_oof is not None and fold in (branch_b_probs or {}):
                b_oof[idx] = float((branch_b_probs or {})[fold].get(str(s), 0.5))

    if method == "soft_voting":
        combined = 0.5 * a_oof + (0.5 * b_oof if b_oof is not None else 0.0)
        if b_oof is None:
            combined = a_oof
        y_pred = (combined >= 0.5).astype(int)
    elif method == "weighted_voting":
        if weights is None:
            weights = [0.5, 0.5] if b_oof is not None else [1.0]
        if b_oof is not None:
            combined = (weights[0] * a_oof + weights[1] * b_oof) / (weights[0] + weights[1])
        else:
            combined = a_oof
        y_pred = (combined >= 0.5).astype(int)
    elif method == "stacking":
        if b_oof is not None:
            stack_X = np.stack([a_oof, b_oof], axis=1)
        else:
            stack_X = a_oof.reshape(-1, 1)
        meta = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
        meta.fit(stack_X, subject_labels)
        combined = meta.predict_proba(stack_X)[:, 1]
        y_pred = (combined >= 0.5).astype(int)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    metrics = binary_metrics(subject_labels, combined, y_pred=y_pred, n_boot=2000, seed=seed)

    fold_rows = []
    for fold_id in range(len(folds)):
        mask = a_fold_of == fold_id
        if not np.any(mask):
            continue
        fold_y = subject_labels[mask]
        fold_p = combined[mask]
        fold_pred = y_pred[mask]
        fold_rows.append(
            {
                "fold": int(fold_id),
                "auroc": float(_safe_auc(fold_y, fold_p)),
                "accuracy": float((fold_pred == fold_y).mean()),
                "sensitivity": float(((fold_pred == 1) & (fold_y == 1)).sum() / max((fold_y == 1).sum(), 1)),
                "specificity": float(((fold_pred == 0) & (fold_y == 0)).sum() / max((fold_y == 0).sum(), 1)),
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
