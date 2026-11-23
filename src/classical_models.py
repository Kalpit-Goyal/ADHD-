"""Classical ML models for Paper 1-style EEG feature classification.

Implements XGBoost, RBF-SVM, Random Forest, Extra Trees, Logistic Regression,
and HistGradientBoosting with subject-wise cross-validation. Feature selection
(SelectKBest with mutual information or ANOVA F-score) is fitted independently
inside each training fold -- never on the full dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier

    _HAS_XGBOOST = True
except Exception:  # pragma: no cover - xgboost optional
    XGBClassifier = None  # type: ignore[assignment,misc]
    _HAS_XGBOOST = False

from src.cv import Fold, assert_no_subject_overlap, load_folds, save_folds
from src.evaluate import binary_metrics, youden_threshold


CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]


@dataclass
class ClassicalResult:
    model_name: str
    feature_set: str
    n_features: int
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
    feature_names: list[str] = field(default_factory=list)
    selected_feature_indices: list[int] = field(default_factory=list)


def make_model(
    name: str,
    seed: int,
    n_features: int | None = None,
    k: int | None = None,
    selector_score: str = "f_classif",
) -> object:
    """Build a classical model pipeline with optional internal feature selection.

    When ``k`` is given, a ``SelectKBest`` step is placed first so that feature
    selection is part of the pipeline and is refit on every training fold.
    """
    score_func: Callable = f_classif if selector_score == "f_classif" else mutual_info_classif

    steps: list = []
    if k is not None:
        steps.append(("select", SelectKBest(score_func, k=min(k, n_features or k))))
    steps.append(("scale", StandardScaler()))

    if name == "xgboost":
        if not _HAS_XGBOOST:
            raise ImportError("xgboost is not installed")
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            n_jobs=1,
            random_state=seed,
            verbosity=0,
        )
        steps.append(("clf", clf))
    elif name == "svm_rbf":
        clf = SVC(
            C=1.0,
            gamma=0.01,
            probability=True,
            class_weight="balanced",
            random_state=seed,
        )
        steps.append(("clf", clf))
    elif name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        steps.append(("clf", clf))
    elif name == "extra_trees":
        clf = ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        steps.append(("clf", clf))
    elif name == "logistic_regression":
        clf = LogisticRegression(
            C=1.0,
            max_iter=3000,
            class_weight="balanced",
            random_state=seed,
        )
        steps.append(("clf", clf))
    elif name == "hist_gb":
        clf = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            random_state=seed,
        )
        steps.append(("clf", clf))
    else:
        raise ValueError(f"Unknown classical model: {name}")

    return Pipeline(steps)


def _subject_probs(
    probabilities: np.ndarray,
    subject_ids: np.ndarray,
) -> pd.Series:
    frame = pd.DataFrame(
        {"subject_id": subject_ids, "probability": probabilities}
    )
    return frame.groupby("subject_id", sort=False)["probability"].mean()


def _directed_edge_indices(n_channels: int) -> tuple[np.ndarray, np.ndarray]:
    """Return all ordered off-diagonal pairs (target, source) for directed connectivity.

    For C channels there are C*(C-1) directed edges (upper AND lower triangle).
    adjacency[target, source] encodes the influence from source to target.
    """
    rows, columns = np.meshgrid(np.arange(n_channels), np.arange(n_channels), indexing="ij")
    mask = rows != columns
    return rows[mask], columns[mask]


def _build_epoch_features(bundle, feature_set: str) -> np.ndarray:
    n = len(bundle.labels)
    node = bundle.node_features.reshape(n, -1)

    probabilities = np.exp(bundle.node_features[:, :, :5])
    relative = probabilities / probabilities.sum(axis=2, keepdims=True)
    relative = relative.reshape(n, -1)

    rows, columns = _directed_edge_indices(bundle.n_channels)
    edges = bundle.adjacency[:, rows, columns]

    if feature_set == "node":
        return node
    if feature_set == "node+adj":
        return np.concatenate([node, edges], axis=1)
    if feature_set == "node+adj+rel":
        return np.concatenate([node, edges, relative], axis=1)
    raise ValueError(f"Unknown feature set: {feature_set}")


def _make_feature_names(n_channels: int, bands: list[str]) -> list[str]:
    names: list[str] = []
    for channel in CHANNELS[:n_channels]:
        for band in bands:
            names.append(f"{channel} | {band} power")
        names.append(f"{channel} | theta/beta ratio")
    rows, columns = _directed_edge_indices(n_channels)
    for target, source in zip(rows.tolist(), columns.tolist()):
        names.append(f"{CHANNELS[target]} <- {CHANNELS[source]} connectivity")
    for channel in CHANNELS[:n_channels]:
        for band in bands:
            names.append(f"{channel} | {band} relative power")
    return names


def evaluate_model(
    bundle,
    feature_set: str,
    model_name: str,
    folds: list[Fold],
    seed: int,
    k: int | None = None,
    selector_score: str = "f_classif",
) -> tuple[dict, pd.DataFrame, list[dict]]:
    """Evaluate one classical model with subject-wise CV.

    Feature selection (if any) is refit inside every training fold. Thresholds
    are chosen on validation subjects only.
    """
    features = _build_epoch_features(bundle, feature_set)
    n_features = features.shape[1]

    probs = np.zeros(len(bundle.labels), dtype=float)
    fold_thresholds: dict[int, float] = {}
    fold_of_subject: dict[str, int] = {}
    fold_rows: list[dict] = []

    all_train_subjects: list[str] = []
    all_val_subjects: list[str] = []
    all_test_subjects: list[str] = []

    for fold, fold_spec in enumerate(folds):
        train_subjects = list(fold_spec.train)
        val_subjects = list(fold_spec.val)
        test_subjects = list(fold_spec.test)

        assert_no_subject_overlap(train_subjects, val_subjects)
        assert_no_subject_overlap(train_subjects, test_subjects)
        assert_no_subject_overlap(val_subjects, test_subjects)

        all_train_subjects.extend(train_subjects)
        all_val_subjects.extend(val_subjects)
        all_test_subjects.extend(test_subjects)

        train_index = np.flatnonzero(np.isin(bundle.subject_ids, train_subjects))
        val_index = np.flatnonzero(np.isin(bundle.subject_ids, val_subjects))
        full_index = np.flatnonzero(np.isin(bundle.subject_ids, train_subjects + val_subjects))
        test_index = np.flatnonzero(np.isin(bundle.subject_ids, test_subjects))

        model = make_model(model_name, seed, n_features=n_features, k=k, selector_score=selector_score)

        # Fit on training subjects only; tune threshold on validation subjects.
        model.fit(features[train_index], bundle.labels[train_index].astype(int))
        val_probabilities = model.predict_proba(features[val_index])[:, 1]
        val_subject_probs = _subject_probs(val_probabilities, bundle.subject_ids[val_index])
        val_y = np.asarray(
            [bundle.subject_labels[str(s)] for s in val_subject_probs.index], dtype=int
        )
        threshold = float(youden_threshold(val_y, val_subject_probs.to_numpy()))
        fold_thresholds[fold] = threshold

        # Refit on train + val, then predict held-out test subjects.
        model.fit(features[full_index], bundle.labels[full_index].astype(int))
        probs[test_index] = model.predict_proba(features[test_index])[:, 1]
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

    metrics = binary_metrics(
        subject_y,
        subject_probs.to_numpy(),
        y_pred=y_pred,
        n_boot=2000,
        seed=seed,
    )

    predictions = pd.DataFrame(
        {
            "subject_id": list(subject_probs.index),
            "y_true": subject_y,
            "y_prob": subject_probs.to_numpy(),
            "y_pred": y_pred,
            "fold": [fold_of_subject[str(s)] for s in subject_probs.index],
            "threshold": [
                fold_thresholds[fold_of_subject[str(s)]]
                for s in subject_probs.index
            ],
        }
    )

    for fold_id, group in predictions.groupby("fold"):
        fold_metrics = binary_metrics(
            group["y_true"].to_numpy(),
            group["y_prob"].to_numpy(),
            y_pred=group["y_pred"].to_numpy(),
            n_boot=200,
            seed=seed + int(fold_id),
        )
        fold_rows.append(
            {
                "fold": int(fold_id),
                "auroc": fold_metrics["auroc"],
                "accuracy": fold_metrics["accuracy"],
                "sensitivity": fold_metrics["sensitivity"],
                "specificity": fold_metrics["specificity"],
                "threshold": fold_thresholds[int(fold_id)],
                "n_test": int(len(group)),
            }
        )

    return metrics, predictions, fold_rows


def run_experiment(
    bundle,
    feature_set: str,
    model_name: str,
    folds: list[Fold],
    seed: int,
    k: int | None = None,
    selector_score: str = "f_classif",
    real_data: bool = True,
) -> ClassicalResult:
    metrics, predictions, fold_rows = evaluate_model(
        bundle, feature_set, model_name, folds, seed, k=k, selector_score=selector_score
    )

    subject_ids = predictions["subject_id"].to_numpy()
    unique_subjects = np.unique(subject_ids)

    return ClassicalResult(
        model_name=model_name,
        feature_set=feature_set,
        n_features=int(_build_epoch_features(bundle, feature_set).shape[1]),
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
        n_subjects=int(len(unique_subjects)),
        n_folds=len(folds),
        n_repeats=1,
        participant_level=True,
        real_data=real_data,
        folds=fold_rows,
        feature_names=_make_feature_names(bundle.n_channels, BANDS),
    )


def save_result(
    result: ClassicalResult,
    model: object,
    output_dir: str | Path,
    scaler: object | None = None,
    selector: object | None = None,
) -> None:
    """Persist a trained classical model with its preprocessing objects."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = f"{result.model_name}_{result.feature_set}"
    joblib.dump(model, output_dir / f"{safe_name}.joblib")

    metadata = {
        "model_name": result.model_name,
        "feature_set": result.feature_set,
        "n_features": result.n_features,
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
        "channels": CHANNELS,
        "bands": {name: list(limits) for name, limits in zip(BANDS, [(1,4),(4,8),(8,13),(13,30),(30,45)])},
        "feature_names": result.feature_names,
        "folds": result.folds,
    }
    (output_dir / f"{safe_name}_metadata.json").write_text(
        __import__("json").dumps(metadata, indent=2), encoding="utf-8"
    )


def load_model(
    model_name: str,
    feature_set: str,
    output_dir: str | Path = "models",
) -> object:
    """Load a previously saved classical model."""
    output_dir = Path(output_dir)
    safe_name = f"{model_name}_{feature_set}"
    return joblib.load(output_dir / f"{safe_name}.joblib")


def load_metadata(
    model_name: str,
    feature_set: str,
    output_dir: str | Path = "models",
) -> dict:
    import json
    output_dir = Path(output_dir)
    safe_name = f"{model_name}_{feature_set}"
    return json.loads((output_dir / f"{safe_name}_metadata.json").read_text(encoding="utf-8"))
