"""Evaluation metrics for binary ADHD classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected calibration error with equal-width probability bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)

    error = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        error += float(mask.mean()) * abs(
            float(y_true[mask].mean()) - float(y_prob[mask].mean())
        )
    return float(error)


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold that maximises sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    scores = tpr - fpr
    best = int(np.argmax(scores))
    threshold = float(thresholds[best])
    if not np.isfinite(threshold):
        return 0.5
    return threshold


def _bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric,
    n_boot: int = 1000,
    seed: int = 42,
) -> list[float] | None:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = len(y_true)

    for _ in range(n_boot):
        index = rng.integers(0, n, size=n)
        y_sample = y_true[index]
        p_sample = y_prob[index]
        if len(np.unique(y_sample)) < 2:
            continue
        try:
            value = float(metric(y_sample, p_sample))
        except ValueError:
            continue
        if np.isfinite(value):
            values.append(value)

    if len(values) < 50:
        return None
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def _bootstrap_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> list[float] | None:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    n = len(y_true)

    for _ in range(n_boot):
        index = rng.integers(0, n, size=n)
        values.append(float(accuracy_score(y_true[index], y_pred[index])))

    if len(values) < 50:
        return None
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def binary_metrics(
    y_true: np.ndarray | list[int],
    y_prob: np.ndarray | list[float],
    threshold: float = 0.5,
    n_boot: int = 1000,
    seed: int = 42,
    y_pred: np.ndarray | list[int] | None = None,
) -> dict:
    """Compute discrimination, calibration, and clinical-utility metrics.

    If ``y_pred`` is given it is used directly (for per-fold thresholds);
    otherwise predictions are ``y_prob >= threshold``.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    if len(np.unique(y_true)) < 2:
        raise ValueError("Both classes are required to compute binary metrics")

    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred, dtype=int)
        if len(y_pred) != len(y_true):
            raise ValueError("y_true and y_pred must have the same length")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")

    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))

    return {
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()),
        "threshold": float(threshold),
        "auroc": auroc,
        "auroc_ci": _bootstrap_ci(y_true, y_prob, roc_auc_score, n_boot, seed),
        "auprc": auprc,
        "auprc_ci": _bootstrap_ci(
            y_true, y_prob, average_precision_score, n_boot, seed
        ),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_ci": _bootstrap_accuracy(y_true, y_pred, n_boot, seed),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def format_metrics(metrics: dict) -> str:
    """Compact human-readable summary."""
    def interval(key: str) -> str:
        values = metrics.get(key)
        if not values:
            return ""
        return f" [{values[0]:.3f}, {values[1]:.3f}]"

    return (
        f"AUROC {metrics['auroc']:.3f}{interval('auroc_ci')} | "
        f"AUPRC {metrics['auprc']:.3f} | "
        f"accuracy {metrics['accuracy']:.3f}{interval('accuracy_ci')} | "
        f"sensitivity {metrics['sensitivity']:.3f} | "
        f"specificity {metrics['specificity']:.3f} | "
        f"Brier {metrics['brier']:.3f} | ECE {metrics['ece']:.3f}"
    )
