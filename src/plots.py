"""Plots for model evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path}")


def plot_roc_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2)
    axes[0].plot([0, 1], [0, 1], "--", color="#999999", lw=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"ROC curve (AUROC = {auroc:.3f})")
    axes[0].grid(alpha=0.25)

    axes[1].plot(recall, precision, color="#2e8b57", lw=2)
    axes[1].axhline(
        float(np.mean(y_true)),
        linestyle="--",
        color="#999999",
        lw=1,
        label=f"Prevalence = {np.mean(y_true):.2f}",
    )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-recall curve (AUPRC = {auprc:.3f})")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, output_dir / "roc_pr.png")


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str | Path,
    n_bins: int = 10,
) -> None:
    output_dir = Path(output_dir)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)

    mean_predicted: list[float] = []
    fraction_positive: list[float] = []
    counts: list[int] = []

    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        mean_predicted.append(float(y_prob[mask].mean()))
        fraction_positive.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    axes[0].plot([0, 1], [0, 1], "--", color="#999999", lw=1)
    axes[0].plot(mean_predicted, fraction_positive, "o-", color="#1f77b4", lw=2)
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Fraction positive")
    axes[0].set_title("Calibration curve")
    axes[0].grid(alpha=0.25)

    axes[1].bar(
        mean_predicted,
        counts,
        width=0.08,
        color="#8fbcd4",
        edgecolor="white",
    )
    axes[1].set_xlabel("Mean predicted probability")
    axes[1].set_ylabel("Subjects")
    axes[1].set_title("Probability distribution")
    axes[1].grid(axis="y", alpha=0.25)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, output_dir / "calibration.png")


def plot_confusion(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str | Path,
    threshold: float = 0.5,
    y_pred: np.ndarray | None = None,
) -> None:
    output_dir = Path(output_dir)
    if y_pred is None:
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred, dtype=int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, axis = plt.subplots(figsize=(4.6, 4.0))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    labels = ["Control", "ADHD"]
    axis.set_xticks([0, 1], labels)
    axis.set_yticks([0, 1], labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion matrix (subject-level)")

    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "#222222",
                fontsize=13,
                fontweight="bold",
            )

    fig.tight_layout()
    _save(fig, output_dir / "confusion_matrix.png")


def plot_history(history: pd.DataFrame, output_dir: str | Path) -> None:
    if history.empty:
        return
    output_dir = Path(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for fold, group in history.groupby("fold"):
        axes[0].plot(group["epoch"], group["train_loss"], label=f"fold {fold}", alpha=0.85)
        axes[1].plot(group["epoch"], group["val_auroc"], label=f"fold {fold}", alpha=0.85)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[0].set_title("Training loss")
    axes[0].grid(alpha=0.25)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation AUROC")
    axes[1].set_title("Validation AUROC")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, output_dir / "training_history.png")


def plot_fold_metrics(folds: pd.DataFrame, output_dir: str | Path) -> None:
    if folds.empty:
        return
    output_dir = Path(output_dir)

    fig, axis = plt.subplots(figsize=(7.5, 4.2))
    positions = np.arange(len(folds))
    axis.bar(positions - 0.18, folds["auroc"], width=0.34, label="AUROC", color="#1f77b4")
    axis.bar(positions + 0.18, folds["accuracy"], width=0.34, label="Accuracy", color="#2e8b57")
    axis.set_xticks(positions, [f"Fold {int(v)}" for v in folds["fold"]])
    axis.set_ylim(0.4, 1.0)
    axis.set_ylabel("Score")
    axis.set_title("Subject-level performance by fold")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, output_dir / "fold_metrics.png")


def plot_leakage_demo(
    epoch_accuracy: float,
    subject_accuracy: float,
    output_path: str | Path,
    epoch_auroc: float | None = None,
    subject_auroc: float | None = None,
) -> None:
    output_path = Path(output_path)
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    labels = ["Epoch-level split\n(leaky)", "Subject-level split\n(clean)"]
    positions = np.arange(2)
    width = 0.34

    accuracy_values = [epoch_accuracy * 100, subject_accuracy * 100]
    bars = axis.bar(
        positions - width / 2,
        accuracy_values,
        width=width,
        color="#d9534f",
        label="Accuracy",
    )
    for bar, value in zip(bars, accuracy_values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    if epoch_auroc is not None and subject_auroc is not None:
        auroc_values = [epoch_auroc * 100, subject_auroc * 100]
        bars = axis.bar(
            positions + width / 2,
            auroc_values,
            width=width,
            color="#1f77b4",
            label="AUROC",
        )
        for bar, value in zip(bars, auroc_values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.0,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 108)
    axis.set_ylabel("Score (%)")
    axis.set_title("Same data, different split")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, output_path)
