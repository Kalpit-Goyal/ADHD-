"""Generate all visualizations from saved experiment results."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

from src.plots import _save

RESULTS = Path("results")
FIGURES = RESULTS / "figures"


def load_summary() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for sub in ["classical_paper1", "deep_models", "hybrid"]:
        p = RESULTS / sub / "summary.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_model_comparison(df: pd.DataFrame) -> None:
    """Bar chart of AUROC and accuracy across all experiments."""
    if df.empty or "roc_auc" not in df.columns:
        return
    df = df.dropna(subset=["roc_auc"]).copy()
    if df.empty:
        return

    # Build a readable label per row
    labels = []
    for _, r in df.iterrows():
        parts = [str(r.get("experiment", ""))]
        if pd.notna(r.get("model")):
            parts.append(str(r["model"]))
        if pd.notna(r.get("feature_set")):
            parts.append(str(r["feature_set"]))
        labels.append(" + ".join(parts))

    x = np.arange(len(labels))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    auroc = axes[0].bar(
        x - width / 2, df["roc_auc"], width=width, label="AUROC", color="#1f77b4"
    )
    acc = axes[1].bar(
        x - width / 2, df["accuracy"], width=width, label="Accuracy", color="#2e8b57"
    )

    for ax, bars, vals in [
        (axes[0], auroc, df["roc_auc"]),
        (axes[1], acc, df["accuracy"]),
    ]:
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0.45, 0.85 if ax is axes[0] else 0.8)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_title("AUROC by experiment")
    axes[1].set_title("Accuracy by experiment")
    fig.tight_layout()
    _save(fig, FIGURES / "model_comparison.png")


def plot_roc_pr_from_oof(csv_path: Path, label: str, out_dir: Path) -> None:
    """ROC + PR curves from an OOF predictions CSV."""
    if not csv_path.exists():
        print(f"  skipping {csv_path} (not found)")
        return
    df = pd.read_csv(csv_path)
    if "y_true" not in df.columns or "y_prob" not in df.columns:
        print(f"  skipping {csv_path} (missing y_true/y_prob)")
        return

    y_true = df["y_true"].to_numpy()
    y_prob = df["y_prob"].to_numpy()

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUROC = {auroc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#999999", lw=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"ROC curve — {label}")
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].grid(alpha=0.25)

    axes[1].plot(recall, precision, color="#2e8b57", lw=2, label=f"AUPRC = {auprc:.3f}")
    axes[1].axhline(float(np.mean(y_true)), linestyle="--", color="#999999", lw=1)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-recall curve — {label}")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, out_dir / f"{label}_roc_pr.png")


def plot_confusion_from_oof(csv_path: Path, label: str, out_dir: Path) -> None:
    """Confusion matrix from an OOF predictions CSV."""
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if "y_true" not in df.columns or "y_prob" not in df.columns:
        return
    y_true = df["y_true"].to_numpy()
    y_prob = df["y_prob"].to_numpy()
    y_pred = (y_prob >= 0.5).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, axis = plt.subplots(figsize=(4.6, 4.0))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    labels = ["Control", "ADHD"]
    axis.set_xticks([0, 1], labels)
    axis.set_yticks([0, 1], labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(f"Confusion matrix — {label}")
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
    _save(fig, out_dir / f"{label}_confusion.png")


def plot_sensitivity_specificity(df: pd.DataFrame) -> None:
    """Bar chart of sensitivity vs specificity."""
    if df.empty or "sensitivity" not in df.columns:
        return
    df = df.dropna(subset=["sensitivity", "specificity"]).copy()
    if df.empty:
        return

    labels = []
    for _, r in df.iterrows():
        parts = [str(r.get("experiment", ""))]
        if pd.notna(r.get("model")):
            parts.append(str(r["model"]))
        if pd.notna(r.get("feature_set")):
            parts.append(str(r["feature_set"]))
        labels.append(" + ".join(parts))

    x = np.arange(len(labels))
    width = 0.38

    fig, axis = plt.subplots(figsize=(14, 5.2))
    sens = axis.bar(
        x - width / 2, df["sensitivity"], width=width, label="Sensitivity", color="#d9534f"
    )
    spec = axis.bar(
        x + width / 2, df["specificity"], width=width, label="Specificity", color="#1f77b4"
    )
    for bar, v in zip(sens, df["sensitivity"]):
        axis.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, v in zip(spec, df["specificity"]):
        axis.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    axis.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
    axis.set_ylim(0.4, 0.9)
    axis.set_ylabel("Score")
    axis.set_title("Sensitivity vs Specificity by experiment")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, FIGURES / "sensitivity_specificity.png")


def plot_feature_set_ablation(df: pd.DataFrame) -> None:
    """AUROC by feature set (node / node+adj / node+adj+rel) for classical models."""
    classical = df[df["experiment"].isin(["E1", "E2", "E3", "E4", "E5", "E6"])].copy()
    if classical.empty or "feature_set" not in classical.columns:
        return

    models = classical["model"].unique()
    feature_sets = ["node", "node+adj", "node+adj+rel"]

    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.8), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = classical[classical["model"] == model]
        for fs in feature_sets:
            row = sub[sub["feature_set"] == fs]
            if row.empty:
                continue
            ax.bar(fs, float(row["roc_auc"].iloc[0]), label=fs, alpha=0.85)
        ax.set_title(model.replace("_", " ").title())
        ax.set_ylim(0.5, 0.8)
        ax.set_ylabel("AUROC")
        ax.tick_params(axis="x", rotation=15, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("AUROC by feature set — classical models (E1-E6)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, FIGURES / "feature_set_ablation.png")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_summary()
    print(f"Loaded {len(df)} result rows from summaries")

    # Overall comparison
    plot_model_comparison(df)
    plot_sensitivity_specificity(df)
    plot_feature_set_ablation(df)

    # Per-model ROC/PR + confusion from OOF CSVs
    oof_map: dict[str, Path] = {}

    # Classical models
    classical_dir = RESULTS / "classical_paper1"
    if classical_dir.exists():
        for p in classical_dir.glob("*_predictions.csv"):
            stem = p.stem.replace("_predictions", "")
            oof_map[stem] = p

    # Deep models
    deep_dir = RESULTS / "deep_models"
    if deep_dir.exists():
        for p in deep_dir.glob("*_oof_predictions.csv"):
            stem = p.stem.replace("_oof_predictions", "")
            oof_map[stem] = p

    for label, path in sorted(oof_map.items()):
        print(f"  plotting {label} from {path}")
        plot_roc_pr_from_oof(path, label, FIGURES)
        plot_confusion_from_oof(path, label, FIGURES)

    # Fold metrics for deep models
    for p in sorted((RESULTS / "deep_models").glob("*_fold_metrics.csv")) if (RESULTS / "deep_models").exists() else []:
        label = p.stem.replace("_fold_metrics", "")
        folds = pd.read_csv(p)
        from src.plots import plot_fold_metrics
        plot_fold_metrics(folds, FIGURES / label)

    # Training history for deep models
    for p in sorted((RESULTS / "deep_models").glob("*_history.csv")) if (RESULTS / "deep_models").exists() else []:
        label = p.stem.replace("_history", "")
        history = pd.read_csv(p)
        from src.plots import plot_history
        plot_history(history, FIGURES / label)

    print(f"\nAll figures written to {FIGURES}")


if __name__ == "__main__":
    main()
