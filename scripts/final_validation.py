"""Repeated, subject-wise validation of the best classical EEG models.

Every repeat uses a fresh stratified 5-fold subject split. Inside each outer
fold, models are fit on 80% of the training subjects, the decision threshold is
chosen on the held-out 20% (Youden), and then the models are refit on all
training subjects before predicting the untouched test subjects.

Reports mean +/- SD across repeats and pooled out-of-fold metrics with
per-fold thresholds. No test subject is used for fitting or threshold choice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classical_baseline import subject_probabilities
from src.config import load_config
from src.data import FeatureBundle, build_feature_bundle, read_manifest
from src.evaluate import binary_metrics, youden_threshold


def make_svm(c: float, gamma: str | float = "scale", seed: int = 0):
    return make_pipeline(
        StandardScaler(),
        SVC(
            C=c,
            gamma=gamma,
            probability=True,
            class_weight="balanced",
            random_state=seed,
        ),
    )


def make_extra_trees(seed: int, leaf: int = 5):
    return ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=leaf,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def make_select_svm(seed: int, k: int = 150):
    return make_pipeline(
        SelectKBest(f_classif, k=k),
        StandardScaler(),
        SVC(
            C=1.0,
            gamma=0.01,
            probability=True,
            class_weight="balanced",
            random_state=seed,
        ),
    )


def build_epoch_features(bundle: FeatureBundle) -> dict[str, np.ndarray]:
    n = len(bundle.labels)
    node = bundle.node_features.reshape(n, -1)

    probabilities = np.exp(bundle.node_features[:, :, :5])
    relative = probabilities / probabilities.sum(axis=2, keepdims=True)
    relative = relative.reshape(n, -1)

    rows, columns = np.triu_indices(bundle.n_channels, k=1)
    edges = bundle.adjacency[:, rows, columns]

    return {
        "node": node,
        "node+adj": np.concatenate([node, edges], axis=1),
        "node+adj+rel": np.concatenate([node, edges, relative], axis=1),
    }


def aggregate_subjects(
    bundle: FeatureBundle,
    features: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate epoch features per subject (mean and/or std)."""
    subjects = np.asarray(sorted(bundle.subject_labels), dtype=object)
    labels = np.asarray(
        [bundle.subject_labels[str(s)] for s in subjects], dtype=int
    )
    rows = []
    for subject in subjects:
        mask = bundle.subject_ids == subject
        block = features[mask]
        pieces = []
        if "mean" in mode:
            pieces.append(block.mean(axis=0))
        if "std" in mode:
            pieces.append(block.std(axis=0))
        rows.append(np.concatenate(pieces))
    return subjects, labels, np.stack(rows)


def _fold_subject_predictions(
    probabilities: np.ndarray,
    subject_ids: np.ndarray,
) -> pd.Series:
    return subject_probabilities(probabilities, subject_ids)


def _metrics_from_oof(
    probs: np.ndarray,
    bundle: FeatureBundle,
    fold_thresholds: dict[int, float],
    fold_of_subject: dict[str, int],
    seed: int,
) -> tuple[dict, pd.DataFrame, list[dict]]:
    subject_probs = subject_probabilities(probs, bundle.subject_ids)
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

    fold_rows = []
    for fold, group in predictions.groupby("fold"):
        fold_metrics = binary_metrics(
            group["y_true"].to_numpy(),
            group["y_prob"].to_numpy(),
            y_pred=group["y_pred"].to_numpy(),
            n_boot=200,
            seed=seed + int(fold),
        )
        fold_rows.append(
            {
                "fold": int(fold),
                "auroc": fold_metrics["auroc"],
                "accuracy": fold_metrics["accuracy"],
                "sensitivity": fold_metrics["sensitivity"],
                "specificity": fold_metrics["specificity"],
                "threshold": fold_thresholds[int(fold)],
            }
        )
    return metrics, predictions, fold_rows


def evaluate_epochs(
    bundle: FeatureBundle,
    features: np.ndarray,
    models: list,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[dict, pd.DataFrame, list[dict]]:
    probs = np.zeros(len(bundle.labels), dtype=float)
    fold_thresholds: dict[int, float] = {}
    fold_of_subject: dict[str, int] = {}

    for fold, (train_subjects, test_subjects) in enumerate(folds):
        inner_train, inner_val = train_test_split(
            train_subjects,
            test_size=0.2,
            stratify=[bundle.subject_labels[str(s)] for s in train_subjects],
            random_state=seed + fold,
        )
        train_index = np.flatnonzero(np.isin(bundle.subject_ids, inner_train))
        val_index = np.flatnonzero(np.isin(bundle.subject_ids, inner_val))
        full_index = np.flatnonzero(np.isin(bundle.subject_ids, train_subjects))
        test_index = np.flatnonzero(np.isin(bundle.subject_ids, test_subjects))

        val_probabilities = []
        for model in models:
            fitted = clone(model).fit(
                features[train_index], bundle.labels[train_index].astype(int)
            )
            val_probabilities.append(
                fitted.predict_proba(features[val_index])[:, 1]
            )
        val_subjects = _fold_subject_predictions(
            np.mean(val_probabilities, axis=0), bundle.subject_ids[val_index]
        )
        val_y = np.asarray(
            [bundle.subject_labels[str(s)] for s in val_subjects.index], dtype=int
        )
        threshold = float(youden_threshold(val_y, val_subjects.to_numpy()))
        fold_thresholds[fold] = threshold

        test_probabilities = []
        for model in models:
            fitted = clone(model).fit(
                features[full_index], bundle.labels[full_index].astype(int)
            )
            test_probabilities.append(
                fitted.predict_proba(features[test_index])[:, 1]
            )
        probs[test_index] = np.mean(test_probabilities, axis=0)
        for subject in test_subjects:
            fold_of_subject[str(subject)] = fold

    metrics, predictions, fold_rows = _metrics_from_oof(
        probs, bundle, fold_thresholds, fold_of_subject, seed
    )
    return metrics, predictions, fold_rows


def evaluate_subjects(
    features: np.ndarray,
    subjects: np.ndarray,
    labels: np.ndarray,
    models: list,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[dict, pd.DataFrame, list[dict]]:
    probs = np.zeros(len(subjects), dtype=float)
    fold_thresholds: dict[int, float] = {}
    fold_of_subject: dict[str, int] = {}
    index_of = {str(subject): i for i, subject in enumerate(subjects)}

    for fold, (train_subjects, test_subjects) in enumerate(folds):
        inner_train, inner_val = train_test_split(
            train_subjects,
            test_size=0.2,
            stratify=[labels[index_of[str(s)]] for s in train_subjects],
            random_state=seed + fold,
        )
        train_index = np.asarray([index_of[str(s)] for s in inner_train])
        val_index = np.asarray([index_of[str(s)] for s in inner_val])
        full_index = np.asarray([index_of[str(s)] for s in train_subjects])
        test_index = np.asarray([index_of[str(s)] for s in test_subjects])

        val_probabilities = []
        for model in models:
            fitted = clone(model).fit(features[train_index], labels[train_index])
            val_probabilities.append(fitted.predict_proba(features[val_index])[:, 1])
        val_prob = np.mean(val_probabilities, axis=0)
        threshold = float(youden_threshold(labels[val_index], val_prob))
        fold_thresholds[fold] = threshold

        test_probabilities = []
        for model in models:
            fitted = clone(model).fit(features[full_index], labels[full_index])
            test_probabilities.append(fitted.predict_proba(features[test_index])[:, 1])
        probs[test_index] = np.mean(test_probabilities, axis=0)
        for subject in test_subjects:
            fold_of_subject[str(subject)] = fold

    y_pred = np.asarray(
        [
            int(probs[i] >= fold_thresholds[fold_of_subject[str(subject)]])
            for i, subject in enumerate(subjects)
        ],
        dtype=int,
    )
    metrics = binary_metrics(labels, probs, y_pred=y_pred, n_boot=2000, seed=seed)
    predictions = pd.DataFrame(
        {
            "subject_id": [str(s) for s in subjects],
            "y_true": labels,
            "y_prob": probs,
            "y_pred": y_pred,
            "fold": [fold_of_subject[str(s)] for s in subjects],
            "threshold": [
                fold_thresholds[fold_of_subject[str(s)]] for s in subjects
            ],
        }
    )
    fold_rows = []
    for fold, group in predictions.groupby("fold"):
        fold_metrics = binary_metrics(
            group["y_true"].to_numpy(),
            group["y_prob"].to_numpy(),
            y_pred=group["y_pred"].to_numpy(),
            n_boot=200,
            seed=seed + int(fold),
        )
        fold_rows.append(
            {
                "fold": int(fold),
                "auroc": fold_metrics["auroc"],
                "accuracy": fold_metrics["accuracy"],
                "sensitivity": fold_metrics["sensitivity"],
                "specificity": fold_metrics["specificity"],
                "threshold": fold_thresholds[int(fold)],
            }
        )
    return metrics, predictions, fold_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eeg_cte_real.yaml")
    parser.add_argument("--output", default="results/final_validation")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    records = read_manifest(config.data.root, config.data.manifest)
    bundle = build_feature_bundle(records, config.data, config.features, verbose=True)
    print(
        f"Loaded {len(bundle.subject_labels)} subjects | "
        f"{len(bundle.labels)} epochs"
    )

    epoch_features = build_epoch_features(bundle)
    subject_features = {}
    for feature_name, features in epoch_features.items():
        for mode in ("mean", "mean+std"):
            subject_features[(feature_name, mode)] = aggregate_subjects(
                bundle, features, mode
            )

    subjects = np.asarray(sorted(bundle.subject_labels), dtype=object)
    labels = np.asarray(
        [bundle.subject_labels[str(s)] for s in subjects], dtype=int
    )

    specs = {
        "epoch_svm_c0.3": dict(
            kind="epoch",
            feature="node+adj+rel",
            models=[make_svm(0.3)],
        ),
        "epoch_svm_c1_g0.01": dict(
            kind="epoch",
            feature="node+adj+rel",
            models=[make_svm(1.0, 0.01)],
        ),
        "epoch_ensemble_svm": dict(
            kind="epoch",
            feature="node+adj+rel",
            models=[make_svm(0.3), make_svm(1.0, 0.01)],
        ),
        "epoch_ensemble_svm_et": dict(
            kind="epoch",
            feature="node+adj+rel",
            models=[make_svm(0.3), make_svm(1.0, 0.01), make_extra_trees(0)],
        ),
        "epoch_select150_svm": dict(
            kind="epoch",
            feature="node+adj+rel",
            models=[make_select_svm(0, 150)],
        ),
        "subject_mean_svm_c0.3": dict(
            kind="subject",
            feature="node+adj+rel",
            agg="mean",
            models=[make_svm(0.3)],
        ),
        "subject_mean_svm_c1_g0.01": dict(
            kind="subject",
            feature="node+adj+rel",
            agg="mean",
            models=[make_svm(1.0, 0.01)],
        ),
        "subject_mean+std_svm_c1_g0.01": dict(
            kind="subject",
            feature="node+adj+rel",
            agg="mean+std",
            models=[make_svm(1.0, 0.01)],
        ),
        "subject_mean_ensemble_svm": dict(
            kind="subject",
            feature="node+adj+rel",
            agg="mean",
            models=[make_svm(0.3), make_svm(1.0, 0.01)],
        ),
    }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    prediction_frames = []
    for name, spec in specs.items():
        for repeat in range(args.repeats):
            seed = config.training.seed + repeat * 1000
            splitter = StratifiedKFold(
                n_splits=args.folds, shuffle=True, random_state=seed
            )
            folds = [
                (subjects[train], subjects[test])
                for train, test in splitter.split(subjects, labels)
            ]

            if spec["kind"] == "epoch":
                metrics, predictions, fold_rows = evaluate_epochs(
                    bundle, epoch_features[spec["feature"]], spec["models"], folds, seed
                )
            else:
                subject_data = subject_features[(spec["feature"], spec["agg"])]
                metrics, predictions, fold_rows = evaluate_subjects(
                    subject_data[2],
                    subject_data[0],
                    subject_data[1],
                    spec["models"],
                    folds,
                    seed,
                )

            rows.append(
                {
                    "config": name,
                    "repeat": repeat + 1,
                    "seed": seed,
                    "auroc": metrics["auroc"],
                    "accuracy": metrics["accuracy"],
                    "sensitivity": metrics["sensitivity"],
                    "specificity": metrics["specificity"],
                    "mean_threshold": float(
                        np.mean([item["threshold"] for item in fold_rows])
                    ),
                    "fold_auroc_mean": float(
                        np.mean([item["auroc"] for item in fold_rows])
                    ),
                    "fold_auroc_sd": float(
                        np.std([item["auroc"] for item in fold_rows])
                    ),
                    "fold_accuracy_mean": float(
                        np.mean([item["accuracy"] for item in fold_rows])
                    ),
                    "fold_accuracy_sd": float(
                        np.std([item["accuracy"] for item in fold_rows])
                    ),
                }
            )
            predictions.insert(0, "repeat", repeat + 1)
            predictions.insert(0, "config", name)
            prediction_frames.append(predictions)
            print(
                f"{name:30s} repeat {repeat + 1} | "
                f"AUROC {metrics['auroc']:.3f} | "
                f"accuracy {metrics['accuracy']:.3f}"
            )

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("config", as_index=False)
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            sensitivity_mean=("sensitivity", "mean"),
            specificity_mean=("specificity", "mean"),
        )
        .sort_values("accuracy_mean", ascending=False)
    )
    frame.to_csv(output_dir / "repeats.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        output_dir / "predictions.csv", index=False
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "repeats": args.repeats,
                "folds": args.folds,
                "decision_rule": "per-fold Youden threshold on validation subjects",
                "results": summary.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nMean +/- SD across repeats (sorted by accuracy):")
    for _, row in summary.iterrows():
        print(
            f"{row['config']:30s} | "
            f"AUROC {row['auroc_mean']:.3f} +/- {row['auroc_sd']:.3f} | "
            f"accuracy {row['accuracy_mean']:.3f} +/- {row['accuracy_sd']:.3f} | "
            f"sens {row['sensitivity_mean']:.3f} | "
            f"spec {row['specificity_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
