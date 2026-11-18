"""Fast classical baselines on the EEG features, with subject-wise CV.

For each outer fold: fit on 80% of the training subjects, tune the decision
threshold on the held-out 20%, refit on all training subjects, then evaluate
on the untouched test subjects.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import (
    FeatureBundle,
    build_feature_bundle,
    indices_for_subjects,
    read_manifest,
)
from src.evaluate import binary_metrics, youden_threshold


def make_models(seed: int) -> dict:
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, max_iter=3000, class_weight="balanced", random_state=seed
            ),
        ),
        "svm_rbf": make_pipeline(
            StandardScaler(),
            SVC(
                C=1.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "hist_gb": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, random_state=seed
        ),
    }


def build_features(bundle: FeatureBundle, mode: str) -> np.ndarray:
    node = bundle.node_features.reshape(len(bundle.labels), -1)
    if mode == "node":
        return node

    rows, columns = np.triu_indices(bundle.n_channels, k=1)
    edges = bundle.adjacency[:, rows, columns]
    return np.concatenate([node, edges], axis=1)


def subject_probabilities(
    probabilities: np.ndarray,
    subject_ids: np.ndarray,
) -> pd.Series:
    frame = pd.DataFrame(
        {"subject_id": subject_ids, "probability": probabilities}
    )
    return frame.groupby("subject_id", sort=False)["probability"].mean()


def evaluate(
    bundle: FeatureBundle,
    features: np.ndarray,
    model,
    folds: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    probs = np.zeros(len(bundle.labels), dtype=float)
    thresholds: list[float] = []

    for train_subjects, test_subjects in folds:
        inner_train, inner_val = train_test_split(
            train_subjects,
            test_size=0.2,
            stratify=[bundle.subject_labels[str(s)] for s in train_subjects],
            random_state=seed,
        )

        train_index = indices_for_subjects(bundle, inner_train)
        val_index = indices_for_subjects(bundle, inner_val)
        model.fit(features[train_index], bundle.labels[train_index].astype(int))

        val_probabilities = model.predict_proba(features[val_index])[:, 1]
        val_subjects = subject_probabilities(
            val_probabilities, bundle.subject_ids[val_index]
        )
        val_y = np.asarray(
            [bundle.subject_labels[str(s)] for s in val_subjects.index], dtype=int
        )
        threshold = youden_threshold(val_y, val_subjects.to_numpy())
        thresholds.append(float(threshold))

        full_index = indices_for_subjects(bundle, train_subjects)
        test_index = indices_for_subjects(bundle, test_subjects)
        model.fit(features[full_index], bundle.labels[full_index].astype(int))
        probs[test_index] = model.predict_proba(features[test_index])[:, 1]

    subject_probs = subject_probabilities(probs, bundle.subject_ids)
    subject_y = np.asarray(
        [bundle.subject_labels[str(s)] for s in subject_probs.index], dtype=int
    )
    # Each subject was predicted by exactly one fold; use that fold's threshold.
    subject_ids = list(subject_probs.index)
    threshold_by_subject = {}
    for (_, test_subjects), threshold in zip(folds, thresholds):
        for subject in test_subjects:
            threshold_by_subject[str(subject)] = threshold
    y_pred = np.asarray(
        [
            int(subject_probs[subject] >= threshold_by_subject[str(subject)])
            for subject in subject_ids
        ],
        dtype=int,
    )

    metrics = binary_metrics(
        subject_y,
        subject_probs.to_numpy(),
        y_pred=y_pred,
        n_boot=1000,
        seed=seed,
    )
    metrics["mean_threshold"] = float(np.mean(thresholds))
    predictions = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "y_true": subject_y,
            "y_prob": subject_probs.to_numpy(),
            "y_pred": y_pred,
        }
    )
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eeg_cte_real.yaml")
    parser.add_argument("--output", default="results/classical_baseline")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    config.training.n_splits = args.folds

    records = read_manifest(config.data.root, config.data.manifest)
    bundle = build_feature_bundle(records, config.data, config.features, verbose=False)
    print(
        f"Loaded {len(bundle.subject_labels)} subjects, "
        f"{len(bundle.labels)} epochs"
    )

    subjects = np.asarray(sorted(bundle.subject_labels), dtype=object)
    labels = np.asarray(
        [bundle.subject_labels[str(s)] for s in subjects], dtype=int
    )
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=config.training.seed
    )
    folds = [
        (subjects[train], subjects[test])
        for train, test in splitter.split(subjects, labels)
    ]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for feature_mode in ("node", "node+adj"):
        features = build_features(bundle, feature_mode)
        for name, model in make_models(config.training.seed).items():
            metrics, predictions = evaluate(
                bundle, features, model, folds, config.training.seed
            )
            rows.append(
                {
                    "features": feature_mode,
                    "model": name,
                    "auroc": metrics["auroc"],
                    "accuracy": metrics["accuracy"],
                    "sensitivity": metrics["sensitivity"],
                    "specificity": metrics["specificity"],
                }
            )
            predictions.to_csv(
                output_dir / f"{feature_mode}_{name}_predictions.csv", index=False
            )
            print(
                f"{feature_mode:8s} {name:12s} | "
                f"AUROC {metrics['auroc']:.3f} | "
                f"accuracy {metrics['accuracy']:.3f} | "
                f"sens {metrics['sensitivity']:.3f} | "
                f"spec {metrics['specificity']:.3f}"
            )

    frame = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    frame.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(frame.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    print("\nBest by AUROC:")
    print(frame.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
