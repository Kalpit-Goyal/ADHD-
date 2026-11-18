"""Demonstrate epoch-level leakage versus subject-level validation.

This uses the same spectral features, the same classifier, and the same data.
Only the split changes. Run it after generating synthetic data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import (
    build_feature_bundle,
    indices_for_subjects,
    read_manifest,
)
from src.plots import plot_leakage_demo


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
    )
    classifier.fit(x_train, y_train)
    return classifier.predict_proba(x_test)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eeg_cte.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", default="results/leakage_demo")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.data_root:
        config.data.root = args.data_root

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_manifest(config.data.root, config.data.manifest)
    bundle = build_feature_bundle(records, config.data, config.features, verbose=True)

    x = bundle.node_features.reshape(len(bundle.labels), -1)
    y = bundle.labels.astype(int)

    # 1) Epoch-level split: epochs from the same child appear in train and test.
    epoch_split = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=config.training.seed
    )
    epoch_probabilities = np.zeros(len(y), dtype=float)
    for train_index, test_index in epoch_split.split(x, y):
        epoch_probabilities[test_index] = _fit_predict(
            x[train_index], y[train_index], x[test_index], config.training.seed
        )

    epoch_accuracy = float(accuracy_score(y, epoch_probabilities >= 0.5))
    epoch_auroc = float(roc_auc_score(y, epoch_probabilities))

    # 2) Subject-level split: entire children are held out.
    subjects = np.asarray(sorted(bundle.subject_labels), dtype=object)
    subject_labels = np.asarray(
        [bundle.subject_labels[str(subject)] for subject in subjects], dtype=int
    )
    subject_split = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=config.training.seed
    )

    subject_probabilities: dict[str, list[float]] = {}
    for train_subjects, test_subjects in subject_split.split(subjects, subject_labels):
        train_index = indices_for_subjects(bundle, subjects[train_subjects])
        test_index = indices_for_subjects(bundle, subjects[test_subjects])
        probabilities = _fit_predict(
            x[train_index], y[train_index], x[test_index], config.training.seed
        )
        for subject_id, probability in zip(
            bundle.subject_ids[test_index], probabilities
        ):
            subject_probabilities.setdefault(str(subject_id), []).append(float(probability))

    subject_ids = sorted(subject_probabilities)
    subject_y = np.asarray(
        [bundle.subject_labels[subject_id] for subject_id in subject_ids], dtype=int
    )
    subject_p = np.asarray(
        [np.mean(subject_probabilities[subject_id]) for subject_id in subject_ids],
        dtype=float,
    )

    subject_accuracy = float(accuracy_score(subject_y, subject_p >= 0.5))
    subject_auroc = float(roc_auc_score(subject_y, subject_p))

    summary = {
        "epoch_level": {
            "accuracy": epoch_accuracy,
            "auroc": epoch_auroc,
            "n": int(len(y)),
        },
        "subject_level": {
            "accuracy": subject_accuracy,
            "auroc": subject_auroc,
            "n": int(len(subject_y)),
        },
        "accuracy_gap": epoch_accuracy - subject_accuracy,
        "auroc_gap": epoch_auroc - subject_auroc,
    }

    (output_dir / "leakage_demo.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_leakage_demo(
        epoch_accuracy=epoch_accuracy,
        subject_accuracy=subject_accuracy,
        epoch_auroc=epoch_auroc,
        subject_auroc=subject_auroc,
        output_path=output_dir / "leakage_demo.png",
    )

    print("\nEpoch-level:  "
          f"accuracy {epoch_accuracy:.3f} | AUROC {epoch_auroc:.3f}")
    print("Subject-level:"
          f" accuracy {subject_accuracy:.3f} | AUROC {subject_auroc:.3f}")
    print(
        f"Gap: {summary['accuracy_gap'] * 100:.1f} accuracy points, "
        f"{summary['auroc_gap']:.3f} AUROC"
    )
    print(f"Saved {output_dir / 'leakage_demo.json'}")


if __name__ == "__main__":
    main()
