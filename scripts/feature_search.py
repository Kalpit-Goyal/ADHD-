"""Small search over classical models and EEG feature sets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from classical_baseline import evaluate
from src.config import load_config
from src.data import build_feature_bundle, read_manifest


def build_feature_sets(bundle) -> dict[str, np.ndarray]:
    n = len(bundle.labels)
    node = bundle.node_features.reshape(n, -1)

    probabilities = np.exp(bundle.node_features[:, :, :5])
    relative = probabilities / probabilities.sum(axis=2, keepdims=True)
    relative = relative.reshape(n, -1)

    rows, columns = np.triu_indices(bundle.n_channels, k=1)
    upper = bundle.adjacency[:, rows, columns]

    return {
        "node+adj": np.concatenate([node, upper], axis=1),
        "node+adj+rel": np.concatenate([node, upper, relative], axis=1),
        "node+rel": np.concatenate([node, relative], axis=1),
    }


def make_models(seed: int) -> dict:
    def svm(c: float, gamma: str | float = "scale"):
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

    return {
        "svm_c0.3": svm(0.3),
        "svm_c1": svm(1.0),
        "svm_c3": svm(3.0),
        "svm_c1_g0.01": svm(1.0, 0.01),
        "et_leaf3": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "et_leaf5": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "et_leaf10": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eeg_cte_real.yaml")
    parser.add_argument("--output", default="results/feature_search")
    args = parser.parse_args()

    config = load_config(args.config)
    records = read_manifest(config.data.root, config.data.manifest)
    bundle = build_feature_bundle(records, config.data, config.features, verbose=False)

    subjects = np.asarray(sorted(bundle.subject_labels), dtype=object)
    labels = np.asarray(
        [bundle.subject_labels[str(s)] for s in subjects], dtype=int
    )
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(
        n_splits=config.training.n_splits,
        shuffle=True,
        random_state=config.training.seed,
    )
    folds = [
        (subjects[train], subjects[test])
        for train, test in splitter.split(subjects, labels)
    ]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for feature_name, features in build_feature_sets(bundle).items():
        for model_name, model in make_models(config.training.seed).items():
            metrics, predictions = evaluate(
                bundle, features, model, folds, config.training.seed
            )
            rows.append(
                {
                    "features": feature_name,
                    "model": model_name,
                    "n_features": int(features.shape[1]),
                    "auroc": metrics["auroc"],
                    "accuracy": metrics["accuracy"],
                    "sensitivity": metrics["sensitivity"],
                    "specificity": metrics["specificity"],
                    "mean_threshold": metrics["mean_threshold"],
                }
            )
            predictions.to_csv(
                output_dir / f"{feature_name}_{model_name}.csv", index=False
            )
            print(
                f"{feature_name:14s} {model_name:12s} | "
                f"AUROC {metrics['auroc']:.3f} | "
                f"accuracy {metrics['accuracy']:.3f} | "
                f"sens {metrics['sensitivity']:.3f} | "
                f"spec {metrics['specificity']:.3f}"
            )

    frame = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    frame.to_csv(output_dir / "summary.csv", index=False)
    print("\nTop by AUROC:")
    print(frame.head(8).to_string(index=False))
    print("\nTop by accuracy:")
    print(frame.sort_values("accuracy", ascending=False).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
