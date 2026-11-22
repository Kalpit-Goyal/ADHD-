"""Train classical ML models (E1-E5) with subject-wise CV on real data.

Experiments:
  E1: XGBoost using Paper 1 features
  E2: RBF-SVM using Paper 1 features
  E3: Random Forest
  E4: Extra Trees
  E5: Logistic Regression
  (bonus) HistGradientBoosting

Feature selection (SelectKBest) is fitted inside each training fold only.
Thresholds are tuned on validation subjects only. Test subjects are held out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classical_models import (
    ClassicalResult,
    _build_epoch_features,
    evaluate_model,
    run_experiment,
    save_result,
)
from src.config import load_config
from src.cv import Fold, make_repeated_folds, load_folds, save_folds, read_subject_labels
from src.data import build_feature_bundle, read_manifest
from src.evaluate import binary_metrics


MODELS = ["xgboost", "svm_rbf", "random_forest", "extra_trees", "logistic_regression", "hist_gb"]
FEATURE_SETS = ["node", "node+adj", "node+adj+rel"]


def build_folds(
    labels: dict[str, int],
    n_splits: int = 5,
    seed: int = 42,
    official_folds_path: Optional[str] = None,
) -> list[Fold]:
    from src.cv import make_repeated_folds, load_folds

    if official_folds_path:
        return load_folds(official_folds_path)
    return make_repeated_folds(labels, n_splits=n_splits, n_repeats=1, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eeg_cte_real.yaml")
    parser.add_argument("--output", default="results/classical_paper1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=150, help="SelectKBest k (None = all features)")
    parser.add_argument(
        "--selector", default="f_classif", choices=["f_classif", "mutual_info"]
    )
    parser.add_argument(
        "--official-folds", default=None, help="Path to official CTE-Net folds JSON"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    records = read_manifest(config.data.root, config.data.manifest)
    bundle = build_feature_bundle(records, config.data, config.features, verbose=True)
    print(
        f"Loaded {len(bundle.subject_labels)} subjects | "
        f"{len(bundle.labels)} epochs"
    )

    labels = read_subject_labels(Path(config.data.root) / config.data.manifest)
    folds = build_folds(labels, n_splits=args.folds, seed=args.seed, official_folds_path=args.official_folds)
    print(f"Using {len(folds)} subject-wise folds")

    # Save folds for reuse by other experiments.
    folds_dir = Path(args.output) / "folds"
    save_folds(folds, folds_dir / "folds.json", metadata={"n_splits": args.folds, "seed": args.seed})

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    all_results: list[ClassicalResult] = []

    for model_name in MODELS:
        for feature_set in FEATURE_SETS:
            print(f"\n=== {model_name} | {feature_set} ===")
            try:
                metrics, predictions, fold_rows = evaluate_model(
                    bundle,
                    feature_set,
                    model_name,
                    folds,
                    args.seed,
                    k=args.k,
                    selector_score=args.selector,
                )

                subject_ids = predictions["subject_id"].to_numpy()
                unique_subjects = np.unique(subject_ids)
                n_features = _build_epoch_features(bundle, feature_set).shape[1]

                result = ClassicalResult(
                    model_name=model_name,
                    feature_set=feature_set,
                    n_features=int(n_features),
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
                    real_data=True,
                    folds=fold_rows,
                )
                all_results.append(result)

                save_result(
                    result,
                    model=None,  # model not persisted here; see note below
                    output_dir=output_dir / "models",
                )

                predictions.to_csv(
                    output_dir / f"{model_name}_{feature_set}_predictions.csv",
                    index=False,
                )

                row = {
                    "experiment": f"E{_experiment_number(model_name)}",
                    "model": model_name,
                    "feature_set": feature_set,
                    "n_features": int(n_features),
                    "n_subjects": int(len(unique_subjects)),
                    "n_folds": len(folds),
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
                    "participant_level": True,
                    "real_data": True,
                }
                rows.append(row)

                print(
                    f"{model_name:16s} {feature_set:12s} | "
                    f"AUROC {result.roc_auc:.3f} | "
                    f"accuracy {result.accuracy:.3f} | "
                    f"sens {result.sensitivity:.3f} | "
                    f"spec {result.specificity:.3f}"
                )
            except Exception as exc:  # pragma: no cover - report and continue
                print(f"  {model_name} | {feature_set}: FAILED -- {exc}")
                rows.append(
                    {
                        "experiment": f"E{_experiment_number(model_name)}",
                        "model": model_name,
                        "feature_set": feature_set,
                        "error": str(exc),
                        "real_data": True,
                    }
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    print("\n=== Classical ML summary (E1-E5) ===")
    print(frame[["experiment", "model", "feature_set", "roc_auc", "accuracy", "sensitivity", "specificity"]].to_string(index=False))


def _experiment_number(model_name: str) -> int:
    mapping = {
        "xgboost": 1,
        "svm_rbf": 2,
        "random_forest": 3,
        "extra_trees": 4,
        "logistic_regression": 5,
        "hist_gb": 6,
    }
    return mapping.get(model_name, 0)


if __name__ == "__main__":
    main()
