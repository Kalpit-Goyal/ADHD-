"""Train deep learning models (E6-E8) with subject-wise CV on real data.

Experiments:
  E6: EEGNet (raw-signal CNN baseline)
  E7: Original CTE-Net (attempted reconstruction - src/ctenet.py)
  E8: Existing CTE-Net-inspired graph-attention model (src/models.py)

Saves model weights to models/ for each fold. Uses subject-wise 5-fold CV.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.cv import Fold, load_folds, make_repeated_folds, read_subject_labels, save_folds
from src.data import build_feature_bundle, read_manifest
from src.train import resolve_device, run_cross_validation, set_seed


MODELS = {
    "eegnet": "eegnet",
    "ctenet_original": "ctenet_original",
    "ctenet_inspired": "cte_net",
}


def build_folds(
    labels: dict[str, int],
    n_splits: int = 5,
    seed: int = 42,
    official_folds_path: Optional[str] = None,
) -> list[Fold]:
    if official_folds_path:
        return load_folds(official_folds_path)
    return make_repeated_folds(labels, n_splits=n_splits, n_repeats=1, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eeg_cte_real.yaml")
    parser.add_argument("--output", default="results/deep_models")
    parser.add_argument("--model", default=None, choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--official-folds", default=None, help="Path to official CTE-Net folds JSON"
    )
    parser.add_argument(
        "--device", default="cuda", choices=["auto", "cpu", "cuda"],
        help="Device for deep models. 'cuda' is required; CPU fallback is prohibited.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.training.n_splits = args.folds
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.lr = args.lr
    config.training.device = args.device
    config.training.seed = args.seed

    records = read_manifest(config.data.root, config.data.manifest)
    bundle = build_feature_bundle(records, config.data, config.features, verbose=True)
    print(
        f"Loaded {len(bundle.subject_labels)} subjects | "
        f"{len(bundle.labels)} epochs"
    )

    labels = read_subject_labels(Path(config.data.root) / config.data.manifest)
    folds = build_folds(labels, n_splits=args.folds, seed=args.seed, official_folds_path=args.official_folds)
    print(f"Using {len(folds)} subject-wise folds")

    folds_dir = Path(args.output) / "folds"
    save_folds(folds, folds_dir / "folds.json", metadata={"n_splits": args.folds, "seed": args.seed})

    models_to_run = (
        list(MODELS.keys()) if args.model == "all" or args.model is None else [args.model]
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for model_key in models_to_run:
        model_name = MODELS[model_key]
        print(f"\n=== {model_key} -> {model_name} ===")
        config.model.name = model_name

        try:
            result = run_cross_validation(bundle, config)

            oof = result["oof"]
            subject_metrics = None
            from src.evaluate import binary_metrics

            subject_metrics = binary_metrics(
                oof["y_true"].to_numpy(),
                oof["y_prob"].to_numpy(),
                y_pred=oof["y_pred"].to_numpy(),
                n_boot=2000,
                seed=args.seed,
            )

            row = {
                "experiment": {"eegnet": "E6", "ctenet_original": "E7", "ctenet_inspired": "E8"}[model_key],
                "model": model_key,
                "architecture": model_name,
                "n_subjects": int(len(bundle.subject_labels)),
                "n_epochs": int(len(bundle.labels)),
                "n_folds": len(folds),
                "accuracy": float(subject_metrics["accuracy"]),
                "precision": float(subject_metrics["precision"]),
                "recall": float(subject_metrics["recall"]),
                "f1": float(subject_metrics["f1"]),
                "roc_auc": float(subject_metrics["auroc"]),
                "pr_auc": float(subject_metrics["auprc"]),
                "sensitivity": float(subject_metrics["sensitivity"]),
                "specificity": float(subject_metrics["specificity"]),
                "brier": float(subject_metrics["brier"]),
                "ece": float(subject_metrics["ece"]),
                "participant_level": True,
                "real_data": True,
                "model_dir": str(result.get("output_dir", "")),
            }
            rows.append(row)

            print(
                f"{model_key:18s} | AUROC {subject_metrics['auroc']:.3f} | "
                f"accuracy {subject_metrics['accuracy']:.3f} | "
                f"sens {subject_metrics['sensitivity']:.3f} | "
                f"spec {subject_metrics['specificity']:.3f}"
            )

            # Save OOF predictions.
            oof.to_csv(output_dir / f"{model_key}_oof_predictions.csv", index=False)
            result["folds"].to_csv(output_dir / f"{model_key}_fold_metrics.csv", index=False)
            result["history"].to_csv(output_dir / f"{model_key}_history.csv", index=False)

            (output_dir / f"{model_key}_metrics.json").write_text(
                json.dumps(
                    {
                        "model": model_key,
                        "architecture": model_name,
                        "n_subjects": int(len(bundle.subject_labels)),
                        "n_folds": len(folds),
                        "accuracy": float(subject_metrics["accuracy"]),
                        "roc_auc": float(subject_metrics["auroc"]),
                        "sensitivity": float(subject_metrics["sensitivity"]),
                        "specificity": float(subject_metrics["specificity"]),
                        "f1": float(subject_metrics["f1"]),
                        "participant_level": True,
                        "real_data": True,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        except Exception as exc:  # pragma: no cover - report and continue
            print(f"  {model_key}: FAILED -- {exc}")
            rows.append(
                {
                    "experiment": {"eegnet": "E6", "ctenet_original": "E7", "ctenet_inspired": "E8"}.get(model_key, "?"),
                    "model": model_key,
                    "error": str(exc),
                    "real_data": True,
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n=== Deep model summary (E6-E8) ===")
    print(
        frame[["experiment", "model", "roc_auc", "accuracy", "sensitivity", "specificity"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
