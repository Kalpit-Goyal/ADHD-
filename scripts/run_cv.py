"""Run subject-wise cross-validation for the ADHD EEG pipeline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, load_config
from src.data import build_feature_bundle, read_manifest
from src.evaluate import binary_metrics, format_metrics
from src.plots import (
    plot_calibration,
    plot_confusion,
    plot_fold_metrics,
    plot_history,
    plot_roc_pr,
)
from src.train import run_cross_validation


def apply_quick_mode(config: Config) -> None:
    """Small, fast settings for a laptop smoke test."""
    config.training.n_splits = min(3, config.training.n_splits)
    config.training.epochs = min(12, config.training.epochs)
    config.training.patience = min(5, config.training.patience)
    config.training.batch_size = min(16, config.training.batch_size)
    config.model.d_model = min(32, config.model.d_model)
    config.model.n_heads = 4 if config.model.d_model % 4 == 0 else 2
    config.model.n_layers = 1
    config.output.dir = str(Path(config.output.dir) / "quick")


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _clean_json(value):
    """Recursively replace non-finite floats with None for valid JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_clean_json(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eeg_cte.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=None, choices=["cte_net", "eegnet"])
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--quick", action="store_true", help="Fast smoke-test settings")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat the whole subject-wise CV with different seeds and average",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.data_root:
        config.data.root = args.data_root
    if args.model:
        config.model.name = args.model
        if args.output is None:
            config.output.dir = f"results/{args.model}"
    if args.output:
        config.output.dir = args.output
    if args.device:
        config.training.device = args.device
    if args.quick:
        apply_quick_mode(config)

    output_dir = Path(config.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Config: {args.config}")
    print(f"Data root: {config.data.root}")
    print(f"Model: {config.model.name}")
    print(f"Output: {output_dir}")

    if "synthetic" in Path(config.data.root).name.lower():
        print(
            "\nWARNING: synthetic data. These results test the pipeline, "
            "not real ADHD detection.\n"
        )

    records = read_manifest(config.data.root, config.data.manifest)
    class_counts = pd.Series([record.label for record in records]).value_counts()
    print(
        f"Manifest: {len(records)} subjects | "
        f"ADHD {int(class_counts.get(1, 0))} | "
        f"control {int(class_counts.get(0, 0))}"
    )

    bundle = build_feature_bundle(records, config.data, config.features, verbose=True)
    print(
        f"Built {len(bundle.labels)} epochs | "
        f"{bundle.n_node_features} node features | "
        f"adjacency {bundle.adjacency.shape[1]} x {bundle.adjacency.shape[2]}"
    )

    result = run_cross_validation(bundle, config)

    repeats = max(1, int(args.repeats))
    repeat_results = [result]
    base_seed = config.training.seed
    for repeat in range(1, repeats):
        config.training.seed = base_seed + repeat * 1000
        print(
            f"\n=== Repeat {repeat + 1}/{repeats} "
            f"(seed {config.training.seed}) ==="
        )
        repeat_results.append(run_cross_validation(bundle, config))
    config.training.seed = base_seed

    repeat_metrics = []
    for repeat, repeat_result in enumerate(repeat_results):
        repeat_oof = repeat_result["oof"]
        metrics = binary_metrics(
            repeat_oof["y_true"].to_numpy(),
            repeat_oof["y_prob"].to_numpy(),
            y_pred=repeat_oof["y_pred"].to_numpy(),
            n_boot=500,
            seed=base_seed + repeat,
        )
        repeat_metrics.append(
            {
                "repeat": repeat + 1,
                "auroc": metrics["auroc"],
                "accuracy": metrics["accuracy"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
            }
        )

    if repeats > 1:
        oof_all = pd.concat(
            [item["oof"] for item in repeat_results], ignore_index=True
        )
        result["oof"] = (
            oof_all.groupby("subject_id", as_index=False)
            .agg(
                y_true=("y_true", "first"),
                y_prob=("y_prob", "mean"),
                y_pred=("y_pred", "mean"),
            )
        )
        result["oof"]["fold"] = 0
        result["oof"]["threshold"] = float("nan")
        result["oof"]["y_pred"] = (result["oof"]["y_pred"] >= 0.5).astype(int)

    subject_metrics = binary_metrics(
        result["oof"]["y_true"].to_numpy(),
        result["oof"]["y_prob"].to_numpy(),
        y_pred=result["oof"]["y_pred"].to_numpy(),
        n_boot=2000,
        seed=base_seed,
    )
    epoch_metrics = binary_metrics(
        result["epoch"]["y_true"].to_numpy(),
        result["epoch"]["y_prob"].to_numpy(),
        n_boot=1000,
        seed=config.training.seed,
    )

    summary = {
        "model": config.model.name,
        "connectivity": config.data.connectivity,
        "n_subjects": int(len(bundle.subject_labels)),
        "n_epochs": int(len(bundle.labels)),
        "repeats": repeats,
        "decision_rule": (
            "per-fold Youden threshold chosen on validation subjects"
            if repeats == 1
            else "majority vote across repeats; per-fold Youden thresholds"
        ),
        "subject_level": subject_metrics,
        "epoch_level": epoch_metrics,
        "repeat_metrics": repeat_metrics,
        "folds": json.loads(result["folds"].to_json(orient="records")),
    }

    result["oof"].to_csv(output_dir / "oof_predictions.csv", index=False)
    result["epoch"].to_csv(output_dir / "epoch_predictions.csv", index=False)
    result["folds"].to_csv(output_dir / "fold_metrics.csv", index=False)
    result["history"].to_csv(output_dir / "training_history.csv", index=False)

    (output_dir / "metrics.json").write_text(
        json.dumps(_clean_json(summary), indent=2, default=_json_default),
        encoding="utf-8",
    )
    (output_dir / "config_used.yaml").write_text(
        yaml.safe_dump(asdict(config), sort_keys=False),
        encoding="utf-8",
    )

    plot_roc_pr(
        result["oof"]["y_true"].to_numpy(),
        result["oof"]["y_prob"].to_numpy(),
        output_dir,
    )
    plot_calibration(
        result["oof"]["y_true"].to_numpy(),
        result["oof"]["y_prob"].to_numpy(),
        output_dir,
    )
    plot_confusion(
        result["oof"]["y_true"].to_numpy(),
        result["oof"]["y_prob"].to_numpy(),
        output_dir,
        y_pred=result["oof"]["y_pred"].to_numpy(),
    )
    plot_history(result["history"], output_dir)
    plot_fold_metrics(result["folds"], output_dir)

    print("\nSubject-level (primary):")
    print("  " + format_metrics(subject_metrics))
    if repeats > 1:
        accuracies = [item["accuracy"] for item in repeat_metrics]
        aurocs = [item["auroc"] for item in repeat_metrics]
        print(
            f"\nAcross {repeats} repeats: "
            f"AUROC {np.mean(aurocs):.3f} ± {np.std(aurocs):.3f} | "
            f"accuracy {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}"
        )
    print("\nEpoch-level (correlated; descriptive only):")
    print("  " + format_metrics(epoch_metrics))
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
