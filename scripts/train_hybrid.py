"""Train hybrid models (E9-E12) with subject-wise CV on real data.

Experiments:
  E9: Feature-level fusion (Paper 1 features + CTE-Net deep features)
  E10: Soft-voting ensemble (XGBoost + CTE-Net)
  E11: Weighted soft-voting ensemble
  E12: Stacking ensemble (Logistic Regression meta-classifier, nested CV)

All experiments use identical subject-wise folds. Deep model is trained
per fold on training subjects only, with validation for early stopping.
No test-set information is used for weighting or threshold tuning.

The deep branch is required by default. Use --skip-deep for explicitly
labelled classical-only ablations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classical_models import (
    _build_epoch_features,
    _subject_probs,
    evaluate_model,
    make_model,
)
from src.config import load_config
from src.cv import Fold, load_folds, make_repeated_folds, read_subject_labels, save_folds
from src.data import build_feature_bundle, read_manifest
from src.evaluate import binary_metrics, youden_threshold
from src.hybrid_model import (
    HybridResult,
    run_feature_fusion,
    run_prediction_ensemble,
    save_hybrid_result,
)
from src.models import build_model as build_deep_model
from src.train import aggregate_subject_probs, resolve_device


def build_folds(
    labels: dict[str, int],
    n_splits: int = 5,
    seed: int = 42,
    official_folds_path: Optional[str] = None,
) -> list[Fold]:
    if official_folds_path:
        return load_folds(official_folds_path)
    return make_repeated_folds(labels, n_splits=n_splits, n_repeats=1, seed=seed)


def train_deep_model_per_fold(
    bundle,
    train_subjects: list[str],
    val_subjects: list[str],
    test_subjects: list[str],
    config,
    device: torch.device,
    model_name: str,
    seed: int,
    fold: int,
) -> dict[str, float]:
    """Train a deep model on train subjects only and return test subject probs.

    Uses validation subjects for early stopping and best-checkpoint selection.
    Test subjects are never seen during training or tuning.
    """
    from torch.utils.data import DataLoader

    from src.data import EpochDataset
    from src.train import train_one_fold

    # Use train_one_fold from src.train for consistent training procedure.
    # It returns test_probs, test_subject_ids, test_epoch_indices, etc.
    result = train_one_fold(
        bundle,
        np.asarray(train_subjects),
        np.asarray(val_subjects),
        np.asarray(test_subjects),
        config,
        device,
        fold,
    )

    # Aggregate epoch probabilities to subject level using mean.
    test_subject_probs = aggregate_subject_probs(
        result["test_subject_ids"], result["test_probs"]
    )

    return {
        str(sid): float(prob)
        for sid, prob in test_subject_probs.items()
    }, result["threshold"]


def make_deep_model_factory(
    config, device: torch.device, model_name: str
) -> Callable[[], torch.nn.Module]:
    """Return a factory that builds a fresh deep model for a given fold.

    The factory constructs the model from config and moves it to device.
    Each call returns a new untrained model instance.
    """
    def factory() -> torch.nn.Module:
        config.model.name = model_name
        model = build_deep_model(
            config.model,
            n_channels=bundle_n_channels(config),
            n_features=bundle_n_features(config),
            n_times=bundle_n_times(config),
        )
        return model.to(device)
    return factory


def bundle_n_channels(config) -> int:
    return int(getattr(config.data, "n_channels", 19))


def bundle_n_features(config) -> int:
    return int(getattr(config.data, "n_features", 6))


def bundle_n_times(config) -> int:
    return int(getattr(config.data, "epoch_seconds", 5.0)
                 * getattr(config.data, "fs", 128))


def collect_branch_a_probs(
    bundle,
    folds: list[Fold],
    seed: int,
    feature_set: str = "node+adj+rel",
    k: int = 150,
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, float]]]:
    """Collect XGBoost out-of-fold test and validation probabilities per fold.

    Returns (test_probs, val_probs) where each is
    {fold_index: {subject_id: participant_level_prob}}.
    Uses aggregate_subject_probs to average epoch probabilities per subject
    (never overwrites with dict zip).
    """
    branch_a_test_probs: dict[int, dict[str, float]] = {}
    branch_a_val_probs: dict[int, dict[str, float]] = {}
    a_features = _build_epoch_features(bundle, feature_set)

    for fold, fold_spec in enumerate(folds):
        train_subjects = list(fold_spec.train)
        val_subjects = list(fold_spec.val)
        test_subjects = list(fold_spec.test)

        train_index = np.flatnonzero(np.isin(bundle.subject_ids, train_subjects))
        val_index = np.flatnonzero(np.isin(bundle.subject_ids, val_subjects))
        full_index = np.flatnonzero(
            np.isin(bundle.subject_ids, train_subjects + val_subjects)
        )
        test_index = np.flatnonzero(np.isin(bundle.subject_ids, test_subjects))

        model = make_model("xgboost", seed, n_features=a_features.shape[1], k=k)
        model.fit(
            a_features[train_index], bundle.labels[train_index].astype(int)
        )

        # Validation predictions (participant-level) for threshold tuning.
        val_prob = model.predict_proba(a_features[val_index])[:, 1]
        val_subject_prob = aggregate_subject_probs(
            bundle.subject_ids[val_index], val_prob
        )
        branch_a_val_probs[fold] = {
            str(sid): float(prob) for sid, prob in val_subject_prob.items()
        }

        # Tune threshold on validation subjects.
        val_y = np.asarray(
            [bundle.subject_labels[str(s)] for s in branch_a_val_probs[fold]],
            dtype=int,
        )
        val_p = np.asarray(
            [branch_a_val_probs[fold][s] for s in branch_a_val_probs[fold]],
            dtype=float,
        )
        threshold = float(youden_threshold(val_y, val_p))

        # Refit on train+val.
        model.fit(
            a_features[full_index], bundle.labels[full_index].astype(int)
        )

        # Predict on test and aggregate to participant level.
        test_prob = model.predict_proba(a_features[test_index])[:, 1]
        test_subject_prob = aggregate_subject_probs(
            bundle.subject_ids[test_index], test_prob
        )
        branch_a_test_probs[fold] = {
            str(sid): float(prob) for sid, prob in test_subject_prob.items()
        }

    return branch_a_test_probs, branch_a_val_probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eeg_cte_real.yaml")
    parser.add_argument("--output", default="results/hybrid")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deep-epochs", type=int, default=30)
    parser.add_argument("--deep-batch", type=int, default=32)
    parser.add_argument("--deep-lr", type=float, default=1e-3)
    parser.add_argument(
        "--device", default="cuda", choices=["auto", "cpu", "cuda"],
        help="Device for deep models. 'cuda' is required; CPU fallback is prohibited.",
    )
    parser.add_argument(
        "--deep-model", default="ctenet_original",
        choices=["ctenet_original", "cte_net"],
        help="Deep model architecture for hybrid experiments. "
        "ctenet_original = paper reconstruction (E7); cte_net = inspired graph model (E8).",
    )
    parser.add_argument(
        "--official-folds", default=None, help="Path to official CTE-Net folds JSON"
    )
    parser.add_argument(
        "--skip-deep", action="store_true", help="Run classical-only ablations (not saved as E9-E12)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.training.n_splits = args.folds
    config.training.epochs = args.deep_epochs
    config.training.batch_size = args.deep_batch
    config.training.lr = args.deep_lr
    config.training.device = args.device
    config.training.seed = args.seed

    records = read_manifest(config.data.root, config.data.manifest)
    bundle = build_feature_bundle(records, config.data, config.features, verbose=True)
    print(
        f"Loaded {len(bundle.subject_labels)} subjects | "
        f"{len(bundle.labels)} epochs"
    )

    # Verify real data, not synthetic.
    if str(config.data.root).endswith("synthetic"):
        raise ValueError("Refusing to run hybrid experiments on synthetic data")

    labels = read_subject_labels(Path(config.data.root) / config.data.manifest)
    folds = build_folds(labels, n_splits=args.folds, seed=args.seed, official_folds_path=args.official_folds)
    print(f"Using {len(folds)} subject-wise folds")

    folds_dir = Path(args.output) / "folds"
    save_folds(folds, folds_dir / "folds.json", metadata={"n_splits": args.folds, "seed": args.seed})

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    print(f"Device: {device} | CUDA available: {torch.cuda.is_available()}")

    # ----- Branch A: XGBoost OOF probabilities per fold -----
    print("\n=== Branch A (XGBoost) OOF predictions ===")
    branch_a_probs, branch_a_val_probs = collect_branch_a_probs(bundle, folds, args.seed)

    # ----- Branch B: Deep model OOF probabilities per fold -----
    branch_b_probs: Optional[dict[int, dict[str, float]]] = None
    deep_model_factory: Optional[Callable[[], torch.nn.Module]] = None

    if args.skip_deep:
        print("\n=== SKIP-DEEP: Classical-only ablation (not E9-E12) ===")
    else:
        print("\n=== Training deep model per fold ===")
        deep_model_factory = make_deep_model_factory(config, device, args.deep_model)

        # Train deep model per fold using the SAME folds as the hybrid pipeline.
        branch_b_probs = {}
        deep_thresholds: dict[int, float] = {}

        for fold, fold_spec in enumerate(folds):
            train_subjects = list(fold_spec.train)
            val_subjects = list(fold_spec.val)
            test_subjects = list(fold_spec.test)

            fold_probs, threshold = train_deep_model_per_fold(
                bundle,
                train_subjects,
                val_subjects,
                test_subjects,
                config,
                device,
                args.deep_model,
                args.seed,
                fold,
            )
            branch_b_probs[fold] = fold_probs
            deep_thresholds[fold] = threshold

            print(
                f"  Fold {fold+1}/{len(folds)}: "
                f"train {len(train_subjects)} | "
                f"val {len(val_subjects)} | "
                f"test {len(test_subjects)} | "
                f"threshold {threshold:.3f}"
            )

        print(f"  Collected deep OOF probs for {len(branch_b_probs)} folds")

    # ----- E9: Feature-level fusion -----
    print("\n=== E9: Feature-level fusion ===")
    fusion_result = run_feature_fusion(
        bundle,
        folds,
        args.seed,
        deep_model_factory=deep_model_factory,
        deep_config=None,
        device=device,
        model_name=args.deep_model,
        real_data=True,
    )
    save_hybrid_result(fusion_result, output_dir / "models")
    print(
        f"  E9 feature_fusion: AUROC {fusion_result.roc_auc:.3f} | "
        f"accuracy {fusion_result.accuracy:.3f} | "
        f"sens {fusion_result.sensitivity:.3f} | spec {fusion_result.specificity:.3f}"
    )

    # ----- E10: Soft voting -----
    print("\n=== E10: Soft voting ensemble ===")
    soft_result = run_prediction_ensemble(
        bundle, folds, args.seed,
        branch_a_probs=branch_a_probs,
        branch_a_val_probs=branch_a_val_probs,
        branch_b_probs=branch_b_probs,
        method="soft_voting",
        real_data=True,
        deep_model_factory=deep_model_factory,
        device=device,
        model_name=args.deep_model,
    )
    save_hybrid_result(soft_result, output_dir / "models")
    print(
        f"  E10 soft_voting: AUROC {soft_result.roc_auc:.3f} | "
        f"accuracy {soft_result.accuracy:.3f} | "
        f"sens {soft_result.sensitivity:.3f} | spec {soft_result.specificity:.3f}"
    )

    # ----- E11: Weighted soft voting -----
    print("\n=== E11: Weighted soft voting ensemble ===")
    weighted_result = run_prediction_ensemble(
        bundle, folds, args.seed,
        branch_a_probs=branch_a_probs,
        branch_a_val_probs=branch_a_val_probs,
        branch_b_probs=branch_b_probs,
        method="weighted_voting",
        weights=None,
        real_data=True,
        deep_model_factory=deep_model_factory,
        device=device,
        model_name=args.deep_model,
    )
    save_hybrid_result(weighted_result, output_dir / "models")
    print(
        f"  E11 weighted_voting: AUROC {weighted_result.roc_auc:.3f} | "
        f"accuracy {weighted_result.accuracy:.3f} | "
        f"sens {weighted_result.sensitivity:.3f} | spec {weighted_result.specificity:.3f}"
    )

    # ----- E12: Stacking (proper nested CV) -----
    print("\n=== E12: Stacking ensemble ===")
    stacking_result = run_prediction_ensemble(
        bundle, folds, args.seed,
        branch_a_probs=branch_a_probs,
        branch_a_val_probs=branch_a_val_probs,
        branch_b_probs=branch_b_probs,
        method="stacking",
        real_data=True,
        deep_model_factory=deep_model_factory,
        device=device,
        model_name=args.deep_model,
    )
    save_hybrid_result(stacking_result, output_dir / "models")
    print(
        f"  E12 stacking: AUROC {stacking_result.roc_auc:.3f} | "
        f"accuracy {stacking_result.accuracy:.3f} | "
        f"sens {stacking_result.sensitivity:.3f} | spec {stacking_result.specificity:.3f}"
    )

    # ----- Summary -----
    rows = []
    for r in [fusion_result, soft_result, weighted_result, stacking_result]:
        if r is None:
            continue
        rows.append(
            {
                "experiment": r.experiment,
                "branch_a_model": r.branch_a_model,
                "branch_b_model": r.branch_b_model,
                "n_subjects": r.n_subjects,
                "n_folds": r.n_folds,
                "accuracy": r.accuracy,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "roc_auc": r.roc_auc,
                "pr_auc": r.pr_auc,
                "sensitivity": r.sensitivity,
                "specificity": r.specificity,
                "brier": r.brier,
                "ece": r.ece,
                "participant_level": True,
                "real_data": True,
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n=== Hybrid model summary (E9-E12) ===")
    print(
        frame[["experiment", "branch_a_model", "branch_b_model", "roc_auc", "accuracy", "sensitivity", "specificity"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
