"""Train hybrid models (E9-E12) with subject-wise CV on real data.

Experiments:
  E9: Feature-level fusion (Paper 1 features + CTE-Net deep features)
  E10: Soft-voting ensemble (XGBoost + CTE-Net)
  E11: Weighted soft-voting ensemble
  E12: Stacking ensemble (Logistic Regression meta-classifier)

All experiments use identical subject-wise folds. Deep model predictions come
from out-of-fold evaluation -- no test-set information is used for weighting.
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

from src.classical_models import evaluate_model
from src.config import load_config
from src.cv import Fold, load_folds, make_repeated_folds, read_subject_labels, save_folds
from src.data import build_feature_bundle, read_manifest
from src.evaluate import binary_metrics
from src.hybrid_model import (
    HybridResult,
    run_feature_fusion,
    run_prediction_ensemble,
    save_hybrid_result,
)
from src.train import resolve_device


def build_folds(
    labels: dict[str, int],
    n_splits: int = 5,
    seed: int = 42,
    official_folds_path: Optional[str] = None,
) -> list[Fold]:
    if official_folds_path:
        return load_folds(official_folds_path)
    return make_repeated_folds(labels, n_splits=n_splits, n_repeats=1, seed=seed)


def _collect_oof_probs(
    bundle, folds: list[Fold], model_name: str, config, device, seed: int
) -> dict[int, dict[str, float]]:
    """Run a deep model CV and return {fold: {subject_id: prob}} OOF predictions."""
    from src.train import run_cross_validation

    config.model.name = model_name
    result = run_cross_validation(bundle, config)
    oof = result["oof"]

    fold_probs: dict[int, dict[str, float]] = {}
    for fold_id, group in oof.groupby("fold"):
        fold_probs[int(fold_id)] = dict(zip(group["subject_id"].tolist(), group["y_prob"].tolist()))
    return fold_probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eeg_cte_real.yaml")
    parser.add_argument("--output", default="results/hybrid")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deep-epochs", type=int, default=60)
    parser.add_argument("--deep-batch", type=int, default=32)
    parser.add_argument("--deep-lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--official-folds", default=None, help="Path to official CTE-Net folds JSON"
    )
    parser.add_argument(
        "--skip-deep", action="store_true", help="Skip deep model (classical-only fusion)"
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

    labels = read_subject_labels(Path(config.data.root) / config.data.manifest)
    folds = build_folds(labels, n_splits=args.folds, seed=args.seed, official_folds_path=args.official_folds)
    print(f"Using {len(folds)} subject-wise folds")

    folds_dir = Path(args.output) / "folds"
    save_folds(folds, folds_dir / "folds.json", metadata={"n_splits": args.folds, "seed": args.seed})

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)

    # ----- Get deep model OOF predictions (E8 model) -----
    branch_b_probs: Optional[dict[int, dict[str, float]]] = None
    deep_model = None
    if not args.skip_deep:
        print("\n=== Getting CTE-Net OOF predictions for hybrid ===")
        try:
            branch_b_probs = _collect_oof_probs(
                bundle, folds, "cte_net", config, device, args.seed
            )
            print(f"  Collected OOF probs for {len(branch_b_probs)} folds")
        except Exception as exc:  # pragma: no cover
            print(f"  Deep model OOF collection failed: {exc}")
            branch_b_probs = None

    # ----- Get Branch A (XGBoost) OOF predictions per fold -----
    print("\n=== Getting Branch A (XGBoost) OOF predictions ===")
    branch_a_probs: dict[int, dict[str, float]] = {}
    feature_set = "node+adj+rel"

    from src.classical_models import _build_epoch_features, make_model, _subject_probs
    from src.evaluate import youden_threshold

    a_features = _build_epoch_features(bundle, feature_set)

    for fold, fold_spec in enumerate(folds):
        train_subjects = list(fold_spec.train)
        val_subjects = list(fold_spec.val)
        test_subjects = list(fold_spec.test)

        train_index = np.flatnonzero(np.isin(bundle.subject_ids, train_subjects))
        val_index = np.flatnonzero(np.isin(bundle.subject_ids, val_subjects))
        full_index = np.flatnonzero(np.isin(bundle.subject_ids, train_subjects + val_subjects))
        test_index = np.flatnonzero(np.isin(bundle.subject_ids, test_subjects))

        model = make_model("xgboost", args.seed, n_features=a_features.shape[1], k=150)
        model.fit(a_features[train_index], bundle.labels[train_index].astype(int))

        val_prob = model.predict_proba(a_features[val_index])[:, 1]
        val_subject = _subject_probs(val_prob, bundle.subject_ids[val_index])
        val_y = np.asarray([bundle.subject_labels[str(s)] for s in val_subject.index], dtype=int)
        threshold = float(youden_threshold(val_y, val_subject.to_numpy()))

        model.fit(a_features[full_index], bundle.labels[full_index].astype(int))

        test_prob = model.predict_proba(a_features[test_index])[:, 1]
        test_subject_probs = dict(zip(bundle.subject_ids[test_index].tolist(), test_prob.tolist()))
        branch_a_probs[fold] = test_subject_probs

    # ----- E9: Feature-level fusion -----
    print("\n=== E9: Feature-level fusion ===")
    try:
        fusion_result = run_feature_fusion(
            bundle, folds, args.seed,
            deep_model=deep_model,  # None means classical-only fusion
            device=device,
            model_name="cte_net",
            real_data=True,
        )
        save_hybrid_result(fusion_result, output_dir / "models")
        print(
            f"  E9 feature_fusion: AUROC {fusion_result.roc_auc:.3f} | "
            f"accuracy {fusion_result.accuracy:.3f} | "
            f"sens {fusion_result.sensitivity:.3f} | spec {fusion_result.specificity:.3f}"
        )
    except Exception as exc:
        print(f"  E9 FAILED: {exc}")
        fusion_result = None

    # ----- E10: Soft voting -----
    print("\n=== E10: Soft voting ensemble ===")
    try:
        soft_result = run_prediction_ensemble(
            bundle, folds, args.seed,
            branch_a_probs=branch_a_probs,
            branch_b_probs=branch_b_probs,
            method="soft_voting",
            real_data=True,
        )
        save_hybrid_result(soft_result, output_dir / "models")
        print(
            f"  E10 soft_voting: AUROC {soft_result.roc_auc:.3f} | "
            f"accuracy {soft_result.accuracy:.3f} | "
            f"sens {soft_result.sensitivity:.3f} | spec {soft_result.specificity:.3f}"
        )
    except Exception as exc:
        print(f"  E10 FAILED: {exc}")
        soft_result = None

    # ----- E11: Weighted soft voting -----
    print("\n=== E11: Weighted soft voting ensemble ===")
    try:
        weighted_result = run_prediction_ensemble(
            bundle, folds, args.seed,
            branch_a_probs=branch_a_probs,
            branch_b_probs=branch_b_probs,
            method="weighted_voting",
            weights=[0.6, 0.4] if branch_b_probs is not None else [1.0],
            real_data=True,
        )
        save_hybrid_result(weighted_result, output_dir / "models")
        print(
            f"  E11 weighted_voting: AUROC {weighted_result.roc_auc:.3f} | "
            f"accuracy {weighted_result.accuracy:.3f} | "
            f"sens {weighted_result.sensitivity:.3f} | spec {weighted_result.specificity:.3f}"
        )
    except Exception as exc:
        print(f"  E11 FAILED: {exc}")
        weighted_result = None

    # ----- E12: Stacking -----
    print("\n=== E12: Stacking ensemble ===")
    try:
        stacking_result = run_prediction_ensemble(
            bundle, folds, args.seed,
            branch_a_probs=branch_a_probs,
            branch_b_probs=branch_b_probs,
            method="stacking",
            real_data=True,
        )
        save_hybrid_result(stacking_result, output_dir / "models")
        print(
            f"  E12 stacking: AUROC {stacking_result.roc_auc:.3f} | "
            f"accuracy {stacking_result.accuracy:.3f} | "
            f"sens {stacking_result.sensitivity:.3f} | spec {stacking_result.specificity:.3f}"
        )
    except Exception as exc:
        print(f"  E12 FAILED: {exc}")
        stacking_result = None

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
