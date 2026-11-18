"""Train and persist the deployable ADHD screening model.

This mirrors the best single model from ``scripts/final_validation.py``:
SelectKBest(150) -> StandardScaler -> RBF-SVM on per-epoch
node + connectivity + relative-power features.

The two demo recordings used by the web app are excluded from training so
their predictions are genuinely out-of-sample. The deployed artifact is a
research/education model, not a diagnostic device.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import FeatureBundle, build_feature_bundle, read_manifest

CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

# Demo recordings shown in the web app (held out from training).
DEMO_SUBJECTS = {"v10p": "adhd", "v107": "control"}


def build_epoch_features(bundle: FeatureBundle) -> np.ndarray:
    """Same feature layout used in final_validation.py."""
    n = len(bundle.labels)
    node = bundle.node_features.reshape(n, -1)

    probabilities = np.exp(bundle.node_features[:, :, :5])
    relative = probabilities / probabilities.sum(axis=2, keepdims=True)
    relative = relative.reshape(n, -1)

    rows, columns = np.triu_indices(bundle.n_channels, k=1)
    edges = bundle.adjacency[:, rows, columns]

    return np.concatenate([node, edges, relative], axis=1)


def make_feature_names(n_channels: int, bands: list[str]) -> list[str]:
    names: list[str] = []
    # Node features: channel-major, bands then theta/beta ratio.
    for channel in CHANNELS[:n_channels]:
        for band in bands:
            names.append(f"{channel} | {band} power")
        names.append(f"{channel} | theta/beta ratio")

    # Directed connectivity edges (upper triangle, row-major).
    rows, columns = np.triu_indices(n_channels, k=1)
    for target, source in zip(rows, columns):
        names.append(
            f"{CHANNELS[target]} <-> {CHANNELS[source]} connectivity"
        )

    # Relative band power: channel-major, five bands.
    for channel in CHANNELS[:n_channels]:
        for band in bands:
            names.append(f"{channel} | {band} relative power")

    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eeg_cte_real.yaml")
    parser.add_argument("--output", default="models")
    parser.add_argument("--k", type=int, default=150)
    parser.add_argument(
        "--include-demo-subjects",
        action="store_true",
        help="Train on all 121 subjects (demo predictions become in-sample).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    records = [
        record
        for record in read_manifest(config.data.root, config.data.manifest)
        if args.include_demo_subjects or record.subject_id not in DEMO_SUBJECTS
    ]

    print(f"Training on {len(records)} subjects ...")
    bundle = build_feature_bundle(records, config.data, config.features, verbose=True)
    features = build_epoch_features(bundle)
    labels = bundle.labels.astype(int)

    print(
        f"Feature matrix: {features.shape[0]} epochs x {features.shape[1]} features "
        f"from {len(bundle.subject_labels)} subjects"
    )

    model = make_pipeline(
        SelectKBest(f_classif, k=min(args.k, features.shape[1])),
        StandardScaler(),
        SVC(
            C=1.0,
            gamma=0.01,
            probability=True,
            class_weight="balanced",
            random_state=0,
        ),
    )
    model.fit(features, labels)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "adhd_svm_select150.joblib"
    joblib.dump(model, artifact_path)

    # Persist the cached feature matrix for future experiments.
    np.savez_compressed(
        output_dir / "feature_cache.npz",
        features=features.astype(np.float32),
        labels=labels,
        subject_ids=bundle.subject_ids,
        node_features=bundle.node_features,
        adjacency=bundle.adjacency,
    )

    names = make_feature_names(bundle.n_channels, BANDS)
    selector = model.named_steps["selectkbest"]
    selected = [names[i] for i in selector.get_support(indices=True)]

    metrics = {}
    ensemble_path = ROOT / "results" / "final_validation" / "ensemble_summary.json"
    if ensemble_path.exists():
        ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
        if ensemble.get("results"):
            first = ensemble["results"][0]
            metrics = {
                "auroc": first.get("auroc"),
                "auroc_ci": first.get("auroc_ci"),
                "accuracy": first.get("accuracy"),
                "accuracy_ci": first.get("accuracy_ci"),
                "sensitivity": first.get("sensitivity"),
                "specificity": first.get("specificity"),
                "n_subjects": first.get("n"),
            }

    metadata = {
        "model": "SelectKBest(150) -> StandardScaler -> RBF-SVM",
        "artifact": str(artifact_path.as_posix()),
        "threshold": 0.5,
        "positive_class": "ADHD-like pattern",
        "n_train_subjects": len(bundle.subject_labels),
        "n_train_epochs": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "k": int(selector.k),
        "excluded_subjects": sorted(DEMO_SUBJECTS),
        "channels": CHANNELS,
        "bands": {
            name: list(limits)
            for name, limits in config.features.bands.items()
        },
        "selected_features": selected,
        "metrics": metrics,
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "notes": (
            "Research prototype. Subject-wise validated on 121 children from one "
            "site. Not a diagnostic device."
        ),
    }
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Saved model      : {artifact_path}")
    print(f"Saved metadata   : {output_dir / 'model_metadata.json'}")
    print(f"Selected features: {len(selected)}")


if __name__ == "__main__":
    main()
