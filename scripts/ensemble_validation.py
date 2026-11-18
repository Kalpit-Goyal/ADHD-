import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import binary_metrics

p = pd.read_csv("results/final_validation/predictions.csv")
y = p.groupby("subject_id").y_true.first()
results = []


def summarize(name, configs):
    preds = []
    probs = []
    for config in configs:
        frame = p[p.config == config]
        preds.append(
            frame.pivot_table(
                index="subject_id", columns="repeat", values="y_pred"
            ).mean(axis=1)
        )
        probs.append(
            frame.pivot_table(
                index="subject_id", columns="repeat", values="y_prob"
            ).mean(axis=1)
        )
    vote = (pd.concat(preds, axis=1).mean(axis=1) >= 0.5).astype(int).reindex(y.index)
    prob = pd.concat(probs, axis=1).mean(axis=1).reindex(y.index)
    m = binary_metrics(
        y.to_numpy(),
        prob.to_numpy(),
        y_pred=vote.to_numpy(),
        n_boot=2000,
        seed=42,
    )
    print(
        f"{name:42s} acc {m['accuracy']:.3f} CI {m['accuracy_ci']} | "
        f"AUROC {m['auroc']:.3f} CI {m['auroc_ci']} | "
        f"sens {m['sensitivity']:.3f} spec {m['specificity']:.3f}"
    )
    results.append({"ensemble": name, "configs": configs, **m})


summarize("select150 only", ["epoch_select150_svm"])
summarize(
    "select150 + c0.3 + c1g0.01",
    ["epoch_select150_svm", "epoch_svm_c0.3", "epoch_svm_c1_g0.01"],
)
summarize(
    "all epoch SVM",
    [
        "epoch_select150_svm",
        "epoch_svm_c0.3",
        "epoch_svm_c1_g0.01",
        "epoch_ensemble_svm",
        "epoch_ensemble_svm_et",
    ],
)
summarize(
    "all configs except mean+std",
    [
        "epoch_select150_svm",
        "epoch_svm_c0.3",
        "epoch_svm_c1_g0.01",
        "epoch_ensemble_svm",
        "epoch_ensemble_svm_et",
        "subject_mean_svm_c1_g0.01",
        "subject_mean_ensemble_svm",
    ],
)

Path("results/final_validation/ensemble_summary.json").write_text(
    json.dumps(
        {
            "decision_rule": "majority vote across 3 repeats; mean probability for AUROC",
            "n_subjects": int(len(y)),
            "results": results,
        },
        indent=2,
    ),
    encoding="utf-8",
)
