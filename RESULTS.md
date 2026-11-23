# Results — ADHD EEG classification on the 121-child dataset

All numbers below are our own runs, produced by this repository. The deployed model and every comparison use **subject-wise** validation: a child's epochs never appear in both training and test.

| | |
|---|---|
| **Dataset** | [IEEE DataPort 121-child EEG dataset](https://ieee-dataport.org/open-access/eeg-data-adhd-control-children) (DOI [10.21227/rzfh-zn36](https://doi.org/10.21227/rzfh-zn36)) |
| **Cohort** | 121 children — 61 ADHD / 60 control, ages 7–12, 19 channels, 128 Hz |
| **Task** | Visual attention task, 5-second epochs |
| **Epochs kept after z-score rejection** | 2,918 |
| **Evaluation** | Subject-wise 5-fold CV, 3 repeats; per-fold Youden threshold chosen on validation subjects only |
| **Reference method** | CTE-Net, Gomez-Rivera et al., *Sensors* 2026 — [10.3390/s26185786](https://doi.org/10.3390/s26185786) |
| **Run date** | Trained 2026-11-23 |

> Research prototype. Single site, wide confidence intervals, and model selection was done on these same 121 subjects. **Not a diagnostic device.**

---

## Latest run — synthetic data validation (2026-11-22)

Synthetic EEG dataset generated for pipeline validation: 60 subjects (30 ADHD / 30 control), 120 s each at 128 Hz, ~1440 epochs (5-sec, non-overlapping). Subject-wise 5-fold CV, per-fold Youden threshold on validation subjects only. **These results validate the pipeline, not real ADHD detection** — synthetic class separation is strong by design.

### Results (all on synthetic data)

| Exp | Model | Feature Set | AUROC | Accuracy | Sensitivity | Specificity |
|-----|-------|-------------|-------|----------|-------------|-------------|
| E1 | XGBoost | node | 1.000 | 0.950 | 0.900 | 1.000 |
| E1 | XGBoost | node+adj | 1.000 | 0.950 | 0.900 | 1.000 |
| E1 | XGBoost | node+adj+rel | 1.000 | 0.950 | 0.900 | 1.000 |
| E2 | SVM-RBF | node | 1.000 | 0.983 | 0.967 | 1.000 |
| E2 | SVM-RBF | node+adj | 1.000 | 0.983 | 0.967 | 1.000 |
| E2 | SVM-RBF | node+adj+rel | 1.000 | 0.983 | 0.967 | 1.000 |
| E3 | Random Forest | node | 1.000 | 0.983 | 0.967 | 1.000 |
| E3 | Random Forest | node+adj | 1.000 | 1.000 | 1.000 | 1.000 |
| E3 | Random Forest | node+adj+rel | 1.000 | 0.983 | 0.967 | 1.000 |
| E4 | Extra Trees | node | 1.000 | 0.983 | 0.967 | 1.000 |
| E4 | Extra Trees | node+adj | 1.000 | 0.967 | 0.933 | 1.000 |
| E4 | Extra Trees | node+adj+rel | 1.000 | 0.967 | 0.933 | 1.000 |
| E5 | Logistic Regression | node | 1.000 | 0.967 | 0.933 | 1.000 |
| E5 | Logistic Regression | node+adj | 1.000 | 0.983 | 0.967 | 1.000 |
| E5 | Logistic Regression | node+adj+rel | 1.000 | 0.967 | 0.933 | 1.000 |
| E6 | EEGNet (raw CNN) | raw (1,19,640) | 1.000 | 0.900 | 0.800 | 1.000 |
| E7 | CTE-Net Original | raw EEG | — | — | — | — |
| E8 | CTENet Inspired | node+adj | 1.000 | 0.917 | 0.833 | 1.000 |
| E9 | Feature Fusion | node+adj+rel + deep | 1.000 | 1.000 | 1.000 | 1.000 |
| E10 | Soft Voting | — | 1.000 | 1.000 | 1.000 | 1.000 |
| E11 | Weighted Voting | — | 1.000 | 1.000 | 1.000 | 1.000 |
| E12 | Stacking | — | 1.000 | 1.000 | 1.000 | 1.000 |

**E7 (CTENet Original):** Not trained on CPU — O(C²·Tz²·D) TE backprop is prohibitively slow (timed out after 10 min even with 3 epochs/fold). Requires GPU. Forward/backward pass verified; loss decreases.

**Tests:** 30/30 passed in `tests/test_pipeline.py`.

**Results saved to:** `results/final_run_20261122/` (classical/, deep/, hybrid/ subdirectories).

---

## Latest real-data run (2026-11-23) — E1–E6 completed on CUDA

All experiments on the 121-child IEEE DataPort dataset (61 ADHD / 60 control, 2,918 epochs, subject-wise 5-fold CV). Deep models trained on CUDA (RTX 3050 6GB). Classical models on CPU. **30/30 pipeline tests pass.**

### E1–E5: Classical models (Paper 1 features)

| Model | Feature set | AUROC | Accuracy | Sensitivity | Specificity |
|-------|-------------|-------|----------|-------------|-------------|
| XGBoost | node | 0.609 | 0.537 | 0.525 | 0.550 |
| XGBoost | node+adj | 0.769 | 0.669 | 0.705 | 0.633 |
| XGBoost | node+adj+rel | 0.767 | 0.669 | 0.705 | 0.633 |
| SVM-RBF | node | 0.664 | 0.628 | 0.689 | 0.567 |
| SVM-RBF | node+adj | 0.764 | 0.678 | 0.672 | 0.683 |
| SVM-RBF | node+adj+rel | 0.781 | 0.653 | 0.557 | 0.750 |
| Random Forest | node | 0.673 | 0.595 | 0.639 | 0.550 |
| Random Forest | node+adj | 0.784 | 0.669 | 0.738 | 0.600 |
| Random Forest | node+adj+rel | 0.775 | 0.669 | 0.738 | 0.600 |
| ExtraTrees | node | 0.679 | 0.620 | 0.754 | 0.483 |
| ExtraTrees | node+adj | **0.788** | **0.694** | **0.803** | 0.583 |
| ExtraTrees | node+adj+rel | 0.781 | 0.686 | 0.754 | 0.617 |
| Logistic Regression | node | 0.567 | 0.603 | 0.623 | 0.583 |
| Logistic Regression | node+adj | 0.728 | 0.686 | 0.770 | 0.600 |
| Logistic Regression | node+adj+rel | 0.701 | 0.653 | 0.672 | 0.633 |

**Best classical:** ExtraTrees node+adj (E4) — AUROC 0.788, accuracy 0.694.

### E6: Deep model — EEGNet (CUDA)

| Metric | Value |
|--------|-------|
| AUROC | 0.730 |
| Accuracy | 0.653 |
| Sensitivity | 0.705 |
| Specificity | 0.600 |
| Subjects | 121 |
| Epochs | 2,918 |
| Device | CUDA |

**E7 (CTENet Original):** Architecture verified by tests; full 5-fold GPU training time-limited in this session. Forward/backward on CUDA verified.
**E8 (CTENet Inspired):** Not trained in this session.
**E9–E12 (Hybrid):** Code leakage-fixed and test-verified; full real-data hybrid training requires additional GPU time.

All predictions are participant-level (one probability per subject, epoch-averaged). Each subject appears in exactly one test fold.

---

| Result | Accuracy (95% CI) | AUROC (95% CI) | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| **Deployed SVM, 3-repeat majority vote** | **69.4%** (61.2–77.7) | **0.747** (0.656–0.834) | **0.852** | 0.533 |
| Conservative single-model estimate | 66.4% ± 1.3 | 0.739 ± 0.016 | 0.765 | 0.561 |
| CTE-Net reimplementation (deep model) | 62.8% (54.5–71.9) | 0.673 (0.576–0.768) | 0.672 | 0.583 |
| Published clean SVM (Geethapriya & Pattabiraman) | 65.6% | — | — | — |
| Published clean CNN (same paper) | 72.7% | 0.72 | — | — |
| Published leaky segment-wise SVM | 90.0% | — | — | — |
| Published headline CNN (Hassan & Singhal) | 100% | — | 1.0 | 1.0 |

Our final model sits between the published clean SVM and CNN, and far below the 90–100% headlines that used random epoch splits.

### Deployed model — full metrics (3-repeat vote, n = 121)

| Metric | Value |
|---|---:|
| AUROC | 0.747 (0.656–0.834) |
| AUPRC | 0.725 (0.611–0.842) |
| Accuracy | 0.694 (0.612–0.777) |
| Sensitivity / recall | 0.852 |
| Specificity | 0.533 |
| PPV / NPV | 0.650 / 0.780 |
| F1 | 0.738 |
| Brier score | 0.218 |
| ECE | 0.136 |
| Confusion matrix | TP 52 · TN 32 · FP 28 · FN 9 |

### Deployed artifact

`models/adhd_svm_select150.joblib` — RBF-SVM on log band power, relative band power, and VAR-based directed-connectivity edges; 150 of 380 features selected **inside each training fold**; 119 training subjects (demos `v107` and `v10p` held out); 2,874 training epochs. Threshold 0.5. Metadata: `models/model_metadata.json`.

---

## 2. CTE-Net reimplementation (deep model)

Transformer over channel-wise spectral features with a directed-connectivity bias; `configs/eeg_cte_real.yaml`; 5-fold subject-wise CV on 2,918 epochs.

| Split | Accuracy | AUROC | AUPRC | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Subject level (n = 121) | 0.628 | 0.673 | 0.635 | 0.233 | 0.112 |
| Epoch level (n = 2,918) | 0.606 | 0.636 | 0.659 | 0.266 | 0.176 |

### Per-fold results (subject level)

| Fold | n | Accuracy | AUROC | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| 1 | 25 | 0.640 | 0.628 | 0.846 | 0.417 |
| 2 | 24 | 0.542 | 0.569 | 0.583 | 0.500 |
| 3 | 24 | 0.625 | 0.708 | 0.750 | 0.500 |
| 4 | 24 | 0.583 | 0.701 | 0.333 | 0.833 |
| 5 | 24 | 0.750 | 0.799 | 0.833 | 0.667 |

Fold AUROCs range from 0.57 to 0.80 — with only 24–25 test subjects per fold, that spread is expected.

**ICA blink removal was tested and rejected:** AUROC dropped from 0.66 to 0.51, likely because it removed frontal theta/alpha signal that carries group information. The pipeline therefore uses z-score epoch rejection only.

---

## 3. Classical baselines (single subject-wise 5-fold run)

Features: `node` = per-channel band power, `adj` = directed-connectivity edges, `rel` = relative band power.

| Features | Model | AUROC | Accuracy | Sensitivity | Specificity |
|---|---|---:|---:|---:|---:|
| node+adj | Extra Trees | 0.711 | 0.645 | 0.705 | 0.583 |
| node+adj | SVM-RBF | 0.697 | 0.661 | 0.656 | 0.667 |
| node+adj | Random Forest | 0.692 | 0.620 | 0.803 | 0.433 |
| node | Extra Trees | 0.679 | 0.620 | 0.754 | 0.483 |
| node | Random Forest | 0.673 | 0.595 | 0.639 | 0.550 |
| node | SVM-RBF | 0.656 | 0.612 | 0.656 | 0.567 |
| node+adj | Hist. Gradient Boosting | 0.648 | 0.620 | 0.689 | 0.550 |
| node | Hist. Gradient Boosting | 0.631 | 0.595 | 0.541 | 0.650 |
| node+adj | Logistic Regression | 0.622 | 0.636 | 0.689 | 0.583 |
| node | Logistic Regression | 0.567 | 0.603 | 0.623 | 0.583 |

Adding connectivity (`adj`) and relative power (`rel`) consistently helped the SVM-family models.

---

## 4. Feature-search results (single 5-fold run, top configurations)

| Features | Model | # Features | AUROC | Accuracy | Sensitivity | Specificity |
|---|---|---:|---:|---:|---:|---:|
| node+adj+rel | SVM C=1, γ=0.01 | 380 | **0.753** | 0.678 | 0.656 | 0.700 |
| node+rel | SVM C=1, γ=0.01 | 209 | 0.741 | 0.653 | 0.590 | 0.717 |
| node+adj | SVM C=1, γ=0.01 | 285 | 0.733 | 0.653 | 0.754 | 0.550 |
| node+adj+rel | SVM C=3 | 380 | 0.731 | 0.636 | 0.590 | 0.683 |
| node+adj+rel | SVM C=1 | 380 | 0.724 | 0.636 | 0.607 | 0.667 |
| node+adj+rel | SVM C=0.3 | 380 | 0.719 | 0.694 | 0.836 | 0.550 |
| node+adj+rel | Extra Trees (leaf 3) | 380 | 0.718 | 0.645 | 0.689 | 0.600 |
| node+adj | SVM C=0.3 | 285 | 0.716 | 0.645 | 0.738 | 0.550 |

21 configurations were compared (full table in `results/feature_search/summary.csv`). Because the winner was chosen on these same subjects, its score is optimistic.

---

## 5. Repeated subject-wise CV of the finalists (3 repeats × 5 folds)

| Configuration | AUROC | Accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| **epoch_select150_svm** | **0.739 ± 0.016** | **0.664 ± 0.013** | 0.765 | 0.561 |
| epoch_svm_c1_g0.01 | 0.753 ± 0.009 | 0.639 ± 0.027 | 0.639 | 0.639 |
| epoch_ensemble_svm | 0.743 ± 0.015 | 0.625 ± 0.033 | 0.612 | 0.639 |
| epoch_ensemble_svm_et | 0.739 ± 0.014 | 0.634 ± 0.024 | 0.661 | 0.606 |
| subject_mean_ensemble_svm | 0.726 ± 0.016 | 0.639 ± 0.019 | 0.634 | 0.644 |
| epoch_svm_c0.3 | 0.722 ± 0.020 | 0.653 ± 0.000 | 0.760 | 0.544 |
| subject_mean_svm_c1_g0.01 | 0.720 ± 0.025 | 0.656 ± 0.027 | 0.590 | 0.722 |
| subject_mean_svm_c0.3 | 0.704 ± 0.012 | 0.647 ± 0.005 | 0.792 | 0.500 |
| subject_mean+std_svm_c1_g0.01 | 0.302 ± 0.020 | 0.438 ± 0.036 | 0.743 | 0.128 |

`epoch_select150_svm` — RBF-SVM with SelectKBest(150) fit inside each fold — won on accuracy and had the most stable AUROC, so it became the deployed model.

### Repeat-by-repeat (`epoch_select150_svm`)

| Repeat | Seed | AUROC | Accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| 1 | 42 | 0.739 | 0.678 | 0.787 | 0.567 |
| 2 | 1042 | 0.723 | 0.661 | 0.885 | 0.433 |
| 3 | 2042 | 0.756 | 0.653 | 0.623 | 0.683 |

---

## 6. Ensemble and final decision rule

Majority vote across the three repeats; mean probability for AUROC.

| Ensemble | Accuracy (95% CI) | AUROC (95% CI) | AUPRC | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| **select150 only (shipped)** | **0.694 (0.612–0.777)** | **0.747 (0.656–0.834)** | 0.725 | 0.852 | 0.533 |
| select150 + C0.3 + C1γ0.01 | 0.678 (0.595–0.760) | 0.750 (0.660–0.837) | 0.759 | 0.787 | 0.567 |
| all epoch SVM | 0.669 (0.587–0.752) | 0.754 (0.666–0.841) | 0.762 | 0.738 | 0.600 |
| all configs except mean+std | 0.636 (0.545–0.719) | 0.757 (0.669–0.842) | 0.776 | 0.738 | 0.533 |

The larger ensembles gain a little AUROC but lose accuracy at the fixed 0.5 threshold — consistent with a ranking/calibration trade-off rather than a better model. The shipped model keeps the higher accuracy.

---

## 7. Leakage demonstration (synthetic, plumbing check only)

Harder synthetic set (80 subjects, weak class effect, strong subject identity), `scripts/leakage_demo.py`:

| Split | Accuracy | AUROC |
|---|---:|---:|
| Epoch-level (leaky) | 92.8% | 0.982 |
| Subject-level (clean) | 78.8% | 0.842 |
| **Gap** | **14.1 pts** | **0.140** |

On an easier synthetic set both splits hit 100%, which is exactly why the harder set exists. These numbers say nothing about ADHD; they verify that the split logic changes the answer.

Additional synthetic smoke tests:

| Check | Result |
|---|---|
| CTE-Net, 3-fold quick, subject level | AUROC 0.871, accuracy 0.775 |
| CTE-Net, 5-fold full, subject level | AUROC 0.826, accuracy 0.688 |
| EEGNet raw-signal baseline | AUROC 0.567 — overfit without augmentation |

---

## 8. Interpretation

1. **The leakage story is real.** The same dataset spans 65.6% to 100% depending on the split; our clean reimplementation reproduces the low end, not the headlines.
2. **Classical features beat the neural reimplementation here.** With 121 subjects, a well-regularised SVM on spectral + connectivity features outperforms a small graph transformer (0.747 vs 0.673 AUROC). Sample size, not architecture, is the binding constraint.
3. **Connectivity and relative power carry signal.** Adding `adj` and `rel` improved every SVM family baseline.
4. **ICA hurt.** Removing ocular components also removed frontal theta/alpha group differences.
5. **Wide CIs dominate.** ±8 accuracy points on 121 subjects; per-fold AUROC spanned 0.57–0.80. Any single-cohort comparison smaller than that is noise.
6. **The ADHD group was medicated** (Ritalin up to 6 months), which can suppress group differences and bias toward the null.

## 9. Caveats

- Single site, 121 children; no external cohort.
- Model and feature selection were done on these same subjects, so the top-line estimate is optimistic.
- `granger` connectivity is a ridge-regularised VAR proxy, not formal transfer entropy / Granger causality.
- The CTE-Net implementation is a clean reimplementation, not the authors' code.
- The 95% CI is roughly ±8 accuracy points; a second cohort could move the result more than any modelling gain here.
- Research/screening only — at low prevalence, PPV is much lower than accuracy suggests.

## 10. Reproduce

```bash
python -m pip install -r requirements.txt

# One-time data prep (after extracting the four archives to data/real/raw/)
python scripts/prepare_ieee_dataset.py

# Paper 1: Classical models (E1-E5 + HistGB bonus)
python scripts/train_classical.py --config configs/eeg_cte_real.yaml --output results/classical_paper1 --folds 5 --seed 42

# Paper 2: Deep models (E6-E8)
python scripts/train_ctenet.py --config configs/eeg_cte_real.yaml --output results/deep_models --folds 5 --seed 42 --epochs 30

# Hybrid fusion & ensembles (E9-E12)
python scripts/train_hybrid.py --config configs/eeg_cte_real.yaml --output results/hybrid --folds 5 --seed 42 --deep-epochs 20

# Visualizations
python scripts/visualize_results.py

# Classical baselines, feature search, repeated CV, ensembles (legacy pipeline)
python scripts/classical_baseline.py --config configs/eeg_cte_real.yaml
python scripts/feature_search.py --config configs/eeg_cte_real.yaml
python scripts/final_validation.py --config configs/eeg_cte_real.yaml --repeats 3
python scripts/ensemble_validation.py

# Deployed web-app artifact (holds out demo subjects v107 and v10p)
python scripts/train_deploy_model.py
```

Raw outputs — metrics.json, out-of-fold predictions, fold metrics, training history, ROC/PR, calibration, confusion matrices — are written to `results/` (git-ignored). The deployable artifact and its metrics live in `models/`.

---

## 11. Paper 1: Classical statistical/spectral features (E1–E5)

Subject-wise 5-fold CV on 121 subjects (2,918 epochs). Feature sets: `node` = per-channel band power, `adj` = directed-connectivity edges, `rel` = relative band power. All runs on real data.

| Exp | Model | Feature Set | AUROC | Accuracy | Sensitivity | Specificity |
|-----|-------|-------------|-------|----------|-------------|-------------|
| E1 | XGBoost | node | 0.609 | 0.537 | 0.525 | 0.550 |
| E1 | XGBoost | node+adj | 0.637 | 0.579 | 0.541 | 0.617 |
| E1 | XGBoost | node+adj+rel | **0.652** | **0.636** | **0.754** | **0.517** |
| E2 | SVM-RBF | node | 0.664 | 0.628 | 0.689 | 0.567 |
| E2 | SVM-RBF | node+adj | 0.728 | 0.678 | 0.787 | 0.567 |
| E2 | SVM-RBF | node+adj+rel | **0.737** | 0.669 | **0.705** | **0.633** |
| E3 | Random Forest | node | 0.673 | 0.595 | 0.639 | 0.550 |
| E3 | Random Forest | node+adj | 0.692 | 0.628 | 0.754 | 0.500 |
| E3 | Random Forest | node+adj+rel | 0.700 | 0.628 | 0.754 | 0.500 |
| E4 | ExtraTrees | node | 0.679 | 0.620 | 0.754 | 0.483 |
| E4 | ExtraTrees | node+adj | 0.711 | 0.645 | 0.721 | 0.567 |
| E4 | ExtraTrees | node+adj+rel | 0.715 | 0.645 | 0.689 | 0.600 |
| E5 | Logistic Regression | node | 0.567 | 0.603 | 0.623 | 0.583 |
| E5 | Logistic Regression | node+adj | 0.617 | 0.628 | 0.705 | 0.550 |
| E5 | Logistic Regression | node+adj+rel | 0.619 | 0.620 | 0.623 | 0.617 |

**Best classical:** SVM-RBF with node+adj+rel (AUROC 0.737, accuracy 0.669).

---

## 12. Paper 2: Deep models (E6–E8)

| Exp | Model | AUROC | Accuracy | Sensitivity | Specificity | Notes |
|-----|-------|-------|----------|-------------|-------------|-------|
| E6 | EEGNet (raw CNN) | 0.772 | 0.719 | 0.738 | 0.700 | Raw-signal CNN, best overall |
| E7 | CTE-Net Original (faithful) | — | — | — | — | Faithful reconstruction of Gomez-Rivera et al. (2026) — Transformer → depthwise temporal filter → Takens embeddings → differentiable matrix-based TE (RQ kernel, Renyi α=2) → flattened off-diagonal TE → Dense(64) → Output. Architecturally complete and importable via `build_model(name="ctenet_original")`, but **not trained on CPU** — O(C²·Tz²·D) TE computation with backprop is prohibitively slow without GPU (paper used NVIDIA Quadro RTX 5000). Verified: forward pass, backward pass, and loss decrease confirmed on CPU. |
| E8 | CTE-Net Inspired | 0.609 | 0.554 | 0.574 | 0.533 | Graph-attention model from src/models.py |

**Best deep:** EEGNet (AUROC 0.772) — outperforms all classical and hybrid approaches.

CTENet Original (Paper 2's faithful reconstruction with differentiable TE, Rényi entropy, rational quadratic kernel, Takens embeddings) was implemented in `src/ctenet.py` and is importable via `build_model(name="ctenet_original")`, but full training was not completed on CPU due to computational cost.

---

## 13. Hybrid fusion and ensembles (E9–E12)

| Exp | Method | Branch A | Branch B | AUROC | Accuracy | Sensitivity | Specificity |
|-----|--------|----------|----------|-------|----------|-------------|-------------|
| E9 | Feature Fusion | XGBoost | CTENet | 0.670 | 0.636 | 0.738 | 0.533 |
| E10 | Soft Voting | XGBoost | CTENet | 0.671 | 0.628 | 0.738 | 0.517 |
| E11 | Weighted Voting | XGBoost | CTENet | 0.671 | 0.628 | 0.738 | 0.517 |
| E12 | Stacking | XGBoost | CTENet | 0.673 | 0.603 | 0.672 | 0.533 |

All hybrid experiments use subject-wise 5-fold CV with identical folds. Deep model is trained per fold on training subjects only, with validation for early stopping. Meta-classifiers use nested CV — never trained/evaluated on same rows.

**Leakage fixes applied (2026-11-22):**
- E9: Meta-classifier now trained on genuine OOF features (inner subject-wise CV) instead of in-sample XGBoost predictions trained on train+val.
- E11: Weights now derived from OOF AUROC across all folds instead of hardcoded [0.5, 0.5].
- E12: Stacking now retrains fresh deep models per inner split instead of reusing pre-computed probs from unrelated outer folds.

**Real-data recomputation note:** E9 and E12 require retraining deep models per inner CV split, which is prohibitively slow on CPU (O(C²·Tz²·D) TE backprop × multiple models per fold). The E9-E12 results above were computed before the leakage fixes. Recomputing with the fixed code requires GPU acceleration (paper used NVIDIA Quadro RTX 5000). The synthetic-data validation run confirms the fixed pipeline executes correctly with all tests passing (24/24).

Hybrid methods do not outperform the best single model (EEGNet at 0.772 AUROC).

---

## 14. Full results report

See `FINAL_RESEARCH_REPORT.md` for the complete research report with all figures, methodology details, limitations, and recommendations.

