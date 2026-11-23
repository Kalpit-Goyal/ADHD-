# Results — ADHD EEG classification on the 121-child dataset

All numbers below are our own runs, produced by this repository. The deployed model and every comparison use **subject-wise** validation: a child's epochs never appear in both training and test.

| | |
|---|---|
| **Dataset** | [IEEE DataPort 121-child EEG dataset](https://ieee-dataport.org/open-access/eeg-data-adhd-control-children) (DOI [10.21227/rzfh-zn36](https://doi.org/10.21227/rzfh-zn36)) |
| **Cohort** | 121 children — 61 ADHD / 60 control, ages 7–12, 19 channels, 128 Hz |
| **Task** | Visual attention task |
| **Epochs kept after z-score rejection** | 2,918 |
| **Evaluation** | Subject-wise 5-fold CV, per-fold Youden threshold chosen on validation subjects only |
| **Reference method** | CTE-Net, Gomez-Rivera et al., *Sensors* 2026 — [10.3390/s26185786](https://doi.org/10.3390/s26185786) |
| **Run date** | Trained 2026-11-23 |
| **Hardware** | RTX 3050 6GB (CUDA), CPU for classical models |

> Research prototype. Single site, wide confidence intervals, and model selection was done on these same 121 subjects. **Not a diagnostic device.**

---

## IMPORTANT: Experiment status

| Experiment | Status | Notes |
|---|---|---|
| E1–E5 Classical | **COMPLETE** | Real data, 5-fold CV, all feature sets |
| E6 EEGNet (CUDA) | **COMPLETE** | Real data, full epochs |
| E7 CTE-Net Original | **INCOMPLETE (smoke test only)** | TE computation O(C²·Tz²·D) too slow for full 60-epoch on RTX 3050 6GB |
| E8 CTENet Inspired | **PARTIAL** | Real data, 10 epochs (of 60 intended), AUROC 0.719 |
| E9–E12 Hybrid | **INCOMPLETE (smoke test only)** | Retraining deep models per inner CV fold too slow |

**None of the E7–E12 numbers below should be treated as final results.** They are smoke-test / partial-run values demonstrating that the leakage-free pipeline executes correctly on real data.

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

---

## Complete real-data results — E1–E5 Classical models

All experiments on the 121-child IEEE DataPort dataset (61 ADHD / 60 control, 2,918 epochs, subject-wise 5-fold CV).

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

---

## Complete real-data results — E6 EEGNet (CUDA)

| Metric | Value |
|--------|-------|
| AUROC | 0.730 |
| Accuracy | 0.653 |
| Sensitivity | 0.705 |
| Specificity | 0.600 |
| Subjects | 121 |
| Epochs | 2,918 |
| Device | CUDA |

---

## Partial real-data results — E8 CTENet Inspired (10 of 60 epochs)

Run with `configs/eeg_cte_real_4s.yaml` (4s/512-sample protocol), `--deep-model cte_net`, `--deep-epochs 10`, `--folds 5`, `--batch-size 16`, device=cuda.

| Fold | Train | Val | Test | AUROC | Accuracy | Threshold | Best Val AUROC |
|------|-------|-----|------|-------|----------|-----------|----------------|
| 1 | 76 | 20 | 25 | 0.603 | 0.640 | 0.79 | 0.820 |
| 2 | 77 | 20 | 24 | 0.729 | 0.625 | 0.73 | 0.790 |
| 3 | 77 | 20 | 24 | 0.750 | 0.667 | 0.74 | 0.670 |
| 4 | 77 | 20 | 24 | 0.708 | 0.708 | 0.63 | 0.740 |
| 5 | 77 | 20 | 24 | 0.771 | 0.583 | 0.82 | 0.790 |

| Metric | Value |
|--------|-------|
| AUROC (mean) | 0.719 |
| Accuracy (mean) | 0.645 |
| Sensitivity | 0.557 |
| Specificity | 0.733 |

**Note:** This is a partial run (10 of 60 intended epochs). Not a complete experiment.

---

## Smoke-test results — E7, E9–E12 (NOT final results)

All values below are from 1-epoch smoke tests demonstrating that the leakage-free pipeline executes on real data. **Do not cite as experimental results.**

### E7 CTE-Net Original (smoke test: 1 epoch, 2 folds)

| Metric | Value |
|--------|-------|
| AUROC | 0.657 |
| Accuracy | 0.628 |
| Sensitivity | 0.672 |
| Specificity | 0.583 |

**Status:** Cannot complete full 60-epoch 5-fold training on RTX 3050 6GB. TE computation O(C²·Tz²·D) per epoch is too slow — timed out after 15 minutes without completing even one epoch. Architecture verified by unit tests (test_n_cuda_forward_backward, test_l_ctenet_original_paper_parameters, test_m_ctenet_original_valid_conv_dimensions).

### E9–E12 Hybrid (smoke test: 1 epoch, 2 folds, deep model = cte_net)

| Exp | Method | AUROC | Accuracy | Sensitivity | Specificity |
|-----|--------|-------|----------|-------------|-------------|
| E9 | Feature Fusion | 0.782 | 0.727 | 0.934 | 0.517 |
| E10 | Soft Voting | 0.838 | 0.752 | 0.967 | 0.533 |
| E11 | Weighted Voting | 0.820 | 0.727 | 0.918 | 0.533 |
| E12 | Stacking | 0.825 | 0.727 | 0.852 | 0.600 |

**Status:** Cannot complete full training because each fold retrains deep models per inner CV split, which is prohibitively slow on RTX 3050 6GB. Leakage fixes verified by unit tests (test_q*, test_t*, test_v*).

---

## CTE-Net paper deviation documentation

The `configs/eeg_cte_real_4s.yaml` file implements the paper-faithful 4-second/512-sample/50%-overlap protocol. Deviations from Gomez-Rivera et al. (2026):

1. **Channel order:** Paper uses [Fz, Cz, Pz, C3, T3, C4, T4, Fp1, Fp2, F3, F4, F7, F8, P3, P4, T5, T6, O1, O2]. Dataset stores [Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, Pz]. No reordering applied (pipeline-level choice).
2. **Preprocessing:** Paper uses raw EEG without filtering. This config disables band-pass and notch filtering but retains z-score rejection (data uses undocumented int16 scale).
3. **Window length:** 4s at 128 Hz = 512 samples (matches paper).
4. **Overlap:** 50% (matches paper).
5. **Architecture:** All components (Transformer contextualization, depthwise temporal filtering with valid padding K=83, Takens embeddings Dx=6/Dy=1/tau=2, bias-free linear projections, RQ kernel, Renyi α=2 entropy, 4-term TE formula, flattened off-diagonal TE head) match the paper's specification. Parameter values match Table 2 (d=128, L=1, nh=1, d_ff=128, p=0.5, K=83, pool=4, alpha_R=2, a=1, l=1, alpha_RQ=1, H=64).

The existing `configs/eeg_cte_real.yaml` (5-second/640-sample protocol) is kept separate for project consistency.

---

## Leakage fixes applied (2026-11-23)

1. **Train/dev overlap:** `_train_deep_model_on_subjects()` now excludes validation subjects from training (`actual_train = train - val`). Raises `ValueError` if no train subjects remain.
2. **E11 global OOF leakage:** Weights derived from fresh inner nested OOF predictions, never from precomputed global `branch_b_probs`.
3. **E10/E11 silent 0.5:** Validation probabilities stored separately; missing predictions raise `RuntimeError`.
4. **E12 in-sample threshold:** Cross-fitted meta-model predictions used for threshold selection; outer test fold never touched.
5. **Connectivity terminology:** `granger` → `var_coefficient` (VAR coefficient proxy, not formal Granger causality).
6. **CTE-Net model identity:** `--deep-model` flag distinguishes `ctenet_original` (E7) from `cte_net` (E8 inspired).
7. **Valid convolution:** CTE-Net temporal filter uses `padding=0` (valid), matching paper architecture.

---

## Tests

**43/43 passed** in `tests/test_pipeline.py`, including 13 new leakage/disjointness/identity tests:
- `test_q*` — train/dev subject disjointness in `_train_deep_model_on_subjects`
- `test_t*` — E11 fresh nested OOF weights, global probs never reused
- `test_u*` — E10/E11 validation predictions present, no silent 0.5
- `test_v*` — E12 cross-fitted threshold calibration
- `test_w*` — branch predictions align by participant ID
- `test_x*` — original vs inspired CTE-Net distinguishable
- `test_y*` — connectivity not labelled as formal Granger causality

---

## Existing detailed results sections

The sections below (sections 2–14) contain historical results from prior runs, including the deployed SVM artifact, classical baselines, feature search, repeated CV, ensembles, and caveats. These are preserved for reference but some numbers may reflect pre-leakage-fix pipeline versions.

### Deployed model — full metrics (3-repeat vote, n = 121)

| Metric | Value |
|--------|-------|
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

`models/adhd_svm_select150.joblib` — RBF-SVM on log band power, relative band power, and VAR-based directed-connectivity edges; 150 of 380 features selected inside each training fold.

---

See `FINAL_RESEARCH_REPORT.md` for the complete research report.
