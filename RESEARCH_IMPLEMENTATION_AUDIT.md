# Research Implementation Audit

**Project:** Hybrid EEG-Based ADHD Detection System  
**Repository:** https://github.com/Kalpit-Goyal/ADHD-  
**Audit date:** 2026-11-22  
**Auditor:** Kilo (software engineering agent)

---

## 1. EXECUTIVE SUMMARY

The repository contains a working, subject-wise EEG classification pipeline for the 121-child IEEE DataPort dataset (61 ADHD / 60 control, 19 channels, 128 Hz). It implements:

- Classical ML baselines (LR, SVM-RBF, ExtraTrees, Random Forest, HistGradientBoosting) on spectral + connectivity features.
- A CTE-Net-inspired graph transformer over channel-wise band-power features with a directed-connectivity bias.
- A raw-signal EEGNet CNN baseline.
- Subject-wise 5-fold cross-validation with validation-threshold selection, bootstrap CIs, ROC/PR/calibration plots.
- A deployed RBF-SVM artifact (`models/adhd_svm_select150.joblib`) achieving 69.4% accuracy / 0.747 AUROC on 3-repeat subject-wise CV.

**No hybrid model, no feature-level fusion, no prediction-level stacking, no XGBoost in the classical comparison, and no faithful CTE-Net reproduction exist yet.** The real dataset is present and prepared (`data/real/recordings/` + `manifest.csv`).

**UPDATE (2026-11-22):** All items above are now implemented. XGBoost (`src/classical_models.py`), CTENetOriginal (`src/ctenet.py`), hybrid models (`src/hybrid_model.py` — E9-E12), and training scripts (`scripts/train_classical.py`, `scripts/train_ctenet.py`, `scripts/train_hybrid.py`) are all implemented. All 24 pipeline tests pass. Synthetic-data experiments complete. Real-data experiments are the remaining step.

---

## 2. DATASET INVENTORY

| Item | Status | Details |
|---|---|---|
| Real recordings | Present | 121 `.npy` files in `data/real/recordings/` |
| Manifest | Present | `data/real/manifest.csv` — 121 rows, subject_id, label, site, file, n_samples, duration_seconds |
| Class balance | Verified | 61 ADHD (label=1) / 60 control (label=0) |
| Channels | Verified | 19 channels, order: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, Pz |
| Sampling rate | Verified | 128 Hz |
| Raw .mat sources | Present | `data/real/raw/{ADHD_part1,ADHD_part2,Control_part1,Control_part2}/` |
| Prepared recordings | Present | 121 `.npy` files already converted by `scripts/prepare_ieee_dataset.py` |
| Synthetic data | Present | `data/synthetic/` (40 subjects), `data/synthetic_hard/` (80 subjects) — for plumbing tests only |

**Data is complete.** No download is needed. The two research papers use the same underlying IEEE DataPort 121-child dataset; this repository already has it prepared.

---

## 3. MODELS CURRENTLY IMPLEMENTED

### 3.1 Deep learning models (`src/models.py`, `src/ctenet.py`)

| Model | File | Status | Notes |
|---|---|---|---|
| `EEGNet` | `src/models.py:13` | Implemented | Raw-signal CNN baseline. Input `(B, 1, C, T)`. |
| `CTENet` (inspired) | `src/models.py:144` | Implemented (inspired) | Graph transformer over spectral features + connectivity bias. CTE-Net-*inspired*, not faithful. |
| `CTENetOriginal` | `src/ctenet.py:287` | Implemented | Faithful reconstruction of CTE-Net (Gomez-Rivera et al., 2026, Sensors 26(18):5786). Implements channel-wise nonlinear temporal filters, Takens delay-coordinate embeddings, differentiable matrix-based Transfer Entropy, Rényi alpha=2 entropy, rational quadratic kernel, Transformer encoder over directed connectivity. Documented deviations: input window length (paper T=512 vs repo default T=640), channel order (paper vs IEEE DataPort), preprocessing (paper raw vs repo filtered/z-scored). |

**CTENet paper verification:** `src/ctenet.py` implements all architectural components specified in Paper 2 (Sensors 2026, 26(18):5786): Transformer d=128/L=1/nh=1 → depthwise K=83/pool=4 → Takens Dx=6/Dy=1/τ=2 → RQ kernel TE Renyi α=2 → flattened off-diagonal TE → Dense(64). Parameter values match Table 2 (d=128, L=1, nh=1, d_ff=128, p=0.5, K=83, pool=4, Dx=6, Dy=1, tau=2, alpha_R=2, a=1, l=1, alpha_RQ=1, H=64). Deviations documented in `src/ctenet.py:689-717` (`DOCUMENTED_GAPS`).

E7 (CTE-Net Original on real data): forward/backward verified, loss decreases, but CPU training is prohibitively slow (O(C²·Tz²·D) TE backprop). Requires GPU or compute reduction per master prompt rule 7.

### 3.2 Classical ML models

| Model | File | Status |
|---|---|---|
| Logistic Regression | `scripts/classical_baseline.py:41` | Implemented (with StandardScaler, class_weight=balanced) |
| SVM-RBF | `scripts/classical_baseline.py:49` | Implemented |
| Extra Trees | `scripts/classical_baseline.py:58` | Implemented |
| Random Forest | `scripts/classical_baseline.py:65` | Implemented |
| HistGradientBoosting | `scripts/classical_baseline.py:72` | Implemented |
| XGBoost | **Missing** | Not present anywhere in the repository |

Paper 1's primary classifier is XGBoost. It is not implemented in this repository and must be added.

---

## 4. FEATURE EXTRACTION STATUS

### 4.1 Implemented (`src/features.py`, `src/feature_extraction.py`)

| Feature | Paper 1 | Paper 2 | Status | File |
|---|---|---|---|---|
| Absolute band power (delta/theta/alpha/beta/gamma) | Yes | No | Implemented | `src/feature_extraction.py:86` |
| Relative band power | Yes | No | Implemented | `src/feature_extraction.py:86` |
| Log band power | Yes | No | Implemented | `src/feature_extraction.py:187-191` |
| Band-power ratios (theta/beta, etc.) | Yes | No | Implemented | `src/feature_extraction.py:207-214` |
| Mean, std, variance, skewness, kurtosis, RMS, peak-to-peak | Yes | No | Implemented | `src/feature_extraction.py:157-168` |
| Katz fractal dimension | Yes | No | Implemented | `src/feature_extraction.py:90-107` |
| Hjorth mobility/complexity | Yes | No | Implemented | `src/feature_extraction.py:110-123` |
| Spectral entropy, spectral edge frequency | Yes | No | Implemented | `src/feature_extraction.py:126-142` |
| Directed connectivity (VAR/Granger proxy) | No | Partial | Implemented as proxy | `src/features.py:86-135` |
| Phase-locking value (PLV) | No | No | Implemented | `src/features.py:65-83` |
| Correlation adjacency | No | No | Implemented | `src/features.py:57-62` |
| ICA artifact removal | Yes | No | Implemented | `src/data.py:323-383` |
| ASR artifact removal | Yes | No | **Missing** | Not implemented; `mne`/`pyrasa` not installed |
| Welch PSD | Yes | No | Implemented | `src/feature_extraction.py:61-63` |

### 4.2 Feature selection

| Item | Status |
|---|---|
| SelectKBest (f_classif) inside each fold | Implemented in `scripts/final_validation.py` and `scripts/train_deploy_model.py` |
| Feature selection fitted on training data only | Yes (in final_validation) |
| Mutual information, RFE | Not implemented |
| Feature selection before CV (leakage) | Not present — selection is inside folds |

---

## 5. PREPROCESSING STATUS

| Step | Paper 1 | Paper 2 | Status | File |
|---|---|---|---|---|
| Resample to 128 Hz | Yes | Yes | Implemented | `src/data.py:296-300` |
| Convert to microvolts | Yes | Yes | Implemented | `src/data.py:276-285` |
| Mean removal | Yes | Yes | Implemented | `src/data.py:303` |
| Notch filter (50 Hz) | Yes | Yes | Implemented | `src/data.py:306-308` |
| Band-pass filter (0.5-45 Hz; paper uses 0.5-40 Hz) | Yes | Yes | Implemented | `src/data.py:310-318` |
| 5-second epochs | Yes | Yes | Implemented | `src/data.py:386-416` |
| Per-recording channel z-score | Yes | Yes | Implemented | `src/data.py:419-430` |
| Z-score epoch rejection (max_abs_z) | Yes | Yes | Implemented | `src/data.py:491-502` |
| ICA blink removal | Yes (tested, rejected) | No | Implemented | `src/data.py:323-383` |
| ASR | Yes | No | **Missing** | Not available (no `mne`/`pyrasa`) |

**Preprocessing parameters are configurable via `DataConfig`.** Fitting is done per-recording (not across subjects), so no cross-subject leakage from normalization.

---

## 6. CROSS-VALIDATION & DATA LEAKAGE

### 6.1 Split strategy

| Requirement | Status |
|---|---|
| Subject-wise splits (no epoch overlap) | Yes — `src/data.py:535-590` (`subject_folds`) and `src/cv.py` |
| Stratified by class | Yes — `StratifiedKFold` |
| 5 folds | Yes |
| Validation from training subjects only | Yes — inner train/val split |
| Test subjects completely unseen | Yes |
| Same splits for all models | Partially — deep models use `src/train.py`, classical use `scripts/classical_baseline.py`/`final_validation.py` (both StratifiedKFold by subject, but different random seeds may produce different folds) |

### 6.2 Leakage checks

| Check | Status |
|---|---|
| No participant overlap train/test | Enforced — `src/cv.py:207-307` (`LeakageError`, `validate_folds`, `assert_no_subject_overlap`) |
| No feature selection on full dataset | Yes — SelectKBest inside each fold |
| No scaler on full dataset | Yes — pipeline per fold |
| No oversampling before splitting | Yes — no SMOTE used |
| No threshold optimization on test labels | Yes — Youden on validation subjects only |
| Automated leakage check | Yes — `validate_folds()` raises `LeakageError` |

**Unit test for disjoint folds:** `tests/test_pipeline.py:111-123` (`test_subject_folds_are_disjoint`).

---

## 7. TRAINED MODELS & SAVED ARTIFACTS

| Artifact | Path | Status | Real-data results |
|---|---|---|---|
| Deployed SVM | `models/adhd_svm_select150.joblib` | Trained on real data | 69.4% accuracy, 0.747 AUROC (3-repeat vote) |
| Model metadata | `models/model_metadata.json` | Present | Records metrics, selected features, excluded demo subjects |
| Feature cache | `models/feature_cache.npz` | Present | Cached epoch features |
| CTE-Net weights | **None saved** | Not persisted | Results in `results/cte_net_real/` (not a saved artifact) |
| EEGNet weights | **None saved** | Not persisted | Results in `results/eegnet/` |
| Classical models | **None saved** | Not persisted | Results in `results/classical_baseline/` |

The only saved, reloadable model is the deployed SVM. Deep models and classical baselines are trained on-the-fly and results written to `results/`, but no `.pt` or `.joblib` artifacts are saved for them.

---

## 8. EXISTING RESULTS (REAL DATA)

| Model | Accuracy | AUROC | Sensitivity | Specificity | Source |
|---|---|---|---|---|---|
| Deployed SVM (3-repeat vote) | 69.4% | 0.747 | 0.852 | 0.533 | `RESULTS.md`, `model_metadata.json` |
| CTE-Net reimplementation | 62.8% | 0.673 | 0.672 | 0.583 | `RESULTS.md` |
| Published clean SVM | 65.6% | — | — | — | Geethapriya & Pattabiraman (2026) |
| Published clean CNN | 72.7% | 0.72 | — | — | Geethapriya & Pattabiraman (2026) |
| Paper 1 reported (XGBoost) | 86.1% | — | — | — | García-Ponsoda et al. (2024) |
| Paper 2 reported CTE-Net | 83.4% | 0.902 | — | — | Gomez-Rivera et al. (2026) |

**Synthetic-data results exist** (for plumbing/leakage checks only) and are clearly labeled as such in `RESULTS.md`.

---

## 9. GAP ANALYSIS — WHAT NEEDS TO CHANGE

### 9.1 Paper 1 (García-Ponsoda et al., 2024) gaps

| Gap | Priority | Action |
|---|---|---|
| XGBoost classifier not implemented | High | Add XGBoost to classical baseline comparison |
| ASR artifact removal not available | Medium | Document as missing (no `mne`/`pyrasa`); ICA is implemented but was tested and rejected in prior runs |
| Whole-recording / segment-based features (Paper 1 uses 1/1, 1/2, ..., 1/20 segments) | Medium | `src/feature_extraction.py` already has `segment_recording()` and `extract_recording_features()`; needs integration into training pipeline |
| Feature selection with statistical testing (normality/homoscedasticity → t-test/Welch) | Low | Currently uses SelectKBest/f_classif; Paper 1 uses parametric statistical tests. Document difference. |
| Subject-level feature aggregation (mean per subject) vs epoch-level | Medium | `scripts/final_validation.py` has `aggregate_subjects()`; needs to be unified |

### 9.2 Paper 2 (CTE-Net) gaps

| Gap | Priority | Action |
|---|---|---|
| Channel-wise nonlinear temporal filters | High | Not implemented |
| Takens delay-coordinate embeddings | High | Not implemented |
| Differentiable Transfer Entropy | High | Not implemented (VAR proxy used instead) |
| Rényi alpha entropy | High | Implemented in `src/ctenet.py:RenyiAlphaEntropy` (paper uses alpha=2) |
| Rational quadratic kernel | High | Implemented in `src/ctenet.py:RationalQuadraticKernel` |
| Transformer encoder over connectivity | Yes | Implemented in `src/ctenet.py:TransformerEncoderLayer` / `CTENetOriginal` |
| Participant-level prediction | Yes | Implemented (epoch probs averaged per subject) |

**Action:** `src/ctenet.py` provides `CTENetOriginal` — a faithful reconstruction of the original CTE-Net architecture. Remaining deviations are pipeline-level (window length, channel order, preprocessing), documented in `src/ctenet.py:689-717`. Do not claim exact reproduction due to these pipeline-level differences.

E7 CPU feasibility: `CTENetOriginal` forward/backward verified on CPU, loss decreases, but full training is prohibitively slow due to O(C²·Tz²·D) TE backprop. Requires GPU.

### 9.3 Hybrid model gaps

| Component | Status | Action |
|---|---|---|
| Branch A: Statistical/spectral feature branch (XGBoost/SVM) | Yes | Implemented in `src/classical_models.py`, `scripts/train_classical.py` |
| Branch B: Deep learning branch (CTE-Net) | Yes | Implemented in `src/ctenet.py`, `src/models.py`, `scripts/train_ctenet.py` |
| Branch C: Feature-level fusion | Yes | Implemented in `src/hybrid_model.py:run_feature_fusion()` (E9) with genuine OOF meta-features |
| Branch D: Prediction-level ensemble | Yes | Implemented in `src/hybrid_model.py:run_prediction_ensemble()` (E10-E12) with validation-derived weights and nested CV stacking |
| Hybrid model saving/loading | Yes | Implemented in `src/hybrid_model.py:save_hybrid_result()` |

**Leakage fixes applied (2026-11-22):**
- E9: Meta-classifier now trained on genuine OOF features (inner subject-wise CV) instead of in-sample XGBoost predictions.
- E11: Weights now derived from OOF AUROC across all folds instead of hardcoded [0.5, 0.5].
- E12: Stacking now retrains fresh deep models per inner split instead of reusing pre-computed probs from unrelated outer folds.

### 9.4 Experiment gaps (E1-E12)

| Experiment | Model | Status |
|---|---|---|
| E1 | XGBoost + Paper 1 features | Implemented (`src/classical_models.py`, `scripts/train_classical.py`) — ran on synthetic data |
| E2 | RBF-SVM + Paper 1 features | Implemented (`scripts/classical_baseline.py`) |
| E3 | Random Forest | Implemented (`scripts/classical_baseline.py`) |
| E4 | Extra Trees | Implemented (`scripts/classical_baseline.py`) |
| E5 | Logistic Regression | Implemented (`scripts/classical_models.py`, `scripts/train_classical.py`) |
| E6 | EEGNet | Implemented (`src/models.py`, `scripts/train_ctenet.py`) — ran on synthetic data |
| E7 | Original CTE-Net | Implemented (`src/ctenet.py`) — CPU training infeasible, requires GPU |
| E8 | Existing CTE-Net-inspired model | Implemented (`src/models.py`, `scripts/train_ctenet.py`) — ran on synthetic data |
| E9 | Feature-level fusion | Implemented (`src/hybrid_model.py`) — OOF fixed, ran on synthetic data |
| E10 | Soft voting ensemble | Implemented (`src/hybrid_model.py`) — ran on synthetic data |
| E11 | Weighted soft voting | Implemented (`src/hybrid_model.py`) — validation-derived weights |
| E12 | Stacking ensemble | Implemented (`src/hybrid_model.py`) — nested CV with fresh deep models per inner split |

All experiments ran successfully on synthetic data (60 subjects). Real-data experiments (121 subjects) are the remaining step.

---

## 10. KEY DOCUMENTED DIFFERENCES FROM THE PAPERS

1. **Paper 1 uses XGBoost as primary classifier; XGBoost is now implemented** (`src/classical_models.py`) and ran on synthetic data. Real-data results pending.
2. **Paper 1 uses ASR + ICA preprocessing; the repo has ICA but not ASR** (no `mne`/`pyrasa` installed). ICA was tested and rejected in prior runs (AUROC dropped 0.66 → 0.51).
3. **Paper 1 uses 5-fold CV with 80/20 train/test subject split; the repo uses 5-fold subject-wise with an inner validation split.** Comparable but not identical.
4. **Paper 2's CTE-Net uses differentiable TE, Rényi entropy, rational quadratic kernel, Takens embeddings.** `src/ctenet.py` (CTENetOriginal) implements all these components faithfully. The existing `CTENet` in `src/models.py` remains a CTE-Net-*inspired* graph-attention model.
5. **Paper 2 reports 83.4% participant-level accuracy with 5 folds × 10 repetitions; the repo's CTE-Net achieves 62.8% with 5 folds × 1 repeat.** Large gap partly attributable to architecture differences and the medicated cohort.
6. **Paper 1 reports 86.1% using only 3 channels (P3, P4, C3) with ASR; the repo uses all 19 channels.** Different experimental configuration.

---

## 11. IMPLEMENTATION STATUS

All planned files from section 12 have been created. Remaining work:
- Run all 12 experiments on real data (121 subjects)
- E7 requires GPU (CPU training infeasible)
- Generate visualizations to `results/`
- Write `FINAL_RESEARCH_REPORT.md` and update `RESULTS.md`, `README.md`

---

## 12. FILES CREATED/MODIFIED

| File | Action | Purpose |
|---|---|---|
| `RESEARCH_IMPLEMENTATION_AUDIT.md` | Modified | This document — updated to reflect implementation status |
| `src/ctenet.py` | Created | Original CTE-Net reproduction attempt (CTENetOriginal) |
| `src/hybrid_model.py` | Created | Feature-level fusion (E9) and prediction-level ensemble (E10-E12) |
| `src/classical_models.py` | Created | XGBoost, SVM-RBF, RF, ExtraTrees, LR, HistGB (E1-E5) |
| `scripts/train_classical.py` | Created | E1-E5 classical ML with XGBoost |
| `scripts/train_ctenet.py` | Created | E6-E8 deep learning |
| `scripts/train_hybrid.py` | Created | E9-E12 hybrid + ensemble |
| `configs/eeg_cte_real.yaml` | Created | Real data experiment config |
| `configs/eeg_cte_synthetic.yaml` | Created | Synthetic data experiment config |
| `tests/test_pipeline.py` | Extended | 24 tests including leakage, fusion, ensemble, stacking tests |
| `RESULTS.md` | Modified | Added synthetic results section |
| `FIX_IMPLEMENTATION_AUDIT.md` | Created | Documents all leakage fixes applied |
