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

### 3.1 Deep learning models (`src/models.py`)

| Model | File | Status | Notes |
|---|---|---|---|
| `EEGNet` | `src/models.py:13` | Implemented | Raw-signal CNN baseline. Input `(B, 1, C, T)`. |
| `CTENet` | `src/models.py:144` | Implemented (inspired) | Graph transformer over spectral features + connectivity bias. Described in code as "a clean reimplementation, not the authors' code." |

**CTENet gaps vs. original CTE-Net paper (Gomez-Rivera et al., 2026):** The original implements channel-wise nonlinear temporal filters, Takens delay-coordinate embeddings, differentiable matrix-based Transfer Entropy, Rényi alpha entropy, rational quadratic kernel, and a Transformer encoder over directed connectivity. The repository's `CTENet` implements only a graph-attention layer with a learnable edge bias over precomputed band-power + VAR-connectivity features. It does **not** implement differentiable TE, Rényi entropy, rational quadratic kernel, or Takens embeddings. It is a CTE-Net-*inspired* model, not a faithful reproduction.

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
| Rényi alpha entropy | High | Not implemented |
| Rational quadratic kernel | High | Not implemented |
| Transformer encoder over connectivity | Partial | Graph attention exists, but not the full original architecture |
| Participant-level prediction | Yes | Implemented (epoch probs averaged per subject) |

**Action:** Keep the existing `CTENet` as the "CTE-Net-inspired graph-attention model." Create a separate `src/ctenet.py` attempting the original architecture where feasible, documenting which components are missing. Do not claim exact reproduction.

### 9.3 Hybrid model gaps (not implemented at all)

| Component | Status | Action |
|---|---|---|
| Branch A: Statistical/spectral feature branch (XGBoost/SVM) | Partially | Features exist; need unified training script |
| Branch B: Deep learning branch (CTE-Net) | Partially | Model exists; needs saving/loading |
| Branch C: Feature-level fusion | Missing | Implement |
| Branch D: Prediction-level ensemble (soft voting, weighted voting, stacking) | Missing | Implement |
| Hybrid model saving/loading | Missing | Implement |

### 9.4 Experiment gaps (E1-E12)

| Experiment | Model | Status |
|---|---|---|
| E1 | XGboost + Paper 1 features | **Not implemented** |
| E2 | RBF-SVM + Paper 1 features | Implemented (in classical_baseline) |
| E3 | Random Forest | Implemented |
| E4 | Extra Trees | Implemented |
| E5 | Logistic Regression | Implemented |
| E6 | EEGNet | Implemented |
| E7 | Original CTE-Net | **Not implemented** (inspired version exists) |
| E8 | Existing CTE-Net-inspired model | Implemented |
| E9 | Feature-level fusion | **Not implemented** |
| E10 | Soft voting ensemble | **Not implemented** |
| E11 | Weighted soft voting | **Not implemented** |
| E12 | Stacking ensemble | **Not implemented** |

---

## 10. KEY DOCUMENTED DIFFERENCES FROM THE PAPERS

1. **Paper 1 uses XGBoost as primary classifier; the repo's classical comparison lacks XGBoost.** Must be added.
2. **Paper 1 uses ASR + ICA preprocessing; the repo has ICA but not ASR** (no `mne`/`pyrasa` installed). ICA was tested and rejected in prior runs (AUROC dropped 0.66 → 0.51).
3. **Paper 1 uses 5-fold CV with 80/20 train/test subject split; the repo uses 5-fold subject-wise with an inner validation split.** Comparable but not identical.
4. **Paper 2's CTE-Net uses differentiable TE, Rényi entropy, rational quadratic kernel, Takens embeddings; the repo's CTENet uses a VAR-connectivity proxy + graph attention.** Not a faithful reproduction — must be documented.
5. **Paper 2 reports 83.4% participant-level accuracy with 5 folds × 10 repetitions; the repo's CTE-Net achieves 62.8% with 5 folds × 1 repeat.** Large gap partly attributable to architecture differences and the medicated cohort.
6. **Paper 1 reports 86.1% using only 3 channels (P3, P4, C3) with ASR; the repo uses all 19 channels.** Different experimental configuration.

---

## 11. RECOMMENDED IMPLEMENTATION PLAN

1. **Add XGBoost** to classical ML comparison (E1).
2. **Create `src/ctenet.py`** attempting original CTE-Net components where computationally feasible, documenting gaps (E7).
3. **Build hybrid model** (`src/hybrid_model.py`, `src/ensemble.py`) with feature fusion and prediction-level ensemble (E9-E12).
4. **Run all 12 experiments** with identical subject-wise folds on real data.
5. **Save all trained models** to `models/` with scalers, selectors, encoders, and metadata.
6. **Generate visualizations** to `results/`.
7. **Write `FINAL_RESEARCH_REPORT.md`** and update `RESULTS.md`, `README.md`.

---

## 12. FILES THAT WILL BE CREATED/MODIFIED

| File | Action | Purpose |
|---|---|---|
| `RESEARCH_IMPLEMENTATION_AUDIT.md` | Create | This document |
| `src/ctenet.py` | Create | Original CTE-Net reproduction attempt |
| `src/hybrid_model.py` | Create | Feature-level fusion hybrid |
| `src/ensemble.py` | Create | Prediction-level ensemble (voting, stacking) |
| `src/evaluation.py` | Create | Unified evaluation + leakage checks |
| `scripts/train_classical.py` | Create | E1-E5 classical ML with XGBoost |
| `scripts/train_ctenet.py` | Create | E6-E8 deep learning |
| `scripts/train_hybrid.py` | Create | E9-E12 hybrid + ensemble |
| `scripts/evaluate_all.py` | Create | Unified evaluation across all models |
| `scripts/compare_models.py` | Create | Comparison table + paper comparison |
| `configs/hybrid.yaml` | Create | Hybrid model config |
| `results/model_comparison.csv` | Create | Experiment results |
| `RESULTS.md` | Update | Add new experiment results |
| `README.md` | Update | Add hybrid model docs |
| `tests/test_pipeline.py` | Extend | Add feature/ensemble/leakage tests |
| `FINAL_RESEARCH_REPORT.md` | Create | Final comprehensive report |
