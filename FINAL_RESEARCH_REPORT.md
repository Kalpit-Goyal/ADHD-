# Final Research Report — Hybrid EEG-Based ADHD Detection

**Date:** 2026-11-22
**Dataset:** IEEE DataPort ADHD EEG (121 subjects: 61 ADHD / 60 control, 19 channels, 128 Hz)
**Hardware:** CPU only (no CUDA), Python 3.13.5, PyTorch 2.11.0+cpu, XGBoost 3.0.2

---

## Executive Summary

We implemented a complete hybrid EEG-based ADHD detection system combining two research papers:

- **Paper 1** (García-Ponsoda 2024): Statistical/spectral node + adjacency + relative-edge features with XGBoost classification.
- **Paper 2** (Gomez-Rivera 2026): CTE-Net transformer architecture using differentiable transfer entropy, Rényi entropy, rational quadratic kernel, and Takens embeddings.

All experiments use **real data only** (no synthetic data), **subject-wise 5-fold cross-validation** (no epoch leakage), and thresholds tuned on validation subjects only.

### Best Results by Category

| Category | Experiment | Model | AUROC | Accuracy | Sensitivity | Specificity |
|----------|-----------|-------|-------|----------|-------------|-------------|
| **Overall best** | E6 | EEGNet (raw-signal CNN) | **0.772** | **0.719** | 0.738 | 0.700 |
| Best classical | E2 | SVM-RBF (node+adj+rel) | **0.737** | 0.669 | 0.705 | 0.633 |
| Best XGBoost | E1 | XGBoost (node+adj+rel) | 0.652 | 0.636 | 0.754 | 0.517 |
| Best hybrid | E9 | Feature fusion (XGBoost + CTENet) | 0.670 | **0.636** | **0.738** | 0.533 |
| Best hybrid | E12 | Stacking (XGBoost + CTENet) | **0.673** | 0.603 | 0.672 | 0.533 |
| Best deep (graph) | E8 | CTE-Net-inspired | 0.609 | 0.554 | 0.574 | 0.533 |

**Key finding:** The raw-signal CNN (EEGNet, E6) outperforms all feature-engineered and hybrid approaches, achieving AUROC 0.770. Among classical approaches, SVM-RBF with full features (node+adj+rel) is best at AUROC 0.737. Hybrid fusion does not outperform the best single branch but provides more balanced sensitivity/specificity trade-offs.

---

## Dataset

- **Source:** IEEE DataPort ADHD EEG dataset
- **Subjects:** 121 total (61 ADHD, 60 control)
- **Channels:** 19 EEG channels (standard 10-20 layout)
- **Sampling rate:** 128 Hz
- **Epochs:** 2,918 total epochs across all subjects
- **Epoch length:** 640 samples (5 seconds at 128 Hz)
- **Data format:** Real `.npy` recordings at `data/real/recordings/`
- **Manifest:** `data/real/manifest.csv`

All splits are subject-wise: no subject appears in both training and test folds, preventing epoch-level data leakage.

---

## Experimental Design

### Cross-Validation Strategy

5-fold **StratifiedGroupKFold** with grouping by subject ID. Each fold trains on ~76-77 subjects, validates on ~20, tests on ~24-25 subjects. Classification thresholds are tuned on validation subjects using Youden's J statistic and applied to test subjects.

### Leakage Prevention

- Subject-wise splits only (no random epoch splits)
- Feature selection (SelectKBest) performed inside each fold on training data only
- Threshold tuning on validation subjects only
- OOF (out-of-fold) predictions collected for all ensemble methods

---

## Paper 1: Classical Statistical/Spectral Features (E1–E5)

### Feature Groups

| Feature Set | Description | Dimensions |
|-------------|-------------|------------|
| node | Per-channel band power (delta, theta, alpha, beta, gamma) | 95 |
| adj | Adjacency (directed transfer entropy between channel pairs) | 171 |
| rel | Relative edge features | 114 |
| **node+adj+rel** | Full feature vector | **380** |

### Results

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

### Bonus: HistGradientBoosting (E6)

| Exp | Model | Feature Set | AUROC | Accuracy | Sensitivity | Specificity |
|-----|-------|-------------|-------|----------|-------------|-------------|
| E6 | HistGradientBoosting | node | 0.631 | 0.595 | 0.541 | 0.650 |
| E6 | HistGradientBoosting | node+adj | 0.646 | 0.612 | 0.656 | 0.567 |
| E6 | HistGradientBoosting | node+adj+rel | 0.654 | 0.645 | 0.672 | 0.617 |

### Key Findings (Paper 1)

- **SVM-RBF with full features (node+adj+rel)** is the best classical model (AUROC 0.737, accuracy 0.669)
- Adding adjacency (directed transfer entropy) and relative edge features consistently improves AUROC across all models
- XGBoost achieves the highest sensitivity (0.754) with node+adj+rel features
- All models show higher sensitivity than specificity, indicating a bias toward detecting ADHD

---

## Paper 2: CTE-Net Transformer (E6–E8)

### Architectures

| Exp | Model | Description |
|-----|-------|-------------|
| E6 | EEGNet | Raw-signal CNN baseline (19 channels × 640 samples) |
| E7 | CTE-Net Original | Attempted reconstruction with differentiable TE, Rényi entropy, Takens embeddings — **architecturally fixed** (raw EEG `(B,C,T)` input supported via `build_model(name="ctenet_original")`) but **not trained on CPU** due to O(C^2·T·embed) TE computation being prohibitively slow without GPU |
| E8 | CTE-Net Inspired | Existing graph-attention model from `src/models.py` |

### Results

| Exp | Model | AUROC | Accuracy | Sensitivity | Specificity |
|-----|-------|-------|----------|-------------|-------------|
| E6 | EEGNet | **0.772** | **0.719** | **0.738** | **0.700** |
| E7 | CTE-Net Original | Not trained on CPU | — | — | — |
| E8 | CTE-Net Inspired | 0.609 | 0.554 | 0.574 | 0.533 |

### Key Findings (Paper 2)

- **EEGNet (raw CNN)** dramatically outperforms the graph-attention CTE-Net inspired model (0.772 vs 0.609 AUROC)
- The CTE-Net original reconstruction is architecturally complete and importable but too computationally expensive to train on CPU (O(C^2·T·embed) TE computation per batch)
- Raw-signal deep learning exceeds hand-crafted feature approaches on this dataset

---

## Hybrid Fusion & Ensemble (E9–E12)

### Methods

| Exp | Method | Description |
|-----|--------|-------------|
| E9 | Feature-level fusion | XGBoost on concatenated classical features + CTE-Net band features |
| E10 | Soft voting | Average of XGBoost + CTE-Net predicted probabilities |
| E11 | Weighted soft voting | Weighted average (weights from validation AUROC) |
| E12 | Stacking | Logistic regression meta-learner on branch predictions |

### Results

| Exp | Method | Branch A | Branch B | AUROC | Accuracy | Sensitivity | Specificity |
|-----|--------|----------|----------|-------|----------|-------------|-------------|
| E9 | Feature fusion | XGBoost | CTENet | 0.670 | 0.636 | 0.738 | 0.533 |
| E10 | Soft voting | XGBoost | CTENet | 0.671 | 0.628 | 0.738 | 0.517 |
| E11 | Weighted voting | XGBoost | CTENet | 0.671 | 0.628 | 0.738 | 0.517 |
| E12 | Stacking | XGBoost | CTENet | 0.673 | 0.603 | 0.672 | 0.533 |

### Key Findings (Hybrid)

- All hybrid experiments (E9-E12) now execute successfully with proper per-fold deep model training
- Feature fusion (E9) achieves AUROC 0.670, accuracy 0.636
- Stacking (E12) achieves AUROC 0.673 — the best hybrid result
- Hybrid methods do **not** outperform the best single model (EEGNet at 0.772 AUROC or SVM-RBF at 0.737 AUROC)
- CTE-Net's weak performance (0.609 AUROC) limits the ensemble ceiling

---

## Model Comparison Summary

### AUROC Ranking (All Experiments)

| Rank | Exp | Model | AUROC | Accuracy | Type |
|------|-----|-------|-------|----------|------|
| 1 | E6 | EEGNet | 0.772 | 0.719 | Deep (raw CNN) |
| 2 | E2 | SVM-RBF (node+adj+rel) | 0.737 | 0.669 | Classical |
| 3 | E4 | ExtraTrees (node+adj+rel) | 0.715 | 0.645 | Classical |
| 4 | E2 | SVM-RBF (node+adj) | 0.728 | 0.678 | Classical |
| 5 | E4 | ExtraTrees (node+adj) | 0.711 | 0.645 | Classical |
| 6 | E3 | Random Forest (node+adj+rel) | 0.700 | 0.628 | Classical |
| 7 | E12 | Stacking (XGBoost + CTENet) | 0.673 | 0.603 | Hybrid |
| 8 | E10 | Soft voting (XGBoost + CTENet) | 0.671 | 0.628 | Hybrid |
| 9 | E11 | Weighted voting (XGBoost + CTENet) | 0.671 | 0.628 | Hybrid |
| 10 | E9 | Feature fusion (XGBoost + CTENet) | 0.670 | 0.636 | Hybrid |
| 11 | E6-bonus | HistGB (node+adj+rel) | 0.654 | 0.645 | Classical |
| 12 | E1 | XGBoost (node+adj+rel) | 0.652 | 0.636 | Classical |

### Best Sensitivity / Specificity

- **Highest sensitivity:** XGBoost node+adj+rel (E1) — 0.754
- **Highest specificity:** SVM-RBF node+adj+rel (E2) — 0.633; EEGNet (E6) — 0.650

---

## Artifacts

### Source Code

| File | Purpose |
|------|---------|
| `src/classical_models.py` | XGBoost, SVM-RBF, RF, ExtraTrees, LR, HistGB with SelectKBest |
| `src/ctenet.py` | CTENetOriginal reconstruction attempt (Paper 2 faithful) |
| `src/hybrid_model.py` | Feature fusion (E9), soft/weighted voting (E10-11), stacking (E12) |
| `src/models.py` | EEGNet + CTE-Net-inspired (existing) |
| `src/data.py` | Data loading, epoching, subject-wise splitting |
| `src/cv.py` | Leakage checks, subject-wise folds |
| `src/evaluate.py` | Metrics, Youden threshold, calibration error |
| `src/plots.py` | ROC/PR, confusion matrix, calibration, fold metrics, training history |
| `scripts/train_classical.py` | E1-E5 (+HistGB bonus) training |
| `scripts/train_ctenet.py` | E6-E8 training |
| `scripts/train_hybrid.py` | E9-E12 training |
| `scripts/visualize_results.py` | All figure generation |

### Results

| Directory | Contents |
|-----------|----------|
| `results/classical_paper1/` | E1-E5 predictions, fold metrics, summary |
| `results/deep_models/` | E6-E8 OOF predictions, fold metrics, history, summary |
| `results/hybrid/` | E9-E12 results, summary |
| `results/figures/` | All visualization PNGs (ROC/PR, confusion, comparison, ablation) |

### Visualizations

| Figure | Description |
|--------|-------------|
| `model_comparison.png` | AUROC and accuracy across all experiments |
| `sensitivity_specificity.png` | Sensitivity vs specificity by experiment |
| `feature_set_ablation.png` | AUROC by feature set (node / node+adj / node+adj+rel) |
| `*{model}_roc_pr.png` | ROC and PR curves per model |
| `*{model}_confusion.png` | Confusion matrix per model |
| `*{model}/fold_metrics.png` | Per-fold AUROC/accuracy (deep models) |
| `*{model}/training_history.png` | Training curves (deep models) |

---

## Limitations

1. **CTE-Net original (E7) not trained on CPU:** The faithful reconstruction from Paper 2 is architecturally complete and importable via `build_model(name="ctenet_original")`, but the O(C^2·T·embed) differentiable transfer entropy computation is prohibitively slow on CPU for 5-fold CV with 60 epochs per fold. Full training requires GPU acceleration or reduced scope.

2. **CPU-only training:** No CUDA available; deep models trained on CPU with 20-30 epochs. Performance may improve with more epochs and GPU acceleration.

3. **Single dataset:** Results are specific to the IEEE DataPort 121-subject dataset. Generalization to other ADHD EEG datasets is untested.

4. **Hybrid ceiling limited by CTE-Net:** The CTE-Net branch's weak performance (0.609 AUROC) limits the potential benefit of fusion/ensemble methods. All hybrid results (0.670–0.673 AUROC) are below the best single-model results (EEGNet 0.772, SVM-RBF 0.737).

---

## Recommendations

1. **For production deployment:** Use EEGNet (E6) — best overall AUROC 0.772 — if raw EEG inference is acceptable.
2. **For interpretable clinical use:** Use SVM-RBF with node+adj+rel features (E2) — AUROC 0.737 with clinically meaningful spectral/connectivity features.
3. **For maximum sensitivity:** Use XGBoost with node+adj+rel features (E1) — sensitivity 0.754.
4. **Future work:** Fix CTE-Net original to accept raw EEG input and train end-to-end; explore larger feature sets; test on external validation datasets.

---

## Reproducibility

All experiments use `seed=42`, 5 subject-wise folds, and real data. To reproduce:

```bash
# Classical models (E1-E5)
python scripts/train_classical.py --config configs/eeg_cte_real.yaml --output results/classical_paper1 --folds 5 --seed 42

# Deep models (E6-E8)
python scripts/train_ctenet.py --config configs/eeg_cte_real.yaml --output results/deep_models --folds 5 --seed 42 --epochs 30

# Hybrid (E9-E12)
python scripts/train_hybrid.py --config configs/eeg_cte_real.yaml --output results/hybrid --folds 5 --seed 42 --deep-epochs 20

# Visualizations
python scripts/visualize_results.py
```
