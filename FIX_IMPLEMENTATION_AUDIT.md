# FIX_IMPLEMENTATION_AUDIT.md — ADHD Hybrid EEG Research Project

**Repository:** https://github.com/Kalpit-Goyal/ADHD-
**Latest commit audited:** `810e4f82515eaa30f85c6999f210b040b6c6b461`
**Audit date:** 2026-11-22

This document catalogues every critical bug found during the repository audit,
why it is incorrect, the exact correction required, and how the correction will
be tested. Previous results are preserved as `SUPERSEDED` and are not
overwritten.

---

## 1. `scripts/train_hybrid.py` — Deep model never trained (E9 blocker)

**File:** `scripts/train_hybrid.py:115`
**Function:** `main()`
**Existing behaviour:** `deep_model = None` is hard-coded. The deep-learning
model is never instantiated, trained, or assigned. `None` is passed to
`run_feature_fusion()`, which then substitutes zero-filled deep features.
**Why incorrect:** E9 feature-level fusion does not actually combine the
classical and deep-learning branches. It concatenates XGBoost probabilities
with zeros, producing invalid fusion results.
**Exact correction:**
1. Train a deep model (CTE-Net-inspired or original) per fold using training
   subjects only, with validation for early stopping.
2. Load the best validation checkpoint.
3. Pass the trained model to `run_feature_fusion()` instead of `None`.
4. If deep model training fails, raise an informative exception — do not
   silently continue with zeros.
**How tested:** Unit test verifies `_deep_features()` returns non-zero,
correctly-dimensioned, subject-aligned representations from a trained model.

---

## 2. `src/hybrid_model.py` — Zero-filled deep features silently accepted

**File:** `src/hybrid_model.py:174-175`
**Function:** `_deep_features()`
**Existing behaviour:** When a subject has no extracted deep features, zeros are
silently inserted:
```python
aligned.append(np.zeros(d_model, dtype=np.float32))
```
**Why incorrect:** Missing deep features are indistinguishable from learned
near-zero representations. This silently corrupts fusion inputs and masks
model failures.
**Exact correction:** Raise an informative exception when any expected subject
is missing from extracted deep features. Never substitute zeros.
**How tested:** Unit test verifies exception is raised when a subject is
missing; verifies all expected subject IDs are present in output.

---

## 3. `src/hybrid_model.py` — Stacking data leakage (E12 invalid)

**File:** `src/hybrid_model.py:443-451`
**Function:** `run_prediction_ensemble()` (method="stacking")
**Existing behaviour:**
```python
meta.fit(stack_X, subject_labels)
combined = meta.predict_proba(stack_X)[:, 1]
```
The meta-classifier is trained on OOF predictions and immediately evaluated on
the **same** `stack_X`. The meta-model sees its training data at evaluation time.
**Why incorrect:** This is classic data leakage. Reported stacking metrics are
optimistically biased and not scientifically valid.
**Exact correction:** Implement proper nested stacking:
1. Outer training participants are split into inner subject-wise folds.
2. Each base model is trained on inner-train participants, predicts inner-val.
3. OOF predictions for every outer training participant are collected.
4. Meta-classifier is trained on these OOF predictions.
5. Base models are retrained on full outer-train participants.
6. Meta-classifier predicts on outer-test base-model predictions.
**How tested:** Unit test verifies meta-model is never fit/evaluated on the
same prediction rows (no in-sample stacking features).

---

## 4. `scripts/train_hybrid.py` — Classical probs dict overwrites (E10-E12)

**File:** `scripts/train_hybrid.py:158`
**Existing behaviour:**
```python
test_subject_probs = dict(zip(bundle.subject_ids[test_index].tolist(), test_prob.tolist()))
```
If a participant has multiple epochs in `test_index`, later epoch probabilities
overwrite earlier ones in the dict.
**Why incorrect:** Participants with multiple epochs lose all but the last
epoch's probability. This is not participant-level aggregation — it is
arbitrary epoch selection.
**Exact correction:** Use `aggregate_subject_probs()` (already exists in
`src/train.py:92`) to average epoch probabilities per participant before
storing in `branch_a_probs` / `branch_b_probs`.
**How tested:** Unit test verifies participant aggregation produces arithmetic
mean, not last-epoch overwrite.

---

## 5. `scripts/train_hybrid.py` — Deep model folds not aligned with hybrid folds

**File:** `scripts/train_hybrid.py:52-65` (`_collect_oof_probs`)
**Existing behaviour:** `_collect_oof_probs()` calls `run_cross_validation()`,
which internally calls `subject_folds()` to create its own folds. These folds
are independent of the `folds` supplied to the hybrid pipeline. The deep-model
OOF predictions do not align with the classical-model folds.
**Why incorrect:** E10-E12 combine predictions from mismatched folds. Subject
X may be in fold 1 test for XGBoost but fold 3 test for CTE-Net. Ensemble
metrics are meaningless.
**Exact correction:** Train the deep model using the exact same `folds` list
passed to the hybrid pipeline. Extract per-fold OOF predictions aligned to
those fold boundaries. Do not call `run_cross_validation()` independently.
**How tested:** Unit test verifies fold alignment — same participant IDs in
same fold across all branches.

---

## 6. `src/hybrid_model.py` — No participant alignment check in `_deep_features`

**File:** `src/hybrid_model.py:120-178`
**Function:** `_deep_features()`
**Existing behaviour:** Features are extracted per epoch and averaged per
subject, but there is no verification that the extracted representation
corresponds to the correct `subject_ids_subset` entry.
**Why incorrect:** A batching or indexing error could assign subject A's
features to subject B without detection.
**Exact correction:** Add strict alignment check: verify every subject in
`subject_ids_subset` has a non-empty feature list, and that the output row
order matches `subject_ids_subset` order.
**How tested:** Unit test verifies subject alignment and non-zero outputs.

---

## 7. `src/hybrid_model.py` — `save_hybrid_result()` does not save trained pipeline

**File:** `src/hybrid_model.py:505-533`
**Function:** `save_hybrid_result()`
**Existing behaviour:** Saves only a metadata JSON. No trained models, scalers,
feature selectors, fusion classifiers, or ensemble weights are persisted.
**Why incorrect:** No deployable hybrid inference pipeline exists. `load_hybrid_model()`
and `predict_hybrid()` cannot be implemented.
**Exact correction:** Save trained classical model (joblib), deep model
state_dict (torch), feature scaler (joblib), fusion classifier (joblib),
ensemble weights (JSON), feature names, channel order, preprocessing config,
and model architecture config.
**How tested:** Unit test verifies save/reload produces identical predictions.

---

## 8. `src/models.py` — Original CTE-Net not integrated into training pipeline

**File:** `src/models.py:200-230` (`build_model`)
**Existing behaviour:** `build_model()` only supports `eegnet` and `cte_net`
(inspired). `CTENetOriginal` from `src/ctenet.py` is not registered.
**Why incorrect:** E7 (original CTE-Net) cannot be trained through the
standard pipeline. The model exists in `src/ctenet.py` but is unreachable.
**Exact correction:** Add `ctenet_original` to `build_model()` that constructs
`CTENetOriginal` with raw EEG input `(B, C, T)`.
**How tested:** Unit test verifies `build_model()` returns a `CTENetOriginal`
instance that accepts raw EEG input and produces `(B,)` logits.

---

## 9. `requirements.txt` — Missing `xgboost` dependency

**File:** `requirements.txt`
**Existing behaviour:** `xgboost` is not declared. `src/classical_models.py`
imports it with a try/except, silently degrading if absent.
**Why incorrect:** XGBoost is required for E1 and E9. Clean installation may
lack it.
**Exact correction:** Add `xgboost>=1.7` to `requirements.txt`.
**How tested:** `pip install -r requirements.txt` succeeds and `import xgboost`
works.

---

## 10. `src/hybrid_model.py` — `run_feature_fusion` deep model not trained per fold

**File:** `src/hybrid_model.py:181-387`
**Function:** `run_feature_fusion()`
**Existing behaviour:** The function receives `deep_model` as an argument but
does not train it. The caller passes `None`. Even if a model were passed, it
would be the same model instance across folds, not retrained per fold on
training subjects only.
**Why incorrect:** Feature fusion uses either zeros or an untrained/shared
model, not per-fold trained deep representations.
**Exact correction:** `run_feature_fusion` should accept a model constructor
and config, train a fresh deep model per fold on train subjects, use val for
early stopping, load best checkpoint, extract features for train/val/test.
**How tested:** Integration test verifies deep model is trained per fold and
features are non-zero.

---

## 11. `src/classical_models.py` — `evaluate_model` epoch-level prob assignment

**File:** `src/classical_models.py:264`
**Function:** `evaluate_model()`
**Existing behaviour:**
```python
probs[test_index] = model.predict_proba(features[test_index])[:, 1]
```
This assigns epoch-level probabilities to an epoch-sized array. While
`_subject_probs()` later averages them, the intermediate `probs` array is
epoch-sized and the fold_thresholds are looked up by subject. The mapping
between epoch probabilities and fold thresholds is indirect.
**Why incorrect:** Confusing epoch-level vs participant-level data flow.
Potential for mismatch if a subject's epochs span different indices than
expected.
**Exact correction:** Keep participant-level probabilities throughout
`evaluate_model`. Use `aggregate_subject_probs()` explicitly at each stage.
**How tested:** Unit test verifies participant-level aggregation correctness.

---

## 12. `src/cv.py` — `subject_folds` vs `make_repeated_folds` inconsistency

**File:** `src/data.py:535-590` (`subject_folds`) and `src/cv.py:201-257` (`make_repeated_folds`)
**Existing behaviour:** Two different fold-creation functions exist with
different signatures and return types. `subject_folds` returns
`list[tuple]`, `make_repeated_folds` returns `list[Fold]`. Different scripts
use different functions, creating inconsistency.
**Why incorrect:** Experiments may use different fold structures, breaking
cross-experiment alignment.
**Exact correction:** Unify on `Fold` dataclass from `src/cv.py`. All scripts
use `make_repeated_folds()` or `load_folds()` to produce `list[Fold]`.
**How tested:** Unit test verifies all experiments consume the same `list[Fold]`.

---

## 13. `src/ctenet.py` — `DifferentiableTransferEntropy._embed` uses wrong variable

**File:** `src/ctenet.py:252`
**Function:** `DifferentiableTransferEntropy._embed()`
**Existing behaviour:**
```python
end = n_times - (self.embed_dim - 1) * self.delay if False else n_times - (self.embed_dim - 1) * self.embed_delay
```
The `if False` branch is dead code. `self.delay` is not an attribute of
`DifferentiableTransferEntropy` (it has `self.embed_delay`). If the `if False`
were ever changed, it would raise `AttributeError`.
**Why incorrect:** Dead code referencing non-existent attribute. Confusing.
**Exact correction:** Remove the dead `if False` branch, keep only the correct
`self.embed_delay` computation.
**How tested:** Unit test exercises `_embed()` directly.

---

## Summary of Fixes Required

| # | File | Issue | Severity |
|---|------|-------|----------|
| 1 | `scripts/train_hybrid.py` | `deep_model = None`, never trained | Critical |
| 2 | `src/hybrid_model.py` | Zero-filled features silently accepted | Critical |
| 3 | `src/hybrid_model.py` | Stacking trains/evaluates on same data | Critical |
| 4 | `scripts/train_hybrid.py` | Dict overwrites multi-epoch probs | Critical |
| 5 | `scripts/train_hybrid.py` | Deep model folds not aligned | Critical |
| 6 | `src/hybrid_model.py` | No subject alignment check | High |
| 7 | `src/hybrid_model.py` | No trained model artifacts saved | High |
| 8 | `src/models.py` | Original CTE-Net not in `build_model()` | High |
| 9 | `requirements.txt` | Missing `xgboost` | Medium |
| 10| `src/hybrid_model.py` | Deep model not trained per fold | Critical |
| 11| `src/classical_models.py` | Epoch-level prob assignment confusing | Medium |
| 12| `src/cv.py`/`src/data.py` | Two inconsistent fold APIs | Medium |
| 13| `src/ctenet.py` | Dead code with wrong attribute | Low |
