# Data layout

The pipeline expects one recording per subject plus a `manifest.csv`.

## Manifest

```csv
subject_id,label,site,file
sub-001,1,A,sub-001.npy
sub-002,0,A,sub-002.npy
```

| Column | Required | Meaning |
|---|---|---|
| `subject_id` | yes | Unique subject identifier. Used for grouping splits. |
| `label` | yes | `1` = ADHD, `0` = control. Also accepts `ADHD`/`control`, `case`/`HC`, etc. |
| `site` | no | Scanner or recruitment site. Not used by the default pipeline, but keep it for site-wise analysis. |
| `file` | no | Relative path under the data root. If omitted, the loader looks for `<subject_id>.npy`, `.csv`, `.txt`, `.mat`, or `.set`. |

## Recording formats

| Format | Shape | Notes |
|---|---|---|
| `.npy` | `(n_times, n_channels)` or `(n_channels, n_times)` | Preferred. Orientation is inferred, or you can set `n_channels` in the config. |
| `.csv` / `.txt` | rows = time, columns = channels | A `time`/`timestamp` column is dropped automatically. |
| `.mat` | any nested numeric array | The largest numeric array is used. MATLAB structs are searched automatically. If one file holds several subjects, split them into one file per subject first. |
| `.set` | EEGLAB | Requires `pip install mne`. |

Recordings are resampled to `data.fs` (default 128 Hz), converted to microvolts, notch-filtered, band-passed, and split into 5-second epochs. Set `data.n_channels` to match your montage.

## Public datasets

### Start here: the 121-child EEG dataset

- **Link:** <https://ieee-dataport.org/open-access/eeg-data-adhd-control-children>
- **DOI:** 10.21227/rzfh-zn36
- **Contents:** 61 ADHD / 60 controls, ages 7–12, 19 channels, 128 Hz, visual attention task, `.mat` files.
- **Access:** open access with a **free IEEE account** (no membership needed). CC BY 4.0.
- **Why:** this is the dataset behind most 98–100% EEG papers and the leakage example in the review. It is small enough to run on a laptop and lets you compare directly against published subject-wise results.
- **Channel order:** Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T7, T8, P7, P8, Fz, Cz, Pz.

**Preparing it** — extract the four archives so that `data/real/raw/` contains `ADHD_part1/`, `ADHD_part2/`, `Control_part1/`, and `Control_part2/`, then run:

```bash
python scripts/prepare_ieee_dataset.py
```

This writes one float32 `.npy` per child to `data/real/recordings/` and creates `data/real/manifest.csv` (121 rows, 61/60). Then:

```bash
python scripts/run_cv.py --config configs/eeg_cte_real.yaml --quick   # smoke test
python scripts/run_cv.py --config configs/eeg_cte_real.yaml           # full 5-fold
```

The int16 values in this dataset are in an undocumented scale, so the microvolt threshold is disabled; epochs are rejected in z-score units instead (`max_abs_z: 10`, about 87% kept).

### Other EEG / multimodal options

- **BALLADEER ADHD** — EEG + eye tracking + electrodermal activity, children and adolescents, 12.7 GB. <https://ieee-dataport.org/documents/balladeer-adhd> · DOI 10.21227/nevp-3a70. Check the access tier before planning around it.
- **103-child Iranian EEG dataset** — 49 ADHD / 54 controls, ages 6–10. <https://ieee-dataport.org/documents/eeg-data-adhdcontrol-children>. Useful as a second, independent EEG cohort.
- **Healthy Brain Network (HBN)** — community sample with EEG, MRI, eye tracking, and questionnaires, including many ADHD participants. <http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/MRI_EEG.html>. Also mirrored on OpenNeuro/COINS.

### Neuroimaging

- **ADHD-200** — 947 datasets from 8 sites, 776 with ADHD phenotypes (285 ADHD / 491 controls). Raw data: <http://fcon_1000.projects.nitrc.org/indi/adhd200/> (free NITRC account). Preprocessed: <http://preprocessed-connectomes-project.org/adhd200/download.html>.
- **ABCD** — 11,880 children, 21 sites, multimodal. <https://nda.nih.gov/abcd>. Requires an NDA account and a data use certification.
- **NeuroIMAGE** — 1,069 children from ADHD and control families, multimodal. Access by collaboration request through Radboud University.

### Rules

- Keep raw data out of git (`data/` is ignored).
- Verify the licence and any ethics/DUA requirements before downloading.
- Never split epochs randomly. Use the subject-wise splits in `src/data.py`.
- If you use more than one site, report site-wise metrics and consider leave-one-site-out validation.
