# PhysioNet Challenge 2026 — Technical documentation (team pipeline)

This document explains **what data** the solution uses, **which signals** become features, **how they are processed**, and **how the model is trained and run**, aligned with `team_code.py` **feature_version `v23`**. It is written so a newcomer can follow the code without guessing.

---

## Document language

The body is in **English** for stable UTF-8 storage and international review. The sections below are the **full** technical reference (data, signals, indices, training, inference, env vars). For a Korean version, export this file and run it through a translation tool, or maintain a team translation separately (avoid duplicating indices by hand).

---

## 1. End-to-end pipeline

1. **Inputs**: Challenge BIDS-style layout — demographics CSV, physiological EDF, CAISR algorithmic EDF, plus `channel_table.csv` for naming/bipolar rules (`helper_code`).
2. **Per-record base features**: `extract_all_features()` returns **`N_BASE_FEATURES = 337`** floats (hand-crafted sleep/EEG/ECG/resp features).
3. **Deep embedding**: A **1D-CNN** on a **C3–M2** raw clip produces **`CNN_EMB_DIM` (default 32)** dimensions, trained with **weighted BCE** (same `sample_weight` as trees).
4. **Philosopher’s Stone (PS)**: **4 scalar cognition scores + 50 PCA components** of a **1024-D** latent → **`N_PS_FEATS = 54`**.
5. **Concatenation**: **`337 + 32 + 54 = 423`** columns for training/inference.
6. **Deliberate feature suppression**: Coherence block **[P]** and site one-hot in **[A]** are **zeroed** (always). Optional **`ABLATE_BLOCKS`** zeros other index ranges for experiments.
7. **Site-wise z-normalisation**: Selected amplitude-sensitive indices (bands, spindle amplitudes, complexity, slow-wave amplitudes/slopes) are **z-scored per `SiteID`**; unknown sites use **`__global__`** stats saved in the model.
8. **Hyperparameters + feature count**: **Optuna** tunes LightGBM hyperparameters and **`fs_pct`** (feature-importance percentile threshold) using a **leave-one-site-out style** objective.
9. **Feature mask**: **K-fold mean LightGBM importances** (no full-data leakage for FS) → boolean **`feat_mask`**.
10. **Classifier ensemble**: Default **`CLASSIFIER=trees`** blends **LightGBM + XGBoost + ExtraTrees** probabilities. **`mlp`** / **`hybrid`** add an **MLP** head (MLP fit **without** `sample_weight` due to sklearn API).
11. **Artifact**: `joblib` bundle, default filename **`challenge_model_v23.joblib`** (overridable via env).

---

## 2. Data sources per record

| Source | Role | Code |
|--------|------|------|
| Demographics CSV (`DEMOGRAPHICS_FILE`) | Patient/session/site labels, diagnosis, demographics, **Time_to_Event** | `find_patients`, `load_demographics`, `load_diagnoses`, TTE weight lookup |
| `{site}/{patient}_ses-{session}.edf` | EEG, ECG, etc. | `load_signal_data`, `_build_channel_dict` |
| `{site}/{patient}_ses-{session}_caisr_annotations.edf` | CAISR stages & derived signals | `stage_caisr`, CAISR probability features |
| `channel_table.csv` | Rename + bipolar construction | `_build_channel_dict` |

**Sleep stage encoding** (`stage_caisr`, ~1/30 Hz, one value per 30 s epoch):

`1=N3, 2=N2, 3=N1, 4=REM, 5=Wake, 9=Unavailable` (see `S_N3`, … in code).

---

## 3. Channel selection and standardisation

Raw channel names differ across sites. The code:

1. Applies rename rules (`load_rename_rules`, `standardize_channel_names_rename_only`).
2. Builds bipolar pairs via **`BIPOLAR_RULES`** (e.g. C3–M2 → key **`c3-m2`**).
3. Uses fixed EEG slots **`EEG_SLOT = {'c3-m2':0, 'c4-m1':1, 'f3-m2':2}`** so **feature indices are stable** across cohorts.

**Design intent**: central/centro-parietal/frontal bipolar EEG are standard in sleep–cognition literature; fixing slots avoids “column drift” when a site lacks a channel (missing derivations yield zeros in downstream feature functions).

---

## 4. Base feature vector v23 (337 dims)

Indices are **0-based**; intervals are **inclusive start, exclusive end** where noted.

| Block | Dim | Index (approx.) | Summary |
|-------|-----|-----------------|---------|
| [A] Demographics | 13 | 0–12 | Age, sex one-hot, race, BMI, **site one-hot (3)** — **zeroed at train & inference** |
| [B] Sleep macro | 12 | 13–24 | TST, SE, WASO, stage %, REM latency, transitions, TIB, … |
| [C] EEG bandpower | 90 | 25–114 | 3 channels × 5 stages × 6 bands, **log1p** |
| [D] Resp / SpO2 | 8 | 115–122 | AHI, arousal index, SpO2 stats, ODI, … |
| [E] ECG HRV | 9 | 123–131 | **6 time-domain** + **3 frequency-domain** (log LF, log HF, LF/HF) — v23 extension |
| [F] CAISR stage probs | 5 | 132–136 | |
| [G] Custom spindles | 18 | 137–154 | N2 sigma envelope features (density, amplitude, duration, ratios, …) |
| [H] SO–spindle coupling | 3 | 155–157 | delta–sigma PAC (MVL) |
| [I] EEG complexity | 12 | 158–169 | Hjorth + spectral entropy (N2/N3/REM, C3-M2) |
| [J] Sleep fragmentation | 5 | 170–174 | |
| [K] SEF | 3 | 175–177 | |
| [L] Waveform kurtosis | 6 | 178–183 | |
| [M] Band kurtosis | 36 | 184–219 | |
| [N] N3 spectral ratios | 3 | 220–222 | |
| **[O] Custom slow waves** | **15** | **223–237** | **N2+N3, 0.5–2 Hz, zero-crossing SO candidates: density, \|neg\|, pos, PTP, slope (per channel)** |
| [P] EEG coherence | 75 | 238–312 | **Always zeroed in training & inference** (intentional drop) |
| [Q] REM spectral ratios | 9 | 313–321 | |
| [R] REM SFAR | 3 | 322–324 | |
| [T] Half-night spectral | 12 | 325–336 | |

**Total**: `N_BASE_FEATURES = 337`.

**Note**: Some header comments in `team_code.py` may still mention older layouts (e.g. v22, “319”). The **authoritative** definition is **`N_BASE_FEATURES`, `FEATURE_VERSION`, and `extract_all_features()` concatenation order**.

---

## 5. Why certain signals are emphasised or dropped

- **[C], [G], [I], [O] (amplitude parts)**: Absolute EEG-derived magnitudes depend on amplifier gain, impedance, and referencing → **site z-norm** on listed index sets.
- **[P] coherence**: Potentially informative but often **site/hardware sensitive** → **hard zero** to reduce domain shortcutting.
- **[A] site one-hot**: Risk of **site–label leakage** on external sites → **hard zero**.
- **[E] HRV frequency**: Captures sympathovagal balance from resampled RR PSD (Welch); rationale noted in code comments (clinical literature pointers).
- **[O] slow waves**: NREM slow oscillation morphology linked to cognition in prior work; detector uses band-pass + zero-crossing + duration/amplitude gates.

---

## 6. CNN branch (32-D by default)

- **Segment**: `extract_eeg_cnn_segment` — **C3–M2**, length **`CNN_INPUT_LEN` (8192 samples default)**, centre crop or **zero-pad**, then **global z-score** of the clip.
- **Architecture**: `EEGMiniCNN` — stacked **Conv1d** → **AdaptiveAvgPool1d** → **Linear** embedding + auxiliary logit for BCE.
- **Training**: `CNN_EPOCHS` full-batch Adam steps on **CPU**, **weighted BCE** with `sample_weight`.
- **Persistence**: `cnn_state` (`state_dict`); inference uses `_cnn_embed_from_state`.

---

## 7. Philosopher’s Stone branch (54-D)

- **Scalars**: `brain_health_score`, `total_cognition_score`, `fluid_cognition_score`, `crystallized_cognition_score`.
- **Latents**: 1024-D `lhl_*` columns when available from **`ps_cache_baked.csv`** or live **`run_philosopher`**.
- **Training**: Fit **`PCA(n_components=50, random_state=42)`** on rows with non-zero latents; transform all rows → **50-D**; concatenate with 4 scalars.
- **Inference**: Prefer baked cache; else run PS on EDF if model loads; apply saved **`ps_pca`**.

---

## 8. Post-concatenation processing (training)

On **`X_arr`** of shape `(N, 423)`:

1. Zero **coherence** columns **`238:313`**.
2. Zero **site one-hot** **`10:13`**.
3. Optional **`ABLATE_BLOCKS`**: keys `T`, `REMRAT`, `SFAR`, `SPIN`, `SW` → slice table `_ABLATE_SLICES`.
4. **Site normalisation** on **`_SNORM_IDX`**: bandpower **[C]**, spindle block **[G]**, complexity **[I]**, and **slow-wave amplitude/slope indices** in **[O]** (density excluded from norm list — see code comments).

Store **`site_norm`** dict (`SiteID` → `(mean, std)` plus **`__global__`**) and **`site_norm_idx`** in the joblib bundle.

---

## 9. Sample weights

1. **Time-to-event shaping** for **cognitive impairment positives** (`y==1`): if enough valid TTE values, up-weight **earlier converters** (linear map between 5th–95th percentile of TTE).
2. **Per-site balancing** (`BALANCE_SITE_WEIGHT=1` default): rescale weights so **each `SiteID` contributes equal total mass** — reduces majority-site dominance in boosted trees and CNN.

---

## 10. Optuna (LightGBM + `fs_pct`)

- **Search space**: `num_leaves`, `min_child_samples`, `reg_lambda`, `reg_alpha`, `colsample_bytree`, `subsample`, **`fs_pct`** ∈ [40, 75].
- **Inner LOSO loop** (per trial): for each qualifying held-out site, fit **FS LightGBM** on train → importance threshold at **`fs_pct`** percentile → fit **task LightGBM** on masked features → AUROC on held-out site.
- **Objective**: `LOSO_OBJECTIVE=blend` → `(1-w)*mean(LOSO AUROCs) + w*min(...)`, `w=LOSO_MIN_WEIGHT` (default 0.5). `min` mode uses only the minimum site AUROC.
- **Trials**: `OPTUNA_N_TRIALS` (default **80**; can lower for smoke tests).

---

## 11. Feature selection (no full-data leakage)

- For each CV fold, fit LightGBM on **train indices only**, accumulate **feature_importances_**.
- Average importances across folds → threshold at **`_best_fs_pct`** percentile from Optuna → **`feat_mask`**.
- If too few features, fallback to 30th percentile threshold.

---

## 12. Reported CV and LOSO

- **CV splitter**: **`StratifiedGroupKFold`** by **`SiteID`** when `GROUP_CV_BY_SITE=1` (default) — avoids optimistic same-site leakage between train/val in metrics.
- **Per-fold metric**: `_run_fold` trains **LGBM (+optional dual LGBM)**, **XGB**, **ExtraTrees** on **masked** features, then blends probabilities **identically** to `_tree_prob` at inference.
- **LOSO prints**: For each site with enough train/test support, same `_run_fold` on **all-other-sites → this-site**.

Console lines: `[CV] 5-fold CV AUROC`, `[CV] LOSO AUROC (min=..., cross-site estimate)`.

---

## 13. Final training and saved models

- **Full-data fits** on **`X_sel = X_arr[:, feat_mask]`** with **`sample_weight`** for **LGBM**, optional **second LGBM**, **XGB**, **ExtraTrees**.
- **MLP** (optional): `StandardScaler` on `X_sel`; `MLPClassifier` **without** sample weights.
- **Dump** via `joblib.dump({...})` including **`feat_mask`, `site_norm`, `ps_pca`, `cnn_state`, `classifier_mode`, `ablate_blocks`**, tree models, etc.

---

## 14. Inference (`run_model`)

1. Rebuild **423-D** row: `extract_all_features` + CNN embed + `_get_ps_features`.
2. Assert dimension **`N_BASE_FEATURES + CNN_EMB_DIM + N_PS_FEATS`**.
3. Apply **same hard zeros** (coherence, site one-hot) and **saved ablations**.
4. Apply **site z-norm** using record’s **`site_id`** or **`__global__`** fallback.
5. Apply **`feat_mask`**.
6. Compute probability:
   - **`trees`**: `_tree_prob()` blend.
   - **`mlp`**: MLP only (if present).
   - **`hybrid`**: average of `_tree_prob()` and MLP probability.
7. Threshold **0.5** for binary label.

---

## 15. Environment variables (operational)

| Variable | Purpose |
|----------|---------|
| `CLASSIFIER` | `trees` / `mlp` / `hybrid` |
| `ABLATE_BLOCKS` | Comma list: `T`, `REMRAT`, `SFAR`, `SPIN`, `SW` |
| `GROUP_CV_BY_SITE` | `1` (default) or `0` |
| `BALANCE_SITE_WEIGHT` | `1` default |
| `LOSO_OBJECTIVE`, `LOSO_MIN_WEIGHT` | Optuna objective |
| `OPTUNA_N_TRIALS` | Default 80 |
| `USE_DUAL_LGBM` | `1` enables second LGBM seed |
| `CNN_INPUT_LEN`, `CNN_EMB_DIM`, `CNN_EPOCHS`, `CNN_LR` | CNN |
| `MODEL_FILENAME` / `CHALLENGE_MODEL_FILE` | Load/save basename |

---

## 16. Caching

- Per-record NPZ under `model_folder/feature_cache/{pid}_ses-{ses}_{FEATURE_VERSION}.npz`.
- If stored size ≠ `N_BASE_FEATURES`, it is **deleted** and recomputed.

---

## 17. Implementation notes vs header comments

- The file header still mentions a “5-model stacking ensemble” with a meta-learner; **the implemented deployment blend** is the **LGBM (+optional dual) + XGB + ExtraTrees** probability mix (and optional MLP/hybrid). Treat **this document and `train_model` / `run_model`** as ground truth.
- `extract_all_features` docstring may still say “319”; **actual** is **`N_BASE_FEATURES=337`** for v23.

---

## 18. One-sentence summary

**Hand-craft 337 interpretable sleep/EEG/ECG features, append a supervised 32-D CNN embedding from C3–M2 and 54-D Philosopher’s Stone features, suppress site/coherence shortcuts, site-normalise amplitude-sensitive EEG-derived blocks, select features with Optuna-tuned importance cut-offs under grouped CV, then blend gradient-boosted trees with extra randomized trees (and optionally an MLP) to output a cognition-risk probability.**

---

*Canonical reference: `team_code.py` (`FEATURE_VERSION = 'v23'`). Update this file when feature indices or training logic change.*

---

## Appendix: checklist for reviewers (quick audit)

- [ ] Base feature count is **337** (`N_BASE_FEATURES`) and concatenation is **337 + 32 + 54 = 423**.
- [ ] Coherence columns **238:313** and site one-hot **10:13** are **zeroed** in both `train_model` and `run_model`.
- [ ] Site z-normalisation indices **`site_norm_idx`** match **`_SNORM_IDX`** in code.
- [ ] Saved **`feat_mask`** is applied before ensemble / MLP.
- [ ] **`classifier_mode`** matches training (`trees` / `mlp` / `hybrid`).
- [ ] Optional **`ABLATE_BLOCKS`** at train time is stored as **`ablate_blocks`** and replayed at inference.
