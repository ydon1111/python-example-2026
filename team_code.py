#!/usr/bin/env python
# team_code.py — PhysioNet Challenge 2026
#
# Feature groups (total = 341):
#   [A]  demographics          13   age, sex×3, race×5, BMI, site×3
#   [B]  sleep macrostructure  12   SL, TST, SE, WASO, stage%, REMlat, trans, TIB
#   [C]  EEG bandpower         90   3ch × 5stages × 6bands (delta–gamma) — log1p
#   [D]  resp / SpO2            8   AHI, arousal_idx, PLMI, SpO2 stats, ODI
#   [E]  ECG HRV                9   mean_RR, SDNN, RMSSD, HR, pNN50, NN_range,   *** v22 ***
#                                   log_LF, log_HF, LF/HF ratio (freq domain)
#   [F]  CAISR probabilities    5   prob_w/n3/n2/n1/r
#   [G]  Custom spindles        18  6 feats × 3ch — sigma-band envelope det.    *** NOVEL ***
#                                   sigma_power, density, amplitude, duration,
#                                   sigma/alpha, sigma/delta (N2 sleep)
#   [H]  SO-spindle coupling    3   delta-sigma PAC (MVL) per EEG channel        *** NOVEL ***
#   [I]  EEG complexity        12   Hjorth (act/mob/cmp) + spectral entropy      *** NOVEL ***
#                                   for N2, N3, REM (C3-M2)
#   [J]  sleep fragmentation    5   stage entropy, wake bouts, NREM frag,        *** NOVEL ***
#                                   stage stability
#   [K]  spectral edge freq     3   SEF90 for N2, N3, REM (C3-M2)               *** NOVEL ***
#   [L]  waveform kurtosis      6   kurt(raw EEG) N2,N3 × C3-M2,C4-M1,F3-M2    *** NOVEL ***
#   [M]  band power kurtosis   36   kurt(band power across epochs)               *** NOVEL ***
#                                   3ch × 3stages(N1,N2,N3) × 4bands
#   [N]  N3 spectral ratios     3   delta/alpha, delta/theta, sigma/delta (N3)   *** NOVEL ***
#   [O]  Custom slow waves      15  5 feats × 3ch — SO envelope det. N2+N3       *** v22 ***
#                                   density, neg_amp, pos_amp, ptp, slope
#                                   Mander et al. Nature Neurosci 2015
#   [P]  EEG coherence         75   3pairs × 5bands × 5stages (MSC)             *** NOVEL ***
#                                   pairs: C3-C4, C3-F3, C4-F3
#                                   bands: delta,theta,alpha,sigma,beta
#                                   Sun et al. SLEEP 2023 — AUROC 0.78
#   [Q]  REM spectral ratios    9   theta/alpha, theta/beta, sigma/alpha × 3ch   *** NOVEL ***
#                                   in REM sleep — thalamo-cortical biomarker
#   [R]  REM slow-fast ratio    3   log(slow/fast) in REM per EEG channel        *** NOVEL ***
#                                   SFAR: slow=(delta+theta), fast=(alpha+sigma+beta)
#   [S]  Philosopher's Stone    4   brain_health_score + total/fluid/crystallized *** v22 ***
#                                   cognition scores only (no latents) — NEJM AI 2026
#   [T]  half-night spectral   12   delta/theta/alpha/sigma power ratio 1st/2nd half
#
# Model: 5-model Stacking Ensemble (v22)
#   Level-0: LightGBM, XGBoost, ExtraTrees, RandomForest, LogisticRegression
#   Level-1: LogisticRegression (passthrough=True, 5-fold OOF)
#   Feature selection: Optuna-tuned threshold by LightGBM importance

import joblib
import numpy as np
import os
from scipy import signal as sp_signal
from scipy import stats as sp_stats
from tqdm import tqdm

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from helper_code import (
    DEMOGRAPHICS_FILE, PHYSIOLOGICAL_DATA_SUBFOLDER,
    ALGORITHMIC_ANNOTATIONS_SUBFOLDER, HEADERS,
    find_patients, load_signal_data, load_demographics,
    load_age, load_sex, load_bmi, load_diagnoses,
    get_standardized_race,
    load_rename_rules, standardize_channel_names_rename_only,
    derive_bipolar_signal,
)

# ─── Constants ────────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV     = os.path.join(SCRIPT_DIR, 'channel_table.csv')
CACHE_SUBDIR    = 'feature_cache'
MODEL_FILE      = 'model.sav'
FEATURE_VERSION = 'v22'  # bump when feature layout changes to invalidate old cache
PS_DIR          = '/ps'  # Philosopher's Stone repo path in Docker
PS_CACHE_FILE   = 'ps_cache.csv'  # pre-computed PS scores saved in model_folder
PS_BAKED_CACHE  = os.path.join(SCRIPT_DIR, 'ps_cache_baked.csv')  # baked into Docker image
PS_N_PCA        = 20    # PCA components from 1024-D latent → explains ~90% variance
PS_FEAT_DIM     = 4 + PS_N_PCA   # total PS features (4 scalars + 20 PCA components)

# ── Philosopher's Stone batch cache (populated during train_model) ──────────
# dict: '{pid}_ses-{ses}' → np.array(1028,) = [4 scores | 1024 latents]
# None = not yet loaded (inference path); dict = pre-computed (training path)
_PS_CACHE = None  # type: ignore[assignment]
_PS_PCA   = None  # sklearn PCA fitted on training latents; saved in model.sav

# stage_caisr: fs=1/30 Hz → one sample per 30-second PSG epoch
# Encoding: 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake, 9=Unavailable
S_N3, S_N2, S_N1, S_REM, S_WAKE, S_UNK = 1, 2, 3, 4, 5, 9
VALID_STAGES = [S_N3, S_N2, S_N1, S_REM, S_WAKE]
EPOCH_SEC    = 30

# EEG frequency bands
BANDS = [
    ('delta', 0.5,  4.0),
    ('theta', 4.0,  8.0),
    ('alpha', 8.0, 12.0),
    ('sigma',12.0, 16.0),
    ('beta', 16.0, 30.0),
    ('gamma',30.0, 50.0),
]
N_BANDS  = len(BANDS)   # 6
N_STAGES = len(VALID_STAGES)  # 5

# Fixed EEG channel slots — same slot = same feature position across all sites
EEG_SLOT = {'c3-m2': 0, 'c4-m1': 1, 'f3-m2': 2}
N_EEG_CH = len(EEG_SLOT)  # 3

# Bipolar derivation rules: (target_name, positive_ch, [negative_ch(s)])
BIPOLAR_RULES = [
    ('c3-m2', 'c3', ['m2']),
    ('c4-m1', 'c4', ['m1']),
    ('f3-m2', 'f3', ['m2']),
    ('f4-m1', 'f4', ['m1']),
    ('o1-m2', 'o1', ['m2']),
    ('o2-m1', 'o2', ['m1']),
    ('e1-m2', 'e1', ['m2']),
    ('e2-m2', 'e2', ['m2']),
]

N_KURTOSIS_BANDS  = 4   # delta, theta, alpha, sigma
N_KURTOSIS_STAGES = 3   # N1, N2, N3
MAX_SEQ_LEN       = 1200  # max epochs per recording (~10 h at 1/30 Hz)
SEQ_CHANNELS      = 5     # caisr_prob_w/n1/n2/n3/r
N_FEATURES = (10 + 12 + N_EEG_CH*N_STAGES*N_BANDS + 8 + 6 + 5 + 8 + 3 + 12 + 5 + 3  # 162
              + 6 + N_EEG_CH*N_KURTOSIS_STAGES*N_KURTOSIS_BANDS + 3)                  # +45 = 207


# ─── Required Challenge Functions ─────────────────────────────────────────────

def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV):
    demo_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    records   = find_patients(demo_file)

    if verbose:
        print(f'[train] {len(records)} records | feature_version={FEATURE_VERSION}')

    os.makedirs(model_folder, exist_ok=True)
    cache_dir = os.path.join(model_folder, CACHE_SUBDIR)
    os.makedirs(cache_dir, exist_ok=True)

    # ── Pre-compute PS features once for all records ──────────────────────────
    _precompute_ps_batch(data_folder, model_folder, records, verbose=verbose)

    features, labels, sites = [], [], []
    pbar = tqdm(records, desc='Extracting', disable=not verbose)
    for rec in pbar:
        pid = rec[HEADERS['bids_folder']]
        try:
            X     = extract_all_features(rec, data_folder, csv_path, cache_dir)
            label = load_diagnoses(demo_file, pid)
            features.append(X)
            labels.append(label)
            sites.append(rec.get(HEADERS['site_id'], 'S0001'))
        except Exception as e:
            tqdm.write(f'  ! {pid}: {e}')

    X_arr    = np.array(features, dtype=np.float32)
    X_arr    = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
    y_arr    = np.array(labels,   dtype=int)
    site_arr = np.array(sites)

    if verbose:
        print(f'[train] X={X_arr.shape} | pos={y_arr.sum()} neg={(y_arr==0).sum()}')

    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.base import clone

    # ── Optuna: tune LightGBM + feature-selection threshold ──────────────────
    _best_lgbm_params = {}
    _best_fs_pct      = 40
    if HAS_LGBM:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def _lgbm_objective(trial):
                p = dict(
                    n_estimators      = 800,
                    learning_rate     = 0.02,
                    num_leaves        = trial.suggest_int('num_leaves', 10, 63),
                    min_child_samples = trial.suggest_int('min_child_samples', 10, 80),
                    reg_lambda        = trial.suggest_float('reg_lambda', 0.1, 20.0, log=True),
                    reg_alpha         = trial.suggest_float('reg_alpha', 0.01, 2.0, log=True),
                    colsample_bytree  = trial.suggest_float('colsample_bytree', 0.4, 0.9),
                    subsample         = trial.suggest_float('subsample', 0.5, 0.9),
                    subsample_freq    = 1,
                    random_state      = 42, n_jobs=1, verbose=-1,
                )
                fs_pct = trial.suggest_int('fs_pct', 25, 55)
                clf_fs = lgb.LGBMClassifier(**p)
                clf_fs.fit(X_arr, y_arr)
                imp = clf_fs.feature_importances_
                mask = imp >= np.percentile(imp, fs_pct)
                Xs = X_arr[:, mask]
                clf = lgb.LGBMClassifier(**p)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                return cross_val_score(clf, Xs, y_arr, cv=skf,
                                       scoring='roc_auc', n_jobs=-1).mean()

            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42),
            )
            study.optimize(_lgbm_objective, n_trials=80,
                           show_progress_bar=verbose)
            _best_lgbm_params = {k: v for k, v in study.best_params.items()
                                  if k != 'fs_pct'}
            _best_fs_pct = study.best_params.get('fs_pct', 40)
            _best_lgbm_params.update(dict(n_estimators=800, learning_rate=0.02,
                                          subsample_freq=1, random_state=42,
                                          n_jobs=-1, verbose=-1))
            if verbose:
                print(f'[Optuna] Best CV AUROC={study.best_value:.4f} | '
                      f'fs_pct={_best_fs_pct} | params={study.best_params}')
        except Exception as e:
            if verbose:
                print(f'[Optuna] Skipped ({e})')

    # ── Base model factories ──────────────────────────────────────────────────
    _lgbm_defaults = dict(
        n_estimators=800, num_leaves=31, learning_rate=0.02,
        min_child_samples=30, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_alpha=0.2, reg_lambda=3.0,
        random_state=42, n_jobs=-1, verbose=-1,
    )

    def _make_lgbm():
        if HAS_LGBM:
            params = _best_lgbm_params if _best_lgbm_params else _lgbm_defaults
            return lgb.LGBMClassifier(**params)
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=500, max_depth=4,
            learning_rate=0.02, subsample=0.8, random_state=42,
        )

    def _make_xgb():
        if HAS_XGB:
            return xgb.XGBClassifier(
                n_estimators=800,
                max_depth=4,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.2,
                reg_lambda=3.0,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
        return None

    def _make_rf():
        return RandomForestClassifier(
            n_estimators=500,
            max_features='sqrt',
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

    def _make_et():
        return ExtraTreesClassifier(
            n_estimators=500,
            max_features='sqrt',
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

    def _make_lr():
        # Linear model with scaling — adds diversity to tree-based ensemble
        return Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(C=0.05, max_iter=3000, solver='saga',
                                      random_state=42)),
        ])

    # ── Feature selection: drop bottom 40% by LGBM importance ────────────────
    if verbose:
        print('[FS] Fitting LGBM for feature importance ...')
    clf_fs = _make_lgbm()
    clf_fs.fit(X_arr, y_arr)
    importances = clf_fs.feature_importances_
    threshold   = np.percentile(importances, _best_fs_pct)
    feat_mask   = importances >= threshold
    X_sel       = X_arr[:, feat_mask]
    if verbose:
        print(f'[FS] Selected {feat_mask.sum()}/{len(feat_mask)} features (threshold={threshold:.4f})')

    # ── Build stacking ensemble (passthrough=True) ────────────────────────────
    # passthrough=True: meta-learner sees base OOF predictions + original features
    # This lets LogReg directly use strong single features as well as ensemble output
    estimators = [
        ('lgbm', _make_lgbm()),
        ('rf',   _make_rf()),
        ('et',   _make_et()),
        ('lr',   _make_lr()),
    ]
    xgb_clf = _make_xgb()
    if xgb_clf is not None:
        estimators.append(('xgb', xgb_clf))

    final_estimator = LogisticRegression(C=0.1, max_iter=2000, random_state=42)

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1,
        passthrough=True,
    )

    # ── 5-fold stratified CV to report honest AUROC ───────────────────────────
    if verbose:
        print(f'[CV] 5-fold CV | {len(estimators)} base models ...')
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Site-stratified: combine site+diagnosis so folds are balanced per site
    strat_labels = np.array([f"{s}_{l}" for s, l in zip(site_arr, y_arr)])
    aucs = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_sel, strat_labels), 1):
        fold_stack = StackingClassifier(
            estimators=[(n, clone(m)) for n, m in estimators],
            final_estimator=clone(final_estimator),
            cv=5, stack_method='predict_proba',
            passthrough=True, n_jobs=-1,
        )
        fold_stack.fit(X_sel[tr_idx], y_arr[tr_idx])
        prob = fold_stack.predict_proba(X_sel[val_idx])[:, 1]
        auc  = roc_auc_score(y_arr[val_idx], prob)
        aucs.append(auc)
        if verbose:
            print(f'  Fold {fold}: AUROC={auc:.4f}')
    if verbose:
        print(f'[CV] Mean AUROC = {np.mean(aucs):.4f}  std = {np.std(aucs):.4f}')

    # ── Train final stacking model on full data ───────────────────────────────
    if verbose:
        print('[train] Fitting final stacking model on full data ...')
    stack.fit(X_sel, y_arr)

    joblib.dump({
        'stack':           stack,
        'feat_mask':       feat_mask,
        'feature_version': FEATURE_VERSION,
        'ps_pca':          _PS_PCA,
    }, os.path.join(model_folder, MODEL_FILE))

    if verbose:
        print('[train] Done.')


def load_model(model_folder, verbose):
    return joblib.load(os.path.join(model_folder, MODEL_FILE))


def run_model(model, record, data_folder, verbose):
    global _PS_PCA
    _PS_PCA   = model.get('ps_pca')   # restore PCA transform for inference
    stack     = model['stack']
    feat_mask = model.get('feat_mask')

    X = extract_all_features(record, data_folder, DEFAULT_CSV, cache_dir=None)
    X = X.reshape(1, -1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if feat_mask is not None:
        X = X[:, feat_mask]

    prob   = float(stack.predict_proba(X)[0][1])
    binary = prob >= 0.5
    return binary, prob


# ─── Feature Extraction ───────────────────────────────────────────────────────


def feat_half_night_spectral(phys, fs_dict, stages):
    """[T] 12 features: delta/theta/alpha/sigma power ratio (2nd half / 1st half)
    for c4-m1, plus absolute powers in each half.  Captures how EEG evolves
    through the night — slow-wave sleep pressure changes differ in CI."""
    out = np.zeros(12, dtype=np.float32)
    ch  = 'c4-m1'
    if ch not in phys:
        return out
    sig  = phys[ch].astype(np.float32)
    eeg_fs = float(fs_dict.get(ch, 200.0))
    n    = len(sig)
    if n < int(eeg_fs * 3600):   # need at least 1 hour
        return out
    mid  = n // 2
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'sigma': (11, 16)}
    try:
        from scipy.signal import welch
        for bi, (bname, (lo, hi)) in enumerate(bands.items()):
            def _bp(x):
                nperseg = min(int(eeg_fs * 4), len(x) // 4)
                if nperseg < 16:
                    return 0.0
                f, pxx = welch(x, fs=eeg_fs, nperseg=nperseg)
                m = (f >= lo) & (f < hi)
                return float(np.log1p(pxx[m].mean())) if m.any() else 0.0
            p1 = _bp(sig[:mid])
            p2 = _bp(sig[mid:])
            out[bi * 3]     = p1
            out[bi * 3 + 1] = p2
            out[bi * 3 + 2] = p2 - p1   # change: positive = power increases in 2nd half
    except Exception:
        pass
    return out


def extract_all_features(record, data_folder, csv_path=DEFAULT_CSV, cache_dir=None):
    """Returns float32 array of shape (N_FEATURES,) = (162,)."""
    pid = record[HEADERS['bids_folder']]
    sid = record[HEADERS['site_id']]
    ses = record[HEADERS['session_id']]

    if cache_dir:
        cache_file = os.path.join(cache_dir,
                                  f'{pid}_ses-{ses}_{FEATURE_VERSION}.npz')
        if os.path.exists(cache_file):
            return np.load(cache_file)['features']

    demo_file    = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_data = load_demographics(demo_file, pid, ses)

    phys_path = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                             sid, f'{pid}_ses-{ses}.edf')
    phys_raw, phys_fs_raw = (load_signal_data(phys_path)
                              if os.path.exists(phys_path) else ({}, {}))

    algo_path = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                             sid, f'{pid}_ses-{ses}_caisr_annotations.edf')
    algo_data, _ = (load_signal_data(algo_path)
                    if os.path.exists(algo_path) else ({}, {}))

    phys, fs = _build_channel_dict(phys_raw, phys_fs_raw, csv_path)

    # stage_caisr: fs = 1/30 Hz → one sample per 30-s epoch
    stages = np.asarray(algo_data.get('stage_caisr', []), dtype=float)

    # ── Feature groups ──────────────────────────────────────────────────────
    f_demo   = feat_demographics(patient_data)          # [A]  13
    f_macro  = feat_sleep_macro(stages)                 # [B]  12
    f_eeg    = feat_eeg_bandpower(phys, fs, stages)     # [C]  90
    f_resp   = feat_resp_spo2(algo_data, phys)          # [D]   8
    f_hrv    = feat_ecg_hrv(phys, fs)                  # [E]   9  (v22: +LF/HF)
    f_prob   = feat_caisr_probs(algo_data)              # [F]   5
    f_spin   = feat_custom_spindles(phys, fs, stages)   # [G]  18  NOVEL
    f_sw     = feat_custom_sw(phys, fs, stages)         # [O]  15  v22 NEW
    f_coup   = feat_so_spindle_coupling(phys, fs, stages)     # [H]   3  NOVEL
    f_cplx   = feat_eeg_complexity(phys, fs, stages)          # [I]  12  NOVEL
    f_frag   = feat_sleep_fragmentation(stages)                # [J]   5  NOVEL
    f_sef    = feat_spectral_edge(phys, fs, stages)            # [K]   3  NOVEL
    f_wkurt  = feat_waveform_kurtosis(phys, fs, stages)        # [L]   6  NOVEL
    f_bkurt  = feat_bandpower_kurtosis(phys, fs, stages)       # [M]  36  NOVEL
    f_n3rat  = feat_n3_ratios(phys, fs, stages)                # [N]   3  NOVEL
    f_coh    = feat_eeg_coherence(phys, fs, stages)            # [P]  75  NOVEL
    f_remrat = feat_rem_spectral_ratios(phys, fs, stages)      # [Q]   9  NOVEL
    f_sfar   = feat_rem_sfar(phys, fs, stages)                 # [R]   3  NOVEL
    f_ps     = feat_ps_scalars(patient_data)            # [S]   4  v22 (scalars only)
    f_halves = feat_half_night_spectral(phys, fs, stages)      # [T]  12  NOVEL

    features = np.concatenate([
        f_demo, f_macro, f_eeg, f_resp, f_hrv, f_prob,
        f_spin, f_sw, f_coup, f_cplx, f_frag, f_sef,
        f_wkurt, f_bkurt, f_n3rat,
        f_coh, f_remrat, f_sfar, f_ps, f_halves,
    ]).astype(np.float32)

    if cache_dir:
        np.savez_compressed(cache_file, features=features)

    return features


def _build_channel_dict(raw, raw_fs, csv_path):
    if not raw:
        return {}, {}

    if os.path.exists(csv_path):
        rules = load_rename_rules(csv_path)
        rename_map, drop = standardize_channel_names_rename_only(list(raw.keys()), rules)
    else:
        rename_map, drop = {}, []

    std, fs = {}, {}
    for old, sig in raw.items():
        if old in drop:
            continue
        new = rename_map.get(old, old.lower())
        std[new] = sig
        fs[new]  = raw_fs.get(old, 200.0)

    for target, pos, negs in BIPOLAR_RULES:
        if target in std:
            continue
        if pos not in std or not all(n in std for n in negs):
            continue
        if len({fs[c] for c in [pos] + negs}) > 1:
            continue
        ref    = std[negs[0]] if len(negs) == 1 else tuple(std[n] for n in negs)
        signal = derive_bipolar_signal(std[pos], ref)
        if signal is not None:
            std[target] = signal
            fs[target]  = fs[pos]

    return std, fs


def _get_n2_epochs(sig, eeg_fs, stages):
    """Collect all N2 epoch arrays for a given EEG signal."""
    epoch_samp = int(EPOCH_SEC * eeg_fs)
    epochs = []
    for i, sv in enumerate(stages.astype(int)):
        if sv != S_N2:
            continue
        start = i * epoch_samp
        end   = start + epoch_samp
        if end > len(sig):
            break
        epochs.append(sig[start:end])
    return epochs


def _bandpass(sig, fs, lo, hi):
    """Bandpass filter a signal."""
    nyq = fs / 2.0
    lo_ = max(lo / nyq, 1e-4)
    hi_ = min(hi / nyq, 0.999)
    if lo_ >= hi_:
        return sig
    b, a = sp_signal.butter(4, [lo_, hi_], btype='band')
    return sp_signal.filtfilt(b, a, sig)


# ─── [A] Demographics ─────────────────────────────────────────────────────────

def feat_demographics(data):
    """13 features: age(1) + sex×3 + race×5 + BMI(1) + site×3."""
    age = np.array([load_age(data)], dtype=float)

    sex_vec = np.zeros(3)
    sex_vec[{'Female': 0, 'Male': 1}.get(load_sex(data), 2)] = 1.0

    race_vec = np.zeros(5)
    race_key = get_standardized_race(data).lower()
    race_vec[{'asian':0,'black':1,'others':2,'unavailable':3,'white':4}.get(race_key, 2)] = 1.0

    bmi = np.array([load_bmi(data)], dtype=float)

    site_vec = np.zeros(3, dtype=np.float32)
    site_id  = str(data.get('SiteID', ''))
    site_vec[{'S0001': 0, 'I0006': 1, 'I0002': 2}.get(site_id, 0)] = 1.0

    return np.concatenate([age, sex_vec, race_vec, bmi, site_vec])


# ─── [B] Sleep macrostructure ─────────────────────────────────────────────────

def feat_sleep_macro(stages):
    """12 features from stage_caisr (one value per 30-s epoch)."""
    out = np.zeros(12)
    if len(stages) == 0:
        return out

    valid = stages < S_UNK
    sleep = valid & (stages != S_WAKE)
    s_idx = np.where(sleep)[0]
    if len(s_idx) == 0:
        return out

    onset    = s_idx[0]
    sl_min   = onset * EPOCH_SEC / 60.0
    post     = stages[onset:]

    tst_ep   = int(np.sum((post != S_WAKE) & (post < S_UNK)))
    tib_ep   = int(np.sum(valid))
    waso_ep  = int(np.sum(post == S_WAKE))
    tst_min  = tst_ep  * EPOCH_SEC / 60.0
    tib_min  = tib_ep  * EPOCH_SEC / 60.0
    waso_min = waso_ep * EPOCH_SEC / 60.0
    tst_hr   = tst_min / 60.0
    se       = tst_min / tib_min if tib_min > 0 else 0.0

    n3_pct  = np.sum(post == S_N3)  / tst_ep if tst_ep > 0 else 0.0
    n2_pct  = np.sum(post == S_N2)  / tst_ep if tst_ep > 0 else 0.0
    n1_pct  = np.sum(post == S_N1)  / tst_ep if tst_ep > 0 else 0.0
    rem_pct = np.sum(post == S_REM) / tst_ep if tst_ep > 0 else 0.0

    rem_ep  = np.where(post == S_REM)[0]
    rem_lat = rem_ep[0] * EPOCH_SEC / 60.0 if len(rem_ep) > 0 else tib_min

    valid_p = post[post < S_UNK]
    trans   = (np.count_nonzero(np.diff(valid_p)) / tst_hr
               if tst_hr > 0 and len(valid_p) > 1 else 0.0)

    return np.array([sl_min, tst_min, se, waso_min,
                     n3_pct, n2_pct, n1_pct, rem_pct,
                     rem_lat, trans, tst_hr, tib_min/60.0])


# ─── [C] EEG bandpower ────────────────────────────────────────────────────────

def feat_eeg_bandpower(phys, fs_dict, stages):
    """90 features: 3ch × 5stages × 6bands."""
    result   = np.zeros(N_EEG_CH * N_STAGES * N_BANDS)
    n_epochs = len(stages)
    if n_epochs == 0:
        return result
    stages_int = stages.astype(int)
    for ch, slot in EEG_SLOT.items():
        if ch not in phys:
            continue
        bp     = _bandpower_by_stage(phys[ch], fs_dict.get(ch, 200.0),
                                     stages_int, n_epochs)
        offset = slot * N_STAGES * N_BANDS
        result[offset: offset + N_STAGES * N_BANDS] = bp.flatten()
    return result


def _bandpower_by_stage(sig, eeg_fs, stages_int, n_epochs):
    """Returns (N_STAGES, N_BANDS) mean relative Welch PSD per stage.
    Each band power is normalized by total 0.5-50 Hz power to remove
    site-level amplitude differences (electrode impedance, amplifier gain).
    """
    out        = np.zeros((N_STAGES, N_BANDS))
    epoch_samp = int(EPOCH_SEC * eeg_fs)
    if epoch_samp == 0 or len(sig) < epoch_samp:
        return out
    nperseg = min(int(eeg_fs * 4), epoch_samp)
    accum   = [[] for _ in range(N_STAGES)]
    for i in range(n_epochs):
        sv = stages_int[i]
        if sv not in VALID_STAGES:
            continue
        start = i * epoch_samp
        end   = start + epoch_samp
        if end > len(sig):
            break
        f, psd = sp_signal.welch(sig[start:end], fs=eeg_fs, nperseg=nperseg)
        bp = np.array([
            np.trapz(psd[m], f[m]) if (m := (f >= flo) & (f <= fhi)).any() else 0.0
            for _, flo, fhi in BANDS
        ])
        bp = np.log1p(bp)  # log(1+x): stabilises scale differences across sites
        accum[VALID_STAGES.index(sv)].append(bp)
    for s_idx in range(N_STAGES):
        if accum[s_idx]:
            out[s_idx] = np.mean(accum[s_idx], axis=0)
    return out


# ─── [D] Resp / SpO2 ──────────────────────────────────────────────────────────

def feat_resp_spo2(algo_data, phys):
    """8 features: AHI, arousal_idx, PLMI, mean_SpO2, min_SpO2,
                   ODI3/hr, ODI4/hr, frac_below_90."""
    out = np.zeros(8)

    resp   = np.asarray(algo_data.get('resp_caisr', []), dtype=float)
    r_hr   = len(resp) / 3600.0
    if r_hr > 0:
        out[0] = np.count_nonzero(np.diff((resp > 0).astype(int), prepend=0) == 1) / r_hr

    arous  = np.asarray(algo_data.get('arousal_caisr', []), dtype=float)
    a_hr   = len(arous) / 2.0 / 3600.0
    if a_hr > 0:
        out[1] = np.count_nonzero(np.diff((arous > 0).astype(int), prepend=0) == 1) / a_hr

    limb   = np.asarray(algo_data.get('limb_caisr', []), dtype=float)
    l_hr   = len(limb) / 3600.0
    if l_hr > 0:
        out[2] = np.count_nonzero(np.diff((limb > 0).astype(int), prepend=0) == 1) / l_hr

    spo2 = None
    for ch in ('sao2', 'spo2'):
        if ch in phys:
            spo2 = phys[ch].astype(float)
            break
    if spo2 is not None and len(spo2) > 0:
        valid = spo2[(spo2 >= 50) & (spo2 <= 100)]
        if len(valid) > 0:
            m      = np.mean(valid)
            sph    = len(spo2) / 3600.0
            out[3] = m
            out[4] = np.min(valid)
            out[5] = (np.count_nonzero(np.diff((valid < m-3).astype(int), prepend=0)==1)
                      / sph if sph > 0 else 0.0)
            out[6] = (np.count_nonzero(np.diff((valid < m-4).astype(int), prepend=0)==1)
                      / sph if sph > 0 else 0.0)
            out[7] = np.mean(valid < 90)
    return out


# ─── [E] ECG HRV ──────────────────────────────────────────────────────────────

def feat_ecg_hrv(phys, fs_dict):
    """9 HRV features: 6 time-domain + 3 frequency-domain (LF, HF, LF/HF).

    Time-domain: mean_RR, SDNN, RMSSD, HR, pNN50, NN_range
    Freq-domain: log_LF (0.04-0.15 Hz), log_HF (0.15-0.4 Hz), LF/HF ratio
    Frequency domain computed on uniformly resampled RR series at 4 Hz.
    """
    ecg, ekg_fs = None, 200.0
    for ch in ('ekg', 'ecg'):
        if ch in phys:
            ecg    = phys[ch].astype(float)
            ekg_fs = fs_dict.get(ch, 200.0)
            break
    if ecg is None or len(ecg) < ekg_fs * 60:
        return np.zeros(9)
    try:
        nyq  = ekg_fs / 2.0
        b, a = sp_signal.butter(2, [max(5./nyq, 1e-3), min(40./nyq, 0.99)], btype='band')
        ecg_f   = sp_signal.filtfilt(b, a, ecg)
        peaks,_ = sp_signal.find_peaks(ecg_f,
                                        height=np.percentile(ecg_f, 90),
                                        distance=int(0.35 * ekg_fs))
        if len(peaks) < 10:
            return np.zeros(9)
        rr = np.diff(peaks) / ekg_fs * 1000.0
        rr = rr[(rr > 300) & (rr < 2000)]
        if len(rr) < 5:
            return np.zeros(9)
        mean_rr = np.mean(rr)
        time_feats = np.array([mean_rr, np.std(rr),
                                np.sqrt(np.mean(np.diff(rr)**2)),
                                60000. / mean_rr,
                                np.mean(np.abs(np.diff(rr)) > 50),
                                np.max(rr) - np.min(rr)])

        # Frequency-domain HRV: resample RR series at 4 Hz, then Welch PSD
        freq_feats = np.zeros(3)
        try:
            rr_sec    = rr / 1000.0                          # ms → s
            rr_times  = np.cumsum(rr_sec)                    # cumulative time axis
            rr_times  = rr_times - rr_times[0]               # start at 0
            interp_fs = 4.0
            t_uniform = np.arange(0, rr_times[-1], 1.0 / interp_fs)
            if len(t_uniform) > 32:
                rr_interp = np.interp(t_uniform, rr_times, rr_sec)
                nperseg   = min(256, len(rr_interp) // 4)
                if nperseg >= 8:
                    f_rr, pxx = sp_signal.welch(rr_interp, fs=interp_fs, nperseg=nperseg)
                    lf_m = (f_rr >= 0.04) & (f_rr < 0.15)
                    hf_m = (f_rr >= 0.15) & (f_rr < 0.40)
                    lf   = float(np.trapz(pxx[lf_m], f_rr[lf_m])) if lf_m.any() else 0.0
                    hf   = float(np.trapz(pxx[hf_m], f_rr[hf_m])) if hf_m.any() else 0.0
                    freq_feats = np.array([np.log1p(lf), np.log1p(hf),
                                           lf / hf if hf > 0 else 0.0])
        except Exception:
            pass

        return np.concatenate([time_feats, freq_feats])
    except Exception:
        return np.zeros(9)


# ─── [F] CAISR probabilities ──────────────────────────────────────────────────

def feat_caisr_probs(algo_data):
    """5 features: mean valid (non-9) stage probs [w, n3, n2, n1, r]."""
    out  = np.zeros(5)
    for i, k in enumerate(['caisr_prob_w', 'caisr_prob_n3', 'caisr_prob_n2',
                            'caisr_prob_n1', 'caisr_prob_r']):
        arr   = np.asarray(algo_data.get(k, []), dtype=float)
        valid = arr[arr < 9.0]
        if len(valid) > 0:
            out[i] = np.clip(np.mean(valid), 0., 1.)
    return out


# ─── [G] YASA Sleep spindle features ★ NOVEL ─────────────────────────────────
#
# Motivation: YASA (Vallat & Walker, eLife 2021) implements a validated spindle
# detection algorithm (Lacourse et al. 2019) with per-event metrics.
# Spindle degradation is an early biomarker of thalamo-cortical dysfunction in MCI/AD.
# References: Helfrich et al. Nature Comm 2018; Winer et al. Nature Comm 2019.

def _stages_to_yasa_hypno(stages):
    """Convert CAISR stage encoding to YASA hypnogram at epoch resolution.
    CAISR: N3=1, N2=2, N1=3, REM=4, Wake=5, Unk=9
    YASA:  W=0,  N1=1, N2=2, N3=3, R=4,   Art=-1
    """
    hypno = np.full(len(stages), -1, dtype=int)
    hypno[stages == S_WAKE] = 0
    hypno[stages == S_N1]   = 1
    hypno[stages == S_N2]   = 2
    hypno[stages == S_N3]   = 3
    hypno[stages == S_REM]  = 4
    return hypno


def feat_yasa_spindles(phys, fs_dict, stages):
    """[G] 24 YASA spindle features (8 metrics × 3 EEG channels).

    Per channel (C3-M2, C4-M1, F3-M2):
      [0] duration_mean     — mean spindle duration (s)
      [1] amplitude_mean    — mean peak-to-peak amplitude (µV)
      [2] rms_mean          — mean RMS amplitude
      [3] relpower_mean     — mean relative sigma power during spindle
      [4] frequency_mean    — mean spindle frequency (Hz)
      [5] oscillations_mean — mean number of oscillations per spindle
      [6] density           — spindles per minute of N2 sleep
      [7] frequency_std     — frequency variability (Hz)
    """
    out = np.zeros(24)
    if len(stages) == 0:
        return out

    try:
        import yasa
    except ImportError:
        return out

    hypno_ep = _stages_to_yasa_hypno(stages)
    n2_ep    = np.sum(hypno_ep == 2)
    if n2_ep == 0:
        return out

    channels = ['c3-m2', 'c4-m1', 'f3-m2']
    for ch_idx, ch in enumerate(channels):
        if ch not in phys:
            continue
        sig = phys[ch]
        sf  = fs_dict.get(ch, 200.0)

        epoch_samp = int(EPOCH_SEC * sf)
        hypno_samp = np.repeat(hypno_ep, epoch_samp)
        sig_trim   = sig[:len(hypno_samp)]
        hypno_samp = hypno_samp[:len(sig_trim)]

        try:
            sp = yasa.spindles_detect(
                sig_trim, sf=sf, hypno=hypno_samp,
                include=(2,), freq_sp=(11, 16), min_distance=500,
            )
            if sp is not None:
                df = sp.summary()
                if len(df) > 0:
                    b = ch_idx * 8
                    out[b+0] = df['Duration'].mean()
                    out[b+1] = df['Amplitude'].mean()
                    out[b+2] = df['RMS'].mean()
                    out[b+3] = df['RelPower'].mean()
                    out[b+4] = df['Frequency'].mean()
                    out[b+5] = df['Oscillations'].mean()
                    out[b+6] = len(df) / (n2_ep * EPOCH_SEC / 60.0)
                    out[b+7] = df['Frequency'].std() if len(df) > 1 else 0.0
        except Exception:
            pass

    return out


# ─── [O] YASA Slow Wave features ★ NOVEL ─────────────────────────────────────
#
# Slow oscillations (0.5-2 Hz) in N2/N3 are the primary vehicle for
# hippocampo-neocortical memory replay.  Slope and amplitude of slow waves
# are degraded early in MCI due to cortical thinning.
# Reference: Mander et al. Nature Neurosci 2015; Lucey et al. Nature Comm 2021.

def feat_yasa_sw(phys, fs_dict, stages):
    """[O] 15 YASA slow wave features (5 metrics × 3 EEG channels) in N2+N3.

    Per channel (C3-M2, C4-M1, F3-M2):
      [0] duration_mean   — mean slow wave duration (s)
      [1] neg_amp_mean    — mean negative peak amplitude (µV)
      [2] pos_amp_mean    — mean positive peak amplitude (µV)
      [3] ptp_mean        — mean peak-to-peak amplitude (µV)
      [4] slope_mean      — mean negative half-wave slope (µV/s)
    """
    out = np.zeros(15)
    if len(stages) == 0:
        return out

    try:
        import yasa
    except ImportError:
        return out

    hypno_ep  = _stages_to_yasa_hypno(stages)
    n23_ep    = np.sum((hypno_ep == 2) | (hypno_ep == 3))
    if n23_ep == 0:
        return out

    channels = ['c3-m2', 'c4-m1', 'f3-m2']
    for ch_idx, ch in enumerate(channels):
        if ch not in phys:
            continue
        sig = phys[ch]
        sf  = fs_dict.get(ch, 200.0)

        epoch_samp = int(EPOCH_SEC * sf)
        hypno_samp = np.repeat(hypno_ep, epoch_samp)
        sig_trim   = sig[:len(hypno_samp)]
        hypno_samp = hypno_samp[:len(sig_trim)]

        try:
            sw = yasa.sw_detect(
                sig_trim, sf=sf, hypno=hypno_samp,
                include=(2, 3),
            )
            if sw is not None:
                df = sw.summary()
                if len(df) > 0:
                    b = ch_idx * 5
                    out[b+0] = df['Duration'].mean()
                    out[b+1] = abs(df['ValNegPeak'].mean())
                    out[b+2] = df['ValPosPeak'].mean()
                    out[b+3] = df['PTP'].mean()
                    out[b+4] = abs(df['Slope'].mean())
        except Exception:
            pass

    return out


# ─── [G] Custom spindle detection ★ NOVEL ────────────────────────────────────
#
# Replaces YASA spindle/slow-wave features (v9).
# YASA was producing all-zero features due to CHAN000 channel labeling bug,
# which added noise and hurt AUROC.
# This implementation uses sigma-band (11-16 Hz) envelope thresholding via scipy.
# Faster (~0.5s vs 25s per patient), no external dependency.
# Reference: Lacourse et al. 2019 (basis for YASA's algorithm).

def feat_custom_spindles(phys, fs_dict, stages):
    """[G] 18 custom spindle features (6 per channel × 3 EEG channels) in N2 sleep.

    Per channel (C3-M2, C4-M1, F3-M2):
      [0] sigma_power      — log mean sigma band power in N2
      [1] spindle_density  — events per minute of N2 (threshold: mean+1.5*std)
      [2] mean_amplitude   — mean peak envelope amplitude during spindle
      [3] mean_duration    — mean spindle duration (s), valid: 0.5–3.0 s
      [4] sigma_alpha_ratio — sigma / alpha power in N2
      [5] sigma_delta_ratio — sigma / delta power in N2
    """
    out = np.zeros(18)
    if len(stages) == 0:
        return out

    for ch_idx, ch in enumerate(['c3-m2', 'c4-m1', 'f3-m2']):
        if ch not in phys:
            continue
        sig        = phys[ch]
        eeg_fs     = fs_dict.get(ch, 200.0)
        epoch_samp = int(EPOCH_SEC * eeg_fs)
        nperseg    = min(int(eeg_fs * 4), epoch_samp)

        # Collect N2 epochs
        n2_bufs, n2_count = [], 0
        for i, sv in enumerate(stages.astype(int)):
            if sv != S_N2:
                continue
            start = i * epoch_samp
            end   = start + epoch_samp
            if end > len(sig):
                break
            n2_bufs.append(sig[start:end])
            n2_count += 1

        if not n2_bufs:
            continue

        n2_concat = np.concatenate(n2_bufs)
        n2_min    = n2_count * EPOCH_SEC / 60.0

        # PSD for power ratios
        f, psd = sp_signal.welch(n2_concat, fs=eeg_fs, nperseg=nperseg)
        def _bp(lo, hi):
            m = (f >= lo) & (f <= hi)
            return np.trapz(psd[m], f[m]) if m.any() else 0.0

        sigma_p = _bp(11.0, 16.0)
        alpha_p = _bp(8.0, 12.0)
        delta_p = _bp(0.5,  4.0)

        # Sigma envelope for spindle detection
        sigma_filt = _bandpass(n2_concat, eeg_fs, 11.0, 16.0)
        sigma_env  = np.abs(sp_signal.hilbert(sigma_filt))

        thresh = np.mean(sigma_env) + 1.5 * np.std(sigma_env)
        above  = (sigma_env > thresh).astype(int)
        starts = np.where(np.diff(above, prepend=0) == 1)[0]
        ends   = np.where(np.diff(above, append=0) == -1)[0]

        min_samp = int(0.5  * eeg_fs)
        max_samp = int(3.0  * eeg_fs)
        valid = [(s, e) for s, e in zip(starts, ends)
                 if min_samp <= (e - s) <= max_samp]

        b = ch_idx * 6
        out[b+0] = np.log1p(sigma_p)
        out[b+1] = len(valid) / n2_min if n2_min > 0 else 0.0
        if valid:
            out[b+2] = float(np.mean([np.max(sigma_env[s:e]) for s, e in valid]))
            out[b+3] = float(np.mean([(e - s) / eeg_fs for s, e in valid]))
        out[b+4] = sigma_p / alpha_p if alpha_p > 0 else 0.0
        out[b+5] = sigma_p / delta_p if delta_p > 0 else 0.0

    return out


# ─── [O] Custom Slow Wave detection ★ NOVEL v22 ──────────────────────────────
#
# Motivation: Slow oscillation (SO) amplitude and slope in N2/N3 are among the
# most replicated biomarkers of cognitive decline — cortical thinning reduces
# SO amplitude and sharpness even before clinical MCI.
# Method: 0.5-2 Hz bandpass → negative half-wave detection by peak-finding.
# References: Mander et al. Nature Neurosci 2015; Lucey et al. Nature Comm 2021.

def feat_custom_sw(phys, fs_dict, stages):
    """[O] 15 custom slow-wave features (5 per channel × 3 EEG channels) in N2+N3.

    Per channel (C3-M2, C4-M1, F3-M2):
      [0] sw_density    — slow waves per minute of N2+N3
      [1] neg_amp_mean  — mean absolute negative half-wave amplitude
      [2] pos_amp_mean  — mean positive half-wave amplitude
      [3] ptp_mean      — mean peak-to-peak amplitude
      [4] slope_mean    — mean negative half-wave slope (amp / half-period)
    """
    out = np.zeros(15)
    if len(stages) == 0:
        return out

    for ch_idx, ch in enumerate(['c3-m2', 'c4-m1', 'f3-m2']):
        if ch not in phys:
            continue
        sig        = phys[ch]
        eeg_fs     = fs_dict.get(ch, 200.0)
        epoch_samp = int(EPOCH_SEC * eeg_fs)

        # Collect N2+N3 epochs
        n23_bufs, n23_count = [], 0
        for i, sv in enumerate(stages.astype(int)):
            if sv not in (S_N2, S_N3):
                continue
            start = i * epoch_samp
            end   = start + epoch_samp
            if end > len(sig):
                break
            n23_bufs.append(sig[start:end])
            n23_count += 1

        if not n23_bufs or n23_count < 2:
            continue

        n23_concat = np.concatenate(n23_bufs)
        n23_min    = n23_count * EPOCH_SEC / 60.0

        try:
            # Bandpass for slow oscillations (0.5–2 Hz)
            so_filt = _bandpass(n23_concat, eeg_fs, 0.5, 2.0)

            # Find negative peaks (invert signal so find_peaks locates troughs)
            min_dist = int(0.5 * eeg_fs)   # minimum 0.5 s between SO troughs
            neg_peaks, _ = sp_signal.find_peaks(-so_filt, distance=min_dist)

            if len(neg_peaks) < 3:
                continue

            neg_amps, pos_amps, ptps, slopes = [], [], [], []

            for k in range(len(neg_peaks) - 1):
                n0 = neg_peaks[k]
                n1 = neg_peaks[k + 1]
                interval = n1 - n0

                # Duration gate: 0.5–2 s
                duration = interval / eeg_fs
                if not (0.5 <= duration <= 2.0):
                    continue

                neg_amp = so_filt[n0]          # negative value
                # Positive peak within the interval
                seg     = so_filt[n0:n1]
                pos_amp = float(np.max(seg))

                ptp = pos_amp - neg_amp
                if ptp < 0.1:                  # noise threshold (µV scale)
                    continue

                # Slope ≈ |neg_amp| / half-period
                half_dur = duration / 2.0
                slope    = abs(neg_amp) / half_dur if half_dur > 0 else 0.0

                neg_amps.append(abs(neg_amp))
                pos_amps.append(pos_amp)
                ptps.append(ptp)
                slopes.append(slope)

            b = ch_idx * 5
            if neg_amps:
                out[b + 0] = len(neg_amps) / n23_min if n23_min > 0 else 0.0
                out[b + 1] = float(np.mean(neg_amps))
                out[b + 2] = float(np.mean(pos_amps))
                out[b + 3] = float(np.mean(ptps))
                out[b + 4] = float(np.mean(slopes))
        except Exception:
            pass

    return out


# ─── [H] SO-Spindle coupling (Phase-Amplitude Coupling) ★ NOVEL ──────────────
#
# Motivation: Slow oscillation (SO, 0.5-2 Hz) — spindle (12-16 Hz) coupling
# reflects hippocampo-neocortical memory consolidation.  This coupling is
# disrupted in MCI/AD even before cognitive symptoms.
# Method: Mean Vector Length (MVL) PAC (Tort et al., J Neurophysiol 2010).
# References: Staresina et al. Nature Neurosci 2015; Mander et al. Neuron 2015.

def feat_so_spindle_coupling(phys, fs_dict, stages):
    """3 features: delta(SO)-sigma PAC (MVL) for C3-M2, C4-M1, F3-M2."""
    out = np.zeros(3)
    if len(stages) == 0:
        return out

    for idx, ch in enumerate(['c3-m2', 'c4-m1', 'f3-m2']):
        if ch not in phys:
            continue
        sig    = phys[ch]
        eeg_fs = fs_dict.get(ch, 200.0)
        epochs = _get_n2_epochs(sig, eeg_fs, stages)
        if not epochs:
            continue

        mvl_list = []
        for epoch in epochs:
            try:
                # SO phase (0.5–2 Hz)
                so_filt = _bandpass(epoch, eeg_fs, 0.5, 2.0)
                so_phase = np.angle(sp_signal.hilbert(so_filt))

                # Spindle amplitude envelope (12–16 Hz)
                sp_filt = _bandpass(epoch, eeg_fs, 12.0, 16.0)
                sp_amp  = np.abs(sp_signal.hilbert(sp_filt))

                # MVL = |mean(A * e^{i*phi})|  (Tort et al. 2010)
                mvl = np.abs(np.mean(sp_amp * np.exp(1j * so_phase)))
                mvl_list.append(mvl)
            except Exception:
                pass

        if mvl_list:
            out[idx] = np.mean(mvl_list)

    return out


# ─── [I] EEG complexity ★ NOVEL ───────────────────────────────────────────────
#
# Motivation: Neurodegeneration reduces EEG signal complexity.
# Hjorth parameters (1970) capture signal variance, mean frequency, and
# bandwidth. Spectral entropy captures distributional complexity of PSD.
# Computed per sleep stage (N2, N3, REM) on C3-M2 → 3 stages × 4 metrics = 12.

def feat_eeg_complexity(phys, fs_dict, stages):
    """12 features: [Hjorth_act, Hjorth_mob, Hjorth_cmp, SpectralEntropy]
                    × [N2, N3, REM]  for C3-M2."""
    out = np.zeros(12)
    ch  = 'c3-m2'
    if ch not in phys or len(stages) == 0:
        return out

    sig       = phys[ch]
    eeg_fs    = fs_dict.get(ch, 200.0)
    epoch_samp = int(EPOCH_SEC * eeg_fs)
    nperseg    = min(int(eeg_fs * 4), epoch_samp)

    stage_map = {S_N2: 0, S_N3: 1, S_REM: 2}
    accum = {k: [] for k in stage_map}

    for i, sv in enumerate(stages.astype(int)):
        if sv not in stage_map:
            continue
        start = i * epoch_samp
        end   = start + epoch_samp
        if end > len(sig):
            break
        epoch = sig[start:end]

        # Hjorth parameters
        act = np.var(epoch)
        d1  = np.diff(epoch)
        mob = np.sqrt(np.var(d1) / act) if act > 0 else 0.0
        d2  = np.diff(d1)
        mob2 = np.sqrt(np.var(d2) / np.var(d1)) if np.var(d1) > 0 else 0.0
        cmp  = mob2 / mob if mob > 0 else 0.0

        # Spectral entropy
        f, psd = sp_signal.welch(epoch, fs=eeg_fs, nperseg=nperseg)
        psd_n  = psd / (psd.sum() + 1e-12)  # normalize to probability
        sp_ent = -np.sum(psd_n * np.log2(psd_n + 1e-12))  # Shannon entropy

        accum[sv].append([act, mob, cmp, sp_ent])

    for stage_val, slot in stage_map.items():
        if accum[stage_val]:
            mean_vals = np.mean(accum[stage_val], axis=0)
            out[slot*4: slot*4+4] = mean_vals

    return out


# ─── [J] Sleep fragmentation ★ NOVEL ─────────────────────────────────────────
#
# Motivation: Sleep fragmentation (frequent brief awakenings) is a hallmark of
# MCI/AD even before global cognitive decline.  We capture both the rate and
# structure of nighttime awakenings via information-theoretic metrics.

def feat_sleep_fragmentation(stages):
    """5 features: stage_entropy, wake_bouts/hr, mean_wake_bout_min,
                   NREM_frag/hr (NREM→wake), stage_stability."""
    out = np.zeros(5)
    if len(stages) == 0:
        return out

    valid = stages[stages < S_UNK].astype(int)
    if len(valid) < 2:
        return out

    # Stage sequence entropy (Shannon)
    counts = np.array([np.sum(valid == s) for s in VALID_STAGES], dtype=float)
    probs  = counts / counts.sum()
    probs  = probs[probs > 0]
    out[0] = -np.sum(probs * np.log2(probs))  # max ≈ log2(5) ≈ 2.32

    # Wake bouts per hour and mean bout duration
    tst_hr = np.sum(valid != S_WAKE) * EPOCH_SEC / 3600.0
    wake_bin = (valid == S_WAKE).astype(int)
    starts   = np.where(np.diff(wake_bin, prepend=0) == 1)[0]
    ends     = np.where(np.diff(wake_bin, append=0) == -1)[0]
    if len(starts) > 0 and tst_hr > 0:
        out[1] = len(starts) / tst_hr
        bout_lens = (ends - starts) * EPOCH_SEC / 60.0
        out[2] = np.mean(bout_lens)

    # NREM fragmentation: NREM → Wake transitions per TST hour
    nrem   = np.isin(valid, [S_N3, S_N2, S_N1]).astype(int)
    wake_v = (valid == S_WAKE).astype(int)
    nrem2w = np.sum((np.diff(nrem) == -1) & (wake_v[1:] == 1))
    out[3] = nrem2w / tst_hr if tst_hr > 0 else 0.0

    # Stage stability: mean diagonal of stage transition matrix
    trans_mat = np.zeros((5, 5))
    for a, b in zip(valid[:-1], valid[1:]):
        if a in VALID_STAGES and b in VALID_STAGES:
            trans_mat[VALID_STAGES.index(a), VALID_STAGES.index(b)] += 1
    row_sums = trans_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_prob = trans_mat / row_sums
    out[4] = np.mean(np.diag(trans_prob))

    return out


# ─── [K] Spectral edge frequency ★ NOVEL ─────────────────────────────────────
#
# Motivation: SEF90 (frequency below which 90% of EEG power lies) shifts
# leftward (slower frequencies) as cortical activity decreases in dementia.
# Computed per key sleep stage on C3-M2.

def feat_spectral_edge(phys, fs_dict, stages):
    """3 features: SEF90 for N2, N3, REM (C3-M2, in Hz)."""
    out = np.zeros(3)
    ch  = 'c3-m2'
    if ch not in phys or len(stages) == 0:
        return out

    sig       = phys[ch]
    eeg_fs    = fs_dict.get(ch, 200.0)
    epoch_samp = int(EPOCH_SEC * eeg_fs)
    nperseg    = min(int(eeg_fs * 4), epoch_samp)
    stage_map  = {S_N2: 0, S_N3: 1, S_REM: 2}
    accum      = {k: [] for k in stage_map}

    for i, sv in enumerate(stages.astype(int)):
        if sv not in stage_map:
            continue
        start = i * epoch_samp
        end   = start + epoch_samp
        if end > len(sig):
            break
        f, psd = sp_signal.welch(sig[start:end], fs=eeg_fs, nperseg=nperseg)
        # Restrict to 0.5–50 Hz
        mask = (f >= 0.5) & (f <= 50.0)
        f_, p_ = f[mask], psd[mask]
        if len(p_) == 0:
            continue
        cumpower = np.cumsum(p_)
        threshold = 0.90 * cumpower[-1]
        idx = np.searchsorted(cumpower, threshold)
        sef90 = f_[min(idx, len(f_)-1)]
        accum[sv].append(sef90)

    for stage_val, slot in stage_map.items():
        if accum[stage_val]:
            out[slot] = np.mean(accum[stage_val])

    return out


# ─── [L] Waveform kurtosis ★ NOVEL ────────────────────────────────────────────
#
# Motivation: Waveform kurtosis captures the peakedness of the raw EEG signal,
# reflecting K-complex activity in N2 and slow-wave sharpness in N3.
# This was the single strongest predictor in the Brain Age Index study (n=7071).
# Reference: PMC12486002 (2025) — 13-feature BAI validated across 5 cohorts.

def feat_waveform_kurtosis(phys, fs_dict, stages):
    """6 features: kurtosis of concatenated raw EEG for N2, N3 × C3-M2, C4-M1, F3-M2."""
    out = np.zeros(6)
    if len(stages) == 0:
        return out
    stage_slots = {S_N2: 0, S_N3: 1}
    for ch_idx, ch in enumerate(['c3-m2', 'c4-m1', 'f3-m2']):
        if ch not in phys:
            continue
        sig        = phys[ch]
        eeg_fs     = fs_dict.get(ch, 200.0)
        epoch_samp = int(EPOCH_SEC * eeg_fs)
        for stage_val, stage_slot in stage_slots.items():
            buf = []
            for i, sv in enumerate(stages.astype(int)):
                if sv != stage_val:
                    continue
                start = i * epoch_samp
                end   = start + epoch_samp
                if end > len(sig):
                    break
                buf.append(sig[start:end])
            if len(buf) >= 2:
                concat = np.concatenate(buf)
                out[ch_idx * 2 + stage_slot] = float(sp_stats.kurtosis(concat))
    return out


# ─── [M] Band power kurtosis ★ NOVEL ─────────────────────────────────────────
#
# Motivation: Kurtosis of the distribution of per-epoch band powers captures
# how "spiky" or irregular the brain oscillation is across the night —
# a marker of neurodegeneration-related rhythm instability.
# Reference: Brain Age Index (PMC12486002) — band power kurtosis across N1-N3.

def feat_bandpower_kurtosis(phys, fs_dict, stages):
    """36 features: kurtosis of per-epoch band power distribution
    3ch × 3stages (N1,N2,N3) × 4bands (delta,theta,alpha,sigma)."""
    KB_STAGES = [S_N1, S_N2, S_N3]
    KB_BANDS  = [('delta', 0.5, 4.0), ('theta', 4.0, 8.0),
                 ('alpha', 8.0, 12.0), ('sigma', 12.0, 16.0)]
    N_KS, N_KB = len(KB_STAGES), len(KB_BANDS)
    out = np.zeros(N_EEG_CH * N_KS * N_KB)

    for ch_idx, ch in enumerate(['c3-m2', 'c4-m1', 'f3-m2']):
        if ch not in phys:
            continue
        sig        = phys[ch]
        eeg_fs     = fs_dict.get(ch, 200.0)
        epoch_samp = int(EPOCH_SEC * eeg_fs)
        nperseg    = min(int(eeg_fs * 4), epoch_samp)
        accum      = {s: [[] for _ in range(N_KB)] for s in KB_STAGES}

        for i, sv in enumerate(stages.astype(int)):
            if sv not in KB_STAGES:
                continue
            start = i * epoch_samp
            end   = start + epoch_samp
            if end > len(sig):
                break
            f, psd = sp_signal.welch(sig[start:end], fs=eeg_fs, nperseg=nperseg)
            for b_idx, (_, flo, fhi) in enumerate(KB_BANDS):
                m  = (f >= flo) & (f <= fhi)
                bp = np.trapz(psd[m], f[m]) if m.any() else 0.0
                accum[sv][b_idx].append(bp)

        for s_idx, stage_val in enumerate(KB_STAGES):
            for b_idx in range(N_KB):
                vals = accum[stage_val][b_idx]
                if len(vals) >= 4:
                    idx = ch_idx * N_KS * N_KB + s_idx * N_KB + b_idx
                    out[idx] = float(sp_stats.kurtosis(vals))
    return out


# ─── [N] N3 spectral ratios ★ NOVEL ──────────────────────────────────────────
#
# Motivation: In N3 sleep, MCI/dementia patients show EEG slowing (delta increase,
# alpha/sigma decrease). Delta/alpha and delta/theta ratios in N3 are among the
# most replicated EEG biomarkers of cognitive decline.
# Reference: SAGE 2025 (AUC 0.76), Brain Age Index (PMC12486002).

def feat_n3_ratios(phys, fs_dict, stages):
    """3 features: delta/alpha ratio, delta/theta ratio, sigma/delta ratio in N3 (C3-M2)."""
    out = np.zeros(3)
    ch  = 'c3-m2'
    if ch not in phys or len(stages) == 0:
        return out

    sig        = phys[ch]
    eeg_fs     = fs_dict.get(ch, 200.0)
    epoch_samp = int(EPOCH_SEC * eeg_fs)
    nperseg    = min(int(eeg_fs * 4), epoch_samp)
    d, t, a, s = [], [], [], []

    for i, sv in enumerate(stages.astype(int)):
        if sv != S_N3:
            continue
        start = i * epoch_samp
        end   = start + epoch_samp
        if end > len(sig):
            break
        f, psd = sp_signal.welch(sig[start:end], fs=eeg_fs, nperseg=nperseg)
        def bp(lo, hi):
            m = (f >= lo) & (f <= hi)
            return np.trapz(psd[m], f[m]) if m.any() else 0.0
        d.append(bp(0.5, 4.0))
        t.append(bp(4.0, 8.0))
        a.append(bp(8.0, 12.0))
        s.append(bp(12.0, 16.0))

    if not d:
        return out
    dm, tm, am, sm = np.mean(d), np.mean(t), np.mean(a), np.mean(s)
    out[0] = dm / am if am > 0 else 0.0  # delta/alpha — EEG slowing marker
    out[1] = dm / tm if tm > 0 else 0.0  # delta/theta — cognitive decline marker
    out[2] = sm / dm if dm > 0 else 0.0  # sigma/delta — spindle vs slow-wave balance
    return out


# ─── [P] EEG Inter-channel Coherence ★ NOVEL ─────────────────────────────────
#
# Motivation: Inter-hemispheric EEG coherence reflects long-range cortical
# connectivity.  MCI/dementia shows disrupted thalamocortical and cortico-cortical
# synchronization, reducing coherence in slow (delta/theta) and fast (sigma/beta)
# bands during NREM sleep.
# Reference: Sun et al. SLEEP 2023 — coherence features achieved AUROC 0.78.

def feat_eeg_coherence(phys, fs_dict, stages):
    """[P] 75 EEG inter-channel coherence features.
    3 channel pairs × 5 frequency bands × 5 sleep stages.

    Pairs: (C3-M2, C4-M1), (C3-M2, F3-M2), (C4-M1, F3-M2)
    Bands: delta(0.5-4), theta(4-8), alpha(8-12), sigma(12-16), beta(16-30) Hz
    Stages: N3, N2, N1, REM, Wake

    Uses mean magnitude-squared coherence (MSC) per epoch, then averaged per stage.
    """
    PAIRS = [('c3-m2', 'c4-m1'), ('c3-m2', 'f3-m2'), ('c4-m1', 'f3-m2')]
    COH_BANDS = [
        ('delta', 0.5,  4.0),
        ('theta', 4.0,  8.0),
        ('alpha', 8.0, 12.0),
        ('sigma',12.0, 16.0),
        ('beta', 16.0, 30.0),
    ]
    N_PAIRS = len(PAIRS)   # 3
    N_CB    = len(COH_BANDS)  # 5
    out = np.zeros(N_PAIRS * N_STAGES * N_CB)  # 75

    if len(stages) == 0:
        return out

    stages_int = stages.astype(int)

    for p_idx, (ch1, ch2) in enumerate(PAIRS):
        if ch1 not in phys or ch2 not in phys:
            continue
        sig1 = phys[ch1]
        sig2 = phys[ch2]
        # Use the slower fs if channels differ (shouldn't happen in practice)
        sf = min(fs_dict.get(ch1, 200.0), fs_dict.get(ch2, 200.0))
        ep = int(EPOCH_SEC * sf)
        nps = min(int(sf * 4), ep)

        accum = {sv: [] for sv in VALID_STAGES}

        for i, sv in enumerate(stages_int):
            if sv not in VALID_STAGES:
                continue
            start = i * ep
            end   = start + ep
            if end > len(sig1) or end > len(sig2):
                break
            try:
                f, Cxy = sp_signal.coherence(sig1[start:end], sig2[start:end],
                                              fs=sf, nperseg=nps)
                band_coh = []
                for _, flo, fhi in COH_BANDS:
                    m    = (f >= flo) & (f <= fhi)
                    vals = Cxy[m][~np.isnan(Cxy[m])] if m.any() else np.array([])
                    band_coh.append(float(np.mean(vals)) if len(vals) > 0 else 0.0)
                accum[sv].append(band_coh)
            except Exception:
                pass

        for s_idx, sv in enumerate(VALID_STAGES):
            if accum[sv]:
                mean_coh = np.mean(accum[sv], axis=0)  # (N_CB,)
                base = p_idx * N_STAGES * N_CB + s_idx * N_CB
                out[base: base + N_CB] = mean_coh

    return out


# ─── [Q] REM Spectral Ratios ★ NOVEL ─────────────────────────────────────────
#
# Motivation: REM sleep EEG ratios reflect thalamocortical oscillatory dynamics.
# In MCI/AD, REM theta activity is reduced while alpha/sigma power increases,
# creating characteristic ratio shifts detectable before overt symptoms.
# Reference: Gao et al. 2022; Westerberg et al. 2021.

def feat_rem_spectral_ratios(phys, fs_dict, stages):
    """[Q] 9 REM spectral ratio features.
    theta/alpha, theta/beta, sigma/alpha × 3 EEG channels (C3-M2, C4-M1, F3-M2) in REM.
    """
    out = np.zeros(9)
    if len(stages) == 0:
        return out

    for ch_idx, ch in enumerate(['c3-m2', 'c4-m1', 'f3-m2']):
        if ch not in phys:
            continue
        sig        = phys[ch]
        eeg_fs     = fs_dict.get(ch, 200.0)
        epoch_samp = int(EPOCH_SEC * eeg_fs)
        nperseg    = min(int(eeg_fs * 4), epoch_samp)
        t_list, a_list, b_list, s_list = [], [], [], []

        for i, sv in enumerate(stages.astype(int)):
            if sv != S_REM:
                continue
            start = i * epoch_samp
            end   = start + epoch_samp
            if end > len(sig):
                break
            f, psd = sp_signal.welch(sig[start:end], fs=eeg_fs, nperseg=nperseg)
            def _bp(lo, hi):
                m = (f >= lo) & (f <= hi)
                return np.trapz(psd[m], f[m]) if m.any() else 0.0
            t_list.append(_bp(4.0,  8.0))
            a_list.append(_bp(8.0, 12.0))
            b_list.append(_bp(16.0, 30.0))
            s_list.append(_bp(12.0, 16.0))

        if t_list:
            tm = np.mean(t_list)
            am = np.mean(a_list)
            bm = np.mean(b_list)
            sm = np.mean(s_list)
            base = ch_idx * 3
            out[base+0] = tm / am if am > 0 else 0.0  # theta/alpha REM
            out[base+1] = tm / bm if bm > 0 else 0.0  # theta/beta  REM
            out[base+2] = sm / am if am > 0 else 0.0  # sigma/alpha REM

    return out


# ─── [R] REM Slow-Fast Activity Ratio (SFAR) ★ NOVEL ─────────────────────────
#
# Motivation: The Slow-Fast Activity Ratio (SFAR) captures the global balance of
# low-frequency (slow: delta+theta) vs high-frequency (fast: alpha+sigma+beta)
# power in REM sleep.  SFAR in REM is elevated in dementia patients due to EEG
# slowing (loss of fast rhythms).
# Reference: MFDE-SFAR framework; Moretti et al. 2012.

def feat_rem_sfar(phys, fs_dict, stages):
    """[R] 3 REM slow-fast activity ratio features (one per EEG channel).
    log(slow / fast) during REM:  slow = delta+theta (0.5-8 Hz),
                                  fast  = alpha+sigma+beta (8-30 Hz).
    """
    out = np.zeros(3)
    if len(stages) == 0:
        return out

    for ch_idx, ch in enumerate(['c3-m2', 'c4-m1', 'f3-m2']):
        if ch not in phys:
            continue
        sig        = phys[ch]
        eeg_fs     = fs_dict.get(ch, 200.0)
        epoch_samp = int(EPOCH_SEC * eeg_fs)
        nperseg    = min(int(eeg_fs * 4), epoch_samp)
        ratios = []

        for i, sv in enumerate(stages.astype(int)):
            if sv != S_REM:
                continue
            start = i * epoch_samp
            end   = start + epoch_samp
            if end > len(sig):
                break
            f, psd = sp_signal.welch(sig[start:end], fs=eeg_fs, nperseg=nperseg)
            slow_m = (f >= 0.5) & (f < 8.0)
            fast_m = (f >= 8.0) & (f <= 30.0)
            slow = np.trapz(psd[slow_m], f[slow_m]) if slow_m.any() else 0.0
            fast = np.trapz(psd[fast_m], f[fast_m]) if fast_m.any() else 0.0
            if fast > 0:
                ratios.append(np.log(slow / fast + 1e-10))

        if ratios:
            out[ch_idx] = np.mean(ratios)

    return out


# ─── [S] Philosopher's Stone Transfer Learning ★ NOVEL ───────────────────────
#
# Philosopher's Stone (Ganglberger et al., NEJM AI 2026) is a deep learning
# model trained on 36,000 PSG recordings from 27,000 subjects (FHS, MESA,
# MrOS, SOF, KoGES, MGH cohorts).  It produces a 1024-D latent embedding and
# interpretable brain health scores from overnight C4-M1 EEG.
# We use the 4 scalar outputs: brain_health_score + 3 cognition scores.
# Reference: https://github.com/bdsp-core/philosophers-stone
# Model: huggingface.co/wolfgang-ganglberger/philosophers-stone

def _ps_write_h5(sig, eeg_fs, h5_path):
    """Write a single-channel EEG signal in PS-compatible HDF5 format.

    PS requires exactly 200 Hz; resample if the source differs.
    """
    import h5py
    from math import gcd
    from scipy.signal import resample_poly

    sig = sig.astype(np.float32)
    target_fs = 200
    src_fs = int(round(eeg_fs))
    if src_fs != target_fs:
        g = gcd(src_fs, target_fs)
        sig = resample_poly(sig, target_fs // g, src_fs // g).astype(np.float32)
    # Cap at 6h at 200 Hz: peak RSS scales with length.
    # 7.23h -> 32.51 GB peak, exceeds Docker 31 GB limit; 6h -> ~27 GB.
    sig = sig[:6 * 3600 * 200]

    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('signals/c4-m1', data=sig.reshape(-1, 1))
        f.attrs['sampling_rate'] = 200.0  # always 200 Hz after resampling
        f.attrs['unit_voltage']  = 'uV'


def _precompute_ps_batch(data_folder, model_folder, records, verbose=True):
    """Run Philosopher's Stone on all records in ONE subprocess call.

    Results are saved to <model_folder>/ps_cache.csv and loaded into _PS_CACHE.
    H5 files and philosopher.py output are written to <model_folder>/ps_batch/
    (persistent across runs) so that partial results survive timeout/crash.
    """
    global _PS_CACHE, _PS_PCA
    import pandas as pd
    import subprocess
    from sklearn.decomposition import PCA

    cache_csv = os.path.join(model_folder, PS_CACHE_FILE)
    lhl_cols  = [f'lhl_{i}' for i in range(1, 1025)]

    def _load_cache_df(path):
        """Load cache CSV and return (dict, pca) with latents + fitted PCA."""
        df = pd.read_csv(path)
        cache = {}
        latents_list = []
        keys_list    = []
        for _, row in df.iterrows():
            scores = np.array([
                float(row.get('brain_health_score',          0.0) or 0.0),
                float(row.get('total_cognition_score',        0.0) or 0.0),
                float(row.get('fluid_cognition_score',        0.0) or 0.0),
                float(row.get('crystallized_cognition_score', 0.0) or 0.0),
            ], dtype=np.float32)
            has_latent = any(c in df.columns for c in ['lhl_1', 'lhl_2'])
            if has_latent:
                latent = np.array([float(row.get(c, 0.0) or 0.0) for c in lhl_cols],
                                  dtype=np.float32)
                cache[row['patient_key']] = np.concatenate([scores, latent])
                latents_list.append(latent)
                keys_list.append(row['patient_key'])
            else:
                cache[row['patient_key']] = scores
        pca = None
        if latents_list:
            pca = PCA(n_components=min(PS_N_PCA, len(latents_list)))
            pca.fit(np.stack(latents_list))
        return cache, pca

    # Fast path 1: model_folder cache
    if os.path.exists(cache_csv):
        _PS_CACHE, _PS_PCA = _load_cache_df(cache_csv)
        if verbose:
            has_lat = any(v.shape[0] > 4 for v in _PS_CACHE.values())
            print(f'[PS] Loaded {len(_PS_CACHE)} pre-computed PS features from cache '
                  f'({"with" if has_lat else "without"} latents).')
        return

    # Fast path 2: baked-in cache (Docker image)
    if os.path.exists(PS_BAKED_CACHE):
        if verbose:
            print(f'[PS] Loading baked PS cache from {PS_BAKED_CACHE}...')
        _PS_CACHE, _PS_PCA = _load_cache_df(PS_BAKED_CACHE)
        # Copy baked cache to model_folder for future runs
        import shutil
        shutil.copy2(PS_BAKED_CACHE, cache_csv)
        if verbose:
            print(f'[PS] Loaded {len(_PS_CACHE)} patients from baked cache.')
        return

    if not os.path.exists(PS_DIR):
        _PS_CACHE = {}
        return

    demo_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    if verbose:
        print(f'[PS] Pre-computing Philosopher Stone features for {len(records)} patients...')

    # Use a persistent directory so partial results survive timeout/crash
    batch_dir = os.path.join(model_folder, 'ps_batch')
    os.makedirs(batch_dir, exist_ok=True)

    manifest_rows = []
    filepath_to_key = {}

    for rec in tqdm(records, desc='[PS] H5', disable=not verbose):
        pid = rec[HEADERS['bids_folder']]
        sid = rec[HEADERS['site_id']]
        ses = rec[HEADERS['session_id']]
        key = f'{pid}_ses-{ses}'

        phys_path = os.path.join(
            data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
            sid, f'{pid}_ses-{ses}.edf')
        if not os.path.exists(phys_path):
            continue

        try:
            patient_data = load_demographics(demo_file, pid, ses)
            age = load_age(patient_data)
            if np.isnan(float(age)) or float(age) <= 0:
                age = 65.0
            sex = 1 if load_sex(patient_data) == 'Male' else 0

            phys_raw, phys_fs_raw = load_signal_data(phys_path)
            phys, fs = _build_channel_dict(phys_raw, phys_fs_raw, DEFAULT_CSV)
            if 'c4-m1' not in phys:
                continue

            h5_path = os.path.join(batch_dir, f'{key}.h5')
            if not os.path.exists(h5_path):
                _ps_write_h5(phys['c4-m1'], fs.get('c4-m1', 200.0), h5_path)

            manifest_rows.append({'filepath': h5_path, 'age': float(age), 'sex': int(sex)})
            filepath_to_key[h5_path] = key
        except Exception as e:
            if verbose:
                tqdm.write(f'  ! [PS] {pid}: {e}')

    if not manifest_rows:
        _PS_CACHE = {}
        return

    manifest_csv = os.path.join(batch_dir, 'manifest.csv')
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)

    results_csv = os.path.join(batch_dir, 'phi_results.csv')

    # Resume: skip already-processed patients if phi_results.csv exists
    existing_res_df = pd.read_csv(results_csv) if os.path.exists(results_csv) else pd.DataFrame()
    already_done = set(existing_res_df['filepath'].astype(str)) if not existing_res_df.empty else set()

    remaining_rows = [r for r in manifest_rows if r['filepath'] not in already_done]
    if verbose:
        print(f'[PS] {len(already_done)} already done, {len(remaining_rows)} remaining.')

    if remaining_rows:
        remaining_csv = os.path.join(batch_dir, 'manifest_remaining.csv')
        pd.DataFrame(remaining_rows).to_csv(remaining_csv, index=False)
        # philosopher.py appends to phi_results.csv when it already exists
        # We rename existing so philosopher writes a fresh file, then we merge
        phi_partial = results_csv + '.partial'
        if os.path.exists(results_csv):
            os.rename(results_csv, phi_partial)

        if verbose:
            print(f'[PS] Running philosopher.py on {len(remaining_rows)} records (~{len(remaining_rows)*100//3600}h)...')

        try:
            subprocess.run(
                ['python', 'philosopher.py',
                 '--manifest_csv', remaining_csv,
                 '--outdir', batch_dir,
                 '--no-save-plots', '--no-save-json'],
                capture_output=False,
                timeout=3600 * 30,
                cwd=PS_DIR,
            )
        except subprocess.TimeoutExpired:
            if verbose:
                print('[PS] philosopher.py timed out — using partial results.')
        except Exception as e:
            if verbose:
                print(f'[PS] philosopher.py error: {e}')

        # Merge partial + new results
        frames = []
        if os.path.exists(phi_partial):
            frames.append(pd.read_csv(phi_partial))
        if os.path.exists(results_csv):
            frames.append(pd.read_csv(results_csv))
        if frames:
            merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=['filepath'], keep='last')
            merged.to_csv(results_csv, index=False)
        elif os.path.exists(phi_partial):
            os.rename(phi_partial, results_csv)

    # Read whatever results exist (full or partial); include latent vectors
    _PS_CACHE = {}
    rows_out  = []
    latents_list = []
    keys_list    = []
    if os.path.exists(results_csv):
        res_df = pd.read_csv(results_csv)
        for _, r in res_df.iterrows():
            fp  = str(r.get('filepath', ''))
            key = filepath_to_key.get(fp)
            if key is None:
                continue
            scores = np.array([
                float(r.get('brain_health_score',          0.0) or 0.0),
                float(r.get('total_cognition_score',        0.0) or 0.0),
                float(r.get('fluid_cognition_score',        0.0) or 0.0),
                float(r.get('crystallized_cognition_score', 0.0) or 0.0),
            ], dtype=np.float32)
            latent = np.array([float(r.get(f'lhl_{i}', 0.0) or 0.0) for i in range(1, 1025)],
                              dtype=np.float32)
            _PS_CACHE[key] = np.concatenate([scores, latent])
            latents_list.append(latent)
            keys_list.append(key)
            row_d = {'patient_key': key,
                     'brain_health_score':          scores[0],
                     'total_cognition_score':        scores[1],
                     'fluid_cognition_score':        scores[2],
                     'crystallized_cognition_score': scores[3]}
            for i, v in enumerate(latent, 1):
                row_d[f'lhl_{i}'] = float(v)
            rows_out.append(row_d)
    else:
        if verbose:
            print('[PS] philosopher.py produced no results CSV.')

    # Fit PCA on all collected latents
    if latents_list:
        _PS_PCA = PCA(n_components=min(PS_N_PCA, len(latents_list)))
        _PS_PCA.fit(np.stack(latents_list))
        if verbose:
            var = float(_PS_PCA.explained_variance_ratio_.sum())
            print(f'[PS] PCA fitted on {len(latents_list)} latents '
                  f'({PS_N_PCA} components → {var:.1%} variance).')

    pd.DataFrame(rows_out).to_csv(cache_csv, index=False)
    if verbose:
        print(f'[PS] Saved {len(_PS_CACHE)} PS features → {cache_csv}')

def _apply_ps_pca(cached):
    """Convert cached 1028-D array [4 scores | 1024 latents] → PS_FEAT_DIM array."""
    out = np.zeros(PS_FEAT_DIM, dtype=np.float32)
    out[:4] = cached[:4]
    if _PS_PCA is not None and len(cached) > 4:
        latent = cached[4:].reshape(1, -1)
        pca_vec = _PS_PCA.transform(latent)[0].astype(np.float32)
        n = min(len(pca_vec), PS_N_PCA)
        out[4:4+n] = pca_vec[:n]
    return out


def feat_philosophers_stone(phys, fs_dict, patient_data):
    """[S] PS_FEAT_DIM Philosopher's Stone features (graceful fallback to zeros).

    Returns 4 scalar scores + PS_N_PCA PCA components of the 1024-D latent vector.

    Training path: reads pre-computed values from _PS_CACHE + applies _PS_PCA.
    Inference path: runs philosopher.py subprocess, applies saved _PS_PCA.
    """
    out = np.zeros(PS_FEAT_DIM, dtype=np.float32)

    ch = 'c4-m1'
    if ch not in phys:
        return out

    # ── Training path: use pre-computed cache ─────────────────────────────
    global _PS_CACHE
    if _PS_CACHE is not None:
        pid = patient_data.get(HEADERS['bids_folder'], '')
        ses = patient_data.get(HEADERS['session_id'], '')
        key = f'{pid}_ses-{ses}'
        cached = _PS_CACHE.get(key)
        if cached is not None:
            return _apply_ps_pca(cached)
        return out  # missing from cache → zeros (graceful)

    # ── Inference path: single subprocess call (PS model loaded once) ─────
    if not os.path.exists(PS_DIR):
        return out

    try:
        import subprocess
        import tempfile
        import pandas as pd

        sig    = phys[ch].astype(np.float32)
        eeg_fs = fs_dict.get(ch, 200.0)
        age    = load_age(patient_data)
        if np.isnan(float(age)) or float(age) <= 0:
            age = 65.0
        sex = 1 if load_sex(patient_data) == 'Male' else 0

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, 'eeg.h5')
            _ps_write_h5(sig, eeg_fs, h5_path)

            manifest_csv = os.path.join(tmpdir, 'manifest.csv')
            pd.DataFrame([{'filepath': h5_path, 'age': float(age), 'sex': int(sex)}]).to_csv(
                manifest_csv, index=False)

            subprocess.run(
                ['python', 'philosopher.py',
                 '--manifest_csv', manifest_csv,
                 '--outdir', tmpdir,
                 '--no-save-plots', '--no-save-json'],
                capture_output=True, text=True,
                timeout=300, cwd=PS_DIR,
            )

            results_csv = os.path.join(tmpdir, 'phi_results.csv')
            if os.path.exists(results_csv):
                df  = pd.read_csv(results_csv)
                if len(df) > 0:
                    row    = df.iloc[0]
                    scores = np.array([
                        float(row.get('brain_health_score',          0.0) or 0.0),
                        float(row.get('total_cognition_score',        0.0) or 0.0),
                        float(row.get('fluid_cognition_score',        0.0) or 0.0),
                        float(row.get('crystallized_cognition_score', 0.0) or 0.0),
                    ], dtype=np.float32)
                    latent = np.array([float(row.get(f'lhl_{i}', 0.0) or 0.0)
                                       for i in range(1, 1025)], dtype=np.float32)
                    cached = np.concatenate([scores, latent])
                    out = _apply_ps_pca(cached)
    except Exception:
        pass  # graceful fallback — zeros indicate PS unavailable

    return out


def feat_ps_scalars(patient_data):
    """[S] 4 Philosopher's Stone scalar features (no latent PCA).

    Returns brain_health_score, total/fluid/crystallized cognition scores.
    Uses pre-computed _PS_CACHE; graceful zero-fallback when unavailable.
    Lighter than full PS (no PCA noise from latents).
    """
    out = np.zeros(4, dtype=np.float32)
    global _PS_CACHE
    if _PS_CACHE is None:
        return out
    pid    = patient_data.get(HEADERS['bids_folder'], '')
    ses    = patient_data.get(HEADERS['session_id'],  '')
    key    = f'{pid}_ses-{ses}'
    cached = _PS_CACHE.get(key)
    if cached is not None:
        out[:4] = cached[:4]
    return out


# ─── CAISR Sequence Extraction ────────────────────────────────────────────────

def _extract_caisr_seq(record, data_folder):
    """Returns (MAX_SEQ_LEN, SEQ_CHANNELS) float32 array of CAISR stage probs.

    Channels: [prob_w, prob_n1, prob_n2, prob_n3, prob_r]
    Unavailable epochs (value=9) are zeroed out.
    Sequences shorter than MAX_SEQ_LEN are zero-padded at the end.
    """
    pid = record[HEADERS['bids_folder']]
    sid = record[HEADERS['site_id']]
    ses = record[HEADERS['session_id']]

    algo_path = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                             sid, f'{pid}_ses-{ses}_caisr_annotations.edf')
    algo_data, _ = (load_signal_data(algo_path)
                    if os.path.exists(algo_path) else ({}, {}))

    keys = ['caisr_prob_w', 'caisr_prob_n1', 'caisr_prob_n2',
            'caisr_prob_n3', 'caisr_prob_r']
    channels = []
    for k in keys:
        arr = np.asarray(algo_data.get(k, []), dtype=np.float32)
        arr[arr >= 9.0] = 0.0          # unavailable → 0
        arr = np.clip(arr, 0.0, 1.0)
        channels.append(arr)

    if not any(len(c) > 0 for c in channels):
        return np.zeros((MAX_SEQ_LEN, SEQ_CHANNELS), dtype=np.float32)

    T = max(len(c) for c in channels)
    # Align all channels to same length
    aligned = np.zeros((T, SEQ_CHANNELS), dtype=np.float32)
    for i, c in enumerate(channels):
        L = min(len(c), T)
        aligned[:L, i] = c[:L]

    # Truncate or pad to MAX_SEQ_LEN
    if T >= MAX_SEQ_LEN:
        return aligned[:MAX_SEQ_LEN]
    padded = np.zeros((MAX_SEQ_LEN, SEQ_CHANNELS), dtype=np.float32)
    padded[:T] = aligned
    return padded


# ─── 1D-CNN for CAISR Sequence ────────────────────────────────────────────────

class SleepCNN(nn.Module if HAS_TORCH else object):
    """Small 1D-CNN: (batch, T, 5) → binary logit.

    Architecture is intentionally small to avoid overfitting on ~780 samples.
    Uses BatchNorm + Dropout for regularisation.
    """
    def __init__(self):
        if not HAS_TORCH:
            return
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(SEQ_CHANNELS, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (B, T, C) → permute to (B, C, T) for Conv1d
        return self.head(self.encoder(x.permute(0, 2, 1))).squeeze(1)


def _train_cnn(seq_arr, y, n_epochs=40, batch_size=32, lr=1e-3, verbose=False):
    """Train SleepCNN. seq_arr: (N, T, C), y: (N,). Returns trained model."""
    if not HAS_TORCH:
        return None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t = torch.tensor(seq_arr, dtype=torch.float32)
    y_t = torch.tensor(y,       dtype=torch.float32)
    dl  = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    model = SleepCNN().to(device)
    opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for ep in range(n_epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sched.step()
        if verbose and (ep + 1) % 10 == 0:
            print(f'  [CNN] epoch {ep+1}/{n_epochs}')
    model.eval()
    return model


def _predict_cnn(model, seq_arr):
    """Returns 1-D numpy array of probabilities. seq_arr: (N, T, C)."""
    if not HAS_TORCH or model is None:
        return np.full(len(seq_arr), 0.5)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t    = torch.tensor(seq_arr, dtype=torch.float32).to(device)
        logits = model(X_t)
        probs  = torch.sigmoid(logits).cpu().numpy()
    return probs
