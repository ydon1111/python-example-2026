#!/usr/bin/env python
# team_code.py — PhysioNet Challenge 2026
#
# Feature groups (total = 319):
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
FEATURE_VERSION = 'v21'  # bump when feature layout changes to invalidate old cache
PS_DIR          = '/ps'  # Philosopher's Stone repo path in Docker
PS_CACHE_FILE   = 'ps_cache.csv'  # pre-computed PS scores saved in model_folder
PS_BAKED_CACHE  = os.path.join(SCRIPT_DIR, 'ps_cache_baked.csv')  # baked into Docker image
PS_N_PCA        = 50    # PCA components from 1024-D latent → explains ~98% variance
PS_LATENT_DIM   = 1024  # PS latent vector dimensionality

# ── Philosopher's Stone ───────────────────────────────────────────────────────
_PS_CACHE      = None   # baked scalar dict loaded lazily during inference
_PS_MODEL_OBJ  = None   # pre-loaded torch model (loaded once in load_model)
PS_SCALAR_COLS = ['brain_health_score', 'total_cognition_score',
                  'fluid_cognition_score', 'crystallized_cognition_score']
N_PS_FEATS     = len(PS_SCALAR_COLS) + PS_N_PCA   # 4 scalars + 50 PCA = 54

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

def _load_ps_baked(verbose=False):
    """Load PS baked scalar cache → dict: patient_key → np.array(4)."""
    import pandas as pd
    try:
        df = pd.read_csv(PS_BAKED_CACHE, index_col='patient_key',
                         usecols=['patient_key'] + PS_SCALAR_COLS)
        d = {k: np.array(row, dtype=np.float32)
             for k, row in df[PS_SCALAR_COLS].iterrows()}
        if verbose:
            print(f'[PS] baked cache loaded: {len(d)} patients')
        return d
    except Exception as e:
        if verbose:
            print(f'[PS] baked cache unavailable: {e}')
        return {}


def _load_ps_latents(verbose=False):
    """Load PS latent vectors from baked cache → dict: patient_key → np.array(1024)."""
    import pandas as pd
    try:
        header = pd.read_csv(PS_BAKED_CACHE, nrows=0)
        lat_cols = [c for c in header.columns if c.startswith('lhl_')]
        if not lat_cols:
            return {}
        df = pd.read_csv(PS_BAKED_CACHE, index_col='patient_key',
                         usecols=['patient_key'] + lat_cols)
        d = {k: np.array(row, dtype=np.float32)
             for k, row in df[lat_cols].iterrows()}
        if verbose:
            print(f'[PS] latents loaded: {len(d)} patients, {len(lat_cols)} dims')
        return d
    except Exception as e:
        if verbose:
            print(f'[PS] latents unavailable: {e}')
        return {}


def _load_ps_model(verbose=False):
    """Load Philosopher's Stone torch model once; return model or None."""
    global _PS_MODEL_OBJ
    if _PS_MODEL_OBJ is not None:
        return _PS_MODEL_OBJ
    try:
        import sys
        if PS_DIR not in sys.path:
            sys.path.insert(0, PS_DIR)
        from philosopher import run_philosopher  # noqa: F401 — just check import
        from phi_utils.philosopher_utils import Config as PhiConfig
        from phi_utils.philosopher_utils import load_model as ps_load_model
        cfg = PhiConfig()
        _PS_MODEL_OBJ = ps_load_model(cfg)
        if verbose:
            print('[PS] model loaded successfully')
    except Exception as e:
        _PS_MODEL_OBJ = None
        if verbose:
            print(f'[PS] model load failed: {e}')
    return _PS_MODEL_OBJ


def _get_ps_features(record, data_folder, ps_model=None, ps_cache=None,
                     ps_latents=None, ps_pca=None):
    """Return 4 scalars + PS_N_PCA PCA components (total N_PS_FEATS=54); zeros on failure."""
    pid  = record[HEADERS['bids_folder']]
    ses  = record[HEADERS['session_id']]
    site = record.get(HEADERS['site_id'], 'S0001')
    key  = f'{pid}_ses-{ses}'

    scalars = np.zeros(4, dtype=np.float32)
    latents = np.zeros(PS_LATENT_DIM, dtype=np.float32)

    # 1. Try baked cache (fast)
    if ps_cache and key in ps_cache:
        scalars = ps_cache[key]
    if ps_latents and key in ps_latents:
        latents = ps_latents[key]

    # 2. If latents still zero, run PS model (inference path)
    if not np.any(latents != 0):
        try:
            import sys, tempfile, pandas as pd
            if PS_DIR not in sys.path:
                sys.path.insert(0, PS_DIR)
            from philosopher import run_philosopher

            phys_path = os.path.join(
                data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site,
                f'{pid}_ses-{ses}.edf')
            if os.path.exists(phys_path):
                demo_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
                pdata = load_demographics(demo_file, pid, ses)
                age   = float(load_age(pdata) or 0)
                sex   = 1 if load_sex(pdata) == 'Male' else 0
                manifest = pd.DataFrame({'filepath': [phys_path],
                                         'age': [age], 'sex': [sex]})
                with tempfile.TemporaryDirectory() as tmpdir:
                    result_df, _ = run_philosopher(
                        manifest, outdir=tmpdir, model=ps_model,
                        save_summary=False, save_json=False,
                        save_plots=False, show_progress=False, verbose=False,
                    )
                if result_df is not None and len(result_df) > 0:
                    row = result_df.iloc[0]
                    scalars = np.array([row.get(c, 0.0) for c in PS_SCALAR_COLS],
                                       dtype=np.float32)
                    lat_cols = sorted([c for c in result_df.columns
                                       if c.startswith('lhl_')])
                    if lat_cols:
                        latents = np.array([row.get(c, 0.0) for c in lat_cols],
                                           dtype=np.float32)
        except Exception:
            pass

    # 3. Apply PCA to latents → 50 components
    pca_feats = np.zeros(PS_N_PCA, dtype=np.float32)
    if ps_pca is not None and np.any(latents != 0):
        try:
            pca_feats = ps_pca.transform(latents.reshape(1, -1))[0].astype(np.float32)
            pca_feats = np.nan_to_num(pca_feats, nan=0.0)
        except Exception:
            pass

    return np.concatenate([scalars, pca_feats])   # shape (54,)


def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV):
    demo_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    records   = find_patients(demo_file)

    if verbose:
        print(f'[train] {len(records)} records | feature_version={FEATURE_VERSION}')

    os.makedirs(model_folder, exist_ok=True)
    cache_dir = os.path.join(model_folder, CACHE_SUBDIR)
    os.makedirs(cache_dir, exist_ok=True)

    # Load PS baked cache (scalars + latents) before extraction loop
    _ps_cache   = _load_ps_baked(verbose=verbose)
    _ps_latents = _load_ps_latents(verbose=verbose)

    features, labels, sites = [], [], []
    ps_scalar_rows, ps_latent_rows = [], []
    pbar = tqdm(records, desc='Extracting', disable=not verbose)
    for rec in pbar:
        pid = rec[HEADERS['bids_folder']]
        ses = rec[HEADERS['session_id']]
        key = f'{pid}_ses-{ses}'
        try:
            X     = extract_all_features(rec, data_folder, csv_path, cache_dir)
            label = load_diagnoses(demo_file, pid)
            features.append(X)
            labels.append(label)
            sites.append(rec.get(HEADERS['site_id'], 'S0001'))
            ps_scalar_rows.append(_ps_cache.get(key, np.zeros(4, dtype=np.float32)))
            ps_latent_rows.append(_ps_latents.get(key, np.zeros(PS_LATENT_DIM, dtype=np.float32)))
        except Exception as e:
            tqdm.write(f'  ! {pid}: {e}')

    X_arr    = np.array(features, dtype=np.float32)
    X_arr    = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
    y_arr    = np.array(labels,   dtype=int)
    site_arr = np.array(sites)

    # Build PS features: 4 scalars + PCA50 on 1024-D latents = 54 total
    from sklearn.decomposition import PCA as _PCA
    scalar_arr = np.nan_to_num(np.array(ps_scalar_rows, dtype=np.float32), nan=0.0)
    lat_arr    = np.nan_to_num(np.array(ps_latent_rows, dtype=np.float32), nan=0.0)
    has_lat    = np.any(lat_arr != 0, axis=1)
    ps_pca     = None
    pca_feats  = np.zeros((len(lat_arr), PS_N_PCA), dtype=np.float32)
    if has_lat.sum() > PS_N_PCA:
        ps_pca = _PCA(n_components=PS_N_PCA, random_state=42)
        ps_pca.fit(lat_arr[has_lat])
        pca_feats[has_lat] = ps_pca.transform(lat_arr[has_lat]).astype(np.float32)
        pca_feats = np.nan_to_num(pca_feats, nan=0.0)
        if verbose:
            print(f'[PS] PCA {PS_N_PCA} components | '
                  f'var={ps_pca.explained_variance_ratio_.sum():.1%} | '
                  f'coverage={has_lat.mean():.1%}')
    PS_arr = np.concatenate([scalar_arr, pca_feats], axis=1)  # (N, 54)
    X_arr  = np.concatenate([X_arr, PS_arr], axis=1)          # 319+54 = 373 features
    if verbose:
        print(f'[PS] features appended | scalars(4)+PCA50 | coverage={has_lat.mean():.1%}')

    # [P] EEG coherence (idx 220-294): zero out — site-specific hardware noise.
    X_arr[:, 220:295] = 0.0

    # [A] Site one-hot (idx 10-12): zero out — prevents model from learning
    # site→label shortcuts that won't generalise to unseen validation sites.
    X_arr[:, 10:13] = 0.0

    # Site-normalise absolute EEG features to remove hardware/electrode bias:
    #   [C] EEG bandpower log1p (25-114): absolute PSD varies by amplifier/impedance
    #   [G] spindle amplitude/power (134-151): YASA absolute values
    #   [I] EEG complexity/Hjorth activity (155-166): variance-based, amplitude-dependent
    # Z-norm per site; unseen validation sites fall back to __global__ stats.
    _SNORM_IDX = np.array(
        list(range(25, 115)) +    # [C] EEG bandpower (90 feats)
        list(range(134, 152)) +   # [G] spindle amplitude (18 feats)
        list(range(155, 167)),    # [I] Hjorth + spectral entropy (12 feats)
        dtype=int
    )
    _site_norm = {}
    for _s in np.unique(site_arr):
        _m = (site_arr == _s)
        _mu = X_arr[_m][:, _SNORM_IDX].mean(axis=0)
        _sd = np.maximum(X_arr[_m][:, _SNORM_IDX].std(axis=0), 1e-6)
        _site_norm[_s] = (_mu, _sd)
        X_arr[np.ix_(_m, _SNORM_IDX)] = (X_arr[np.ix_(_m, _SNORM_IDX)] - _mu) / _sd
    _site_norm['__global__'] = (
        X_arr[:, _SNORM_IDX].mean(axis=0),
        np.maximum(X_arr[:, _SNORM_IDX].std(axis=0), 1e-6),
    )

    if verbose:
        print(f'[train] X={X_arr.shape} | pos={y_arr.sum()} neg={(y_arr==0).sum()} '
              f'| coherence+site zeroed | C/G/I site-normed')

    from sklearn.model_selection import StratifiedKFold, cross_val_score

    # ── Optuna: tune LightGBM + feature-selection threshold ──────────────────
    # IMPORTANT: feature selection is done INSIDE each fold to prevent leakage.
    _best_lgbm_params = {}
    _best_fs_pct      = 40
    if HAS_LGBM:
        try:
            import optuna
            from sklearn.metrics import roc_auc_score as _roc
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            _opt_strat = np.array([f"{s}_{l}" for s, l in zip(site_arr, y_arr)])

            # LOSO splits: hold out each minority site as a full test set.
            # Tests true cross-site generalisation — mirrors unseen validation sites.
            _unique_sites = np.unique(site_arr)
            _loso_splits = []
            for _hold in _unique_sites:
                _tr = np.where(site_arr != _hold)[0]
                _te = np.where(site_arr == _hold)[0]
                if len(_tr) >= 100 and len(_te) >= 20:
                    _loso_splits.append((_tr, _te))

            def _lgbm_objective(trial):
                p = dict(
                    n_estimators      = 500,
                    learning_rate     = 0.05,
                    num_leaves        = trial.suggest_int('num_leaves', 10, 50),
                    min_child_samples = trial.suggest_int('min_child_samples', 20, 100),
                    reg_lambda        = trial.suggest_float('reg_lambda', 1.0, 30.0, log=True),
                    reg_alpha         = trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),
                    colsample_bytree  = trial.suggest_float('colsample_bytree', 0.4, 0.8),
                    subsample         = trial.suggest_float('subsample', 0.5, 0.85),
                    subsample_freq    = 1,
                    random_state      = 42, n_jobs=1, verbose=-1,
                )
                fs_pct = trial.suggest_int('fs_pct', 30, 65)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                fold_aucs = []
                # Standard stratified folds
                for tr_idx, val_idx in skf.split(X_arr, _opt_strat):
                    X_tr, y_tr = X_arr[tr_idx], y_arr[tr_idx]
                    X_val, y_val = X_arr[val_idx], y_arr[val_idx]
                    fs_clf = lgb.LGBMClassifier(**p)
                    fs_clf.fit(X_tr, y_tr)
                    imp  = fs_clf.feature_importances_
                    mask = imp >= np.percentile(imp, fs_pct)
                    if mask.sum() < 10:
                        mask = imp >= np.percentile(imp, 30)
                    clf = lgb.LGBMClassifier(**p)
                    clf.fit(X_tr[:, mask], y_tr)
                    prob = clf.predict_proba(X_val[:, mask])[:, 1]
                    try:
                        fold_aucs.append(_roc(y_val, prob))
                    except Exception:
                        fold_aucs.append(0.5)
                # LOSO folds (weighted ×2 each — directly tests cross-site generalisation)
                for tr_idx, val_idx in _loso_splits:
                    X_tr, y_tr = X_arr[tr_idx], y_arr[tr_idx]
                    X_val, y_val = X_arr[val_idx], y_arr[val_idx]
                    fs_clf = lgb.LGBMClassifier(**p)
                    fs_clf.fit(X_tr, y_tr)
                    imp  = fs_clf.feature_importances_
                    mask = imp >= np.percentile(imp, fs_pct)
                    if mask.sum() < 10:
                        mask = imp >= np.percentile(imp, 30)
                    clf = lgb.LGBMClassifier(**p)
                    clf.fit(X_tr[:, mask], y_tr)
                    prob = clf.predict_proba(X_val[:, mask])[:, 1]
                    try:
                        loso_auc = _roc(y_val, prob)
                        fold_aucs.extend([loso_auc, loso_auc])  # weight ×2
                    except Exception:
                        fold_aucs.extend([0.5, 0.5])
                return float(np.mean(fold_aucs))

            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42),
            )
            study.optimize(_lgbm_objective, n_trials=50,
                           show_progress_bar=verbose)
            _best_lgbm_params = {k: v for k, v in study.best_params.items()
                                  if k != 'fs_pct'}
            _best_fs_pct = study.best_params.get('fs_pct', 40)
            _best_lgbm_params.update(dict(n_estimators=500, learning_rate=0.05,
                                          subsample_freq=1, random_state=42,
                                          n_jobs=-1, verbose=-1))
            if verbose:
                print(f'[Optuna] Best nested-CV AUROC={study.best_value:.4f} | '
                      f'fs_pct={_best_fs_pct} | params={study.best_params}')
        except Exception as e:
            if verbose:
                print(f'[Optuna] Skipped ({e})')

    # ── Model factory ─────────────────────────────────────────────────────────
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

    # ── XGBoost factory — regularisation aligned with Optuna-tuned LGBM ──────
    def _make_xgb():
        if not HAS_XGB:
            return None
        # Mirror LGBM regularisation level so both models generalise similarly
        _lp = _best_lgbm_params if _best_lgbm_params else _lgbm_defaults
        return xgb.XGBClassifier(
            n_estimators=500, max_depth=3, learning_rate=0.05,
            subsample=_lp.get('subsample', 0.8),
            colsample_bytree=_lp.get('colsample_bytree', 0.7),
            reg_alpha=float(_lp.get('reg_alpha', 2.0)),
            reg_lambda=float(_lp.get('reg_lambda', 10.0)),
            min_child_weight=max(1, int(_lp.get('min_child_samples', 30) // 5)),
            eval_metric='logloss', random_state=42,
            n_jobs=-1, verbosity=0,
        )

    # ── Feature selection on full data (for final model only) ────────────────
    if verbose:
        print('[FS] Fitting LGBM for feature importance (full data) ...')
    clf_fs = _make_lgbm()
    clf_fs.fit(X_arr, y_arr)
    importances = clf_fs.feature_importances_
    threshold   = np.percentile(importances, _best_fs_pct)
    feat_mask   = importances >= threshold
    X_sel       = X_arr[:, feat_mask]
    if verbose:
        print(f'[FS] Selected {feat_mask.sum()}/{len(feat_mask)} features (threshold={threshold:.4f})')

    # ── Nested CV + LOSO — honest AUROC + cross-site generalisation ─────────
    from sklearn.metrics import roc_auc_score
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    strat_labels = np.array([f"{s}_{l}" for s, l in zip(site_arr, y_arr)])
    aucs, loso_aucs = [], []
    if verbose:
        print('[CV] 5-fold nested-CV + LOSO (LGBM + XGB) ...')

    def _run_fold(tr_idx, val_idx):
        fs_f = _make_lgbm()
        fs_f.fit(X_arr[tr_idx], y_arr[tr_idx])
        imp_f = fs_f.feature_importances_
        fmask = imp_f >= np.percentile(imp_f, _best_fs_pct)
        if fmask.sum() < 10:
            fmask = imp_f >= np.percentile(imp_f, 30)
        lg = _make_lgbm(); lg.fit(X_arr[tr_idx][:, fmask], y_arr[tr_idx])
        pr = lg.predict_proba(X_arr[val_idx][:, fmask])[:, 1]
        xg = _make_xgb()
        if xg is not None:
            xg.fit(X_arr[tr_idx][:, fmask], y_arr[tr_idx])
            pr = (pr + xg.predict_proba(X_arr[val_idx][:, fmask])[:, 1]) / 2.0
        return roc_auc_score(y_arr[val_idx], pr), fmask.sum()

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_arr, strat_labels), 1):
        auc, nf = _run_fold(tr_idx, val_idx)
        aucs.append(auc)
        if verbose:
            print(f'  Fold {fold}: AUROC={auc:.4f} | n_feat={nf}')

    for _hold in np.unique(site_arr):
        _tr = np.where(site_arr != _hold)[0]
        _te = np.where(site_arr == _hold)[0]
        if len(_tr) < 100 or len(_te) < 20:
            continue
        auc, nf = _run_fold(_tr, _te)
        loso_aucs.append(auc)
        if verbose:
            print(f'  LOSO {_hold}: AUROC={auc:.4f} | n_feat={nf} | test_n={len(_te)}')

    if verbose:
        print(f'[CV] Nested-CV AUROC = {np.mean(aucs):.4f}  std = {np.std(aucs):.4f}')
        if loso_aucs:
            print(f'[CV] LOSO AUROC     = {np.mean(loso_aucs):.4f}  (cross-site estimate)')

    # ── Train final models on full data ───────────────────────────────────────
    if verbose:
        print('[train] Fitting final LGBM + XGB on full data ...')
    final_lgbm = _make_lgbm()
    final_lgbm.fit(X_sel, y_arr)
    final_xgb = _make_xgb()
    if final_xgb is not None:
        final_xgb.fit(X_sel, y_arr)

    joblib.dump({
        'stack':           final_lgbm,
        'xgb_model':       final_xgb,
        'feat_mask':       feat_mask,
        'feature_version': FEATURE_VERSION,
        'site_norm':       _site_norm,
        'site_norm_idx':   _SNORM_IDX,
        'ps_pca':          ps_pca,          # PCA fitted on PS latents (may be None)
    }, os.path.join(model_folder, MODEL_FILE))

    if verbose:
        print('[train] Done.')


def load_model(model_folder, verbose):
    m = joblib.load(os.path.join(model_folder, MODEL_FILE))
    # Pre-load PS model + caches once (saves ~30-60s per patient during inference)
    m['ps_model']   = _load_ps_model(verbose=verbose)
    m['ps_cache']   = _load_ps_baked(verbose=False)    # scalar baked cache
    m['ps_latents'] = _load_ps_latents(verbose=False)  # latent baked cache
    return m


def run_model(model, record, data_folder, verbose):
    lgbm_model = model['stack']
    xgb_model  = model.get('xgb_model')
    feat_mask  = model.get('feat_mask')

    X = extract_all_features(record, data_folder, DEFAULT_CSV, cache_dir=None)
    X = X.reshape(1, -1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Zero out features that were zeroed during training
    X[0, 220:295] = 0.0   # coherence
    X[0, 10:13]   = 0.0   # site one-hot

    # Append PS features: 4 scalars + PCA50 latents = 54 values
    ps_feat = _get_ps_features(record, data_folder,
                               ps_model=model.get('ps_model'),
                               ps_cache=model.get('ps_cache'),
                               ps_latents=model.get('ps_latents'),
                               ps_pca=model.get('ps_pca'))
    X = np.concatenate([X, ps_feat.reshape(1, -1)], axis=1)   # (1, 373)

    # Apply site-level normalisation to EEG amplitude features
    _snorm = model.get('site_norm')
    _sidx  = model.get('site_norm_idx')
    if _snorm is not None and _sidx is not None:
        site_id = record.get(HEADERS['site_id'], '__global__')
        _mu, _sd = _snorm.get(site_id, _snorm['__global__'])
        X[0, _sidx] = (X[0, _sidx] - _mu) / _sd

    if feat_mask is not None:
        X = X[:, feat_mask]

    prob = float(lgbm_model.predict_proba(X)[0][1])
    if xgb_model is not None:
        prob = (prob + float(xgb_model.predict_proba(X)[0][1])) / 2.0

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
    f_hrv    = feat_ecg_hrv(phys, fs)                  # [E]   6
    f_prob   = feat_caisr_probs(algo_data)              # [F]   5
    f_spin   = feat_custom_spindles(phys, fs, stages)   # [G]  18  NOVEL
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
    f_halves = feat_half_night_spectral(phys, fs, stages)      # [T]  12  NOVEL

    features = np.concatenate([
        f_demo, f_macro, f_eeg, f_resp, f_hrv, f_prob,
        f_spin, f_coup, f_cplx, f_frag, f_sef,
        f_wkurt, f_bkurt, f_n3rat,
        f_coh, f_remrat, f_sfar, f_halves,
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
        return np.zeros(6)
    try:
        nyq  = ekg_fs / 2.0
        b, a = sp_signal.butter(2, [max(5./nyq, 1e-3), min(40./nyq, 0.99)], btype='band')
        ecg_f   = sp_signal.filtfilt(b, a, ecg)
        peaks,_ = sp_signal.find_peaks(ecg_f,
                                        height=np.percentile(ecg_f, 90),
                                        distance=int(0.35 * ekg_fs))
        if len(peaks) < 10:
            return np.zeros(6)
        rr = np.diff(peaks) / ekg_fs * 1000.0
        rr = rr[(rr > 300) & (rr < 2000)]
        if len(rr) < 5:
            return np.zeros(6)
        mean_rr = np.mean(rr)
        time_feats = np.array([mean_rr, np.std(rr),
                                np.sqrt(np.mean(np.diff(rr)**2)),
                                60000. / mean_rr,
                                np.mean(np.abs(np.diff(rr)) > 50),
                                np.max(rr) - np.min(rr)])
        return time_feats
    except Exception:
        return np.zeros(6)


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
