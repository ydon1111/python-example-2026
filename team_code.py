#!/usr/bin/env python
# team_code.py — PhysioNet Challenge 2026
#
# Feature groups (total = 162):
#   [A]  demographics          10   age, sex×3, race×5, BMI
#   [B]  sleep macrostructure  12   SL, TST, SE, WASO, stage%, REMlat, trans, TIB
#   [C]  EEG bandpower         90   3ch × 5stages × 6bands (delta–gamma)
#   [D]  resp / SpO2            8   AHI, arousal_idx, PLMI, SpO2 stats, ODI
#   [E]  ECG HRV                6   mean_RR, SDNN, RMSSD, HR, pNN50, NN_range
#   [F]  CAISR probabilities    5   prob_w/n3/n2/n1/r
#   [G]  sleep spindle         8   sigma power, fast/slow spindle, density,     *** NOVEL ***
#                                   sigma/delta ratio, lateralization, sigma/theta
#   [H]  SO-spindle coupling    3   delta-sigma PAC (MVL) per EEG channel        *** NOVEL ***
#   [I]  EEG complexity        12   Hjorth (act/mob/cmp) + spectral entropy      *** NOVEL ***
#                                   for N2, N3, REM (C3-M2)
#   [J]  sleep fragmentation    5   stage entropy, wake bouts, NREM frag,        *** NOVEL ***
#                                   stage stability
#   [K]  spectral edge freq     3   SEF90 for N2, N3, REM (C3-M2)               *** NOVEL ***
#
# Model: LightGBM (fallback: GradientBoostingClassifier)

import joblib
import numpy as np
import os
from scipy import signal as sp_signal
from tqdm import tqdm

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

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
FEATURE_VERSION = 'v2'   # bump when feature layout changes to invalidate old cache

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

N_FEATURES = 10 + 12 + N_EEG_CH*N_STAGES*N_BANDS + 8 + 6 + 5 + 8 + 3 + 12 + 5 + 3  # 162


# ─── Required Challenge Functions ─────────────────────────────────────────────

def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV):
    demo_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    records   = find_patients(demo_file)

    if verbose:
        print(f'[train] {len(records)} records | feature_version={FEATURE_VERSION}')

    os.makedirs(model_folder, exist_ok=True)
    cache_dir = os.path.join(model_folder, CACHE_SUBDIR)
    os.makedirs(cache_dir, exist_ok=True)

    features, labels = [], []
    pbar = tqdm(records, desc='Extracting', disable=not verbose)
    for rec in pbar:
        pid = rec[HEADERS['bids_folder']]
        try:
            X     = extract_all_features(rec, data_folder, csv_path, cache_dir)
            label = load_diagnoses(demo_file, pid)
            features.append(X)
            labels.append(label)
        except Exception as e:
            tqdm.write(f'  ! {pid}: {e}')

    X_arr = np.array(features, dtype=np.float32)
    y_arr = np.array(labels,   dtype=int)

    if verbose:
        print(f'[train] X={X_arr.shape} | pos={y_arr.sum()} neg={(y_arr==0).sum()}')

    if HAS_LGBM:
        clf = lgb.LGBMClassifier(
            n_estimators=500,
            num_leaves=47,
            learning_rate=0.03,
            min_child_samples=15,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            n_estimators=300, max_depth=4,
            learning_rate=0.03, subsample=0.8, random_state=42,
        )

    # ── 5-fold stratified cross-validation ───────────────────────────────────
    if verbose:
        print('[CV] Running 5-fold stratified CV ...')
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs   = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_arr, y_arr), 1):
        if HAS_LGBM:
            cv_clf = lgb.LGBMClassifier(
                n_estimators=500, num_leaves=47, learning_rate=0.03,
                min_child_samples=15, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
                class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1,
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            cv_clf = GradientBoostingClassifier(
                n_estimators=300, max_depth=4,
                learning_rate=0.03, subsample=0.8, random_state=42,
            )
        cv_clf.fit(X_arr[tr_idx], y_arr[tr_idx])
        prob   = cv_clf.predict_proba(X_arr[val_idx])[:, 1]
        auc    = roc_auc_score(y_arr[val_idx], prob)
        aucs.append(auc)
        if verbose:
            print(f'  Fold {fold}: AUROC={auc:.4f}')
    if verbose:
        print(f'[CV] Mean AUROC = {np.mean(aucs):.4f}  std = {np.std(aucs):.4f}')
    # ─────────────────────────────────────────────────────────────────────────

    clf.fit(X_arr, y_arr)
    joblib.dump({'model': clf, 'feature_version': FEATURE_VERSION},
                os.path.join(model_folder, MODEL_FILE))

    if verbose:
        print('[train] Done.')


def load_model(model_folder, verbose):
    return joblib.load(os.path.join(model_folder, MODEL_FILE))


def run_model(model, record, data_folder, verbose):
    clf    = model['model']
    X      = extract_all_features(record, data_folder, DEFAULT_CSV, cache_dir=None)
    X      = X.reshape(1, -1).astype(np.float32)
    binary = bool(clf.predict(X)[0])
    prob   = float(clf.predict_proba(X)[0][1])
    return binary, prob


# ─── Feature Extraction ───────────────────────────────────────────────────────

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
    f_demo   = feat_demographics(patient_data)          # [A]  10
    f_macro  = feat_sleep_macro(stages)                 # [B]  12
    f_eeg    = feat_eeg_bandpower(phys, fs, stages)     # [C]  90
    f_resp   = feat_resp_spo2(algo_data, phys)          # [D]   8
    f_hrv    = feat_ecg_hrv(phys, fs)                  # [E]   6
    f_prob   = feat_caisr_probs(algo_data)              # [F]   5
    f_spin   = feat_spindle(phys, fs, stages)           # [G]   8  NOVEL
    f_coup   = feat_so_spindle_coupling(phys, fs, stages)  # [H] 3  NOVEL
    f_cplx   = feat_eeg_complexity(phys, fs, stages)   # [I]  12  NOVEL
    f_frag   = feat_sleep_fragmentation(stages)         # [J]   5  NOVEL
    f_sef    = feat_spectral_edge(phys, fs, stages)     # [K]   3  NOVEL

    features = np.concatenate([
        f_demo, f_macro, f_eeg, f_resp, f_hrv, f_prob,
        f_spin, f_coup, f_cplx, f_frag, f_sef,
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
    """10 features: age(1) + sex×3 + race×5 + BMI(1)."""
    age = np.array([load_age(data)], dtype=float)

    sex_vec = np.zeros(3)
    sex_vec[{'Female': 0, 'Male': 1}.get(load_sex(data), 2)] = 1.0

    race_vec = np.zeros(5)
    race_key = get_standardized_race(data).lower()
    race_vec[{'asian':0,'black':1,'others':2,'unavailable':3,'white':4}.get(race_key, 2)] = 1.0

    bmi = np.array([load_bmi(data)], dtype=float)
    return np.concatenate([age, sex_vec, race_vec, bmi])


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
    """Returns (N_STAGES, N_BANDS) mean Welch PSD per stage."""
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
    """6 time-domain HRV features: mean_RR, SDNN, RMSSD, HR, pNN50, NN_range."""
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
        return np.array([mean_rr, np.std(rr),
                         np.sqrt(np.mean(np.diff(rr)**2)),
                         60000. / mean_rr,
                         np.mean(np.abs(np.diff(rr)) > 50),
                         np.max(rr) - np.min(rr)])
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


# ─── [G] Sleep spindle features ★ NOVEL ──────────────────────────────────────
#
# Motivation: Sleep spindles (12-16 Hz transient oscillations during N2) are
# generated by thalamo-cortical circuits.  Tau accumulation in the thalamus
# (early Alzheimer's) degrades spindle density and shifts fast-to-slow ratio.
# References: Helfrich et al. Nature Comm 2018; Winer et al. Nature Comm 2019.

def feat_spindle(phys, fs_dict, stages):
    """8 spindle features computed from N2 EEG epochs.

    [0] sigma_power_n2       — mean N2 sigma (12-16 Hz) band power  (C3-M2)
    [1] slow_spindle_power   — slow spindle (12-14 Hz) power        (C3-M2)
    [2] fast_spindle_power   — fast spindle (14-16 Hz) power        (C3-M2)
    [3] fast_slow_ratio      — fast / slow spindle power ratio
    [4] spindle_density      — spindle peaks per minute in N2       (C3-M2)
    [5] sigma_delta_ratio    — sigma / delta power in N2            (C3-M2)
    [6] sigma_theta_ratio    — sigma / theta power in N2            (C3-M2)
    [7] spindle_laterality   — (C3 sigma - C4 sigma) / (C3 + C4)
    """
    out = np.zeros(8)
    if len(stages) == 0:
        return out

    ch  = 'c3-m2'
    ch2 = 'c4-m1'
    if ch not in phys:
        return out

    sig    = phys[ch]
    eeg_fs = fs_dict.get(ch, 200.0)
    epochs = _get_n2_epochs(sig, eeg_fs, stages)
    if not epochs:
        return out

    nperseg   = min(int(eeg_fs * 4), int(EPOCH_SEC * eeg_fs))
    sigma_p   = []   # 12-16 Hz
    slow_p    = []   # 12-14 Hz
    fast_p    = []   # 14-16 Hz
    delta_p   = []   # 0.5-4 Hz
    theta_p   = []   # 4-8 Hz
    densities = []

    sigma_filt_cache = _bandpass(sig, eeg_fs, 12.0, 16.0)
    epoch_samp = int(EPOCH_SEC * eeg_fs)

    for i, sv in enumerate(stages.astype(int)):
        if sv != S_N2:
            continue
        start = i * epoch_samp
        end   = start + epoch_samp
        if end > len(sig):
            break
        epoch = sig[start:end]

        f, psd = sp_signal.welch(epoch, fs=eeg_fs, nperseg=nperseg)
        def bp(lo, hi):
            m = (f >= lo) & (f <= hi)
            return np.trapz(psd[m], f[m]) if m.any() else 0.0

        sigma_p.append(bp(12, 16))
        slow_p.append(bp(12, 14))
        fast_p.append(bp(14, 16))
        delta_p.append(bp(0.5, 4))
        theta_p.append(bp(4, 8))

        # Spindle density: peaks in sigma envelope > 1 std above mean
        env = np.abs(sp_signal.hilbert(sigma_filt_cache[start:end]))
        thr = np.mean(env) + np.std(env)
        pks, _ = sp_signal.find_peaks(env, height=thr,
                                       distance=int(0.5 * eeg_fs))  # min 0.5s apart
        densities.append(len(pks) / (EPOCH_SEC / 60.0))  # peaks per minute

    if not sigma_p:
        return out

    s_mean  = np.mean(sigma_p)
    sl_mean = np.mean(slow_p)
    sf_mean = np.mean(fast_p)
    d_mean  = np.mean(delta_p)
    t_mean  = np.mean(theta_p)

    out[0] = s_mean
    out[1] = sl_mean
    out[2] = sf_mean
    out[3] = sf_mean / sl_mean if sl_mean > 0 else 0.0
    out[4] = np.mean(densities)
    out[5] = s_mean / d_mean   if d_mean  > 0 else 0.0
    out[6] = s_mean / t_mean   if t_mean  > 0 else 0.0

    # Laterality index (C3 vs C4 sigma in N2)
    if ch2 in phys:
        sig2   = phys[ch2]
        ep2    = _get_n2_epochs(sig2, fs_dict.get(ch2, 200.0), stages)
        if ep2:
            def _sigma_power(e):
                f2, p2 = sp_signal.welch(e, fs=eeg_fs, nperseg=nperseg)
                m2 = (f2 >= 12) & (f2 <= 16)
                return np.trapz(p2[m2], f2[m2]) if m2.any() else 0.0
            s2 = [_sigma_power(e) for e in ep2]
            c4_s = np.mean(s2)
            denom = s_mean + c4_s
            out[7] = (s_mean - c4_s) / denom if denom > 0 else 0.0

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
