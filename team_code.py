#!/usr/bin/env python
# team_code.py — PhysioNet Challenge 2026
# Features: demographics(10) + sleep_macro(12) + EEG_bandpower(90) +
#           resp_spo2(8) + ecg_hrv(6) + caisr_probs(5) = 131 total
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
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV  = os.path.join(SCRIPT_DIR, 'channel_table.csv')
CACHE_SUBDIR = 'feature_cache'
MODEL_FILE   = 'model.sav'

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

# Feature layout (total = 131)
# [0:10]    demographics
# [10:22]   sleep macrostructure
# [22:112]  EEG bandpower  (3 ch x 5 stages x 6 bands = 90)
# [112:120] resp/SpO2
# [120:126] ECG HRV
# [126:131] CAISR probabilities
N_FEATURES = 10 + 12 + N_EEG_CH * N_STAGES * N_BANDS + 8 + 6 + 5


# ─── Required Challenge Functions ─────────────────────────────────────────────

def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV):
    demo_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    records   = find_patients(demo_file)

    if verbose:
        print(f'[train] {len(records)} records found.')

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
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.05,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, subsample=0.8, random_state=42,
        )

    clf.fit(X_arr, y_arr)
    joblib.dump({'model': clf}, os.path.join(model_folder, MODEL_FILE))

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
    """Returns float32 array of shape (N_FEATURES,)."""
    pid = record[HEADERS['bids_folder']]
    sid = record[HEADERS['site_id']]
    ses = record[HEADERS['session_id']]

    # Cache check
    if cache_dir:
        cache_file = os.path.join(cache_dir, f'{pid}_ses-{ses}.npz')
        if os.path.exists(cache_file):
            return np.load(cache_file)['features']

    # Load demographics
    demo_file    = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_data = load_demographics(demo_file, pid, ses)

    # Load physiological signals
    phys_path = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                             sid, f'{pid}_ses-{ses}.edf')
    phys_raw, phys_fs_raw = (load_signal_data(phys_path)
                              if os.path.exists(phys_path) else ({}, {}))

    # Load CAISR algorithmic annotations
    algo_path = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                             sid, f'{pid}_ses-{ses}_caisr_annotations.edf')
    algo_data, _ = (load_signal_data(algo_path)
                    if os.path.exists(algo_path) else ({}, {}))

    # Standardize channel names + derive bipolar channels
    phys, fs = _build_channel_dict(phys_raw, phys_fs_raw, csv_path)

    # stage_caisr: fs=1/30 Hz -> one sample per 30-second epoch
    stages = np.asarray(algo_data.get('stage_caisr', []), dtype=float)

    # Extract feature groups
    f_demo  = feat_demographics(patient_data)          # 10
    f_macro = feat_sleep_macro(stages)                 # 12
    f_eeg   = feat_eeg_bandpower(phys, fs, stages)     # 90
    f_resp  = feat_resp_spo2(algo_data, phys)          #  8
    f_hrv   = feat_ecg_hrv(phys, fs)                  #  6
    f_prob  = feat_caisr_probs(algo_data)              #  5

    features = np.concatenate([f_demo, f_macro, f_eeg,
                                f_resp, f_hrv, f_prob]).astype(np.float32)

    if cache_dir:
        np.savez_compressed(cache_file, features=features)

    return features


def _build_channel_dict(raw, raw_fs, csv_path):
    """Rename channels using channel_table.csv, then derive missing bipolar channels."""
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

    # Derive bipolar channels for sites with raw electrodes (e.g. I0006)
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


# ─── Feature Groups ───────────────────────────────────────────────────────────

def feat_demographics(data):
    """10 features: age(1) + sex_onehot(3) + race_onehot(5) + bmi(1)."""
    age = np.array([load_age(data)], dtype=float)

    sex_vec = np.zeros(3)
    sex_vec[{'Female': 0, 'Male': 1}.get(load_sex(data), 2)] = 1.0

    race_vec = np.zeros(5)
    race_key = get_standardized_race(data).lower()
    race_vec[{'asian':0,'black':1,'others':2,'unavailable':3,'white':4}.get(race_key, 2)] = 1.0

    bmi = np.array([load_bmi(data)], dtype=float)
    return np.concatenate([age, sex_vec, race_vec, bmi])


def feat_sleep_macro(stages):
    """12 features derived from stage_caisr (one value per 30-s epoch)."""
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
    tib_hr   = tib_min / 60.0
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
                     rem_lat, trans, tst_hr, tib_hr])


def feat_eeg_bandpower(phys, fs_dict, stages):
    """90 features: 3 channels x 5 sleep stages x 6 frequency bands.
    Fixed channel slots ensure consistent feature positions across all sites."""
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
    """Welch PSD averaged per sleep stage. Returns (N_STAGES, N_BANDS)."""
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


def feat_resp_spo2(algo_data, phys):
    """8 features: AHI, arousal_idx, PLMI, mean_SpO2, min_SpO2,
                   ODI3/hr, ODI4/hr, frac_below_90."""
    out = np.zeros(8)

    # AHI -- resp_caisr at 1 Hz
    resp   = np.asarray(algo_data.get('resp_caisr', []), dtype=float)
    r_hr   = len(resp) / 3600.0
    if r_hr > 0:
        out[0] = np.count_nonzero(np.diff((resp > 0).astype(int), prepend=0) == 1) / r_hr

    # Arousal index -- arousal_caisr at 2 Hz
    arous  = np.asarray(algo_data.get('arousal_caisr', []), dtype=float)
    a_hr   = len(arous) / 2.0 / 3600.0
    if a_hr > 0:
        out[1] = np.count_nonzero(np.diff((arous > 0).astype(int), prepend=0) == 1) / a_hr

    # PLMI -- limb_caisr at 1 Hz
    limb   = np.asarray(algo_data.get('limb_caisr', []), dtype=float)
    l_hr   = len(limb) / 3600.0
    if l_hr > 0:
        out[2] = np.count_nonzero(np.diff((limb > 0).astype(int), prepend=0) == 1) / l_hr

    # SpO2
    spo2 = None
    for ch in ('sao2', 'spo2'):
        if ch in phys:
            spo2 = phys[ch].astype(float)
            break
    if spo2 is not None and len(spo2) > 0:
        valid  = spo2[(spo2 >= 50) & (spo2 <= 100)]
        if len(valid) > 0:
            m       = np.mean(valid)
            sph     = len(spo2) / 3600.0
            out[3]  = m
            out[4]  = np.min(valid)
            out[5]  = (np.count_nonzero(np.diff((valid < m-3).astype(int), prepend=0)==1)
                       / sph if sph > 0 else 0.0)
            out[6]  = (np.count_nonzero(np.diff((valid < m-4).astype(int), prepend=0)==1)
                       / sph if sph > 0 else 0.0)
            out[7]  = np.mean(valid < 90)

    return out


def feat_ecg_hrv(phys, fs_dict):
    """6 time-domain HRV features: mean_RR, SDNN, RMSSD, mean_HR, pNN50, NN_range."""
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
        lo   = max(5.0  / nyq, 1e-3)
        hi   = min(40.0 / nyq, 0.99)
        b, a = sp_signal.butter(2, [lo, hi], btype='band')
        ecg_f = sp_signal.filtfilt(b, a, ecg)

        dist     = int(0.35 * ekg_fs)
        thresh   = np.percentile(ecg_f, 90)
        peaks, _ = sp_signal.find_peaks(ecg_f, height=thresh, distance=dist)

        if len(peaks) < 10:
            return np.zeros(6)

        rr = np.diff(peaks) / ekg_fs * 1000.0
        rr = rr[(rr > 300) & (rr < 2000)]
        if len(rr) < 5:
            return np.zeros(6)

        mean_rr = np.mean(rr)
        return np.array([
            mean_rr,
            np.std(rr),
            np.sqrt(np.mean(np.diff(rr) ** 2)),
            60000.0 / mean_rr,
            np.mean(np.abs(np.diff(rr)) > 50),
            np.max(rr) - np.min(rr),
        ])
    except Exception:
        return np.zeros(6)


def feat_caisr_probs(algo_data):
    """5 features: mean valid (non-9) CAISR stage probabilities [w, n3, n2, n1, r]."""
    out  = np.zeros(5)
    keys = ['caisr_prob_w', 'caisr_prob_n3', 'caisr_prob_n2',
            'caisr_prob_n1', 'caisr_prob_r']
    for i, k in enumerate(keys):
        arr   = np.asarray(algo_data.get(k, []), dtype=float)
        valid = arr[arr < 9.0]
        if len(valid) > 0:
            out[i] = np.clip(np.mean(valid), 0.0, 1.0)
    return out