"""
BCI Competition IV Dataset 2a — Preprocessing for ZUNA
Mirrors the DREAMER pipeline structure.

Per-trial pipeline:
  1. Baseline subtraction  (pre-cue period mean, per channel)
  2. Resample              250 Hz → 256 Hz
  3. High-pass filter      0.5 Hz, 4th-order Butterworth (applied to full trial)
  4. Notch filter          50 Hz  (confirmed SNR +7–26 dB across all subjects)
  5. Z-score normalisation per channel over full trial
  7. Slice into non-overlapping 5s windows (1280 samples @ 256 Hz)
     → [-1s, +4s] epoch is exactly 1280 samples = 1 window per trial
  8. Drop artifact-flagged trials

Output: preprocessed.pt (single pooled file, all subjects & sessions)
  X           (N, 22, 1280)  float32   - EEG windows
  y           (N,)           int64     - 0-indexed class (0=left, 1=right, 2=feet, 3=tongue)
  subject_ids (N,)           int64     - 1-indexed subject ID (1–9)
  trial_ids   (N,)           int64     - 1-indexed trial number within subject/session
  coords      (22, 3)        float64   - standard_1020 head-frame positions (metres)
  ch_names    list[str]      - 22 EEG channel names
"""

import numpy as np
import scipy.io as sio
import scipy.signal
import torch
from pathlib import Path
from math import gcd

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path('BCICIV_2a_mat')
OUTPUT_PATH = Path('preprocessed.pt')

ORIG_FS        = 250
TARGET_FS      = 256
WINDOW_SEC     = 5
WINDOW_SAMPLES = WINDOW_SEC * TARGET_FS   # 1280

PRE_S    = 1.0    # seconds before cue  (baseline + pre-cue)
POST_S   = 4.0    # seconds after  cue  (MI window)
N_EEG    = 22     # EEG channels to keep (EOG channels 23–25 dropped)

HP_FREQ    = 0.5   # Hz
NOTCH_FREQ = 50.0  # Hz
NOTCH_Q    = 30.0


# ── Electrode layout ───────────────────────────────────────────────────────────
CH_NAMES = [
    'Fz',  'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5',  'C3',  'C1',  'Cz',  'C2',  'C4',  'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1',  'Pz',  'P2',  'POz',
]

COORDS_3D = np.array([
    [ 0.00031,  0.05851,  0.06646],  # Fz
    [-0.06018,  0.02272,  0.05554],  # FC3
    [-0.03406,  0.02601,  0.07999],  # FC1
    [ 0.00038,  0.02739,  0.08867],  # FCz
    [ 0.03478,  0.02644,  0.07881],  # FC2
    [ 0.06229,  0.02372,  0.05563],  # FC4
    [-0.08028, -0.01376,  0.02916],  # C5
    [-0.06536, -0.01163,  0.06436],  # C3
    [-0.03616, -0.00998,  0.08975],  # C1
    [ 0.00040, -0.00917,  0.10024],  # Cz
    [ 0.03767, -0.00962,  0.08841],  # C2
    [ 0.06712, -0.01090,  0.06358],  # C4
    [ 0.08346, -0.01278,  0.02921],  # C6
    [-0.06356, -0.04701,  0.06562],  # CP3
    [-0.03551, -0.04729,  0.09131],  # CP1
    [ 0.00039, -0.04732,  0.09943],  # CPz
    [ 0.03838, -0.04707,  0.09069],  # CP2
    [ 0.06661, -0.04664,  0.06558],  # CP4
    [-0.02862, -0.08052,  0.07544],  # P1
    [ 0.00032, -0.08111,  0.08261],  # Pz
    [ 0.03192, -0.08049,  0.07672],  # P2
    [ 0.00022, -0.10218,  0.05061],  # POz
], dtype=np.float64)


# ── Signal processing helpers ──────────────────────────────────────────────────
def resample_signal(data: np.ndarray) -> np.ndarray:
    """Resample (samples, channels) from ORIG_FS to TARGET_FS."""
    up   = TARGET_FS // gcd(ORIG_FS, TARGET_FS)
    down = ORIG_FS   // gcd(ORIG_FS, TARGET_FS)
    return scipy.signal.resample_poly(data, up, down, axis=0)


def highpass_filter(data: np.ndarray) -> np.ndarray:
    """4th-order Butterworth high-pass at HP_FREQ, zero-phase."""
    nyq = TARGET_FS / 2
    b, a = scipy.signal.butter(4, HP_FREQ / nyq, btype='high')
    return scipy.signal.filtfilt(b, a, data, axis=0)


def notch_filter(data: np.ndarray, freq: float) -> np.ndarray:
    """IIR notch at freq Hz, zero-phase."""
    nyq = TARGET_FS / 2
    b, a = scipy.signal.iirnotch(freq / nyq, Q=NOTCH_Q)
    return scipy.signal.filtfilt(b, a, data, axis=0)


def zscore_normalize(data: np.ndarray) -> np.ndarray:
    """Z-score per channel over the full trial."""
    mean = data.mean(axis=0, keepdims=True)
    std  = data.std( axis=0, keepdims=True)
    std  = np.where(std < 1e-8, 1e-8, std)
    return (data - mean) / std


def slice_windows(data: np.ndarray) -> np.ndarray:
    """
    Slice (samples, channels) into non-overlapping 5s windows.
    Returns (n_windows, channels, WINDOW_SAMPLES).
    Trailing samples that don't fill a full window are dropped.
    """
    n_windows = len(data) // WINDOW_SAMPLES
    data = data[: n_windows * WINDOW_SAMPLES]
    return data.reshape(n_windows, WINDOW_SAMPLES, data.shape[1]).transpose(0, 2, 1)


# ── Per-trial pipeline ─────────────────────────────────────────────────────────
def process_trial(epoch: np.ndarray) -> np.ndarray:
    """
    Full pipeline for one cue-locked epoch.
    epoch shape: (1250 samples @ 250 Hz, 22 channels)
                 = [-1s, +4s] relative to cue onset

    Returns: (n_windows, 22, 1280)  — typically 1 window per trial
    """
    pre_samples_orig = int(PRE_S * ORIG_FS)   # 250 samples = 1s pre-cue

    # 1. Baseline subtraction: subtract mean of pre-cue period per channel
    baseline_mean = epoch[:pre_samples_orig].mean(axis=0, keepdims=True)
    data = epoch - baseline_mean

    # 2. Resample 250 → 256 Hz
    data = resample_signal(data)

    # 3. High-pass filter on full resampled trial
    data = highpass_filter(data)

    # 4. Notch filter (50 Hz)
    data = notch_filter(data, NOTCH_FREQ)

    # 5. Z-score per channel
    data = zscore_normalize(data)

    # 6. Slice into non-overlapping 5s windows
    windows = slice_windows(data)   # (n_windows, 22, 1280)

    return windows.astype(np.float32)


# ── Per-subject helpers ────────────────────────────────────────────────────────
def get_mi_runs(d):
    return [i for i in range(len(d))
            if hasattr(d[i].trial, '__len__') and len(d[i].trial) > 1]


def extract_epochs(run, pre_s=PRE_S, post_s=POST_S, fs=ORIG_FS, n_eeg=N_EEG):
    """Extract cue-locked epochs from a single run."""
    X    = run.X[:, :n_eeg].astype(np.float64)
    pre  = int(pre_s  * fs)
    post = int(post_s * fs)

    epochs, labels, artifacts = [], [], []
    for onset, label, art in zip(run.trial.ravel(), run.y.ravel(), run.artifacts.ravel()):
        t0, t1 = onset - pre, onset + post
        if t0 < 0 or t1 > X.shape[0]:
            continue
        epochs.append(X[t0:t1])        # (1250, 22)
        labels.append(int(label))
        artifacts.append(int(art))

    return epochs, labels, artifacts


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print('BCI IV 2a preprocessing (DREAMER-style)')
    print(f'  {ORIG_FS} Hz → {TARGET_FS} Hz  |  window {WINDOW_SEC}s = {WINDOW_SAMPLES} samples')
    print(f'  HP {HP_FREQ} Hz  |  Notch {NOTCH_FREQ} Hz  |  Z-score\n')

    X_list, y_list, subj_list, trial_list = [], [], [], []

    subjects = [f'A{i:02d}' for i in range(1, 10)]

    for subj in subjects:
        subj_id = int(subj[1:])
        trial_counter = 0
        n_kept = 0
        n_artifact = 0

        for split in ['T', 'E']:
            path = DATA_DIR / f'{subj}{split}.mat'
            if not path.exists():
                continue

            raw = sio.loadmat(str(path), struct_as_record=False, squeeze_me=True)
            d   = raw['data']

            for run_idx in get_mi_runs(d):
                run = d[run_idx]
                epochs, labels, artifacts = extract_epochs(run)

                for epoch, label, art in zip(epochs, labels, artifacts):
                    trial_counter += 1

                    if art == 1:
                        n_artifact += 1
                        continue

                    windows = process_trial(epoch)   # (n_windows, 22, 1280)
                    n_w = len(windows)
                    if n_w == 0:
                        continue

                    # 0-index the label (1–4 → 0–3)
                    y_cls = label - 1

                    X_list.append(windows)
                    y_list.extend([y_cls]    * n_w)
                    subj_list.extend([subj_id]    * n_w)
                    trial_list.extend([trial_counter] * n_w)
                    n_kept += n_w

        print(f'  {subj}: {trial_counter} trials  '
              f'| artifact rejected: {n_artifact}  '
              f'| windows kept: {n_kept}')

    X           = np.concatenate(X_list, axis=0)            # (N, 22, 1280)
    y           = np.array(y_list,     dtype=np.int64)
    subject_ids = np.array(subj_list,  dtype=np.int64)
    trial_ids   = np.array(trial_list, dtype=np.int64)

    print(f'\nSaving to {OUTPUT_PATH}')
    torch.save({
        'X':           torch.from_numpy(X),
        'y':           torch.from_numpy(y),
        'subject_ids': torch.from_numpy(subject_ids),
        'trial_ids':   torch.from_numpy(trial_ids),
        'coords':      torch.from_numpy(COORDS_3D),
        'ch_names':    CH_NAMES,
        'fs':          TARGET_FS,
        'pre_s':       PRE_S,
        'post_s':      POST_S,
    }, str(OUTPUT_PATH))

    print(f'\n── Summary ───────────────────────────────────────────────')
    print(f'  X shape:       {X.shape}  (windows, channels, samples)')
    print(f'  Total windows: {len(X)}')
    print(f'  Class counts:  {np.bincount(y).tolist()}  (0=left, 1=right, 2=feet, 3=tongue)')
    print(f'  Subjects:      {np.unique(subject_ids).tolist()}')
    print(f'  File size:     {OUTPUT_PATH.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
