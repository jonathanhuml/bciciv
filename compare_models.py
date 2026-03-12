#!/usr/bin/env python3
"""
compare_models.py – BCI Competition IV Dataset 2a motor imagery benchmark
==========================================================================

Three methods compared on 4-class motor imagery prediction
(labels 0–3: left hand, right hand, feet, tongue) using
Stratified 5-Fold cross-validation on pooled EEG windows
(no leave-one-subject-out):

  1. Welch PSD → PCA → Multinomial Logistic Regression   (sklearn baseline)
  2. LUNA-Base (frozen backbone) → Multinomial Logistic Regression
  3. ZUNAenc (frozen encoder output) → Multinomial Logistic Regression

All three methods use the same LogisticRegression classifier
(solver='lbfgs', multinomial softmax, class_weight='balanced').

Usage
-----
    python compare_models.py                        # all methods
    python compare_models.py --skip-luna            # skip LUNA
    python compare_models.py --skip-zuna            # skip ZUNA
    python compare_models.py --no-vis               # skip latent-space plots

Requirements:
    torch>=2.5  scipy  numpy  scikit-learn  matplotlib  mne
    braindecode  huggingface-hub  safetensors  zuna
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_PATH = Path("preprocessed.pt")
OUT_DIR   = Path("results")

SFREQ     = 256
N_CHANS   = 22
N_TIMES   = 1280    # 5 s × 256 Hz
N_CLASSES = 4       # 0=left hand, 1=right hand, 2=feet, 3=tongue

BCI_CH_NAMES = [
    "Fz",  "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1",  "Pz",  "P2",  "POz",
]

# PSD baseline
PSD_FMIN    = 1.0
PSD_FMAX    = 45.0
PSD_NPERSEG = 256
PSD_N_PCA   = 50

# LUNA (HuggingFace)
LUNA_REPO = "thorir/LUNA"
LUNA_FILE = "LUNA_base.safetensors"

# ZUNA tokenisation
ZUNA_N_FINE   = 32
ZUNA_N_COARSE = N_TIMES // ZUNA_N_FINE     # 40 coarse-time tokens per channel
ZUNA_SEQ_LEN  = N_CHANS * ZUNA_N_COARSE   # 880 tokens per window (22 × 40)
ZUNA_DATA_NORM = 1.0                        # data already z-scored
ZUNA_XYZ_EXTREMES = torch.tensor([[-0.12, -0.12, -0.12], [0.12, 0.12, 0.12]])
ZUNA_NUM_BINS  = 50

# ZUNA (HuggingFace)
ZUNA_REPO    = "Zyphra/ZUNA"
ZUNA_WEIGHTS = "model-00001-of-00001.safetensors"
ZUNA_CONFIG  = "config.json"

# Classifier (shared across all methods)
LOGREG_C    = 1.0
LOGREG_ITER = 1000

# Evaluation
N_FOLDS          = 5
BATCH_SIZE        = 128
ZUNA_BATCH_SIZE   = 4     # 4 × 880 = 3520 packed tokens
SEED              = 42

# Visualisation
TSNE_MAX_SAMPLES = 3000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Data ───────────────────────────────────────────────────────────────────────

def load_data(path: Path) -> dict[str, torch.Tensor]:
    print(f"Loading {path} …")
    data        = torch.load(str(path), weights_only=False)
    X           = data["X"]            # (N, 22, 1280) float32
    y           = data["y"]            # (N,) int64  values 0–3
    subject_ids = data["subject_ids"]  # (N,) int64
    trial_ids   = data["trial_ids"]    # (N,) int64
    print(f"  Windows  : {X.shape}  |  subjects : {subject_ids.unique().numel()}")
    print(f"  Class dist (0=L, 1=R, 2=F, 3=T): {y.bincount().tolist()}")
    return {"X": X, "y": y, "subject_ids": subject_ids, "trial_ids": trial_ids}


# ── Channel positions ──────────────────────────────────────────────────────────

def make_bciciv_chs_info() -> list[dict]:
    """MNE-style chs list for LUNA (uses standard_1020 montage)."""
    import mne
    montage = mne.channels.make_standard_montage("standard_1020")
    info    = mne.create_info(ch_names=BCI_CH_NAMES, sfreq=SFREQ, ch_types="eeg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info.set_montage(montage, match_case=False)
    return info["chs"]


def make_bciciv_chan_pos() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        chan_pos          (22, 3)  float32 – xyz in metres (standard_1020)
        chan_pos_discrete (22, 3)  int64   – discretised for ZUNA
    """
    import mne
    from apps.AY2latent_bci.eeg_data import discretize_chan_pos

    montage = mne.channels.make_standard_montage("standard_1020")
    info    = mne.create_info(ch_names=BCI_CH_NAMES, sfreq=SFREQ, ch_types="eeg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info.set_montage(montage, match_case=False)

    pos  = torch.tensor(
        np.array([ch["loc"][:3] for ch in info["chs"]], dtype=np.float32)
    )  # (22, 3)
    disc = discretize_chan_pos(pos, ZUNA_XYZ_EXTREMES, ZUNA_NUM_BINS)  # (22, 3)
    return pos, disc


# ── PSD baseline ───────────────────────────────────────────────────────────────

def welch_psd_features(X: np.ndarray) -> np.ndarray:
    """
    Welch PSD per channel, log-transformed, flattened.

    Parameters
    ----------
    X : (N, 22, 1280)  float32

    Returns
    -------
    features : (N, 22 * n_freq_bins)  float64
    """
    freqs, psd = welch(X, fs=SFREQ, nperseg=PSD_NPERSEG, axis=-1)
    band = (freqs >= PSD_FMIN) & (freqs <= PSD_FMAX)
    psd  = np.log(psd[:, :, band] + 1e-12)   # (N, 22, n_freqs)
    return psd.reshape(psd.shape[0], -1)       # (N, 22*n_freqs)


# ── LUNA ───────────────────────────────────────────────────────────────────────

class _FeatureHook:
    """
    Captures the pooled representation from LUNA's final_layer by hooking
    the last nn.Linear (the pre-head projection).

    Output shape: (batch, D_luna)  where D_luna ≈ 256 for LUNA-Base.
    """
    def __init__(self, model: nn.Module):
        self.features: torch.Tensor | None = None

        target      = getattr(model, "final_layer", None)
        search_root = target if target is not None else model

        last_linear = None
        for m in search_root.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        assert last_linear is not None, "No nn.Linear found for feature hook"
        self._handle = last_linear.register_forward_pre_hook(self._hook)

    def _hook(self, module: nn.Module, args: tuple) -> None:
        self.features = args[0].detach().cpu()

    def remove(self) -> None:
        self._handle.remove()


def load_luna(chs_info: list[dict]) -> nn.Module:
    from braindecode.models import LUNA
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    print("  Loading LUNA-Base weights from HuggingFace …")
    model = LUNA(
        n_chans=N_CHANS,
        n_outputs=N_CLASSES,
        n_times=N_TIMES,
        sfreq=SFREQ,
        chs_info=chs_info,
    )
    path = hf_hub_download(repo_id=LUNA_REPO, filename=LUNA_FILE)
    sd   = load_file(path)

    mapping = model.mapping.copy()
    mapping["cross_attn.temparature"] = "cross_attn.temperature"
    mapped_sd = {mapping.get(k, k): v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(mapped_sd, strict=False)
    print(f"  LUNA weights — missing: {len(missing)}, unexpected: {len(unexpected)}")
    if missing:
        print(f"    missing keys: {missing[:5]}{'…' if len(missing) > 5 else ''}")

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model.to(DEVICE)


@torch.no_grad()
def extract_luna_features(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """
    Forward all windows through frozen LUNA; return pooled pre-head features.

    Shape: (N, D_luna)  where D_luna ≈ 256 for LUNA-Base.
    """
    hook   = _FeatureHook(model)
    loader = DataLoader(
        torch.utils.data.TensorDataset(X),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    all_feats     = []
    printed_shape = False
    for (batch,) in loader:
        model(batch.to(DEVICE))
        all_feats.append(hook.features)
        if not printed_shape and hook.features is not None:
            print(f"  LUNA encoder output: {hook.features.shape}"
                  f"  (batch, D_luna={hook.features.shape[-1]})")
            printed_shape = True
    hook.remove()
    return torch.cat(all_feats, dim=0).numpy()


# ── ZUNA ───────────────────────────────────────────────────────────────────────

def load_zuna() -> tuple[nn.Module, nn.Module, int]:
    """Returns (encoder, enc_dec, latent_dim). Both are frozen, eval, on DEVICE."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file as safe_load
    from lingua.args import dataclass_from_dict
    from apps.AY2latent_bci.transformer import DecoderTransformerArgs, EncoderDecoder

    print("  Loading ZUNA weights from HuggingFace …")
    cfg_path = hf_hub_download(repo_id=ZUNA_REPO, filename=ZUNA_CONFIG)
    with open(cfg_path) as f:
        cfg = json.load(f)
    model_args: DecoderTransformerArgs = dataclass_from_dict(
        DecoderTransformerArgs, cfg["model"]
    )

    weights_path = hf_hub_download(repo_id=ZUNA_REPO, filename=ZUNA_WEIGHTS)
    sd_raw = safe_load(weights_path, device="cpu")
    sd     = {k.removeprefix("model."): v for k, v in sd_raw.items()}

    enc_dec = EncoderDecoder(model_args)
    enc_dec.load_state_dict(sd, strict=True)

    encoder    = enc_dec.encoder
    latent_dim = model_args.encoder_output_dim

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_dec = sum(p.numel() for p in enc_dec.parameters()) - n_enc
    print(f"  ZUNA encoder — latent_dim={latent_dim}, params={n_enc:,}")
    print(f"  ZUNA decoder — params={n_dec:,}")
    print(f"  ZUNA encoder raw output: (1, B×{ZUNA_SEQ_LEN}, {latent_dim})")
    print(f"  After time-pool per channel: (B, {N_CHANS}, {ZUNA_N_COARSE}, {latent_dim})"
          f" → mean(dim=2) → (B, {N_CHANS}, {latent_dim})"
          f" → flatten → (B, {N_CHANS * latent_dim})")

    for p in enc_dec.parameters():
        p.requires_grad_(False)
    enc_dec.eval()

    _explore_zuna_shapes(enc_dec, latent_dim)

    return encoder.to(DEVICE), enc_dec.to(DEVICE), latent_dim


def _explore_zuna_shapes(enc_dec: nn.Module, latent_dim: int) -> None:
    """
    Run a single dummy window through the ZUNA encoder to verify shapes.
    Prints encoder output shape and catches any forward-pass errors early.
    """
    from apps.AY2latent_bci.eeg_data import chop_and_reshape_signals

    print("\n  [ZUNA shape check] Running dummy forward pass …")
    try:
        dummy_eeg  = torch.zeros(N_CHANS, N_TIMES, dtype=torch.float32)
        dummy_pos  = torch.zeros(N_CHANS, 3, dtype=torch.float32)
        dummy_disc = torch.zeros(N_CHANS, 3, dtype=torch.long)

        eeg_r, _, cpd_r, _, tc_r, seq_len = chop_and_reshape_signals(
            eeg_signal        = dummy_eeg,
            chan_pos          = dummy_pos,
            chan_pos_discrete = dummy_disc,
            chan_dropout      = [],
            tf                = ZUNA_N_FINE,
            use_coarse_time   = "B",
        )
        packed_tokens = eeg_r.unsqueeze(0)                              # (1, SEQ_LEN, N_FINE)
        tok_idx       = torch.cat([cpd_r, tc_r], dim=1).unsqueeze(0)   # (1, SEQ_LEN, 4)
        seq_lens      = torch.tensor([int(seq_len)], dtype=torch.long)

        print(f"  Dummy packed_tokens: {packed_tokens.shape}"
              f"  (1, ZUNA_SEQ_LEN={ZUNA_SEQ_LEN}, ZUNA_N_FINE={ZUNA_N_FINE})")

        with torch.no_grad():
            enc_out, _ = enc_dec.encoder(
                token_values=packed_tokens,
                seq_lens=seq_lens,
                tok_idx=tok_idx,
                attn_impl="flex_attention",
            )
            print(f"  Encoder output shape: {enc_out.shape}"
                  f"  (1, B×{ZUNA_SEQ_LEN}, latent_dim={latent_dim})")
    except Exception as e:
        print(f"  Shape check failed: {e}")
    print()


class _BCICIVZUNADataset(Dataset):
    """Converts preprocessed (C, T) windows to ZUNA's packed-token format."""

    def __init__(
        self,
        X: torch.Tensor,
        chan_pos: torch.Tensor,
        chan_pos_discrete: torch.Tensor,
    ):
        self.X         = X
        self.chan_pos  = chan_pos
        self.chan_disc = chan_pos_discrete

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        from apps.AY2latent_bci.eeg_data import chop_and_reshape_signals

        x = self.X[idx].float() / ZUNA_DATA_NORM   # (22, 1280)
        eeg_r, _, cpd_r, _, tc_r, seq_len = chop_and_reshape_signals(
            eeg_signal        = x,
            chan_pos          = self.chan_pos,
            chan_pos_discrete = self.chan_disc,
            chan_dropout      = [],
            tf                = ZUNA_N_FINE,
            use_coarse_time   = "B",
        )
        tok_idx = torch.cat([cpd_r, tc_r], dim=1)   # (SEQ_LEN, 4)
        return eeg_r, tok_idx, int(seq_len)


def _zuna_collate(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens_list, tok_idx_list, seq_lens = zip(*batch)
    packed_tokens  = torch.stack(tokens_list).reshape(1, -1, ZUNA_N_FINE)
    packed_tok_idx = torch.stack(tok_idx_list).reshape(1, -1, 4)
    seq_lens_t     = torch.tensor(seq_lens, dtype=torch.long)
    return packed_tokens, seq_lens_t, packed_tok_idx


@torch.no_grad()
def extract_zuna_features(
    encoder: nn.Module,
    X: torch.Tensor,
    chan_pos: torch.Tensor,
    chan_pos_disc: torch.Tensor,
) -> np.ndarray:
    """
    Forward all windows through frozen ZUNA encoder.

    Pooling: mean over ZUNA_N_COARSE (=40) time tokens per channel.
    Channels kept separate (not globally pooled).

    Shape pipeline:
      encoder raw  : (1, B×880, 32)
      reshape      : (B, 22, 40, 32)
      mean(time)   : (B, 22, 32)
      flatten      : (B, 704)

    Returns
    -------
    features : (N, N_CHANS × latent_dim)  i.e. (N, 704)
    """
    dataset = _BCICIVZUNADataset(X, chan_pos, chan_pos_disc)
    loader  = DataLoader(
        dataset, batch_size=ZUNA_BATCH_SIZE,
        shuffle=False, collate_fn=_zuna_collate,
    )

    all_feats     = []
    printed_shape = False
    for packed_tokens, seq_lens, tok_idx in loader:
        B = seq_lens.shape[0]
        packed_tokens = packed_tokens.to(DEVICE)
        seq_lens      = seq_lens.to(DEVICE)
        tok_idx       = tok_idx.to(DEVICE)

        enc_out, _ = encoder(
            token_values=packed_tokens,
            seq_lens=seq_lens,
            tok_idx=tok_idx,
            attn_impl="flex_attention",
        )

        if not printed_shape:
            print(f"  ZUNA encoder raw output: {enc_out.shape}"
                  f"  (1, B×{ZUNA_SEQ_LEN}, latent_dim={enc_out.shape[-1]})")

        latent_dim = enc_out.shape[-1]
        latent = (
            enc_out.squeeze(0)
                   .reshape(B, N_CHANS, ZUNA_N_COARSE, latent_dim)
                   .mean(dim=2)
                   .reshape(B, N_CHANS * latent_dim)
                   .cpu()
        )

        if not printed_shape:
            print(f"  After time-pool + flatten: {latent.shape}"
                  f"  (B, {N_CHANS}×{latent_dim}={N_CHANS * latent_dim})")
            printed_shape = True

        all_feats.append(latent)

    return torch.cat(all_feats, dim=0).numpy()


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_windows(
    features: np.ndarray,
    labels: np.ndarray,
    use_pca: bool = False,
) -> dict[str, float]:
    """
    Stratified 5-fold CV on pooled windows (no leave-one-subject-out).
    Classifier: multinomial LogReg, class_weight='balanced'.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    bal_accs, accs = [], []

    for train_idx, test_idx in skf.split(features, labels):
        X_tr, y_tr = features[train_idx], labels[train_idx]
        X_te, y_te = features[test_idx],  labels[test_idx]

        steps = [("scale", StandardScaler())]
        if use_pca:
            n_comp = min(PSD_N_PCA, X_tr.shape[1], X_tr.shape[0] - 1)
            steps.append(("pca", PCA(n_components=n_comp, random_state=SEED)))
        steps.append((
            "clf",
            LogisticRegression(
                C=LOGREG_C,
                max_iter=LOGREG_ITER,
                solver="lbfgs",
                random_state=SEED,
                class_weight="balanced",
            ),
        ))

        pipe = Pipeline(steps)
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)

        bal_accs.append(balanced_accuracy_score(y_te, preds))
        accs.append(accuracy_score(y_te, preds))

    return {
        "balanced_acc_mean": float(np.mean(bal_accs)),
        "balanced_acc_std":  float(np.std(bal_accs)),
        "accuracy_mean":     float(np.mean(accs)),
        "accuracy_std":      float(np.std(accs)),
        "per_fold_bal_acc":  bal_accs,
        "per_fold_acc":      accs,
    }


# ── Visualisation ──────────────────────────────────────────────────────────────

def _subsample(
    feats: np.ndarray,
    *arrays: np.ndarray,
    max_n: int = TSNE_MAX_SAMPLES,
) -> tuple[np.ndarray, ...]:
    if len(feats) <= max_n:
        return (feats,) + arrays
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(feats), size=max_n, replace=False)
    return (feats[idx],) + tuple(a[idx] for a in arrays)


def plot_latent_spaces(
    luna_feats:     np.ndarray | None,
    zuna_enc_feats: np.ndarray | None,
    y:              np.ndarray,
    subject_ids:    np.ndarray,
    out_dir:        Path,
) -> None:
    """
    For each model: PCA(2) and t-SNE(2) coloured by MI class and subject.
    Saves <model>_pca.png and <model>_tsne.png.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = ["Left", "Right", "Feet", "Tongue"]

    model_feats: dict[str, np.ndarray] = {}
    if luna_feats is not None:
        model_feats["LUNA"] = luna_feats
    if zuna_enc_feats is not None:
        model_feats["ZUNAenc"] = zuna_enc_feats

    for model_name, feats in model_feats.items():
        for method_name, Reducer, kw in [
            ("pca",  PCA,  {"n_components": 2, "random_state": SEED}),
            ("tsne", TSNE, {"n_components": 2, "random_state": SEED,
                            "perplexity": 30, "n_jobs": -1}),
        ]:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(
                f"{model_name} latent space — {method_name.upper()}",
                fontsize=13, fontweight="bold",
            )

            sub_feats, sub_y, sub_sid = _subsample(feats, y, subject_ids)

            scaler  = StandardScaler()
            scaled  = scaler.fit_transform(sub_feats)
            reducer = Reducer(**kw)
            emb     = reducer.fit_transform(scaled)

            # MI class
            ax = axes[0]
            cmap = plt.get_cmap("tab10", N_CLASSES)
            for cls in range(N_CLASSES):
                mask = sub_y == cls
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           s=8, alpha=0.6, color=cmap(cls), label=class_names[cls])
            ax.set_title("MI Class")
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.legend(markerscale=2, fontsize=8)
            ax.grid(alpha=0.2)

            # Subject
            ax = axes[1]
            unique_subjs = np.unique(sub_sid)
            cmap_s = plt.get_cmap("tab10", len(unique_subjs))
            for i, s in enumerate(unique_subjs):
                mask = sub_sid == s
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           s=8, alpha=0.6, color=cmap_s(i), label=f"S{s:02d}")
            ax.set_title("Subject")
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.legend(markerscale=2, fontsize=8, ncol=3)
            ax.grid(alpha=0.2)

            fig.tight_layout()
            out_path = out_dir / f"{model_name.lower()}_{method_name}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"  Saved: {out_path}")


# ── Results I/O ────────────────────────────────────────────────────────────────

def print_results(results: dict[str, dict]) -> None:
    """results[method] = metrics dict. Chance = 0.25 for 4-class."""
    print(f"\n{'═'*68}")
    print(f"  Task: MOTOR IMAGERY  (4-class, chance = 0.25)")
    print(f"{'═'*68}")
    print(f"  {'Method':<22}  {'Bal. Acc':>10}  {'± std':>7}  "
          f"{'Accuracy':>10}  {'± std':>7}")
    print(f"  {'─'*62}")
    for method, m in results.items():
        print(
            f"  {method:<22}  "
            f"{m['balanced_acc_mean']:>10.4f}  "
            f"{m['balanced_acc_std']:>7.4f}  "
            f"{m['accuracy_mean']:>10.4f}  "
            f"{m['accuracy_std']:>7.4f}"
        )
    print(f"{'═'*68}")


def save_results_csv(results: dict[str, dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"results_{timestamp}.csv"

    fieldnames = [
        "method", "dataset",
        "balanced_acc_mean", "balanced_acc_std",
        "accuracy_mean", "accuracy_std",
    ] + [f"fold_{i+1}_bal_acc" for i in range(N_FOLDS)] \
      + [f"fold_{i+1}_acc"     for i in range(N_FOLDS)]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, m in results.items():
            row = {
                "method":               method,
                "dataset":              "BCICIV_2a",
                "balanced_acc_mean":    round(m["balanced_acc_mean"], 6),
                "balanced_acc_std":     round(m["balanced_acc_std"],  6),
                "accuracy_mean":        round(m["accuracy_mean"],     6),
                "accuracy_std":         round(m["accuracy_std"],      6),
            }
            for i, v in enumerate(m["per_fold_bal_acc"]):
                row[f"fold_{i+1}_bal_acc"] = round(v, 6)
            for i, v in enumerate(m["per_fold_acc"]):
                row[f"fold_{i+1}_acc"] = round(v, 6)
            writer.writerow(row)

    print(f"  Results saved to {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="BCI IV 2a motor imagery benchmark")
    parser.add_argument("--skip-luna", action="store_true")
    parser.add_argument("--skip-zuna", action="store_true")
    parser.add_argument("--no-vis",   action="store_true")
    parser.add_argument("--data",     type=Path, default=DATA_PATH)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Load data ──────────────────────────────────────────────────────────────
    data        = load_data(args.data)
    X           = data["X"]
    y           = data["y"].numpy()
    subject_ids = data["subject_ids"].numpy()

    # ── PSD features ───────────────────────────────────────────────────────────
    print("\n[PSD] Computing Welch PSD features …")
    psd_feats = welch_psd_features(X.numpy())
    print(f"  PSD feature shape: {psd_feats.shape}")

    # ── LUNA features ──────────────────────────────────────────────────────────
    luna_feats = None
    if not args.skip_luna:
        print("\n[LUNA] Extracting frozen latent features …")
        try:
            chs_info   = make_bciciv_chs_info()
            luna_model = load_luna(chs_info)
            luna_feats = extract_luna_features(luna_model, X)
            print(f"  LUNA feature shape: {luna_feats.shape}")
            del luna_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  LUNA skipped — {e}")

    # ── ZUNA encoder features ──────────────────────────────────────────────────
    zuna_enc_feats = None
    if not args.skip_zuna:
        print("\n[ZUNA] Extracting frozen encoder features …")
        zuna_enc = zuna_encdec = None
        try:
            chan_pos, chan_pos_disc   = make_bciciv_chan_pos()
            zuna_enc, zuna_encdec, _ = load_zuna()
        except Exception as e:
            print(f"  ZUNA load failed — {e}")

        if zuna_enc is not None:
            try:
                zuna_enc_feats = extract_zuna_features(zuna_enc, X, chan_pos, chan_pos_disc)
                print(f"  ZUNAenc feature shape: {zuna_enc_feats.shape}")
            except Exception as e:
                import traceback
                print(f"  ZUNAenc failed — {type(e).__name__}: {e}")
                traceback.print_exc()

        if zuna_enc is not None or zuna_encdec is not None:
            del zuna_enc, zuna_encdec
            torch.cuda.empty_cache()

    # ── Evaluation ─────────────────────────────────────────────────────────────
    print(f"\n[EVAL] {N_FOLDS}-fold stratified CV — 4-class motor imagery")
    results: dict[str, dict] = {}

    print("  PSD + PCA + Multinomial LogReg …")
    results["PSD+PCA+LogReg"] = evaluate_windows(psd_feats, y, use_pca=True)

    if luna_feats is not None:
        print("  LUNA (frozen) + Multinomial LogReg …")
        results["LUNA+LogReg"] = evaluate_windows(luna_feats, y, use_pca=False)

    if zuna_enc_feats is not None:
        print("  ZUNAenc (frozen) + Multinomial LogReg …")
        results["ZUNAenc+LogReg"] = evaluate_windows(zuna_enc_feats, y, use_pca=False)

    # ── Results ────────────────────────────────────────────────────────────────
    print_results(results)
    print("\n[SAVE]")
    save_results_csv(results, OUT_DIR)

    # ── Visualisation ──────────────────────────────────────────────────────────
    if not args.no_vis and (luna_feats is not None or zuna_enc_feats is not None):
        print("\n[VIS] Generating latent-space plots …")
        plot_latent_spaces(
            luna_feats     = luna_feats,
            zuna_enc_feats = zuna_enc_feats,
            y              = y,
            subject_ids    = subject_ids,
            out_dir        = OUT_DIR,
        )
        print(f"  Plots saved to {OUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
