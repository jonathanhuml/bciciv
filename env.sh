#!/usr/bin/env bash
# env.sh — BCI IV 2a experiment environment setup
# Usage: source env.sh
# Creates (if needed) and activates a venv, then installs all dependencies.

set -e

VENV_DIR="${VENV_DIR:-.venv}"

# ── Create venv if it doesn't exist ───────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "[env.sh] Creating virtual environment at $VENV_DIR …"
    python3 -m venv "$VENV_DIR"
fi

# ── Activate ───────────────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ── Python dependencies ────────────────────────────────────────────────────────
pip install --upgrade pip
pip install \
    "torch>=2.5" \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    mne \
    braindecode \
    huggingface-hub \
    safetensors \
    zuna

# ── Environment variables ──────────────────────────────────────────────────────

# HuggingFace: cache model weights to a persistent volume if available
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"

# Suppress MNE's verbose output
export MNE_LOGGING_LEVEL=WARNING

# Torch
export TORCH_HOME="${TORCH_HOME:-/opt/torch_cache}"

# Determinism
export PYTHONHASHSEED=42

echo "[env.sh] Environment ready. Venv: $VENV_DIR"
