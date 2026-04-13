#!/usr/bin/env bash
# =============================================================================
# setup_strace_env.sh — Create strace_env conda environment
#
# CPU-only TF + JAX + PyTorch, all pinned to numpy 1.26.4.
# This env is used ONLY for strace workload capture, not for training.
# Training continues to use tf_env (ROCm PyTorch).
# =============================================================================
set -euo pipefail

ENV_NAME="strace_env"
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${CONDA_PREFIX:-$HOME/miniconda3}")"
PY="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
PIP="${CONDA_BASE}/envs/${ENV_NAME}/bin/pip"

echo "============================================================"
echo "  Creating conda environment: ${ENV_NAME}"
echo "  Base: ${CONDA_BASE}"
echo "============================================================"

# Remove existing env if present
if "${CONDA_BASE}/bin/conda" env list | grep -q "^${ENV_NAME} "; then
    echo "  Removing existing ${ENV_NAME}..."
    "${CONDA_BASE}/bin/conda" env remove -y -n "${ENV_NAME}"
fi

"${CONDA_BASE}/bin/conda" create -y -n "${ENV_NAME}" python=3.10
echo ""

echo "--- [1/5] Installing CPU-only PyTorch 2.5.1 ---"
"${PIP}" install --quiet \
    torch==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

echo "--- [2/5] Pinning numpy 1.26.4 ---"
"${PIP}" install --quiet "numpy==1.26.4"

echo "--- [3/5] Installing TensorFlow CPU 2.14.1 ---"
# Install TF CPU; tensorflow-io-gcs-filesystem may fail on Ubuntu 24 (glibc mismatch).
# The strace workloads only use local filesystem ops, so the GCS plugin is not required.
if "${PIP}" install --quiet \
    "tensorflow-cpu==2.14.1" \
    "ml-dtypes==0.2.0" \
    "wrapt==1.14.1" 2>&1; then
    echo "  TF installed OK"
else
    echo "  WARNING: TF install failed, trying without tensorflow-io-gcs-filesystem..."
    "${PIP}" install --quiet --no-deps "tensorflow-cpu==2.14.1"
    "${PIP}" install --quiet \
        "keras==2.14.0" \
        "tensorflow-estimator==2.14.0" \
        "ml-dtypes==0.2.0" \
        "wrapt==1.14.1" \
        "absl-py>=1.0.0" \
        "astunparse>=1.6.0" \
        "flatbuffers>=23.5.26" \
        "gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1" \
        "google-pasta>=0.1.1" \
        "grpcio<2.0,>=1.24.3" \
        "h5py>=2.9.0" \
        "libclang>=13.0.0" \
        "opt-einsum>=2.3.2" \
        "packaging" \
        "protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3" \
        "setuptools" \
        "six>=1.12.0" \
        "termcolor>=1.1.0" \
        "typing-extensions>=3.6.6" \
        "tensorboard<2.15,>=2.14"
fi

echo "--- [4/5] Installing JAX 0.4.30 CPU ---"
"${PIP}" install --quiet "jax==0.4.30" "jaxlib==0.4.30"

echo "--- [5/5] Re-pinning numpy (TF/JAX may have upgraded it) ---"
"${PIP}" install --quiet "numpy==1.26.4"

echo ""
echo "============================================================"
echo "  Verifying installations"
echo "============================================================"
"${PY}" - <<'PYEOF'
import sys

results = {}

try:
    import torch
    results['torch'] = torch.__version__
except Exception as e:
    results['torch'] = f'FAILED: {e}'

try:
    import tensorflow as tf
    results['tensorflow'] = tf.__version__
except Exception as e:
    results['tensorflow'] = f'FAILED: {e}'

try:
    import jax
    results['jax'] = jax.__version__
except Exception as e:
    results['jax'] = f'FAILED: {e}'

try:
    import numpy as np
    results['numpy'] = np.__version__
except Exception as e:
    results['numpy'] = f'FAILED: {e}'

print()
all_ok = True
for pkg, ver in results.items():
    status = '✓' if 'FAILED' not in str(ver) else '✗'
    print(f"  {status} {pkg}: {ver}")
    if 'FAILED' in str(ver):
        all_ok = False

print()
if all_ok:
    print("  All packages OK — strace_env is ready.")
else:
    print("  WARNING: some packages failed (see above).")
    print("  Strace capture will skip broken frameworks.")
PYEOF

echo ""
echo "  Python binary: ${PY}"
echo "============================================================"
