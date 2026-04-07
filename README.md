# Milestone 4 — Continual Learning, Monitoring & Human-in-the-Loop

This folder contains the complete reproducible pipeline for the behavioral classifier, covering all work from Milestones 1–4: dataset generation, preprocessing, model training, robustness evaluation, monitoring, continual learning (CL), human-in-the-loop (HITL) simulation, and active learning.

---

## Project Overview

**Goal:** Multi-label classification of AI API call sequences into 5 behavioral categories:
`file_access` | `network_access` | `process_mgmt` | `pure_calculation` | `code_execution`

**Architecture:** `DualInputClassifier` — dual-branch CNN+BiLSTM model (368,781 params, 1.47 MB)
- Syscall branch: word-level tokenizer (vocab=59, max_len=512)
- API path branch: char-level tokenizer (vocab=33, max_len=64)

**Best checkpoint metrics (test set):**
| Metric | Value |
|---|---|
| Accuracy | 0.9677 |
| F1 (macro) | 0.9788 |
| F1 (micro) | 0.9841 |
| F1 (weighted) | 0.9838 |
| AUROC (macro) | 0.9953 |
| PR-AUC (macro) | 0.9897 |
| 5-Fold CV F1 | 0.9815 ± 0.0016 |

---

## Folder Structure

```
Milestone4/
├── M4_Pipeline.ipynb       # End-to-end Jupyter notebook (all 13 pipeline sections)
├── README.md                      # This file
├── data/
│   ├── training_data_fuzz/        # Preprocessed dataset (JSONL splits)
│   │   ├── train.jsonl            # 91,915 training records
│   │   ├── val.jsonl              #  9,509 validation records
│   │   ├── test.jsonl             #  9,509 test records
│   │   └── metadata.json          # Label schema, class weights, splits info
│   └── checkpoints/
│       ├── best_model.pt          # Trained DualInputClassifier checkpoint
│       ├── tokenizer.json         # Syscall word-level tokenizer (vocab=59)
│       └── path_tokenizer.json    # API path char-level tokenizer (vocab=33)
├── scripts/                       # All pipeline Python scripts (see descriptions below)
├── figures/                       # Generated plots from M4 experiments
│   ├── m4_learning_curves.png     # Training F1 / loss over epochs
│   ├── m4_metrics_bar.png         # Bar chart of all 9 evaluation metrics
│   ├── m4_cv.png                  # 5-Fold cross-validation F1 per fold
│   ├── m4_dashboard.png           # 4-panel M4 performance dashboard
│   ├── cl_dashboard.png           # CL drift before/after F1 dashboard
│   ├── cl_per_metric.png          # All 8 metrics across CL drift steps
│   ├── cl_heatmap.png             # CL improvement delta heatmap
│   └── cl_loss.png                # BCE loss convergence per drift step
└── outputs/                       # Artifacts saved by notebook runs
    └── pipeline_results.json      # Written by Section 13 of the notebook
```

---

## Dependencies

Python 3.10 or 3.11 is required. Install into your environment:

```bash
pip install torch==2.5.1          # CPU: add +cpu suffix; GPU: see Hardware section
pip install scikit-learn matplotlib numpy tqdm nbformat python-docx
```

**AMD GPU (ROCm 6.2) — training environment used in this project:**
```bash
pip install torch==2.5.1+rocm6.2 --index-url https://download.pytorch.org/whl/rocm6.2
```

**Required environment variables for AMD GPU:**
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export TORCH_BLAS_PREFER_HIPBLASLT=0
```

CPU training works without these variables and without ROCm.

---

## Pipeline Stages & Run Commands

All commands assume your working directory is `Milestone4/`. Use relative paths as shown.

### Stage 1 — Data Extraction (upstream, optional)

Extracts API call sequences from Python source files. Requires the full source corpus from the parent project.

```bash
python scripts/deep_api_extractor.py \
    --input-dir /path/to/python/sources \
    --output-file api_calls.jsonl
```

### Stage 2 — Behavioral Labeling (upstream, optional)

Labels API call sequences with behavioral categories using static analysis.

```bash
python scripts/api_behavior_labeler.py \
    --input api_calls.jsonl \
    --output labeled_dataset.jsonl
```

### Stage 3 — Strace Syscall Capture (upstream, optional)

Captures real Linux syscall sequences for each behavioral category via strace.

```bash
# Requires a conda env with CPU-only PyTorch + TF + JAX
python scripts/strace_capture.py \
    --env-python ~/miniconda3/envs/strace_env/bin/python \
    --output strace_syscall_map.json

python scripts/apply_strace_syscalls.py \
    --input labeled_dataset.jsonl \
    --output labeled_dataset_strace.jsonl \
    --strace-map strace_syscall_map.json
```

### Stage 4 — Strace Fuzzing (upstream, optional)

Generates additional fuzzed training samples using the strace API fuzzer.

```bash
python scripts/strace_api_fuzzer.py \
    --input labeled_dataset_strace.jsonl \
    --output labeled_dataset_fuzz.jsonl \
    --fuzz-count 15000
```

### Stage 5 — Preprocessing

Tokenizes and splits the dataset. **The preprocessed dataset is already included in `data/training_data_fuzz/`, so this step can be skipped.**

```bash
python scripts/preprocess_dataset.py \
    --input labeled_dataset_fuzz.jsonl \
    --output-dir data/training_data_fuzz \
    --input-mode syscall \
    --calc-cap 25000
```

### Stage 6 — Training

Trains the DualInputClassifier. **A trained checkpoint is already included in `data/checkpoints/best_model.pt`.**

```bash
# AMD GPU (ROCm):
HSA_OVERRIDE_GFX_VERSION=10.3.0 TORCH_BLAS_PREFER_HIPBLASLT=0 \
python scripts/train.py \
    --mode train \
    --data-dir data/training_data_fuzz \
    --checkpoint-dir data/checkpoints \
    --tokenizer-type syscall \
    --dual-input \
    --epochs 20 \
    --batch-size 256 \
    --cnn-filters 64 \
    --lstm-hidden 64

# CPU:
python scripts/train.py \
    --mode train \
    --data-dir data/training_data_fuzz \
    --checkpoint-dir data/checkpoints \
    --tokenizer-type syscall \
    --dual-input \
    --epochs 20 \
    --batch-size 256
```

Training artifacts saved to `data/checkpoints/`:
- `best_model.pt` — best checkpoint by val F1
- `last_model.pt` — final epoch checkpoint

### Stage 7 — Threshold Sweep

Finds the optimal classification threshold on the validation set.

```bash
python scripts/threshold_sweep.py \
    --checkpoint data/checkpoints/best_model.pt \
    --tokenizer data/checkpoints/tokenizer.json \
    --val-data data/training_data_fuzz/val.jsonl
```

### Stage 8 — Robustness & Milestone 3 Experiments

```bash
python scripts/milestone3_experiments.py \
    --checkpoint data/checkpoints/best_model.pt \
    --tokenizer data/checkpoints/tokenizer.json \
    --test-data data/training_data_fuzz/test.jsonl
```

### Stage 9 — Milestone 4 Experiments (Monitoring, CL, HITL, Active Learning)

```bash
python scripts/milestone4_experiments.py \
    --checkpoint data/checkpoints/best_model.pt \
    --tokenizer data/checkpoints/tokenizer.json \
    --path-tokenizer data/checkpoints/path_tokenizer.json \
    --test-data data/training_data_fuzz/test.jsonl \
    --val-data data/training_data_fuzz/val.jsonl \
    --train-data data/training_data_fuzz/train.jsonl
```

### Stage 10 — Full Notebook (recommended entry point)

The notebook runs all pipeline stages end-to-end and saves results to `outputs/`.

```bash
cd Milestone4/
jupyter notebook M4_Pipeline.ipynb
# or
jupyter lab M4_Pipeline.ipynb
```

Run cells sequentially. Section 13 saves `outputs/pipeline_results.json` as the primary output artifact.

---

## Scripts Reference

| Script | Description |
|---|---|
| `deep_api_extractor.py` | Extracts API call sequences from Python source files |
| `api_behavior_labeler.py` | Labels extracted API calls with behavioral categories |
| `strace_workloads.py` | Defines 15 workloads (5 categories × 3 frameworks) for strace capture |
| `strace_capture.py` | Runs workloads under strace, computes baseline-subtracted syscall maps |
| `apply_strace_syscalls.py` | Replaces derived syscalls in dataset with real strace-captured ones |
| `strace_api_fuzzer.py` | Generates fuzzed variants of labeled records for data augmentation |
| `preprocess_dataset.py` | Tokenizes, splits, and balances dataset into train/val/test JSONL |
| `train.py` | DualInputClassifier training with early stopping, learning rate scheduling |
| `threshold_sweep.py` | Sweeps classification thresholds 0.10–0.60 on val set, reports per-label F1 |
| `milestone3_experiments.py` | Robustness tests: token dropout, char noise, label noise, OOD detection |
| `milestone4_experiments.py` | CL drift, HITL simulation, active learning, distribution monitoring |
| `collect_metrics.py` | Collects and aggregates evaluation metrics across checkpoints |

---

## Outputs

All outputs from notebook and script runs are saved under `outputs/` and `figures/`:

| Output | Description |
|---|---|
| `outputs/pipeline_results.json` | Full metrics from end-to-end notebook run (Section 13) |
| `figures/m4_learning_curves.png` | F1 and loss curves across training epochs |
| `figures/m4_metrics_bar.png` | Bar chart: all 9 evaluation metrics at best threshold |
| `figures/m4_cv.png` | 5-Fold cross-validation F1 per fold with mean ± std |
| `figures/m4_dashboard.png` | 4-panel M4 performance summary dashboard |
| `figures/cl_dashboard.png` | CL drift: F1 before/after fine-tuning per drift scenario |
| `figures/cl_per_metric.png` | All 8 metrics across 4 CL drift steps with before/after arrows |
| `figures/cl_heatmap.png` | Improvement delta heatmap across drift steps and metrics |
| `figures/cl_loss.png` | BCE loss convergence curves per drift step (3 fine-tuning epochs) |

---

## Notebook Sections

The notebook (`M4_Pipeline.ipynb`) covers:

1. Environment setup and path configuration
2. Tokenizer and model loading (DualInputClassifier)
3. Dataset loading and label distribution inspection
4. Baseline inference on test set
5. Full metrics suite: accuracy, precision, recall, F1 (macro/micro/weighted), AUROC, PR-AUC
6. 5-Fold cross-validation
7. Robustness experiments (token dropout, char noise)
8. Distribution monitoring (PSI, ECE, entropy-based drift detection)
9. Adversarial stress tests (FGSM embedding attack)
10. Continual learning (4 drift scenarios, head-only fine-tuning, replay buffer)
11. HITL simulation (uncertainty-based query, oracle labeling, model update)
12. Active learning (margin-based uncertainty, 5 query cycles)
13. End-to-end demo and results export to `outputs/pipeline_results.json`

---

## Hardware Notes

- **AMD GPU (ROCm 6.2):** Set `HSA_OVERRIDE_GFX_VERSION=10.3.0` and `TORCH_BLAS_PREFER_HIPBLASLT=0`. Training on GPU takes ~2–5 min per epoch for the full 91K-record dataset.
- **CPU fallback:** All scripts and the notebook detect `cuda` availability and fall back to CPU automatically. Inference runs at ~3,641 samples/sec on GPU; CPU will be slower.
- **Memory:** The full training set requires ~4 GB RAM. The checkpoint loads in ~20 MB peak GPU memory.

---

## Reproducibility

- All paths in the notebook are relative to `Milestone4/` via `NOTEBOOK_DIR = Path('.').resolve()`.
- No absolute paths appear in any script or notebook cell.
- The included checkpoint and dataset reproduce the reported metrics exactly.
- Random seeds: training uses `torch.manual_seed(42)` and `numpy.random.seed(42)`.
- Class weights: `pos_weight=[4.0, 5.0, 6.0, 2.0, 2.5]` for `[file_access, network_access, process_mgmt, pure_calculation, code_execution]`.
