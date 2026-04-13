# Milestone 1 - Data Selection, Collection & Feature Engineering

## Contents

- `requirements.pdf` - Milestone 1 requirement specification
- `report.docx` - Milestone 1 deliverable report
- `M1_Data_Pipeline.ipynb` - Consolidated notebook with all M1 scripts

### scripts/

| Script | Description |
|---|---|
| `deep_api_extractor.py` | Extracts API call sequences from Python source files in tensorflow/pytorch/jax |
| `api_behavior_labeler.py` | Labels extracted API calls with behavioral categories via static analysis |
| `preprocess_dataset.py` | Tokenizes, splits, and balances dataset into train/val/test JSONL |
| `collect_metrics.py` | Collects and aggregates evaluation metrics across checkpoints |
| `strace_capture.py` | Runs workloads under strace, computes baseline-subtracted syscall maps |
| `strace_workloads.py` | Defines 15 workloads (5 categories x 3 frameworks) for strace capture |
| `strace_api_fuzzer.py` | Generates fuzzed variants of labeled records for data augmentation |
| `apply_strace_syscalls.py` | Replaces derived syscalls in dataset with real strace-captured ones |
| `setup_strace_env.sh` | Sets up the conda environment for strace capture |
| `train.py` | Copy of the M2 training script (dependency for `collect_metrics.py`) |

### data/raw/

Raw labeled datasets and intermediate labeling artifacts:
- `labeled_dataset.jsonl` (414 MB) - char-tokenized base dataset
- `labeled_dataset_fuzz.jsonl` (421 MB) - fuzz-augmented (used by M2-M4)
- `labeled_dataset_strace.jsonl` (431 MB) - strace-enriched variant
- `labeled.jsonl`, `review.jsonl`, `unified.jsonl` - intermediate pipeline artifacts
- `api_syscall_map*.json`, `strace_syscall_map.json` - syscall mappings
- `intermediate/` - scan result CSVs/JSONs from deep_api_extractor.py

### data/splits/

Preprocessed train/val/test splits:
- `training_data/` - original char-tokenized split
- `training_data_fuzz/` - fuzz-augmented split (primary, used by M2-M4)
- `training_data_strace/` - strace-enriched split

## Pipeline Order

```
deep_api_extractor.py -> api_behavior_labeler.py -> strace_capture.py ->
apply_strace_syscalls.py -> strace_api_fuzzer.py -> preprocess_dataset.py
```
