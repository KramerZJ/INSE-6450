# Milestone 2 - Model Selection, Training & Evaluation

## Contents

- `requirements.pdf` - Milestone 2 requirement specification
- `report.docx` - Milestone 2 deliverable report
- `M2_Training_Pipeline.ipynb` - Consolidated notebook with all M2 scripts

### scripts/

| Script | Description |
|---|---|
| `train.py` | DualInputClassifier training with early stopping, LR scheduling |
| `threshold_sweep.py` | Sweeps classification thresholds 0.10-0.60 on val set |

### checkpoints/

- `best_model.pt` - best checkpoint by val F1 (DualInputClassifier, fuzz data)
- `best_model_strace.pt` - best checkpoint for strace variant
- `checkpoint_epoch_{5,10,15,20}.pt` - periodic epoch checkpoints
- `tokenizer.json` - syscall word-level tokenizer (vocab=59)
- `tokenizer_strace.json` - strace tokenizer variant
- `path_tokenizer.json` - API path char-level tokenizer (vocab=33)

### figures/

- `learning_curves.png` - training F1/loss over epochs

### outputs/

- `milestone2_metrics.json` - evaluation metrics from M2 experiments

## Run Commands

Training (from `milestones/milestone2_model/`):
```bash
python scripts/train.py \
    --mode train \
    --data-dir ../milestone1_data/data/splits/training_data_fuzz \
    --checkpoint-dir ./checkpoints \
    --tokenizer-type syscall --dual-input \
    --epochs 20 --batch-size 256 --cnn-filters 64 --lstm-hidden 64
```

Threshold sweep:
```bash
python scripts/threshold_sweep.py \
    --checkpoint ./checkpoints/best_model.pt \
    --tokenizer ./checkpoints/tokenizer.json \
    --val-data ../milestone1_data/data/splits/training_data_fuzz/val.jsonl
```
