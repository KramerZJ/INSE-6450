# Milestone 3 - Robustness, Adversarial Testing & Monitoring

## Contents

- `requirements.pdf` - Milestone 3 requirement specification
- `report.docx` - Milestone 3 deliverable report
- `M3_Robustness.ipynb` - Consolidated notebook with all M3 scripts

### scripts/

| Script | Description |
|---|---|
| `milestone3_experiments.py` | Robustness tests: token dropout, char noise, label noise, OOD detection |
| `train.py` | Copy of the M2 training script (dependency for `milestone3_experiments.py`) |

### figures/

- `robustness_curves.png` - robustness degradation curves
- `adversarial.png` - adversarial attack results
- `calibration.png` - calibration curves
- `monitoring.png` - monitoring dashboard
- `label_distribution.png` - label distribution analysis

### outputs/

- `milestone3_results.json` - full experiment results

## Run Command

From `milestones/milestone3_robustness/`:
```bash
python scripts/milestone3_experiments.py \
    --data-dir ../milestone1_data/data/splits/training_data_fuzz \
    --checkpoint ../milestone2_model/checkpoints/best_model.pt \
    --tokenizer ../milestone2_model/checkpoints/tokenizer.json
```
