#!/usr/bin/env python3
"""
Milestone 3 — Full Experiment Suite
=====================================

Runs ALL experiments needed for Milestone 3 and outputs results as JSON.
Takes ~15-25 minutes depending on hardware.

Experiments:
  1. Stress tests (token dropout, char noise, truncation, OOD)
  2. Adversarial attacks (FGSM at multiple epsilon, char substitution)
  3. Calibration analysis (reliability diagram data, confidence histograms)
  4. Failure case mining (worst predictions with full details)
  5. Latency comparison (clean vs corrupted)
  6. Drift simulation + adaptation (shift priors → retrain head → before/after)
  7. Monitoring metrics (drift statistics on held-out data)

Usage:
    python milestone3_experiments.py \
        --data-dir ./training_data \
        --checkpoint ./checkpoints/best_model.pt \
        --tokenizer ./checkpoints/tokenizer.json
"""

import os
import sys
import json
import time
import copy
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, ".")
from train import CNNBiLSTMClassifier, CodeTokenizer, APICallDataset, MultiLabelMetrics

LABELS = ["file_access", "network_access", "process_mgmt", "pure_calculation", "code_execution"]
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_model_and_data(args):
    """Load model, tokenizer, test dataset."""
    tokenizer = CodeTokenizer(max_length=512)
    tokenizer.load(args.tokenizer)

    model = CNNBiLSTMClassifier(vocab_size=tokenizer.vocab_size)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()  # MIOpen requires train mode for LSTM backward

    test_dataset = APICallDataset(os.path.join(args.data_dir, "test.jsonl"), tokenizer)

    return model, tokenizer, test_dataset, device


def evaluate_dataset(model, dataset, device, batch_size=64):
    """Run evaluation, return metrics dict + all probs/labels."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    metrics = MultiLabelMetrics()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            lens = batch["length"].to(device)
            labs = batch["labels"].to(device)
            logits = model(ids, lens)
            metrics.update(logits, labs)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labs.cpu().numpy())

    results = metrics.compute()
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    return results, all_probs, all_labels


# ════════════════════════════════════════════════════════
# 1. STRESS TESTS
# ════════════════════════════════════════════════════════

class StressTestedDataset(torch.utils.data.Dataset):
    """Wraps a dataset and applies corruption to input_ids."""

    def __init__(self, base_dataset, corruption_fn):
        self.base = base_dataset
        self.corrupt = corruption_fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        item["input_ids"] = self.corrupt(item["input_ids"].clone())
        return item


def token_dropout(ids, rate=0.1):
    """Randomly zero out tokens."""
    mask = torch.rand(ids.shape) < rate
    mask[0] = False  # keep BOS
    ids[mask] = 0  # PAD
    return ids

def char_noise(ids, rate=0.1, vocab_size=112):
    """Replace tokens with random chars."""
    mask = torch.rand(ids.shape) < rate
    mask[0] = False
    noise = torch.randint(4, vocab_size, ids.shape)
    ids[mask] = noise[mask]
    return ids

def truncate_context(ids, keep_frac=0.5):
    """Keep only first fraction of non-pad tokens."""
    nonpad = (ids != 0).sum().item()
    cutoff = max(2, int(nonpad * keep_frac))
    ids[cutoff:] = 0
    return ids

def run_stress_tests(model, test_dataset, device, tokenizer):
    """Run all stress tests and return results."""
    logger.info("=" * 60)
    logger.info("1. STRESS TESTS")
    logger.info("=" * 60)

    results = {}

    # Clean baseline
    logger.info("  Clean baseline...")
    clean_res, _, _ = evaluate_dataset(model, test_dataset, device)
    results["clean"] = {
        "macro_f1": clean_res["macro_f1"],
        "exact_match": clean_res["exact_match"],
        "hamming_loss": clean_res["hamming_loss"],
    }

    # Token dropout at various rates
    for rate in [0.05, 0.10, 0.20, 0.30, 0.50]:
        logger.info(f"  Token dropout rate={rate}...")
        corrupted = StressTestedDataset(test_dataset, lambda ids, r=rate: token_dropout(ids, r))
        res, _, _ = evaluate_dataset(model, corrupted, device)
        results[f"token_dropout_{rate}"] = {
            "macro_f1": res["macro_f1"],
            "exact_match": res["exact_match"],
            "hamming_loss": res["hamming_loss"],
        }

    # Char noise at various rates
    for rate in [0.05, 0.10, 0.20, 0.30]:
        logger.info(f"  Char noise rate={rate}...")
        corrupted = StressTestedDataset(
            test_dataset,
            lambda ids, r=rate: char_noise(ids, r, tokenizer.vocab_size)
        )
        res, _, _ = evaluate_dataset(model, corrupted, device)
        results[f"char_noise_{rate}"] = {
            "macro_f1": res["macro_f1"],
            "exact_match": res["exact_match"],
            "hamming_loss": res["hamming_loss"],
        }

    # Truncation
    for frac in [0.75, 0.50, 0.25]:
        logger.info(f"  Truncation keep={frac}...")
        corrupted = StressTestedDataset(test_dataset, lambda ids, f=frac: truncate_context(ids, f))
        res, _, _ = evaluate_dataset(model, corrupted, device)
        results[f"truncate_{frac}"] = {
            "macro_f1": res["macro_f1"],
            "exact_match": res["exact_match"],
            "hamming_loss": res["hamming_loss"],
        }

    # OOD: random garbage input
    logger.info("  OOD: random tokens...")
    ood_dataset = StressTestedDataset(
        test_dataset,
        lambda ids: torch.randint(4, tokenizer.vocab_size, ids.shape)
    )
    res_ood, probs_ood, _ = evaluate_dataset(model, ood_dataset, device)
    results["ood_random"] = {
        "macro_f1": res_ood["macro_f1"],
        "exact_match": res_ood["exact_match"],
        "avg_max_confidence": round(float(probs_ood.max(axis=1).mean()), 4),
        "avg_confidence_all": round(float(probs_ood.mean()), 4),
    }

    # OOD: English text (not code)
    logger.info("  OOD: English text...")

    class OODTextDataset(torch.utils.data.Dataset):
        def __init__(self, tokenizer, n=500):
            self.tokenizer = tokenizer
            self.texts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is a subset of artificial intelligence.",
                "The weather today is sunny with a high of 75 degrees.",
                "Please remember to submit your assignment by Friday.",
                "The stock market experienced significant volatility today.",
            ] * (n // 5)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            tokens, length = self.tokenizer.encode_padded(self.texts[idx])
            return {
                "input_ids": torch.tensor(tokens, dtype=torch.long),
                "length": torch.tensor(length, dtype=torch.long),
                "labels": torch.zeros(5, dtype=torch.float32),
            }

    ood_eng = OODTextDataset(tokenizer)
    _, probs_eng, _ = evaluate_dataset(model, ood_eng, device)
    results["ood_english_text"] = {
        "avg_max_confidence": round(float(probs_eng.max(axis=1).mean()), 4),
        "avg_confidence_all": round(float(probs_eng.mean()), 4),
        "frac_any_label_above_0.5": round(float((probs_eng > 0.5).any(axis=1).mean()), 4),
    }

    for k, v in results.items():
        logger.info(f"  {k}: {v}")

    return results


# ════════════════════════════════════════════════════════
# 2. ADVERSARIAL ATTACKS
# ════════════════════════════════════════════════════════

def fgsm_attack(model, input_ids, lengths, labels, epsilon, device):
    """FGSM on the embedding layer."""
    model.train()  # MIOpen requires train mode for LSTM backward
    input_ids = input_ids.to(device)
    lengths = lengths.to(device)
    labels = labels.to(device)

    # Get embeddings and make them require grad
    embeds = model.embedding(input_ids)
    embeds = embeds.detach().requires_grad_(True)

    # Forward through rest of model manually
    x_cnn = embeds.permute(0, 2, 1)
    conv_outs = [conv(x_cnn) for conv in model.convs]
    x = torch.cat(conv_outs, dim=1).permute(0, 2, 1)

    lengths_cpu = lengths.cpu().clamp(min=1)
    packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
    lstm_out, _ = model.lstm(packed)
    lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

    attn_w = model.attention(lstm_out)
    mask = torch.arange(lstm_out.size(1), device=device).unsqueeze(0) >= lengths.unsqueeze(1)
    attn_w = attn_w.masked_fill(mask.unsqueeze(2), float('-inf'))
    attn_w = torch.softmax(attn_w, dim=1)
    context = (lstm_out * attn_w).sum(dim=1)

    logits = model.classifier(context)
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()

    # FGSM perturbation on embeddings
    perturbed = embeds + epsilon * embeds.grad.sign()

    # Forward again with perturbed embeddings
    with torch.no_grad():
        x_cnn = perturbed.permute(0, 2, 1)
        conv_outs = [conv(x_cnn) for conv in model.convs]
        x = torch.cat(conv_outs, dim=1).permute(0, 2, 1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        lstm_out, _ = model.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        attn_w = model.attention(lstm_out)
        attn_w = attn_w.masked_fill(mask.unsqueeze(2), float('-inf'))
        attn_w = torch.softmax(attn_w, dim=1)
        context = (lstm_out * attn_w).sum(dim=1)
        adv_logits = model.classifier(context)

    return adv_logits


def char_substitution_attack(ids, targets, vocab_size=112, n_swaps=5):
    """Swap n random non-pad tokens with random chars (black-box attack)."""
    ids = ids.clone()
    for i in range(ids.size(0)):
        nonpad = (ids[i] != 0).nonzero(as_tuple=True)[0]
        if len(nonpad) < 3:
            continue
        # Skip BOS/EOS
        candidates = nonpad[1:-1]
        if len(candidates) == 0:
            continue
        n = min(n_swaps, len(candidates))
        swap_idx = candidates[torch.randperm(len(candidates))[:n]]
        ids[i, swap_idx] = torch.randint(4, vocab_size, (n,))
    return ids


def run_adversarial(model, test_dataset, device, tokenizer):
    """Run adversarial evaluations."""
    logger.info("\n" + "=" * 60)
    logger.info("2. ADVERSARIAL ATTACKS")
    logger.info("=" * 60)

    results = {}
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # FGSM at multiple epsilon
    for eps in [0.01, 0.05, 0.1, 0.2, 0.5]:
        logger.info(f"  FGSM epsilon={eps}...")
        metrics = MultiLabelMetrics()
        for batch in loader:
            adv_logits = fgsm_attack(
                model, batch["input_ids"], batch["length"], batch["labels"], eps, device
            )
            metrics.update(adv_logits, batch["labels"].to(device))
        res = metrics.compute()
        results[f"fgsm_eps_{eps}"] = {
            "macro_f1": res["macro_f1"],
            "exact_match": res["exact_match"],
            "per_label_f1": {name: res[name]["f1"] for name in LABELS},
        }
        logger.info(f"    macro_f1={res['macro_f1']}, exact_match={res['exact_match']}")

    # Character substitution (black-box) at multiple swap counts
    for n_swaps in [3, 5, 10, 20, 50]:
        logger.info(f"  Char substitution n_swaps={n_swaps}...")
        corrupted = StressTestedDataset(
            test_dataset,
            lambda ids, n=n_swaps: char_substitution_attack(
                ids.unsqueeze(0), None, tokenizer.vocab_size, n
            ).squeeze(0)
        )
        res, _, _ = evaluate_dataset(model, corrupted, device)
        results[f"char_sub_{n_swaps}"] = {
            "macro_f1": res["macro_f1"],
            "exact_match": res["exact_match"],
        }
        logger.info(f"    macro_f1={res['macro_f1']}, exact_match={res['exact_match']}")

    return results


# ════════════════════════════════════════════════════════
# 3. CALIBRATION ANALYSIS
# ════════════════════════════════════════════════════════

def run_calibration(model, test_dataset, device):
    """Compute calibration metrics: reliability diagram data, confidence stats."""
    logger.info("\n" + "=" * 60)
    logger.info("3. CALIBRATION ANALYSIS")
    logger.info("=" * 60)

    _, all_probs, all_labels = evaluate_dataset(model, test_dataset, device)

    results = {}

    # Per-label calibration (10 bins)
    n_bins = 10
    for li, name in enumerate(LABELS):
        probs = all_probs[:, li]
        labels = all_labels[:, li]

        bins = np.linspace(0, 1, n_bins + 1)
        bin_accs = []
        bin_confs = []
        bin_counts = []

        for b in range(n_bins):
            mask = (probs >= bins[b]) & (probs < bins[b + 1])
            if mask.sum() == 0:
                bin_accs.append(None)
                bin_confs.append(None)
                bin_counts.append(0)
            else:
                bin_accs.append(round(float(labels[mask].mean()), 4))
                bin_confs.append(round(float(probs[mask].mean()), 4))
                bin_counts.append(int(mask.sum()))

        # ECE (Expected Calibration Error)
        ece = 0
        total = len(probs)
        for b in range(n_bins):
            if bin_counts[b] > 0:
                ece += (bin_counts[b] / total) * abs(bin_accs[b] - bin_confs[b])

        results[name] = {
            "bin_accuracies": bin_accs,
            "bin_confidences": bin_confs,
            "bin_counts": bin_counts,
            "ece": round(ece, 4),
        }
        logger.info(f"  {name}: ECE={ece:.4f}")

    # Overall confidence histogram
    max_probs = all_probs.max(axis=1)
    results["confidence_histogram"] = {
        "mean_max_conf": round(float(max_probs.mean()), 4),
        "std_max_conf": round(float(max_probs.std()), 4),
        "frac_above_0.9": round(float((max_probs > 0.9).mean()), 4),
        "frac_above_0.95": round(float((max_probs > 0.95).mean()), 4),
        "frac_below_0.5": round(float((max_probs < 0.5).mean()), 4),
    }
    logger.info(f"  Confidence: mean_max={max_probs.mean():.4f}, frac>0.9={float((max_probs > 0.9).mean()):.4f}")

    # Macro ECE
    macro_ece = np.mean([results[name]["ece"] for name in LABELS])
    results["macro_ece"] = round(macro_ece, 4)
    logger.info(f"  Macro ECE: {macro_ece:.4f}")

    return results


# ════════════════════════════════════════════════════════
# 4. FAILURE CASE MINING
# ════════════════════════════════════════════════════════

def mine_failures(model, test_dataset, device):
    """Find worst predictions for the failure table."""
    logger.info("\n" + "=" * 60)
    logger.info("4. FAILURE CASE MINING")
    logger.info("=" * 60)

    _, all_probs, all_labels = evaluate_dataset(model, test_dataset, device)
    preds = (all_probs > 0.5).astype(int)

    failures = []
    for i in range(len(test_dataset)):
        if not np.array_equal(preds[i], all_labels[i]):
            record = test_dataset.records[i]
            # Compute per-sample loss as severity
            loss = 0
            for j in range(5):
                loss += abs(all_probs[i][j] - all_labels[i][j])

            pred_labels = [LABELS[j] for j in range(5) if preds[i][j] == 1]
            true_labels = [LABELS[j] for j in range(5) if all_labels[i][j] == 1]
            max_conf = float(all_probs[i].max())

            failures.append({
                "index": i,
                "api_path": record.get("api_path", ""),
                "text_snippet": record.get("text", "")[:120],
                "predicted": pred_labels,
                "expected": true_labels,
                "probabilities": {LABELS[j]: round(float(all_probs[i][j]), 3) for j in range(5)},
                "max_confidence": round(max_conf, 3),
                "severity": round(float(loss), 3),
            })

    # Sort by severity
    failures.sort(key=lambda x: -x["severity"])

    # Take top 10
    top_failures = failures[:10]

    logger.info(f"  Total misclassified: {len(failures)}/{len(test_dataset)} ({len(failures)/len(test_dataset)*100:.1f}%)")
    for i, f in enumerate(top_failures):
        logger.info(f"  #{i+1}: {f['api_path'][:60]}")
        logger.info(f"       pred={f['predicted']}, true={f['expected']}, conf={f['max_confidence']}")

    return {"total_failures": len(failures), "total_samples": len(test_dataset), "top_failures": top_failures}


# ════════════════════════════════════════════════════════
# 5. LATENCY: CLEAN vs CORRUPTED
# ════════════════════════════════════════════════════════

def run_latency_comparison(model, test_dataset, device, tokenizer):
    """Compare latency on clean vs corrupted inputs."""
    logger.info("\n" + "=" * 60)
    logger.info("5. LATENCY: CLEAN vs CORRUPTED")
    logger.info("=" * 60)

    results = {}

    for label, ds in [
        ("clean", test_dataset),
        ("token_dropout_0.2", StressTestedDataset(test_dataset, lambda ids: token_dropout(ids, 0.2))),
        ("char_noise_0.2", StressTestedDataset(test_dataset, lambda ids: char_noise(ids, 0.2, tokenizer.vocab_size))),
    ]:
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        latencies = []

        # Warmup
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= 3: break
                model(batch["input_ids"].to(device), batch["length"].to(device))

        with torch.no_grad():
            for batch in loader:
                t0 = time.perf_counter()
                model(batch["input_ids"].to(device), batch["length"].to(device))
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000)

        arr = np.array(latencies)
        total_samples = len(ds)
        total_time = arr.sum() / 1000

        results[label] = {
            "batch64_p50_ms": round(float(np.percentile(arr, 50)), 2),
            "batch64_p90_ms": round(float(np.percentile(arr, 90)), 2),
            "throughput_samples_sec": round(total_samples / total_time, 1),
        }
        logger.info(f"  {label}: p50={results[label]['batch64_p50_ms']}ms, throughput={results[label]['throughput_samples_sec']}/sec")

    return results


# ════════════════════════════════════════════════════════
# 6. DRIFT SIMULATION + ADAPTATION
# ════════════════════════════════════════════════════════

def run_drift_adaptation(model, test_dataset, tokenizer, device, data_dir):
    """Simulate distribution drift, then adapt model, report before/after."""
    logger.info("\n" + "=" * 60)
    logger.info("6. DRIFT SIMULATION + ADAPTATION")
    logger.info("=" * 60)

    results = {}

    # Load train data for adaptation
    train_dataset = APICallDataset(os.path.join(data_dir, "train.jsonl"), tokenizer)

    # Simulate drift: filter to only network/process heavy samples (rare classes)
    # This simulates a new codebase that is heavily distributed/RPC-focused
    logger.info("  Creating drifted dataset (network + process heavy)...")
    drift_indices = []
    for i, r in enumerate(test_dataset.records):
        labels = r.get("label_vector", [0]*5)
        # Keep if has network_access or process_mgmt
        if labels[1] == 1 or labels[2] == 1:
            drift_indices.append(i)
    # Also add some random pure_calculation to make it a real mix
    other_indices = [i for i in range(len(test_dataset)) if i not in set(drift_indices)]
    random.seed(42)
    random.shuffle(other_indices)
    drift_indices.extend(other_indices[:len(drift_indices) // 2])
    random.shuffle(drift_indices)

    drifted_test = Subset(test_dataset, drift_indices)
    logger.info(f"  Drifted test set: {len(drifted_test)} samples (vs {len(test_dataset)} original)")

    # Evaluate original model on drifted data
    logger.info("  Evaluating original model on drifted data...")
    before_res, _, _ = evaluate_dataset(model, drifted_test, device)
    results["before_adaptation"] = {
        "dataset_size": len(drifted_test),
        "macro_f1": before_res["macro_f1"],
        "exact_match": before_res["exact_match"],
        "per_label_f1": {name: before_res[name]["f1"] for name in LABELS},
    }
    logger.info(f"  Before: macro_f1={before_res['macro_f1']}")

    # Adaptation: fine-tune classifier head on a small sample of drift-like training data
    logger.info("  Fine-tuning classifier head on drift-like training subset...")
    # Select similar training samples
    drift_train_indices = []
    for i, r in enumerate(train_dataset.records):
        labels = r.get("label_vector", [0]*5)
        if labels[1] == 1 or labels[2] == 1:
            drift_train_indices.append(i)
    random.shuffle(drift_train_indices)
    drift_train_indices = drift_train_indices[:5000]  # small adaptation set

    drift_train = Subset(train_dataset, drift_train_indices)
    drift_loader = DataLoader(drift_train, batch_size=64, shuffle=True)

    # Copy model, freeze everything except classifier
    adapted_model = copy.deepcopy(model)
    for name, param in adapted_model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, adapted_model.parameters()),
        lr=5e-4, weight_decay=1e-4
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.87, 1.55, 3.93, 0.44, 0.78]).to(device)
    )

    # Fine-tune for 3 epochs
    adapted_model.train()
    adapt_start = time.perf_counter()
    for epoch in range(3):
        total_loss = 0
        for batch in drift_loader:
            ids = batch["input_ids"].to(device)
            lens = batch["length"].to(device)
            labs = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = adapted_model(ids, lens)
            loss = criterion(logits, labs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.info(f"    Adaptation epoch {epoch+1}: loss={total_loss/len(drift_loader):.4f}")
    adapt_time = time.perf_counter() - adapt_start

    # Evaluate adapted model
    adapted_model.eval()
    logger.info("  Evaluating adapted model on drifted data...")
    after_res, _, _ = evaluate_dataset(adapted_model, drifted_test, device)
    results["after_adaptation"] = {
        "macro_f1": after_res["macro_f1"],
        "exact_match": after_res["exact_match"],
        "per_label_f1": {name: after_res[name]["f1"] for name in LABELS},
        "adaptation_time_sec": round(adapt_time, 1),
        "adaptation_samples": len(drift_train_indices),
        "adaptation_epochs": 3,
    }
    logger.info(f"  After: macro_f1={after_res['macro_f1']}")

    # Also check adapted model on ORIGINAL test set (ensure no regression)
    logger.info("  Checking adapted model on original test set...")
    orig_res, _, _ = evaluate_dataset(adapted_model, test_dataset, device)
    results["adapted_on_original"] = {
        "macro_f1": orig_res["macro_f1"],
        "exact_match": orig_res["exact_match"],
    }
    logger.info(f"  Adapted on original: macro_f1={orig_res['macro_f1']}")

    # Model size comparison
    orig_params = sum(p.numel() for p in model.parameters())
    adapted_params = sum(p.numel() for p in adapted_model.parameters())
    results["model_comparison"] = {
        "original_params": orig_params,
        "adapted_params": adapted_params,
        "params_changed": "classifier head only (same total params)",
    }

    return results


# ════════════════════════════════════════════════════════
# 7. MONITORING METRICS
# ════════════════════════════════════════════════════════

def run_monitoring(model, test_dataset, device, data_dir, tokenizer):
    """Compute drift statistics simulating a monitoring scenario."""
    logger.info("\n" + "=" * 60)
    logger.info("7. MONITORING METRICS")
    logger.info("=" * 60)

    results = {}

    # Split test into "reference" (first half) and "current" (second half)
    n = len(test_dataset)
    ref_indices = list(range(0, n // 2))
    cur_indices = list(range(n // 2, n))

    ref_ds = Subset(test_dataset, ref_indices)
    cur_ds = Subset(test_dataset, cur_indices)

    # Evaluate both
    ref_res, ref_probs, ref_labels = evaluate_dataset(model, ref_ds, device)
    cur_res, cur_probs, cur_labels = evaluate_dataset(model, cur_ds, device)

    # PSI (Population Stability Index) for each label's predicted probability
    def compute_psi(ref_probs, cur_probs, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        ref_hist = np.histogram(ref_probs, bins=bins)[0] / len(ref_probs)
        cur_hist = np.histogram(cur_probs, bins=bins)[0] / len(cur_probs)
        # Avoid log(0)
        ref_hist = np.clip(ref_hist, 1e-6, None)
        cur_hist = np.clip(cur_hist, 1e-6, None)
        psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))
        return round(float(psi), 6)

    psi_results = {}
    for i, name in enumerate(LABELS):
        psi = compute_psi(ref_probs[:, i], cur_probs[:, i])
        psi_results[name] = psi
        logger.info(f"  PSI({name}): {psi}")
    results["psi_per_label"] = psi_results

    # F1 comparison (simulating rolling window)
    results["reference_metrics"] = {
        "macro_f1": ref_res["macro_f1"],
        "exact_match": ref_res["exact_match"],
    }
    results["current_metrics"] = {
        "macro_f1": cur_res["macro_f1"],
        "exact_match": cur_res["exact_match"],
    }

    # Label distribution comparison
    ref_label_rates = ref_labels.mean(axis=0).tolist()
    cur_label_rates = cur_labels.mean(axis=0).tolist()
    results["label_rates"] = {
        "reference": {LABELS[i]: round(ref_label_rates[i], 4) for i in range(5)},
        "current": {LABELS[i]: round(cur_label_rates[i], 4) for i in range(5)},
    }

    # Confidence drift
    ref_max_conf = ref_probs.max(axis=1).mean()
    cur_max_conf = cur_probs.max(axis=1).mean()
    results["confidence_drift"] = {
        "reference_mean_max_conf": round(float(ref_max_conf), 4),
        "current_mean_max_conf": round(float(cur_max_conf), 4),
        "drift": round(float(cur_max_conf - ref_max_conf), 4),
    }
    logger.info(f"  Confidence drift: {results['confidence_drift']}")

    return results


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Milestone 3 Experiments")
    parser.add_argument("--data-dir", type=str, default="./training_data")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--tokenizer", type=str, default="./checkpoints/tokenizer.json")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    model, tokenizer, test_dataset, device = load_model_and_data(args)
    logger.info(f"Device: {device}, Test samples: {len(test_dataset)}")

    all_results = {}

    all_results["stress_tests"] = run_stress_tests(model, test_dataset, device, tokenizer)
    all_results["adversarial"] = run_adversarial(model, test_dataset, device, tokenizer)
    all_results["calibration"] = run_calibration(model, test_dataset, device)
    all_results["failures"] = mine_failures(model, test_dataset, device)
    all_results["latency_comparison"] = run_latency_comparison(model, test_dataset, device, tokenizer)
    all_results["drift_adaptation"] = run_drift_adaptation(model, test_dataset, tokenizer, device, args.data_dir)
    all_results["monitoring"] = run_monitoring(model, test_dataset, device, args.data_dir, tokenizer)

    # Save
    out_path = os.path.join(args.data_dir, "milestone3_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("ALL RESULTS JSON")
    print("=" * 60)
    print(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
