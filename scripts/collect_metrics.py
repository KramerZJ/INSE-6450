#!/usr/bin/env python3
"""
Milestone 2 Metrics Collector
Run this and paste the FULL output back to Claude.

Usage:
    python collect_metrics.py \
        --data-dir ./training_data \
        --checkpoint ./checkpoints/best_model.pt \
        --tokenizer ./checkpoints/tokenizer.json
"""

import os
import sys
import json
import time
import platform
import argparse
import numpy as np

import torch
import torch.nn as nn

# ── Import model and tokenizer from train.py ──
# We'll redefine minimally to avoid import issues
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import CNNBiLSTMClassifier, CodeTokenizer, SyscallTokenizer, load_tokenizer, APICallDataset, MultiLabelMetrics, run_baseline

LABELS = ["file_access", "network_access", "process_mgmt", "pure_calculation", "code_execution"]


def collect_all(args):
    results = {}

    # ════════════════════════════════════════
    # 1. HARDWARE INFO
    # ════════════════════════════════════════
    print("=" * 60)
    print("1. HARDWARE INFO")
    print("=" * 60)

    hw = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu": "unknown",
        "ram_gb": "unknown",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    # CPU info
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    hw["cpu"] = line.split(":")[1].strip()
                    break
    except:
        hw["cpu"] = platform.processor() or "unknown"

    # RAM
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if "MemTotal" in line:
                    kb = int(line.split()[1])
                    hw["ram_gb"] = round(kb / 1024 / 1024, 1)
                    break
    except:
        pass

    # GPU info
    if torch.cuda.is_available():
        hw["gpu_name"] = torch.cuda.get_device_name(0)
        hw["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        hw["hip_version"] = getattr(torch.version, "hip", None)
    else:
        hw["gpu_name"] = "N/A (CPU only)"
        hw["gpu_vram_gb"] = "N/A"

    results["hardware"] = hw
    for k, v in hw.items():
        print(f"  {k}: {v}")

    # ════════════════════════════════════════
    # 2. MODEL SIZE
    # ════════════════════════════════════════
    print("\n" + "=" * 60)
    print("2. MODEL SIZE")
    print("=" * 60)

    ckpt_path = args.checkpoint
    ckpt_size_mb = os.path.getsize(ckpt_path) / 1024 / 1024
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Load tokenizer (auto-detect type from saved file)
    with open(args.tokenizer, "r") as _f:
        _saved_tok = json.load(_f)
    tokenizer = load_tokenizer(args.tokenizer, _saved_tok.get("type", "char"), max_length=512)

    # Load model
    model = CNNBiLSTMClassifier(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024

    model_info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "param_size_mb": round(param_size_mb, 2),
        "checkpoint_size_mb": round(ckpt_size_mb, 2),
        "training_epoch": checkpoint.get("epoch", "?"),
        "best_f1": checkpoint.get("best_f1", "?"),
    }
    results["model_size"] = model_info
    for k, v in model_info.items():
        print(f"  {k}: {v}")

    # ════════════════════════════════════════
    # 3. FLOPS ESTIMATION
    # ════════════════════════════════════════
    print("\n" + "=" * 60)
    print("3. FLOPS ESTIMATION")
    print("=" * 60)

    # Analytical FLOPS estimate for one forward pass
    seq_len = 512
    embed_dim = 64
    cnn_filters = 128
    n_kernels = 3
    kernel_sizes = [3, 5, 7]
    lstm_hidden = 128
    lstm_layers = 2
    num_labels = 5

    # Embedding: lookup (negligible)
    # CNN: 3 conv layers
    cnn_flops = 0
    for ks in kernel_sizes:
        # Conv1d: 2 * out_channels * in_channels * kernel_size * output_length
        cnn_flops += 2 * cnn_filters * embed_dim * ks * seq_len

    # BiLSTM: 2 directions * layers * seq_len * (8 * hidden * input_size + 8 * hidden^2)
    lstm_input = cnn_filters * n_kernels  # 384
    lstm_flops = 0
    for layer in range(lstm_layers):
        inp = lstm_input if layer == 0 else lstm_hidden * 2
        # Each LSTM cell: 4 gates, each is (inp + hidden) * hidden multiply-adds
        per_step = 2 * 4 * (inp + lstm_hidden) * lstm_hidden  # *2 for bidirectional
        lstm_flops += seq_len * per_step

    # Attention: linear layers
    attn_flops = 2 * seq_len * (lstm_hidden * 2) * (lstm_hidden) + 2 * seq_len * lstm_hidden

    # Classifier: FC layers
    fc_flops = 2 * (lstm_hidden * 2) * (lstm_hidden) + 2 * lstm_hidden * num_labels

    total_flops = cnn_flops + lstm_flops + attn_flops + fc_flops
    gflops = total_flops / 1e9

    flops_info = {
        "cnn_flops": f"{cnn_flops:,}",
        "lstm_flops": f"{lstm_flops:,}",
        "attention_flops": f"{attn_flops:,}",
        "classifier_flops": f"{fc_flops:,}",
        "total_flops": f"{total_flops:,}",
        "total_gflops": round(gflops, 4),
    }
    results["flops"] = flops_info
    for k, v in flops_info.items():
        print(f"  {k}: {v}")

    # ════════════════════════════════════════
    # 4. INFERENCE BENCHMARKS
    # ════════════════════════════════════════
    print("\n" + "=" * 60)
    print("4. INFERENCE BENCHMARKS")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load test data
    test_dataset = APICallDataset(os.path.join(args.data_dir, "test.jsonl"), tokenizer)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    batch_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Warmup
    print("  Warming up...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 20:
                break
            _ = model(batch["input_ids"].to(device), batch["length"].to(device))

    # Single-sample latency (measure 500 samples)
    print("  Measuring single-sample latency (500 samples)...")
    latencies = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 500:
                break
            start = time.perf_counter()
            _ = model(batch["input_ids"].to(device), batch["length"].to(device))
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

    latencies_arr = np.array(latencies)
    p50 = np.percentile(latencies_arr, 50)
    p90 = np.percentile(latencies_arr, 90)
    p95 = np.percentile(latencies_arr, 95)
    p99 = np.percentile(latencies_arr, 99)

    # Batch throughput
    print("  Measuring batch throughput (batch_size=64)...")
    total_samples = 0
    batch_start = time.perf_counter()
    batch_latencies = []
    with torch.no_grad():
        for batch in batch_loader:
            bs = batch["input_ids"].size(0)
            t0 = time.perf_counter()
            _ = model(batch["input_ids"].to(device), batch["length"].to(device))
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            batch_latencies.append((t1 - t0) * 1000)
            total_samples += bs
    batch_total_time = time.perf_counter() - batch_start
    throughput = total_samples / batch_total_time

    # Memory at inference
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for i, batch in enumerate(batch_loader):
                if i >= 5:
                    break
                _ = model(batch["input_ids"].to(device), batch["length"].to(device))
        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        inference_mem_info = f"{peak_mem:.1f} MB (GPU)"
    else:
        import resource
        peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB to MB
        inference_mem_info = f"{peak_mem:.1f} MB (RSS, CPU)"

    inference_info = {
        "device": str(device),
        "single_sample_p50_ms": round(p50, 2),
        "single_sample_p90_ms": round(p90, 2),
        "single_sample_p95_ms": round(p95, 2),
        "single_sample_p99_ms": round(p99, 2),
        "batch64_avg_latency_ms": round(np.mean(batch_latencies), 2),
        "throughput_samples_per_sec": round(throughput, 1),
        "total_test_samples": total_samples,
        "total_inference_time_sec": round(batch_total_time, 2),
        "peak_inference_memory": inference_mem_info,
    }
    results["inference"] = inference_info
    for k, v in inference_info.items():
        print(f"  {k}: {v}")

    # ════════════════════════════════════════
    # 5. AUROC & PR-AUC (required by rubric)
    # ════════════════════════════════════════
    print("\n" + "=" * 60)
    print("5. AUROC & PR-AUC")
    print("=" * 60)

    from sklearn.metrics import roc_auc_score, average_precision_score

    all_probs = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in batch_loader:
            logits = model(batch["input_ids"].to(device), batch["length"].to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch["labels"].numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    auc_results = {}
    for i, name in enumerate(LABELS):
        if all_labels[:, i].sum() > 0:
            auroc = roc_auc_score(all_labels[:, i], all_probs[:, i])
            prauc = average_precision_score(all_labels[:, i], all_probs[:, i])
            auc_results[name] = {"auroc": round(auroc, 4), "pr_auc": round(prauc, 4)}
            print(f"  {name:>20s}: AUROC={auroc:.4f}  PR-AUC={prauc:.4f}")

    # Macro averages
    aurocs = [v["auroc"] for v in auc_results.values()]
    praucs = [v["pr_auc"] for v in auc_results.values()]
    auc_results["macro_auroc"] = round(np.mean(aurocs), 4)
    auc_results["macro_pr_auc"] = round(np.mean(praucs), 4)
    print(f"  {'macro_auroc':>20s}: {auc_results['macro_auroc']:.4f}")
    print(f"  {'macro_pr_auc':>20s}: {auc_results['macro_pr_auc']:.4f}")

    results["auc"] = auc_results

    # ════════════════════════════════════════
    # 6. BASELINE COMPARISON
    # ════════════════════════════════════════
    print("\n" + "=" * 60)
    print("6. BASELINE: TF-IDF + Logistic Regression")
    print("=" * 60)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.metrics import f1_score, accuracy_score

        def load_split(path):
            texts, labels = [], []
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    r = json.loads(line.strip())
                    texts.append(r.get("text", ""))
                    labels.append(r.get("label_vector", [0] * 5))
            return texts, np.array(labels)

        train_texts, train_labels = load_split(os.path.join(args.data_dir, "train.jsonl"))
        test_texts, test_labels = load_split(os.path.join(args.data_dir, "test.jsonl"))

        print("  Fitting TF-IDF...")
        tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), max_features=50000, sublinear_tf=True)
        X_train = tfidf.fit_transform(train_texts)
        X_test = tfidf.transform(test_texts)

        print("  Training Logistic Regression...")
        t0 = time.perf_counter()
        clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced'), n_jobs=-1)
        clf.fit(X_train, train_labels)
        baseline_train_time = time.perf_counter() - t0

        preds = clf.predict(X_test)
        macro_f1 = f1_score(test_labels, preds, average='macro', zero_division=0)
        exact = (preds == test_labels).all(axis=1).mean()
        hamming = (preds != test_labels).mean()

        # Per-label F1
        per_label = {}
        for i, name in enumerate(LABELS):
            f1 = f1_score(test_labels[:, i], preds[:, i], zero_division=0)
            per_label[name] = round(f1, 4)

        baseline_info = {
            "macro_f1": round(macro_f1, 4),
            "exact_match": round(exact, 4),
            "hamming_loss": round(hamming, 4),
            "per_label_f1": per_label,
            "train_time_sec": round(baseline_train_time, 1),
        }
        results["baseline"] = baseline_info
        print(f"  macro_f1: {macro_f1:.4f}")
        print(f"  exact_match: {exact:.4f}")
        print(f"  hamming_loss: {hamming:.4f}")
        print(f"  train_time: {baseline_train_time:.1f}s")
        for name, f1 in per_label.items():
            print(f"    {name:>20s}: F1={f1:.4f}")

    except Exception as e:
        print(f"  Baseline failed: {e}")
        results["baseline"] = {"error": str(e)}

    # ════════════════════════════════════════
    # 7. TRAINING LOG (if checkpoint has it)
    # ════════════════════════════════════════
    print("\n" + "=" * 60)
    print("7. CHECKPOINT INFO")
    print("=" * 60)

    ckpt_info = {
        "epoch": checkpoint.get("epoch", "?"),
        "best_f1": checkpoint.get("best_f1", "?"),
        "results": checkpoint.get("results", {}),
    }
    results["checkpoint"] = ckpt_info
    for k, v in ckpt_info.items():
        print(f"  {k}: {v}")

    # ════════════════════════════════════════
    # DUMP ALL RESULTS
    # ════════════════════════════════════════
    print("\n" + "=" * 60)
    print("ALL RESULTS JSON")
    print("=" * 60)
    print(json.dumps(results, indent=2, default=str))

    # Save to file too
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "milestone2_model", "outputs", "milestone2_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect Milestone 2 metrics")
    parser.add_argument("--data-dir", type=str, default="./data/splits/training_data_fuzz")
    parser.add_argument("--checkpoint", type=str, default="../milestone2_model/checkpoints/best_model.pt")
    parser.add_argument("--tokenizer", type=str, default="../milestone2_model/checkpoints/tokenizer.json")
    args = parser.parse_args()
    collect_all(args)

if __name__ == "__main__":
    main()
