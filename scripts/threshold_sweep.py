#!/usr/bin/env python3
"""
threshold_sweep.py — Find optimal classification threshold on val set.

Loads best_model.pt, runs inference on val.jsonl, sweeps thresholds 0.1–0.6,
prints per-label and macro F1/precision/recall at each threshold.

Usage:
    python dataset/threshold_sweep.py \
        --checkpoint checkpoints/best_model.pt \
        --tokenizer checkpoints/tokenizer.json \
        --val-data dataset/training_data_fuzz/val.jsonl
"""

import os
os.environ.setdefault('HSA_OVERRIDE_GFX_VERSION', '10.3.0')
os.environ.setdefault('TORCH_BLAS_PREFER_HIPBLASLT', '0')

import sys
import json
import argparse
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABELS = ["file_access", "network_access", "process_mgmt", "pure_calculation", "code_execution"]
NUM_LABELS = len(LABELS)


# ── Inline minimal model + tokenizer (avoids importing all of train.py) ──

class SyscallTokenizer:
    PAD = 0; UNK = 1; BOS = 2; EOS = 3; SPECIAL_OFFSET = 4

    def __init__(self):
        self.token_to_idx = {}
        self.max_length = 256
        self.vocab_size = 0
        self.fitted = False

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.token_to_idx = data["token_to_idx"]
        self.max_length = data["max_length"]
        self.vocab_size = data["vocab_size"]
        self.fitted = True

    def encode_padded(self, text):
        tokens = [self.BOS]
        for tok in text.lower().split()[:self.max_length - 2]:
            tokens.append(self.token_to_idx.get(tok, self.UNK))
        tokens.append(self.EOS)
        length = len(tokens)
        if len(tokens) < self.max_length:
            tokens += [self.PAD] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
            length = self.max_length
        return tokens, length


class CNNBiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, cnn_filters=64,
                 cnn_kernel_sizes=(3, 5, 7), lstm_hidden=64,
                 lstm_layers=2, dropout=0.3, num_labels=NUM_LABELS):
        super().__init__()
        effective_vocab_size = max(vocab_size, 256)
        self.embedding = nn.Embedding(effective_vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, cnn_filters, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(cnn_filters),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
            )
            for ks in cnn_kernel_sizes
        ])
        cnn_out_dim = cnn_filters * len(cnn_kernel_sizes)
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            bidirectional=True, dropout=dropout if lstm_layers > 1 else 0,
        )
        lstm_out_dim = lstm_hidden * 2
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2), nn.Tanh(),
            nn.Linear(lstm_out_dim // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, lstm_out_dim // 2), nn.ReLU(),
            nn.BatchNorm1d(lstm_out_dim // 2), nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, num_labels),
        )

    def forward(self, input_ids, lengths=None):
        x = self.embedding.weight[input_ids]
        x_cnn = x.permute(0, 2, 1)
        conv_outputs = [conv(x_cnn) for conv in self.convs]
        x = torch.cat(conv_outputs, dim=1).permute(0, 2, 1)
        if lengths is not None:
            lengths_cpu = lengths.cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        if lengths is not None:
            mask = torch.arange(lstm_out.size(1), device=lstm_out.device).unsqueeze(0)
            mask = mask >= lengths.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(mask.unsqueeze(2), float('-inf'))
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)
        return self.classifier(context)


def compute_metrics(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    per_label = {}
    for i, name in enumerate(LABELS):
        tp = ((preds[:, i] == 1) & (labels[:, i] == 1)).sum()
        fp = ((preds[:, i] == 1) & (labels[:, i] == 0)).sum()
        fn = ((preds[:, i] == 0) & (labels[:, i] == 1)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_label[name] = {"p": p, "r": r, "f1": f1, "tp": int(tp), "fp": int(fp), "fn": int(fn)}

    macro_f1 = np.mean([per_label[n]["f1"] for n in LABELS])
    macro_p = np.mean([per_label[n]["p"] for n in LABELS])
    macro_r = np.mean([per_label[n]["r"] for n in LABELS])

    exact = (preds == labels).all(axis=1).mean()
    hamming = (preds != labels).mean()

    return per_label, macro_f1, macro_p, macro_r, exact, hamming


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    parser.add_argument("--val-data", default="dataset/training_data_fuzz/val.jsonl")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load tokenizer
    tokenizer = SyscallTokenizer()
    tokenizer.load(args.tokenizer)
    logger.info(f"Tokenizer vocab_size={tokenizer.vocab_size}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})
    logger.info(f"Checkpoint config: {cfg}")

    model = CNNBiLSTMClassifier(
        vocab_size=cfg.get("vocab_size", tokenizer.vocab_size),
        embed_dim=cfg.get("embed_dim", 64),
        cnn_filters=cfg.get("cnn_filters", 64),
        lstm_hidden=cfg.get("lstm_hidden", 64),
        lstm_layers=cfg.get("lstm_layers", 2),
        dropout=cfg.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Model loaded.")

    # Load val data
    records = []
    with open(args.val_data) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Val records: {len(records)}")

    # Inference
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, len(records), args.batch_size):
            batch = records[i:i + args.batch_size]
            input_ids_list = []
            lengths_list = []
            labels_list = []
            for r in batch:
                ids, length = tokenizer.encode_padded(r.get("text", ""))
                input_ids_list.append(ids)
                lengths_list.append(length)
                labels_list.append(r.get("label_vector", [0] * NUM_LABELS))

            input_ids = torch.tensor(input_ids_list, dtype=torch.long).to(device)
            lengths = torch.tensor(lengths_list, dtype=torch.long).to(device)
            labels_t = torch.tensor(labels_list, dtype=torch.float32)

            logits = model(input_ids, lengths)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels_t.numpy())

            if (i // args.batch_size) % 10 == 0:
                logger.info(f"  Processed {i + len(batch)}/{len(records)}")

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    logger.info(f"Inference done. probs shape: {all_probs.shape}")

    # Threshold sweep
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    print("\n" + "=" * 100)
    print(f"THRESHOLD SWEEP — val set ({len(records)} records)")
    print("=" * 100)
    print(f"{'Thr':>5}  {'MacroF1':>8}  {'MacroP':>8}  {'MacroR':>8}  {'ExactMatch':>10}  {'Hamming':>8}")
    print("-" * 60)

    best_f1 = 0.0
    best_thr = 0.5
    results = []
    for thr in thresholds:
        per_label, mf1, mp, mr, exact, hamming = compute_metrics(all_probs, all_labels, thr)
        results.append((thr, per_label, mf1, mp, mr, exact, hamming))
        marker = " ◄" if mf1 > best_f1 else ""
        print(f"{thr:>5.2f}  {mf1:>8.4f}  {mp:>8.4f}  {mr:>8.4f}  {exact:>10.4f}  {hamming:>8.4f}{marker}")
        if mf1 > best_f1:
            best_f1 = mf1
            best_thr = thr

    print("\n" + "=" * 100)
    print(f"Best threshold: {best_thr:.2f}  (Macro F1 = {best_f1:.4f})")
    print("=" * 100)

    # Per-label breakdown at best threshold
    _, best_per_label, *_ = next(r for r in results if r[0] == best_thr)
    print(f"\nPer-label at threshold={best_thr:.2f}:")
    print(f"{'Label':>20}  {'P':>7}  {'R':>7}  {'F1':>7}  {'TP':>6}  {'FP':>6}  {'FN':>6}")
    print("-" * 68)
    for name in LABELS:
        m = best_per_label[name]
        print(f"{name:>20}  {m['p']:>7.4f}  {m['r']:>7.4f}  {m['f1']:>7.4f}  {m['tp']:>6}  {m['fp']:>6}  {m['fn']:>6}")

    # Also show default 0.5 for comparison
    _, def_per_label, def_mf1, def_mp, def_mr, def_exact, def_hamming = next(r for r in results if r[0] == 0.50)
    print(f"\nComparison — threshold=0.50: Macro F1={def_mf1:.4f}, P={def_mp:.4f}, R={def_mr:.4f}, Exact={def_exact:.4f}")
    print(f"Comparison — threshold={best_thr:.2f}: Macro F1={best_f1:.4f}")

    return best_thr, best_f1


if __name__ == "__main__":
    main()
