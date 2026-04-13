#!/usr/bin/env python3
"""
Dataset Preprocessor for Multi-Label API Classification
=========================================================

Takes the labeled dataset and produces a balanced, split, training-ready output.

Label merging (13 → 5):
  file_read + file_write + dir_read + dir_write    → file_access
  network_send + network_receive                   → network_access
  process_create + process_abort + process_sleep    → process_mgmt
  calc_math + calc_data_mgmt + calc_encode_decode  → pure_calculation
  code_execution                                   → code_execution

Then:
  - Drops low-confidence records from overrepresented classes
  - Undersamples pure_calculation
  - Oversamples rare classes
  - Produces stratified train/val/test split
  - Outputs training-ready JSONL with label vectors

Usage:
    python preprocess_dataset.py \
        --input labeled_dataset.jsonl \
        --output-dir ./training_data \
        --calc-cap 25000 \
        --min-confidence 0.4 \
        --oversample-floor 5000

    # Quick check of what it would do (dry run):
    python preprocess_dataset.py \
        --input labeled_dataset.jsonl \
        --dry-run
"""

import json
import random
import logging
import argparse
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Label Mapping
# =============================================================================

MERGE_MAP = {
    # File access
    "file_read":           "file_access",
    "file_write":          "file_access",
    "dir_read":            "file_access",
    "dir_write":           "file_access",
    # Network access
    "network_send":        "network_access",
    "network_receive":     "network_access",
    # Process management
    "process_create":      "process_mgmt",
    "process_abort":       "process_mgmt",
    "process_sleep":       "process_mgmt",
    # Pure calculation (all three merged)
    "calc_math":           "pure_calculation",
    "calc_data_mgmt":      "pure_calculation",
    "calc_encode_decode":  "pure_calculation",
    # Code execution (unchanged)
    "code_execution":      "code_execution",
}

MERGED_LABELS = [
    "file_access",
    "network_access",
    "process_mgmt",
    "pure_calculation",
    "code_execution",
]


def merge_labels(original_labels: list[str]) -> list[str]:
    """Map 13 fine-grained labels → 5 merged labels."""
    merged = set()
    for label in original_labels:
        mapped = MERGE_MAP.get(label)
        if mapped:
            merged.add(mapped)
    return sorted(merged)


def label_vector(labels: list[str]) -> list[int]:
    """Convert label list to binary vector."""
    return [1 if l in labels else 0 for l in MERGED_LABELS]


# =============================================================================
# Preprocessing Pipeline
# =============================================================================

class DatasetPreprocessor:

    def __init__(self, args):
        self.args = args
        self.rng = random.Random(args.seed)

    def run(self):
        logger.info("=" * 60)
        logger.info("DATASET PREPROCESSING PIPELINE")
        logger.info("=" * 60)

        # Step 1: Load
        records = self._load(self.args.input)

        # Step 2: Merge labels
        records = self._merge_labels(records)

        # Step 3: Print initial distribution
        logger.info("\nAFTER MERGE (before balancing):")
        self._print_distribution(records)

        # Step 4: Drop low confidence from overrepresented classes
        records = self._drop_low_confidence(records)

        # Step 5: Remove synthetic from val/test candidates
        real_records = [r for r in records if not r.get("is_synthetic", False)]
        synthetic_records = [r for r in records if r.get("is_synthetic", False)]
        logger.info(f"\nReal records: {len(real_records):,}, Synthetic: {len(synthetic_records):,}")

        # Step 6: Undersample pure_calculation
        real_records = self._undersample(real_records, "pure_calculation", self.args.calc_cap)

        # Step 7: Oversample rare classes
        real_records = self._oversample(real_records, self.args.oversample_floor)

        # Step 8: Print balanced distribution
        logger.info("\nAFTER BALANCING:")
        self._print_distribution(real_records)

        if self.args.dry_run:
            logger.info("\n[DRY RUN] Stopping here. No files written.")
            return

        # Step 9: Split into train/val/test
        train, val, test = self._stratified_split(
            real_records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
        )

        # Step 10: Add synthetic data to train only
        train.extend(synthetic_records)
        self.rng.shuffle(train)

        logger.info(f"\nFINAL SPLIT:")
        logger.info(f"  Train: {len(train):,}")
        logger.info(f"  Val:   {len(val):,}")
        logger.info(f"  Test:  {len(test):,}")

        logger.info("\n  Train distribution:")
        self._print_distribution(train, indent=4)
        logger.info("\n  Val distribution:")
        self._print_distribution(val, indent=4)
        logger.info("\n  Test distribution:")
        self._print_distribution(test, indent=4)

        # Step 11: Write outputs
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._write(train, output_dir / "train.jsonl")
        self._write(val, output_dir / "val.jsonl")
        self._write(test, output_dir / "test.jsonl")

        # Write metadata
        self._write_metadata(output_dir, train, val, test)

        logger.info(f"\nAll files written to {output_dir}/")

    # ── Loading ──

    def _load(self, path: str) -> list[dict]:
        records = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        logger.info(f"Loaded {len(records):,} records from {path}")
        return records

    # ── Label Merging ──

    def _merge_labels(self, records: list[dict]) -> list[dict]:
        for r in records:
            original = r.get("labels", r.get("assigned_labels", []))
            r["original_labels"] = original
            r["labels"] = merge_labels(original)
            r["label_vector"] = label_vector(r["labels"])
            # Primary label = for stratification (use the rarest label in the record)
            r["primary_label"] = self._primary_label(r["labels"])
        logger.info(f"Merged 13 labels → 5 labels")
        return records

    def _primary_label(self, labels: list[str]) -> str:
        """Pick the rarest label as primary (for stratification)."""
        if not labels:
            return "__none__"
        # Priority: rarer labels matter more for stratification
        priority = {
            "file_access": 0,
            "network_access": 1,
            "process_mgmt": 2,
            "code_execution": 3,
            "pure_calculation": 4,
        }
        return min(labels, key=lambda l: priority.get(l, 99))

    # ── Confidence Filtering ──

    def _drop_low_confidence(self, records: list[dict]) -> list[dict]:
        """Drop low confidence records, but only from overrepresented classes."""
        threshold = self.args.min_confidence

        # Count current distribution
        label_counts = Counter()
        for r in records:
            for l in r["labels"]:
                label_counts[l] += 1

        median_count = sorted(label_counts.values())[len(label_counts) // 2] if label_counts else 0

        kept = []
        dropped = 0
        for r in records:
            conf = r.get("label_confidence", r.get("confidence", 1.0))

            if conf >= threshold:
                kept.append(r)
                continue

            # Low confidence — keep if it has a rare label
            has_rare = any(label_counts.get(l, 0) < median_count for l in r["labels"])
            if has_rare:
                kept.append(r)
            else:
                dropped += 1

        logger.info(f"Dropped {dropped:,} low-confidence records (threshold={threshold}, only from overrepresented classes)")
        return kept

    # ── Undersampling ──

    def _undersample(self, records: list[dict], target_label: str, cap: int) -> list[dict]:
        """Cap the number of records where the primary label is target_label."""
        target_records = []
        other_records = []

        for r in records:
            # A record is "primarily target" if it ONLY has target_label
            if r["labels"] == [target_label]:
                target_records.append(r)
            else:
                other_records.append(r)

        before = len(target_records)
        if len(target_records) > cap:
            self.rng.shuffle(target_records)
            target_records = target_records[:cap]

        logger.info(f"Undersampled '{target_label}' (single-label): {before:,} → {len(target_records):,}")

        return other_records + target_records

    # ── Oversampling ──

    def _oversample(self, records: list[dict], floor: int) -> list[dict]:
        """Duplicate records from underrepresented classes up to floor."""
        # Count by primary label
        by_primary = defaultdict(list)
        for r in records:
            by_primary[r["primary_label"]].append(r)

        augmented = []
        for label, group in by_primary.items():
            if label == "__none__":
                augmented.extend(group)
                continue

            if len(group) < floor:
                # Oversample by duplicating
                extra_needed = floor - len(group)
                extras = [self.rng.choice(group).copy() for _ in range(extra_needed)]
                for e in extras:
                    e["is_oversampled"] = True
                logger.info(f"Oversampled '{label}': {len(group):,} → {len(group) + len(extras):,} (+{len(extras):,})")
                augmented.extend(group)
                augmented.extend(extras)
            else:
                augmented.extend(group)

        return augmented

    # ── Splitting ──

    def _stratified_split(self, records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """Stratified split by primary_label, ensuring all labels in all splits."""
        by_label = defaultdict(list)
        for r in records:
            by_label[r["primary_label"]].append(r)

        train, val, test = [], [], []

        for label, group in by_label.items():
            self.rng.shuffle(group)
            n = len(group)
            n_val = max(1, int(n * val_ratio))
            n_test = max(1, int(n * test_ratio))
            n_train = n - n_val - n_test

            # Don't put oversampled records in val/test
            real_in_group = [r for r in group if not r.get("is_oversampled")]
            oversampled_in_group = [r for r in group if r.get("is_oversampled")]

            if len(real_in_group) < n_val + n_test + 2:
                # Too few real records — just split what we have
                split_val = real_in_group[:n_val]
                split_test = real_in_group[n_val:n_val + n_test]
                split_train = real_in_group[n_val + n_test:] + oversampled_in_group
            else:
                split_val = real_in_group[:n_val]
                split_test = real_in_group[n_val:n_val + n_test]
                split_train = real_in_group[n_val + n_test:] + oversampled_in_group

            val.extend(split_val)
            test.extend(split_test)
            train.extend(split_train)

        self.rng.shuffle(train)
        self.rng.shuffle(val)
        self.rng.shuffle(test)

        return train, val, test

    # ── Output ──

    def _write(self, records: list[dict], path: Path):
        """Write records as training-ready JSONL."""
        syscall_field = self.args.syscall_field
        use_syscalls = self.args.input_mode == "syscall"

        with open(str(path), "w", encoding="utf-8") as f:
            for r in records:
                if use_syscalls:
                    raw = r.get(syscall_field, r.get("system_calls", None))
                    if isinstance(raw, list):
                        text = " ".join(str(s) for s in raw)
                    elif isinstance(raw, str) and raw:
                        text = raw
                    else:
                        # Fallback: no syscall data found, use code context
                        text = r.get("code_context", r.get("call_expression", ""))
                else:
                    text = r.get("code_context", r.get("call_expression", ""))

                out = {
                    "text": text,
                    "api_path": r.get("api_path", ""),
                    "framework": r.get("framework", ""),
                    "labels": r["labels"],
                    "label_vector": r["label_vector"],
                    "original_labels": r.get("original_labels", []),
                    "is_synthetic": r.get("is_synthetic", False),
                    "is_oversampled": r.get("is_oversampled", False),
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(records):,} records to {path}")

    def _write_metadata(self, output_dir: Path, train, val, test):
        """Write dataset metadata for training scripts."""
        def dist(records):
            counts = Counter()
            for r in records:
                for l in r["labels"]:
                    counts[l] += 1
            return dict(sorted(counts.items(), key=lambda x: -x[1]))

        # Compute class weights from training set
        train_dist = dist(train)
        total = sum(train_dist.values())
        n_classes = len(MERGED_LABELS)
        weights = {}
        for label in MERGED_LABELS:
            count = train_dist.get(label, 1)
            weights[label] = round(total / (n_classes * count), 4)

        metadata = {
            "label_schema": {
                "labels": MERGED_LABELS,
                "num_labels": len(MERGED_LABELS),
                "merge_map": MERGE_MAP,
            },
            "splits": {
                "train": {
                    "count": len(train),
                    "distribution": dist(train),
                    "synthetic_count": sum(1 for r in train if r.get("is_synthetic")),
                    "oversampled_count": sum(1 for r in train if r.get("is_oversampled")),
                },
                "val": {
                    "count": len(val),
                    "distribution": dist(val),
                },
                "test": {
                    "count": len(test),
                    "distribution": dist(test),
                },
            },
            "class_weights": weights,
            "class_weights_tensor_order": [weights[l] for l in MERGED_LABELS],
            "training_notes": {
                "loss": "Binary Cross-Entropy (multi-label)",
                "weight_usage": "Multiply each label's BCE loss by its class weight",
                "input_mode": self.args.input_mode,
                "input_field": (
                    f"text (syscall sequence from '{self.args.syscall_field}' field, space-separated syscall names)"
                    if self.args.input_mode == "syscall"
                    else "text (code context string)"
                ),
                "recommended_tokenizer": (
                    "syscall (word-level, one token per syscall name)"
                    if self.args.input_mode == "syscall"
                    else "char (character-level)"
                ),
                "output_field": "label_vector (5-dim binary vector)",
                "label_order": MERGED_LABELS,
                "threshold": "Use 0.5 for predictions, tune on val set",
                "metrics": "Per-label F1 + macro F1 + exact match ratio",
            },
            "pytorch_example": {
                "loss_setup": (
                    "weights = torch.tensor(class_weights_tensor_order)\n"
                    "criterion = nn.BCEWithLogitsLoss(pos_weight=weights)"
                ),
                "prediction": (
                    "probs = torch.sigmoid(logits)\n"
                    "preds = (probs > 0.5).int()"
                ),
            },
            "tensorflow_example": {
                "loss_setup": (
                    "# In model.compile:\n"
                    "# Use custom weighted BCE or sample_weight\n"
                    "loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)"
                ),
            },
        }

        meta_path = output_dir / "metadata.json"
        with open(str(meta_path), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote metadata to {meta_path}")

    # ── Utilities ──

    def _print_distribution(self, records, indent=2):
        counts = Counter()
        multi_label = 0
        for r in records:
            for l in r["labels"]:
                counts[l] += 1
            if len(r["labels"]) > 1:
                multi_label += 1

        total = len(records)
        prefix = " " * indent
        for label in MERGED_LABELS:
            count = counts.get(label, 0)
            pct = count / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 2)
            logger.info(f"{prefix}{label:>18}: {count:>8,}  ({pct:5.1f}%)  {bar}")

        logger.info(f"{prefix}{'TOTAL':>18}: {total:>8,}")
        logger.info(f"{prefix}{'multi-label':>18}: {multi_label:>8,}  ({multi_label/max(total,1)*100:.1f}%)")

        # Show imbalance ratio
        vals = [c for c in counts.values() if c > 0]
        if vals:
            ratio = max(vals) / min(vals)
            logger.info(f"{prefix}{'imbalance ratio':>18}: {ratio:.1f}x")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess labeled dataset for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Label merging (13 → 5):
  file_read + file_write + dir_read + dir_write   → file_access
  network_send + network_receive                   → network_access
  process_create + process_abort + process_sleep   → process_mgmt
  calc_math + calc_data_mgmt + calc_encode_decode  → pure_calculation
  code_execution                                   → code_execution

Examples:
  # Standard run:
  python preprocess_dataset.py \\
      --input labeled_dataset.jsonl \\
      --output-dir ./training_data

  # Aggressive undersampling:
  python preprocess_dataset.py \\
      --input labeled_dataset.jsonl \\
      --output-dir ./training_data \\
      --calc-cap 15000

  # Preview without writing:
  python preprocess_dataset.py \\
      --input labeled_dataset.jsonl \\
      --dry-run
        """,
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input labeled JSONL (from api_behavior_labeler.py)")
    parser.add_argument("--output-dir", type=str, default="./training_data",
                        help="Output directory for train/val/test splits (default: ./training_data)")
    parser.add_argument("--calc-cap", type=int, default=25000,
                        help="Max pure_calculation single-label records (default: 25000)")
    parser.add_argument("--min-confidence", type=float, default=0.4,
                        help="Drop records below this confidence from overrepresented classes (default: 0.4)")
    parser.add_argument("--oversample-floor", type=int, default=5000,
                        help="Minimum records per primary label after oversampling (default: 5000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show distribution changes without writing files")
    parser.add_argument("--input-mode", type=str, choices=["text", "syscall"], default="text",
                        help="Input feature: 'text' uses code_context, 'syscall' uses the syscall sequence field (default: text)")
    parser.add_argument("--syscall-field", type=str, default="syscalls",
                        help="Field name in source data that contains the syscall list/string (default: syscalls)")

    args = parser.parse_args()
    DatasetPreprocessor(args).run()


if __name__ == "__main__":
    main()
