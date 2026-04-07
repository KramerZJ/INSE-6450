#!/usr/bin/env python3
"""
apply_strace_syscalls.py — Apply strace-derived syscalls to labeled_dataset.jsonl.

Reads labeled_dataset.jsonl record by record (streaming, no full load),
replaces the 'syscalls' field with real strace-observed syscalls from
strace_syscall_map.json, and writes labeled_dataset_strace.jsonl.

Usage:
    python dataset/apply_strace_syscalls.py \\
        --input labeled_dataset.jsonl \\
        --output labeled_dataset_strace.jsonl \\
        --strace-map dataset/strace_syscall_map.json

    # Preview first 10 records without writing:
    python dataset/apply_strace_syscalls.py ... --dry-run 10
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Label merge map (mirrors preprocess_dataset.py) ──────────────────────────
# Maps 13 fine-grained labels → 5 merged behavioral categories.

MERGE_MAP: dict[str, str] = {
    "file_read":          "file_access",
    "file_write":         "file_access",
    "dir_read":           "file_access",
    "dir_write":          "file_access",
    "network_send":       "network_access",
    "network_receive":    "network_access",
    "process_create":     "process_mgmt",
    "process_abort":      "process_mgmt",
    "process_sleep":      "process_mgmt",
    "calc_math":          "pure_calculation",
    "calc_data_mgmt":     "pure_calculation",
    "calc_encode_decode": "pure_calculation",
    "code_execution":     "code_execution",
}

# Priority order for merging multi-label syscall lists
MERGED_LABELS: list[str] = [
    "file_access",
    "network_access",
    "process_mgmt",
    "pure_calculation",
    "code_execution",
]

# Fallback if strace map missing a (framework, category) entry
FALLBACK_MAP: dict[str, list[str]] = {
    "file_access":      ["openat", "read", "write", "close"],
    "network_access":   ["socket", "connect", "sendto", "recvfrom"],
    "process_mgmt":     ["clone", "execve", "wait4"],
    "pure_calculation": ["mmap", "munmap"],
    "code_execution":   ["execve", "clone3", "mmap"],
}

# Framework name normalization (dataset uses 'tensorflow', not 'tf')
FW_ALIASES: dict[str, str] = {
    "tensorflow": "tensorflow",
    "tf":         "tensorflow",
    "pytorch":    "pytorch",
    "torch":      "pytorch",
    "jax":        "jax",
}


def load_strace_map(path: str) -> dict[str, dict[str, list[str]]]:
    """Load strace_syscall_map.json. Returns {framework: {category: [syscalls]}}."""
    with open(path) as f:
        raw = json.load(f)
    # Strip metadata key
    strace_map = {k: v for k, v in raw.items() if not k.startswith("_")}
    logger.info(f"Loaded strace map from {path}")
    for fw, cats in strace_map.items():
        for cat, syscalls in cats.items():
            logger.info(f"  {fw:12s} | {cat:20s} | {syscalls}")
    return strace_map


def derive_strace_syscalls(
    original_labels: list[str],
    framework: str,
    strace_map: dict[str, dict[str, list[str]]],
) -> list[str]:
    """
    Compute strace-based syscall list for a record.

    1. Map fine-grained labels → merged categories via MERGE_MAP
    2. For each merged category (in MERGED_LABELS priority order):
       extend result with strace_map[framework][category], deduplicating
    3. Fall back to FALLBACK_MAP if (framework, category) not in strace_map

    Returns an ordered, deduplicated list of syscall names.
    """
    fw = FW_ALIASES.get(framework, framework)
    fw_map = strace_map.get(fw, {})

    # Step 1: resolve merged categories
    merged: set[str] = set()
    for label in original_labels:
        mc = MERGE_MAP.get(label)
        if mc:
            merged.add(mc)

    # Step 2: union syscalls in priority order
    seen: set[str] = set()
    result: list[str] = []
    for category in MERGED_LABELS:
        if category not in merged:
            continue
        syscalls = fw_map.get(category) or FALLBACK_MAP.get(category, [])
        for sc in syscalls:
            if sc not in seen:
                seen.add(sc)
                result.append(sc)

    return result


def process_dataset(
    input_path: str,
    output_path: str,
    strace_map: dict[str, dict[str, list[str]]],
    dry_run: int = 0,
) -> None:
    """
    Stream labeled_dataset.jsonl line-by-line, replace 'syscalls', write output.

    Args:
        input_path:  Path to labeled_dataset.jsonl
        output_path: Path to write labeled_dataset_strace.jsonl
        strace_map:  Loaded strace syscall map
        dry_run:     If > 0, preview that many records and exit without writing
    """
    stats: dict[str, int] = defaultdict(int)
    fallback_combos: set[tuple[str, str]] = set()
    sample_records: list[dict] = []

    logger.info(f"\nProcessing: {input_path}")
    if dry_run:
        logger.info(f"DRY RUN — showing first {dry_run} records")

    out_f = None if dry_run else open(output_path, "w")

    try:
        with open(input_path) as in_f:
            for i, line in enumerate(in_f):
                line = line.strip()
                if not line:
                    continue

                rec = json.loads(line)
                stats["total"] += 1

                fw = rec.get("framework", "unknown")
                orig_labels = rec.get("labels", [])
                old_syscalls = rec.get("syscalls", [])

                new_syscalls = derive_strace_syscalls(orig_labels, fw, strace_map)

                # Track fallback usage
                for label in orig_labels:
                    mc = MERGE_MAP.get(label)
                    if mc:
                        fw_norm = FW_ALIASES.get(fw, fw)
                        if mc not in strace_map.get(fw_norm, {}):
                            fallback_combos.add((fw_norm, mc))

                rec["syscalls"] = new_syscalls
                rec["syscalls_source"] = "strace"

                stats[f"fw_{fw}"] += 1

                if dry_run:
                    sample_records.append({
                        "api_path": rec.get("api_path", ""),
                        "framework": fw,
                        "labels": orig_labels,
                        "old_syscalls": old_syscalls,
                        "new_syscalls": new_syscalls,
                    })
                    if len(sample_records) >= dry_run:
                        break
                else:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if stats["total"] % 50_000 == 0:
                    logger.info(f"  Processed {stats['total']:,} records...")

    finally:
        if out_f:
            out_f.close()

    # ── Report ────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("APPLY COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Records processed: {stats['total']:,}")
    for k, v in sorted(stats.items()):
        if k.startswith("fw_"):
            logger.info(f"  {k[3:]:12s}: {v:,}")

    if fallback_combos:
        logger.warning(f"\n  Fallback used for {len(fallback_combos)} (framework, category) pairs:")
        for combo in sorted(fallback_combos):
            logger.warning(f"    {combo}")
    else:
        logger.info("\n  No fallbacks needed — all (framework, category) pairs covered.")

    if dry_run:
        logger.info(f"\n{'─'*60}")
        logger.info("SAMPLE RECORDS (dry run)")
        logger.info(f"{'─'*60}")
        for rec in sample_records:
            print(json.dumps(rec, indent=2))
    else:
        logger.info(f"\n  Output written to: {output_path}")
        out_size = os.path.getsize(output_path) / 1024**2
        logger.info(f"  Output size: {out_size:.1f} MB")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Apply strace-derived syscalls to labeled_dataset.jsonl"
    )
    parser.add_argument(
        "--input",
        default="labeled_dataset.jsonl",
        help="Input labeled dataset (default: labeled_dataset.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="labeled_dataset_strace.jsonl",
        help="Output path (default: labeled_dataset_strace.jsonl)",
    )
    parser.add_argument(
        "--strace-map",
        default="dataset/strace_syscall_map.json",
        help="Path to strace_syscall_map.json (default: dataset/strace_syscall_map.json)",
    )
    parser.add_argument(
        "--dry-run",
        type=int,
        default=0,
        metavar="N",
        help="Preview N records without writing output (default: 0 = disabled)",
    )
    args = parser.parse_args()

    # Resolve paths
    input_path = os.path.expanduser(args.input)
    output_path = os.path.expanduser(args.output)
    map_path = os.path.expanduser(args.strace_map)

    for p, name in [(input_path, "--input"), (map_path, "--strace-map")]:
        if not os.path.exists(p):
            logger.error(f"{name} not found: {p}")
            sys.exit(1)

    strace_map = load_strace_map(map_path)
    process_dataset(input_path, output_path, strace_map, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
