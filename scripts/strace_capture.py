#!/usr/bin/env python3
"""
strace_capture.py — Capture real syscalls from framework workloads using strace.

For each (framework, category) pair:
  1. Run baseline (framework import only) under strace -f -c
  2. Run workload under strace -f -c
  3. Compute delta: syscalls NEW or significantly increased vs baseline
  4. Store in strace_syscall_map.json

Usage:
    python dataset/strace_capture.py \\
        --env-python /home/kramer/miniconda3/envs/strace_env/bin/python \\
        --output dataset/strace_syscall_map.json \\
        --timeout 45

    # Preview workloads without running:
    python dataset/strace_capture.py --env-python ... --dry-run
"""

import os
import sys
import re
import json
import time
import shutil
import tempfile
import argparse
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from strace_workloads import FRAMEWORKS, CATEGORIES, get_baseline, get_workload

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Syscall noise filter ──────────────────────────────────────────────────────
# These appear in massive counts during Python interpreter startup and import.
# Their baseline counts are so high that delta-based detection is unreliable.
# We exclude them from the final signature (they carry no label-specific signal).
NOISE_SYSCALLS: frozenset = frozenset({
    "newfstatat", "fstat", "stat", "lstat",   # metadata on every .so/.pyc
    "getdents64",                               # directory listing during import
    "lseek",                                    # .pyc seek
    "getcwd",                                   # cwd check
    "ioctl",                                    # terminal probing
    "fcntl",                                    # file flags
    "pread64",                                  # .pyc content reads
    "brk",                                      # heap growth (always present)
    "set_robust_list",                          # thread setup
    "set_tid_address",                          # thread setup
    "arch_prctl",                               # CPU feature detection
    "rseq",                                     # restartable sequences
    "getrandom",                                # entropy (always)
    "gettid",                                   # thread id
    "getpid",                                   # process id
    "getuid", "getgid", "geteuid", "getegid",  # identity
})

# Fallback if a workload completely fails to produce signal
FALLBACK_MAP: dict[str, list[str]] = {
    "file_access":      ["openat", "read", "write", "close", "unlink"],
    "network_access":   ["socket", "connect", "bind", "listen", "accept4", "sendto", "recvfrom"],
    "process_mgmt":     ["clone3", "clone", "execve", "wait4", "pipe2"],
    "pure_calculation": ["mmap", "munmap", "mprotect"],
    "code_execution":   ["execve", "clone3", "openat", "mmap", "mprotect"],
}

# Strace syscall group filter (avoids capturing futex/sched noise from trace=all)
STRACE_FILTER = "%file,%network,%process,%desc,%memory,%signal"


# ─── Core functions ────────────────────────────────────────────────────────────

def parse_strace_summary(log_path: str) -> dict[str, int]:
    """
    Parse a `strace -c -o <log>` output file into {syscall_name: call_count}.

    strace -c output format (with optional errors column):
        % time     seconds  usecs/call     calls    errors syscall
        ------ ----------- ----------- --------- --------- ----------------
          42.1    0.003100          12       250           openat
           6.3    0.000460           5        90        10 read
        ------ ----------- ----------- --------- --------- ----------------
         100.0    0.007360          17       420        10 total

    Regex handles both 5-column (no errors) and 6-column (with errors) rows.
    """
    if not os.path.exists(log_path):
        return {}

    counts: dict[str, int] = {}
    # Group 1 = call count, Group 2 = syscall name
    pattern = re.compile(
        r'^\s*[\d.]+\s+[\d.]+\s+\d+\s+(\d+)\s+(?:\d+\s+)?([a-z_][a-z0-9_]*)\s*$'
    )
    try:
        with open(log_path) as f:
            for line in f:
                m = pattern.match(line)
                if m:
                    name = m.group(2)
                    if name != "total":
                        counts[name] = int(m.group(1))
    except Exception as e:
        logger.warning(f"  Failed to parse {log_path}: {e}")
    return counts


def run_strace(
    env_python: str,
    code: str,
    log_path: str,
    timeout_secs: int = 45,
    strace_filter: str = STRACE_FILTER,
    label: str = "",
) -> dict[str, int]:
    """
    Run code under strace -f -c and return parsed syscall counts.
    Returns {} on timeout or failure — never raises.
    """
    cmd = [
        "/usr/bin/timeout", str(timeout_secs),
        "/usr/bin/strace", "-f", "-c",
        "-e", f"trace={strace_filter}",
        "-o", log_path,
        env_python, "-c", code,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_secs + 10,
        )
        if result.returncode == 124:
            logger.warning(f"  [{label}] TIMEOUT after {timeout_secs}s — skipping")
            return {}
        if result.returncode not in (0, 1):
            logger.warning(f"  [{label}] exit code {result.returncode}")
            if result.stderr:
                logger.warning(f"  [{label}] stderr: {result.stderr[:300]}")
        return parse_strace_summary(log_path)
    except subprocess.TimeoutExpired:
        logger.warning(f"  [{label}] outer timeout expired")
        return {}
    except Exception as e:
        logger.warning(f"  [{label}] strace failed: {e}")
        return {}


def compute_delta(
    baseline: dict[str, int],
    workload: dict[str, int],
    min_delta: int = 3,
) -> list[str]:
    """
    Return syscalls that carry signal from the workload (not just baseline noise).

    Inclusion rules (applied in order, union result):
    1. Syscall is NEW in workload (not present in baseline at all)
    2. Syscall count increased by > min_delta AND it is not in NOISE_SYSCALLS

    Result ordered by delta count descending (highest signal first).
    """
    new_syscalls: list[tuple[str, int]] = []

    all_names = set(workload) | set(baseline)
    for name in all_names:
        w_count = workload.get(name, 0)
        b_count = baseline.get(name, 0)
        delta = w_count - b_count

        if delta <= 0:
            continue

        if name in NOISE_SYSCALLS:
            continue

        if name not in baseline:
            # Completely new syscall — always include
            new_syscalls.append((name, delta))
        elif delta > min_delta:
            # Significantly increased count
            new_syscalls.append((name, delta))

    # Sort by delta descending
    new_syscalls.sort(key=lambda x: -x[1])
    return [name for name, _ in new_syscalls]


def check_strace_available() -> str:
    """Return strace path or raise."""
    path = shutil.which("strace") or "/usr/bin/strace"
    if not os.path.exists(path):
        raise FileNotFoundError("strace not found. Install with: sudo apt install strace")
    result = subprocess.run([path, "--version"], capture_output=True, text=True)
    version_line = result.stdout.splitlines()[0] if result.stdout else "unknown"
    logger.info(f"strace: {path} ({version_line})")
    return path


def check_framework_available(env_python: str, framework: str) -> bool:
    """Check if a framework can be imported in env_python."""
    import_name = {"pytorch": "torch", "tensorflow": "tensorflow", "jax": "jax"}[framework]
    code = f"import {import_name}; print('ok')"
    try:
        r = subprocess.run(
            [env_python, "-c", code],
            capture_output=True, text=True, timeout=30
        )
        return r.returncode == 0 and "ok" in r.stdout
    except Exception:
        return False


# ─── Main capture loop ─────────────────────────────────────────────────────────

def capture_all(
    env_python: str,
    output_path: str,
    timeout_secs: int = 45,
    dry_run: bool = False,
) -> dict:
    """
    Run strace on all 15 (framework, category) workloads and write output_path.
    Returns the full results dict.
    """
    check_strace_available()

    # Check which frameworks are actually importable
    available = {}
    logger.info("Checking framework availability in strace_env...")
    for fw in FRAMEWORKS:
        ok = check_framework_available(env_python, fw)
        available[fw] = ok
        status = "✓" if ok else "✗ (will use fallback)"
        logger.info(f"  {fw}: {status}")

    results: dict = {
        "_metadata": {
            "capture_date": datetime.now(timezone.utc).isoformat(),
            "strace_binary": shutil.which("strace") or "/usr/bin/strace",
            "env_python": env_python,
            "filter": STRACE_FILTER,
            "timeout_secs": timeout_secs,
            "delta_method": "new_in_workload OR (delta>3 AND not noise_syscall)",
            "frameworks_available": available,
        }
    }

    total = len(FRAMEWORKS) * len(CATEGORIES)
    done = 0

    with tempfile.TemporaryDirectory(prefix="strace_capture_") as tmpdir:
        for fw in FRAMEWORKS:
            results[fw] = {}
            logger.info(f"\n{'='*60}")
            logger.info(f"Framework: {fw.upper()}")
            logger.info(f"{'='*60}")

            if dry_run:
                logger.info(f"  [DRY RUN] Baseline code:\n{get_baseline(fw)}")

            # Run baseline once per framework
            baseline_log = os.path.join(tmpdir, f"baseline_{fw}.log")
            baseline_counts: dict[str, int] = {}

            if not dry_run and available[fw]:
                logger.info(f"  Running baseline (import only)...")
                t0 = time.perf_counter()
                baseline_counts = run_strace(
                    env_python, get_baseline(fw), baseline_log,
                    timeout_secs=timeout_secs,
                    label=f"{fw}/baseline",
                )
                elapsed = time.perf_counter() - t0
                logger.info(f"  Baseline done in {elapsed:.1f}s — {len(baseline_counts)} syscall types")

            for cat in CATEGORIES:
                done += 1
                label = f"{fw}/{cat}"
                logger.info(f"\n  [{done}/{total}] {label}")

                workload_code = get_workload(cat, fw)

                if dry_run:
                    logger.info(f"  [DRY RUN] Workload code:\n{workload_code}")
                    results[fw][cat] = FALLBACK_MAP.get(cat, [])
                    continue

                if not available[fw]:
                    logger.warning(f"  Framework not available — using fallback")
                    results[fw][cat] = FALLBACK_MAP.get(cat, [])
                    continue

                workload_log = os.path.join(tmpdir, f"workload_{fw}_{cat}.log")
                t0 = time.perf_counter()
                workload_counts = run_strace(
                    env_python, workload_code, workload_log,
                    timeout_secs=timeout_secs,
                    label=label,
                )
                elapsed = time.perf_counter() - t0

                if not workload_counts:
                    logger.warning(f"  No strace output — using fallback for {label}")
                    results[fw][cat] = FALLBACK_MAP.get(cat, [])
                    continue

                syscalls = compute_delta(baseline_counts, workload_counts)

                if not syscalls:
                    logger.warning(f"  Delta is empty — using fallback for {label}")
                    syscalls = FALLBACK_MAP.get(cat, [])

                results[fw][cat] = syscalls
                logger.info(f"  Done in {elapsed:.1f}s — syscalls: {syscalls}")

    # Write output
    if not dry_run:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nSaved strace map to: {output_path}")

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("SYSCALL MAP SUMMARY")
    logger.info("=" * 60)
    for fw in FRAMEWORKS:
        for cat in CATEGORIES:
            syscalls = results.get(fw, {}).get(cat, [])
            logger.info(f"  {fw:12s} | {cat:20s} | {syscalls}")

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Capture real syscalls via strace")
    parser.add_argument(
        "--env-python",
        required=True,
        help="Path to Python binary in strace_env (e.g. ~/miniconda3/envs/strace_env/bin/python)",
    )
    parser.add_argument(
        "--output",
        default="dataset/strace_syscall_map.json",
        help="Output path for strace_syscall_map.json",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
        help="Timeout in seconds per strace run (default: 45)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print workload code without running strace",
    )
    args = parser.parse_args()

    env_python = os.path.expanduser(args.env_python)
    if not os.path.exists(env_python):
        logger.error(f"env-python not found: {env_python}")
        sys.exit(1)

    capture_all(
        env_python=env_python,
        output_path=args.output,
        timeout_secs=args.timeout,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
