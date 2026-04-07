#!/usr/bin/env python3
"""
strace_api_fuzzer.py — Per-API strace capture using fuzzed calls.

Instead of running one generic workload per (framework, category), this script:
  1. Picks unique api_paths from the dataset
  2. Generates executable Python that actually CALLS each specific API with
     fuzzed (synthesized) inputs via importlib + inspect
  3. Runs each call under strace with baseline subtraction
  4. Records per-API syscall signatures in api_syscall_map.json

Usage:
    # Small test: 30 APIs across frameworks
    python dataset/strace_api_fuzzer.py \\
        --env-python ~/miniconda3/envs/strace_env/bin/python \\
        --dataset labeled_dataset_strace.jsonl \\
        --output dataset/api_syscall_map.json \\
        --sample 30 --timeout 20

    # Full run (all unique APIs)
    python dataset/strace_api_fuzzer.py \\
        --env-python ~/miniconda3/envs/strace_env/bin/python \\
        --dataset labeled_dataset_strace.jsonl \\
        --output dataset/api_syscall_map.json \\
        --timeout 20
"""

import os
import sys
import re
import json
import time
import shutil
import random
import hashlib
import tempfile
import argparse
import logging
import subprocess
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from strace_capture import (
    parse_strace_summary, compute_delta, check_strace_available,
    NOISE_SYSCALLS, STRACE_FILTER, FALLBACK_MAP
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── Merged label → fallback syscalls (used when fuzzing fails) ──────────────
MERGE_MAP = {
    "file_read": "file_access", "file_write": "file_access",
    "dir_read": "file_access", "dir_write": "file_access",
    "network_send": "network_access", "network_receive": "network_access",
    "process_create": "process_mgmt", "process_abort": "process_mgmt",
    "process_sleep": "process_mgmt",
    "calc_math": "pure_calculation", "calc_data_mgmt": "pure_calculation",
    "calc_encode_decode": "pure_calculation",
    "code_execution": "code_execution",
}

# ─── Framework setup preambles ────────────────────────────────────────────────
_NO_CACHE = "import sys; sys.dont_write_bytecode = True  # suppress __pycache__ writes\n"

FRAMEWORK_SETUP = {
    "pytorch": _NO_CACHE + """\
import os, sys, traceback
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
""",
    "tensorflow": _NO_CACHE + """\
import os, sys, traceback
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tensorflow as tf
import numpy as np
""",
    "jax": _NO_CACHE + """\
import os, sys, traceback
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax, jax.numpy as jnp
import numpy as np
""",
}

# ─── Per-framework fuzz value factories ───────────────────────────────────────
# Each value is a Python expression string that produces a concrete object.
FUZZ_FACTORIES = {
    "pytorch": {
        "tensor":        "torch.randn(4, 4).detach()",
        "tensor_1d":     "torch.randn(16)",
        "tensor_3d":     "torch.randn(2, 4, 4)",
        "int_tensor":    "torch.randint(0, 10, (4,))",
        "bool_tensor":   "torch.ones(4, dtype=torch.bool)",
        "long_tensor":   "torch.zeros(4, dtype=torch.long)",
        "int":           "4",
        "float":         "1.0",
        "bool":          "False",
        "str":           '"test"',
        "none":          "None",
        "list":          "[1, 2, 3, 4]",
        "tuple":         "(4, 4)",
        "dtype":         "torch.float32",
        "device":        '"cpu"',
        "module":        "nn.Linear(4, 4)",
        "optimizer":     "torch.optim.SGD([torch.randn(4,4,requires_grad=True)], lr=0.01)",
    },
    "tensorflow": {
        "tensor":        "tf.random.normal([4, 4])",
        "tensor_1d":     "tf.random.normal([16])",
        "tensor_3d":     "tf.random.normal([2, 4, 4])",
        "int_tensor":    "tf.constant([1, 2, 3, 4])",
        "bool_tensor":   "tf.constant([True, False, True, False])",
        "int":           "4",
        "float":         "1.0",
        "bool":          "False",
        "str":           '"test"',
        "none":          "None",
        "list":          "[1, 2, 3, 4]",
        "tuple":         "(4, 4)",
        "dtype":         "tf.float32",
        "shape":         "tf.TensorShape([4, 4])",
    },
    "jax": {
        "tensor":        "jnp.ones((4, 4))",
        "tensor_1d":     "jnp.ones(16)",
        "tensor_3d":     "jnp.ones((2, 4, 4))",
        "int_tensor":    "jnp.array([1, 2, 3, 4])",
        "bool_tensor":   "jnp.array([True, False, True, False])",
        "int":           "4",
        "float":         "1.0",
        "bool":          "False",
        "str":           '"test"',
        "none":          "None",
        "key":           "jax.random.PRNGKey(0)",
        "list":          "[1, 2, 3, 4]",
        "tuple":         "(4, 4)",
        "dtype":         "jnp.float32",
    },
}

# ─── Parameter name → fuzz value heuristics ──────────────────────────────────
# Matched as substring in parameter name (lowercase).
PARAM_NAME_HINTS = [
    # Tensor-like first (highest priority)
    (["input", "inputs", "x", "tensor", "data", "features", "logit",
      "hidden", "embed", "value", "values", "query", "key", "v",
      "weight", "bias", "grad", "gradient", "output", "target_val",
      "pred", "label_val", "score"],   "tensor"),
    # Second tensor argument
    (["y", "other", "b", "w"],          "tensor"),
    # Index/shape tensors
    (["index", "indices", "idx"],       "int_tensor"),
    (["mask",],                          "bool_tensor"),
    # Integers
    (["num_", "n_", "_size", "size", "count", "steps", "k", "d_",
      "dim", "axis", "depth", "length", "width", "height",
      "channels", "classes", "heads", "groups", "stride",
      "padding", "dilation", "rank"],   "int"),
    # Floats
    (["rate", "ratio", "scale", "alpha", "beta", "gamma", "epsilon",
      "momentum", "lr", "learning_rate", "eps", "p", "prob"],  "float"),
    # Booleans
    (["training", "inplace", "bias", "affine", "track",
      "requires_grad", "normalize"],   "bool"),
    # Shape tuples
    (["shape", "size"],                 "tuple"),
    # Data type
    (["dtype"],                         "dtype"),
    # Device
    (["device"],                        "device"),
    # String (low priority)
    (["mode", "reduction", "padding_mode"],  "str"),
]


def _param_to_fuzz_key(name: str, annotation, framework: str) -> str:
    """Map a parameter name + annotation to a fuzz factory key."""
    name_lower = name.lower()

    # Check annotation first
    ann_str = str(annotation).lower()
    if "tensor" in ann_str or "ndarray" in ann_str or "array" in ann_str:
        return "tensor"
    if "int" in ann_str and "float" not in ann_str:
        return "int"
    if "float" in ann_str:
        return "float"
    if "bool" in ann_str:
        return "bool"
    if "str" in ann_str:
        return "str"
    if "dtype" in ann_str:
        return "dtype"

    # Check parameter name hints
    for name_hints, fuzz_key in PARAM_NAME_HINTS:
        if any(h in name_lower for h in name_hints):
            if fuzz_key in FUZZ_FACTORIES.get(framework, {}):
                return fuzz_key

    # Default: tensor (most common in ML frameworks)
    return "tensor"


def generate_introspect_snippet(api_path: str, framework: str) -> str:
    """
    Generate a Python snippet that:
      1. Imports the api via importlib (walking from longest importable prefix)
      2. Inspects its signature
      3. Calls it with fuzz-generated arguments
      4. Handles failures silently (so strace still gets baseline syscalls)
    """
    fuzz = FUZZ_FACTORIES.get(framework, FUZZ_FACTORIES["pytorch"])
    setup = FRAMEWORK_SETUP.get(framework, "")

    # Build per-fuzz-key assignment block
    fuzz_assignments = "\n".join(
        f"    _FUZZ['{k}'] = {v}" for k, v in fuzz.items()
    )

    snippet = f"""\
{setup}
import importlib, inspect

_FUZZ = {{}}
try:
{fuzz_assignments}
except Exception:
    pass

def _fuzz_param(name, param, fw):
    ann = param.annotation
    name_l = name.lower()
    ann_s = str(ann).lower()
    # Annotation-based
    if any(t in ann_s for t in ['tensor', 'ndarray', 'array']):
        return _FUZZ.get('tensor')
    if 'int' in ann_s and 'float' not in ann_s:
        return _FUZZ.get('int', 4)
    if 'float' in ann_s:
        return _FUZZ.get('float', 1.0)
    if 'bool' in ann_s:
        return _FUZZ.get('bool', False)
    if 'str' in ann_s:
        return _FUZZ.get('str', 'test')
    if 'dtype' in ann_s:
        return _FUZZ.get('dtype')
    # Name-based
    for hints, key in [
        (['input','inputs','x','tensor','data','features','logit','hidden',
          'embed','value','query','weight','bias','grad','output','score',
          'pred'], 'tensor'),
        (['y','other','b','w'], 'tensor'),
        (['index','indices','idx'], 'int_tensor'),
        (['mask'], 'bool_tensor'),
        (['num_','n_','size','count','steps','k','dim','axis','depth',
          'length','width','height','channels','classes','heads','groups',
          'stride','padding','dilation','rank'], 'int'),
        (['rate','ratio','scale','alpha','beta','gamma','epsilon','momentum',
          'lr','learning_rate','eps','p','prob'], 'float'),
        (['training','inplace','affine','track','requires_grad'], 'bool'),
        (['shape'], 'tuple'),
        (['dtype'], 'dtype'),
        (['device'], 'device'),
    ]:
        if any(h in name_l for h in hints):
            v = _FUZZ.get(key)
            if v is not None:
                return v
    return _FUZZ.get('tensor')

_api_path = {repr(api_path)}
_parts = _api_path.split('.')

try:
    # Walk from longest prefix down to find importable module
    _obj = None
    for _i in range(len(_parts), 0, -1):
        _modpath = '.'.join(_parts[:_i])
        try:
            _mod = importlib.import_module(_modpath)
            _obj = _mod
            for _attr in _parts[_i:]:
                _obj = getattr(_obj, _attr, None)
                if _obj is None:
                    break
            if _obj is not None:
                break
        except (ImportError, ModuleNotFoundError, Exception):
            continue

    if _obj is None or not callable(_obj):
        raise ValueError(f'Could not find callable: {{_api_path}}')

    # Inspect signature
    try:
        _sig = inspect.signature(_obj)
    except (ValueError, TypeError):
        # No signature (C extension etc.) — try calling with no args
        _obj()
        raise StopIteration

    _pos_args = []
    _kw_args = {{}}
    _max_pos = 6  # cap to avoid explosive combinatorics

    for _pname, _param in _sig.parameters.items():
        if _param.kind == inspect.Parameter.VAR_POSITIONAL:
            break
        if _param.kind == inspect.Parameter.VAR_KEYWORD:
            break
        if _param.kind == inspect.Parameter.KEYWORD_ONLY:
            if _param.default is inspect.Parameter.empty:
                _v = _fuzz_param(_pname, _param, {repr(framework)})
                if _v is not None:
                    _kw_args[_pname] = _v
        else:
            if len(_pos_args) >= _max_pos:
                break
            if _param.default is inspect.Parameter.empty:
                _v = _fuzz_param(_pname, _param, {repr(framework)})
                if _v is not None:
                    _pos_args.append(_v)

    _result = _obj(*_pos_args, **_kw_args)

    # Force evaluation for lazy frameworks
    if hasattr(_result, 'block_until_ready'):
        _result.block_until_ready()
    if hasattr(_result, 'numpy'):
        try:
            _result.numpy()
        except Exception:
            pass

except StopIteration:
    pass  # Called with no args successfully
except Exception as _e:
    pass  # Failure is expected for many APIs; strace still captures baseline delta
"""
    return snippet


def generate_direct_snippet(api_path: str, framework: str, call_expression: str) -> str:
    """
    Alternative strategy: adapt the actual call_expression from the dataset record.
    Replaces variable references with fuzz literals.
    """
    fuzz = FUZZ_FACTORIES.get(framework, {})
    setup = FRAMEWORK_SETUP.get(framework, "")

    # Provide common variable aliases so the expression might work as-is
    preamble = "\n".join(f"{k} = {v}" for k, v in fuzz.items())

    # Common aliases used in test code
    alias_block = {
        "pytorch": """
x = torch.randn(4, 4)
y = torch.randn(4, 4)
a = torch.randn(4, 4)
b = torch.randn(4, 4)
model = nn.Linear(4, 4)
tensor = x
input_tensor = x
weight = torch.randn(4, 4)
bias = torch.zeros(4)
""",
        "tensorflow": """
x = tf.random.normal([4, 4])
y = tf.random.normal([4, 4])
a = tf.random.normal([4, 4])
b = tf.random.normal([4, 4])
tensor = x
input_tensor = x
""",
        "jax": """
x = jnp.ones((4, 4))
y = jnp.ones((4, 4))
a = jnp.ones((4, 4))
b = jnp.ones((4, 4))
key = jax.random.PRNGKey(0)
key1 = jax.random.PRNGKey(1)
tensor = x
input_array = x
scores = jnp.ones((16,))
k = 4
recall = 0.9
""",
    }.get(framework, "")

    return f"""\
{setup}
{alias_block}
try:
    _result = {call_expression}
    if hasattr(_result, 'block_until_ready'):
        _result.block_until_ready()
except Exception:
    pass
"""


def run_strace_snippet(
    env_python: str,
    code: str,
    log_path: str,
    timeout_secs: int = 20,
    label: str = "",
) -> dict[str, int]:
    """Run code under strace -f -c, return parsed syscall counts.
    Uses python -B to suppress __pycache__ writes so they don't pollute the delta."""
    cmd = [
        "/usr/bin/timeout", str(timeout_secs),
        "/usr/bin/strace", "-f", "-c",
        "-e", f"trace={STRACE_FILTER}",
        "-o", log_path,
        env_python, "-B", "-c", code,  # -B: don't write .pyc files
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_secs + 10
        )
        if result.returncode == 124:
            logger.debug(f"  [{label}] TIMEOUT")
            return {}
        return parse_strace_summary(log_path)
    except subprocess.TimeoutExpired:
        return {}
    except Exception as e:
        logger.debug(f"  [{label}] error: {e}")
        return {}


# ─── Main capture logic ───────────────────────────────────────────────────────

def collect_unique_apis(dataset_path: str, sample: int = 0) -> list[dict]:
    """
    Read the dataset and collect unique (api_path, framework, labels, call_expression).
    If sample > 0, return a stratified sample across frameworks and merged categories.
    """
    seen: set[str] = set()
    apis: list[dict] = []

    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = f"{r.get('framework')}::{r.get('api_path')}"
            if key in seen:
                continue
            seen.add(key)
            cats = set()
            for lbl in r.get("labels", []):
                mc = MERGE_MAP.get(lbl)
                if mc:
                    cats.add(mc)
            apis.append({
                "api_path": r.get("api_path", ""),
                "framework": r.get("framework", ""),
                "labels": r.get("labels", []),
                "merged_cats": sorted(cats),
                "call_expression": r.get("call_expression", ""),
                "confidence": r.get("label_confidence", 0.5),
            })

    logger.info(f"Unique APIs: {len(apis)}")

    if sample > 0:
        # Stratified sample: pick evenly across (framework, category) cells
        buckets: dict[str, list[dict]] = defaultdict(list)
        for api in apis:
            fw = api["framework"]
            for cat in api["merged_cats"]:
                buckets[f"{fw}:{cat}"].append(api)
        # Shuffle each bucket and interleave
        sampled: list[dict] = []
        seen_paths: set[str] = set()
        per_bucket = max(1, sample // max(len(buckets), 1))
        for bucket_key, bucket_apis in sorted(buckets.items()):
            random.shuffle(bucket_apis)
            for api in bucket_apis[:per_bucket]:
                if api["api_path"] not in seen_paths:
                    sampled.append(api)
                    seen_paths.add(api["api_path"])
        # Top up to sample count from remainder
        random.shuffle(apis)
        for api in apis:
            if len(sampled) >= sample:
                break
            if api["api_path"] not in seen_paths:
                sampled.append(api)
                seen_paths.add(api["api_path"])
        return sampled[:sample]

    return apis


def fuzz_and_capture(
    env_python: str,
    apis: list[dict],
    baseline_map: dict[str, dict[str, int]],  # {framework: {syscall: count}}
    tmpdir: str,
    timeout_secs: int = 20,
) -> dict[str, dict]:
    """
    For each API entry, try two fuzzing strategies and capture syscalls.
    Returns {api_path: {"syscalls": [...], "status": "...", "framework": "..."}}
    """
    results = {}
    total = len(apis)

    for i, api in enumerate(apis, 1):
        api_path = api["api_path"]
        framework = api["framework"]
        call_expr = api["call_expression"]
        baseline = baseline_map.get(framework, {})
        label = f"{framework}/{api_path}"
        short = api_path.split(".")[-1]

        logger.info(f"  [{i}/{total}] {label}")

        # Deduplicate if same api_path seen for this framework
        result_key = f"{framework}::{api_path}"
        if result_key in results:
            continue

        # Safe filename hash
        fname_hash = hashlib.md5(result_key.encode()).hexdigest()[:10]

        syscalls = []
        status = "fail"

        # ── Strategy 1: introspect + fuzz ────────────────────────────────────
        log1 = os.path.join(tmpdir, f"s1_{fname_hash}.log")
        code1 = generate_introspect_snippet(api_path, framework)
        counts1 = run_strace_snippet(env_python, code1, log1,
                                     timeout_secs=timeout_secs, label=f"{short}/introspect")
        if counts1:
            syscalls = compute_delta(baseline, counts1, min_delta=2)
            if syscalls:
                status = "introspect"
                logger.info(f"    → introspect OK: {syscalls}")

        # ── Strategy 2: direct call_expression ───────────────────────────────
        if not syscalls and call_expr:
            log2 = os.path.join(tmpdir, f"s2_{fname_hash}.log")
            code2 = generate_direct_snippet(api_path, framework, call_expr)
            counts2 = run_strace_snippet(env_python, code2, log2,
                                         timeout_secs=timeout_secs, label=f"{short}/direct")
            if counts2:
                syscalls = compute_delta(baseline, counts2, min_delta=2)
                if syscalls:
                    status = "direct"
                    logger.info(f"    → direct OK: {syscalls}")

        # ── Fallback: category-level syscalls ─────────────────────────────────
        if not syscalls:
            cats = api.get("merged_cats", [])
            fallback_sysc: list[str] = []
            seen_fb: set[str] = set()
            for cat in cats:
                for sc in FALLBACK_MAP.get(cat, []):
                    if sc not in seen_fb:
                        seen_fb.add(sc)
                        fallback_sysc.append(sc)
            syscalls = fallback_sysc
            status = "fallback"
            logger.info(f"    → fallback: {syscalls}")

        results[result_key] = {
            "api_path": api_path,
            "framework": framework,
            "syscalls": syscalls,
            "status": status,
            "merged_cats": api.get("merged_cats", []),
        }

    return results


_COMPREHENSIVE_BASELINES = {
    "pytorch": """\
import sys; sys.dont_write_bytecode = True
import os
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
# Pre-warm lazy submodules so their import I/O is captured in baseline
import torch.fx, torch.fx.traceback
import torch._inductor, torch._dynamo
import torch.distributed
import torch.distributed.elastic
import torch.onnx
import torch.jit
import torch.serialization
import torch.nn.attention
import torch.utils, torch.utils.data
_x = torch.randn(4, 4)
_y = torch.matmul(_x, _x)
del _x, _y
""",
    "tensorflow": """\
import sys; sys.dont_write_bytecode = True
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tensorflow as tf
import numpy as np
_x = tf.constant(0.0)
del _x
""",
    "jax": """\
import sys; sys.dont_write_bytecode = True
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax, jax.numpy as jnp
import numpy as np
_x = jnp.zeros(1)
_x.block_until_ready()
del _x
""",
}


def run_baselines(
    env_python: str,
    tmpdir: str,
    timeout_secs: int,
) -> dict[str, dict[str, int]]:
    """Run comprehensive baseline strace for each framework once."""
    from strace_workloads import FRAMEWORKS
    baselines = {}
    for fw in FRAMEWORKS:
        log = os.path.join(tmpdir, f"baseline_{fw}.log")
        code = _COMPREHENSIVE_BASELINES[fw]
        logger.info(f"  Baseline for {fw}...")
        counts = run_strace_snippet(env_python, code, log,
                                    timeout_secs=timeout_secs * 3, label=f"{fw}/baseline")
        baselines[fw] = counts
        logger.info(f"    {len(counts)} syscall types in baseline")
    return baselines


def apply_api_syscalls_to_dataset(
    dataset_path: str,
    output_path: str,
    api_map: dict[str, dict],
) -> None:
    """
    Stream dataset, replace syscalls with per-API strace results where available,
    fall back to existing syscalls_source='strace' (category-level) otherwise.
    """
    stats = defaultdict(int)
    fallback_to_category = 0

    with open(dataset_path) as inf, open(output_path, "w") as outf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            fw = r.get("framework", "")
            api_path = r.get("api_path", "")
            result_key = f"{fw}::{api_path}"

            entry = api_map.get(result_key)
            if entry and entry.get("syscalls"):
                r["syscalls"] = entry["syscalls"]
                r["syscalls_source"] = f"strace_fuzz_{entry['status']}"
                stats[f"status_{entry['status']}"] += 1
            else:
                # Keep existing category-level syscalls
                fallback_to_category += 1
                stats["status_category"] += 1

            stats["total"] += 1
            outf.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"  Total records: {stats['total']:,}")
    for k in sorted(stats):
        if k.startswith("status_"):
            logger.info(f"  {k[7:]:15s}: {stats[k]:,}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-API strace capture with fuzzing"
    )
    parser.add_argument("--env-python", required=True,
                        help="Python binary in strace_env")
    parser.add_argument("--dataset", default="labeled_dataset_strace.jsonl",
                        help="Input dataset (default: labeled_dataset_strace.jsonl)")
    parser.add_argument("--output", default="dataset/api_syscall_map.json",
                        help="Output JSON map (default: dataset/api_syscall_map.json)")
    parser.add_argument("--apply-to", default="",
                        help="If set, also apply the map to this dataset path")
    parser.add_argument("--apply-output", default="",
                        help="Output path for the applied dataset")
    parser.add_argument("--sample", type=int, default=0,
                        help="Test N APIs only (0 = all, default: 0)")
    parser.add_argument("--timeout", type=int, default=20,
                        help="Timeout per strace run in seconds (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    env_python = os.path.expanduser(args.env_python)
    if not os.path.exists(env_python):
        logger.error(f"env-python not found: {env_python}")
        sys.exit(1)

    check_strace_available()

    # Collect unique APIs
    logger.info(f"Reading dataset: {args.dataset}")
    apis = collect_unique_apis(args.dataset, sample=args.sample)
    logger.info(f"APIs to fuzz: {len(apis)}")

    with tempfile.TemporaryDirectory(prefix="strace_fuzz_") as tmpdir:
        # Run baselines once per framework
        logger.info("Running baselines...")
        baselines = run_baselines(env_python, tmpdir, args.timeout)

        # Fuzz and capture per-API
        logger.info(f"\nFuzzing {len(apis)} APIs...")
        results = fuzz_and_capture(env_python, apis, baselines, tmpdir, args.timeout)

    # Build output map keyed by "framework::api_path"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved API syscall map: {args.output} ({len(results)} entries)")

    # Summary
    status_counts: dict[str, int] = defaultdict(int)
    for entry in results.values():
        status_counts[entry["status"]] += 1
    logger.info("Status breakdown:")
    for s, c in sorted(status_counts.items()):
        logger.info(f"  {s:15s}: {c}")

    # Optionally apply to dataset
    if args.apply_to and args.apply_output:
        logger.info(f"\nApplying to: {args.apply_to}")
        apply_api_syscalls_to_dataset(args.apply_to, args.apply_output, results)
        logger.info(f"Written to: {args.apply_output}")


if __name__ == "__main__":
    main()
