#!/usr/bin/env python3
"""
Multi-Label API Behavior Labeler
=================================

Labels extracted API call sites with behavioral categories:
  - File access (read, write, dir_read, dir_write)
  - Network access (send, receive)
  - Process management (create, abort, sleep)
  - Pure calculation (math, data_mgmt, encode_decode)
  - Code execution (user-controlled function execution)

Each API call can have MULTIPLE labels (multi-label classification).

Usage:
    # Label an existing extracted dataset:
    python api_behavior_labeler.py \
        --input ./unified_dataset.jsonl \
        --output ./labeled_dataset.jsonl

    # Also export the review queue (ambiguous cases):
    python api_behavior_labeler.py \
        --input ./unified_dataset.jsonl \
        --output ./labeled_dataset.jsonl \
        --review-queue ./review_queue.jsonl

    # Run with docstring analysis (requires repo access):
    python api_behavior_labeler.py \
        --input ./unified_dataset.jsonl \
        --output ./labeled_dataset.jsonl \
        --tf-repo ./tensorflow \
        --pytorch-repo ./pytorch \
        --jax-repo ./jax
"""

import ast
import re
import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Label Definitions
# =============================================================================

class Labels:
    """All possible behavioral labels."""

    # File access
    FILE_READ       = "file_read"
    FILE_WRITE      = "file_write"
    DIR_READ        = "dir_read"
    DIR_WRITE       = "dir_write"

    # Network access
    NETWORK_SEND    = "network_send"
    NETWORK_RECEIVE = "network_receive"

    # Process management
    PROCESS_CREATE  = "process_create"
    PROCESS_ABORT   = "process_abort"
    PROCESS_SLEEP   = "process_sleep"

    # Pure calculation
    CALC_MATH       = "calc_math"
    CALC_DATA_MGMT  = "calc_data_mgmt"
    CALC_ENCODE_DECODE = "calc_encode_decode"

    # Code execution
    CODE_EXECUTION  = "code_execution"

    # For grouping in output
    GROUPS = {
        "file_access":         [FILE_READ, FILE_WRITE, DIR_READ, DIR_WRITE],
        "network_access":      [NETWORK_SEND, NETWORK_RECEIVE],
        "process_management":  [PROCESS_CREATE, PROCESS_ABORT, PROCESS_SLEEP],
        "pure_calculation":    [CALC_MATH, CALC_DATA_MGMT, CALC_ENCODE_DECODE],
        "code_execution":      [CODE_EXECUTION],
    }

    ALL = [
        FILE_READ, FILE_WRITE, DIR_READ, DIR_WRITE,
        NETWORK_SEND, NETWORK_RECEIVE,
        PROCESS_CREATE, PROCESS_ABORT, PROCESS_SLEEP,
        CALC_MATH, CALC_DATA_MGMT, CALC_ENCODE_DECODE,
        CODE_EXECUTION,
    ]


# =============================================================================
# Syscall Map — maps each fine-grained label to the Linux syscalls it implies
# =============================================================================
#
# Used to enrich labeled records with a `syscalls` field so that downstream
# training can learn from OS-level system call signatures rather than (or in
# addition to) raw code text.
#
# Ordering within each list is roughly by frequency of occurrence for that
# label, which lets word-level models pick up positional patterns too.

SYSCALL_MAP: dict[str, list[str]] = {
    Labels.FILE_READ: [
        "openat", "open", "read", "pread64", "readv", "preadv",
        "close", "fstat", "stat", "lstat", "access", "faccessat",
    ],
    Labels.FILE_WRITE: [
        "openat", "open", "write", "pwrite64", "writev", "pwritev",
        "close", "fsync", "fdatasync", "truncate", "ftruncate",
        "unlink", "unlinkat", "rename", "renameat",
    ],
    Labels.DIR_READ: [
        "openat", "getdents64", "getdents", "stat", "lstat",
        "access", "faccessat", "readlink",
    ],
    Labels.DIR_WRITE: [
        "mkdir", "mkdirat", "rmdir", "rename", "renameat",
        "symlink", "symlinkat", "link", "linkat",
        "chmod", "fchmod", "chown", "fchown",
    ],
    Labels.NETWORK_SEND: [
        "socket", "connect", "bind", "send", "sendto",
        "sendmsg", "sendmmsg", "write", "writev",
    ],
    Labels.NETWORK_RECEIVE: [
        "socket", "listen", "accept", "accept4",
        "recv", "recvfrom", "recvmsg", "recvmmsg",
        "read", "readv",
    ],
    Labels.PROCESS_CREATE: [
        "clone", "clone3", "fork", "vfork",
        "execve", "execveat", "posix_spawn",
    ],
    Labels.PROCESS_ABORT: [
        "exit_group", "exit", "kill", "tkill", "tgkill",
        "sigaction", "rt_sigaction", "raise",
    ],
    Labels.PROCESS_SLEEP: [
        "nanosleep", "clock_nanosleep", "futex",
        "poll", "ppoll", "select", "pselect6",
        "epoll_wait", "epoll_pwait",
    ],
    Labels.CALC_MATH: [
        # Pure CPU computation — no syscalls required.
        # A small set of memory syscalls may appear for large tensor allocation.
        "mmap", "munmap",
    ],
    Labels.CALC_DATA_MGMT: [
        "mmap", "munmap", "mprotect", "madvise",
        "mremap", "brk",
    ],
    Labels.CALC_ENCODE_DECODE: [
        # Pure CPU — typically no syscalls; mmap for large buffers.
        "mmap", "munmap",
    ],
    Labels.CODE_EXECUTION: [
        "execve", "execveat",
        "mmap", "mprotect",
        "clone", "fork",
        "openat", "close",  # dlopen path
    ],
}


def derive_syscalls(labels: list[str]) -> list[str]:
    """
    Build an ordered, deduplicated syscall sequence for a set of fine-grained
    behavioral labels.  Ordering preserves the label priority in Labels.ALL so
    the sequence is deterministic and consistent across records.
    """
    seen: set[str] = set()
    result: list[str] = []
    for label in Labels.ALL:               # stable order
        if label in labels:
            for sc in SYSCALL_MAP.get(label, []):
                if sc not in seen:
                    seen.add(sc)
                    result.append(sc)
    return result


# =============================================================================
# Layer 1: Rule-Based Labeler
# =============================================================================

class RuleBasedLabeler:
    """
    Assigns labels based on API path patterns, argument names,
    and known framework semantics.

    Rules are organized by framework and by label. Each rule is a dict with:
      - "path_patterns": list of regex patterns to match against api_path
      - "path_keywords": list of substrings to match in api_path (case-insensitive)
      - "arg_keywords":  list of arg names that suggest this behavior
      - "context_keywords": list of keywords to look for in surrounding code

    A rule fires if ANY of its conditions match.
    """

    def __init__(self):
        self.rules = self._build_rules()

    def label(self, record: dict) -> tuple[list[str], float]:
        """
        Assign labels to a single API call record.
        Returns (labels, confidence) where confidence is 0.0-1.0
        """
        api_path = record.get("api_path", "").lower()
        func_name = api_path.split(".")[-1] if api_path else ""
        kwarg_names = [k.lower() for k in record.get("kwarg_names", [])]
        code_context = record.get("code_context", "").lower()
        framework = record.get("framework", "")

        labels = set()
        match_strengths = []  # track how strongly each label matched

        for label, rule_set in self.rules.items():
            strength = self._evaluate_rule(
                rule_set, api_path, func_name, kwarg_names, code_context, framework
            )
            if strength > 0:
                labels.add(label)
                match_strengths.append(strength)

        # If nothing matched, try fallback heuristics
        if not labels:
            labels, match_strengths = self._fallback_heuristics(
                api_path, func_name, kwarg_names, code_context
            )

        confidence = min(1.0, sum(match_strengths) / max(len(match_strengths), 1))
        return sorted(labels), confidence

    def _evaluate_rule(self, rule_set, api_path, func_name, kwarg_names, context, framework):
        """Evaluate a rule set against the call. Returns match strength 0.0-1.0."""
        strength = 0.0

        # Check framework-specific rules first (higher priority)
        fw_rules = rule_set.get(f"_{framework}", {})
        general_rules = rule_set.get("_general", {})

        for rules in [fw_rules, general_rules]:
            # Path pattern match (strongest signal)
            for pattern in rules.get("path_patterns", []):
                if re.search(pattern, api_path):
                    strength = max(strength, 0.95)

            # Path keyword match (strong signal)
            for keyword in rules.get("path_keywords", []):
                if keyword in api_path:
                    strength = max(strength, 0.85)

            # Function name exact match
            for name in rules.get("func_names", []):
                if func_name == name:
                    strength = max(strength, 0.90)

            # Argument name match (moderate signal)
            for arg_kw in rules.get("arg_keywords", []):
                if arg_kw in kwarg_names or any(arg_kw in k for k in kwarg_names):
                    strength = max(strength, 0.60)

            # Context keyword match (weaker signal)
            for ctx_kw in rules.get("context_keywords", []):
                if ctx_kw in context:
                    strength = max(strength, 0.40)

        return strength

    def _fallback_heuristics(self, api_path, func_name, kwarg_names, context):
        """Last resort: very broad pattern matching."""
        labels = set()
        strengths = []

        # If path looks mathematical and nothing else matched
        math_indicators = ["add", "sub", "mul", "div", "matmul", "dot", "sum", "prod",
                           "max", "min", "abs", "sqrt", "exp", "log", "pow", "sin", "cos",
                           "tan", "sigmoid", "relu", "softmax", "tanh", "norm"]
        if any(ind == func_name or ind in api_path.split(".")[-1] for ind in math_indicators):
            labels.add(Labels.CALC_MATH)
            strengths.append(0.70)

        # Data manipulation default
        data_indicators = ["reshape", "transpose", "concat", "stack", "split", "slice",
                           "gather", "scatter", "permute", "expand", "squeeze", "view",
                           "contiguous", "clone", "copy", "fill", "zero", "ones", "empty",
                           "arange", "linspace", "meshgrid", "broadcast", "tile", "repeat",
                           "pad", "crop", "flip", "roll", "sort", "unique", "where",
                           "index", "select", "narrow", "chunk"]
        if any(ind == func_name or ind in api_path.split(".")[-1] for ind in data_indicators):
            labels.add(Labels.CALC_DATA_MGMT)
            strengths.append(0.65)

        # If still nothing, mark as data management (safest default for tensor ops)
        if not labels and any(fw in api_path for fw in ["torch", "tensorflow", "tf.", "jax"]):
            labels.add(Labels.CALC_DATA_MGMT)
            strengths.append(0.30)

        return labels, strengths

    def _build_rules(self) -> dict:
        """Build the complete rule database."""
        return {
            # ===========================================================
            # FILE_READ
            # ===========================================================
            Labels.FILE_READ: {
                "_general": {
                    "path_patterns": [
                        r"read_file",
                        r"load_model",
                        r"load_weights",
                        r"load_checkpoint",
                        r"from_saved_model",
                        r"parse_.*_example",
                        r"from_pretrained",
                        r"deserializ",
                    ],
                    "path_keywords": [
                        "read_file", "load_dataset", "load_model", "load_weights",
                        "from_saved", "restore", "import_graph", "parse_example",
                        "read_record", "tfrecord", "from_generator",
                        "load_state_dict", "load_checkpoint",
                        "open_dataset", "from_file", "from_csv", "from_json",
                        "from_parquet", "from_tfrecords", "from_tensor_slices",
                        "read_csv", "read_json", "read_numpy", "from_numpy_file",
                        "cache_dataset",  # reads cache from disk
                    ],
                    "func_names": [
                        "load", "restore", "read", "open", "parse",
                        "from_saved_model", "load_model", "load_weights",
                        "load_state_dict", "from_pretrained",
                    ],
                    "arg_keywords": [
                        "filepath", "file_path", "filename", "file_name",
                        "load_path", "restore_path", "input_path",
                        "ckpt_path", "checkpoint_path", "model_path",
                        # Removed: "path" — too generic, matches search_path, module_path, exec_path, etc.
                    ],
                    "context_keywords": [
                        "open(", "with open", "read()", "rb",
                    ],
                },
                "_tensorflow": {
                    "path_patterns": [
                        r"gen_io_ops\.read",
                        r"gen_dataset_ops\.(cache|text_line|csv|fixed_length_record|tfrecord)",
                        r"gen_parsing_ops\.",
                        r"saved_model.*load",
                    ],
                    "path_keywords": [
                        "gen_io_ops.read", "gen_parsing_ops",
                        "saved_model_load", "checkpoint.restore",
                    ],
                },
                "_pytorch": {
                    "path_patterns": [
                        r"torch\.load",
                        r"torch\.jit\.load",
                        r"safetensors.*load",
                    ],
                    "path_keywords": [
                        "torch.load", "torch.jit.load",
                        "torchvision.io.read", "torchaudio.load",
                        "torch.utils.data.dataloader",
                    ],
                },
                "_jax": {
                    "path_patterns": [
                        r"jax\..*checkpoint.*restore",
                        r"orbax.*restore",
                    ],
                    "path_keywords": [
                        "checkpoint.restore", "from_state_dict",
                    ],
                },
            },

            # ===========================================================
            # FILE_WRITE
            # ===========================================================
            Labels.FILE_WRITE: {
                "_general": {
                    "path_patterns": [
                        r"write_file",
                        r"save_model",
                        r"save_weights",
                        r"save_checkpoint",
                        r"to_saved_model",
                        r"seriali[zs]",
                        r"export_model",
                    ],
                    "path_keywords": [
                        "write_file", "save_model", "save_weights",
                        "save_checkpoint", "export", "to_saved_model",
                        "save_state_dict", "write_graph",
                        "write_summary", "write_event", "write_tensor",
                        "to_csv", "to_json", "to_parquet",
                    ],
                    "func_names": [
                        "save", "dump", "write", "export", "serialize",
                        "save_model", "save_weights", "save_state_dict",
                    ],
                    "arg_keywords": [
                        "save_path", "output_path", "output_file",
                        "export_dir", "write_path", "ckpt_path",
                    ],
                    "context_keywords": [
                        "wb", ".write(", "save(",
                    ],
                },
                "_tensorflow": {
                    "path_patterns": [
                        r"gen_io_ops\.write",
                        r"gen_summary_ops\.",
                        r"checkpoint\.save",
                    ],
                    "path_keywords": [
                        "gen_io_ops.write", "tf.io.write",
                        "summary.write", "tfrecord_writer",
                        "saved_model.save", "checkpoint.save",
                    ],
                },
                "_pytorch": {
                    "path_patterns": [
                        r"torch\.save",
                        r"torch\.jit\.save",
                        r"torch\.onnx\.export",
                    ],
                    "path_keywords": [
                        "torch.save", "torch.jit.save",
                        "torch.onnx.export", "tensorboard",
                    ],
                },
                "_jax": {
                    "path_keywords": [
                        "checkpoint.save", "orbax.save",
                    ],
                },
            },

            # ===========================================================
            # DIR_READ
            # ===========================================================
            Labels.DIR_READ: {
                "_general": {
                    "path_patterns": [
                        r"list_dir",
                        r"listdir",
                        r"walk",
                        r"glob",
                        r"scandir",
                    ],
                    "path_keywords": [
                        "listdir", "list_dir", "walk", "glob",
                        "scandir", "list_files", "file_list",
                        "matching_files", "get_files",
                    ],
                    "func_names": [
                        "listdir", "walk", "glob", "scandir",
                    ],
                },
                "_tensorflow": {
                    "path_keywords": [
                        "gfile.listdir", "gfile.walk", "gfile.glob",
                        "matching_files", "gen_io_ops.matching",
                    ],
                },
            },

            # ===========================================================
            # DIR_WRITE
            # ===========================================================
            Labels.DIR_WRITE: {
                "_general": {
                    "path_patterns": [
                        r"make_?dir",
                        r"mkdir",
                        r"rmdir",
                        r"remove_?dir",
                        r"create_?dir",
                    ],
                    "path_keywords": [
                        "mkdir", "makedirs", "rmdir", "rmtree",
                        "create_dir", "make_dirs", "ensure_dir",
                    ],
                    "func_names": [
                        "mkdir", "makedirs", "rmdir", "rmtree",
                    ],
                },
                "_tensorflow": {
                    "path_keywords": [
                        "gfile.mkdir", "gfile.makedirs", "gfile.rmtree",
                    ],
                },
            },

            # ===========================================================
            # NETWORK_SEND
            # ===========================================================
            Labels.NETWORK_SEND: {
                "_general": {
                    "path_patterns": [
                        r"rpc_(?:call|send|async|sync)",
                        r"send_tensor",
                        r"all_reduce",
                        r"all_gather",
                        r"http.*(?:post|put|patch|request)",
                        # Removed: r"broadcast" — matches tensor shape ops (broadcast_in_dim, broadcast_to)
                        # Removed: r"scatter"   — matches tensor indexing ops (lax.scatter, torch.scatter)
                    ],
                    "path_keywords": [
                        "rpc_call", "rpc_send", "rpc_sync", "rpc_async",
                        "send", "isend", "all_reduce",
                        "all_gather", "reduce_scatter", "all_to_all",
                        "send_tensor", "collective_send",
                        # Removed: "broadcast" — tensor shape broadcasting, not network broadcast
                        # Removed: "scatter"   — tensor scatter op, not distributed scatter
                    ],
                    "func_names": [
                        "send", "isend", "broadcast", "all_reduce",
                        "rpc_sync", "rpc_async", "rpc_call",
                    ],
                    "arg_keywords": [
                        "dst", "dest", "remote", "endpoint", "port", "url",
                        # Removed: "target" (matches recall_target, target_shape, etc.)
                        # Removed: "host" (matches host_memory, host_device, etc.)
                        # Removed: "address" (matches address_space, memory ops)
                    ],
                },
                "_tensorflow": {
                    "path_patterns": [
                        r"gen_rpc_ops\.rpc_(?:call|send|client)",
                        r"gen_collective_ops\.",
                        r"cross_replica_ops\.",
                    ],
                    "path_keywords": [
                        "gen_rpc_ops", "collective_ops",
                        "cross_device_ops", "cross_replica",
                        "distribute.experimental.rpc",
                    ],
                },
                "_pytorch": {
                    "path_patterns": [
                        r"torch\.distributed\.(?:send|isend|broadcast|all_reduce|all_gather)",
                        r"torch\.distributed\.rpc",
                        r"c10d\.",
                    ],
                    "path_keywords": [
                        "torch.distributed.send", "torch.distributed.broadcast",
                        "torch.distributed.rpc", "process_group.send",
                        "c10d.", "nccl.", "gloo.",
                    ],
                },
                "_jax": {
                    "path_patterns": [
                        r"jax\.lax\.p(?:sum|mean|max|min|all_gather)",
                        r"jax\.distributed",
                    ],
                    "path_keywords": [
                        "lax.psum", "lax.pmean", "lax.pmax",
                        "lax.all_gather", "lax.ppermute",
                        "jax.distributed",
                    ],
                },
            },

            # ===========================================================
            # NETWORK_RECEIVE
            # ===========================================================
            Labels.NETWORK_RECEIVE: {
                "_general": {
                    "path_patterns": [
                        r"rpc_(?:recv|receive|check_status|get_result)",
                        r"recv_tensor",
                        r"http.*(?:get|response|fetch)",
                    ],
                    "path_keywords": [
                        "recv", "irecv", "receive",
                        "rpc_check_status", "rpc_get",
                        "recv_tensor", "collective_recv",
                        "fetch", "download",
                    ],
                    "func_names": [
                        "recv", "irecv", "receive", "fetch", "download",
                    ],
                },
                "_tensorflow": {
                    "path_patterns": [
                        r"gen_rpc_ops\.rpc_(?:check_status|get_value)",
                    ],
                    "path_keywords": [
                        "gen_rpc_ops.rpc_check_status",
                    ],
                },
                "_pytorch": {
                    "path_patterns": [
                        r"torch\.distributed\.(?:recv|irecv)",
                    ],
                    "path_keywords": [
                        "torch.distributed.recv",
                        "process_group.recv",
                    ],
                },
            },

            # ===========================================================
            # PROCESS_CREATE
            # ===========================================================
            Labels.PROCESS_CREATE: {
                "_general": {
                    "path_patterns": [
                        r"spawn",
                        r"fork",
                        r"start_server",
                        r"create_server",
                        r"multiprocessing.*start",
                        r"launch.*worker",
                    ],
                    "path_keywords": [
                        "spawn", "fork", "start_server", "create_server",
                        "launch", "start_worker", "multiprocessing",
                        "new_process", "create_worker", "subprocess",
                        "thread_pool", "process_pool", "executor",
                    ],
                    "func_names": [
                        "spawn", "fork", "start", "launch",
                    ],
                },
                "_tensorflow": {
                    "path_patterns": [
                        r"gen_rpc_ops\.rpc_(?:server|client)",
                        r"server\.start",
                        r"cluster_resolver",
                    ],
                    "path_keywords": [
                        "gen_rpc_ops.rpc_server", "gen_rpc_ops.rpc_client",
                        "server.start", "distribute.Server",
                        "multi_worker", "parameter_server",
                        "coordinator", "cluster_resolver",
                    ],
                },
                "_pytorch": {
                    "path_patterns": [
                        r"torch\.multiprocessing\.spawn",
                        r"torch\.distributed\.launch",
                        r"elastic\.agent",
                    ],
                    "path_keywords": [
                        "mp.spawn", "multiprocessing.spawn",
                        "distributed.launch", "init_process_group",
                        "elastic", "rendezvous",
                    ],
                },
                "_jax": {
                    "path_keywords": [
                        "jax.distributed.initialize",
                        "multihost", "multi_process",
                    ],
                },
            },

            # ===========================================================
            # PROCESS_ABORT
            # ===========================================================
            Labels.PROCESS_ABORT: {
                "_general": {
                    "path_patterns": [
                        r"abort",
                        r"terminate",
                        r"kill",
                        r"shutdown",
                        r"destroy",
                    ],
                    "path_keywords": [
                        "abort", "terminate", "kill", "shutdown",
                        "destroy_process_group", "close_server",
                        "stop_server", "cancel",
                    ],
                    "func_names": [
                        "abort", "terminate", "kill", "shutdown",
                        "destroy", "close", "stop", "cancel",
                    ],
                },
                "_tensorflow": {
                    "path_keywords": [
                        "gen_rpc_ops.delete_rpc", "gen_rpc_ops.shutdown",
                        "server.stop", "coordinator.request_stop",
                    ],
                },
                "_pytorch": {
                    "path_keywords": [
                        "destroy_process_group", "rpc.shutdown",
                    ],
                },
            },

            # ===========================================================
            # PROCESS_SLEEP
            # ===========================================================
            Labels.PROCESS_SLEEP: {
                "_general": {
                    "path_patterns": [
                        r"sleep",
                        r"wait",
                        r"barrier",
                        r"synchronize",
                        r"block_until",
                    ],
                    "path_keywords": [
                        "sleep", "wait", "barrier", "synchronize",
                        "block_until", "await_result", "join",
                        "wait_for_ready", "event.wait",
                    ],
                    "func_names": [
                        "sleep", "wait", "barrier", "synchronize", "join",
                    ],
                },
                "_tensorflow": {
                    "path_keywords": [
                        "gen_rpc_ops.rpc_check_status",  # blocks until ready
                        "coordinator.wait", "barrier",
                    ],
                },
                "_pytorch": {
                    "path_keywords": [
                        "barrier", "torch.cuda.synchronize",
                        "dist.barrier", "future.wait",
                    ],
                },
                "_jax": {
                    "path_keywords": [
                        "block_until_ready", "jax.effects_barrier",
                    ],
                },
            },

            # ===========================================================
            # CALC_MATH
            # ===========================================================
            Labels.CALC_MATH: {
                "_general": {
                    "path_patterns": [
                        r"(?:^|\.)(add|sub|mul|div|matmul|dot|mm|bmm|einsum)(?:\.|$|_)",
                        r"(?:^|\.)conv(?:olution|[12]d|_general|_transpose)",
                        r"(?:^|\.)(relu|sigmoid|tanh|softmax|gelu|silu|swish|elu|leaky_relu)",
                        r"(?:^|\.)(batch_norm|layer_norm|group_norm|instance_norm)",
                        r"(?:^|\.)(cross_entropy|nll_loss|mse_loss|l1_loss|binary_cross)",
                        r"(?:^|\.)(svd|eig|cholesky|qr|lu|det|inv|solve|lstsq)",
                        r"(?:^|\.)(fft|ifft|rfft|irfft|stft)",
                        r"gen_math_ops\.",
                        r"gen_nn_ops\.",
                        r"gen_linalg_ops\.",
                        r"gen_spectral_ops\.",
                        r"ops\.aten\.",
                    ],
                    "path_keywords": [
                        # Arithmetic
                        "add", "subtract", "multiply", "divide", "matmul",
                        "dot", "einsum", "outer", "inner", "cross", "cumsum",
                        "cumprod", "addmm", "baddbmm",
                        # Activations
                        "relu", "sigmoid", "tanh", "softmax", "gelu",
                        "leaky_relu", "elu", "silu", "swish", "mish",
                        # Normalization
                        "batch_norm", "layer_norm", "group_norm",
                        # Loss
                        "cross_entropy", "nll_loss", "mse_loss", "l1_loss",
                        "binary_cross_entropy", "huber_loss", "triplet_margin",
                        # Linear algebra
                        "svd", "eig", "cholesky", "qr_", "lu_", "det",
                        "inverse", "solve", "lstsq", "norm",
                        # Convolution
                        "conv1d", "conv2d", "conv3d", "conv_transpose",
                        "conv_general_dilated", "depthwise_conv",
                        # Pooling
                        "max_pool", "avg_pool", "adaptive_pool",
                        # Signal processing
                        "fft", "ifft", "rfft", "stft",
                        # Reduction
                        "reduce_sum", "reduce_mean", "reduce_max", "reduce_min",
                        "reduce_prod",
                        # Gradient computation
                        "gradient", "grad", "backward", "jvp", "vjp",
                    ],
                },
                "_tensorflow": {
                    "path_keywords": [
                        "gen_math_ops", "gen_nn_ops", "gen_linalg_ops",
                        "gen_spectral_ops", "gen_audio_ops",
                    ],
                },
                "_pytorch": {
                    "path_keywords": [
                        "aten.add", "aten.mm", "aten.addmm", "aten.bmm",
                        "aten.convolution", "aten.native_batch_norm",
                        "aten._softmax", "aten.relu", "aten.sigmoid",
                        "torch._C._nn",
                    ],
                },
                "_jax": {
                    "path_keywords": [
                        "lax.add", "lax.mul", "lax.dot", "lax.conv",
                        "lax.dot_general", "lax.conv_general_dilated",
                        "lax.reduce", "lax.cumsum", "lax.cumprod",
                        "_src.lax", "lax.linalg",
                    ],
                },
            },

            # ===========================================================
            # CALC_DATA_MGMT
            # ===========================================================
            Labels.CALC_DATA_MGMT: {
                "_general": {
                    "path_patterns": [
                        r"(?:^|\.)(reshape|transpose|permute|concat|stack|split|slice|gather|scatter)",
                        r"(?:^|\.)(squeeze|unsqueeze|expand|view|contiguous|clone|copy)",
                        r"(?:^|\.)(zeros|ones|empty|full|arange|linspace|eye|rand|randn)",
                        r"(?:^|\.)(to_tensor|as_tensor|from_numpy|numpy|to_numpy)",
                        r"gen_array_ops\.",
                        r"gen_state_ops\.",
                        r"gen_resource_variable_ops\.",
                    ],
                    "path_keywords": [
                        # Shape manipulation
                        "reshape", "transpose", "permute", "flatten",
                        "squeeze", "unsqueeze", "expand_dims", "broadcast",
                        "tile", "repeat", "roll", "flip",
                        # Combining/splitting
                        "concat", "concatenate", "stack", "vstack", "hstack",
                        "split", "chunk", "unbind", "unstack",
                        # Indexing/selection
                        "gather", "scatter", "index_select", "masked_select",
                        "where", "take", "put", "index_put", "advanced_index",
                        "slice", "narrow", "select", "dynamic_slice",
                        # Creation
                        "zeros", "ones", "empty", "full", "arange", "linspace",
                        "eye", "rand", "randn", "normal", "uniform",
                        "zeros_like", "ones_like", "empty_like", "full_like",
                        # Conversion
                        "to_tensor", "as_tensor", "from_numpy", "numpy",
                        "cast", "type_as", "to_dtype", "astype", "convert_to_tensor",
                        # Memory management
                        "clone", "copy", "contiguous", "pin_memory",
                        "to_device", "cpu", "cuda", "gpu", "device",
                        # Variable/state ops
                        "assign", "assign_add", "assign_sub", "variable",
                        "get_variable", "create_variable",
                        # Sorting/searching
                        "sort", "argsort", "topk", "unique", "searchsorted",
                    ],
                },
                "_tensorflow": {
                    "path_keywords": [
                        "gen_array_ops", "gen_state_ops",
                        "gen_resource_variable_ops", "gen_data_flow_ops",
                        "gen_random_ops", "gen_stateless_random_ops",
                    ],
                },
                "_pytorch": {
                    "path_keywords": [
                        "aten.reshape", "aten.t", "aten.transpose",
                        "aten.view", "aten.expand", "aten.permute",
                        "aten.cat", "aten.stack", "aten.split",
                        "aten.gather", "aten.scatter", "aten.index",
                        "aten.clone", "aten.contiguous",
                        "aten.zeros", "aten.ones", "aten.empty",
                        "aten._unsafe_view",
                    ],
                },
                "_jax": {
                    "path_keywords": [
                        "lax.reshape", "lax.transpose", "lax.broadcast",
                        "lax.concatenate", "lax.slice", "lax.dynamic_slice",
                        "lax.gather", "lax.scatter", "lax.pad",
                        "lax.sort", "lax.iota",
                        "jax.random", "jax.numpy",
                    ],
                },
            },

            # ===========================================================
            # CALC_ENCODE_DECODE
            # ===========================================================
            Labels.CALC_ENCODE_DECODE: {
                "_general": {
                    "path_patterns": [
                        r"(?:^|\.)(encode|decode|seriali[zs]|deseriali[zs])",
                        r"(?:^|\.)(compress|decompress|quantiz|dequantiz)",
                        r"(?:^|\.)(hash|fingerprint|checksum)",
                        r"(?:^|\.)(tokeniz|detokeniz|vocab)",
                        r"(?:^|\.)(base64|utf8|unicode|ascii)",
                        r"(?:^|\.)(?:de|en)code_(?:image|jpeg|png|wav|audio|video|raw|proto)",
                        r"gen_string_ops\.",
                        r"gen_image_ops\.(?:en|de)code",
                        r"gen_audio_ops\.(?:en|de)code",
                        r"gen_encode_ops\.",
                    ],
                    "path_keywords": [
                        "encode", "decode", "serialize", "deserialize",
                        "compress", "decompress", "zip", "unzip", "gzip",
                        "quantize", "dequantize", "fake_quant",
                        "hash", "fingerprint",
                        "tokenize", "detokenize", "bpe", "sentencepiece",
                        "base64", "utf8", "unicode",
                        "encode_jpeg", "decode_jpeg", "encode_png", "decode_png",
                        "decode_raw", "decode_csv", "parse_tensor",
                        "to_bytes", "from_bytes", "pack", "unpack",
                    ],
                },
                "_tensorflow": {
                    "path_keywords": [
                        "gen_string_ops", "gen_image_ops.decode",
                        "gen_image_ops.encode", "gen_audio_ops.decode",
                        "gen_audio_ops.encode", "gen_encode_ops",
                        "gen_parsing_ops", "serialize_tensor",
                        "deserialize", "proto",
                    ],
                },
                "_pytorch": {
                    "path_keywords": [
                        "quantize", "dequantize", "torch.quantization",
                        "ops.quantized", "fake_quantize",
                        "torchvision.io.decode", "torchvision.io.encode",
                        "torchaudio.transforms",
                    ],
                },
                "_jax": {
                    "path_keywords": [
                        "serialization", "msgpack",
                    ],
                },
            },

            # ===========================================================
            # CODE_EXECUTION
            # ===========================================================
            Labels.CODE_EXECUTION: {
                "_general": {
                    "path_patterns": [
                        r"(?:^|\.)(eval|exec|compile|apply|call|run|invoke)",
                        r"py_func",
                        r"py_function",
                        r"numpy_function",
                        r"map_fn",
                        r"while_loop",
                        r"cond\b",
                        r"switch_case",
                        r"jit",
                        r"autograph",
                        r"trace",
                        r"make_jaxpr",
                    ],
                    "path_keywords": [
                        # Direct code execution
                        "py_func", "py_function", "numpy_function",
                        "eager_py_func", "wrapped_function",
                        # Functional primitives that execute user code
                        "map_fn", "map_and_batch", "parallel_map",
                        "flat_map", "interleave",
                        # Control flow (executes user-provided callables)
                        "while_loop", "fori_loop", "scan",
                        "cond", "switch_case", "case",
                        # JIT compilation (converts user code to executable)
                        "jit", "tf.function", "autograph",
                        "compile", "torch.compile", "dynamo",
                        # Function tracing
                        "trace", "make_jaxpr", "concrete_function",
                        "get_concrete_function",
                        # Callbacks & hooks
                        "register_hook", "register_forward_hook",
                        "register_backward_hook", "add_callback",
                        "custom_gradient", "custom_jvp", "custom_vjp",
                        # Apply functions
                        "apply", "apply_gradients", "run_functions",
                    ],
                    "func_names": [
                        "apply", "run", "execute",
                        "jit", "compile", "trace",
                        # Removed: "call" — too generic, every Python class has __call__
                    ],
                },
                "_tensorflow": {
                    "path_patterns": [
                        r"gen_functional_ops\.",
                        r"gen_control_flow_ops\.",
                        r"gen_script_ops\.",
                    ],
                    "path_keywords": [
                        "gen_functional_ops", "gen_control_flow_ops",
                        "gen_script_ops", "eager_py_func",
                        "tf.function", "autograph.to_graph",
                        "session.run",
                    ],
                },
                "_pytorch": {
                    "path_patterns": [
                        r"torch\._dynamo",
                        r"torch\.compile",
                        r"torch\.jit\.(trace|script)",
                    ],
                    "path_keywords": [
                        "torch.jit.trace", "torch.jit.script",
                        "torch._dynamo", "torch.compile",
                        "torch._inductor", "torch.fx.symbolic_trace",
                        "torch.autograd.Function.apply",
                    ],
                },
                "_jax": {
                    "path_patterns": [
                        r"jax\.jit",
                        r"jax\.make_jaxpr",
                        r"jax\.eval_shape",
                    ],
                    "path_keywords": [
                        "jax.jit", "jax.pmap", "jax.vmap",
                        "jax.make_jaxpr", "jax.eval_shape",
                        "lax.while_loop", "lax.fori_loop",
                        "lax.scan", "lax.cond", "lax.switch",
                        "jax.custom_jvp", "jax.custom_vjp",
                        "jax.checkpoint", "jax.remat",
                        "callback", "debug.callback",
                        "pure_callback", "io_callback",
                    ],
                },
            },
        }


# =============================================================================
# Layer 2: Docstring / Signature Analyzer
# =============================================================================

class DocstringAnalyzer:
    """
    Infers behavioral labels by analyzing function docstrings and
    type annotations from the actual source code.

    Looks for clues like:
      - "Reads a file..." → file_read
      - "Returns: Tensor" → likely calc_*
      - "Args: filename (str)" → likely file_*
      - "Sends tensor to..." → network_send
    """

    # Keywords in docstrings that suggest specific behaviors
    DOCSTRING_SIGNALS = {
        Labels.FILE_READ: [
            r"reads?\s+(a\s+)?file", r"load(s|ing)?\s+from\s+disk",
            r"open(s|ing)?\s+(a\s+)?file", r"deserializ",
            r"restore(s|ing)?\s+from", r"read(s|ing)?\s+from\s+(?:disk|path|filesystem)",
        ],
        Labels.FILE_WRITE: [
            r"writes?\s+(to\s+)?(a\s+)?file", r"save(s|ing)?\s+to\s+disk",
            r"export(s|ing)?\s+to", r"serializ(e|ing)\s+to",
            r"dump(s|ing)?\s+to", r"persist(s|ing)?",
        ],
        Labels.NETWORK_SEND: [
            r"sends?\s+(a\s+)?tensor", r"sends?\s+(data|message|request)",
            r"remote\s+procedure\s+call", r"rpc", r"transmit",
            # Removed: r"broadcast(s|ing)?" — tensor docstrings use "broadcast" for shape ops
        ],
        Labels.NETWORK_RECEIVE: [
            r"receives?\s+(a\s+)?tensor", r"receives?\s+(data|message|response)",
            r"fetch(es|ing)?", r"download(s|ing)?",
        ],
        Labels.PROCESS_CREATE: [
            r"spawn(s|ing)?", r"fork(s|ing)?", r"create(s|ing)?\s+(a\s+)?process",
            r"launch(es|ing)?", r"start(s|ing)?\s+(a\s+)?server",
        ],
        Labels.PROCESS_SLEEP: [
            r"wait(s|ing)?(\s+for)?", r"block(s|ing)?(\s+until)?",
            r"sleep(s|ing)?", r"barrier", r"synchroniz",
        ],
        Labels.CALC_MATH: [
            r"comput(e|es|ing)", r"calculat(e|es|ing)",
            r"matrix\s+multipli", r"element.?wise",
            r"convol(ve|ution)", r"activation\s+function",
            r"loss\s+function", r"gradient",
        ],
        Labels.CALC_ENCODE_DECODE: [
            r"encod(e|es|ing)", r"decod(e|es|ing)",
            r"compress(es|ing)?", r"quantiz(e|es|ing)",
            r"seriali[zs]", r"hash(es|ing)?",
        ],
        Labels.CODE_EXECUTION: [
            r"execut(e|es|ing)\s+(?:a\s+)?(?:user|python|callable)",
            r"call(s|ing)?\s+(?:the\s+)?(?:user|python|function)",
            r"trac(e|es|ing)\s+(?:a\s+)?function",
            r"compil(e|es|ing)\s+(?:a\s+)?function",
            r"JIT", r"just.in.time",
        ],
    }

    # Type annotation patterns that suggest behavior
    TYPE_SIGNALS = {
        Labels.FILE_READ: [
            r"(?:file|path|filename).*(?:str|Path|PathLike)",
            r"IO\[", r"TextIO", r"BinaryIO", r"BufferedReader",
        ],
        Labels.FILE_WRITE: [
            r"(?:output|save|write).*(?:str|Path|PathLike)",
            r"BufferedWriter",
        ],
        Labels.CALC_MATH: [
            r"(?:Tensor|ndarray|Array).*->.*(?:Tensor|ndarray|Array)",
        ],
    }

    def __init__(self, repo_paths: dict = None):
        """
        Args:
            repo_paths: dict of {"tensorflow": path, "pytorch": path, "jax": path}
        """
        self.repo_paths = repo_paths or {}
        self._docstring_cache = {}

    def analyze(self, record: dict) -> list[str]:
        """Infer additional labels from docstring analysis."""
        api_path = record.get("api_path", "")
        framework = record.get("framework", "")

        docstring = self._get_docstring(api_path, framework)
        if not docstring:
            return []

        labels = set()
        docstring_lower = docstring.lower()

        for label, patterns in self.DOCSTRING_SIGNALS.items():
            for pattern in patterns:
                if re.search(pattern, docstring_lower):
                    labels.add(label)
                    break

        return sorted(labels)

    def _get_docstring(self, api_path: str, framework: str) -> Optional[str]:
        """Try to retrieve the docstring for an API function."""
        cache_key = f"{framework}:{api_path}"
        if cache_key in self._docstring_cache:
            return self._docstring_cache[cache_key]

        docstring = None
        repo_path = self.repo_paths.get(framework)

        if repo_path:
            docstring = self._find_docstring_in_repo(api_path, framework, repo_path)

        self._docstring_cache[cache_key] = docstring
        return docstring

    def _find_docstring_in_repo(self, api_path: str, framework: str, repo_path: str) -> Optional[str]:
        """Search the repo for the function's docstring."""
        # Convert api_path to a likely file path
        parts = api_path.split(".")
        func_name = parts[-1]

        # Try to find the file by walking path segments
        repo = Path(repo_path)
        # Build candidate paths
        for i in range(len(parts) - 1, 0, -1):
            candidate = repo / "/".join(parts[:i]) / "__init__.py"
            if not candidate.exists():
                candidate = Path(str(repo / "/".join(parts[:i])) + ".py")
            if not candidate.exists():
                continue

            try:
                source = candidate.read_text(errors="ignore")
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if node.name == func_name:
                            return ast.get_docstring(node) or ""
            except (SyntaxError, UnicodeDecodeError):
                continue

        return None


# =============================================================================
# Layer 3: Review Queue & Conflict Resolver
# =============================================================================

class ReviewQueueBuilder:
    """
    Identifies records that need human review:
    - Low confidence labels
    - Conflicting signals (e.g., looks like both file_read and network_receive)
    - Unknown API paths with no rule match
    """

    CONFLICT_PAIRS = [
        # These pairs rarely co-occur, so flag for review
        ({Labels.FILE_READ}, {Labels.NETWORK_RECEIVE}),
        ({Labels.FILE_WRITE}, {Labels.NETWORK_SEND}),
        ({Labels.CALC_MATH}, {Labels.CODE_EXECUTION}),
    ]

    def __init__(self, confidence_threshold: float = 0.5):
        self.threshold = confidence_threshold
        self.queue = []

    def check(self, record: dict, labels: list[str], confidence: float,
              rule_labels: list[str], doc_labels: list[str]):
        """Check if a record needs review."""
        reasons = []

        # Low confidence
        if confidence < self.threshold:
            reasons.append(f"low_confidence ({confidence:.2f})")

        # No labels assigned
        if not labels:
            reasons.append("no_labels")

        # Conflicting signals between rule and docstring
        if doc_labels and rule_labels:
            rule_set = set(rule_labels)
            doc_set = set(doc_labels)
            if rule_set != doc_set:
                diff = rule_set.symmetric_difference(doc_set)
                if diff:
                    reasons.append(f"rule_doc_conflict: {diff}")

        # Unlikely label combinations
        label_set = set(labels)
        for pair_a, pair_b in self.CONFLICT_PAIRS:
            if pair_a.issubset(label_set) and pair_b.issubset(label_set):
                reasons.append(f"conflicting_labels: {pair_a} + {pair_b}")

        if reasons:
            self.queue.append({
                **record,
                "assigned_labels": labels,
                "confidence": confidence,
                "review_reasons": reasons,
                "rule_labels": rule_labels,
                "docstring_labels": doc_labels,
            })

    def get_queue(self) -> list[dict]:
        return self.queue


# =============================================================================
# Multi-Label Combination Rules
# =============================================================================

class MultiLabelResolver:
    """
    Handles known multi-label patterns where specific APIs
    always exhibit multiple behaviors.

    For example:
    - rpc_call → always [network_send, network_receive, code_execution]
    - torch.save → always [file_write, calc_encode_decode]
    - tf.function → always [code_execution]
    """

    # Known multi-label patterns: if api_path matches key, ALWAYS include these labels
    KNOWN_MULTI_LABELS = {
        # --- RPC (always send + receive + code exec) ---
        "rpc_call": [Labels.NETWORK_SEND, Labels.NETWORK_RECEIVE, Labels.CODE_EXECUTION],
        "rpc_sync": [Labels.NETWORK_SEND, Labels.NETWORK_RECEIVE, Labels.CODE_EXECUTION],
        "rpc_async": [Labels.NETWORK_SEND, Labels.NETWORK_RECEIVE, Labels.CODE_EXECUTION],

        # --- Client creation (process + network) ---
        "rpc_client": [Labels.PROCESS_CREATE, Labels.NETWORK_SEND],
        "rpc_server": [Labels.PROCESS_CREATE, Labels.NETWORK_RECEIVE],

        # --- Save/Load (file + encoding) ---
        "torch.save": [Labels.FILE_WRITE, Labels.CALC_ENCODE_DECODE],
        "torch.load": [Labels.FILE_READ, Labels.CALC_ENCODE_DECODE],
        "tf.io.write_file": [Labels.FILE_WRITE, Labels.CALC_ENCODE_DECODE],
        "tf.io.read_file": [Labels.FILE_READ, Labels.CALC_ENCODE_DECODE],
        "save_model": [Labels.FILE_WRITE, Labels.CALC_ENCODE_DECODE],
        "load_model": [Labels.FILE_READ, Labels.CALC_ENCODE_DECODE],
        "save_weights": [Labels.FILE_WRITE, Labels.CALC_ENCODE_DECODE],
        "load_weights": [Labels.FILE_READ, Labels.CALC_ENCODE_DECODE],
        "save_checkpoint": [Labels.FILE_WRITE, Labels.CALC_ENCODE_DECODE],
        "load_checkpoint": [Labels.FILE_READ, Labels.CALC_ENCODE_DECODE],
        "save_state_dict": [Labels.FILE_WRITE, Labels.CALC_ENCODE_DECODE],
        "load_state_dict": [Labels.FILE_READ, Labels.CALC_ENCODE_DECODE],

        # --- Distributed collectives (network + math) ---
        "all_reduce": [Labels.NETWORK_SEND, Labels.NETWORK_RECEIVE, Labels.CALC_MATH],
        "all_gather": [Labels.NETWORK_SEND, Labels.NETWORK_RECEIVE, Labels.CALC_DATA_MGMT],
        "reduce_scatter": [Labels.NETWORK_SEND, Labels.NETWORK_RECEIVE, Labels.CALC_MATH],
        "psum": [Labels.NETWORK_SEND, Labels.NETWORK_RECEIVE, Labels.CALC_MATH],
        "pmean": [Labels.NETWORK_SEND, Labels.NETWORK_RECEIVE, Labels.CALC_MATH],

        # --- JIT/Compilation (code execution + whatever the function does) ---
        "tf.function": [Labels.CODE_EXECUTION],
        "torch.compile": [Labels.CODE_EXECUTION],
        "torch.jit.trace": [Labels.CODE_EXECUTION],
        "torch.jit.script": [Labels.CODE_EXECUTION],
        "jax.jit": [Labels.CODE_EXECUTION],

        # --- Data pipelines (file read + code execution for map functions) ---
        "map_fn": [Labels.CODE_EXECUTION],
        "flat_map": [Labels.CODE_EXECUTION],
        "interleave": [Labels.CODE_EXECUTION],
        "from_generator": [Labels.CODE_EXECUTION],

        # --- Image encode/decode (encoding + potentially file access) ---
        "decode_jpeg": [Labels.CALC_ENCODE_DECODE],
        "decode_png": [Labels.CALC_ENCODE_DECODE],
        "encode_jpeg": [Labels.CALC_ENCODE_DECODE],
        "encode_png": [Labels.CALC_ENCODE_DECODE],
        "decode_image": [Labels.CALC_ENCODE_DECODE],

        # --- Process group init (process + network) ---
        "init_process_group": [Labels.PROCESS_CREATE, Labels.NETWORK_SEND, Labels.NETWORK_RECEIVE],
        "destroy_process_group": [Labels.PROCESS_ABORT],
        "jax.distributed.initialize": [Labels.PROCESS_CREATE, Labels.NETWORK_SEND, Labels.NETWORK_RECEIVE],
    }

    def resolve(self, labels: list[str], api_path: str) -> list[str]:
        """Add known multi-label expansions."""
        expanded = set(labels)
        path_lower = api_path.lower()
        func_name = api_path.split(".")[-1].lower()

        for pattern, forced_labels in self.KNOWN_MULTI_LABELS.items():
            pattern_lower = pattern.lower()
            if pattern_lower in path_lower or func_name == pattern_lower:
                expanded.update(forced_labels)

        return sorted(expanded)


# =============================================================================
# Main Labeling Pipeline
# =============================================================================

class LabelingPipeline:
    """Orchestrates the full labeling process."""

    def __init__(self, args):
        self.args = args
        self.rule_labeler = RuleBasedLabeler()
        self.multi_resolver = MultiLabelResolver()
        self.review_builder = ReviewQueueBuilder(
            confidence_threshold=args.review_threshold
        )

        # Optional docstring analyzer
        self.doc_analyzer = None
        repo_paths = {}
        if args.tf_repo:
            repo_paths["tensorflow"] = args.tf_repo
        if args.pytorch_repo:
            repo_paths["pytorch"] = args.pytorch_repo
        if args.jax_repo:
            repo_paths["jax"] = args.jax_repo
        if repo_paths:
            self.doc_analyzer = DocstringAnalyzer(repo_paths)

        self.stats = defaultdict(int)
        self.label_counts = defaultdict(int)
        self.co_occurrence = defaultdict(int)

    def run(self):
        """Run the full labeling pipeline."""
        logger.info("=" * 70)
        logger.info("MULTI-LABEL API BEHAVIOR LABELING PIPELINE")
        logger.info("=" * 70)

        input_path = Path(self.args.input)
        output_path = Path(self.args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = self._load_records(input_path)
        labeled_records = []

        for i, record in enumerate(records):
            if i > 0 and i % 5000 == 0:
                logger.info(f"  Labeled {i}/{len(records)} records...")

            labeled = self._label_one(record)
            labeled_records.append(labeled)

        # Write labeled dataset
        self._write_output(labeled_records, output_path)

        # Write review queue
        if self.args.review_queue:
            self._write_review_queue(Path(self.args.review_queue))

        # Write label stats
        stats_path = output_path.with_name(output_path.stem + "_label_stats.json")
        self._write_stats(stats_path)

        self._print_summary(len(records))

    def _label_one(self, record: dict) -> dict:
        """Label a single record."""

        # Layer 1: Rule-based
        rule_labels, confidence = self.rule_labeler.label(record)
        self.stats["rule_labeled"] += 1 if rule_labels else 0

        # Layer 2: Docstring analysis (if available)
        doc_labels = []
        if self.doc_analyzer:
            doc_labels = self.doc_analyzer.analyze(record)
            if doc_labels:
                self.stats["doc_enhanced"] += 1

        # Merge: union of rule and docstring labels
        merged = sorted(set(rule_labels) | set(doc_labels))

        # Multi-label resolution (add known co-occurring labels)
        final_labels = self.multi_resolver.resolve(merged, record.get("api_path", ""))

        # Layer 3: Review queue check
        self.review_builder.check(record, final_labels, confidence, rule_labels, doc_labels)

        # Update stats
        for label in final_labels:
            self.label_counts[label] += 1
        for i, l1 in enumerate(final_labels):
            for l2 in final_labels[i+1:]:
                pair = tuple(sorted([l1, l2]))
                self.co_occurrence[f"{pair[0]} + {pair[1]}"] += 1
        if len(final_labels) > 1:
            self.stats["multi_label"] += 1
        if not final_labels:
            self.stats["unlabeled"] += 1

        # Build output record
        record["labels"] = final_labels
        record["label_groups"] = self._group_labels(final_labels)
        record["label_confidence"] = round(confidence, 3)
        record["label_source"] = {
            "rule_based": rule_labels,
            "docstring": doc_labels,
            "multi_label_resolved": [l for l in final_labels if l not in merged],
        }
        # Binary vector for ML training
        record["label_vector"] = [1 if l in final_labels else 0 for l in Labels.ALL]
        # Syscall sequence derived from behavioral labels — used when training
        # with --input-mode syscall in preprocess_dataset.py
        record["syscalls"] = derive_syscalls(final_labels)

        return record

    def _group_labels(self, labels: list[str]) -> dict:
        """Group labels by category for readability."""
        groups = {}
        for group_name, group_labels in Labels.GROUPS.items():
            active = [l for l in labels if l in group_labels]
            if active:
                groups[group_name] = active
        return groups

    def _load_records(self, path: Path) -> list[dict]:
        """Load records from JSONL."""
        records = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        logger.info(f"Loaded {len(records)} records from {path}")
        return records

    def _write_output(self, records: list[dict], path: Path):
        """Write labeled records to JSONL."""
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(records)} labeled records to {path}")

    def _write_review_queue(self, path: Path):
        """Write the review queue."""
        queue = self.review_builder.get_queue()
        with open(path, "w") as f:
            for item in queue:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(queue)} items to review queue at {path}")

    def _write_stats(self, path: Path):
        """Write label distribution statistics."""
        stats = {
            "label_counts": dict(sorted(self.label_counts.items(), key=lambda x: -x[1])),
            "top_co_occurrences": dict(sorted(
                self.co_occurrence.items(), key=lambda x: -x[1])[:30]
            ),
            "pipeline_stats": dict(self.stats),
            "label_definitions": {
                label: {
                    "group": group_name,
                    "count": self.label_counts.get(label, 0),
                }
                for group_name, group_labels in Labels.GROUPS.items()
                for label in group_labels
            },
        }
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Wrote label stats to {path}")

    def _print_summary(self, total: int):
        """Print labeling summary."""
        logger.info("\n" + "=" * 70)
        logger.info("LABELING SUMMARY")
        logger.info("=" * 70)

        logger.info(f"\n  Total records:     {total:,}")
        logger.info(f"  Rule-labeled:      {self.stats['rule_labeled']:,}")
        logger.info(f"  Doc-enhanced:      {self.stats['doc_enhanced']:,}")
        logger.info(f"  Multi-label:       {self.stats['multi_label']:,}")
        logger.info(f"  Unlabeled:         {self.stats['unlabeled']:,}")
        logger.info(f"  Review queue:      {len(self.review_builder.queue):,}")

        logger.info("\n  LABEL DISTRIBUTION:")
        for label in Labels.ALL:
            count = self.label_counts.get(label, 0)
            pct = (count / max(total, 1)) * 100
            bar = "█" * int(pct)
            logger.info(f"    {label:>22}: {count:>8,}  ({pct:5.1f}%)  {bar}")

        logger.info("\n  TOP CO-OCCURRENCES:")
        for pair, count in sorted(self.co_occurrence.items(), key=lambda x: -x[1])[:10]:
            logger.info(f"    {pair:>45}: {count:,}")

        logger.info("\n  LABEL GROUP COVERAGE:")
        for group_name, group_labels in Labels.GROUPS.items():
            group_total = sum(self.label_counts.get(l, 0) for l in group_labels)
            logger.info(f"    {group_name:>22}: {group_total:,}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Label API Behavior Labeler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Label an extracted dataset:
  python api_behavior_labeler.py \\
      --input ./unified_dataset.jsonl \\
      --output ./labeled_dataset.jsonl

  # With docstring analysis and review queue:
  python api_behavior_labeler.py \\
      --input ./unified_dataset.jsonl \\
      --output ./labeled_dataset.jsonl \\
      --review-queue ./review_queue.jsonl \\
      --tf-repo ./tensorflow \\
      --pytorch-repo ./pytorch \\
      --jax-repo ./jax

  # Lower review threshold to flag more for review:
  python api_behavior_labeler.py \\
      --input ./unified_dataset.jsonl \\
      --output ./labeled_dataset.jsonl \\
      --review-threshold 0.7
        """,
    )

    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL from deep_api_extractor.py")
    parser.add_argument("--output", type=str, required=True,
                        help="Output labeled JSONL")
    parser.add_argument("--review-queue", type=str, default=None,
                        help="Output JSONL for records needing human review")
    parser.add_argument("--review-threshold", type=float, default=0.5,
                        help="Confidence below this goes to review queue (default: 0.5)")
    parser.add_argument("--tf-repo", type=str, default=None,
                        help="TensorFlow repo path (enables docstring analysis)")
    parser.add_argument("--pytorch-repo", type=str, default=None,
                        help="PyTorch repo path (enables docstring analysis)")
    parser.add_argument("--jax-repo", type=str, default=None,
                        help="JAX repo path (enables docstring analysis)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline = LabelingPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
