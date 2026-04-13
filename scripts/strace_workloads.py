"""
strace_workloads.py — Executable Python code snippets for strace capture.

Each workload is a self-contained string that can be run as:
    python -c "<workload_code>"

The workloads are designed to actually TRIGGER the OS syscalls characteristic
of each behavioral category, so strace captures real signals.

Frameworks: pytorch, tensorflow, jax
Categories: file_access, network_access, process_mgmt, pure_calculation, code_execution
"""

FRAMEWORKS = ["pytorch", "tensorflow", "jax"]
CATEGORIES = [
    "file_access",
    "network_access",
    "process_mgmt",
    "pure_calculation",
    "code_execution",
]

# ─── Baselines ────────────────────────────────────────────────────────────────
# Minimal framework warm-up — used as baseline for delta computation.
# Captures all import-time syscalls (SO loading, .pyc reads, etc.) that we want
# to subtract from the workload runs.

_BASELINES: dict[str, str] = {
    "pytorch": """\
import torch as _t
_x = _t.zeros(1)
del _x
""",
    "tensorflow": """\
import os as _os
_os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tensorflow as _tf
_x = _tf.constant(0.0)
del _x
""",
    "jax": """\
import os as _os
_os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax as _j
import jax.numpy as _jnp
_x = _jnp.zeros(1)
_x.block_until_ready()
del _x
""",
}

# ─── Workloads ────────────────────────────────────────────────────────────────

_WORKLOADS: dict[tuple[str, str], str] = {}

# ── file_access ───────────────────────────────────────────────────────────────

_WORKLOADS[("pytorch", "file_access")] = """\
import os, tempfile, torch
x = torch.randn(512, 512)
with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
    fname = f.name
torch.save(x, fname)
y = torch.load(fname, weights_only=True)
os.unlink(fname)
# Also exercise directory listing
import glob
_ = glob.glob(tempfile.gettempdir() + '/*.pt')
"""

_WORKLOADS[("tensorflow", "file_access")] = """\
import os, tempfile
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tensorflow as tf
with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
    fname = f.name
x = tf.constant([1.0, 2.0, 3.0, 4.0])
tf.io.write_file(fname, tf.io.serialize_tensor(x))
raw = tf.io.read_file(fname)
y = tf.io.parse_tensor(raw, out_type=tf.float32)
os.unlink(fname)
# Also write text
with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='wb') as f:
    fname2 = f.name
tf.io.write_file(fname2, 'hello strace')
_ = tf.io.read_file(fname2)
os.unlink(fname2)
"""

_WORKLOADS[("jax", "file_access")] = """\
import os, tempfile
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax.numpy as jnp
import numpy as np
x = jnp.ones((512, 512))
with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
    fname = f.name
np.save(fname, np.array(x))
y = jnp.array(np.load(fname))
os.unlink(fname)
# Also exercise npz
with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
    fname2 = f.name
np.savez(fname2, a=np.array(x), b=np.array(x[:10]))
data = np.load(fname2)
_ = data['a']
os.unlink(fname2 + '.npz' if not fname2.endswith('.npz') else fname2)
"""

# ── network_access ────────────────────────────────────────────────────────────
# Uses a pure Python TCP socket loop — reliable regardless of distributed
# framework support. Framework import is included so baseline subtraction works.

_NET_CORE = """\
import socket, threading, time

def _run_net():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', 0))
    port = server.getsockname()[1]
    server.listen(1)
    buf = []
    def _serve():
        try:
            conn, _ = server.accept()
            buf.append(conn.recv(4096))
            conn.sendall(b'pong' * 64)
            conn.close()
        finally:
            server.close()
    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c.connect(('127.0.0.1', port))
    c.sendall(b'ping' * 64)
    _ = c.recv(4096)
    c.close()
    t.join(timeout=5)

_run_net()
# Second round to exercise more send/recv syscalls
_run_net()
"""

_WORKLOADS[("pytorch", "network_access")] = """\
import torch
_x = torch.zeros(1)
""" + _NET_CORE

_WORKLOADS[("tensorflow", "network_access")] = """\
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tensorflow as tf
_x = tf.constant(0.0)
""" + _NET_CORE

_WORKLOADS[("jax", "network_access")] = """\
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax.numpy as jnp
_x = jnp.zeros(1)
""" + _NET_CORE

# ── process_mgmt ──────────────────────────────────────────────────────────────

_WORKLOADS[("pytorch", "process_mgmt")] = """\
import subprocess, multiprocessing as mp, sys, torch
torch.zeros(1)

# subprocess.run — triggers clone/execve
r = subprocess.run([sys.executable, '-c', 'import sys; sys.exit(0)'],
                   capture_output=True, timeout=15)

# multiprocessing spawn — triggers clone3 + execve in child
def _worker(q):
    q.put(42)

ctx = mp.get_context('spawn')
q = ctx.Queue()
p = ctx.Process(target=_worker, args=(q,))
p.start()
p.join(timeout=15)
_ = q.get_nowait() if not q.empty() else None

# Second subprocess for extra signal
r2 = subprocess.run([sys.executable, '-c', 'print("ok")'],
                    capture_output=True, timeout=10)
"""

_WORKLOADS[("tensorflow", "process_mgmt")] = """\
import os, subprocess, concurrent.futures, sys
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tensorflow as tf
tf.constant(0.0)

# subprocess
r = subprocess.run([sys.executable, '-c', 'pass'],
                   capture_output=True, timeout=15)

# ProcessPoolExecutor — triggers clone + execve
def _task():
    return 99

with concurrent.futures.ProcessPoolExecutor(max_workers=1) as ex:
    fut = ex.submit(_task)
    try:
        _ = fut.result(timeout=15)
    except Exception:
        pass

r2 = subprocess.run([sys.executable, '-c', 'print("tf ok")'],
                    capture_output=True, timeout=10)
"""

_WORKLOADS[("jax", "process_mgmt")] = """\
import os, subprocess, multiprocessing as mp, sys
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax.numpy as jnp
jnp.zeros(1)

# subprocess
r = subprocess.run([sys.executable, '-c', 'pass'],
                   capture_output=True, timeout=15)

# fork-based multiprocessing
ctx = mp.get_context('fork')
p = ctx.Process(target=lambda: None)
p.start()
p.join(timeout=10)

r2 = subprocess.run([sys.executable, '-c', 'print("jax ok")'],
                    capture_output=True, timeout=10)
"""

# ── pure_calculation ──────────────────────────────────────────────────────────
# Large tensors force the OS to allocate memory via mmap/brk.
# FFT and linear algebra trigger distinct memory access patterns.

_WORKLOADS[("pytorch", "pure_calculation")] = """\
import torch
N = 2048
x = torch.randn(N, N)
y = torch.randn(N, N)
# matmul — large intermediate allocation
z = torch.matmul(x, y)
# norm
w = torch.linalg.norm(z)
# fft — encode/decode flavor
out = torch.fft.fft2(x)
# element-wise
q = torch.relu(z) + torch.sigmoid(y)
del x, y, z, w, out, q
"""

_WORKLOADS[("tensorflow", "pure_calculation")] = """\
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tensorflow as tf
N = 2048
x = tf.random.normal([N, N])
y = tf.random.normal([N, N])
z = tf.linalg.matmul(x, y)
w = tf.linalg.norm(z)
q = tf.nn.relu(z) + tf.math.sigmoid(y)
del x, y, z, w, q
"""

_WORKLOADS[("jax", "pure_calculation")] = """\
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax, jax.numpy as jnp
N = 2048
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (N, N))
y = jax.random.normal(jax.random.PRNGKey(1), (N, N))
z = jnp.dot(x, y)
w = jnp.sum(z)
q = jnp.fft.fft2(x)
z.block_until_ready()
del x, y, z, w, q
"""

# ── code_execution ────────────────────────────────────────────────────────────
# JIT compilation writes cache files, may spawn subprocesses for compilation
# backends, and uses large mmap for compiled artifacts.

_WORKLOADS[("pytorch", "code_execution")] = """\
import torch

# torch.compile — Dynamo traces the function, spawning compilation workers
@torch.compile(backend='eager')
def _model(x, y):
    h = torch.matmul(x, y)
    h = torch.relu(h)
    return h + torch.linalg.norm(h)

x = torch.randn(128, 128)
y = torch.randn(128, 128)
out1 = _model(x, y)          # first call: compilation
out2 = _model(x + 0.01, y)   # second call: cache hit

# torch.jit.script — TorchScript compilation (different path from Dynamo)
@torch.jit.script
def _script_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b) + a.sum()

_ = _script_fn(x, y)
"""

_WORKLOADS[("tensorflow", "code_execution")] = """\
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tensorflow as tf

# @tf.function — XLA/XRT tracing, may write cache files
@tf.function
def _compute(x, y):
    z = tf.linalg.matmul(x, y)
    return z + tf.math.reduce_sum(y)

x = tf.random.normal([128, 128])
y = tf.random.normal([128, 128])
out1 = _compute(x, y)    # first call: traces and compiles
out2 = _compute(x, y)    # second call: uses cached trace

# Also trigger tf.function with input_signature for XLA compilation
@tf.function(input_signature=[
    tf.TensorSpec(shape=[128, 128], dtype=tf.float32),
    tf.TensorSpec(shape=[128, 128], dtype=tf.float32),
])
def _compiled(x, y):
    return tf.linalg.matmul(x, y)

_ = _compiled(x, y)
"""

_WORKLOADS[("jax", "code_execution")] = """\
import os, tempfile
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

# Set XLA compilation cache dir so JAX actually writes .pb files
_cache_dir = os.path.join(tempfile.gettempdir(), 'jax_xla_cache_strace')
os.makedirs(_cache_dir, exist_ok=True)

import jax, jax.numpy as jnp
try:
    jax.config.update('jax_compilation_cache_dir', _cache_dir)
except Exception:
    pass

# jax.jit — XLA compilation
@jax.jit
def _compute(x, y):
    return jnp.dot(x, y) + jnp.sum(y)

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (128, 128))
y = jax.random.normal(jax.random.PRNGKey(1), (128, 128))
out1 = _compute(x, y)         # compilation
out1.block_until_ready()
out2 = _compute(x, y)         # cache read
out2.block_until_ready()

# jax.make_jaxpr — forces lowering to XLA HLO
expr = jax.make_jaxpr(_compute)(x, y)

# vmap — triggers additional compilation
_vcompute = jax.vmap(_compute)
xb = jax.random.normal(key, (4, 128, 128))
yb = jax.random.normal(key, (4, 128, 128))
try:
    outb = _vcompute(xb, yb)
    outb.block_until_ready()
except Exception:
    pass
"""


# ─── Public API ───────────────────────────────────────────────────────────────

def get_baseline(framework: str) -> str:
    """Return the baseline (import-only) code for a framework."""
    if framework not in _BASELINES:
        raise ValueError(f"Unknown framework: {framework!r}. Choose from {FRAMEWORKS}")
    return _BASELINES[framework]


def get_workload(category: str, framework: str) -> str:
    """Return the workload code for a (category, framework) pair."""
    if framework not in FRAMEWORKS:
        raise ValueError(f"Unknown framework: {framework!r}. Choose from {FRAMEWORKS}")
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category: {category!r}. Choose from {CATEGORIES}")
    key = (framework, category)
    if key not in _WORKLOADS:
        raise KeyError(f"No workload defined for {key}")
    return _WORKLOADS[key]


def list_all() -> list[tuple[str, str]]:
    """Return all defined (framework, category) pairs."""
    return list(_WORKLOADS.keys())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Print a strace workload code snippet")
    parser.add_argument("--framework", choices=FRAMEWORKS, required=True)
    parser.add_argument("--category", choices=CATEGORIES + ["baseline"], required=True)
    args = parser.parse_args()
    if args.category == "baseline":
        print(get_baseline(args.framework))
    else:
        print(get_workload(args.category, args.framework))
