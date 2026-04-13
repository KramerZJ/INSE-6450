#!/usr/bin/env python3
"""
Deep API Call Extractor for TensorFlow, PyTorch, and JAX
=========================================================

Extracts deeply-qualified API call sites from framework source repos
for training multi-class function classification models.

Produces a unified dataset of (code_context, full_api_path, metadata) tuples
suitable for training sequence classification models (CNN+LSTM, Transformers, etc.)

Usage:
    # Clone repos first:
    # git clone https://github.com/tensorflow/tensorflow.git
    # git clone https://github.com/pytorch/pytorch.git
    # git clone https://github.com/jax-ml/jax.git

    python deep_api_extractor.py \
        --tf-repo ./tensorflow \
        --pytorch-repo ./pytorch \
        --jax-repo ./jax \
        --output ./unified_dataset.jsonl \
        --include-synthetic
"""

import ast
import os
import re
import json
import yaml
import logging
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class APICallSite:
    """A single extracted API call with full context."""
    framework: str                  # "tensorflow", "pytorch", "jax"
    api_path: str                   # Full qualified path, e.g. "tensorflow.distribute.experimental.rpc.kernels.gen_rpc_ops.rpc_client"
    code_context: str               # The surrounding code (call + a few lines of context)
    call_expression: str            # Just the call itself, e.g. "gen_rpc_ops.rpc_client(...)"
    source_file: str                # Where it was found
    line_number: int                # Line in source file
    num_args: int                   # Number of positional args
    kwarg_names: list               # Names of keyword arguments
    depth_level: int                # How deep the API path goes
    api_category: str               # "ops", "nn", "distributed", "autograd", "compiler", "io", "other"
    is_internal: bool               # Whether it uses _private or internal APIs
    is_synthetic: bool = False      # Whether this was synthetically generated

    @property
    def unique_id(self):
        content = f"{self.framework}:{self.api_path}:{self.code_context}"
        return hashlib.md5(content.encode()).hexdigest()


# =============================================================================
# Base AST Extractor
# =============================================================================

class BaseDeepCallExtractor(ast.NodeVisitor):
    """
    Base class for framework-specific AST extractors.

    Handles:
    - Import tracking (import X, from X import Y, from X import Y as Z)
    - Call resolution (resolving chained attribute access to full paths)
    - Context extraction (surrounding lines for each call site)
    """

    FRAMEWORK = "base"
    ROOTS = set()
    COMMON_ALIASES = {}
    INTERNAL_MARKERS = {"_", "__"}

    # Categories for classifying API paths
    CATEGORY_PATTERNS = {}

    def __init__(self, source_code: str, file_path: str):
        self.source = source_code
        self.source_lines = source_code.splitlines()
        self.file_path = file_path
        self.imports = {}           # alias -> full module path
        self.from_imports = {}      # alias -> full qualified name
        self.star_imports = []      # modules imported with *
        self.calls = []             # extracted APICallSite instances

        # Pre-populate common aliases
        for alias, full_path in self.COMMON_ALIASES.items():
            self.from_imports[alias] = full_path

    def extract(self):
        """Parse the source and extract all deep API calls."""
        try:
            tree = ast.parse(self.source)
            self.visit(tree)
        except SyntaxError:
            logger.debug(f"Syntax error in {self.file_path}, skipping")
        return self.calls

    def visit_Import(self, node):
        """Track: import torch, import tensorflow as tf"""
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports[name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Track: from torch.nn import functional as F"""
        if node.module is None:
            self.generic_visit(node)
            return

        for alias in node.names:
            if alias.name == "*":
                self.star_imports.append(node.module)
                continue
            name = alias.asname or alias.name
            self.from_imports[name] = f"{node.module}.{alias.name}"

        self.generic_visit(node)

    def visit_Call(self, node):
        """Extract and resolve every function call."""
        full_path = self._resolve_call_path(node.func)

        if full_path and self._is_framework_call(full_path):
            context = self._get_context(node)
            call_expr = self._get_call_expression(node)
            api_path = self._normalize_path(full_path)

            site = APICallSite(
                framework=self.FRAMEWORK,
                api_path=api_path,
                code_context=context,
                call_expression=call_expr,
                source_file=self.file_path,
                line_number=node.lineno,
                num_args=len(node.args),
                kwarg_names=[kw.arg for kw in node.keywords if kw.arg],
                depth_level=api_path.count("."),
                api_category=self._categorize(api_path),
                is_internal=self._is_internal(api_path),
            )
            self.calls.append(site)

        self.generic_visit(node)

    def _resolve_call_path(self, node) -> Optional[str]:
        """Recursively resolve a call node to its full dotted path."""
        if isinstance(node, ast.Name):
            # Simple name: could be a from-import or a direct import
            name = node.id
            if name in self.from_imports:
                return self.from_imports[name]
            if name in self.imports:
                return self.imports[name]
            # Check star imports
            for mod in self.star_imports:
                if self._is_framework_root(mod):
                    return f"{mod}.{name}"
            return name

        elif isinstance(node, ast.Attribute):
            # Chained: something.attr
            parent = self._resolve_call_path(node.value)
            if parent:
                return f"{parent}.{node.attr}"
            # If parent resolution failed, try to build from attribute chain
            chain = self._get_attribute_chain(node)
            if chain:
                return chain
            return None

        elif isinstance(node, ast.Subscript):
            # Handle torch.ops.aten["add"] style access
            parent = self._resolve_call_path(node.value)
            if parent and isinstance(node.slice, ast.Constant):
                return f"{parent}.{node.slice.value}"
            return parent

        return None

    def _get_attribute_chain(self, node) -> Optional[str]:
        """Build full dotted path from chained attribute access."""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            # Resolve the root
            root = parts[-1]
            if root in self.from_imports:
                parts[-1] = self.from_imports[root]
            elif root in self.imports:
                parts[-1] = self.imports[root]
            return ".".join(reversed(parts))
        return None

    def _is_framework_call(self, path: str) -> bool:
        """Check if the resolved path belongs to this framework."""
        for root in self.ROOTS:
            if path.startswith(root + ".") or path == root:
                return True
        return False

    def _is_framework_root(self, module: str) -> bool:
        """Check if a module belongs to this framework."""
        for root in self.ROOTS:
            if module.startswith(root):
                return True
        return False

    def _get_context(self, node, context_lines=3) -> str:
        """Extract surrounding lines of code for context."""
        start = max(0, node.lineno - 1 - context_lines)
        end = min(len(self.source_lines), node.end_lineno + context_lines if node.end_lineno else node.lineno + context_lines)
        return "\n".join(self.source_lines[start:end])

    def _get_call_expression(self, node) -> str:
        """Extract just the call expression text."""
        try:
            start_line = node.lineno - 1
            end_line = node.end_lineno if node.end_lineno else node.lineno
            lines = self.source_lines[start_line:end_line]
            if lines:
                # Adjust first and last line offsets
                if len(lines) == 1:
                    return lines[0][node.col_offset:node.end_col_offset]
                else:
                    lines[0] = lines[0][node.col_offset:]
                    if node.end_col_offset:
                        lines[-1] = lines[-1][:node.end_col_offset]
                    return "\n".join(lines)
        except (IndexError, TypeError):
            pass
        return ""

    def _normalize_path(self, path: str) -> str:
        """Normalize the API path (framework-specific override point)."""
        return path

    def _categorize(self, path: str) -> str:
        """Categorize an API path into a functional area."""
        path_lower = path.lower()
        for category, patterns in self.CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if pattern in path_lower:
                    return category
        return "other"

    def _is_internal(self, path: str) -> bool:
        """Check if path uses internal/private APIs."""
        parts = path.split(".")
        return any(
            part.startswith("_") and not part.startswith("__init__")
            for part in parts
        )


# =============================================================================
# TensorFlow Extractor
# =============================================================================

class TensorFlowExtractor(BaseDeepCallExtractor):
    """
    Extracts deep TF calls including:
    - gen_*_ops.* (generated op wrappers)
    - tf.raw_ops.* (raw operations)
    - tf.distribute.experimental.rpc.kernels.*
    - _pywrap_* (C++ bindings)
    - Internal test helpers and framework calls
    """

    FRAMEWORK = "tensorflow"
    ROOTS = {"tf", "tensorflow", "keras"}
    COMMON_ALIASES = {
        "tf": "tensorflow",
        "K": "tensorflow.keras.backend",
        "layers": "tensorflow.keras.layers",
    }

    CATEGORY_PATTERNS = {
        "ops": ["gen_", "raw_ops", "_ops.", "math_ops", "array_ops", "nn_ops", "string_ops"],
        "nn": ["keras", "layers", "nn.", "activations", "losses", "metrics"],
        "distributed": ["distribute", "rpc", "collective", "mirrored", "tpu_strategy"],
        "autograd": ["gradient", "autodiff", "GradientTape"],
        "compiler": ["autograph", "function", "concrete_function", "graph", "xla"],
        "io": ["io.", "data.", "TFRecord", "Dataset", "FixedLenFeature"],
        "image": ["image.", "decode_", "encode_", "resize"],
        "signal": ["signal.", "stft", "fft"],
    }

    def _normalize_path(self, path: str) -> str:
        """Normalize TF paths: resolve tf → tensorflow, handle gen_ files."""
        if path.startswith("tf."):
            path = "tensorflow." + path[3:]
        return path


class TensorFlowGenOpsExtractor:
    """
    Separately extract function signatures from gen_*.py files.
    These auto-generated files contain every TF op's Python wrapper.
    """

    def __init__(self, tf_repo_path: str):
        self.repo_path = Path(tf_repo_path)

    def extract_op_signatures(self) -> list[dict]:
        """Find and parse all gen_*.py files for op signatures."""
        signatures = []
        gen_files = list(self.repo_path.rglob("gen_*.py"))
        logger.info(f"Found {len(gen_files)} gen_*.py files in TF repo")

        for gen_file in gen_files:
            try:
                source = gen_file.read_text(errors="ignore")
                tree = ast.parse(source)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                        sig = self._extract_signature(node, gen_file)
                        if sig:
                            signatures.append(sig)
            except (SyntaxError, UnicodeDecodeError):
                continue

        logger.info(f"Extracted {len(signatures)} TF op signatures from gen_* files")
        return signatures

    def _extract_signature(self, node: ast.FunctionDef, file_path: Path) -> Optional[dict]:
        """Extract function name, args, defaults from a FunctionDef."""
        # Build the module path from file location
        rel_path = file_path.relative_to(self.repo_path)
        module_parts = list(rel_path.with_suffix("").parts)

        # Insert "tensorflow" root if not present
        if module_parts and module_parts[0] != "tensorflow":
            module_parts = ["tensorflow"] + module_parts

        module_path = ".".join(module_parts)
        full_path = f"{module_path}.{node.name}"

        args = []
        for arg in node.args.args:
            if arg.arg != "self":
                args.append(arg.arg)

        return {
            "api_path": full_path,
            "function_name": node.name,
            "args": args,
            "kwargs": [arg.arg for arg in node.args.kwonlyargs],
            "has_defaults": len(node.args.defaults) > 0,
            "docstring": ast.get_docstring(node) or "",
        }


# =============================================================================
# PyTorch Extractor
# =============================================================================

class PyTorchExtractor(BaseDeepCallExtractor):
    """
    Extracts deep PyTorch calls including:
    - torch.ops.aten.*.default (ATen dispatcher ops)
    - torch._C._nn.* (C++ bindings)
    - torch._dynamo.* (TorchDynamo compiler)
    - torch._inductor.* (TorchInductor codegen)
    - torch.distributed.rpc.* (distributed RPC)
    - torch._functorch.* (function transforms)
    """

    FRAMEWORK = "pytorch"
    ROOTS = {"torch", "torchvision", "torchaudio", "torch_geometric"}
    COMMON_ALIASES = {
        "F": "torch.nn.functional",
        "nn": "torch.nn",
        "optim": "torch.optim",
        "dist": "torch.distributed",
        "fx": "torch.fx",
    }

    CATEGORY_PATTERNS = {
        "ops": ["ops.aten", "ops.quantized", "ops.mkldnn", "_C._", "aten.", "raw_ops"],
        "nn": ["nn.", "functional.", "modules.", "layers", "activation", "loss"],
        "distributed": ["distributed", "rpc", "c10d", "process_group", "fsdp", "ddp"],
        "autograd": ["autograd", "grad", "backward", "_functions"],
        "compiler": ["_dynamo", "_inductor", "compile", "fx.", "jit.", "_export"],
        "io": ["utils.data", "DataLoader", "Dataset", "save", "load"],
        "quantize": ["quantiz", "qconfig", "observer", "fake_quant"],
    }

    def _normalize_path(self, path: str) -> str:
        """Normalize PyTorch paths."""
        # torch.nn.functional sometimes imported as F
        return path


class PyTorchNativeFunctionsExtractor:
    """
    Parse native_functions.yaml for the complete ATen op registry.
    This is PyTorch's ground truth for all native operations.
    """

    def __init__(self, pytorch_repo_path: str):
        self.repo_path = Path(pytorch_repo_path)

    def extract_native_functions(self) -> list[dict]:
        """Parse native_functions.yaml for all ATen op signatures."""
        yaml_path = self.repo_path / "aten" / "src" / "ATen" / "native" / "native_functions.yaml"

        if not yaml_path.exists():
            logger.warning(f"native_functions.yaml not found at {yaml_path}")
            return []

        logger.info(f"Parsing {yaml_path}")

        try:
            with open(yaml_path, "r") as f:
                # native_functions.yaml is large; use safe_load
                data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to parse native_functions.yaml: {e}")
            # Try manual parsing as fallback
            return self._manual_parse(yaml_path)

        signatures = []
        if not isinstance(data, list):
            return self._manual_parse(yaml_path)

        for entry in data:
            if not isinstance(entry, dict) or "func" not in entry:
                continue

            sig = self._parse_func_string(entry["func"])
            if sig:
                sig["dispatch"] = entry.get("dispatch", {})
                sig["variants"] = entry.get("variants", "")
                signatures.append(sig)

        logger.info(f"Extracted {len(signatures)} PyTorch native function signatures")
        return signatures

    def _parse_func_string(self, func_str: str) -> Optional[dict]:
        """
        Parse a native_functions.yaml func string.
        Format: "name.overload(args) -> return_type"
        Example: "add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor"
        """
        try:
            # Split name from args
            name_part = func_str.split("(")[0].strip()
            args_match = re.search(r"\((.*)\)", func_str)
            return_match = re.search(r"->\s*(.+)$", func_str)

            # Parse name and overload
            if "." in name_part:
                name, overload = name_part.split(".", 1)
            else:
                name, overload = name_part, "default"

            # Parse arguments
            args = []
            kwargs = []
            if args_match:
                args_str = args_match.group(1)
                in_kwargs = False
                for arg in args_str.split(","):
                    arg = arg.strip()
                    if arg == "*":
                        in_kwargs = True
                        continue
                    if arg:
                        # Extract arg name (last word before =)
                        arg_name = arg.split("=")[0].strip().split()[-1] if arg.split("=")[0].strip() else ""
                        if in_kwargs:
                            kwargs.append(arg_name)
                        else:
                            args.append(arg_name)

            return {
                "api_path": f"torch.ops.aten.{name}.{overload}",
                "function_name": name,
                "overload": overload,
                "args": args,
                "kwargs": kwargs,
                "return_type": return_match.group(1).strip() if return_match else "",
                "raw_signature": func_str,
            }
        except Exception:
            return None

    def _manual_parse(self, yaml_path: Path) -> list[dict]:
        """Fallback: regex-based parsing if YAML parser fails (file is huge)."""
        signatures = []
        try:
            content = yaml_path.read_text(errors="ignore")
            func_pattern = re.compile(r"- func:\s*(.+)")
            for match in func_pattern.finditer(content):
                sig = self._parse_func_string(match.group(1).strip())
                if sig:
                    signatures.append(sig)
        except Exception as e:
            logger.error(f"Manual parsing also failed: {e}")
        return signatures


# =============================================================================
# JAX Extractor
# =============================================================================

class JAXExtractor(BaseDeepCallExtractor):
    """
    Extracts deep JAX calls including:
    - jax.lax.* (primitive operations)
    - jax._src.lax.* (internal implementations)
    - jax._src.interpreters.* (XLA/MLIR lowering)
    - jax._src.core.* (Jaxpr tracing)
    - jax._src.ad.* (autodiff internals)
    - jax.experimental.* (experimental APIs)
    """

    FRAMEWORK = "jax"
    ROOTS = {"jax", "jaxlib", "flax", "optax", "haiku"}
    COMMON_ALIASES = {
        "jnp": "jax.numpy",
        "lax": "jax.lax",
        "jsp": "jax.scipy",
        "random": "jax.random",
        "vmap": "jax.vmap",
        "jit": "jax.jit",
        "grad": "jax.grad",
        "pmap": "jax.pmap",
    }

    CATEGORY_PATTERNS = {
        "ops": ["lax.", "lax_", "primitive", "raw_ops", "_src.lax"],
        "nn": ["nn.", "stax", "flax", "haiku", "optax", "activation"],
        "distributed": ["distributed", "pmap", "sharding", "mesh", "shard_map", "pjit"],
        "autograd": ["ad.", "grad", "jvp", "vjp", "custom_jvp", "custom_vjp", "checkpoint"],
        "compiler": ["interpreters", "xla", "mlir", "dispatch", "jaxpr", "_src.core", "stages"],
        "io": ["io.", "serialization", "checkpoint"],
        "numpy": ["numpy", "jnp."],
        "random": ["random.", "prng", "key"],
        "scipy": ["scipy"],
    }

    def _normalize_path(self, path: str) -> str:
        """Normalize JAX paths."""
        return path


class JAXPrimitiveExtractor:
    """
    Extract JAX primitive registrations from source code.
    Primitives are JAX's equivalent of gen_* ops / native_functions.yaml.

    Looks for patterns like:
        add_p = standard_primitive(...)
        conv_general_dilated_p = Primitive('conv_general_dilated')
        reduce_sum_p = standard_primitive(_reduce_sum_shape_rule, ...)
    """

    def __init__(self, jax_repo_path: str):
        self.repo_path = Path(jax_repo_path)

    def extract_primitives(self) -> list[dict]:
        """Find all primitive definitions in JAX source."""
        primitives = []

        # Search in jax/_src/lax/ (primary location for primitives)
        search_dirs = [
            self.repo_path / "jax" / "_src" / "lax",
            self.repo_path / "jax" / "_src",
            self.repo_path / "jax" / "lax",
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            for py_file in search_dir.rglob("*.py"):
                try:
                    source = py_file.read_text(errors="ignore")
                    primitives.extend(self._extract_from_source(source, py_file))
                except Exception:
                    continue

        logger.info(f"Extracted {len(primitives)} JAX primitives")
        return primitives

    def _extract_from_source(self, source: str, file_path: Path) -> list[dict]:
        """Extract primitive definitions from a single file."""
        results = []

        # Pattern 1: xxx_p = standard_primitive(...)
        standard_pattern = re.compile(
            r"(\w+)_p\s*=\s*standard_primitive\s*\(([^)]*)\)"
        )

        # Pattern 2: xxx_p = Primitive('xxx')
        primitive_pattern = re.compile(
            r"(\w+)_p\s*=\s*(?:core\.)?Primitive\s*\(\s*['\"](\w+)['\"]\s*\)"
        )

        # Pattern 3: xxx_p = core.Primitive('xxx')  (alternate import)
        core_primitive_pattern = re.compile(
            r"(\w+)_p\s*=\s*\w+\.Primitive\s*\(\s*['\"](\w+)['\"]\s*\)"
        )

        for pattern in [standard_pattern, primitive_pattern, core_primitive_pattern]:
            for match in pattern.finditer(source):
                var_name = match.group(1)
                prim_name = match.group(2) if pattern != standard_pattern else var_name

                # Build module path
                rel_path = file_path.relative_to(self.repo_path)
                module = ".".join(rel_path.with_suffix("").parts)

                results.append({
                    "api_path": f"jax._src.lax.{prim_name}",
                    "public_path": f"jax.lax.{prim_name}",
                    "primitive_name": prim_name,
                    "variable_name": f"{var_name}_p",
                    "source_module": module,
                    "bind_path": f"jax._src.lax.{prim_name}_p.bind",
                })

        # Pattern 4: Find abstract_eval rules (tells us about primitives and their types)
        eval_pattern = re.compile(
            r"(\w+)_p\.def_(?:abstract_eval|impl)\s*\(\s*(\w+)\s*\)"
        )
        for match in eval_pattern.finditer(source):
            prim_var = match.group(1)
            # We may have already found this primitive, skip duplicates
            # but this confirms the primitive is fully registered

        return results


# =============================================================================
# Synthetic Data Generator
# =============================================================================

class SyntheticCallGenerator:
    """
    Generate synthetic API call variations from extracted signatures.

    Given a real call like:
        gen_rpc_ops.rpc_client("172.24.89.186:1111", 10000, list_registered_methods=True)

    Generates variations like:
        gen_rpc_ops.rpc_client(server_addr, timeout)
        gen_rpc_ops.rpc_client(f"{host}:{port}", DEFAULT_TIMEOUT, list_registered_methods=False)
        gen_rpc_ops.rpc_client(address, timeout_ms, list_registered_methods=enable_discovery)
    """

    # Template values for different argument types
    TENSOR_TEMPLATES = [
        "tf.constant({val})",
        "tf.zeros({shape})",
        "tf.ones({shape})",
        "torch.tensor({val})",
        "torch.randn({shape})",
        "torch.zeros({shape})",
        "jnp.array({val})",
        "jnp.zeros({shape})",
        "jnp.ones({shape})",
        "input_tensor",
        "x",
        "hidden_state",
        "embeddings",
    ]

    STRING_TEMPLATES = [
        '"{val}"',
        "f'{{{var}}}'",
        "address",
        "endpoint",
        "name",
        "config_path",
        "DEFAULT_NAME",
    ]

    INT_TEMPLATES = [
        "{val}",
        "batch_size",
        "num_steps",
        "hidden_dim",
        "DEFAULT_SIZE",
        "config.num_layers",
    ]

    BOOL_TEMPLATES = [
        "True",
        "False",
        "is_training",
        "use_bias",
        "config.enable_feature",
    ]

    def __init__(self, multiplier: int = 5):
        self.multiplier = multiplier

    def generate_from_signatures(self, signatures: list[dict], framework: str) -> list[APICallSite]:
        """Generate synthetic call sites from extracted signatures."""
        synthetic = []
        import random

        for sig in signatures:
            api_path = sig.get("api_path", "")
            func_name = sig.get("function_name", api_path.split(".")[-1])
            args = sig.get("args", [])
            kwargs = sig.get("kwargs", [])

            for i in range(self.multiplier):
                # Generate argument values
                synth_args = []
                for arg_name in args:
                    if arg_name in ("self", "name"):
                        continue
                    synth_args.append(self._generate_arg_value(arg_name, framework))

                synth_kwargs = {}
                # Randomly include some kwargs
                for kwarg_name in kwargs:
                    if random.random() > 0.5:
                        synth_kwargs[kwarg_name] = self._generate_arg_value(kwarg_name, framework)

                # Build the call expression
                args_str = ", ".join(synth_args)
                kwargs_str = ", ".join(f"{k}={v}" for k, v in synth_kwargs.items())
                all_args = ", ".join(filter(None, [args_str, kwargs_str]))

                # Build caller prefix (module path minus the function name)
                path_parts = api_path.rsplit(".", 1)
                if len(path_parts) == 2:
                    caller = path_parts[0].split(".")[-1]  # Last module segment
                    call_expr = f"{caller}.{func_name}({all_args})"
                else:
                    call_expr = f"{func_name}({all_args})"

                # Generate context (fake surrounding code)
                context = self._generate_context(call_expr, framework)

                site = APICallSite(
                    framework=framework,
                    api_path=api_path,
                    code_context=context,
                    call_expression=call_expr,
                    source_file="synthetic",
                    line_number=0,
                    num_args=len(synth_args),
                    kwarg_names=list(synth_kwargs.keys()),
                    depth_level=api_path.count("."),
                    api_category="ops",
                    is_internal="_" in api_path.split(".")[1] if "." in api_path else False,
                    is_synthetic=True,
                )
                synthetic.append(site)

        logger.info(f"Generated {len(synthetic)} synthetic examples for {framework}")
        return synthetic

    def _generate_arg_value(self, arg_name: str, framework: str) -> str:
        """Generate a plausible argument value based on the argument name."""
        import random
        name_lower = arg_name.lower()

        if any(t in name_lower for t in ["tensor", "input", "output", "weight", "bias", "x", "y", "data"]):
            template = random.choice(self.TENSOR_TEMPLATES)
            fw_templates = [t for t in self.TENSOR_TEMPLATES if framework[:2] in t.lower() or t[0].islower()]
            if fw_templates:
                template = random.choice(fw_templates)
            return template.format(val=random.choice([1, 2, 0.5, "[1,2,3]"]),
                                   shape=random.choice(["(3, 4)", "(batch_size, dim)", "(2, 3, 4)"]))

        elif any(t in name_lower for t in ["name", "path", "addr", "host", "url", "string", "key", "format"]):
            return random.choice(self.STRING_TEMPLATES).format(val="example", var="config_value")

        elif any(t in name_lower for t in ["size", "dim", "num", "step", "count", "length", "timeout", "port"]):
            return random.choice(self.INT_TEMPLATES).format(val=random.choice([32, 64, 128, 256, 512, 1024]))

        elif any(t in name_lower for t in ["enable", "use", "is_", "has_", "training", "bias", "reverse"]):
            return random.choice(self.BOOL_TEMPLATES)

        else:
            # Generic: use variable name itself
            return arg_name

    def _generate_context(self, call_expr: str, framework: str) -> str:
        """Generate fake surrounding code context."""
        import random
        prefixes = {
            "tensorflow": [
                "import tensorflow as tf",
                "with tf.device('/GPU:0'):",
                "with tf.GradientTape() as tape:",
                "@tf.function",
            ],
            "pytorch": [
                "import torch",
                "model.train()",
                "with torch.no_grad():",
                "optimizer.zero_grad()",
            ],
            "jax": [
                "import jax",
                "import jax.numpy as jnp",
                "key = jax.random.PRNGKey(0)",
                "@jax.jit",
            ],
        }

        prefix_lines = random.sample(prefixes.get(framework, [""]), min(2, len(prefixes.get(framework, [""]))))
        result_var = random.choice(["result", "output", "out", "y", "logits", "features"])

        return "\n".join(prefix_lines + [f"    {result_var} = {call_expr}"])


# =============================================================================
# File Crawler
# =============================================================================

class RepoCrawler:
    """Crawl a framework repo and extract all API call sites."""

    # Directories to skip
    SKIP_DIRS = {
        "__pycache__", ".git", "node_modules", "third_party",
        "vendor", ".tox", ".eggs", "build", "dist", ".mypy_cache",
    }

    # Prioritized directories (searched first, likely highest quality data)
    PRIORITY_DIRS = {"test", "tests", "testing", "test_"}

    def __init__(self, repo_path: str, extractor_class: type, max_files: int = None):
        self.repo_path = Path(repo_path)
        self.extractor_class = extractor_class
        self.max_files = max_files

    def crawl(self) -> list[APICallSite]:
        """Crawl the repo and return all extracted call sites."""
        all_calls = []
        python_files = self._find_python_files()

        logger.info(f"Crawling {len(python_files)} Python files in {self.repo_path}")

        for i, py_file in enumerate(python_files):
            if self.max_files and i >= self.max_files:
                break

            if i > 0 and i % 500 == 0:
                logger.info(f"  Processed {i}/{len(python_files)} files, {len(all_calls)} calls so far")

            try:
                source = py_file.read_text(errors="ignore")
                if len(source) > 500_000:  # Skip huge generated files
                    continue

                extractor = self.extractor_class(source, str(py_file))
                calls = extractor.extract()
                all_calls.extend(calls)
            except Exception as e:
                logger.debug(f"Error processing {py_file}: {e}")
                continue

        logger.info(f"Extracted {len(all_calls)} total calls from {self.repo_path.name}")
        return all_calls

    def _find_python_files(self) -> list[Path]:
        """Find all Python files, prioritizing test directories."""
        priority_files = []
        other_files = []

        for py_file in self.repo_path.rglob("*.py"):
            # Skip unwanted directories
            if any(skip in py_file.parts for skip in self.SKIP_DIRS):
                continue

            # Prioritize test files
            if any(p in str(py_file).lower() for p in self.PRIORITY_DIRS):
                priority_files.append(py_file)
            else:
                other_files.append(py_file)

        # Test files first (highest quality), then implementation files
        return priority_files + other_files


# =============================================================================
# Dataset Builder
# =============================================================================

class UnifiedDatasetBuilder:
    """
    Orchestrates extraction from all three frameworks and builds
    a unified JSONL dataset for model training.
    """

    def __init__(self, args):
        self.args = args
        self.all_calls = []
        self.all_signatures = []
        self.stats = defaultdict(lambda: defaultdict(int))

    def build(self):
        """Run the full extraction pipeline."""
        logger.info("=" * 70)
        logger.info("UNIFIED DEEP API EXTRACTION PIPELINE")
        logger.info("=" * 70)

        # --- TensorFlow ---
        if self.args.tf_repo:
            self._extract_tensorflow()

        # --- PyTorch ---
        if self.args.pytorch_repo:
            self._extract_pytorch()

        # --- JAX ---
        if self.args.jax_repo:
            self._extract_jax()

        # --- Synthetic augmentation ---
        if self.args.include_synthetic and self.all_signatures:
            self._generate_synthetic()

        # --- Deduplication ---
        self._deduplicate()

        # --- Write output ---
        self._write_dataset()

        # --- Print stats ---
        self._print_stats()

    def _extract_tensorflow(self):
        logger.info("\n[1/3] Extracting from TensorFlow...")
        repo = Path(self.args.tf_repo)

        # AST extraction
        crawler = RepoCrawler(str(repo), TensorFlowExtractor, self.args.max_files)
        calls = crawler.crawl()
        self.all_calls.extend(calls)
        self.stats["tensorflow"]["ast_calls"] = len(calls)

        # gen_* file signatures
        gen_extractor = TensorFlowGenOpsExtractor(str(repo))
        signatures = gen_extractor.extract_op_signatures()
        for sig in signatures:
            sig["framework"] = "tensorflow"
        self.all_signatures.extend(signatures)
        self.stats["tensorflow"]["op_signatures"] = len(signatures)

    def _extract_pytorch(self):
        logger.info("\n[2/3] Extracting from PyTorch...")
        repo = Path(self.args.pytorch_repo)

        # AST extraction
        crawler = RepoCrawler(str(repo), PyTorchExtractor, self.args.max_files)
        calls = crawler.crawl()
        self.all_calls.extend(calls)
        self.stats["pytorch"]["ast_calls"] = len(calls)

        # native_functions.yaml
        native_extractor = PyTorchNativeFunctionsExtractor(str(repo))
        signatures = native_extractor.extract_native_functions()
        for sig in signatures:
            sig["framework"] = "pytorch"
        self.all_signatures.extend(signatures)
        self.stats["pytorch"]["native_functions"] = len(signatures)

    def _extract_jax(self):
        logger.info("\n[3/3] Extracting from JAX...")
        repo = Path(self.args.jax_repo)

        # AST extraction
        crawler = RepoCrawler(str(repo), JAXExtractor, self.args.max_files)
        calls = crawler.crawl()
        self.all_calls.extend(calls)
        self.stats["jax"]["ast_calls"] = len(calls)

        # Primitive extraction
        prim_extractor = JAXPrimitiveExtractor(str(repo))
        primitives = prim_extractor.extract_primitives()
        for prim in primitives:
            prim["framework"] = "jax"
        self.all_signatures.extend(primitives)
        self.stats["jax"]["primitives"] = len(primitives)

    def _generate_synthetic(self):
        logger.info("\n[+] Generating synthetic training data...")
        generator = SyntheticCallGenerator(multiplier=self.args.synthetic_multiplier)

        for framework in ["tensorflow", "pytorch", "jax"]:
            fw_sigs = [s for s in self.all_signatures if s.get("framework") == framework]
            if fw_sigs:
                synthetic = generator.generate_from_signatures(fw_sigs, framework)
                self.all_calls.extend(synthetic)
                self.stats[framework]["synthetic"] = len(synthetic)

    def _deduplicate(self):
        """Remove duplicate call sites based on content hash."""
        seen = set()
        unique = []
        for call in self.all_calls:
            uid = call.unique_id
            if uid not in seen:
                seen.add(uid)
                unique.append(call)

        removed = len(self.all_calls) - len(unique)
        logger.info(f"Deduplication: removed {removed} duplicates, {len(unique)} unique calls remain")
        self.all_calls = unique

    def _write_dataset(self):
        """Write the unified dataset as JSONL."""
        output_path = Path(self.args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Main dataset
        with open(output_path, "w") as f:
            for call in self.all_calls:
                record = asdict(call)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Wrote {len(self.all_calls)} records to {output_path}")

        # Signature registry (separate file)
        sig_path = output_path.with_name(output_path.stem + "_signatures.json")
        with open(sig_path, "w") as f:
            json.dump(self.all_signatures, f, indent=2, ensure_ascii=False)

        logger.info(f"Wrote {len(self.all_signatures)} signatures to {sig_path}")

        # Label taxonomy (unique API paths)
        labels_path = output_path.with_name(output_path.stem + "_labels.json")
        label_counts = defaultdict(int)
        for call in self.all_calls:
            label_counts[call.api_path] += 1

        taxonomy = {
            "total_unique_labels": len(label_counts),
            "labels": dict(sorted(label_counts.items(), key=lambda x: -x[1])),
        }
        with open(labels_path, "w") as f:
            json.dump(taxonomy, f, indent=2, ensure_ascii=False)

        logger.info(f"Wrote {len(label_counts)} unique labels to {labels_path}")

        # Train/val/test split info
        split_path = output_path.with_name(output_path.stem + "_split_guide.json")
        split_guide = {
            "recommended_split": {
                "train": 0.8,
                "validation": 0.1,
                "test": 0.1,
            },
            "strategy": "stratified_by_api_path",
            "notes": [
                "Stratify by api_path to ensure all labels appear in all splits",
                "For transfer learning: train on pytorch+jax, fine-tune on tensorflow",
                "Keep synthetic data in train only, never in val/test",
                "Consider depth_level as a secondary stratification axis",
            ],
            "transfer_learning_plan": {
                "phase_1_pretrain": {
                    "data": "pytorch + jax calls (all)",
                    "objective": "Learn general deep API call patterns",
                },
                "phase_2_finetune": {
                    "data": "tensorflow calls",
                    "objective": "Transfer to TF-specific API taxonomy",
                },
            },
        }
        with open(split_path, "w") as f:
            json.dump(split_guide, f, indent=2)

        logger.info(f"Wrote split guide to {split_path}")

    def _print_stats(self):
        """Print extraction statistics."""
        logger.info("\n" + "=" * 70)
        logger.info("EXTRACTION SUMMARY")
        logger.info("=" * 70)

        total = 0
        for framework in ["tensorflow", "pytorch", "jax"]:
            fw_stats = self.stats[framework]
            if not fw_stats:
                continue

            fw_total = sum(v for k, v in fw_stats.items())
            total += fw_total
            logger.info(f"\n  {framework.upper()}:")
            for key, val in fw_stats.items():
                logger.info(f"    {key}: {val:,}")

        logger.info(f"\n  TOTAL RECORDS: {len(self.all_calls):,}")

        # Depth distribution
        depth_dist = defaultdict(int)
        for call in self.all_calls:
            depth_dist[call.depth_level] += 1

        logger.info("\n  DEPTH DISTRIBUTION:")
        for depth in sorted(depth_dist.keys()):
            bar = "█" * min(50, depth_dist[depth] // max(1, len(self.all_calls) // 1000))
            logger.info(f"    Level {depth}: {depth_dist[depth]:>8,}  {bar}")

        # Category distribution
        cat_dist = defaultdict(int)
        for call in self.all_calls:
            cat_dist[call.api_category] += 1

        logger.info("\n  CATEGORY DISTRIBUTION:")
        for cat, count in sorted(cat_dist.items(), key=lambda x: -x[1]):
            logger.info(f"    {cat:>15}: {count:,}")

        # Framework distribution
        fw_dist = defaultdict(int)
        for call in self.all_calls:
            fw_dist[call.framework] += 1

        logger.info("\n  FRAMEWORK DISTRIBUTION:")
        for fw, count in sorted(fw_dist.items(), key=lambda x: -x[1]):
            logger.info(f"    {fw:>15}: {count:,}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Deep API Call Extractor for TensorFlow, PyTorch, and JAX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from all three frameworks:
  python deep_api_extractor.py \\
      --tf-repo ./tensorflow \\
      --pytorch-repo ./pytorch \\
      --jax-repo ./jax \\
      --output ./dataset/unified.jsonl \\
      --include-synthetic

  # Extract from just PyTorch + JAX (for pre-training):
  python deep_api_extractor.py \\
      --pytorch-repo ./pytorch \\
      --jax-repo ./jax \\
      --output ./dataset/pretrain.jsonl

  # Quick test run with file limit:
  python deep_api_extractor.py \\
      --tf-repo ./tensorflow \\
      --max-files 100 \\
      --output ./dataset/test.jsonl
        """,
    )

    parser.add_argument("--tf-repo", type=str, default=None,
                        help="Path to cloned TensorFlow repo")
    parser.add_argument("--pytorch-repo", type=str, default=None,
                        help="Path to cloned PyTorch repo")
    parser.add_argument("--jax-repo", type=str, default=None,
                        help="Path to cloned JAX repo")
    parser.add_argument("--output", type=str, default="./unified_dataset.jsonl",
                        help="Output JSONL file path (default: ./unified_dataset.jsonl)")
    parser.add_argument("--include-synthetic", action="store_true",
                        help="Generate synthetic training examples from signatures")
    parser.add_argument("--synthetic-multiplier", type=int, default=5,
                        help="Number of synthetic examples per signature (default: 5)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Max Python files to process per repo (for testing)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not any([args.tf_repo, args.pytorch_repo, args.jax_repo]):
        parser.error("At least one repo path must be provided (--tf-repo, --pytorch-repo, --jax-repo)")

    builder = UnifiedDatasetBuilder(args)
    builder.build()


if __name__ == "__main__":
    main()
