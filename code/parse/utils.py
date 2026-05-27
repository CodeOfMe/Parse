"""
Shared utilities for the PARSE framework: device detection, serialization,
path resolution, and measurement helpers.

All modules in parse/ should import from here rather than reimplementing
these functions.
"""

import json
import os
import sys
import time
import platform
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch


# ── Device detection ─────────────────────────────────────────────────

def detect_device(prefer: str = "auto") -> str:
    """Detect the best available compute device.

    Returns one of: "cuda", "rocm", "mps", "cpu".
    """
    if prefer != "auto":
        return prefer

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "rocm") and torch.backends.rocm.is_available():
        return "rocm"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_info(device: str) -> Dict[str, Any]:
    """Return human-readable device information."""
    info = {"device": device, "platform": platform.platform()}
    if device == "mps":
        info["chip"] = platform.processor() or "Apple Silicon"
        try:
            import subprocess
            mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            info["memory_gb"] = int(mem.stdout.strip()) / (1024**3)
        except Exception:
            pass
    elif device in ("cuda", "rocm"):
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    return info


# ── Model architecture detection ─────────────────────────────────────

def get_transformer_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Extract transformer layers from a model, handling multiple architectures."""
    for attr_path in [
        "model.language_model.layers",  # Qwen3.5 multimodal
        "model.model.layers",           # nested
        "model.layers",                  # standard HF
        "transformer.h",                 # GPT-2 style
        "model.decoder.layers",          # T5 decoder
    ]:
        parts = attr_path.split(".")
        obj = model
        try:
            for p in parts:
                obj = getattr(obj, p)
            return list(obj)
        except (AttributeError, TypeError):
            continue
    raise RuntimeError(
        f"Cannot find transformer layers in model of type {type(model).__name__}. "
        f"Expected one of: model.language_model.layers, model.layers, transformer.h"
    )


def detect_model_dims(model: torch.nn.Module) -> Tuple[int, int]:
    """Detect (n_layers, hidden_size) from a model."""
    layers = get_transformer_layers(model)
    n_layers = len(layers)

    # Hidden size: first 2D weight parameter
    for name, param in layers[0].named_parameters():
        if len(param.shape) == 2 and "weight" in name:
            return n_layers, param.shape[1]

    return n_layers, 1024  # Qwen3.5-0.8B fallback


def detect_ffn_params(model: torch.nn.Module) -> List[str]:
    """Detect FFN parameter name patterns in the model for gradient sensitivity."""
    layers = get_transformer_layers(model)
    ffn_names = set()
    ffn_keywords = ["gate_proj", "up_proj", "down_proj", "mlp", "fc1", "fc2",
                    "w1", "w2", "w3", "feed_forward", "ffn"]

    for name, _ in layers[0].named_parameters():
        lower = name.lower()
        if any(kw in lower for kw in ffn_keywords):
            ffn_names.add(name)

    return sorted(ffn_names)


# ── Serialization ────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy and torch types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        if hasattr(obj, 'item') and callable(obj.item):
            try:
                return self.default(obj.item())
            except (ValueError, TypeError):
                pass
        return super().default(obj)


def save_json(data: Any, path: str, indent: int = 2):
    """Save data as JSON with numpy/torch type handling."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder, ensure_ascii=False)


def load_json(path: str) -> Any:
    """Load data from JSON."""
    with open(path) as f:
        return json.load(f)


# ── Correlation helpers ──────────────────────────────────────────────

def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 1D arrays."""
    return float(np.corrcoef(a, b)[0, 1])


def mean_pairwise_r(matrix: np.ndarray) -> float:
    """Mean of all upper-triangle pairwise Pearson correlations."""
    n = matrix.shape[1]
    if n < 2:
        return 1.0
    rs = [pearson_r(matrix[:, i], matrix[:, j])
          for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(rs)) if rs else 0.0


def cross_axis_r(mat_a: np.ndarray, mat_b: np.ndarray) -> Tuple[float, List[float]]:
    """Mean pairwise correlation between two matrices (columns paired)."""
    rs = [pearson_r(mat_a[:, i], mat_b[:, j])
          for i in range(mat_a.shape[1]) for j in range(mat_b.shape[1])]
    return float(np.mean(rs)) if rs else 0.0, rs


def deep_shallow_ratio(matrix: np.ndarray, deep_start: int = 16) -> np.ndarray:
    """Ratio of mean activations in deep layers vs shallow layers."""
    deep = matrix[deep_start:].mean(axis=0)
    shallow = matrix[:6].mean(axis=0)
    return deep / (shallow + 1e-8)


# ── CIT normalization ────────────────────────────────────────────────

def normalize_cit(cit: np.ndarray) -> np.ndarray:
    """Per-category normalization: each category sums to 1 across layers."""
    col_sums = cit.sum(axis=0, keepdims=True)
    return cit / (col_sums + 1e-8)


# ── Timing ───────────────────────────────────────────────────────────

class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = ""):
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        label = f" [{self.name}]" if self.name else ""
        print(f"  Done in {elapsed:.1f}s{label}")
