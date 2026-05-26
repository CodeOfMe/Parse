"""
Model builder for PARSE: wraps HuggingFace Qwen model with CIT support and
DCR routing. Orchestrates the full "diagnose → sculpt → transplant" pipeline.

The model is loaded from a standard HuggingFace checkpoint and modified in-place
using the CIT-guided layer selection and FFN transplantation modules.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer

from .transplant import TransplantFFN, CapabilityRouter
from .cit import ComputeCIT


def build_parse_model(
    model_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
    enable_dcr: bool = True,
) -> Dict[str, Any]:
    """
    Load a Qwen model from HuggingFace and prepare it for PARSE experimentation.

    Returns a dict containing:
        - model: the loaded HuggingFace model
        - tokenizer: the tokenizer
        - transplant: TransplantFFN instance (for FFN surgery)
        - cit_computer: ComputeCIT instance (for importance computation)
        - n_layers: detected number of layers
        - hidden_size: detected hidden dimension

    Parameters
    ----------
    model_path : str
        Path to HuggingFace model checkpoint.
    device : str
        "cuda", "mps", or "cpu".
    torch_dtype : torch.dtype
        Precision for model weights.
    enable_dcr : bool
        Whether to enable Dynamic Capability Router.
    """
    model_path = Path(model_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=torch_dtype if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else "cpu",
        local_files_only=True,
    )
    if device == "cpu":
        model = model.float()
    model.eval()

    # Detect architecture
    n_layers, hidden_size = _detect_model_dims(model)

    # Create tool instances
    transplantation = TransplantFFN(model, device=device, enable_dcr=enable_dcr)
    cit_computer = ComputeCIT(
        model, tokenizer,
        device=device,
        alpha=0.6,
        n_layers=n_layers,
    )

    return {
        "model": model,
        "tokenizer": tokenizer,
        "transplant": transplantation,
        "cit_computer": cit_computer,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "device": device,
    }


def _detect_model_dims(model) -> tuple:
    """Detect architecture dimensions from a loaded model."""
    # Try standard Qwen/LLaMA path
    for attr in ["model.layers", "transformer.h", "model.decoder.layers"]:
        parts = attr.split(".")
        obj = model
        try:
            for p in parts:
                obj = getattr(obj, p)
            n_layers = len(obj)
            # Get hidden_size from first layer's input projection
            for name, param in obj[0].named_parameters():
                if "weight" in name and len(param.shape) == 2:
                    return n_layers, param.shape[1]
        except (AttributeError, TypeError):
            continue

    return 24, 1024  # Qwen3.5-0.8B defaults
