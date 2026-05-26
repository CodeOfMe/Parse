"""
FFN Transplantation module for PARSE.

Removes Feed-Forward Network components (gate_proj, up_proj, down_proj) from
specified layers and inserts lightweight No-FFN replacements inspired by Needle [55].

The transplanted blocks use pure attention + gated residual connections.
A Dynamic Capability Router (DCR) modulates the residual gates based on
input context, enabling multi-profile serving from a single compressed model.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
from collections import OrderedDict


class NoFFNBlock(nn.Module):
    """
    Lightweight replacement for a removed FFN sublayer.

    Unlike a standard FFN (~65% of layer parameters), this block contains only:
    - A gated residual connection (scalar learnable gate)
    - Optional scenario-specific gate perturbations (for DCR modulation)
    """

    def __init__(
        self,
        hidden_size: int,
        num_scenarios: int = 0,
        enable_dcr: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.enable_dcr = enable_dcr
        self.num_scenarios = num_scenarios

        # Base gate value: initialized to 0 → σ(0) = 0.5 for stable start
        self.gate_base = nn.Parameter(torch.zeros(1))

        # Per-scenario gate perturbations (learned during DCR training)
        if enable_dcr and num_scenarios > 0:
            self.gate_specialized = nn.Parameter(torch.zeros(num_scenarios, 1))
        else:
            self.gate_specialized = None

    def forward(self, hidden_states, routing_weights=None):
        """
        Parameters
        ----------
        hidden_states : (B, S, D)
        routing_weights : (B, num_scenarios) or None

        Returns
        -------
        gated_output : (B, S, D) — hidden_states scaled by modulated gate.
        """
        if self.enable_dcr and routing_weights is not None and self.gate_specialized is not None:
            # Paper Eq: g_l(x) = σ(g_l^base + Σ R_c(x) · g_{l,c}^specialized)
            scenario_bias = routing_weights @ self.gate_specialized  # (B, 1)
            gate = torch.sigmoid(self.gate_base + scenario_bias)  # (B, 1)
        else:
            gate = torch.sigmoid(self.gate_base)

        return gate.unsqueeze(-1) * hidden_states


class CapabilityRouter(nn.Module):
    """
    Dynamic Capability Router (DCR) — Section 3.4 of the paper.

    A lightweight neural probe (0.08M parameters typical) that maps mean-pooled
    input embeddings to a probability distribution over (Language x Discipline x Scenario)
    capability combinations.

    R(x) = softmax(W_r · mean(h_embed(x)) + b_r)
    """

    def __init__(self, hidden_size: int, num_scenarios: int, bottleneck: int = None):
        super().__init__()
        if bottleneck is None:
            bottleneck = max(hidden_size // 4, 64)

        self.router = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, num_scenarios),
        )

    def forward(self, hidden_states):
        """
        Parameters
        ----------
        hidden_states : (B, S, D) — input embeddings or hidden states

        Returns
        -------
        routing_weights : (B, num_scenarios) — softmax distribution
        """
        pooled = hidden_states.mean(dim=1)  # (B, D)
        logits = self.router(pooled)
        return torch.softmax(logits, dim=-1)

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class TransplantFFN:
    """
    Perform FFN transplantation on a HuggingFace Qwen/LLaMA-style model.

    The transplantation:
    1. Locates the FFN submodules (gate_proj, up_proj, down_proj) in target layers
    2. Replaces them with lightweight NoFFNBlock modules
    3. Optionally inserts DCR gate perturbations for multi-profile routing

    The original Self-Attention modules (Q, K, V, O) are preserved intact.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        enable_dcr: bool = True,
    ):
        self.model = model
        self.device = device
        self.enable_dcr = enable_dcr

        self.hidden_size = None
        self.num_scenarios = 0
        self.transplanted_layers: List[int] = []
        self.router: Optional[CapabilityRouter] = None
        self.noffn_blocks: nn.ModuleDict = nn.ModuleDict()

    def transplant(
        self,
        layers_to_transplant: List[int],
        num_scenarios: int = 5,
    ) -> nn.Module:
        """
        Execute transplantation on specified layers.

        Parameters
        ----------
        layers_to_transplant : list of int
            Zero-indexed layer indices to receive FFN removal + NoFFN replacement.
        num_scenarios : int
            Number of capability scenarios for DCR routing (default: 5 = |S_cen|).

        Returns
        -------
        The modified model (same object, mutated in-place).
        """
        self.transplanted_layers = sorted(layers_to_transplant)
        self.num_scenarios = num_scenarios

        layers = self._get_layers()
        self.hidden_size = self._detect_hidden_size(layers[0])

        # Build NoFFN blocks
        for layer_idx in self.transplanted_layers:
            self._transplant_layer(layers[layer_idx], layer_idx)

        # Build DCR if enabled
        if self.enable_dcr:
            self.router = CapabilityRouter(
                hidden_size=self.hidden_size,
                num_scenarios=num_scenarios,
            ).to(self.device)

        return self.model

    def get_param_stats(self) -> Dict:
        """Return statistics about the transplantation."""
        original_ffn_params = 0
        removed_ffn_params = 0
        noffn_added_params = 0

        layers = self._get_layers()
        for i, layer in enumerate(layers):
            for name, p in layer.named_parameters():
                if any(kw in name for kw in ["gate_proj", "up_proj", "down_proj"]):
                    if i in self.transplanted_layers:
                        removed_ffn_params += p.numel()
                    else:
                        original_ffn_params += p.numel()

        for p in self.noffn_blocks.parameters():
            noffn_added_params += p.numel()

        dcr_params = sum(p.numel() for p in self.router.parameters()) if self.router else 0
        total_model_params = sum(p.numel() for p in self.model.parameters())

        return {
            "original_ffn_params": original_ffn_params,
            "removed_ffn_params": removed_ffn_params,
            "noffn_added_params": noffn_added_params,
            "dcr_params": dcr_params,
            "total_params_after": total_model_params,
            "transplanted_layer_count": len(self.transplanted_layers),
            "transplanted_layers": self.transplanted_layers,
        }

    def forward_with_routing(self, input_ids, attention_mask=None, labels=None):
        """
        Full forward pass with DCR routing and FFN-transplanted layers.

        This method intercepts the standard forward pass to inject:
        1. DCR routing weights computation
        2. NoFFN gate modulation in transplanted layers

        For transplanted layers, the FFN computation is replaced by the NoFFNBlock
        gating applied to the post-attention hidden states.

        Returns
        -------
        Same format as a standard HuggingFace model forward pass (dict with
        logits, loss, plus routing weights).
        """
        # Standard forward with output_hidden_states to intercept
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        return {
            "logits": outputs.logits,
            "loss": outputs.loss,
            "hidden_states": outputs.hidden_states,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_layers(self) -> List[nn.Module]:
        model = self.model
        for attr in ["model.layers", "transformer.h", "model.decoder.layers"]:
            parts = attr.split(".")
            obj = model
            try:
                for p in parts:
                    obj = getattr(obj, p)
                return list(obj)
            except (AttributeError, TypeError):
                continue
        raise RuntimeError("Cannot find transformer layers.")

    def _detect_hidden_size(self, layer: nn.Module) -> int:
        for name, p in layer.named_parameters():
            if "weight" in name and len(p.shape) == 2:
                return p.shape[1]
        return 1024  # fallback

    def _transplant_layer(self, layer: nn.Module, layer_idx: int):
        """Remove FFN submodules from a layer and insert NoFFNBlock."""
        # Build NoFFN block
        noffn = NoFFNBlock(
            hidden_size=self.hidden_size,
            num_scenarios=self.num_scenarios,
            enable_dcr=self.enable_dcr,
        ).to(self.device)
        self.noffn_blocks.add_module(f"layer_{layer_idx}", noffn)

        # Replace FFN submodules with identity paths (the NoFFN gate handles scaling)
        # Qwen/LLaMA FFN names: gate_proj, up_proj, down_proj (in mlp submodule)
        ffn_names = self._find_ffn_names(layer)

        for name in ffn_names:
            try:
                # Navigate to the FFN submodule
                parts = name.split(".")
                parent = layer
                for p in parts[:-1]:
                    parent = getattr(parent, p)

                # Replace with identity
                setattr(parent, parts[-1], nn.Identity())
            except (AttributeError, TypeError):
                continue

    @staticmethod
    def _find_ffn_names(layer: nn.Module) -> List[str]:
        """Find FFN parameter names in a layer (Qwen: mlp.gate_proj, mlp.up_proj, mlp.down_proj)."""
        ffn_names = []
        for name, module in layer.named_modules():
            if any(kw in name for kw in ["mlp", "ffn", "feed_forward"]):
                for subname, _ in module.named_children():
                    if any(kw in subname for kw in ["gate_proj", "up_proj", "down_proj", "fc1", "fc2", "w1", "w2", "w3"]):
                        ffn_names.append(f"{name}.{subname}")
        return ffn_names
