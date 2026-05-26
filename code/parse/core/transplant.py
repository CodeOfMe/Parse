"""
TransplantFFN — FFN transplantation with NoFFN blocks and DCR routing.

This module implements the core architectural modification of PARSE:
1. Replaces FFN submodules in selected layers with gated NoFFN pass-through
2. Preserves self-attention submodules (always retained)
3. Routes capability profiles via Dynamic Capability Router (DCR)
4. Always retains standard attention layers (L3,7,11,15,19,23 for Qwen3.5)

Key design decisions (matching the paper):
- NoFFN blocks use gated residual: output = hidden + gate * (identity - hidden)
  This preserves the residual stream while modulating via DCR.
- DCR injects routing weights into each NoFFN gate per the formula:
  g_l(x) = σ(g_l^base + Σ R_c(x) · g_{l,c}^specialized)
- Standard attention layers are NEVER transplanted, matching the hybrid
  architecture insight that standard attention carries critical routing.

Parameter target: 0.08M for DCR (bottleneck=64 for hidden_size=1024)
"""

from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn


class NoFFNBlock(nn.Module):
    """
    Gated pass-through block replacing FFN in transplanted layers.

    Instead of the original FFN computation:
        h' = h + FFN(LN(h))

    The transplanted layer computes:
        h' = h + gate * (pass_through - h) = h * (1 - gate) + pass_through * gate

    where gate = σ(g_base + Σ R_c · g_specialized) modulates the residual
    stream based on DCR routing.

    When gate ≈ 1: full pass-through (useful for preserved capabilities)
    When gate ≈ 0: pure residual (original attention output preserved)
    """

    def __init__(
        self,
        hidden_size: int,
        num_scenarios: int = 5,
        enable_dcr: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.enable_dcr = enable_dcr

        # Base gate: initialized to ~0.5 so sigmoid(0) = 0.5
        # This means transplanted layers start at 50% pass-through
        self.gate_base = nn.Parameter(torch.zeros(1))

        # Per-scenario gate perturbations (learned during flywheel training)
        if enable_dcr and num_scenarios > 0:
            self.gate_specialized = nn.Parameter(torch.zeros(num_scenarios, 1))
        else:
            self.gate_specialized = None

        # Layer normalization to stabilize transplanted output
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, routing_weights=None):
        """
        Parameters
        ----------
        hidden_states : (B, S, D) — post-attention hidden states
        routing_weights : (B, num_scenarios) — from CapabilityRouter

        Returns
        -------
        output : (B, S, D) — gated residual + normalized pass-through
        """
        # Compute gate: g_l(x) = σ(g_base + Σ R_c · g_specialized)
        if self.enable_dcr and routing_weights is not None and self.gate_specialized is not None:
            scenario_bias = routing_weights @ self.gate_specialized  # (B, 1)
            gate = torch.sigmoid(self.gate_base + scenario_bias)  # (B, 1)
        else:
            gate = torch.sigmoid(self.gate_base)  # (1,)

        gate = gate.unsqueeze(-1)  # (B, 1, 1) for broadcasting

        # Gated residual: output = (1 - gate) * hidden + gate * LN(hidden)
        # This preserves the residual stream while allowing DCR to modulate
        # how much of the original (pre-FFN) vs post-attention signal flows
        normalized = self.layer_norm(hidden_states)
        output = (1 - gate) * hidden_states + gate * normalized
        return output


class CapabilityRouter(nn.Module):
    """
    Dynamic Capability Router (DCR) — Section 3.4.

    Maps mean-pooled input embeddings to a probability distribution over
    capability scenarios:
        R(x) = softmax(W_r · mean(h_embed(x)) + b_r)

    Uses bottleneck architecture (hidden_size -> 64 -> num_scenarios)
    to maintain 0.08M parameter budget. The router produces per-sample
    routing weights that modulate NoFFN gates in transplanted layers.

    Because CIT vectors are highly correlated (r=0.994), DCR learns
    to modulate *degree* rather than *direction* — a scalar gate per
    layer rather than categorical routing per capability.
    """

    def __init__(self, hidden_size: int, num_scenarios: int, bottleneck: int = None):
        super().__init__()
        if bottleneck is None:
            bottleneck = max(hidden_size // 16, 64)

        self.router = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, num_scenarios),
        )

    def forward(self, hidden_states):
        """
        Parameters
        ----------
        hidden_states : (B, S, D) — input embeddings

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


# Standard attention layer indices for Qwen3.5-0.8B hybrid architecture
# These layers use standard (full) attention and should ALWAYS be retained
QWEN_STANDARD_ATTN_LAYERS: Set[int] = {3, 7, 11, 15, 19, 23}


class TransplantFFN:
    """
    Perform FFN transplantation on a HuggingFace Qwen/LLaMA-style model.

    The transplantation:
    1. Identifies FFN submodules (gate_proj, up_proj, down_proj) in target layers
    2. Replaces FFN computation with NoFFNBlock gated pass-through
    3. Registers forward hooks so NoFFN blocks are called during inference
    4. Builds a CapabilityRouter for multi-profile DCR routing
    5. Always retains standard attention layers (NEVER transplants them)

    The original Self-Attention modules (Q, K, V, O) are ALWAYS preserved.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        enable_dcr: bool = True,
        standard_attn_layers: Optional[Set[int]] = None,
    ):
        self.model = model
        self.device = device
        self.enable_dcr = enable_dcr
        self.standard_attn_layers = standard_attn_layers or QWEN_STANDARD_ATTN_LAYERS

        self.hidden_size = None
        self.num_scenarios = 0
        self.transplanted_layers: List[int] = []
        self._hooks = []  # forward hooks for NoFFN injection
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
            Standard attention layers are automatically excluded.
        num_scenarios : int
            Number of capability scenarios for DCR routing.

        Returns
        -------
        The modified model (same object, mutated in-place).
        """
        # Remove any standard attention layers from transplant list
        filtered = [
            i for i in sorted(set(layers_to_transplant))
            if i not in self.standard_attn_layers
        ]
        self.transplanted_layers = filtered
        self.num_scenarios = num_scenarios

        layers = self._get_layers()
        self.hidden_size = self._detect_hidden_size(layers[0])

        # Build NoFFN blocks and register hooks
        for layer_idx in self.transplanted_layers:
            noffn = self._transplant_layer(layers[layer_idx], layer_idx)
            if noffn is not None:
                self._register_hook(layers[layer_idx], layer_idx, noffn)

        # Build DCR if enabled
        if self.enable_dcr:
            self.router = CapabilityRouter(
                hidden_size=self.hidden_size,
                num_scenarios=num_scenarios,
            ).to(self.device)

        return self.model

    def forward_with_routing(self, input_ids, attention_mask=None, labels=None):
        """
        Full forward pass with DCR routing and NoFFN gate modulation.

        For each transplanted layer:
        1. Self-attention runs normally (always preserved)
        2. NoFFNBlock gate modulates the post-attention residual
        3. Gate weights are computed from DCR routing

        The NoFFN blocks are called via registered forward hooks, so this
        method performs a standard model forward pass while the hooks inject
        NoFFN gating at the appropriate layers.
        """
        # Compute DCR routing weights from input embeddings
        if self.enable_dcr and self.router is not None:
            with torch.no_grad():
                embed_layer = self._get_embedding_layer()
                input_embeds = embed_layer(input_ids)
                routing_weights = self.router(input_embeds)  # (B, num_scenarios)
        else:
            routing_weights = None

        # Store routing weights for hooks to access
        self._current_routing_weights = routing_weights

        # Standard forward pass — hooks will inject NoFFN gating
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        result = {
            "logits": outputs.logits,
            "loss": outputs.loss,
            "hidden_states": outputs.hidden_states,
            "routing_weights": routing_weights,
        }

        self._current_routing_weights = None
        return result

    def remove_hooks(self):
        """Remove all registered forward hooks (call before model saving)."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def get_param_stats(self) -> Dict:
        """Return statistics about the transplantation."""
        original_ffn_params = 0
        removed_ffn_params = 0

        layers = self._get_layers()
        for i, layer in enumerate(layers):
            for name, p in layer.named_parameters():
                if any(kw in name for kw in ["gate_proj", "up_proj", "down_proj"]):
                    if i in self.transplanted_layers:
                        removed_ffn_params += p.numel()
                    else:
                        original_ffn_params += p.numel()

        noffn_added_params = sum(p.numel() for p in self.noffn_blocks.parameters())
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
            "standard_attn_layers_retained": list(self.standard_attn_layers),
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

    def _get_embedding_layer(self):
        """Get the model's embedding layer for DCR routing."""
        model = self.model
        for attr in ["model.embed_tokens", "model.wte", "transformer.wte", "embed_tokens"]:
            parts = attr.split(".")
            obj = model
            try:
                for p in parts:
                    obj = getattr(obj, p)
                return obj
            except (AttributeError, TypeError):
                continue
        raise RuntimeError("Cannot find embedding layer for DCR routing.")

    def _detect_hidden_size(self, layer: nn.Module) -> int:
        for name, p in layer.named_parameters():
            if "weight" in name and len(p.shape) == 2:
                return p.shape[1]
        return 1024

    def _transplant_layer(self, layer: nn.Module, layer_idx: int) -> Optional[NoFFNBlock]:
        """
        Remove FFN submodules from a layer and create NoFFNBlock.
        Returns the NoFFN block, or None if no FFN found.
        """
        # Build NoFFN block
        noffn = NoFFNBlock(
            hidden_size=self.hidden_size,
            num_scenarios=self.num_scenarios,
            enable_dcr=self.enable_dcr,
        ).to(self.device)
        self.noffn_blocks.add_module(f"layer_{layer_idx}", noffn)

        # Replace FFN linear layers with Identity (parameters freed)
        # In Qwen/LLaMA: mlp.gate_proj, mlp.up_proj, mlp.down_proj
        ffn_names = self._find_ffn_names(layer)
        if not ffn_names:
            return None

        for name in ffn_names:
            try:
                parts = name.split(".")
                parent = layer
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], nn.Identity())
            except (AttributeError, TypeError):
                continue

        return noffn

    def _register_hook(self, layer: nn.Module, layer_idx: int, noffn: NoFFNBlock):
        """
        Register a forward hook on the MLP submodule to inject NoFFN gating.

        The hook intercepts the MLP forward pass (which now uses Identity layers)
        and replaces it with the NoFFNBlock's gated residual computation.
        """
        transplant_ref = self  # closure reference

        def noffn_hook(module, input, output):
            """Forward hook that replaces Identity-FFN output with NoFFN gating."""
            # input is a tuple; for MLP, input[0] is (B, S, D) hidden states
            if isinstance(input, tuple):
                hidden_states = input[0]
            else:
                hidden_states = input

            routing_weights = getattr(transplant_ref, '_current_routing_weights', None)
            noffn_module = transplant_ref.noffn_blocks.get(f"layer_{layer_idx}")
            if noffn_module is not None:
                return noffn_module(hidden_states, routing_weights)
            return output

        # Find the MLP submodule to hook
        mlp = None
        for name, module in layer.named_modules():
            if any(kw in name for kw in ["mlp", "ffn", "feed_forward"]):
                mlp = module
                break

        if mlp is not None:
            hook = mlp.register_forward_hook(noffn_hook)
            self._hooks.append(hook)

    @staticmethod
    def _find_ffn_names(layer: nn.Module) -> List[str]:
        """Find FFN parameter names in a layer."""
        ffn_names = []
        for name, module in layer.named_modules():
            if any(kw in name for kw in ["mlp", "ffn", "feed_forward"]):
                for subname, child in module.named_children():
                    if isinstance(child, nn.Linear) and any(
                        kw in subname for kw in ["gate_proj", "up_proj", "down_proj", "fc1", "fc2", "w1", "w2", "w3"]
                    ):
                        ffn_names.append(f"{name}.{subname}")
        return ffn_names