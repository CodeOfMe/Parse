"""
Capability Importance Tensor (CIT) computation.

For each layer l and capability combination c = (lang, disc, scen), computes:

    CIT(l, c) = α · A(l, c) + (1-α) · G(l, c)

where:
  - A(l,c) = Activation Capacitance (hidden state L1 norm, normalized)
  - G(l,c) = Gradient Sensitivity (|∂L/∂W · W|, normalized)

Uses factorized computation:
    CIT(l, lang, disc, scen) = CIT_lang(l, lang) · CIT_disc(l, disc) · CIT_scen(l, scen)

Reducing complexity from O(L × |L|×|D|×|S|) to O(L × (|L|+|D|+|S|)).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class ComputeCIT:
    """
    Compute the Capability Importance Tensor for a given model on calibration data.

    Parameters
    ----------
    model : nn.Module
        The base language model (e.g., Qwen3.5-0.8B).
    tokenizer : PreTrainedTokenizer
        Tokenizer for encoding calibration prompts.
    device : str
        Compute device.
    alpha : float
        Weight between activation capacitance (α) and gradient sensitivity (1-α).
    n_layers : int
        Total number of Transformer layers in the model.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda",
        alpha: float = 0.6,
        n_layers: int = 24,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.alpha = alpha
        self.n_layers = n_layers

        self.model.to(device)
        self.model.eval()

    def compute_marginal(
        self,
        calibration_data: Dict[str, List[str]],
        axis_name: str,
    ) -> torch.Tensor:
        """
        Compute marginal CIT for a single axis (language, discipline, or scenario).

        Parameters
        ----------
        calibration_data : dict
            Mapping from category name (e.g. "zh", "math", "fc") to list of prompt strings.
        axis_name : str
            One of "lang", "disc", "scen".

        Returns
        -------
        Tensor of shape (n_layers, n_categories) with normalized CIT values.
        """
        categories = list(calibration_data.keys())
        n_categories = len(categories)
        cit_marginal = torch.zeros(self.n_layers, n_categories, device=self.device)

        for c_idx, (cat_name, prompts) in enumerate(calibration_data.items()):
            if not prompts:
                continue

            # Encode all prompts for this category
            encodings = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            # 1. Activation capacitance: register hooks to capture hidden states
            activations = self._collect_activations(encodings)

            # 2. Gradient sensitivity: compute gradients w.r.t. loss
            gradients = self._collect_gradients(encodings)

            # Combine activation and gradient signals (normalize within layer)
            for l in range(self.n_layers):
                A = activations[l] / (activations.max() + 1e-8) if activations[l] > 0 else 0.0
                G = gradients[l] / (gradients.max() + 1e-8) if gradients[l] > 0 else 0.0
                cit_marginal[l, c_idx] = self.alpha * A + (1 - self.alpha) * G

        # Normalize each category's scores to sum to 1 across layers
        cit_marginal = cit_marginal / (cit_marginal.sum(dim=0, keepdim=True) + 1e-8)

        return cit_marginal.cpu()

    def combine_to_full(
        self,
        cit_lang: torch.Tensor,
        cit_disc: torch.Tensor,
        cit_scen: torch.Tensor,
        profile_languages: List[str],
        profile_disciplines: List[str],
        profile_scenarios: List[str],
        lang_categories: List[str],
        disc_categories: List[str],
        scen_categories: List[str],
    ) -> torch.Tensor:
        """
        Combine marginal CIT tensors into full preservation-weighted layer scores.

        Uses factorized product: CIT(l, lang, disc, scen) = CIT_lang(l,lang) · CIT_disc(l,disc) · CIT_scen(l,scen)

        Parameters
        ----------
        cit_lang : (n_layers, n_lang_cats)
        cit_disc : (n_layers, n_disc_cats)
        cit_scen : (n_layers, n_scen_cats)
        profile_languages : list of language keys to preserve
        profile_disciplines : list of discipline keys to preserve
        profile_scenarios : list of scenario keys to preserve
        lang_categories : all language category names
        disc_categories : all discipline category names
        scen_categories : all scenario category names

        Returns
        -------
        S_preserve : (n_layers,) tensor of preservation-weighted importance scores.
        """
        lang_indices = [lang_categories.index(l) for l in profile_languages if l in lang_categories]
        disc_indices = [disc_categories.index(d) for d in profile_disciplines if d in disc_categories]
        scen_indices = [scen_categories.index(s) for s in profile_scenarios if s in scen_categories]

        S_preserve = torch.zeros(self.n_layers)

        for li in lang_indices:
            for di in disc_indices:
                for si in scen_indices:
                    # Product of marginal scores
                    score = cit_lang[:, li] * cit_disc[:, di] * cit_scen[:, si]
                    S_preserve += score

        return S_preserve

    def select_layers(
        self,
        S_preserve: torch.Tensor,
        target_sparsity: float,
    ) -> Tuple[List[int], List[int]]:
        """
        Select which layers to retain (top-K by preservation score) and which to prune.

        Parameters
        ----------
        S_preserve : (n_layers,) preservation-weighted importance.
        target_sparsity : target fraction of layers to prune.

        Returns
        -------
        retained_layers : list of layer indices to keep intact.
        pruned_layers : list of layer indices to transplant (remove FFN).
        """
        K = max(1, int(self.n_layers * (1 - target_sparsity / 2)))
        sorted_indices = torch.argsort(S_preserve, descending=True)
        retained = sorted_indices[:K].tolist()
        pruned = sorted_indices[K:].tolist()
        retained.sort()
        pruned.sort()
        return retained, pruned

    def compute_profile(
        self,
        calibration_data: Dict[str, Dict[str, List[str]]],
        profile_languages: List[str],
        profile_disciplines: List[str],
        profile_scenarios: List[str],
        target_sparsity: float,
    ) -> Dict:
        """
        Full CIT computation pipeline for a given preservation profile.

        Returns a dict with CIT tensors, layer selection, and metadata.
        """
        # Compute marginals
        cit_lang = self.compute_marginal(calibration_data.get("lang", {}), "lang")
        cit_disc = self.compute_marginal(calibration_data.get("disc", {}), "disc")
        cit_scen = self.compute_marginal(calibration_data.get("scen", {}), "scen")

        lang_cats = list(calibration_data.get("lang", {}).keys())
        disc_cats = list(calibration_data.get("disc", {}).keys())
        scen_cats = list(calibration_data.get("scen", {}).keys())

        # Combined preservation score
        S_preserve = self.combine_to_full(
            cit_lang, cit_disc, cit_scen,
            profile_languages, profile_disciplines, profile_scenarios,
            lang_cats, disc_cats, scen_cats,
        )

        retained, pruned = self.select_layers(S_preserve, target_sparsity)

        return {
            "cit_lang": cit_lang,
            "cit_disc": cit_disc,
            "cit_scen": cit_scen,
            "lang_cats": lang_cats,
            "disc_cats": disc_cats,
            "scen_cats": scen_cats,
            "S_preserve": S_preserve,
            "retained_layers": retained,
            "pruned_layers": pruned,
            "n_retained": len(retained),
            "n_pruned": len(pruned),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_activations(self, encodings) -> Dict[int, float]:
        """Run forward pass and collect per-layer hidden state L1 norms."""
        hooks = []
        layer_acts = defaultdict(float)

        def hook_fn(layer_idx):
            def fn(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output
                layer_acts[layer_idx] += out.detach().abs().sum().item()
            return fn

        # Register hooks on each layer's final norm or output
        layers = self._get_transformer_layers()
        for i, layer in enumerate(layers):
            h = layer.register_forward_hook(hook_fn(i))
            hooks.append(h)

        with torch.no_grad():
            self.model(**encodings)

        for h in hooks:
            h.remove()

        return dict(layer_acts)

    def _collect_gradients(self, encodings) -> Dict[int, float]:
        """Run backward pass and collect per-layer gradient sensitivity."""
        params = []
        param_to_layer = {}

        layers = self._get_transformer_layers()
        for i, layer in enumerate(layers):
            for name, p in layer.named_parameters():
                if p.requires_grad and "weight" in name:
                    params.append(p)
                    param_to_layer[id(p)] = i

        labels = encodings["input_ids"].clone()
        outputs = self.model(**encodings, labels=labels)
        loss = outputs.loss
        loss.backward()

        layer_grads = defaultdict(float)
        for p in params:
            if p.grad is not None:
                g = (p.grad * p.data).abs().sum().item()
                layer_grads[param_to_layer[id(p)]] += g

        self.model.zero_grad()
        return dict(layer_grads)

    def _get_transformer_layers(self) -> List[nn.Module]:
        """Extract the list of Transformer layers from the model."""
        model = self.model
        # Qwen / LLaMA convention
        for attr in ["model.layers", "transformer.h", "model.decoder.layers"]:
            parts = attr.split(".")
            obj = model
            try:
                for p in parts:
                    obj = getattr(obj, p)
                return list(obj)
            except (AttributeError, TypeError):
                continue
        raise RuntimeError(
            "Cannot find transformer layers. Expected model.model.layers or similar."
        )
