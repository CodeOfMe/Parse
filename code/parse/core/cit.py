"""
Capability Importance Tensor (CIT) computation.

For each layer l and capability combination c = (lang, disc, scen), computes:

    CIT(l, c) = α · A(l, c) + (1-α) · G(l, c)

where:
  - A(l,c) = Activation Capacitance (hidden state L1 norm per token, per-category summed)
  - G(l,c) = Gradient Sensitivity (|∂L/∂W_FFNe · W_FFNe|, FFN weights only)

Uses factorized computation:
    CIT(l, lang, disc, scen) = CIT_lang(l, lang) · CIT_disc(l, disc) · CIT_scen(l, scen)

Reducing complexity from O(L × |L|×|D|×|S|) to O(L × (|L|+|D|+|S|)).

Two CIT modes are supported:
  1. **Standard CIT** (compute_marginal): Measures absolute importance per capability.
     Produces high cross-axis correlation (r≈0.994) because capabilities share
     similar importance profiles, differing only in magnitude.

  2. **Contrastive CIT** (compute_contrastive): Suppresses shared importance and
     amplifies capability-specific layers. Computes:
         CIT_contrast(l, c) = max(0, CIT(l, c) - λ · mean_{c'≠c}(CIT(l, c')))
     This directly addresses the r≈0.994 limitation by promoting layers where
     capability importance diverges from the cross-category mean.
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
        standard_attn_layers: Optional[set] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.alpha = alpha
        self.n_layers = n_layers
        # Standard attention layers (always retained, never transplanted)
        self.standard_attn_layers = standard_attn_layers or {3, 7, 11, 15, 19, 23}

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

            # Combine activation and gradient signals.
            # Use raw values then normalize per-category across layers to
            # preserve inter-category magnitude differences. The per-element
            # normalization (A/max, G/max) is removed to avoid double-scaling
            # that would destroy relative magnitude information.
            for l in range(self.n_layers):
                cit_marginal[l, c_idx] = self.alpha * activations[l] + (1 - self.alpha) * gradients[l]

        # Normalize each category's scores to sum to 1 across layers.
        # This is the ONLY normalization: per-category, across layers,
        # preserving inter-category magnitude differences.
        cit_marginal = cit_marginal / (cit_marginal.sum(dim=0, keepdim=True) + 1e-8)

        return cit_marginal.cpu()

    def compute_contrastive(
        self,
        calibration_data: Dict[str, List[str]],
        axis_name: str,
        contrast_weight: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute contrastive CIT that maximizes inter-axis discriminability.

        Instead of CIT(l, c) = α·A(l,c) + (1-α)·G(l,c), contrastive CIT
        computes:

            CIT_contrast(l, c) = max(0, CIT_standard(l, c) - contrast_weight · mean_{c'≠c}(CIT_standard(l, c')))

        This suppresses layers that are important for ALL capabilities (high
        cross-axis correlation) and amplifies layers that are selectively important
        for specific capabilities, directly addressing the r≈0.994 limitation.

        Parameters
        ----------
        calibration_data : dict
            Mapping from category name to list of prompt strings.
        axis_name : str
            One of "lang", "disc", "scen".
        contrast_weight : float
            How strongly to suppress shared importance. 0 = standard CIT,
            1 = pure contrastive (only capability-specific layers survive).

        Returns
        -------
        Tensor of shape (n_layers, n_categories) with contrastive-normalized CIT.
        """
        # First compute standard marginal CIT
        standard_cit = self.compute_marginal(calibration_data, axis_name)
        # standard_cit: (n_layers, n_categories), normalized per-column

        categories = list(calibration_data.keys())
        n_categories = len(categories)

        # Compute mean across all other categories for each category
        cit_contrast = torch.zeros_like(standard_cit)
        for c_idx in range(n_categories):
            # Mean of all OTHER categories at each layer
            other_mask = torch.ones(n_categories, dtype=torch.bool)
            other_mask[c_idx] = False
            other_mean = standard_cit[:, other_mask].mean(dim=1)

            # Contrastive: keep only what's ABOVE the cross-category mean
            difference = standard_cit[:, c_idx] - contrast_weight * other_mean
            cit_contrast[:, c_idx] = torch.clamp(difference, min=0)

        # Re-normalize per category to sum to 1 across layers
        col_sums = cit_contrast.sum(dim=0, keepdim=True)
        # Avoid division by zero for categories that became all-zero
        col_sums = torch.where(col_sums > 1e-8, col_sums, torch.ones_like(col_sums))
        cit_contrast = cit_contrast / col_sums

        return cit_contrast.cpu()

    def compute_contrastive_correlation(
        self,
        lang_data: Dict[str, List[str]],
        disc_data: Dict[str, List[str]],
        scen_data: Dict[str, List[str]],
        contrast_weight: float = 0.5,
    ) -> Dict:
        """
        Compute contrastive CIT for all axes and measure the achieved
        reduction in cross-axis correlation.

        Returns a dict with:
        - 'standard_r': mean cross-axis Pearson r using standard CIT
        - 'contrastive_r': mean cross-axis Pearson r using contrastive CIT
        - 'reduction': relative reduction in cross-axis correlation
        - 'cit_lang': contrastive marginal CIT for language axis
        - 'cit_disc': contrastive marginal CIT for discipline axis
        - 'cit_scen': contrastive marginal CIT for scenario axis
        """
        import numpy as np

        # Standard CIT
        cit_lang_std = self.compute_marginal(lang_data, "lang")
        cit_disc_std = self.compute_marginal(disc_data, "disc")

        # Contrastive CIT
        cit_lang_con = self.compute_contrastive(lang_data, "lang", contrast_weight)
        cit_disc_con = self.compute_contrastive(disc_data, "disc", contrast_weight)

        def mean_pairwise_r(cit_matrix):
            n_cats = cit_matrix.shape[1]
            rs = []
            for i in range(n_cats):
                for j in range(i + 1, n_cats):
                    r = np.corrcoef(
                        cit_matrix[:, i].numpy(),
                        cit_matrix[:, j].numpy()
                    )[0, 1]
                    rs.append(r)
            return np.mean(rs)

        # Cross-axis correlation (between lang and disc)
        def cross_axis_r(cit_lang, cit_disc):
            rs = []
            for i in range(cit_lang.shape[1]):
                for j in range(cit_disc.shape[1]):
                    r = np.corrcoef(
                        cit_lang[:, i].numpy(),
                        cit_disc[:, j].numpy()
                    )[0, 1]
                    rs.append(r)
            return np.mean(rs)

        standard_r = cross_axis_r(cit_lang_std, cit_disc_std)
        contrastive_r = cross_axis_r(cit_lang_con, cit_disc_con)

        return {
            "standard_r": standard_r,
            "contrastive_r": contrastive_r,
            "reduction": (standard_r - contrastive_r) / max(standard_r, 1e-8),
            "contrast_weight": contrast_weight,
            "cit_lang": cit_lang_con,
            "cit_disc": cit_disc_con,
        }

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

        Standard attention layers (hybrid architecture: L3,7,11,15,19,23 for Qwen3.5)
        are ALWAYS retained regardless of their preservation score, matching the
        paper's claim that standard attention layers carry critical routing and
        should never be transplanted.

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

        # Always retain standard attention layers
        forced_retain = self.standard_attn_layers & set(range(self.n_layers))

        # Sort by preservation score, excluding forced-retained layers
        sorted_indices = torch.argsort(S_preserve, descending=True)

        retained = set(forced_retain)
        for idx in sorted_indices.tolist():
            if len(retained) >= K:
                break
            retained.add(idx)

        pruned = [i for i in range(self.n_layers) if i not in retained]
        retained = sorted(retained)
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
        """Run backward pass and collect per-layer gradient sensitivity.
        
        Computes G(l,c) = |∂L_c/∂W_l · W_l| where W_l refers to FFN weights only,
        matching the paper's formula scope. Attention parameters are excluded because
        they are preserved (not transplanted) in the PARSE architecture.
        """
        params = []
        param_to_layer = {}

        layers = self._get_transformer_layers()
        ffn_keywords = ["mlp", "feed_forward", "ffn", "gate_proj", "up_proj", "down_proj"]

        for i, layer in enumerate(layers):
            for name, p in layer.named_parameters():
                if p.requires_grad and "weight" in name:
                    # Only include FFN parameters in gradient sensitivity,
                    # since attention weights are always preserved
                    if any(kw in name.lower() for kw in ffn_keywords):
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
