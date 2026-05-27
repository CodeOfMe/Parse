#!/usr/bin/env python3
"""
PARSE Local Experiment Runner — Lightweight CIT diagnostic and evaluation.

Designed for Apple Silicon / CPU. Runs activation-based CIT computation,
generates paper-quality figures, and evaluates baseline model perplexity.
All results are from real measurements; no synthetic data.

Usage:
    python run_local.py                    # Full CIT + figures + eval
    python run_local.py --skip-figures     # CIT + eval only
    python run_local.py --profile P1       # Run for a specific profile
    python run_local.py --device cpu       # Force CPU
"""

import os
import sys
import csv
import time
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# Add code/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))

from parse.config import PROFILES, get_profile, LANGUAGES, DISCIPLINES, SCENARIOS
from parse.data.calibration import build_default_calibration
from parse.utils import (
    detect_device, device_info, get_transformer_layers,
    normalize_cit, mean_pairwise_r, cross_axis_r, deep_shallow_ratio,
    save_json, Timer, NumpyEncoder,
)
from parse.visualization import generate_all_figures

# ── Config ───────────────────────────────────────────────────────────

MODEL_PATH = "models/qwen/Qwen3___5-0___8B"
OUTPUT_DIR = "results/local"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Model loading ────────────────────────────────────────────────────

def load_model(device: str):
    """Load Qwen3.5-0.8B on the given device."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model on {device}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "mps" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True, dtype=dtype,
        device_map=device, local_files_only=True)
    model.eval()

    layers = get_transformer_layers(model)
    n_layers = len(layers)
    print(f"  Loaded in {time.time()-t0:.1f}s — {n_layers} layers")
    return model, tokenizer, layers, n_layers


# ── CIT Computation ──────────────────────────────────────────────────

def compute_activation_cit(model, tokenizer, layers, calibration_data, device, max_len=128):
    """Compute activation-only CIT for one axis.

    Returns (n_layers, n_categories) numpy array and category names.
    """
    categories = list(calibration_data.keys())
    n_cats = len(categories)
    n_layers = len(layers)
    cit = np.zeros((n_layers, n_cats), dtype=np.float64)

    for c_idx, (cat_name, prompts) in enumerate(calibration_data.items()):
        if not prompts:
            continue
        layer_acts = defaultdict(float)
        total_tokens = 0

        for prompt in prompts[:10]:  # Use 10 per category
            enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=max_len).to(device)

            hooks = []
            def hook_fn(layer_idx):
                def fn(module, input, output):
                    out = output[0] if isinstance(output, tuple) else output
                    layer_acts[layer_idx] += out.detach().cpu().abs().sum().item()
                return fn

            for i, layer in enumerate(layers):
                hooks.append(layer.register_forward_hook(hook_fn(i)))

            with torch.no_grad():
                _ = model(**enc)

            for h in hooks:
                h.remove()
            total_tokens += enc["input_ids"].shape[1]
            del enc

        for l in range(n_layers):
            cit[l, c_idx] = layer_acts.get(l, 0) / max(total_tokens, 1)

        print(f"    [{cat_name}] {len(prompts[:10])} prompts, {total_tokens} tokens")

        if device == "mps":
            torch.mps.empty_cache()

    return normalize_cit(cit), categories


def combine_to_full(cit_lang, cit_disc, cit_scen, profile_langs, profile_discs, profile_scens,
                    lang_cats, disc_cats, scen_cats):
    """Factorized CIT combination for a preservation profile."""
    li = [lang_cats.index(l) for l in profile_langs if l in lang_cats]
    di = [disc_cats.index(d) for d in profile_discs if d in disc_cats]
    si = [scen_cats.index(s) for s in profile_scens if s in scen_cats]
    S = np.zeros(cit_lang.shape[0])
    for l in li:
        for d in di:
            for s in si:
                S += cit_lang[:, l] * cit_disc[:, d] * cit_scen[:, s]
    return S


def select_layers(S, n_layers, target_sparsity=0.5, forced_retain=None):
    """Select top-K layers to retain, respecting forced retain set."""
    if forced_retain is None:
        forced_retain = {3, 7, 11, 15, 19, 23}
    K = max(1, int(n_layers * (1 - target_sparsity / 2)))
    forced = forced_retain & set(range(n_layers))
    sorted_idx = np.argsort(S)[::-1]
    retained = set(forced)
    for idx in sorted_idx:
        if len(retained) >= K:
            break
        retained.add(idx)
    pruned = sorted([i for i in range(n_layers) if i not in retained])
    return sorted(retained), pruned


# ── Evaluation ────────────────────────────────────────────────────────

def evaluate_perplexity(model, tokenizer, prompts, device, max_len=256):
    """Compute perplexity on a set of prompts."""
    total_loss = 0.0
    total_tokens = 0
    for prompt in prompts[:5]:  # 5 per category to keep it fast
        enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=max_len).to(device)
        labels = enc["input_ids"].clone()
        with torch.no_grad():
            out = model(**enc, labels=labels)
        total_loss += out.loss.item() * labels.shape[1]
        total_tokens += labels.shape[1]
        del enc, out
    return np.exp(total_loss / max(total_tokens, 1)) if total_tokens else float('inf')


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PARSE Local Experiment Runner")
    parser.add_argument("--device", default="auto", help="Compute device")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--profile", default=None, help="Specific profile (P1-P12)")
    args = parser.parse_args()

    device = detect_device(args.device)
    info = device_info(device)
    print("=" * 60)
    print("PARSE Local Experiment Runner")
    print(f"Device: {device} | {info.get('chip', info.get('gpu_name', ''))}")
    print("=" * 60)

    # ── 1. Load model ──
    with Timer("Model loading"):
        model, tokenizer, layers, n_layers = load_model(device)

    # ── 2. Load calibration data ──
    cal = build_default_calibration()
    print(f"\nCalibration: {len(cal.lang)} lang × {len(cal.disc)} disc × {len(cal.scen)} scen")

    # ── 3. Compute CIT ──
    print("\n[Stage 1] Computing activation-based CIT...")
    with Timer("CIT computation"):
        cit_lang, lang_cats = compute_activation_cit(model, tokenizer, layers, cal.lang, device)
        cit_disc, disc_cats = compute_activation_cit(model, tokenizer, layers, cal.disc, device)
        cit_scen, scen_cats = compute_activation_cit(model, tokenizer, layers, cal.scen, device)

    # ── 4. Correlation analysis ──
    r_lang_lang = mean_pairwise_r(cit_lang)
    r_disc_disc = mean_pairwise_r(cit_disc)
    r_scen_scen = mean_pairwise_r(cit_scen)
    r_lang_disc, _ = cross_axis_r(cit_lang, cit_disc)
    r_lang_scen, _ = cross_axis_r(cit_lang, cit_scen)
    r_disc_scen, _ = cross_axis_r(cit_disc, cit_scen)
    r_cross_mean = (r_lang_disc + r_lang_scen + r_disc_scen) / 3

    # Minimum cross-axis pair
    min_r, min_pair = 1.0, ""
    for i, lc in enumerate(lang_cats):
        for j, dc in enumerate(disc_cats):
            r = np.corrcoef(cit_lang[:, i], cit_disc[:, j])[0, 1]
            if r < min_r:
                min_r, min_pair = r, f"{lc}-{dc}"

    lang_cliff = deep_shallow_ratio(cit_lang)
    disc_cliff = deep_shallow_ratio(cit_disc)
    scen_cliff = deep_shallow_ratio(cit_scen)

    print(f"\n=== CIT Correlation Results ===")
    print(f"  Within-axis:  Lang={r_lang_lang:.4f}  Disc={r_disc_disc:.4f}  Scen={r_scen_scen:.4f}")
    print(f"  Cross-axis:   Lang-Disc={r_lang_disc:.4f}  Lang-Scen={r_lang_scen:.4f}  Disc-Scen={r_disc_scen:.4f}")
    print(f"  Mean cross:   {r_cross_mean:.4f}")
    print(f"  Min pair:     {min_pair} at r={min_r:.4f}")
    print(f"  Capability Cliff: Lang={lang_cliff.mean():.2f}×  Disc={disc_cliff.mean():.2f}×  Scen={scen_cliff.mean():.2f}×")

    # ── 5. Layer selection for all profiles ──
    prof_arg = args.profile
    profile_names = [prof_arg] if prof_arg else list(PROFILES.keys())
    profile_results = {}

    print(f"\n=== Layer Selection ===")
    for pname in profile_names:
        prof = PROFILES[pname] if pname in PROFILES else get_profile(pname)
        S = combine_to_full(cit_lang, cit_disc, cit_scen,
                           prof.languages, prof.disciplines, prof.scenarios,
                           lang_cats, disc_cats, scen_cats)
        retained, pruned = select_layers(S, n_layers)
        profile_results[pname] = {
            "description": prof.description,
            "languages": prof.languages,
            "disciplines": prof.disciplines,
            "scenarios": prof.scenarios,
            "retained": retained,
            "pruned": pruned,
            "n_retained": len(retained),
            "n_pruned": len(pruned),
            "S_preserve": S.tolist(),
        }
        print(f"  {pname}: retain {retained}, prune {pruned} | {prof.description}")

    # ── 6. Baseline PPL evaluation ──
    if not args.skip_eval:
        print(f"\n[Stage 4] Baseline perplexity evaluation...")
        ppl_results = {}
        with Timer("PPL evaluation"):
            for axis_name, cal_data, cats in [
                ("lang", cal.lang, lang_cats),
                ("disc", cal.disc, disc_cats),
                ("scen", cal.scen, scen_cats),
            ]:
                for cat in cats[:3]:  # First 3 per axis to keep fast
                    prompts = cal_data.get(cat, [])
                    if prompts:
                        ppl = evaluate_perplexity(model, tokenizer, prompts, device)
                        ppl_results[f"{axis_name}/{cat}"] = round(ppl, 2)
                        print(f"    {axis_name}/{cat}: PPL={ppl:.2f}")

    # ── 7. Save results ──
    results = {
        "metadata": {
            "model": "Qwen3.5-0.8B",
            "device": device,
            "device_info": info,
            "n_layers": n_layers,
            "cit_method": "activation-only",
            "samples_per_category": 10,
            "standard_attn_layers": [3, 7, 11, 15, 19, 23],
        },
        "correlations": {
            "within_axis": {"lang_lang": round(r_lang_lang, 4), "disc_disc": round(r_disc_disc, 4), "scen_scen": round(r_scen_scen, 4)},
            "cross_axis": {"lang_disc": round(r_lang_disc, 4), "lang_scen": round(r_lang_scen, 4), "disc_scen": round(r_disc_scen, 4), "mean": round(r_cross_mean, 4)},
            "min_pair": {"pair": min_pair, "r": round(min_r, 4)},
        },
        "capability_cliff": {
            "lang_mean": round(float(lang_cliff.mean()), 2),
            "disc_mean": round(float(disc_cliff.mean()), 2),
            "scen_mean": round(float(scen_cliff.mean()), 2),
        },
        "profiles": {p: {k: v for k, v in d.items() if k != "S_preserve"} for p, d in profile_results.items()},
    }
    if not args.skip_eval:
        results["baseline_ppl"] = ppl_results

    save_json(results, os.path.join(OUTPUT_DIR, "experiment_results.json"))

    # Save CIT CSVs
    for name, cit, cats in [("cit_language", cit_lang, lang_cats),
                             ("cit_discipline", cit_disc, disc_cats),
                             ("cit_scenario", cit_scen, scen_cats)]:
        path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer"] + cats)
            for l in range(n_layers):
                w.writerow([l] + [f"{cit[l, i]:.6f}" for i in range(len(cats))])

    # ── 8. Generate figures ──
    if not args.skip_figures:
        print(f"\n[Figures] Generating publication-quality figures...")
        with Timer("Figure generation"):
            generate_all_figures(
                cit_lang, lang_cats, cit_disc, disc_cats, cit_scen, scen_cats,
                profile_results=profile_results,
                output_dir="figures",
                formats=("pdf", "png"),
            )
        print(f"  Saved to figures/")

    # ── Cleanup ──
    del model
    if device == "mps":
        torch.mps.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"COMPLETE — Results saved to {OUTPUT_DIR}/")
    print(f"  experiment_results.json")
    print(f"  cit_language.csv, cit_discipline.csv, cit_scenario.csv")
    if not args.skip_figures:
        print(f"  figures/fig1–fig4 (PDF + PNG)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
