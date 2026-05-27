#!/usr/bin/env python3
"""
PARSE GPU Experiment Runner — Full transplantation + flywheel + evaluation.

Designed for NVIDIA RTX 4060 (8GB VRAM) with CUDA.
Runs the complete 4-stage PARSE pipeline on a standard Transformer architecture.
Assumes the model has explicit FFN submodules (gate_proj, up_proj, down_proj).

Prerequisites:
    pip install torch transformers modelscope numpy tqdm

Usage:
    python run_gpu.py                          # Full pipeline with P1
    python run_gpu.py --profile P6             # Bilingual developer agent
    python run_gpu.py --sparsity 0.5 --rounds 3
    python run_gpu.py --no-flywheel            # Skip recovery (ablation)
    python run_gpu.py --export-gguf            # Export quantized GGUF
"""

import os
import sys
import json
import csv
import time
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))

from parse.config import PARSEConfig, PROFILES, get_profile, profile_to_config
from parse.data.calibration import build_default_calibration, CalibrationData
from parse.utils import (
    detect_device, device_info, get_transformer_layers,
    detect_model_dims, detect_ffn_params,
    normalize_cit, mean_pairwise_r, cross_axis_r, deep_shallow_ratio,
    save_json, load_json, Timer,
)
from parse.core.cit import ComputeCIT
from parse.core.transplant import TransplantFFN
from parse.eval.metrics import evaluate_capabilities, compute_CRR, compute_CCI

OUTPUT_DIR = "results/gpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# Stage 1: CIT Diagnostic Probing
# ══════════════════════════════════════════════════════════════════════

def run_stage1(model, tokenizer, cal, config, device):
    """Compute CIT and select layers."""
    print("\n" + "=" * 50)
    print("STAGE 1: CIT Diagnostic Probing")
    print("=" * 50)

    cit_computer = ComputeCIT(
        model, tokenizer, device=device,
        alpha=config.cit_alpha, n_layers=config.n_layers_total,
    )

    with Timer("CIT marginals"):
        cit_lang = cit_computer.compute_marginal(cal.lang, "lang")
        cit_disc = cit_computer.compute_marginal(cal.disc, "disc")
        cit_scen = cit_computer.compute_marginal(cal.scen, "scen")

    lang_cats = list(cal.lang.keys())
    disc_cats = list(cal.disc.keys())
    scen_cats = list(cal.scen.keys())

    # Correlation
    cit_lang_np, cit_disc_np = cit_lang.numpy(), cit_disc.numpy()
    r_lang_disc, _ = cross_axis_r(cit_lang_np, cit_disc_np)
    lang_cliff = deep_shallow_ratio(cit_lang_np)
    disc_cliff = deep_shallow_ratio(cit_disc_np)

    print(f"  Cross-axis Lang-Disc r: {r_lang_disc:.4f}")
    print(f"  Capability Cliff: Lang={lang_cliff.mean():.2f}×  Disc={disc_cliff.mean():.2f}×")

    # Layer selection
    S = cit_computer.combine_to_full(
        cit_lang, cit_disc, cit_scen,
        config.languages, config.disciplines, config.scenarios,
        lang_cats, disc_cats, scen_cats,
    )
    retained, pruned = cit_computer.select_layers(S, config.target_sparsity)

    print(f"  Profile: {len(retained)} retained, {len(pruned)} pruned")
    print(f"  Retained: {retained}")
    print(f"  Pruned:    {pruned}")

    # Save CIT data
    for name, cit, cats in [("cit_language", cit_lang, lang_cats),
                             ("cit_discipline", cit_disc, disc_cats),
                             ("cit_scenario", cit_scen, scen_cats)]:
        path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer"] + cats)
            for l in range(config.n_layers_total):
                w.writerow([l] + [f"{cit[l, i].item():.6f}" for i in range(len(cats))])

    return retained, pruned, {
        "r_lang_disc": round(r_lang_disc, 4),
        "cliff_lang": round(float(lang_cliff.mean()), 2),
        "cliff_disc": round(float(disc_cliff.mean()), 2),
    }


# ══════════════════════════════════════════════════════════════════════
# Stage 2: Architecture Transplantation
# ══════════════════════════════════════════════════════════════════════

def run_stage2(model, pruned_layers, enable_dcr, device):
    """Transplant FFN in pruned layers."""
    print("\n" + "=" * 50)
    print("STAGE 2: Architecture Transplantation")
    print("=" * 50)

    transplant = TransplantFFN(model, device=device, enable_dcr=enable_dcr)

    with Timer("Transplantation"):
        stats = transplant.transplant(pruned_layers)

    print(f"  FFN params removed:  {stats['ffn_removed']:,}")
    print(f"  NoFFN params added:  {stats['noffn_added']:,}")
    print(f"  DCR params:          {stats['dcr_params']:,}")
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params after:  {total:,}")

    return stats


# ══════════════════════════════════════════════════════════════════════
# Stage 3: Dual-Flywheel Recovery
# ══════════════════════════════════════════════════════════════════════

def run_stage3(model, tokenizer, cal, config, device):
    """Run dual-flywheel recovery training."""
    print("\n" + "=" * 50)
    print("STAGE 3: Dual-Flywheel Recovery")
    print("=" * 50)

    from parse.trainer.flywheel import DualFlywheelTrainer, FlywheelConfig

    flywheel_cfg = FlywheelConfig(
        rounds=config.flywheel_rounds,
        samples_per_round=config.flywheel_samples_per_round,
        learning_rate=config.flywheel_learning_rate,
        batch_size=config.flywheel_batch_size,
        enable_grpo=config.enable_grpo,
        grpo_group_size=config.grpo_group_size,
    )

    trainer = DualFlywheelTrainer(model, tokenizer, flywheel_cfg, cal)

    metrics_history = []
    with Timer("Flywheel recovery"):
        for round_idx in range(config.flywheel_rounds):
            print(f"\n  --- Round {round_idx + 1}/{config.flywheel_rounds} ---")

            # Synthetic flywheel
            synth_metrics = trainer.synthetic_flywheel_step()
            print(f"    Synthetic: loss={synth_metrics.get('loss', 0):.4f}")

            # Self-refining flywheel (GRPO)
            if config.enable_grpo:
                grpo_metrics = trainer.self_refining_flywheel_step()
                reward = grpo_metrics.get('reward_mean', 0)
                print(f"    GRPO:      reward={reward:.4f}")

            metrics_history.append({**synth_metrics, **(grpo_metrics if config.enable_grpo else {})})

    # Save checkpoint
    checkpoint_path = os.path.join(OUTPUT_DIR, "flywheel_checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"  Checkpoint saved to {checkpoint_path}")

    return metrics_history


# ══════════════════════════════════════════════════════════════════════
# Stage 4: Evaluation
# ══════════════════════════════════════════════════════════════════════

def run_stage4(compressed_model, original_model, tokenizer, cal, config, device):
    """Evaluate compressed vs original model."""
    print("\n" + "=" * 50)
    print("STAGE 4: Evaluation")
    print("=" * 50)

    with Timer("Evaluation"):
        eval_results = evaluate_capabilities(
            compressed_model, original_model, tokenizer, cal, device,
            preserved_languages=config.languages,
            preserved_disciplines=config.disciplines,
            preserved_scenarios=config.scenarios,
        )

    crr = compute_CRR(eval_results)
    cci = compute_CCI(eval_results)
    compressed_params = sum(p.numel() for p in compressed_model.parameters())
    original_params = sum(p.numel() for p in original_model.parameters())
    prr = (original_params - compressed_params) / original_params

    print(f"  CRR: {crr:.4f}")
    print(f"  CCI: {cci:.4f}")
    print(f"  PRR: {prr:.2%} ({original_params:,} → {compressed_params:,})")

    return {
        "CRR": round(crr, 4),
        "CCI": round(cci, 4),
        "PRR": round(prr, 4),
        "original_params": original_params,
        "compressed_params": compressed_params,
        "per_capability": eval_results,
    }


# ══════════════════════════════════════════════════════════════════════
# GGUF Export
# ══════════════════════════════════════════════════════════════════════

def run_export(model, tokenizer, config):
    """Export compressed model to GGUF."""
    print("\n" + "=" * 50)
    print("EXPORT: GGUF Conversion")
    print("=" * 50)

    from parse.export import export_to_gguf

    export_dir = os.path.join(OUTPUT_DIR, "gguf_export")
    with Timer("GGUF export"):
        result = export_to_gguf(
            model, tokenizer,
            export_dir=export_dir,
            quantization=config.gguf_quantization,
        )

    if result.get("success"):
        print(f"  Exported: {result.get('quantized_path', 'unknown')}")
    else:
        print(f"  Export failed: {result.get('error', 'unknown')}")

    return result


# ══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PARSE GPU Experiment Runner")
    parser.add_argument("--model-path", default="models/qwen/Qwen3___5-0___8B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile", default="P1")
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--cit-alpha", type=float, default=0.6)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--no-flywheel", action="store_true")
    parser.add_argument("--no-dcr", action="store_true")
    parser.add_argument("--skip-stage1", action="store_true")  # Use cached CIT
    parser.add_argument("--export-gguf", action="store_true")
    parser.add_argument("--gguf-quant", default="Q4_K_M")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    device = detect_device(args.device)
    info = device_info(device)

    print("=" * 60)
    print("PARSE GPU Experiment Runner")
    print(f"Device: {device} | {info.get('gpu_name', info.get('chip', ''))}")
    print(f"Profile: {args.profile} | Sparsity: {args.sparsity}")
    print("=" * 60)

    # ── Resolve profile ──
    if args.profile in PROFILES:
        profile = PROFILES[args.profile]
    else:
        profile = get_profile(args.profile)

    config = profile_to_config(
        profile,
        base_model_path=args.model_path,
        device=device,
        target_sparsity=args.sparsity,
        cit_alpha=args.cit_alpha,
        flywheel_rounds=args.rounds,
        enable_grpo=not args.no_flywheel,
        enable_dcr=not args.no_dcr,
        gguf_quantization=args.gguf_quant,
        output_dir=args.output_dir,
    )

    # ── Load model ──
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading model from {args.model_path}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True,
        torch_dtype=torch.float16, device_map=device, local_files_only=True)
    model.eval()

    layers = get_transformer_layers(model)
    n_layers = len(layers)
    config.n_layers_total = n_layers
    print(f"  Loaded in {time.time()-t0:.1f}s — {n_layers} layers")

    # Verify architecture
    ffn_params = detect_ffn_params(model)
    if not ffn_params:
        print("\n⚠️  WARNING: No standard FFN parameters detected!")
        print("   This model uses a Mamba/linear-attention architecture.")
        print("   Transplantation (Stage 2-3) requires a standard Transformer with FFN.")
        print("   Running CIT diagnostic only. Set --skip-stage1 to reuse cached CIT.\n")
        run_transplant = False
    else:
        print(f"  FFN params detected: {ffn_params[:3]}...")
        run_transplant = True

    cal = build_default_calibration()

    # ── Stage 1 ──
    if not args.skip_stage1:
        retained, pruned, stage1_metrics = run_stage1(model, tokenizer, cal, config, device)
    else:
        print("\n[Skipping Stage 1 — using cached CIT]")
        # Load from local results if available
        local_json = os.path.join("results/local", "experiment_results.json")
        if os.path.exists(local_json):
            data = load_json(local_json)
            p1_data = data["profiles"].get(args.profile, {})
            retained = p1_data.get("retained", list(range(n_layers)))
            pruned = p1_data.get("pruned", [])
            stage1_metrics = data.get("correlations", {})
        else:
            retained, pruned = list(range(n_layers)), []

    # ── Stage 2 ──
    if run_transplant and pruned:
        stage2_stats = run_stage2(model, pruned, config.enable_dcr, device)
    else:
        stage2_stats = {}
        print("\n[Skipping Stage 2 — no FFN architecture or no layers to prune]")

    # ── Stage 3 ──
    if run_transplant and pruned and config.enable_grpo:
        stage3_history = run_stage3(model, tokenizer, cal, config, device)
    else:
        stage3_history = []
        print("\n[Skipping Stage 3 — flywheel disabled or no FFN]")

    # ── Stage 4 ──
    # Reload original for comparison (keep compressed model)
    print("\n[Stage 4] Loading original model for baseline comparison...")
    original = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True,
        torch_dtype=torch.float16, device_map=device, local_files_only=True)
    original.eval()

    stage4_results = run_stage4(model, original, tokenizer, cal, config, device)
    del original

    # ── Export ──
    if args.export_gguf and run_transplant:
        export_result = run_export(model, tokenizer, config)
    else:
        export_result = {}

    # ── Save ──
    full_results = {
        "config": {
            "profile": args.profile,
            "languages": profile.languages,
            "disciplines": profile.disciplines,
            "scenarios": profile.scenarios,
            "sparsity": args.sparsity,
            "cit_alpha": args.cit_alpha,
            "flywheel_rounds": args.rounds,
        },
        "device_info": info,
        "stage1": stage1_metrics if not args.skip_stage1 else {"source": "cached"},
        "stage2": stage2_stats,
        "stage3": stage3_history,
        "stage4": stage4_results,
        "export": export_result,
    }
    save_json(full_results, os.path.join(args.output_dir, "experiment_results.json"))

    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE")
    print(f"  CRR: {stage4_results.get('CRR', 'N/A')}")
    print(f"  PRR: {stage4_results.get('PRR', 'N/A')}")
    print(f"  Results: {args.output_dir}/experiment_results.json")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
