#!/usr/bin/env python3
"""
PARSE Experiment Entry Point

Orchestrates the complete 4-stage capability-preserving compression pipeline:
  1. Diagnostic Probing   — CIT computation across Language × Discipline × Scenario
  2. Architecture Sculpting — layer selection based on preservation profile
  3. Transplantation       — FFN removal + NoFFN insertion + DCR attachment
  4. Dual-Flywheel Recovery — synthetic + self-refining training with GRPO

Supports: CUDA, ROCm (AMD), MPS (Apple Silicon), CPU

Usage:
    python run_experiment.py --strategy parse --sparsity 0.5 --device cuda
    python run_experiment.py --profile P1 --device auto
    python run_experiment.py --languages zh en --disciplines math logic \\
        --scenarios fc math_reasoning --export-gguf
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import argparse

import torch

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from code.parse.config import (
    PreservationProfile, get_profile, PROFILES,
    LANGUAGES, DISCIPLINES, SCENARIOS,
)
from capability_pruning import ExperimentConfig, ExperimentRunner


# ── Device Detection ──────────────────────────────────────────────────

def setup_device(device: str) -> str:
    """Auto-detect or validate device selection."""
    if device == "auto":
        if torch.cuda.is_available():
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                print("[device] ROCm (HIP) detected")
            else:
                print("[device] CUDA detected")
            return "cuda"
        elif torch.backends.mps.is_available():
            print("[device] Apple Silicon (MPS) detected")
            return "mps"
        else:
            print("[device] CPU only")
            return "cpu"

    if device == "rocm":
        print("[device] ROCm → CUDA (HIP)")
        return "cuda"

    valid = {"cuda", "mps", "cpu", "rocm"}
    if device not in valid:
        print(f"[device] Unknown '{device}', falling back to CPU")
        return "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        print("[device] CUDA not available, falling back to CPU")
        return "cpu"

    return device if device != "rocm" else "cuda"


# ── Model Loading ─────────────────────────────────────────────────────

def load_model_and_tokenizer(model_path: str, device: str):
    """Load a HuggingFace Qwen model and tokenizer."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[model] Loading from {model_path}")
    print(f"[model] Device: {device}")

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype if device != "cpu" else torch.float32,
        device_map=device if device != "cpu" else "cpu",
        local_files_only=True,
    )
    model.eval()

    return model, tokenizer


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PARSE: Capability-Preserving Model Compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full PARSE pipeline
  python run_experiment.py --strategy parse --sparsity 0.5 --device cuda

  # Pre-defined profile
  python run_experiment.py --profile P1 --device auto

  # Custom preservation profile + GGUF export
  python run_experiment.py \\
      --languages zh en --disciplines math logic \\
      --scenarios fc math_reasoning --export-gguf

  # Benchmark baselines
  python run_experiment.py --strategy wanda --sparsity 0.5 --device cuda

Profiles (P1-P12):
  P1  = zh/en + math/logic + fc/math_reasoning  (Chinese+English STEM+Agent)
  P2  = zh/en/ja + math/physics + all             (East Asian + STEM)
  P3  = en + math + all                           (English math specialist)
  P4  = zh + all + all                            (Chinese full-capability)
  P5  = all + math/logic/physics + fc             (Multilingual STEM agent)
  P6  = zh/en + all + fc/code                     (Bilingual developer agent)
  P7  = all + math + math_reasoning               (Multilingual math solver)
  P8  = zh/en/ja/fr + all + translation           (Quad-lingual translator)
  P9  = all + all + fc                            (Universal function caller)
  P10 = zh/en + all + all                         (Bilingual full-capability)
  P11 = all + math/logic + all                    (Universal STEM preservation)
  P12 = zh/en + math/logic/physics + fc/code/math_reasoning  (Full targeted)
        """,
    )

    # Model & device
    parser.add_argument("--model_path", type=str,
                        default="models/qwen/Qwen3___5-0___8B",
                        help="Path to HuggingFace model checkpoint")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "rocm", "mps", "cpu"],
                        help="Compute device")

    # Strategy & profile
    parser.add_argument("--strategy", type=str, default="parse",
                        choices=["parse", "wanda", "layerdrop", "magnitude", "hybrid"],
                        help="Compression strategy (parse = full PARSE pipeline)")
    parser.add_argument("--profile", type=str, default=None,
                        choices=list(PROFILES.keys()),
                        help="Pre-defined preservation profile (P1-P12)")
    parser.add_argument("--sparsity", type=float, default=0.5,
                        help="Target sparsity ratio (default: 0.5)")

    # Preservation axes
    parser.add_argument("--languages", type=str, nargs="+", default=None,
                        help=f"Languages to preserve ({', '.join(LANGUAGES)})")
    parser.add_argument("--disciplines", type=str, nargs="+", default=None,
                        help=f"Disciplines to preserve ({', '.join(DISCIPLINES)})")
    parser.add_argument("--scenarios", type=str, nargs="+", default=None,
                        help=f"Scenarios to preserve ({', '.join(SCENARIOS)})")

    # CIT
    parser.add_argument("--cit_alpha", type=float, default=0.6,
                        help="CIT α weight (activation vs gradient)")

    # Flywheel
    parser.add_argument("--flywheel_rounds", type=int, default=3,
                        help="Number of dual-flywheel recovery rounds")
    parser.add_argument("--no_grpo", action="store_true",
                        help="Disable GRPO optimization (use simple refinement)")

    # Output
    parser.add_argument("--output_dir", type=str, default="results/experiments",
                        help="Output directory for results")
    parser.add_argument("--save_model", action="store_true",
                        help="Save compressed model checkpoint")
    parser.add_argument("--export_gguf", action="store_true",
                        help="Export compressed model to GGUF for llama.cpp/MoXing")
    parser.add_argument("--gguf_quant", type=str, default="Q4_K_M",
                        choices=["F16", "Q8_0", "Q4_K_M", "Q5_K_M", "Q2_K"],
                        help="GGUF quantization level")

    args = parser.parse_args()

    # ── Resolve profile ──
    if args.profile:
        profile = get_profile(args.profile)
        print(f"[profile] {args.profile}: {profile.description}")
        languages = profile.languages
        disciplines = profile.disciplines
        scenarios = profile.scenarios
    else:
        languages = args.languages or ["zh", "en"]
        disciplines = args.disciplines or ["math", "logic"]
        scenarios = args.scenarios or ["fc", "math_reasoning"]

    # ── Setup device ──
    device = setup_device(args.device)

    # ── Validate model path ──
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    if not model_path.exists():
        print(f"\n[error] Model path not found: {model_path}")
        print("  Download with ModelScope:")
        print("    modelscope download --model Qwen/Qwen3.5-0.8B --local_dir models/qwen/Qwen3___5-0___8B")
        sys.exit(1)

    # ── Load model ──
    model, tokenizer = load_model_and_tokenizer(str(model_path), device)

    # ── Build experiment config ──
    config = ExperimentConfig(
        base_model_path=str(model_path),
        device=device,
        strategy=args.strategy,
        target_sparsity=args.sparsity,
        preserve_languages=languages,
        preserve_domains=disciplines,
        preserve_scenarios=scenarios,
        output_dir=args.output_dir,
        save_pruned_model=args.save_model,
        cit_alpha=args.cit_alpha,
        flywheel_rounds=args.flywheel_rounds,
        enable_grpo=not args.no_grpo,
        export_gguf=args.export_gguf,
        gguf_quantization=args.gguf_quant,
    )

    # ── Run experiment ──
    runner = ExperimentRunner(model, tokenizer, config)
    results = runner.run_full_experiment()

    # ── Cleanup ──
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    print(f"\n{'='*60}")
    print(f"Experiment complete")
    print(f"  Strategy: {config.strategy}")
    print(f"  Profile:  L={languages} D={disciplines} S={scenarios}")
    print(f"  Sparsity: {config.target_sparsity}")
    print(f"  CRR:      {results.get('evaluation', {}).get('avg_crr', 'N/A')}")
    print(f"{'='*60}")
