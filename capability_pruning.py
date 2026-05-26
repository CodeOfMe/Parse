"""
PARSE experiment configuration and runner.

Orchestrates the complete 4-stage pipeline:
  1. Diagnostic Probing (CIT computation)
  2. Architecture Sculpting (layer selection + FFN transplantation)
  3. Transplantation (FFN removal + NoFFN insertion + DCR attachment)
  4. Dual-Flywheel Recovery (synthetic + self-refining training)

Outputs structured JSON/CSV results for paper writing and visualization.
"""

import json
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

import torch

from code.parse.config import PARSEConfig, PreservationProfile, get_profile, LANGUAGES, DISCIPLINES, SCENARIOS
from code.parse.core.model import build_parse_model
from code.parse.data.calibration import build_default_calibration, CalibrationData
from code.parse.trainer.flywheel import DualFlywheelTrainer, FlywheelConfig
from code.parse.eval.metrics import evaluate_capabilities, compute_CRR, compute_CCI
from code.parse.export import export_to_gguf


@dataclass
class ExperimentConfig:
    """User-facing experiment configuration (matches run_experiment.py CLI args)."""

    base_model_path: str
    device: str = "cuda"
    strategy: str = "parse"  # "parse", "wanda", "layerdrop", "magnitude", "hybrid"
    target_sparsity: float = 0.5

    # Preservation profile
    preserve_languages: List[str] = field(default_factory=lambda: ["zh", "en"])
    preserve_domains: List[str] = field(default_factory=lambda: ["math", "logic"])
    preserve_scenarios: List[str] = field(default_factory=lambda: ["fc", "math_reasoning"])

    # Output
    output_dir: str = "results/experiments"
    save_pruned_model: bool = True

    # CIT
    cit_alpha: float = 0.6
    calibration_samples: int = 15

    # Flywheel
    flywheel_rounds: int = 3
    flywheel_learning_rate: float = 1e-5
    enable_grpo: bool = True

    # Export
    export_gguf: bool = False
    gguf_quantization: str = "Q4_K_M"

    def to_parse_config(self) -> PARSEConfig:
        return PARSEConfig(
            base_model_path=self.base_model_path,
            device=self.device,
            target_sparsity=self.target_sparsity,
            languages=self.preserve_languages,
            disciplines=self.preserve_domains,
            scenarios=self.preserve_scenarios,
            output_dir=self.output_dir,
            cit_alpha=self.cit_alpha,
            calibration_samples_per_category=self.calibration_samples,
            flywheel_rounds=self.flywheel_rounds,
            flywheel_learning_rate=self.flywheel_learning_rate,
            enable_grpo=self.enable_grpo,
            gguf_quantization=self.gguf_quantization,
            export_dir=str(Path(self.output_dir) / "export"),
        )


class ExperimentRunner:
    """
    Complete PARSE experiment runner implementing the 4-stage pipeline.

    Usage:
        config = ExperimentConfig(base_model_path="models/...", ...)
        runner = ExperimentRunner(model, tokenizer, config)
        results = runner.run_full_experiment()
    """

    def __init__(self, model, tokenizer, config: ExperimentConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._parse_config = config.to_parse_config()

        # Timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config.output_dir) / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Accumulate results
        self.results: Dict[str, Any] = {
            "experiment_config": asdict(config),
            "timestamp": timestamp,
            "pipeline": {},
            "evaluation": {},
        }

    def run_full_experiment(self) -> Dict[str, Any]:
        """
        Execute the complete 4-stage PARSE pipeline.

        Returns a dict with all experiment results for JSON/CSV export.
        """
        device = self.config.device
        print(f"\n{'='*60}")
        print(f"PARSE Experiment — {self.config.strategy}")
        print(f"  Model: {self.config.base_model_path}")
        print(f"  Device: {device}")
        print(f"  Profile: L={self.config.preserve_languages} "
              f"D={self.config.preserve_domains} S={self.config.preserve_scenarios}")
        print(f"  Sparsity: {self.config.target_sparsity}")
        print(f"{'='*60}\n")

        # ── Stage 1: Diagnostic Probing ──
        print("Stage 1/4: Diagnostic Probing (CIT computation)...")
        t0 = time.time()
        cit_result = self._run_diagnostic_probing()
        self.results["pipeline"]["diagnostic"] = {
            "duration_s": time.time() - t0,
            "n_retained": cit_result["n_retained"],
            "n_pruned": cit_result["n_pruned"],
            "retained_layers": cit_result["retained_layers"],
            "pruned_layers": cit_result["pruned_layers"],
        }
        print(f"  Retained: {cit_result['n_retained']} layers ({cit_result['retained_layers']})")
        print(f"  Pruned:   {cit_result['n_pruned']} layers ({cit_result['pruned_layers']})")

        # ── Stage 2: Architecture Transplantation ──
        print("\nStage 2/4: Architecture Transplantation (FFN removal + NoFFN insertion)...")
        t0 = time.time()
        transplant_result = self._run_transplantation(cit_result)
        self.results["pipeline"]["transplantation"] = {
            "duration_s": time.time() - t0,
            **transplant_result,
        }
        print(f"  Removed {transplant_result['removed_ffn_params']:,} FFN params")
        print(f"  Added {transplant_result['noffn_added_params']:,} NoFFN params "
              f"(+{transplant_result['dcr_params']:,} DCR)")
        print(f"  Total: {transplant_result['total_params_after']:,} parameters")

        # ── Stage 3: Dual-Flywheel Recovery ──
        print("\nStage 3/4: Dual-Flywheel Recovery...")
        t0 = time.time()
        flywheel_result = self._run_flywheel_recovery()
        self.results["pipeline"]["flywheel"] = {
            "duration_s": time.time() - t0,
            **flywheel_result,
        }

        # ── Stage 4: Evaluation ──
        print("\nStage 4/4: Capability Evaluation...")
        t0 = time.time()
        eval_result = self._run_evaluation()
        self.results["evaluation"] = eval_result
        self.results["pipeline"]["evaluation"] = {"duration_s": time.time() - t0}
        print(f"  CRR: {eval_result.get('avg_crr', 'N/A'):.4f}" if eval_result.get('avg_crr') else "  CRR: N/A")
        print(f"  CCI: {eval_result.get('cci', 'N/A'):.4f}" if eval_result.get('cci') else "  CCI: N/A")

        # ── Export ──
        if self.config.export_gguf:
            print(f"\nExporting GGUF ({self.config.gguf_quantization})...")
            self._run_gguf_export()

        # ── Save Results ──
        self._save_results()
        return self.results

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _run_diagnostic_probing(self) -> Dict:
        """Stage 1: Compute CIT and select layers."""
        result = build_parse_model(
            self.config.base_model_path,
            device=self.config.device,
        )
        cit_computer = result["cit_computer"]
        self.model = result["model"]
        self.tokenizer = result["tokenizer"]
        self._n_layers = result["n_layers"]
        self._hidden_size = result["hidden_size"]

        # Load calibration data
        calib = build_default_calibration()
        calib_dict = {"lang": calib.lang, "disc": calib.disc, "scen": calib.scen}
        self._calib = calib  # store for flywheel

        # Compute profile
        cit_result = cit_computer.compute_profile(
            calibration_data=calib_dict,
            profile_languages=self.config.preserve_languages,
            profile_disciplines=self.config.preserve_domains,
            profile_scenarios=self.config.preserve_scenarios,
            target_sparsity=self.config.target_sparsity,
        )

        # Save CIT data
        self._save_cit_csv(cit_result)

        self._cit_lang = cit_result["cit_lang"]
        self._cit_disc = cit_result["cit_disc"]
        self._cit_scen = cit_result["cit_scen"]
        self._lang_cats = cit_result["lang_cats"]
        self._disc_cats = cit_result["disc_cats"]
        self._scen_cats = cit_result["scen_cats"]

        return cit_result

    def _run_transplantation(self, cit_result: Dict) -> Dict:
        """Stage 2: Perform FFN transplantation on pruned layers."""
        result = build_parse_model(
            self.config.base_model_path,
            device=self.config.device,
            enable_dcr=True,
        )
        transplant = result["transplant"]
        self.model = result["model"]
        self._transplant = transplant

        pruned_layers = cit_result["pruned_layers"]
        n_scenarios = len(self.config.preserve_scenarios)

        self.model = transplant.transplant(
            layers_to_transplant=pruned_layers,
            num_scenarios=n_scenarios,
        )

        return transplant.get_param_stats()

    def _run_flywheel_recovery(self) -> Dict:
        """Stage 3: Dual-flywheel capability recovery."""
        flywheel_config = FlywheelConfig(
            rounds=self._parse_config.flywheel_rounds,
            samples_per_round=self._parse_config.flywheel_samples_per_round,
            batch_size=self._parse_config.flywheel_batch_size,
            learning_rate=self._parse_config.flywheel_learning_rate,
            enable_grpo=self._parse_config.enable_grpo,
            device=self.config.device,
        )

        calib_dict = {"lang": self._calib.lang, "disc": self._calib.disc, "scen": self._calib.scen}

        trainer = DualFlywheelTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=flywheel_config,
            calibration_data=calib_dict,
        )

        profile = {
            "languages": self.config.preserve_languages,
            "disciplines": self.config.preserve_domains,
            "scenarios": self.config.preserve_scenarios,
        }

        metrics = trainer.run_full_recovery(profile)

        return {
            "synthetic_loss_final": metrics["synthetic_loss"][-1] if metrics["synthetic_loss"] else None,
            "self_refining_loss_final": metrics["self_refining_loss"][-1] if metrics["self_refining_loss"] else None,
            "grpo_final_reward": metrics["grpo_reward"][-1] if metrics.get("grpo_reward") and metrics["grpo_reward"] else None,
        }

    def _run_evaluation(self) -> Dict:
        """Stage 4: Evaluate capability retention."""
        calib = self._calib

        # Build eval data: all categories
        eval_data = {}
        for lang in LANGUAGES:
            prompts = calib.lang.get(lang, [])
            if prompts:
                eval_data[f"lang_{lang}"] = prompts
        for disc in DISCIPLINES:
            prompts = calib.disc.get(disc, [])
            if prompts:
                eval_data[f"disc_{disc}"] = prompts
        for scen in SCENARIOS:
            prompts = calib.scen.get(scen, [])
            if prompts:
                eval_data[f"scen_{scen}"] = prompts

        # Original model for CRR (reload fresh)
        result = build_parse_model(
            self.config.base_model_path,
            device=self.config.device,
        )
        original_model = result["model"]

        eval_results = evaluate_capabilities(
            self.model, self.tokenizer, eval_data,
            original_model=original_model,
            device=self.config.device,
        )

        # Del original model
        del original_model
        if self.config.device == "cuda":
            torch.cuda.empty_cache()

        # Compute aggregate metrics
        preserved = []
        for lang in self.config.preserve_languages:
            preserved.append(f"lang_{lang}")
        for disc in self.config.preserve_domains:
            preserved.append(f"disc_{disc}")
        for scen in self.config.preserve_scenarios:
            preserved.append(f"scen_{scen}")

        avg_crr = compute_CRR(eval_results, preserved)
        cci = compute_CCI(eval_results, preserved)

        return {
            "per_capability": eval_results,
            "avg_crr": avg_crr,
            "cci": cci,
            "preserved_capabilities": preserved,
        }

    def _run_gguf_export(self):
        """Export compressed model to GGUF format."""
        paths = export_to_gguf(
            self.model, self.tokenizer,
            output_name=f"parse_{self.config.strategy}",
            output_dir=str(Path(self.config.output_dir) / "export"),
            quantization=self.config.gguf_quantization,
        )
        self.results["gguf_export"] = paths

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _save_cit_csv(self, cit_result: Dict):
        """Save CIT tensors as CSV for figure generation."""
        # Language CIT
        cit_lang = cit_result["cit_lang"]
        with open(self.output_dir / "cit_language.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["layer"] + cit_result["lang_cats"])
            for l in range(cit_lang.shape[0]):
                writer.writerow([l] + cit_lang[l].tolist())

        # Discipline CIT
        cit_disc = cit_result["cit_disc"]
        with open(self.output_dir / "cit_discipline.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["layer"] + cit_result["disc_cats"])
            for l in range(cit_disc.shape[0]):
                writer.writerow([l] + cit_disc[l].tolist())

        # Scenario CIT
        cit_scen = cit_result["cit_scen"]
        with open(self.output_dir / "cit_scenario.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["layer"] + cit_result["scen_cats"])
            for l in range(cit_scen.shape[0]):
                writer.writerow([l] + cit_scen[l].tolist())

    def _save_results(self):
        """Save all experiment results to disk."""
        # JSON (full)
        results_path = self.output_dir / "experiment_results.json"

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, torch.Tensor):
                    return obj.cpu().tolist()
                return super().default(obj)

        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, cls=NumpyEncoder, default=str)

        print(f"\n  Results saved to {results_path}")

        # CSV (key metrics)
        metrics_csv = self.output_dir / "summary_metrics.csv"
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["strategy", self.config.strategy])
            writer.writerow(["target_sparsity", self.config.target_sparsity])
            writer.writerow(["n_layers_retained", self.results["pipeline"]["diagnostic"]["n_retained"]])
            writer.writerow(["n_layers_pruned", self.results["pipeline"]["diagnostic"]["n_pruned"]])
            writer.writerow(["total_params_after", self.results["pipeline"]["transplantation"]["total_params_after"]])
            writer.writerow(["dcr_params", self.results["pipeline"]["transplantation"]["dcr_params"]])
            writer.writerow(["avg_crr", self.results["evaluation"].get("avg_crr", "N/A")])
            writer.writerow(["cci", self.results["evaluation"].get("cci", "N/A")])

        print(f"  Summary metrics saved to {metrics_csv}")
