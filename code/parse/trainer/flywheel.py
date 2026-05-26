"""
Dual-Flywheel Capability Recovery Trainer (Section 3.5, Stage 4).

Implements two complementary training loops after FFN transplantation:

1. **Synthetic Flywheel**: Generates calibration data for each (Language x Discipline x Scenario)
   combination using prompt templates, then fine-tunes the compressed model on these samples.

2. **Self-Refining Flywheel**: The compressed model generates responses; a critic scores
   quality; high-quality traces are re-injected with GRPO-based optimization [23, 32, 33].

The combination recovers capability loss from FFN removal through targeted fine-tuning.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import time


@dataclass
class FlywheelConfig:
    """Configuration for dual-flywheel recovery training."""

    rounds: int = 3
    samples_per_round: int = 64
    batch_size: int = 4
    learning_rate: float = 1e-5
    max_seq_length: int = 512

    # GRPO
    enable_grpo: bool = True
    grpo_group_size: int = 4  # samples per prompt for relative comparison

    # Critic
    critic_threshold: float = 0.6  # minimum score to recycle a sample

    # Device
    device: str = "cuda"
    grad_accumulation_steps: int = 1


class DualFlywheelTrainer:
    """
    Dual data flywheel trainer for post-transplantation capability recovery.

    Parameters
    ----------
    model : nn.Module
        The compressed (post-transplant) model.
    tokenizer : PreTrainedTokenizer
    config : FlywheelConfig
    calibration_data : dict
        Mapping of capability axis -> category -> list of prompts (from CalibrationData).
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: FlywheelConfig,
        calibration_data: Optional[Dict[str, Dict[str, List[str]]]] = None,
    ):
        self.model = model.to(config.device)
        self.tokenizer = tokenizer
        self.config = config
        self.calibration_data = calibration_data or {}

        self.optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
        )

        self.metrics: Dict[str, List[float]] = {
            "round": [],
            "synthetic_loss": [],
            "self_refining_loss": [],
            "grpo_reward": [],
        }

    def run_full_recovery(
        self,
        preservation_profile: Dict[str, List[str]],
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete dual-flywheel recovery cycle.

        Parameters
        ----------
        preservation_profile : dict with keys "languages", "disciplines", "scenarios"
        progress_callback : optional callback(round_idx, stage, loss)

        Returns
        -------
        Training metrics dict.
        """
        for round_idx in range(self.config.rounds):
            t_start = time.time()

            # Stage 1: Synthetic flywheel
            synth_loss = self.synthetic_flywheel_step(
                preservation_profile, round_idx
            )
            self.metrics["synthetic_loss"].append(synth_loss)

            # Stage 2: Self-refining flywheel with GRPO
            sr_loss = self.self_refining_flywheel_step(
                preservation_profile, round_idx
            )
            self.metrics["self_refining_loss"].append(sr_loss)
            self.metrics["round"].append(round_idx)

            elapsed = time.time() - t_start
            print(
                f"  Round {round_idx + 1}/{self.config.rounds} | "
                f"synth_loss={synth_loss:.4f} | sr_loss={sr_loss:.4f} | "
                f"time={elapsed:.1f}s"
            )

            if progress_callback:
                progress_callback(round_idx, "complete", synth_loss + sr_loss)

        return dict(self.metrics)

    def synthetic_flywheel_step(
        self,
        profile: Dict[str, List[str]],
        round_idx: int,
    ) -> float:
        """
        Synthetic data flywheel: generate calibration prompts for the preservation
        profile, encode, and fine-tune the model.

        In this implementation, we use the pre-built calibration data filtered by
        the preservation profile. The model trains on next-token prediction.
        """
        prompts = self._gather_profile_prompts(profile)
        if not prompts:
            return 0.0

        # Sample subset for this round
        n_samples = min(len(prompts), self.config.samples_per_round)
        batch_prompts = prompts[:n_samples]

        # Encode
        encodings = self.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(self.config.device)

        # Training loop over batches
        dataset = TensorDataset(
            encodings["input_ids"], encodings["attention_mask"]
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        total_loss = 0.0
        n_batches = 0
        self.model.train()

        for input_ids, attention_mask in loader:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,  # standard LM objective
            )
            loss = outputs.loss
            loss = loss / self.config.grad_accumulation_steps
            loss.backward()

            if (n_batches + 1) % self.config.grad_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.config.grad_accumulation_steps
            n_batches += 1

        if n_batches % self.config.grad_accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.model.eval()
        return total_loss / max(n_batches, 1)

    def self_refining_flywheel_step(
        self,
        profile: Dict[str, List[str]],
        round_idx: int,
    ) -> float:
        """
        Self-refining flywheel: model generates responses; a critic scores them;
        high-quality traces are recycled for further training.

        When GRPO is enabled, multiple samples are generated per prompt and
        advantages are computed via within-group relative comparison.
        """
        prompts = self._gather_profile_prompts(profile)
        if not prompts:
            return 0.0

        n_samples = min(len(prompts), self.config.samples_per_round // 2)
        batch_prompts = prompts[:n_samples]

        if self.config.enable_grpo:
            return self._grpo_step(batch_prompts)
        else:
            return self._simple_refine_step(batch_prompts)

    def _grpo_step(self, prompts: List[str]) -> float:
        """
        Group Relative Policy Optimization step.

        For each prompt, generate G samples (grpo_group_size), compute rewards
        (simple length-normalized perplexity reduction), estimate advantages by
        within-group standardization, and update.
        """
        G = self.config.grpo_group_size
        total_loss = 0.0

        for prompt in prompts:
            # Generate G responses
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_length,
            ).to(self.config.device)

            with torch.no_grad():
                generated = self.model.generate(
                    **enc,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.8,
                    num_return_sequences=G,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Compute reward for each generation
            rewards = []
            for g in range(G):
                gen_ids = generated[g]
                # Reward: negative log-perplexity (higher = better)
                with torch.no_grad():
                    out = self.model(gen_ids.unsqueeze(0), labels=gen_ids.unsqueeze(0))
                    reward = -out.loss.item()
                rewards.append(reward)

            # GRPO advantage: within-group standardization
            rewards_t = torch.tensor(rewards, device=self.config.device)
            mean_r = rewards_t.mean()
            std_r = rewards_t.std() + 1e-8
            advantages = (rewards_t - mean_r) / std_r

            # Weighted training on best generation
            best_idx = advantages.argmax().item()
            best_gen = generated[best_idx]

            out = self.model(best_gen.unsqueeze(0), labels=best_gen.unsqueeze(0))
            loss = out.loss

            # Scale by advantage
            if advantages[best_idx] > 0:
                self.optimizer.zero_grad()
                (loss * advantages[best_idx]).backward()
                self.optimizer.step()

            total_loss += loss.item()

        self.metrics["grpo_reward"].append(float(rewards_t.mean()))
        return total_loss / max(len(prompts), 1)

    def _simple_refine_step(self, prompts: List[str]) -> float:
        """Basic self-refinement without GRPO (critic scoring only)."""
        total_loss = 0.0
        accepted = 0

        for prompt in prompts:
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_length,
            ).to(self.config.device)

            with torch.no_grad():
                generated = self.model.generate(
                    **enc,
                    max_new_tokens=32,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Simple critic: length and token diversity score
            gen_tokens = generated[0]
            score = self._critic_score(gen_tokens)

            if score >= self.config.critic_threshold:
                out = self.model(gen_tokens.unsqueeze(0), labels=gen_tokens.unsqueeze(0))
                loss = out.loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                accepted += 1

        return total_loss / max(accepted, 1)

    def _critic_score(self, token_ids: torch.Tensor) -> float:
        """Compute a simple quality score for generated output."""
        tokens = token_ids.tolist()
        if len(tokens) < 5:
            return 0.0
        # Diversity proxy: ratio of unique tokens
        unique_ratio = len(set(tokens)) / len(tokens)
        # Length bonus (prefer 8-128 tokens)
        length = len(tokens)
        length_score = min(length / 32.0, 1.0) if length < 32 else max(0.0, 1.0 - (length - 128) / 256)
        return 0.5 * unique_ratio + 0.5 * length_score

    def _gather_profile_prompts(self, profile: Dict[str, List[str]]) -> List[str]:
        """Collect calibration prompts that match the preservation profile."""
        prompts = []

        # Language prompts
        for lang in profile.get("languages", []):
            prompts.extend(self.calibration_data.get("lang", {}).get(lang, []))

        # Discipline prompts
        for disc in profile.get("disciplines", []):
            prompts.extend(self.calibration_data.get("disc", {}).get(disc, []))

        # Scenario prompts
        for scen in profile.get("scenarios", []):
            prompts.extend(self.calibration_data.get("scen", {}).get(scen, []))

        return prompts

    def save_checkpoint(self, path: str):
        """Save trainer state for resumption."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "metrics": self.metrics,
                "config": self.config,
            },
            path,
        )
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Resume from a saved checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.metrics = checkpoint["metrics"]
        print(f"Checkpoint loaded from {path}")
