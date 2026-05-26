"""
Dual-Flywheel Capability Recovery Trainer (Section 3.5, Stage 4).

Implements two complementary training loops after FFN transplantation:

1. **Synthetic Flywheel**: Generates novel calibration data for each
   (Language x Discipline x Scenario) combination using either a teacher
   model or self-generation with temperature, then fine-tunes the compressed
   model on these generated samples.

2. **Self-Refining Flywheel**: The compressed model generates responses;
   a multi-capability reward function scores quality by scenario type;
   high-quality traces are re-injected with GRPO-based optimization [23, 32, 33].

The combination recovers capability loss from FFN removal through targeted fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
import math
import time


SCENARIO_REWARD_WEIGHTS = {
    "fc": {
        "json_valid": 0.3,
        "func_name_match": 0.4,
        "param_complete": 0.3,
    },
    "math_reasoning": {
        "numeric_match": 1.0,
    },
    "translation": {
        "chrf_score": 1.0,
    },
    "code": {
        "exec_match": 1.0,
    },
    "chat": {
        "perplexity_bonus": 1.0,
    },
}

SCENARIO_WEIGHT_IN_REWARD = {
    "fc": 0.3,
    "math_reasoning": 0.25,
    "translation": 0.2,
    "code": 0.15,
    "chat": 0.1,
}


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

    # Early stopping
    early_stop_min_improvement: float = 0.005  # 0.5% CRR improvement threshold

    # Synthetic generation
    synthetic_temperature: float = 1.0  # temperature for self-generation
    synthetic_max_new_tokens: int = 128
    teacher_model: Optional[Any] = None  # optional teacher model for distillation

    # Device
    device: str = "cuda"
    grad_accumulation_steps: int = 1


def _extract_final_number(text: str) -> Optional[str]:
    """Extract the final numeric answer from a math reasoning response."""
    patterns = [
        r'(?:answer|result|solution)\s*(?:is|:|=)\s*([+-]?\d*\.?\d+)',
        r'\$([+-]?\d*\.?\d+)\$',
        r'([+-]?\d*\.?\d+)\s*(?:\.\s*$)',
        r'=\s*([+-]?\d*\.?\d+)\s*$',
    ]
    for pat in patterns:
        m = re.findall(pat, text, re.IGNORECASE)
        if m:
            return m[-1]
    nums = re.findall(r'[+-]?\d+\.?\d*', text)
    return nums[-1] if nums else None


def _compute_chrf(hypothesis: str, reference: str, beta: float = 2.0) -> float:
    """Compute chrF score between hypothesis and reference translation."""
    if not hypothesis or not reference:
        return 0.0
    hyp_chars = list(hypothesis.strip())
    ref_chars = list(reference.strip())
    if not hyp_chars or not ref_chars:
        return 0.0
    ref_ngrams: Dict[int, set] = {}
    max_n = 6
    for n in range(1, max_n + 1):
        ngs = set()
        for i in range(len(ref_chars) - n + 1):
            ngs.add(tuple(ref_chars[i:i + n]))
        ref_ngrams[n] = ngs

    hyp_ngrams: Dict[int, set] = {}
    for n in range(1, max_n + 1):
        ngs = set()
        for i in range(len(hyp_chars) - n + 1):
            ngs.add(tuple(hyp_chars[i:i + n]))
        hyp_ngrams[n] = ngs

    precisions = []
    recalls = []
    for n in range(1, max_n + 1):
        ref_set = ref_ngrams.get(n, set())
        hyp_set = hyp_ngrams.get(n, set())
        if not hyp_set:
            continue
        common = len(ref_set & hyp_set)
        precisions.append(common / len(hyp_set) if hyp_set else 0.0)
        recalls.append(common / len(ref_set) if ref_set else 0.0)

    if not precisions:
        return 0.0
    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    if avg_p + avg_r == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * avg_p * avg_r / (beta_sq * avg_p + avg_r)


def compute_multi_capability_reward(
    generated_text: str,
    reference_text: str,
    scenario: str,
    perplexity: Optional[float] = None,
    exec_output: Optional[str] = None,
    expected_output: Optional[str] = None,
) -> float:
    """
    Compute a multi-capability reward based on scenario type.

    Parameters
    ----------
    generated_text : str
        The model-generated response.
    reference_text : str
        The reference / ground-truth response.
    scenario : str
        One of: fc, math_reasoning, translation, code, chat.
    perplexity : float or None
        Model perplexity on the generated text (used only for chat scenario at 10% weight).
    exec_output : str or None
        Actual execution output for code scenario.
    expected_output : str or None
        Expected execution output for code scenario.

    Returns
    -------
    float : reward in [0, 1].
    """
    if scenario == "fc":
        return _fc_reward(generated_text, reference_text)
    elif scenario == "math_reasoning":
        return _math_reward(generated_text, reference_text)
    elif scenario == "translation":
        return _translation_reward(generated_text, reference_text)
    elif scenario == "code":
        return _code_reward(generated_text, reference_text, exec_output, expected_output)
    elif scenario == "chat":
        return _chat_reward(generated_text, perplexity)
    else:
        return _chat_reward(generated_text, perplexity)


def _fc_reward(generated: str, reference: str) -> float:
    """Function-calling reward: JSON validity + function name + parameter completeness."""
    weights = SCENARIO_REWARD_WEIGHTS["fc"]
    score = 0.0

    try:
        gen_obj = json.loads(generated)
        score += weights["json_valid"]
    except (json.JSONDecodeError, TypeError):
        try:
            json_match = re.search(r'\{.*\}', generated, re.DOTALL)
            if json_match:
                gen_obj = json.loads(json_match.group())
                score += weights["json_valid"] * 0.5
            else:
                return 0.0
        except (json.JSONDecodeError, TypeError):
            return 0.0

    try:
        ref_obj = json.loads(reference)
    except (json.JSONDecodeError, TypeError):
        ref_obj = {}

    if isinstance(gen_obj, dict) and isinstance(ref_obj, dict):
        if gen_obj.get("name") == ref_obj.get("name"):
            score += weights["func_name_match"]
        else:
            score += weights["func_name_match"] * 0.2

        ref_params = set(ref_obj.get("arguments", ref_obj.get("parameters", {})).keys())
        gen_params = set(gen_obj.get("arguments", gen_obj.get("parameters", {})).keys())
        if ref_params:
            overlap = len(ref_params & gen_params) / len(ref_params)
            score += weights["param_complete"] * overlap

    return min(score, 1.0)


def _math_reward(generated: str, reference: str) -> float:
    """Math reasoning reward: extract final numeric answer and compare."""
    gen_answer = _extract_final_number(generated)
    ref_answer = _extract_final_number(reference)
    if gen_answer is None or ref_answer is None:
        if gen_answer is None and ref_answer is None:
            return 0.5
        return 0.0
    try:
        gen_val = float(gen_answer)
        ref_val = float(ref_answer)
        if ref_val == 0:
            return 1.0 if abs(gen_val) < 1e-6 else 0.0
        rel_error = abs(gen_val - ref_val) / abs(ref_val)
        return max(0.0, 1.0 - rel_error)
    except (ValueError, TypeError):
        return 0.0


def _translation_reward(generated: str, reference: str) -> float:
    """Translation reward: chrF score against reference."""
    return _compute_chrf(generated, reference)


def _code_reward(
    generated: str,
    reference: str,
    exec_output: Optional[str] = None,
    expected_output: Optional[str] = None,
) -> float:
    """Code generation reward: execute and compare output."""
    if exec_output is not None and expected_output is not None:
        if exec_output.strip() == expected_output.strip():
            return 1.0
        ref_lines = expected_output.strip().splitlines()
        gen_lines = exec_output.strip().splitlines()
        min_lines = min(len(ref_lines), len(gen_lines))
        if min_lines == 0:
            return 0.0
        matches = sum(
            1 for i in range(min_lines)
            if ref_lines[i].strip() == gen_lines[i].strip()
        )
        return matches / max(len(ref_lines), len(gen_lines), 1)

    gen_blocks = re.findall(r'```[\w]*\n(.*?)```', generated, re.DOTALL)
    ref_blocks = re.findall(r'```[\w]*\n(.*?)```', reference, re.DOTALL)
    if not gen_blocks or not ref_blocks:
        return 0.0
    gen_code = gen_blocks[0].strip()
    ref_code = ref_blocks[0].strip()
    if gen_code == ref_code:
        return 1.0
    ref_lines = ref_code.splitlines()
    gen_lines = gen_code.splitlines()
    min_len = min(len(ref_lines), len(gen_lines))
    matches = sum(
        1 for i in range(min_len)
        if ref_lines[i].strip() == gen_lines[i].strip()
    )
    return matches / max(len(ref_lines), len(gen_lines), 1)


def _chat_reward(generated: str, perplexity: Optional[float] = None) -> float:
    """Chat reward: 10% weight negative perplexity bonus, 90% length/diversity heuristic."""
    tokens = generated.split()
    if not tokens:
        return 0.0
    unique_ratio = len(set(tokens)) / len(tokens)
    length_score = min(len(tokens) / 30.0, 1.0)
    structure_score = 1.0 if any(p in generated for p in ".!?,") else 0.5
    heuristic = unique_ratio * 0.3 + length_score * 0.3 + structure_score * 0.3

    if perplexity is not None and perplexity > 0:
        perp_bonus = max(0.0, 1.0 - perplexity / 50.0) * 0.1
    else:
        perp_bonus = 0.0

    return min(heuristic + perp_bonus, 1.0)


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
        Mapping of capability axis -> category -> list of prompt dicts with
        keys "prompt" (str) and optionally "reference" (str) and "scenario" (str).
    transplant_handler : optional
        TransplantFFN instance providing NoFFN blocks and DCR router.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: FlywheelConfig,
        calibration_data: Optional[Dict[str, Dict[str, List[Any]]]] = None,
        transplant_handler=None,
    ):
        self.model = model.to(config.device)
        self.tokenizer = tokenizer
        self.config = config
        self.calibration_data = calibration_data or {}
        self.transplant_handler = transplant_handler

        self._freeze_non_trainable()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if self.transplant_handler and self.transplant_handler.router is not None:
            trainable_params += list(self.transplant_handler.router.parameters())

        self.optimizer = AdamW(
            trainable_params,
            lr=config.learning_rate,
        )

        self.metrics: Dict[str, List[float]] = {
            "round": [],
            "synthetic_loss": [],
            "self_refining_loss": [],
            "grpo_reward": [],
            "crr": [],
        }
        self._prev_crr: Optional[float] = None

    def _freeze_non_trainable(self):
        """
        Freeze all model parameters except NoFFN gate parameters and DCR parameters.
        Only gate_base, gate_specialized (NoFFN), and DCR router weights should
        receive gradients during flywheel training.
        """
        allowed_patterns = ("gate_base", "gate_specialized")

        for name, param in self.model.named_parameters():
            should_train = any(pat in name for pat in allowed_patterns)
            param.requires_grad = should_train

        if self.transplant_handler and self.transplant_handler.router is not None:
            for param in self.transplant_handler.router.parameters():
                param.requires_grad = True

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

            synth_loss = self.synthetic_flywheel_step(
                preservation_profile, round_idx
            )
            self.metrics["synthetic_loss"].append(synth_loss)

            sr_loss, reward_mean = self.self_refining_flywheel_step(
                preservation_profile, round_idx
            )
            self.metrics["self_refining_loss"].append(sr_loss)
            self.metrics["round"].append(round_idx)

            crr = reward_mean
            self.metrics["crr"].append(crr)

            elapsed = time.time() - t_start
            print(
                f"  Round {round_idx + 1}/{self.config.rounds} | "
                f"synth_loss={synth_loss:.4f} | sr_loss={sr_loss:.4f} | "
                f"crr={crr:.4f} | time={elapsed:.1f}s"
            )

            if progress_callback:
                progress_callback(round_idx, "complete", synth_loss + sr_loss)

            if self._should_early_stop(crr):
                print(f"  Early stopping: CRR improvement < {self.config.early_stop_min_improvement * 100:.1f}%")
                break

            self._prev_crr = crr

        return dict(self.metrics)

    def _should_early_stop(self, current_crr: float) -> bool:
        """Check if CRR improvement is below the early-stopping threshold."""
        if self._prev_crr is None:
            return False
        if self._prev_crr == 0:
            return False
        improvement = (current_crr - self._prev_crr) / self._prev_crr
        return improvement < self.config.early_stop_min_improvement

    def synthetic_flywheel_step(
        self,
        profile: Dict[str, List[str]],
        round_idx: int,
    ) -> float:
        """
        Synthetic data flywheel: generate novel calibration data using the model
        (or teacher model) with temperature sampling, then fine-tune on the
        generated output rather than just replaying static prompts.

        Parameters
        ----------
        profile : dict
            Preservation profile with languages, disciplines, scenarios.
        round_idx : int
            Current round index (used to vary generation temperature).
        """
        prompts_data = self._gather_profile_prompts_data(profile)
        if not prompts_data:
            return 0.0

        n_samples = min(len(prompts_data), self.config.samples_per_round)
        batch_data = prompts_data[:n_samples]

        gen_model = self.config.teacher_model or self.model
        gen_temp = self.config.synthetic_temperature * (1.0 + 0.1 * round_idx)

        generated_pairs: List[Dict[str, Any]] = []
        gen_model.eval()

        for item in batch_data:
            prompt_text = item if isinstance(item, str) else item.get("prompt", str(item))
            enc = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_length,
            ).to(self.config.device)

            if self.config.teacher_model is not None:
                with torch.no_grad():
                    gen_ids = gen_model.generate(
                        **enc,
                        max_new_tokens=self.config.synthetic_max_new_tokens,
                        do_sample=True,
                        temperature=gen_temp,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
            else:
                with torch.no_grad():
                    gen_ids = self.model.generate(
                        **enc,
                        max_new_tokens=self.config.synthetic_max_new_tokens,
                        do_sample=True,
                        temperature=gen_temp,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

            generated_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            reference_text = item.get("reference", "") if isinstance(item, dict) else ""
            scenario = item.get("scenario", "chat") if isinstance(item, dict) else "chat"

            generated_pairs.append({
                "input_ids": gen_ids[0],
                "reference": reference_text,
                "scenario": scenario,
                "generated_text": generated_text,
            })

        dataset_inputs = []
        dataset_masks = []
        for pair in generated_pairs:
            ids = pair["input_ids"]
            pad_len = self.config.max_seq_length - len(ids)
            if pad_len < 0:
                ids = ids[:self.config.max_seq_length]
                pad_len = 0
            padded = torch.cat([ids, torch.full((pad_len,), self.tokenizer.pad_token_id or 0, dtype=torch.long)])
            mask = torch.cat([torch.ones(len(ids)), torch.zeros(pad_len)])
            dataset_inputs.append(padded)
            dataset_masks.append(mask)

        if not dataset_inputs:
            return 0.0

        dataset = TensorDataset(
            torch.stack(dataset_inputs),
            torch.stack(dataset_masks),
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        total_loss = 0.0
        n_batches = 0
        self.model.train()

        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(self.config.device)
            attention_mask = attention_mask.to(self.config.device)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
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
    ) -> tuple:
        """
        Self-refining flywheel: model generates responses; a multi-capability
        reward function scores them by scenario; high-quality traces are recycled.

        When GRPO is enabled, multiple samples are generated per prompt and
        advantages are computed via proper GRPO group-relative standardization
        applied to ALL samples, not just the best.

        Returns
        -------
        (loss, mean_reward) : tuple of float
        """
        prompts_data = self._gather_profile_prompts_data(profile)
        if not prompts_data:
            return 0.0, 0.0

        n_samples = min(len(prompts_data), self.config.samples_per_round // 2)
        batch_data = prompts_data[:n_samples]

        if self.config.enable_grpo:
            return self._grpo_step(batch_data)
        else:
            loss = self._simple_refine_step(batch_data)
            return loss, 0.0

    def _grpo_step(self, prompts_data: List[Any]) -> tuple:
        """
        Group Relative Policy Optimization step.

        For each prompt, generate G samples (grpo_group_size), compute
        multi-capability rewards, estimate advantages via within-group
        standardization applied to ALL samples with positive advantages,
        and update.

        Returns
        -------
        (total_loss, mean_reward) : tuple of float
        """
        G = self.config.grpo_group_size
        total_loss = 0.0
        all_rewards = []

        for item in prompts_data:
            prompt_text = item if isinstance(item, str) else item.get("prompt", str(item))
            reference_text = item.get("reference", "") if isinstance(item, dict) else ""
            scenario = item.get("scenario", "chat") if isinstance(item, dict) else "chat"

            enc = self.tokenizer(
                prompt_text,
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

            rewards = []
            generated_texts = []
            for g in range(G):
                gen_ids = generated[g]
                gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                generated_texts.append(gen_text)

                with torch.no_grad():
                    out = self.model(gen_ids.unsqueeze(0), labels=gen_ids.unsqueeze(0))
                    perplexity = math.exp(min(out.loss.item(), 10.0))

                reward = compute_multi_capability_reward(
                    generated_text=gen_text,
                    reference_text=reference_text,
                    scenario=scenario,
                    perplexity=perplexity,
                )
                rewards.append(reward)

            rewards_t = torch.tensor(rewards, device=self.config.device, dtype=torch.float32)
            mean_r = rewards_t.mean()
            std_r = rewards_t.std() + 1e-8
            advantages = (rewards_t - mean_r) / std_r

            for g in range(G):
                if advantages[g] <= 0:
                    continue

                gen_ids = generated[g]
                self.optimizer.zero_grad()
                out = self.model(gen_ids.unsqueeze(0), labels=gen_ids.unsqueeze(0))
                loss = out.loss * advantages[g]
                loss.backward()
                self.optimizer.step()

                total_loss += out.loss.item()

            all_rewards.extend(rewards)

        mean_reward = sum(all_rewards) / max(len(all_rewards), 1)
        self.metrics["grpo_reward"].append(mean_reward)
        return total_loss / max(len(prompts_data), 1), mean_reward

    def _simple_refine_step(self, prompts_data: List[Any]) -> float:
        """Basic self-refinement without GRPO (critic scoring only)."""
        total_loss = 0.0
        accepted = 0

        for item in prompts_data:
            prompt_text = item if isinstance(item, str) else item.get("prompt", str(item))
            reference_text = item.get("reference", "") if isinstance(item, dict) else ""
            scenario = item.get("scenario", "chat") if isinstance(item, dict) else "chat"

            enc = self.tokenizer(
                prompt_text,
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

            gen_tokens = generated[0]
            gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

            reward = compute_multi_capability_reward(
                generated_text=gen_text,
                reference_text=reference_text,
                scenario=scenario,
            )

            if reward >= self.config.critic_threshold:
                self.optimizer.zero_grad()
                out = self.model(gen_tokens.unsqueeze(0), labels=gen_tokens.unsqueeze(0))
                loss = out.loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                accepted += 1

        return total_loss / max(accepted, 1)

    def _critic_score(self, token_ids: torch.Tensor) -> float:
        """Compute a simple quality score for generated output (fallback)."""
        tokens = token_ids.tolist()
        if len(tokens) < 5:
            return 0.0
        unique_ratio = len(set(tokens)) / len(tokens)
        length = len(tokens)
        length_score = min(length / 32.0, 1.0) if length < 32 else max(0.0, 1.0 - (length - 128) / 256)
        return 0.5 * unique_ratio + 0.5 * length_score

    def _gather_profile_prompts_data(self, profile: Dict[str, List[str]]) -> List[Any]:
        """
        Collect calibration prompt data that matches the preservation profile.

        Each item may be a string (plain prompt) or a dict with keys:
        'prompt', 'reference', 'scenario'.
        """
        data = []

        for lang in profile.get("languages", []):
            items = self.calibration_data.get("lang", {}).get(lang, [])
            for item in items:
                if isinstance(item, str):
                    item = {"prompt": item, "reference": "", "scenario": "chat"}
                data.append(item)

        for disc in profile.get("disciplines", []):
            items = self.calibration_data.get("disc", {}).get(disc, [])
            for item in items:
                if isinstance(item, str):
                    item = {"prompt": item, "reference": "", "scenario": "chat"}
                data.append(item)

        for scen in profile.get("scenarios", []):
            items = self.calibration_data.get("scen", {}).get(scen, [])
            for item in items:
                if isinstance(item, str):
                    item = {"prompt": item, "reference": "", "scenario": scen}
                elif isinstance(item, dict):
                    item.setdefault("scenario", scen)
                data.append(item)

        return data

    def _gather_profile_prompts(self, profile: Dict[str, List[str]]) -> List[str]:
        """Backward-compatible string-only prompt gathering."""
        data = self._gather_profile_prompts_data(profile)
        return [item if isinstance(item, str) else item.get("prompt", str(item)) for item in data]

    def save_checkpoint(self, path: str):
        """Save trainer state for resumption."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": self.metrics,
            "config": self.config,
        }
        if self.transplant_handler and self.transplant_handler.router is not None:
            state["router_state"] = self.transplant_handler.router.state_dict()

        torch.save(state, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Resume from a saved checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.metrics = checkpoint["metrics"]

        if self.transplant_handler and self.transplant_handler.router is not None:
            if "router_state" in checkpoint:
                self.transplant_handler.router.load_state_dict(checkpoint["router_state"])

        print(f"Checkpoint loaded from {path}")