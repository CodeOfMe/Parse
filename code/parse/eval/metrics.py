"""
Capability evaluation metrics for PARSE.

Metrics:
  - Perplexity (PPL): Standard language modeling quality.
  - Capability Retention Ratio (CRR): compressed / original performance per capability.
  - Cross-Capability Interference (CCI): avg degradation of non-preserved capabilities.
  - GSM8K accuracy: math word problem benchmark.
  - BFCL accuracy: Berkeley Function-Calling Leaderboard metric.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import re


def evaluate_capabilities(
    model: nn.Module,
    tokenizer,
    eval_data: Dict[str, List[str]],
    original_model: Optional[nn.Module] = None,
    device: str = "cuda",
    max_length: int = 512,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a PARSE-compressed model across capability dimensions.

    Parameters
    ----------
    model : compressed model
    tokenizer : tokenizer
    eval_data : dict mapping capability_name -> list of evaluation prompts
    original_model : optional original model for CRR computation
    device : compute device

    Returns
    -------
    Dict of {capability: {"ppl": float, "crr": float or None}}
    """
    results = {}
    model.eval()

    for capability, prompts in eval_data.items():
        if not prompts:
            results[capability] = {"ppl": None, "crr": None}
            continue

        # Compute perplexity
        ppl = _compute_perplexity(model, tokenizer, prompts, device, max_length)

        # Compute CRR if original model provided
        crr = None
        if original_model is not None:
            original_ppl = _compute_perplexity(
                original_model, tokenizer, prompts, device, max_length
            )
            if original_ppl is not None and original_ppl > 0:
                crr = original_ppl / max(ppl, 1e-8)  # higher = better retention

        results[capability] = {"ppl": ppl, "crr": crr}

    return results


def compute_CRR(
    eval_results: Dict[str, Dict[str, float]],
    preserved_capabilities: List[str],
) -> float:
    """Average Capability Retention Ratio across preserved capabilities."""
    crr_values = []
    for cap in preserved_capabilities:
        if cap in eval_results and eval_results[cap].get("crr") is not None:
            crr_values.append(eval_results[cap]["crr"])
    return sum(crr_values) / len(crr_values) if crr_values else 0.0


def compute_CCI(
    eval_results: Dict[str, Dict[str, float]],
    preserved_capabilities: List[str],
) -> float:
    """Cross-Capability Interference: average CRR degradation on non-preserved capabilities."""
    non_preserved = set(eval_results.keys()) - set(preserved_capabilities)
    if not non_preserved:
        return 0.0

    degradations = []
    for cap in non_preserved:
        if cap in eval_results and eval_results[cap].get("crr") is not None:
            crr = eval_results[cap]["crr"]
            degradations.append(1.0 - min(crr, 1.0))

    return sum(degradations) / len(degradations) if degradations else 0.0


def evaluate_gsm8k(
    model: nn.Module,
    tokenizer,
    test_samples: List[Dict[str, str]],
    device: str = "cuda",
) -> float:
    """
    Evaluate GSM8K mathematical reasoning accuracy.

    Parameters
    ----------
    model : the language model
    tokenizer
    test_samples : list of {"question": ..., "answer": ...} dicts
    device

    Returns
    -------
    Accuracy (0.0-1.0).
    """
    correct = 0
    model.eval()

    for sample in test_samples:
        question = sample["question"]
        expected_answer = _extract_number(sample.get("answer", ""))

        prompt = f"Question: {question}\nAnswer:"
        enc = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            generated = model.generate(
                **enc,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(generated[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted = _extract_number(response)

        if predicted is not None and expected_answer is not None:
            if abs(predicted - expected_answer) < 1e-6:
                correct += 1

    return correct / len(test_samples) if test_samples else 0.0


def evaluate_bfcl(
    model: nn.Module,
    tokenizer,
    test_samples: List[Dict],
    device: str = "cuda",
) -> float:
    """
    Evaluate Berkeley Function-Calling Leaderboard accuracy.

    Simplified metric: correct function name + valid JSON parameters.

    Parameters
    ----------
    model
    tokenizer
    test_samples : list of {"prompt": ..., "expected_function": ..., "expected_params": {...}}
    device

    Returns
    -------
    Accuracy (0.0-1.0).
    """
    correct = 0
    model.eval()

    for sample in test_samples:
        prompt = sample["prompt"]
        expected_func = sample.get("expected_function", "")

        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **enc,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(generated[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

        # Basic check: does response contain expected function name?
        if expected_func.lower() in response.lower():
            # Check for valid JSON structure
            if "{" in response and "}" in response:
                correct += 1

    return correct / len(test_samples) if test_samples else 0.0


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _compute_perplexity(
    model, tokenizer, prompts, device, max_length
) -> Optional[float]:
    """Compute average perplexity over a set of prompts."""
    total_loss = 0.0
    total_tokens = 0

    for prompt in prompts:
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**enc, labels=enc["input_ids"])
            loss = outputs.loss
            if loss is not None:
                total_loss += loss.item() * enc["input_ids"].numel()
                total_tokens += enc["input_ids"].numel()

    if total_tokens == 0:
        return None

    avg_loss = total_loss / total_tokens
    return torch.exp(torch.tensor(avg_loss)).item()


def _extract_number(text: str) -> Optional[float]:
    """Extract the last number from a text response (GSM8K convention)."""
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if not numbers:
        return None
    # GSM8K convention: last number is the answer
    return float(numbers[-1])
