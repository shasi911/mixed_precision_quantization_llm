"""
Evaluation pipeline for GSM8K, BoolQ, and PIQA.
"""

import logging
import os
import time
import re
import torch
import numpy as np
from datasets import load_dataset
from torch.nn.functional import log_softmax
from tqdm import tqdm

from .utils import (
    extract_number,
    extract_yes_no,
    format_gsm8k_prompt,
    format_boolq_prompt,
    format_piqa_prompt,
)

logger = logging.getLogger("quant_llm.evaluate")


def _load_dataset(path: str, *args, split: str, cache_dir: str | None = None):
    """Load a Hugging Face dataset using the configured reusable cache."""
    dataset_cache_dir = cache_dir or os.getenv("HF_DATASETS_CACHE")
    if dataset_cache_dir:
        logger.info("Loading dataset %s from cache_dir=%s", path, dataset_cache_dir)
    return load_dataset(path, *args, split=split, cache_dir=dataset_cache_dir)

# ─── GSM8K few-shot examples (standard 4-shot) ────────────────────────────────
GSM8K_FEW_SHOT = [
    {
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "answer": "Janet sells 16 - 3 - 4 = 9 eggs daily.\nShe makes 9 * 2 = $18 every day.\n#### 18",
    },
    {
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "answer": "It takes 2/2 = 1 bolt of white fiber.\nSo the total is 2 + 1 = 3 bolts.\n#### 3",
    },
    {
        "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
        "answer": "The repairs increased the value by 80000 * 1.5 = $120,000.\nThe house is now worth 80000 + 120000 = $200,000.\nHe spent 80000 + 50000 = $130,000.\nHis profit is 200000 - 130000 = $70,000.\n#### 70000",
    },
    {
        "question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
        "answer": "He runs 3 * 3 = 9 sprints per week.\nSo he runs 9 * 60 = 540 meters per week.\n#### 540",
    },
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

@torch.inference_mode()
def _generate(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode only the newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


@torch.inference_mode()
def _log_likelihood(model, tokenizer, text: str, device: str) -> float:
    """Compute average per-token log-likelihood for *text*."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        logits = model(**inputs).logits  # (1, seq_len, vocab_size)
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.mean().item()


# ─── Dataset evaluations ──────────────────────────────────────────────────────

def evaluate_gsm8k(
    model,
    tokenizer,
    max_samples: int | None = 200,
    max_new_tokens: int = 256,
    few_shot: int = 4,
    dataset_cache_dir: str | None = None,
) -> dict:
    logger.info("Evaluating on GSM8K (max_samples=%s) …", max_samples)
    device = next(model.parameters()).device
    dataset = _load_dataset("gsm8k", "main", split="test", cache_dir=dataset_cache_dir)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    few_shot_examples = GSM8K_FEW_SHOT[:few_shot]
    correct, total = 0, 0
    latencies = []

    for example in tqdm(dataset, desc="GSM8K"):
        prompt = format_gsm8k_prompt(example["question"], few_shot_examples)
        t0 = time.time()
        response = _generate(model, tokenizer, prompt, max_new_tokens, device)
        latencies.append(time.time() - t0)

        pred = extract_number(response)
        # Ground truth: last number after ####
        gt_match = re.search(r"####\s*([\-\d,\.]+)", example["answer"])
        gt = gt_match.group(1).replace(",", "").strip() if gt_match else None

        if pred is not None and gt is not None and pred == gt:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    logger.info("GSM8K accuracy: %.4f (%d/%d)", acc, correct, total)
    return {
        "dataset": "gsm8k",
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "avg_latency_s": float(np.mean(latencies)),
    }


def evaluate_boolq(
    model,
    tokenizer,
    max_samples: int | None = 200,
    max_new_tokens: int = 16,
    dataset_cache_dir: str | None = None,
) -> dict:
    logger.info("Evaluating on BoolQ (max_samples=%s) …", max_samples)
    device = next(model.parameters()).device
    dataset = _load_dataset("boolq", split="validation", cache_dir=dataset_cache_dir)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct, total = 0, 0
    latencies = []

    for example in tqdm(dataset, desc="BoolQ"):
        prompt = format_boolq_prompt(example["passage"], example["question"])
        t0 = time.time()
        response = _generate(model, tokenizer, prompt, max_new_tokens, device)
        latencies.append(time.time() - t0)

        pred = extract_yes_no(response)
        gt = "yes" if example["answer"] else "no"
        if pred == gt:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    logger.info("BoolQ accuracy: %.4f (%d/%d)", acc, correct, total)
    return {
        "dataset": "boolq",
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "avg_latency_s": float(np.mean(latencies)),
    }


def evaluate_piqa(
    model,
    tokenizer,
    max_samples: int | None = 200,
    dataset_cache_dir: str | None = None,
) -> dict:
    """
    Uses log-likelihood scoring: pick the solution with higher per-token
    log-probability given the goal context.
    """
    logger.info("Evaluating on PIQA (max_samples=%s) …", max_samples)
    device = next(model.parameters()).device
    dataset = _load_dataset("ybisk/piqa", split="validation", cache_dir=dataset_cache_dir)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct, total = 0, 0
    latencies = []

    for example in tqdm(dataset, desc="PIQA"):
        goal = example["goal"]
        sol1 = example["sol1"]
        sol2 = example["sol2"]
        label = example["label"]  # 0 or 1

        ctx = f"Goal: {goal}\nSolution: "
        t0 = time.time()
        ll1 = _log_likelihood(model, tokenizer, ctx + sol1, device)
        ll2 = _log_likelihood(model, tokenizer, ctx + sol2, device)
        latencies.append(time.time() - t0)

        pred = 0 if ll1 > ll2 else 1
        if pred == label:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    logger.info("PIQA accuracy: %.4f (%d/%d)", acc, correct, total)
    return {
        "dataset": "piqa",
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "avg_latency_s": float(np.mean(latencies)),
    }


# ─── Perplexity ───────────────────────────────────────────────────────────────

@torch.inference_mode()
def compute_perplexity(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 512,
) -> float:
    """Compute average perplexity over a list of texts."""
    device = next(model.parameters()).device
    total_log_prob, total_tokens = 0.0, 0

    for text in tqdm(texts, desc="Perplexity"):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(device)
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss
        n_tokens = inputs["input_ids"].shape[1]
        total_log_prob += loss.item() * n_tokens
        total_tokens += n_tokens

    ppl = float(np.exp(total_log_prob / total_tokens)) if total_tokens > 0 else float("inf")
    logger.info("Perplexity: %.2f", ppl)
    return ppl


# ─── Dispatcher ───────────────────────────────────────────────────────────────

def run_all_evaluations(model, tokenizer, eval_cfg: dict) -> dict:
    datasets = eval_cfg.get("datasets", ["gsm8k", "boolq", "piqa"])
    max_samples = eval_cfg.get("max_samples", 200)
    max_new_tokens = eval_cfg.get("max_new_tokens", 256)
    few_shot = eval_cfg.get("few_shot", 4)
    dataset_cache_dir = eval_cfg.get("dataset_cache_dir")

    results = {}
    fn_map = {
        "gsm8k": lambda: evaluate_gsm8k(
            model, tokenizer, max_samples, max_new_tokens, few_shot, dataset_cache_dir
        ),
        "boolq": lambda: evaluate_boolq(
            model, tokenizer, max_samples, dataset_cache_dir=dataset_cache_dir
        ),
        "piqa": lambda: evaluate_piqa(
            model, tokenizer, max_samples, dataset_cache_dir=dataset_cache_dir
        ),
    }
    for ds in datasets:
        if ds in fn_map:
            results[ds] = fn_map[ds]()
        else:
            logger.warning("Unknown dataset: %s (skipping)", ds)
    return results
