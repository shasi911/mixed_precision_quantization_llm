"""
Layer sensitivity analysis.

Strategy
--------
Start from a fully FP16 model.  For each decoder layer in turn:
  1. Quantize only that layer.
  2. Evaluate on a small validation slice.
  3. Record the accuracy drop vs. the FP16 baseline.
  4. Restore the layer to FP16 before moving to the next.

Layers with the largest accuracy drop are the *most sensitive* to quantization
and should be kept at high precision in the mixed-precision strategy.

Usage (standalone)
------------------
python -m src.sensitivity_analysis \
    --model NousResearch/Llama-2-7b-hf \
    --bits 4 \
    --dataset boolq \
    --max_samples 100 \
    --output results/sensitivity.json
"""

import argparse
import copy
import json
import logging
import os
import time

import torch
import torch.nn as nn

from .model_loader import load_fp16_model, load_tokenizer
from .quantization import quantize_module, get_dtype_str
from .evaluate import evaluate_boolq, evaluate_gsm8k, evaluate_piqa

logger = logging.getLogger("quant_llm.sensitivity")


# ─── Layer snapshot / restore ─────────────────────────────────────────────────

def _snapshot_state(module: nn.Module) -> dict:
    """Deep-copy the state dict of a module (kept on CPU to save GPU mem)."""
    return {k: v.cpu().clone() for k, v in module.state_dict().items()}


def _restore_state(module: nn.Module, state: dict, device) -> None:
    """Restore a module from a CPU snapshot back to *device*."""
    # Replace quantized layers with plain Linear layers first
    _replace_quantized_with_linear(module)
    module.load_state_dict({k: v.to(device) for k, v in state.items()})


def _replace_quantized_with_linear(module: nn.Module) -> None:
    """Walk module tree and replace any bitsandbytes layers with nn.Linear."""
    try:
        import bitsandbytes as bnb
        BNB_TYPES = (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit)
    except ImportError:
        return

    for name, child in list(module.named_children()):
        if isinstance(child, BNB_TYPES):
            in_f = child.in_features
            out_f = child.out_features
            has_bias = child.bias is not None
            new_linear = nn.Linear(in_f, out_f, bias=has_bias)
            setattr(module, name, new_linear)
        else:
            _replace_quantized_with_linear(child)


# ─── Single-layer quantization eval ───────────────────────────────────────────

def evaluate_with_one_layer_quantized(
    model,
    tokenizer,
    layer_idx: int,
    bits: int,
    quant_type: str,
    dataset: str,
    max_samples: int,
    device,
    dataset_cache_dir: str | None = None,
) -> float:
    """
    Quantize decoder_layer[layer_idx], evaluate, then restore it.
    Returns accuracy on the chosen dataset.
    """
    target_layer = model.model.layers[layer_idx]

    # Take a CPU snapshot before modifying
    snap = _snapshot_state(target_layer)

    try:
        # Bring layer to CPU, quantize, send back to GPU
        target_layer.cpu()
        compute_dtype = torch.float16
        quantize_module(target_layer, bits=bits, quant_type=quant_type, compute_dtype=compute_dtype)
        target_layer.to(device)

        if dataset == "gsm8k":
            result = evaluate_gsm8k(
                model,
                tokenizer,
                max_samples=max_samples,
                max_new_tokens=128,
                few_shot=2,
                dataset_cache_dir=dataset_cache_dir,
            )
        elif dataset == "boolq":
            result = evaluate_boolq(
                model, tokenizer, max_samples=max_samples, dataset_cache_dir=dataset_cache_dir
            )
        else:
            result = evaluate_piqa(
                model, tokenizer, max_samples=max_samples, dataset_cache_dir=dataset_cache_dir
            )
        acc = result["accuracy"]
    finally:
        # Always restore the layer
        target_layer.cpu()
        _restore_state(target_layer, snap, device)
        target_layer.to(device)

    return acc


# ─── Main sensitivity loop ────────────────────────────────────────────────────

def run_sensitivity_analysis(
    model_name: str,
    cache_dir: str | None,
    bits: int = 4,
    quant_type: str = "nf4",
    dataset: str = "boolq",
    max_samples: int = 100,
    output_path: str = "results/sensitivity.json",
    dataset_cache_dir: str | None = None,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(model_name, cache_dir)

    logger.info("Loading FP16 model for sensitivity analysis …")
    model = load_fp16_model(model_name, cache_dir, torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()

    num_layers = len(model.model.layers)
    logger.info("Model has %d decoder layers.", num_layers)

    # ── FP16 baseline ─────────────────────────────────────────────────────────
    logger.info("Computing FP16 baseline accuracy …")
    if dataset == "gsm8k":
        baseline_result = evaluate_gsm8k(
            model,
            tokenizer,
            max_samples=max_samples,
            max_new_tokens=128,
            few_shot=2,
            dataset_cache_dir=dataset_cache_dir,
        )
    elif dataset == "boolq":
        baseline_result = evaluate_boolq(
            model, tokenizer, max_samples=max_samples, dataset_cache_dir=dataset_cache_dir
        )
    else:
        baseline_result = evaluate_piqa(
            model, tokenizer, max_samples=max_samples, dataset_cache_dir=dataset_cache_dir
        )
    baseline_acc = baseline_result["accuracy"]
    logger.info("FP16 baseline accuracy: %.4f", baseline_acc)

    # ── Per-layer sensitivity ─────────────────────────────────────────────────
    sensitivity_scores: list[dict] = []

    for idx in range(num_layers):
        logger.info("--- Layer %d / %d ---", idx, num_layers - 1)
        t0 = time.time()
        acc = evaluate_with_one_layer_quantized(
            model, tokenizer,
            layer_idx=idx,
            bits=bits,
            quant_type=quant_type,
            dataset=dataset,
            max_samples=max_samples,
            device=device,
            dataset_cache_dir=dataset_cache_dir,
        )
        drop = baseline_acc - acc
        elapsed = time.time() - t0
        logger.info(
            "Layer %d | acc=%.4f | drop=%.4f | %.1fs",
            idx, acc, drop, elapsed,
        )
        sensitivity_scores.append({
            "layer": idx,
            "accuracy": round(acc, 6),
            "drop": round(drop, 6),
        })

    # Sort by drop descending (most sensitive first)
    ranked = sorted(sensitivity_scores, key=lambda x: x["drop"], reverse=True)

    output = {
        "model": model_name,
        "bits": bits,
        "quant_type": quant_type,
        "dataset": dataset,
        "max_samples": max_samples,
        "baseline_accuracy": round(baseline_acc, 6),
        "sensitivity_scores": sensitivity_scores,
        "ranked_by_sensitivity": ranked,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Sensitivity results saved to %s", output_path)

    return output


# ─── CLI entry point ──────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Layer sensitivity analysis")
    p.add_argument("--model", default="NousResearch/Llama-2-7b-hf")
    p.add_argument("--cache_dir", default="./model_cache")
    p.add_argument("--bits", type=int, choices=[4, 8], default=4)
    p.add_argument("--quant_type", default="nf4", choices=["nf4", "fp4"])
    p.add_argument("--dataset", default="boolq", choices=["gsm8k", "boolq", "piqa"])
    p.add_argument("--max_samples", type=int, default=100)
    p.add_argument("--output", default="results/sensitivity.json")
    p.add_argument("--dataset_cache_dir", default=None)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )
    args = _parse_args()
    run_sensitivity_analysis(
        model_name=args.model,
        cache_dir=args.cache_dir,
        bits=args.bits,
        quant_type=args.quant_type,
        dataset=args.dataset,
        max_samples=args.max_samples,
        output_path=args.output,
        dataset_cache_dir=args.dataset_cache_dir,
    )
