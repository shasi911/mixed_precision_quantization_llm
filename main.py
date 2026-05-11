#!/usr/bin/env python3
"""
Layer-Aware Quantization for Efficient LLM Inference
=====================================================

Modes
-----
  baseline    – Full FP16 inference (no quantization)
  quantize    – Layer-aware quantization from YAML config
  uniform     – Uniform INT4/INT8 quantization (comparison baseline)
  sensitivity – Layer-by-layer sensitivity analysis
"""

import argparse
import json
import logging
import os
import time
import torch

from src.utils import load_config, setup_logging
from src.model_loader import (
    load_model_from_config,
    load_fp16_model,
    load_tokenizer,
    load_uniform_quantized_model,
)
from src.evaluate import run_all_evaluations, compute_perplexity
from src.quantization import get_memory_footprint, get_gpu_memory_mb
from src.sensitivity_analysis import run_sensitivity_analysis


def _parse_args():
    p = argparse.ArgumentParser(
        description="Layer-Aware Quantization for LLM Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        default="configs/quantization_config.yaml",
        help="Path to YAML config file",
    )
    p.add_argument(
        "--mode",
        required=True,
        choices=["baseline", "quantize", "uniform", "sensitivity"],
        help=(
            "baseline   : FP16, no quantization\n"
            "quantize   : layer-aware quant from YAML\n"
            "uniform    : all layers quantized uniformly\n"
            "sensitivity: run per-layer sensitivity sweep"
        ),
    )
    # Sensitivity-specific overrides
    p.add_argument("--sens_dataset", default="boolq", choices=["gsm8k", "boolq", "piqa"])
    p.add_argument("--sens_samples", type=int, default=100)
    p.add_argument("--sens_output", default="results/sensitivity.json")
    # Uniform baseline override
    p.add_argument("--uniform_bits", type=int, choices=[4, 8], default=4)
    return p.parse_args()


def _save_results(results: dict, tag: str, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"{tag}_{ts}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logging.getLogger("quant_llm").info("Results saved to %s", path)


def _collect_system_info(model) -> dict:
    info = {
        "memory_footprint_mb": get_memory_footprint(model),
        "gpu_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_allocated_mb"] = get_gpu_memory_mb()
        info["gpu_total_memory_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    return info


def main():
    args = _parse_args()
    config = load_config(args.config)

    out_cfg = config.get("output", {})
    logger = setup_logging(
        log_level=out_cfg.get("log_level", "INFO"),
        log_file=os.path.join("logs", "experiment.log"),
    )
    results_dir = out_cfg.get("results_dir", "./results")
    model_cfg = config["model"]
    eval_cfg = config.get("evaluation", {})
    quant_cfg = config.get("quantization", {})

    # ── Sensitivity analysis ───────────────────────────────────────────────────
    if args.mode == "sensitivity":
        run_sensitivity_analysis(
            model_name=model_cfg["name"],
            cache_dir=model_cfg.get("cache_dir"),
            bits=int(quant_cfg.get("bits", 4)),
            quant_type=quant_cfg.get("quant_type", "nf4"),
            dataset=args.sens_dataset,
            max_samples=args.sens_samples,
            output_path=args.sens_output,
            dataset_cache_dir=eval_cfg.get("dataset_cache_dir"),
        )
        return

    # ── Load model ────────────────────────────────────────────────────────────
    t_load = time.time()

    if args.mode == "baseline":
        logger.info("Mode: FP16 baseline")
        tokenizer = load_tokenizer(model_cfg["name"], model_cfg.get("cache_dir"))
        model = load_fp16_model(
            model_cfg["name"],
            model_cfg.get("cache_dir"),
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        tag = "baseline_fp16"

    elif args.mode == "quantize":
        logger.info("Mode: layer-aware quantization")
        model, tokenizer = load_model_from_config(config)
        tag = "layer_aware_quant"

    elif args.mode == "uniform":
        logger.info("Mode: uniform %d-bit quantization", args.uniform_bits)
        model, tokenizer = load_uniform_quantized_model(
            model_name=model_cfg["name"],
            cache_dir=model_cfg.get("cache_dir"),
            bits=args.uniform_bits,
            quant_type=quant_cfg.get("quant_type", "nf4"),
        )
        tag = f"uniform_int{args.uniform_bits}"

    load_time = time.time() - t_load
    logger.info("Model loaded in %.1f s", load_time)

    # ── System info ───────────────────────────────────────────────────────────
    sys_info = _collect_system_info(model)
    logger.info("System info: %s", sys_info)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    logger.info("Starting evaluation …")
    t_eval = time.time()
    eval_results = run_all_evaluations(model, tokenizer, eval_cfg)
    eval_time = time.time() - t_eval

    # ── Compile final report ──────────────────────────────────────────────────
    report = {
        "mode": args.mode,
        "model": model_cfg["name"],
        "quantization_config": quant_cfg if args.mode != "baseline" else None,
        "load_time_s": round(load_time, 2),
        "eval_time_s": round(eval_time, 2),
        "system": sys_info,
        "results": eval_results,
    }

    # Print summary table
    print("\n" + "=" * 60)
    print(f"  Mode : {args.mode}")
    print(f"  Model: {model_cfg['name']}")
    print("-" * 60)
    print(f"  {'Dataset':<12} {'Accuracy':>10} {'Avg Latency (s)':>18}")
    print("-" * 60)
    for ds, r in eval_results.items():
        print(f"  {ds:<12} {r['accuracy']:>10.4f} {r['avg_latency_s']:>18.3f}")
    print("=" * 60)
    print(f"  GPU memory used: {sys_info.get('gpu_memory_allocated_mb', 0):.1f} MB")
    print("=" * 60 + "\n")

    _save_results(report, tag, results_dir)


if __name__ == "__main__":
    main()
