"""
Model loading with layer-aware quantization.

Flow
----
1.  Load the model in FP16 on CPU (no bitsandbytes flags at load time).
2.  Walk the YAML layer config; for every component marked quantize=true,
    replace its nn.Linear modules with bitsandbytes quantized equivalents.
3.  Move the entire model to GPU – this triggers actual weight quantization.
"""

import logging
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .quantization import (
    get_dtype_str,
    get_memory_footprint,
    get_gpu_memory_mb,
    quantize_module,
    quantize_sublayers,
)

logger = logging.getLogger("quant_llm.model_loader")


def load_tokenizer(model_name: str, cache_dir: str | None = None) -> AutoTokenizer:
    logger.info("Loading tokenizer for %s", model_name)
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_fp16_model(
    model_name: str,
    cache_dir: str | None = None,
    torch_dtype: torch.dtype = torch.float16,
) -> AutoModelForCausalLM:
    """Load model in full precision on CPU first (avoids GPU OOM during patching)."""
    logger.info("Loading %s in %s on CPU …", model_name, torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def _has_sublayer_keys(layer_cfg: dict) -> bool:
    return "self_attn" in layer_cfg or "mlp" in layer_cfg


def apply_layer_quantization(
    model: AutoModelForCausalLM,
    layer_cfg: dict,
    quant_cfg: dict,
) -> None:
    """
    Patch model in-place according to the YAML layer config.

    layer_cfg  – the ``layers`` section of the YAML
    quant_cfg  – the ``quantization`` section of the YAML
    """
    bits = int(quant_cfg.get("bits", 4))
    quant_type = quant_cfg.get("quant_type", "nf4")
    compute_dtype = get_dtype_str(quant_cfg.get("compute_dtype", "float16"))

    def _quant(module, label):
        logger.info("Quantizing %s (%d-bit %s)", label, bits, quant_type)
        quantize_module(module, bits=bits, quant_type=quant_type, compute_dtype=compute_dtype)

    # embed_tokens
    if layer_cfg.get("embed_tokens", {}).get("quantize", False):
        _quant(model.model.embed_tokens, "embed_tokens")

    # final norm
    if layer_cfg.get("norm", {}).get("quantize", False):
        _quant(model.model.norm, "norm")

    # lm_head
    if layer_cfg.get("lm_head", {}).get("quantize", False):
        _quant(model.lm_head, "lm_head")

    # decoder layers
    decoder_cfg = layer_cfg.get("decoder_layers", {})
    num_layers = len(model.model.layers)
    for idx in range(num_layers):
        dcfg = decoder_cfg.get(str(idx), {})
        should_quant = dcfg.get("quantize", False)
        if not should_quant:
            continue

        decoder_layer = model.model.layers[idx]

        if _has_sublayer_keys(dcfg):
            # Fine-grained: only specific projections
            quantize_sublayers(
                decoder_layer,
                sublayer_flags=dcfg,
                bits=bits,
                quant_type=quant_type,
                compute_dtype=compute_dtype,
            )
        else:
            # Coarse: quantize the whole decoder block
            _quant(decoder_layer, f"decoder_layer[{idx}]")


def load_model_from_config(config: dict) -> tuple:
    """
    Main entry point: load model + tokenizer, apply layer-aware quantization,
    and move to GPU.

    Returns
    -------
    (model, tokenizer)
    """
    model_cfg = config["model"]
    model_name = model_cfg["name"]
    cache_dir = model_cfg.get("cache_dir")
    torch_dtype = get_dtype_str(model_cfg.get("dtype", "float16"))

    tokenizer = load_tokenizer(model_name, cache_dir)

    t0 = time.time()
    model = load_fp16_model(model_name, cache_dir, torch_dtype)

    layer_cfg = config.get("layers", {})
    quant_cfg = config.get("quantization", {})

    num_quantized = sum(
        1 for v in config.get("layers", {}).get("decoder_layers", {}).values()
        if v.get("quantize", False)
    )
    logger.info(
        "Applying layer-aware quantization: %d / %d decoder layers will be quantized",
        num_quantized,
        len(model.model.layers),
    )

    apply_layer_quantization(model, layer_cfg, quant_cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Moving model to %s …", device)
    model = model.to(device)
    model.eval()

    elapsed = time.time() - t0
    mem = get_memory_footprint(model)
    logger.info("Model loaded in %.1fs | Memory breakdown (MB): %s", elapsed, mem)
    if torch.cuda.is_available():
        logger.info("GPU memory allocated: %.1f MB", get_gpu_memory_mb())

    return model, tokenizer


def load_uniform_quantized_model(
    model_name: str,
    cache_dir: str | None,
    bits: int = 4,
    quant_type: str = "nf4",
) -> tuple:
    """
    Load with ALL decoder layers (and lm_head) quantized uniformly.
    Used as the uniform-quantization baseline.
    """
    from transformers import BitsAndBytesConfig

    logger.info("Loading uniformly quantized model (%d-bit) …", bits)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=(bits == 4),
        load_in_8bit=(bits == 8),
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        quantization_config=bnb_cfg,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = load_tokenizer(model_name, cache_dir)
    return model, tokenizer
