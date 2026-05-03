"""
Quantization utilities.

Selectively replaces nn.Linear modules with bitsandbytes INT8 or INT4
equivalents.  Quantization of weights is triggered when the module is
moved to CUDA (.cuda() or .to("cuda")).
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger("quant_llm.quantization")

try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    logger.warning("bitsandbytes not found – quantization disabled.")


# ─── Low-level helpers ────────────────────────────────────────────────────────

def _make_linear_int8(linear: nn.Linear) -> "bnb.nn.Linear8bitLt":
    new = bnb.nn.Linear8bitLt(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        has_fp16_weights=False,
        threshold=6.0,
    )
    new.weight = bnb.nn.Int8Params(
        linear.weight.data.clone(),
        requires_grad=False,
        has_fp16_weights=False,
    )
    if linear.bias is not None:
        new.bias = nn.Parameter(linear.bias.data.clone())
    return new


def _make_linear_int4(
    linear: nn.Linear,
    quant_type: str = "nf4",
    compute_dtype: torch.dtype = torch.float16,
) -> "bnb.nn.Linear4bit":
    new = bnb.nn.Linear4bit(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        compute_dtype=compute_dtype,
        compress_statistics=True,
        quant_type=quant_type,
    )
    new.weight = bnb.nn.Params4bit(
        linear.weight.data.clone(),
        requires_grad=False,
        quant_type=quant_type,
    )
    if linear.bias is not None:
        new.bias = nn.Parameter(linear.bias.data.clone())
    return new


# ─── Public API ───────────────────────────────────────────────────────────────

def quantize_module(
    module: nn.Module,
    bits: int = 4,
    quant_type: str = "nf4",
    compute_dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """
    Recursively replace every nn.Linear inside *module* with a quantized
    bitsandbytes equivalent.  Returns the module in-place (also returned for
    convenience).
    """
    if not BNB_AVAILABLE:
        raise RuntimeError("bitsandbytes is required for quantization.")
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if bits == 8:
                new_layer = _make_linear_int8(child)
            else:
                new_layer = _make_linear_int4(child, quant_type, compute_dtype)
            setattr(module, name, new_layer)
            logger.debug("Quantized %s (%d-bit)", name, bits)
        else:
            quantize_module(child, bits, quant_type, compute_dtype)
    return module


def quantize_sublayers(
    decoder_layer: nn.Module,
    sublayer_flags: dict,
    bits: int = 4,
    quant_type: str = "nf4",
    compute_dtype: torch.dtype = torch.float16,
) -> None:
    """
    Fine-grained quantization of specific linear projections inside a single
    decoder block.

    sublayer_flags example::

        {
          "self_attn": {"q_proj": True, "k_proj": False, ...},
          "mlp":       {"gate_proj": True, ...},
        }
    """
    SUBLAYER_PATHS = {
        "self_attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "mlp": ["gate_proj", "up_proj", "down_proj"],
    }
    for group, proj_names in SUBLAYER_PATHS.items():
        group_flags = sublayer_flags.get(group, {})
        parent = getattr(decoder_layer, group, None)
        if parent is None:
            continue
        for proj in proj_names:
            should_quant = group_flags.get(proj, False)
            if not should_quant:
                continue
            linear = getattr(parent, proj, None)
            if linear is None or not isinstance(linear, nn.Linear):
                continue
            if bits == 8:
                new_layer = _make_linear_int8(linear)
            else:
                new_layer = _make_linear_int4(linear, quant_type, compute_dtype)
            setattr(parent, proj, new_layer)
            logger.debug("Quantized %s.%s (%d-bit)", group, proj, bits)


# ─── Memory / dtype reporting ─────────────────────────────────────────────────

def get_memory_footprint(model: nn.Module) -> dict:
    """Return memory footprint broken down by dtype (in MB)."""
    breakdown: dict[str, float] = {}
    for param in model.parameters():
        key = str(param.dtype)
        breakdown[key] = breakdown.get(key, 0.0) + param.numel() * param.element_size()
    return {k: v / (1024 ** 2) for k, v in breakdown.items()}


def get_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def get_dtype_str(dtype_str: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unknown dtype: {dtype_str}. Choose from {list(mapping)}")
    return mapping[dtype_str]
