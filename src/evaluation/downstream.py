"""Downstream quality: does the model still behave after its weights are quantized?

Uses fake quantization (quantize then dequantize the weights back to float) so a
plain CPU forward pass measures the behavioral effect without any low-bit kernel.
"""

import copy

import numpy as np
import torch
from torch import nn

from src.quant.dequant import dequantize
from src.quant.granularity import Granularity, Scheme, quantize


def fake_quantize_linear_weights(
    model: nn.Module,
    bits: int = 8,
    scheme: Scheme = "absmax",
    granularity: Granularity = "per_channel",
    axis: int = 1,
) -> nn.Module:
    """Return a copy of ``model`` with every ``nn.Linear`` weight round-tripped through quantization."""
    q_model = copy.deepcopy(model)
    with torch.no_grad():
        for module in q_model.modules():
            if isinstance(module, nn.Linear):
                w = module.weight.detach().cpu().numpy().astype(np.float32)
                qa = quantize(w, bits=bits, scheme=scheme, granularity=granularity, axis=axis)
                module.weight.copy_(torch.from_numpy(dequantize(qa)))
    return q_model


def fake_quantize_2d_weights(
    model: nn.Module,
    bits: int = 8,
    scheme: Scheme = "absmax",
    granularity: Granularity = "per_channel",
    axis: int = 1,
    skip: tuple[str, ...] = (),
) -> nn.Module:
    """Round-trip every 2-D weight parameter through quantization.

    Works for any module whose weights are 2-D (``nn.Linear``, GPT-2's
    ``Conv1D``, embeddings), so it quantizes a whole small transformer. Names
    containing any substring in ``skip`` are left at full precision.
    """
    q_model = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in q_model.named_parameters():
            if param.ndim != 2 or any(token in name for token in skip):
                continue
            w = param.detach().cpu().numpy().astype(np.float32)
            qa = quantize(w, bits=bits, scheme=scheme, granularity=granularity, axis=axis)
            param.copy_(torch.from_numpy(dequantize(qa)))
    return q_model


def output_drift(
    reference: nn.Module, candidate: nn.Module, inputs: torch.Tensor
) -> dict[str, float]:
    """How far a candidate model's outputs move from the reference, on the same inputs."""
    reference.eval()
    candidate.eval()
    with torch.no_grad():
        a = reference(inputs).flatten()
        b = candidate(inputs).flatten()
    mse = float(torch.mean((a - b) ** 2).item())
    cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
    return {"mse": mse, "cosine": cosine}
