"""Reconstruction error: the clean, deterministic signal that carries the claims.

Downstream quality only demonstrates; these round-trip errors are what the tests
assert, because a small reconstruction error is the precondition for behavior
being preserved (though not a guarantee of it).
"""

import numpy as np

from src.quant.dequant import dequantize
from src.quant.granularity import Granularity, Scheme, quantize


def reconstruction_error(x: np.ndarray, x_hat: np.ndarray) -> dict[str, float]:
    """Error between an array and its dequantized reconstruction."""
    x = np.asarray(x, dtype=np.float32)
    x_hat = np.asarray(x_hat, dtype=np.float32)
    diff = x - x_hat

    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    x_norm = float(np.linalg.norm(x.ravel()))
    rel_fro = float(np.linalg.norm(diff.ravel()) / x_norm) if x_norm > 0 else 0.0
    signal = float(np.mean(x**2))
    snr_db = float(10.0 * np.log10(signal / mse)) if mse > 0 else float("inf")

    return {
        "mse": mse,
        "mae": mae,
        "max_abs": max_abs,
        "rel_fro": rel_fro,
        "snr_db": snr_db,
    }


def quantization_error(
    x: np.ndarray,
    bits: int = 8,
    scheme: Scheme = "absmax",
    granularity: Granularity = "per_tensor",
    axis: int = 1,
) -> dict[str, float]:
    """Quantize ``x``, dequantize, and return the reconstruction error."""
    qa = quantize(x, bits=bits, scheme=scheme, granularity=granularity, axis=axis)
    return reconstruction_error(x, dequantize(qa))
