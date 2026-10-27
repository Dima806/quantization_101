"""Cross-check the from-scratch zero-point int8 path against torch's own quantizer.

Agreement to within one quantization level confirms the hand-written numpy
arithmetic matches a battle-tested implementation. ``onnxruntime`` is an optional
alternative reference and is only imported when installed.
"""

import numpy as np

from src.quant.dequant import dequantize
from src.quant.zeropoint import quantize_zeropoint


def torch_affine_int8(x: np.ndarray) -> np.ndarray:
    """Dequantized values from torch's per-tensor affine (asymmetric) uint8 quantizer."""
    import torch

    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    t = torch.from_numpy(arr)
    xmin = float(t.min().item())
    xmax = float(t.max().item())
    qmin, qmax = 0, 255
    span = xmax - xmin
    scale = span / (qmax - qmin) if span != 0.0 else 1.0
    zero_point = int(round(qmin - xmin / scale))
    zero_point = max(qmin, min(qmax, zero_point))
    qt = torch.quantize_per_tensor(t, scale=scale, zero_point=zero_point, dtype=torch.quint8)
    return qt.dequantize().numpy().astype(np.float32)


def compare_to_torch(x: np.ndarray) -> dict[str, float]:
    """Max abs difference between our zero-point int8 path and torch's, and one quant step."""
    arr = np.asarray(x, dtype=np.float32)
    ours = dequantize(quantize_zeropoint(arr, bits=8, axis=None))
    ref = torch_affine_int8(arr)
    step = float((np.max(arr) - np.min(arr)) / 255.0)
    return {
        "max_abs_diff": float(np.max(np.abs(ours - ref))),
        "quant_step": step,
    }
