"""Absmax (symmetric) quantization, from scratch.

The simplest scheme and the clearest illustration of the idea: scale by the
maximum absolute value so the grid is centred on zero, then round. With
``axis=None`` one scale covers the whole tensor (per-tensor); with an axis set,
each slice along that axis gets its own scale (per-channel), which is what
confines an outlier to its own row.
"""

import numpy as np

from src.bits.widths import BitWidth, code_dtype
from src.quant.dequant import QuantizedArray


def quantize_absmax(x: np.ndarray, bits: int = 8, axis: int | None = None) -> QuantizedArray:
    """Symmetric quantization of ``x`` to ``bits`` via the absolute-maximum scale."""
    x = np.asarray(x, dtype=np.float32)
    bw = BitWidth(bits, signed=True)

    absmax = np.max(np.abs(x), axis=axis, keepdims=axis is not None)
    # Guard the all-zero slice: a zero scale would divide by zero below.
    scale = np.where(absmax == 0.0, 1.0, absmax / bw.qmax).astype(np.float32)

    codes = np.round(x / scale)
    codes = np.clip(codes, bw.qmin, bw.qmax).astype(code_dtype(bits, signed=True))

    return QuantizedArray(
        codes=codes,
        scale=np.asarray(scale, dtype=np.float32),
        bits=bits,
        zero_point=None,
        axis=axis,
        symmetric=True,
    )
