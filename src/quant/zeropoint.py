"""Zero-point (asymmetric) quantization, from scratch.

Maps an arbitrary ``[min, max]`` range onto the unsigned integer grid with an
offset (the zero-point), so the buckets are used efficiently even when the
values are not centred on zero. Reconstructs as ``(code - zero_point) * scale``.
"""

import numpy as np

from src.bits.widths import BitWidth, code_dtype
from src.quant.dequant import QuantizedArray


def quantize_zeropoint(x: np.ndarray, bits: int = 8, axis: int | None = None) -> QuantizedArray:
    """Asymmetric quantization of ``x`` to ``bits`` using a min-to-max affine map."""
    x = np.asarray(x, dtype=np.float32)
    bw = BitWidth(bits, signed=False)
    qmin, qmax = bw.qmin, bw.qmax

    keepdims = axis is not None
    xmin = np.min(x, axis=axis, keepdims=keepdims)
    xmax = np.max(x, axis=axis, keepdims=keepdims)

    span = xmax - xmin
    # Guard a constant slice: a zero span would divide by zero below.
    scale = np.where(span == 0.0, 1.0, span / (qmax - qmin)).astype(np.float32)

    zero_point = np.round(qmin - xmin / scale)
    zero_point = np.clip(zero_point, qmin, qmax).astype(np.float32)

    codes = np.round(x / scale) + zero_point
    codes = np.clip(codes, qmin, qmax).astype(code_dtype(bits, signed=False))

    return QuantizedArray(
        codes=codes,
        scale=np.asarray(scale, dtype=np.float32),
        bits=bits,
        zero_point=np.asarray(zero_point, dtype=np.float32),
        axis=axis,
        symmetric=False,
    )
