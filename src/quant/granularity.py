"""Granularity: one scale for the whole tensor vs one scale per channel/row.

This is the single most important lever for quality, because it is how the
outlier problem is contained: with per-channel scales, an outlier in one row
stretches only that row's scale and leaves every other row's ordinary weights
their full resolution.
"""

from typing import Literal

import numpy as np

from src.quant.absmax import quantize_absmax
from src.quant.dequant import QuantizedArray
from src.quant.zeropoint import quantize_zeropoint

Scheme = Literal["absmax", "zeropoint"]
Granularity = Literal["per_tensor", "per_channel"]


def quantize(
    x: np.ndarray,
    bits: int = 8,
    scheme: Scheme = "absmax",
    granularity: Granularity = "per_tensor",
    axis: int = 1,
) -> QuantizedArray:
    """Unified entry point: pick a scheme and a granularity.

    ``per_tensor`` uses a single scale; ``per_channel`` reduces along ``axis``
    (default 1, i.e. one scale per output row of a weight matrix).
    """
    reduce_axis = None if granularity == "per_tensor" else axis
    if scheme == "absmax":
        return quantize_absmax(x, bits=bits, axis=reduce_axis)
    return quantize_zeropoint(x, bits=bits, axis=reduce_axis)
