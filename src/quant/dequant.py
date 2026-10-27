"""Dequantization and the container that carries a quantized tensor.

A :class:`QuantizedArray` stores only integer codes plus a (possibly
per-channel) scale and optional zero-point. Reconstructing the floats is the
round-trip whose error the project measures and asserts.
"""

from dataclasses import dataclass

import numpy as np

from src.bits.widths import packed_bytes


@dataclass
class QuantizedArray:
    """Integer codes plus the metadata needed to reconstruct the floats.

    Attributes:
        codes: integer codes, shape of the original tensor.
        scale: float32 step size; scalar (per-tensor) or shape ``(rows, 1)`` (per-channel).
        bits: bit-width of the codes.
        zero_point: float offset for asymmetric schemes; ``None`` when symmetric.
        axis: reduced axis for per-channel scales; ``None`` for per-tensor.
        symmetric: whether the scheme is symmetric (absmax) or asymmetric (zero-point).
    """

    codes: np.ndarray
    scale: np.ndarray
    bits: int
    zero_point: np.ndarray | None = None
    axis: int | None = None
    symmetric: bool = True

    def dequantize(self) -> np.ndarray:
        """Reconstruct the float32 tensor from codes, scale, and zero-point."""
        return dequantize(self)

    @property
    def nbytes(self) -> int:
        """Stored footprint: packed codes plus the scale (and zero-point) overhead."""
        total = packed_bytes(int(self.codes.size), self.bits)
        total += int(self.scale.astype(np.float32).nbytes)
        if self.zero_point is not None:
            total += int(self.zero_point.astype(np.float32).nbytes)
        return total


def dequantize(qa: QuantizedArray) -> np.ndarray:
    """Reconstruct floats as ``(codes - zero_point) * scale``."""
    codes = qa.codes.astype(np.float32)
    if qa.zero_point is not None:
        codes = codes - qa.zero_point.astype(np.float32)
    return (codes * qa.scale.astype(np.float32)).astype(np.float32)
