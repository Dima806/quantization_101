"""Integer bit-widths and the size math.

The size claims in this project are *computed*, not asserted by hand: an int8
value is 1 byte against fp32's 4, i.e. one quarter; int4 is half a byte, one
eighth. :func:`size_ratio` and :func:`packed_bytes` are the source of truth.
"""

from dataclasses import dataclass

import numpy as np

FP32_BYTES = 4
FP32_BITS = 32


@dataclass(frozen=True)
class BitWidth:
    """A signed or unsigned integer grid of ``bits`` bits.

    Symmetric (absmax) quantization uses the signed grid; asymmetric
    (zero-point) quantization uses the unsigned grid.
    """

    bits: int
    signed: bool = True

    @property
    def num_levels(self) -> int:
        """Number of distinct integer codes, e.g. 256 for int8, 16 for int4."""
        return 1 << self.bits

    @property
    def qmin(self) -> int:
        """Smallest code. Signed grids are kept symmetric (e.g. -127, not -128)."""
        if self.signed:
            return -(self.num_levels // 2 - 1)
        return 0

    @property
    def qmax(self) -> int:
        """Largest code (127 for signed int8, 255 for unsigned)."""
        if self.signed:
            return self.num_levels // 2 - 1
        return self.num_levels - 1


def code_dtype(bits: int, *, signed: bool = True) -> np.dtype:
    """Smallest numpy integer dtype that holds codes of the given width.

    int4 codes are stored one-per-byte here (see :func:`pack_int4` for the
    truly packed form used by the size math).
    """
    if bits <= 8:
        return np.dtype(np.int8 if signed else np.uint8)
    if bits <= 16:
        return np.dtype(np.int16 if signed else np.uint16)
    return np.dtype(np.int32 if signed else np.uint32)


def packed_bytes(n_elements: int, bits: int) -> int:
    """Bytes needed to store ``n_elements`` codes packed at ``bits`` each."""
    return (n_elements * bits + 7) // 8


def size_ratio(bits: int, *, reference_bits: int = FP32_BITS) -> float:
    """Storage ratio of a ``bits``-wide tensor against fp32: 0.25 for int8, 0.125 for int4."""
    return bits / reference_bits


def pack_int4(codes: np.ndarray) -> np.ndarray:
    """Pack unsigned 4-bit codes (values 0..15) two-per-byte.

    Demonstrates the real 1/8 footprint: N values become ceil(N/2) bytes.
    """
    flat = np.asarray(codes, dtype=np.uint8).ravel()
    if flat.size % 2 == 1:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.uint8)])
    high = flat[0::2] << 4
    low = flat[1::2] & 0x0F
    return (high | low).astype(np.uint8)


def unpack_int4(packed: np.ndarray, n_elements: int) -> np.ndarray:
    """Inverse of :func:`pack_int4`, returning the first ``n_elements`` codes."""
    packed = np.asarray(packed, dtype=np.uint8)
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    out = np.empty(packed.size * 2, dtype=np.uint8)
    out[0::2] = high
    out[1::2] = low
    return out[:n_elements]
