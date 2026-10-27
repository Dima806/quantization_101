"""Size math is computed, not asserted by hand: int8 = 1/4 fp32, int4 = 1/8."""

import numpy as np

from src.bits.widths import pack_int4, packed_bytes, size_ratio, unpack_int4


def test_int8_is_a_quarter_of_fp32():
    assert size_ratio(8) == 0.25


def test_int4_is_an_eighth_of_fp32():
    assert size_ratio(4) == 0.125


def test_packed_bytes_matches_bit_arithmetic():
    assert packed_bytes(1000, 8) == 1000
    assert packed_bytes(1000, 4) == 500
    assert packed_bytes(1001, 4) == 501  # rounds up


def test_int4_pack_roundtrip():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 16, size=1001).astype(np.uint8)
    packed = pack_int4(codes)
    assert packed.size == 501  # ceil(1001 / 2): one eighth of fp32's 4004 bytes
    restored = unpack_int4(packed, codes.size)
    assert np.array_equal(restored, codes)
