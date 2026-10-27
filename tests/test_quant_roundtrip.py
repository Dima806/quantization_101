"""dequant(quant(x)) must land within the bit-width's resolution, and match a reference."""

import numpy as np
import pytest

from src.models.weights import smooth_weight_matrix
from src.quant.dequant import dequantize
from src.quant.granularity import quantize
from src.reference.compare import compare_to_torch


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("scheme", ["absmax", "zeropoint"])
@pytest.mark.parametrize("granularity", ["per_tensor", "per_channel"])
def test_roundtrip_within_resolution(bits, scheme, granularity):
    x = smooth_weight_matrix(128, 128, seed=0)
    qa = quantize(x, bits=bits, scheme=scheme, granularity=granularity, axis=1)
    x_hat = dequantize(qa)
    # Nearest-bucket rounding bounds the error at half the (largest) step size.
    step = float(np.max(qa.scale))
    assert np.max(np.abs(x - x_hat)) <= step / 2 + 1e-4


def test_matches_torch_reference():
    x = smooth_weight_matrix(128, 128, seed=3)
    result = compare_to_torch(x)
    # Agreement to within one quantization level confirms the from-scratch path.
    assert result["max_abs_diff"] <= 1.5 * result["quant_step"]
