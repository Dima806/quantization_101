"""An outlier collapses per-tensor quality; per-channel recovers it; clipping helps."""

import numpy as np

from src.models.weights import planted_outlier_matrix
from src.quant.dequant import dequantize
from src.quant.granularity import quantize
from src.quant.outliers import clip_to_percentile


def _ordinary_error(x: np.ndarray, mask: np.ndarray, **kwargs) -> float:
    """Mean reconstruction error on the ordinary (non-outlier) weights."""
    x_hat = dequantize(quantize(x, **kwargs))
    return float(np.mean(np.abs(x - x_hat)[~mask]))


def test_per_tensor_collapses_and_per_channel_recovers():
    x, mask = planted_outlier_matrix(128, 128, outlier_scale=20.0, n_outliers=1, seed=0)
    per_tensor = _ordinary_error(x, mask, bits=8, scheme="absmax", granularity="per_tensor")
    per_channel = _ordinary_error(
        x, mask, bits=8, scheme="absmax", granularity="per_channel", axis=1
    )
    # The outlier stretches the single per-tensor scale; ordinary weights lose resolution.
    assert per_tensor > 3 * per_channel


def test_clipping_reduces_outlier_damage():
    x, mask = planted_outlier_matrix(128, 128, outlier_scale=20.0, n_outliers=1, seed=1)
    naive = _ordinary_error(x, mask, bits=8, scheme="absmax", granularity="per_tensor")

    clipped = clip_to_percentile(x, percentile=99.9)
    x_hat = dequantize(quantize(clipped, bits=8, scheme="absmax", granularity="per_tensor"))
    clipped_err = float(np.mean(np.abs(x - x_hat)[~mask]))

    assert clipped_err < naive
