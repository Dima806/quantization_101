"""Outlier handling: the explicit fix for the failure this project dramatizes.

Clipping extreme values before quantizing keeps one giant weight from stretching
the range so far that the ordinary weights collapse into a couple of buckets.
Isolation goes further, pulling the outliers out to store separately.
"""

import numpy as np


def outlier_mask(x: np.ndarray, n_sigma: float = 6.0) -> np.ndarray:
    """Boolean mask of entries further than ``n_sigma`` standard deviations from the mean."""
    x = np.asarray(x, dtype=np.float32)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd == 0.0:
        return np.zeros_like(x, dtype=bool)
    return np.abs(x - mu) > n_sigma * sd


def clip_to_percentile(x: np.ndarray, percentile: float = 99.9) -> np.ndarray:
    """Clip ``x`` so that ``|x|`` does not exceed its ``percentile`` absolute value."""
    x = np.asarray(x, dtype=np.float32)
    threshold = float(np.percentile(np.abs(x), percentile))
    return np.clip(x, -threshold, threshold).astype(np.float32)


def clip_to_sigma(x: np.ndarray, n_sigma: float = 6.0) -> np.ndarray:
    """Clip ``x`` to ``mean +/- n_sigma * std``."""
    x = np.asarray(x, dtype=np.float32)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    return np.clip(x, mu - n_sigma * sd, mu + n_sigma * sd).astype(np.float32)


def isolate_outliers(
    x: np.ndarray, n_sigma: float = 6.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split ``x`` into a dense part (outliers zeroed) and the sparse outliers.

    Returns ``(dense, mask, outlier_values)``. The dense part quantizes cleanly;
    the handful of outlier values are kept at full precision and added back.
    """
    x = np.asarray(x, dtype=np.float32)
    mask = outlier_mask(x, n_sigma=n_sigma)
    dense = np.where(mask, 0.0, x).astype(np.float32)
    outlier_values = np.where(mask, x, 0.0).astype(np.float32)
    return dense, mask, outlier_values
