"""Weight sources: seeded synthetics (known truth) and a small real model."""

from src.models.small_lm import TinyMLP, make_tiny_model, random_inputs
from src.models.weights import (
    planted_outlier_matrix,
    smooth_weight_matrix,
)

__all__ = [
    "TinyMLP",
    "make_tiny_model",
    "planted_outlier_matrix",
    "random_inputs",
    "smooth_weight_matrix",
]
