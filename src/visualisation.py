"""Shared plotting helpers for the notebooks and the Streamlit app."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from src.quant.dequant import QuantizedArray, dequantize


def _ensure_ax(ax: Axes | None) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    return ax


def plot_weight_distribution(x: np.ndarray, bins: int = 120, ax: Axes | None = None) -> Axes:
    """Histogram of weight values: smooth and bounded weights quantize well."""
    ax = _ensure_ax(ax)
    ax.hist(np.asarray(x, dtype=np.float32).ravel(), bins=bins, color="#4C78A8")
    ax.set_xlabel("weight value")
    ax.set_ylabel("count")
    ax.set_title("Weight distribution")
    return ax


def plot_quantization_buckets(x: np.ndarray, qa: QuantizedArray, ax: Axes | None = None) -> Axes:
    """Original values vs their dequantized reconstruction (the rounding, made visible)."""
    ax = _ensure_ax(ax)
    original = np.asarray(x, dtype=np.float32).ravel()
    reconstructed = dequantize(qa).ravel()
    ax.scatter(original, reconstructed, s=4, alpha=0.3, color="#4C78A8")
    lo = min(float(np.min(original)), float(np.min(reconstructed)))
    hi = max(float(np.max(original)), float(np.max(reconstructed)))
    ax.plot([lo, hi], [lo, hi], color="#E45756", linewidth=1, label="identity")
    ax.set_xlabel("original weight")
    ax.set_ylabel("dequantized weight")
    ax.set_title(f"int{qa.bits} round-trip")
    ax.legend()
    return ax


def plot_error_vs_bits(df: pd.DataFrame, metric: str = "rel_fro", ax: Axes | None = None) -> Axes:
    """Line of a reconstruction-error metric against bit-width, one line per config."""
    ax = _ensure_ax(ax)
    group_cols = [c for c in ("scheme", "granularity") if c in df.columns]
    if group_cols:
        for keys, sub in df.groupby(group_cols):
            label = "/".join(str(k) for k in np.atleast_1d(keys))
            ordered = sub.sort_values("bits")
            ax.plot(ordered["bits"], ordered[metric], marker="o", label=label)
        ax.legend()
    else:
        ordered = df.sort_values("bits")
        ax.plot(ordered["bits"], ordered[metric], marker="o")
    ax.set_xlabel("bits")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs bit-width")
    return ax
