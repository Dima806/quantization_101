"""Weight matrices to quantize.

Two synthetic generators give the controlled, seeded truth the library favors
(a smooth well-behaved matrix, and one with a planted outlier), and a loader
pulls authentic 2-D weight matrices from a small real model for the notebooks.
"""

import numpy as np


def smooth_weight_matrix(
    rows: int = 256, cols: int = 256, seed: int = 0, scale: float = 0.02
) -> np.ndarray:
    """A smooth, bounded Gaussian matrix: the kind of thing that quantizes well."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((rows, cols)) * scale).astype(np.float32)


def planted_outlier_matrix(
    rows: int = 256,
    cols: int = 256,
    outlier_scale: float = 10.0,
    n_outliers: int = 1,
    seed: int = 0,
    base_scale: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """A smooth matrix with a few planted outliers, plus a mask of their positions.

    Each outlier is ``outlier_scale`` times the largest ordinary magnitude, so
    per-tensor quantization is forced to stretch its range around it. Returns
    ``(matrix, mask)`` where ``mask`` is ``True`` at the planted entries.
    """
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((rows, cols)) * base_scale).astype(np.float32)
    ordinary_absmax = float(np.max(np.abs(w)))

    mask = np.zeros_like(w, dtype=bool)
    n = min(n_outliers, rows)
    rows_idx = rng.choice(rows, size=n, replace=False)
    cols_idx = rng.integers(0, cols, size=n)
    signs = rng.choice(np.array([-1.0, 1.0]), size=n)
    for i in range(n):
        r, c = int(rows_idx[i]), int(cols_idx[i])
        w[r, c] = np.float32(signs[i] * outlier_scale * ordinary_absmax)
        mask[r, c] = True
    return w, mask


def load_model_weights(
    model_name: str = "distilgpt2", max_matrices: int | None = None
) -> dict[str, np.ndarray]:
    """Return the 2-D weight matrices of a small HF model as float32 arrays.

    Notebook-only: this downloads the model on first use. Tests rely on the
    synthetic generators above so they never need the network.
    """
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_name)
    matrices: dict[str, np.ndarray] = {}
    for name, param in model.named_parameters():
        if param.ndim == 2:
            matrices[name] = param.detach().cpu().numpy().astype(np.float32)
            if max_matrices is not None and len(matrices) >= max_matrices:
                break
    return matrices
