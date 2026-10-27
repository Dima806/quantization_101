"""A small model for end-to-end downstream quality.

:class:`TinyMLP` is a self-contained, seeded torch model so tests can measure
the downstream effect of quantization deterministically with no network. For
the notebooks, :func:`load_hf_causal_lm` swaps in a real HF model.
"""

import torch
from torch import nn


class TinyMLP(nn.Module):
    """A tiny deterministic MLP standing in for a real model in tests."""

    def __init__(
        self, in_dim: int = 64, hidden: int = 128, out_dim: int = 16, seed: int = 0
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        return self.net(x)


def make_tiny_model(seed: int = 0) -> TinyMLP:
    """Construct and eval-mode a :class:`TinyMLP`."""
    model = TinyMLP(seed=seed)
    model.eval()
    return model


def random_inputs(n: int = 32, in_dim: int = 64, seed: int = 1) -> torch.Tensor:
    """Deterministic batch of inputs for the tiny model."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, in_dim, generator=gen)


def load_hf_causal_lm(model_name: str = "distilgpt2") -> tuple[object, object]:
    """Load a small HF causal LM and its tokenizer (notebook-only; downloads)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return model, tokenizer
