"""Memory and speed vs bit-width.

The memory claims are computed from :mod:`src.bits.widths`; speed is a simple
wall-clock helper (the project measures quantization quality and size, not
production low-bit throughput, which needs hardware kernels).
"""

import time
from collections.abc import Callable

import pandas as pd

from src.bits.widths import packed_bytes, size_ratio


def tensor_size_bytes(n_elements: int, bits: int) -> int:
    """Packed storage of ``n_elements`` values at ``bits`` each."""
    return packed_bytes(n_elements, bits)


def size_table(n_elements: int, bit_widths: tuple[int, ...] = (32, 8, 4)) -> pd.DataFrame:
    """A table of footprint and ratio-vs-fp32 across bit-widths."""
    rows = []
    for bits in bit_widths:
        n_bytes = packed_bytes(n_elements, bits)
        rows.append(
            {
                "bits": bits,
                "bytes": n_bytes,
                "mib": n_bytes / (1024.0 * 1024.0),
                "ratio_vs_fp32": size_ratio(bits),
            }
        )
    return pd.DataFrame(rows)


def time_call(fn: Callable[[], object], repeats: int = 5) -> float:
    """Best wall-clock time (seconds) of calling ``fn`` ``repeats`` times."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best
