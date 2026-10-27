"""Arena: compare bit-widths, schemes, and granularity on one matrix at once."""

from dataclasses import asdict, dataclass

import pandas as pd

from src.bits.widths import size_ratio
from src.evaluation.reconstruction import quantization_error
from src.quant.granularity import Granularity, Scheme


@dataclass(frozen=True)
class QuantConfig:
    """One point in the quantization design space."""

    bits: int
    scheme: Scheme = "absmax"
    granularity: Granularity = "per_tensor"


def default_configs(bit_widths: tuple[int, ...] = (8, 4)) -> list[QuantConfig]:
    """Every combination of the given bit-widths with both schemes and granularities."""
    schemes: tuple[Scheme, ...] = ("absmax", "zeropoint")
    granularities: tuple[Granularity, ...] = ("per_tensor", "per_channel")
    return [
        QuantConfig(bits=bits, scheme=scheme, granularity=granularity)
        for bits in bit_widths
        for scheme in schemes
        for granularity in granularities
    ]


def arena(x, configs: list[QuantConfig] | None = None) -> pd.DataFrame:
    """Reconstruction error and size ratio for each config, as a tidy DataFrame."""
    configs = configs if configs is not None else default_configs()
    rows = []
    for cfg in configs:
        err = quantization_error(x, bits=cfg.bits, scheme=cfg.scheme, granularity=cfg.granularity)
        rows.append({**asdict(cfg), **err, "size_ratio": size_ratio(cfg.bits)})
    return pd.DataFrame(rows)
