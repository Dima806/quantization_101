"""Typed configuration loaded from ``config/settings.yaml``.

Keeps seeds, bit-widths, and thresholds out of the notebooks so runs are
reproducible and the knobs live in one place.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"


class SyntheticConfig(BaseModel):
    """Shape and outlier structure of the seeded synthetic matrices."""

    rows: int = 256
    cols: int = 256
    outlier_scale: float = 10.0
    n_outliers: int = 1


class Settings(BaseSettings):
    """All tunable knobs for the project."""

    seed: int = 0
    bit_widths: list[int] = [8, 4]
    granularity: str = "per_channel"
    channel_axis: int = 1
    scheme: str = "absmax"
    clip_percentile: float = 99.9
    clip_sigma: float = 6.0
    outlier_sigma: float = 6.0
    model_name: str = "distilgpt2"
    synthetic: SyntheticConfig = SyntheticConfig()


def load_settings(path: Path | None = None) -> Settings:
    """Load settings from YAML, falling back to defaults when the file is absent."""
    target = path or CONFIG_PATH
    if target.exists():
        data = yaml.safe_load(target.read_text()) or {}
        return Settings(**data)
    return Settings()
