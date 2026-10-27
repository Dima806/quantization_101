"""Persist notebook outputs as shareable artifacts.

Every figure is saved as a PNG under ``outputs/figures/`` and every textual or
numerical result as JSON under ``outputs/data/``.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "outputs" / "figures"
DATA_DIR = ROOT / "outputs" / "data"


def _jsonable(obj: Any) -> Any:
    """Coerce numpy scalars/arrays to plain Python for ``json.dumps``."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_figure(fig: Figure, name: str, dpi: int = 150) -> Path:
    """Save ``fig`` as ``outputs/figures/<name>.png`` and return the path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_json(data: dict[str, Any], name: str) -> Path:
    """Save ``data`` as ``outputs/data/<name>.json`` (pretty-printed) and return the path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, default=_jsonable))
    return path
