"""Small filesystem helpers shared by the training and prediction scripts.

Centralizing "make a timestamped run folder and drop a copy of the config
in it" here means both ``scripts/train.py`` and ``scripts/predict.py`` (per
project requirement 6) get the same, consistent layout.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def make_timestamped_dir(base_dir: str | Path, prefix: str = "run") -> Path:
    """Create and return ``base_dir/<prefix>_<YYYYmmdd_HHMMSS>/``.

    A numeric suffix is appended if the exact timestamp already exists
    (e.g. two runs started within the same second).
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base / f"{prefix}_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = base / f"{prefix}_{stamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def save_json(data: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2, default=_json_default))


def _json_default(obj: Any) -> Any:
    # numpy scalars / dates and similar show up in metrics dicts; fall back
    # to str() rather than letting json.dumps raise.
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:
        pass
    return str(obj)


def copy_config(config_path: str | Path, dest_dir: str | Path) -> Path:
    """Copy the exact config file used for a run into its run folder, per
    project requirement 6 (trackability of which settings produced which
    predictions)."""
    dest = Path(dest_dir) / "config.yaml"
    shutil.copy2(config_path, dest)
    return dest
