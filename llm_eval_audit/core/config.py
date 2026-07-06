"""Loads `configs/eval_config.yaml` into a plain nested dict.

Kept as a thin, dependency-light loader (not a schema/validation library)
since the config is small and every consumer already knows which keys it
needs; failing loudly on a missing key via a plain `KeyError` is fine here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/eval_config.yaml") -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
