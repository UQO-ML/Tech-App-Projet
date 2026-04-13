"""Structures communes pour décrire les modèles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelSpec:
    """Spécification standard d'un modèle de classification."""

    name: str
    pipeline: Any
    param_grid: dict[str, Any]
    why: str
