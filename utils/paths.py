"""Shared, machine-independent path defaults."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """Return DATA_ROOT, falling back to the repository's ``data`` directory."""
    configured = os.environ.get("DATA_ROOT")
    return Path(configured).expanduser() if configured else REPO_ROOT / "data"


def material_data_root() -> Path:
    """Return the directory containing the MAOAM material datasets."""
    configured = os.environ.get("MATERIAL_DATA_ROOT")
    return Path(configured).expanduser() if configured else data_root() / "maoam_data"
