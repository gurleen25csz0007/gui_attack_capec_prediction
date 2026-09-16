"""Shared output locations for Stage-04 interactive and batch runs."""

from __future__ import annotations

import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs"
OUTPUT_ROOT = Path(
    os.environ.get("EVALUATION_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))
).expanduser().resolve()


def output_path(*parts: str) -> Path:
    """Return a path below the configured Stage-04 output root."""
    return OUTPUT_ROOT.joinpath(*parts)
