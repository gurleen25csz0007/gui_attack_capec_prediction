"""Canonical Stage-04 splits produced by the selected Stage-03 keyword model."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
METHODOLOGY_DIR = SCRIPT_DIR.parent
KEYWORD_SPLIT_DIR = (
    METHODOLOGY_DIR
    / "03_keyword_making"
    / "outputs"
    / "keybert_modified"
)
TRAIN_PATH = KEYWORD_SPLIT_DIR / "training_final.csv"
VALIDATION_PATH = KEYWORD_SPLIT_DIR / "validation_final.csv"
TEST_PATH = KEYWORD_SPLIT_DIR / "testing_final.csv"

# Compatibility alias used by the existing Stage-04 callers and run metadata.
DATASET_PATH = KEYWORD_SPLIT_DIR
DEFAULT_SEED = 42


def _load_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Selected KeyBERT {split_name} split not found: {path}"
        )
    frame = pd.read_csv(path).reset_index(drop=True)
    required = {
        "cve_id",
        "cleaned_description",
        "keyphrases",
        "capec_id",
        "weakness",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame["cve_id"].isna().any():
        raise ValueError(f"{path} contains missing cve_id values.")
    if frame["cve_id"].astype(str).duplicated().any():
        raise ValueError(f"{path} contains duplicate cve_id values.")
    return frame


def load_fixed_splits(
    split_dir: Path = DATASET_PATH,
    seed: int = DEFAULT_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the fixed KeyBERT 70/15/15 splits and validate CVE isolation.

    ``seed`` remains in the API for existing callers. Stage 03 already created
    the split with seed 42, so Stage 04 must never split or shuffle it again.
    """
    if seed != DEFAULT_SEED:
        raise ValueError(
            f"The selected Stage-03 splits use seed {DEFAULT_SEED}; "
            f"received seed={seed}."
        )
    split_dir = Path(split_dir)
    train = _load_split(split_dir / "training_final.csv", "training")
    validation = _load_split(
        split_dir / "validation_final.csv", "validation"
    )
    test = _load_split(split_dir / "testing_final.csv", "testing")

    split_sets = [
        set(part["cve_id"].astype(str))
        for part in (train, validation, test)
    ]
    if (
        split_sets[0] & split_sets[1]
        or split_sets[0] & split_sets[2]
        or split_sets[1] & split_sets[2]
    ):
        raise RuntimeError("CVE leakage detected between the fixed splits.")
    return train, validation, test
