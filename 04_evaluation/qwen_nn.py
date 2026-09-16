#!/usr/bin/env python3
"""Frozen MiniLM+MLP CAPEC classifier using Qwen-generated keywords."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
QWEN_SPLIT_DIR = (
    SCRIPT_DIR.parent
    / "03_keyword_making"
    / "outputs"
    / "qwen_keywords"
)

# NN_keywords.py retains the same NN hyperparameters and implementation while
# these settings isolate the Qwen input and artifacts.
os.environ["EVALUATION_KEYWORD_SPLIT_DIR"] = str(QWEN_SPLIT_DIR)
os.environ["EVALUATION_KEYWORD_METHOD"] = "qwen_keywords"
os.environ["EVALUATION_MODEL_OUTPUT_NAME"] = "qwen_nn"

runpy.run_path(str(SCRIPT_DIR / "NN_keywords.py"), run_name="__main__")
