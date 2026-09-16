#!/usr/bin/env python3
"""Fine-tuned MiniLM+MLP CAPEC classifier using Qwen-generated keywords."""

from __future__ import annotations

from pathlib import Path

from NN_fine_tune_keywords import DescriptionKeywordMiniLMClassifier
from fine_tune_runner import run


SCRIPT_DIR = Path(__file__).resolve().parent
QWEN_SPLIT_DIR = (
    SCRIPT_DIR.parent
    / "03_keyword_making"
    / "outputs"
    / "qwen_keywords"
)


def run_experiment() -> Path:
    """Run the existing fine-tuned keyword architecture on Qwen splits."""
    return run(
        target="capec",
        use_keywords=True,
        model_factory=DescriptionKeywordMiniLMClassifier,
        split_directory=QWEN_SPLIT_DIR,
        output_name="qwen_nnfinetuned",
        keyword_method="qwen_keywords",
    )


if __name__ == "__main__":
    run_experiment()
