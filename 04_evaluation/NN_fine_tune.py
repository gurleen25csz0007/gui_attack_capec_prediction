#!/usr/bin/env python3
"""CVE description -> fine-tuned MiniLM -> MLP -> multi-label CAPEC.

This is the trainable-encoder counterpart of :mod:`NN`.  It uses the same
fixed Stage-03 splits, cleaned-description input, CAPEC label space, MLP head,
weighted BCE loss, validation threshold search, early stopping, and test
metrics.  The sole method change is end-to-end MiniLM fine-tuning.

The reusable training implementation lives in ``fine_tune_runner.py`` so the
CAPEC and CWE experiments cannot silently drift apart.
"""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel

from fine_tune_runner import TrainingConfig, run


TARGET = "capec"
USE_KEYWORDS = False


class DescriptionMiniLMClassifier(nn.Module):
    """The ``NN.py`` MLP with a trainable MiniLM encoder in front of it."""

    def __init__(self, config: TrainingConfig, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.input_dim = int(self.encoder.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, num_labels),
        )

    @staticmethod
    def mean_pool(output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(output.last_hidden_state.dtype)
        return (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        description = self.encoder(
            input_ids=batch["desc_input_ids"],
            attention_mask=batch["desc_attention_mask"],
        )
        description = self.mean_pool(description, batch["desc_attention_mask"])
        description = torch.nn.functional.normalize(description, dim=1)
        return self.classifier(description)


def run_experiment() -> Path:
    """Train, validate, test, and save the description-only CAPEC model."""
    return run(
        target=TARGET,
        use_keywords=USE_KEYWORDS,
        model_factory=DescriptionMiniLMClassifier,
    )


if __name__ == "__main__":
    run_experiment()
