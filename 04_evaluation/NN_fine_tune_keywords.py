#!/usr/bin/env python3
"""Description + KeyBERT phrases -> fine-tuned MiniLM -> MLP -> CAPEC.

This is the trainable-encoder counterpart of :mod:`NN_keywords`.  Each
keyphrase is encoded separately, the phrase vectors are mean-pooled and L2
normalised, and that vector is concatenated with the normalised description
vector.  Thus the feature method stays the same; only MiniLM is now updated by
back-propagation.

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
USE_KEYWORDS = True


class DescriptionKeywordMiniLMClassifier(nn.Module):
    """Trainable MiniLM with the same two-vector input as ``NN_keywords.py``."""

    def __init__(self, config: TrainingConfig, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.model_name)
        encoder_size = int(self.encoder.config.hidden_size)
        self.input_dim = encoder_size * 2
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
        description = torch.nn.functional.normalize(
            self.mean_pool(description, batch["desc_attention_mask"]), dim=1
        )

        phrases = self.encoder(
            input_ids=batch["keyword_input_ids"],
            attention_mask=batch["keyword_attention_mask"],
        )
        phrases = torch.nn.functional.normalize(
            self.mean_pool(phrases, batch["keyword_attention_mask"]), dim=1
        )

        # Reproduce NN_keywords.py: mean-pool each row's individual phrase
        # embeddings, L2-normalise the result, then concatenate with the
        # description vector. Rows without phrases receive an all-zero vector.
        keywords = torch.zeros(
            (description.shape[0], phrases.shape[1]),
            dtype=phrases.dtype,
            device=phrases.device,
        )
        counts = torch.zeros(
            description.shape[0], dtype=phrases.dtype, device=phrases.device
        )
        valid = batch["keyword_row_indices"] >= 0
        if bool(valid.any()):
            row_indices = batch["keyword_row_indices"][valid]
            keywords = keywords.index_add(0, row_indices, phrases[valid])
            counts.index_add_(
                0,
                row_indices,
                torch.ones_like(row_indices, dtype=phrases.dtype),
            )
        keywords = keywords / counts.clamp_min(1.0).unsqueeze(1)
        keywords = torch.nn.functional.normalize(keywords, dim=1)
        keywords = keywords * batch["has_keywords"].unsqueeze(1)

        return self.classifier(torch.cat([description, keywords], dim=1))


def run_experiment() -> Path:
    """Train, validate, test, and save the keyword-aware CAPEC model."""
    return run(
        target=TARGET,
        use_keywords=USE_KEYWORDS,
        model_factory=DescriptionKeywordMiniLMClassifier,
    )


if __name__ == "__main__":
    run_experiment()
