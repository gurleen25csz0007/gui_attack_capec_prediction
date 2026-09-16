#!/usr/bin/env python3
"""Fine-tune MiniLM end-to-end for CVE description -> multi-label CWE.

Standalone trainable-encoder counterpart of ``NN_1way.py``. It preserves the
fixed splits, CWE labels, MLP, weighted BCE loss, validation threshold search,
early stopping, and test metrics. Only MiniLM's frozen status changes.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, hamming_loss, jaccard_score,
    label_ranking_average_precision_score, precision_score, recall_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = SCRIPT_DIR.parent
DATA_DIR = EVALUATION_DIR.parent / "03_keyword_making" / "outputs" / "keybert_modified"
TRAIN_PATH = DATA_DIR / "training_final.csv"
VALIDATION_PATH = DATA_DIR / "validation_final.csv"
TEST_PATH = DATA_DIR / "testing_final.csv"
OUTPUT_ROOT = Path(os.environ.get(
    "EVALUATION_OUTPUT_ROOT", str(EVALUATION_DIR / "outputs")
)).expanduser().resolve()
OUTPUT_DIR = OUTPUT_ROOT / "CVE_to_CWE" / "NN_fine_tune"

DESCRIPTION_COLUMN = "cleaned_description"
LABEL_COLUMN = "weakness"
CVE_COLUMN = "cve_id"
NO_ID_LABEL = "CWE-noID"
THRESHOLDS = (0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98)
TOP_K_VALUES = (1, 2, 3, 5, 10, 20, 30, 50)


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    random_seed: int = 42
    epochs: int = 50
    patience: int = 5
    effective_batch_size: int = 256
    train_batch_size: int = 32
    eval_batch_size: int = 128
    encoder_learning_rate: float = 2e-5
    classifier_learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    max_length: int = 192
    hidden_size: int = 512
    dropout: float = 0.30
    positive_weight_cap: float = 50.0

    @property
    def accumulation_steps(self) -> int:
        if self.effective_batch_size % self.train_batch_size:
            raise ValueError("effective_batch_size must divide by train_batch_size")
        return self.effective_batch_size // self.train_batch_size


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_cwe_cell(value: Any) -> list[str]:
    """Return unique canonical CWE IDs; missing/no-ID values return []."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null", "[]", "cwe-noid"}:
        return []
    matches = re.findall(r"CWE\s*[-_:]?\s*(\d+)", text, flags=re.I)
    return sorted({f"CWE-{int(item)}" for item in matches})


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {split_name} split: {path}")
    frame = pd.read_csv(path).reset_index(drop=True)
    missing = sorted({CVE_COLUMN, DESCRIPTION_COLUMN, LABEL_COLUMN} - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame[CVE_COLUMN].isna().any() or frame[CVE_COLUMN].astype(str).duplicated().any():
        raise ValueError(f"{path} contains missing or duplicate CVE IDs")
    frame[DESCRIPTION_COLUMN] = frame[DESCRIPTION_COLUMN].fillna("").astype(str)
    frame = frame[frame[DESCRIPTION_COLUMN].str.strip().ne("")].reset_index(drop=True)
    frame["target_list"] = frame[LABEL_COLUMN].map(parse_cwe_cell).map(
        lambda labels: labels if labels else [NO_ID_LABEL]
    )
    return frame


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train = load_split(TRAIN_PATH, "training")
    validation = load_split(VALIDATION_PATH, "validation")
    test = load_split(TEST_PATH, "testing")
    ids = [set(frame[CVE_COLUMN].astype(str)) for frame in (train, validation, test)]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise RuntimeError("CVE leakage detected between fixed splits")
    labels = sorted({
        label for frame in (train, validation, test)
        for values in frame["target_list"] for label in values
        if label != NO_ID_LABEL
    }) + [NO_ID_LABEL]
    return train, validation, test, labels


def make_multihot(rows: Sequence[Sequence[str]], label_to_index: dict[str, int]) -> np.ndarray:
    targets = np.zeros((len(rows), len(label_to_index)), dtype=np.uint8)
    for row_index, row_labels in enumerate(rows):
        for label in row_labels:
            targets[row_index, label_to_index[label]] = 1
    return targets


class DescriptionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, targets: np.ndarray):
        self.descriptions = frame[DESCRIPTION_COLUMN].tolist()
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[str, np.ndarray]:
        return self.descriptions[index], self.targets[index]


class BatchCollator:
    def __init__(self, tokenizer: Any, config: TrainingConfig):
        self.tokenizer, self.config = tokenizer, config

    def __call__(self, rows: Sequence[tuple[str, np.ndarray]]) -> dict[str, torch.Tensor]:
        descriptions, targets = zip(*rows)
        encoded = self.tokenizer(
            list(descriptions), padding=True, truncation=True,
            max_length=self.config.max_length, return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "targets": torch.from_numpy(np.stack(targets).astype(np.float32)),
        }


class DescriptionMiniLMClassifier(nn.Module):
    """MiniLM mean-pooled sentence vector followed by the original NN MLP."""

    def __init__(self, config: TrainingConfig, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.input_dim = int(self.encoder.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, config.hidden_size), nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(),
            nn.Dropout(config.dropout), nn.Linear(config.hidden_size, num_labels),
        )

    @staticmethod
    def mean_pool(output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(output.last_hidden_state.dtype)
        return (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1e-9)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        embedding = self.mean_pool(output, batch["attention_mask"])
        return self.classifier(torch.nn.functional.normalize(embedding, dim=1))


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def postprocess(probabilities: np.ndarray, threshold: float, no_id_index: int) -> np.ndarray:
    predictions = (probabilities >= threshold).astype(np.uint8)
    for row_index, row in enumerate(predictions):
        positive = np.flatnonzero(row)
        if not len(positive):
            predictions[row_index, int(np.argmax(probabilities[row_index]))] = 1
        elif any(index != no_id_index for index in positive):
            predictions[row_index, no_id_index] = 0
    return predictions


def evaluate(
    truth: np.ndarray, probabilities: np.ndarray, threshold: float, no_id_index: int,
) -> tuple[dict[str, float], np.ndarray]:
    predictions = postprocess(probabilities, threshold, no_id_index)
    metrics = {
        "threshold": float(threshold),
        "exact_match_accuracy": float(accuracy_score(truth, predictions)),
        "micro_precision": float(precision_score(truth, predictions, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(truth, predictions, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(truth, predictions, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(truth, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predictions, average="weighted", zero_division=0)),
        "sample_precision": float(precision_score(truth, predictions, average="samples", zero_division=0)),
        "sample_recall": float(recall_score(truth, predictions, average="samples", zero_division=0)),
        "sample_f1": float(f1_score(truth, predictions, average="samples", zero_division=0)),
        "sample_jaccard": float(jaccard_score(truth, predictions, average="samples", zero_division=0)),
        "hamming_loss": float(hamming_loss(truth, predictions)),
        "label_ranking_average_precision": float(
            label_ranking_average_precision_score(truth, probabilities)
        ),
    }
    return metrics, predictions


def select_threshold(
    truth: np.ndarray, probabilities: np.ndarray, no_id_index: int,
) -> tuple[float, dict[str, float]]:
    candidates = [evaluate(truth, probabilities, value, no_id_index)[0] for value in THRESHOLDS]
    best = max(candidates, key=lambda item: item["micro_f1"])
    return float(best["threshold"]), best


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, targets = [], []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        probabilities.append(torch.sigmoid(model(batch)).float().cpu().numpy())
        targets.append(batch["targets"].byte().cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(targets)


def ranking_metrics(
    truth: np.ndarray, probabilities: np.ndarray, no_id_index: int,
) -> dict[str, float | int]:
    scores = probabilities.copy()
    scores[:, no_id_index] = -np.inf
    order = np.argsort(-scores, axis=1)
    results: dict[str, float | int] = {}
    for k in TOP_K_VALUES:
        hits, recalls = [], []
        for row_index in range(len(truth)):
            actual = set(np.flatnonzero(truth[row_index])) - {no_id_index}
            if not actual:
                continue
            top = set(order[row_index, :k])
            hits.append(bool(actual & top))
            recalls.append(len(actual & top) / len(actual))
        results[f"hit@{k}"] = float(np.mean(hits)) if hits else 0.0
        results[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        results[f"evaluated_rows@{k}"] = len(hits)
    return results


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_and_evaluate() -> Path:
    config = TrainingConfig()
    seed_everything(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, validation, test, labels = load_data()
    label_to_index = {label: index for index, label in enumerate(labels)}
    index_to_label = {str(index): label for index, label in enumerate(labels)}
    no_id_index = label_to_index[NO_ID_LABEL]
    y_train = make_multihot(train["target_list"].tolist(), label_to_index)
    y_validation = make_multihot(validation["target_list"].tolist(), label_to_index)
    y_test = make_multihot(test["target_list"].tolist(), label_to_index)
    print(
        f"device={device} labels={len(labels)} train={len(train)} "
        f"validation={len(validation)} test={len(test)}", flush=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = OUTPUT_DIR / "best_cwe_multilabel_mlp.pt"
    final_path = OUTPUT_DIR / "cwe_multilabel_mlp.pt"
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    collator = BatchCollator(tokenizer, config)
    generator = torch.Generator().manual_seed(config.random_seed)
    train_loader = DataLoader(
        DescriptionDataset(train, y_train), batch_size=config.train_batch_size,
        shuffle=True, collate_fn=collator, generator=generator,
    )
    validation_loader = DataLoader(
        DescriptionDataset(validation, y_validation), batch_size=config.eval_batch_size,
        shuffle=False, collate_fn=collator,
    )
    test_loader = DataLoader(
        DescriptionDataset(test, y_test), batch_size=config.eval_batch_size,
        shuffle=False, collate_fn=collator,
    )

    model = DescriptionMiniLMClassifier(config, len(labels)).to(device)
    positive_counts = y_train.sum(axis=0).astype(np.float32)
    positive_weights = np.ones(len(labels), dtype=np.float32)
    present = positive_counts > 0
    positive_weights[present] = (
        len(y_train) - positive_counts[present]
    ) / positive_counts[present]
    positive_weights = np.clip(positive_weights, 1.0, config.positive_weight_cap)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(positive_weights).to(device))
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": config.encoder_learning_rate},
        {"params": model.classifier.parameters(), "lr": config.classifier_learning_rate},
    ], weight_decay=config.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / config.accumulation_steps)
    total_updates = updates_per_epoch * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_updates * config.warmup_ratio),
        num_training_steps=total_updates,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_score, stale_epochs = -1.0, 0
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step, raw_batch in enumerate(train_loader, start=1):
            batch = move_batch(raw_batch, device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                raw_loss = criterion(model(batch), batch["targets"])
                loss = raw_loss / config.accumulation_steps
            scaler.scale(loss).backward()
            total_loss += float(raw_loss.item())
            if step % config.accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        validation_probabilities, validation_truth = predict(model, validation_loader, device)
        threshold, validation_metrics = select_threshold(
            validation_truth, validation_probabilities, no_id_index
        )
        score = validation_metrics["micro_f1"]
        print(
            f"epoch={epoch}/{config.epochs} loss={total_loss / len(train_loader):.4f} "
            f"threshold={threshold:.2f} val_micro_f1={score:.4f}", flush=True,
        )
        if score > best_score:
            best_score, stale_epochs = score, 0
            torch.save({
                "model_state_dict": model.state_dict(), "all_labels": labels,
                "label_to_idx": label_to_index, "idx_to_label": index_to_label,
                "target": "cwe", "use_keywords": False,
                "feature_mode": DESCRIPTION_COLUMN, "embed_model_name": config.model_name,
                "input_dim": model.input_dim, "best_epoch": epoch,
                "threshold": threshold, "validation_metrics": validation_metrics,
                "training_config": asdict(config),
            }, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                print(f"early stopping after epoch {epoch}", flush=True)
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_probabilities, test_truth = predict(model, test_loader, device)
    test_metrics, test_predictions = evaluate(
        test_truth, test_probabilities, checkpoint["threshold"], no_id_index
    )
    top_k = ranking_metrics(test_truth, test_probabilities, no_id_index)
    torch.save(checkpoint, final_path)
    tokenizer.save_pretrained(OUTPUT_DIR / "tokenizer")
    save_json(OUTPUT_DIR / "label_to_idx.json", label_to_index)
    save_json(OUTPUT_DIR / "idx_to_label.json", index_to_label)
    pd.DataFrame({
        "index": range(len(labels)), "label": labels,
        "training_count": y_train.sum(axis=0).astype(int),
    }).to_csv(OUTPUT_DIR / "label_info.csv", index=False)

    output = test.copy()
    output["true_cwe_labels"] = [
        [labels[index] for index in np.flatnonzero(row)] for row in test_truth
    ]
    output["pred_cwe_labels"] = [
        [labels[index] for index in np.flatnonzero(row)] for row in test_predictions
    ]
    for k in (1, 2, 3, 5, 10):
        output[f"top_{k}_predictions"] = [[
            (labels[index], float(row[index])) for index in np.argsort(-row)
            if index != no_id_index
        ][:k] for row in test_probabilities]
    output.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    output.to_csv(OUTPUT_DIR / "topk_hit_results.csv", index=False)
    save_json(OUTPUT_DIR / "metrics.json", {
        "completed": True, "model": "NN_fine_tune", "target": "cwe",
        "source_splits": str(DATA_DIR), "selection_split": "validation",
        "selection_metric": "micro_f1", "best_epoch": checkpoint["best_epoch"],
        "best_threshold": checkpoint["threshold"],
        "validation_metrics": checkpoint["validation_metrics"],
        "best_validation_metrics": checkpoint["validation_metrics"],
        "test_metrics": test_metrics, "top_k_metrics": top_k,
        "top_k_hit_real_cwe_rows": {
            key: value for key, value in top_k.items()
            if key.startswith("hit@") or key.startswith("evaluated_rows@")
        },
        "top_k_recall_real_cwe_rows": {
            key: value for key, value in top_k.items()
            if key.startswith("recall@") or key.startswith("evaluated_rows@")
        },
        "training_config": asdict(config), "elapsed_seconds": time.time() - started,
    })
    print(f"test_micro_f1={test_metrics['micro_f1']:.4f}")
    print(f"saved={OUTPUT_DIR}")
    return OUTPUT_DIR


if __name__ == "__main__":
    train_and_evaluate()
