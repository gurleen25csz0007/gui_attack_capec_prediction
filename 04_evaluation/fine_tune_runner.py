#!/usr/bin/env python3
"""Shared end-to-end MiniLM fine-tuning for the direct-CAPEC NN variants.

The representation and evaluation path intentionally mirror ``NN.py`` and
``NN_keywords.py``. The only modelling change is that MiniLM participates in
back-propagation instead of producing frozen, precomputed embeddings.

The CVE-to-CWE fine-tuning programs are fully standalone in ``CVE_to_CWE``;
this module deliberately contains no CWE experiment implementation.
"""

from __future__ import annotations

import ast
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    label_ranking_average_precision_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from evaluation_data import DATASET_PATH, load_fixed_splits
from evaluation_paths import OUTPUT_ROOT


SCRIPT_DIR = Path(__file__).resolve().parent
CAPEC_CATALOG = SCRIPT_DIR.parent / "01_cwe_capec_ground_truth" / "3000_capec.csv"


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
    description_max_length: int = 192
    keyword_max_length: int = 64
    hidden_size: int = 512
    dropout: float = 0.30
    positive_weight_cap: float = 50.0
    selection_metric: str = "micro_f1"

    @property
    def gradient_accumulation_steps(self) -> int:
        if self.effective_batch_size % self.train_batch_size:
            raise ValueError("effective_batch_size must divide by train_batch_size")
        return self.effective_batch_size // self.train_batch_size


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_ids(value: Any, prefix: str) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null", "[]"}:
        return []
    matches = re.findall(rf"{prefix}\s*[-_:]?\s*(\d+)", text, flags=re.I)
    if not matches:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = []
        if not isinstance(parsed, (list, tuple, set)):
            parsed = [parsed]
        for item in parsed:
            matches.extend(
                re.findall(rf"{prefix}\s*[-_:]?\s*(\d+)", str(item), flags=re.I)
            )
    return sorted({f"{prefix}-{int(item)}" for item in matches})


def parse_keyphrases(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null", "[]"}:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = None
    values = (
        list(parsed)
        if isinstance(parsed, (list, tuple, set))
        else re.split(r"[;,|\n]+", text)
    )
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        phrase = " ".join(str(value).strip().split())
        key = phrase.casefold()
        if phrase and key not in seen:
            seen.add(key)
            result.append(phrase)
    return result


def capec_catalog_labels() -> list[str]:
    frame = pd.read_csv(CAPEC_CATALOG)
    column = next(
        (name for name in ("ID", "id", "capec_id") if name in frame.columns),
        None,
    )
    if column is None:
        raise ValueError(f"Cannot find the CAPEC ID column in {CAPEC_CATALOG}")
    labels: set[str] = set()
    for value in frame[column].dropna():
        labels.update(parse_ids(value, "CAPEC"))
        if re.fullmatch(r"\d+(?:\.0)?", str(value).strip()):
            labels.add(f"CAPEC-{int(float(value))}")
    return sorted(labels)


def prepare_frames(
    target: str,
    split_directory: Path = DATASET_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    if target != "capec":
        raise ValueError("fine_tune_runner.py is only for the direct CAPEC models")
    train, validation, test = load_fixed_splits(split_directory, seed=42)
    prefix = "CAPEC"
    column = "capec_id"
    no_id = f"{prefix}-noID"
    frames = []
    data_labels: set[str] = set()
    for source in (train, validation, test):
        frame = source.copy()
        # The frozen NN scripts remove blank descriptions before constructing
        # their label matrices.  Preserve that exact behaviour here.
        frame["cleaned_description"] = (
            frame["cleaned_description"].fillna("").astype(str)
        )
        frame = frame[
            frame["cleaned_description"].str.strip().ne("")
        ].reset_index(drop=True)
        frame["target_list"] = frame[column].map(lambda value: parse_ids(value, prefix))
        frame["target_list"] = frame["target_list"].map(
            lambda labels: labels if labels else [no_id]
        )
        frame["keyword_list"] = frame["keyphrases"].map(parse_keyphrases)
        frame["keyword_text"] = frame["keyword_list"].map(
            lambda values: " [SEP] ".join(values)
        )
        frame["has_keywords"] = frame["keyword_list"].map(bool)
        data_labels.update(label for labels in frame["target_list"] for label in labels)
        frames.append(frame)
    real_labels = sorted(set(capec_catalog_labels()) | (data_labels - {no_id}))
    return frames[0], frames[1], frames[2], real_labels + [no_id]


def multihot(rows: Sequence[Sequence[str]], label_to_idx: dict[str, int]) -> np.ndarray:
    result = np.zeros((len(rows), len(label_to_idx)), dtype=np.uint8)
    for row, labels in enumerate(rows):
        for label in labels:
            result[row, label_to_idx[label]] = 1
    return result


class TextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, targets: np.ndarray):
        self.descriptions = frame["cleaned_description"].fillna("").astype(str).tolist()
        self.keyword_lists = frame["keyword_list"].tolist()
        self.has_keywords = frame["has_keywords"].astype(bool).tolist()
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[str, list[str], bool, np.ndarray]:
        return (
            self.descriptions[index],
            self.keyword_lists[index],
            self.has_keywords[index],
            self.targets[index],
        )


class BatchCollator:
    def __init__(self, tokenizer: Any, use_keywords: bool, config: TrainingConfig):
        self.tokenizer = tokenizer
        self.use_keywords = use_keywords
        self.config = config

    def __call__(self, rows: Sequence[tuple[str, list[str], bool, np.ndarray]]) -> dict[str, torch.Tensor]:
        descriptions, keyword_lists, has_keywords, targets = zip(*rows)
        desc = self.tokenizer(
            list(descriptions),
            padding=True,
            truncation=True,
            max_length=self.config.description_max_length,
            return_tensors="pt",
        )
        batch = {
            "desc_input_ids": desc["input_ids"],
            "desc_attention_mask": desc["attention_mask"],
            "targets": torch.from_numpy(np.stack(targets).astype(np.float32)),
        }
        if self.use_keywords:
            keyword_phrases: list[str] = []
            keyword_rows: list[int] = []
            for row_index, phrases in enumerate(keyword_lists):
                keyword_phrases.extend(phrases)
                keyword_rows.extend([row_index] * len(phrases))
            # KeyBERT coverage is high, but retain a harmless dummy phrase so
            # an entirely empty batch remains tokenizable.
            if not keyword_phrases:
                keyword_phrases = [""]
                keyword_rows = [-1]
            encoded_keywords = self.tokenizer(
                keyword_phrases,
                padding=True,
                truncation=True,
                max_length=self.config.keyword_max_length,
                return_tensors="pt",
            )
            batch.update({
                "keyword_input_ids": encoded_keywords["input_ids"],
                "keyword_attention_mask": encoded_keywords["attention_mask"],
                "has_keywords": torch.tensor(has_keywords, dtype=torch.float32),
                "keyword_row_indices": torch.tensor(keyword_rows, dtype=torch.long),
            })
        return batch


class FineTunedClassifier(nn.Module):
    def __init__(self, config: TrainingConfig, num_labels: int, use_keywords: bool):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.use_keywords = use_keywords
        encoder_size = int(self.encoder.config.hidden_size)
        input_size = encoder_size * (2 if use_keywords else 1)
        self.classifier = nn.Sequential(
            nn.Linear(input_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, num_labels),
        )
        self.input_dim = input_size

    @staticmethod
    def pool(output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(output.last_hidden_state.dtype)
        return (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1e-9)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        desc = torch.nn.functional.normalize(self.pool(
            self.encoder(
                input_ids=batch["desc_input_ids"],
                attention_mask=batch["desc_attention_mask"],
            ),
            batch["desc_attention_mask"],
        ), dim=1)
        features = desc
        if self.use_keywords:
            phrase_embeddings = torch.nn.functional.normalize(self.pool(
                self.encoder(
                    input_ids=batch["keyword_input_ids"],
                    attention_mask=batch["keyword_attention_mask"],
                ),
                batch["keyword_attention_mask"],
            ), dim=1)
            keywords = torch.zeros(
                (desc.shape[0], phrase_embeddings.shape[1]),
                dtype=phrase_embeddings.dtype,
                device=phrase_embeddings.device,
            )
            counts = torch.zeros(
                desc.shape[0], dtype=phrase_embeddings.dtype,
                device=phrase_embeddings.device,
            )
            valid = batch["keyword_row_indices"] >= 0
            if bool(valid.any()):
                row_indices = batch["keyword_row_indices"][valid]
                keywords = keywords.index_add(
                    0, row_indices, phrase_embeddings[valid]
                )
                counts.index_add_(
                    0, row_indices,
                    torch.ones_like(row_indices, dtype=phrase_embeddings.dtype),
                )
            keywords = keywords / counts.clamp_min(1.0).unsqueeze(1)
            keywords = torch.nn.functional.normalize(keywords, dim=1)
            keywords = keywords * batch["has_keywords"].unsqueeze(1)
            features = torch.cat([desc, keywords], dim=1)
        return self.classifier(features)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def postprocess(probabilities: np.ndarray, threshold: float, no_id: int) -> np.ndarray:
    prediction = (probabilities >= threshold).astype(np.uint8)
    for row in range(len(prediction)):
        positives = np.flatnonzero(prediction[row])
        if not len(positives):
            prediction[row, int(np.argmax(probabilities[row]))] = 1
        elif any(index != no_id for index in positives):
            prediction[row, no_id] = 0
    return prediction


def metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float, no_id: int) -> tuple[dict[str, float], np.ndarray]:
    prediction = postprocess(probabilities, threshold, no_id)
    result = {
        "threshold": float(threshold),
        "exact_match_accuracy": float(accuracy_score(y_true, prediction)),
        "micro_precision": float(precision_score(y_true, prediction, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, prediction, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, prediction, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, prediction, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, prediction, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, prediction, average="weighted", zero_division=0)),
        "sample_precision": float(precision_score(y_true, prediction, average="samples", zero_division=0)),
        "sample_recall": float(recall_score(y_true, prediction, average="samples", zero_division=0)),
        "sample_f1": float(f1_score(y_true, prediction, average="samples", zero_division=0)),
        "sample_jaccard": float(jaccard_score(y_true, prediction, average="samples", zero_division=0)),
        "sklearn_samples_jaccard": float(jaccard_score(y_true, prediction, average="samples", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, prediction)),
        "label_ranking_average_precision": float(label_ranking_average_precision_score(y_true, probabilities)),
    }
    return result, prediction


def best_threshold(y_true: np.ndarray, probabilities: np.ndarray, no_id: int) -> tuple[float, dict[str, float]]:
    thresholds = np.array([
        0.80,
        0.85,
        0.90,
        0.92,
        0.94,
        0.96,
        0.97,
        0.98,
        0.99,
    ])
    scored = [metrics(y_true, probabilities, round(float(value), 4), no_id)[0] for value in thresholds]
    winner = max(scored, key=lambda row: row["micro_f1"])
    return float(winner["threshold"]), winner


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        probabilities.append(torch.sigmoid(model(batch)).float().cpu().numpy())
        targets.append(batch["targets"].byte().cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(targets)


def ranking_metrics(y_true: np.ndarray, probabilities: np.ndarray, no_id: int) -> dict[str, float]:
    result: dict[str, float] = {}
    real_probabilities = probabilities.copy()
    real_probabilities[:, no_id] = -np.inf
    order = np.argsort(-real_probabilities, axis=1)
    for k in (1, 2, 3, 5, 10, 20, 30, 50):
        hits, recalls = [], []
        for row in range(len(y_true)):
            truth = set(np.flatnonzero(y_true[row])) - {no_id}
            top = set(order[row, :k])
            if truth:
                hits.append(bool(truth & top))
                recalls.append(len(truth & top) / len(truth))
        result[f"hit@{k}"] = float(np.mean(hits)) if hits else 0.0
        result[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        result[f"evaluated_rows@{k}"] = len(hits)
    return result


def labels_for_row(row: np.ndarray, labels: Sequence[str]) -> list[str]:
    return [labels[index] for index in np.flatnonzero(row)]


def top_labels_for_row(
    row: np.ndarray,
    labels: Sequence[str],
    no_id: int,
    k: int,
) -> list[tuple[str, float]]:
    order = [index for index in np.argsort(-row) if index != no_id][:k]
    return [(labels[index], float(row[index])) for index in order]


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def run(
    target: str,
    use_keywords: bool,
    model_factory: Callable[[TrainingConfig, int], nn.Module] | None = None,
    split_directory: Path = DATASET_PATH,
    output_name: str | None = None,
    keyword_method: str | None = None,
) -> Path:
    if target != "capec":
        raise ValueError("fine_tune_runner.py is only for the direct CAPEC models")
    config = TrainingConfig()
    seed_everything(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_directory = Path(split_directory).expanduser().resolve()
    train, validation, test, labels = prepare_frames(target, split_directory)
    print(f"device={device}")
    print(
        f"target={target} use_keywords={use_keywords} "
        f"train={len(train)} validation={len(validation)} test={len(test)}"
    )
    label_to_idx = {label: index for index, label in enumerate(labels)}
    no_id = label_to_idx["CAPEC-noID"]
    y_train = multihot(train["target_list"].tolist(), label_to_idx)
    y_validation = multihot(validation["target_list"].tolist(), label_to_idx)
    y_test = multihot(test["target_list"].tolist(), label_to_idx)

    suffix = output_name or (
        "NN_fine_tune_keywords" if use_keywords else "NN_fine_tune"
    )
    output_dir = OUTPUT_ROOT / suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"best_{target}_multilabel_mlp.pt"
    final_checkpoint_path = output_dir / f"{target}_multilabel_mlp.pt"

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    collator = BatchCollator(tokenizer, use_keywords, config)
    generator = torch.Generator().manual_seed(config.random_seed)
    train_loader = DataLoader(
        TextDataset(train, y_train), batch_size=config.train_batch_size,
        shuffle=True, collate_fn=collator, generator=generator,
    )
    validation_loader = DataLoader(
        TextDataset(validation, y_validation), batch_size=config.eval_batch_size,
        shuffle=False, collate_fn=collator,
    )
    test_loader = DataLoader(
        TextDataset(test, y_test), batch_size=config.eval_batch_size,
        shuffle=False, collate_fn=collator,
    )

    model = (
        model_factory(config, len(labels))
        if model_factory is not None
        else FineTunedClassifier(config, len(labels), use_keywords)
    ).to(device)
    counts = y_train.sum(axis=0).astype(np.float32)
    weights = np.ones(len(labels), dtype=np.float32)
    present = counts > 0
    weights[present] = (len(y_train) - counts[present]) / counts[present]
    weights = np.clip(weights, 1.0, config.positive_weight_cap)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": config.encoder_learning_rate},
        {"params": model.classifier.parameters(), "lr": config.classifier_learning_rate},
    ], weight_decay=config.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_updates = updates_per_epoch * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * config.warmup_ratio),
        num_training_steps=total_updates,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_score = -1.0
    stale_epochs = 0
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, raw_batch in enumerate(train_loader, start=1):
            batch = move_batch(raw_batch, device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss = criterion(model(batch), batch["targets"])
                loss = loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            running_loss += float(loss.item()) * config.gradient_accumulation_steps
            should_update = (
                step % config.gradient_accumulation_steps == 0
                or step == len(train_loader)
            )
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        validation_probabilities, validation_truth = predict(model, validation_loader, device)
        threshold, validation_metrics = best_threshold(
            validation_truth, validation_probabilities, no_id
        )
        score = validation_metrics[config.selection_metric]
        print(
            f"epoch={epoch}/{config.epochs} loss={running_loss / len(train_loader):.4f} "
            f"threshold={threshold:.2f} val_micro_f1={score:.4f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "all_labels": labels,
                "label_to_idx": label_to_idx,
                "idx_to_label": {str(index): label for index, label in enumerate(labels)},
                "target": target,
                "use_keywords": use_keywords,
                "feature_mode": "cleaned_description+keywords" if use_keywords else "cleaned_description",
                "embed_model_name": config.model_name,
                "input_dim": model.input_dim,
                "best_epoch": epoch,
                "epoch": epoch,
                "threshold": threshold,
                "validation_metrics": validation_metrics,
                "val_metrics": validation_metrics,
                "training_config": asdict(config),
                "train_rows": len(train),
                "val_rows": len(validation),
                "test_rows": len(test),
            }, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                print(f"early stopping after epoch {epoch}", flush=True)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_probabilities, test_truth = predict(model, test_loader, device)
    test_metrics, test_prediction = metrics(
        test_truth, test_probabilities, float(checkpoint["threshold"]), no_id
    )
    top_k = ranking_metrics(test_truth, test_probabilities, no_id)
    # Keep the same pair of checkpoint names emitted by the frozen scripts.
    torch.save(checkpoint, final_checkpoint_path)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    save_json(output_dir / "label_to_idx.json", label_to_idx)
    save_json(output_dir / "idx_to_label.json", {str(index): label for index, label in enumerate(labels)})
    pd.DataFrame({
        "index": range(len(labels)),
        "label": labels,
        "training_count": y_train.sum(axis=0).astype(int),
    }).to_csv(output_dir / "label_info.csv", index=False)
    prediction_frame = test.copy()
    prefix = "capec"
    prediction_frame[f"true_{prefix}_labels"] = [
        labels_for_row(row, labels) for row in test_truth
    ]
    prediction_frame[f"true_real_{prefix}_labels"] = [
        [label for label in labels_for_row(row, labels) if label != labels[no_id]]
        for row in test_truth
    ]
    prediction_frame[f"pred_{prefix}_labels"] = [
        labels_for_row(row, labels) for row in test_prediction
    ]
    for k in (1, 2, 3, 5, 10):
        top_column = f"top_{k}_predictions"
        prediction_frame[top_column] = [
            top_labels_for_row(row, labels, no_id, k)
            for row in test_probabilities
        ]
        prediction_frame[f"hit_at_{k}"] = [
            (
                int(bool(
                    (set(np.flatnonzero(test_truth[row_index])) - {no_id})
                    & {
                        label_to_idx[label]
                        for label, _score in prediction_frame.iloc[row_index][top_column]
                    }
                ))
                if (set(np.flatnonzero(test_truth[row_index])) - {no_id})
                else None
            )
            for row_index in range(len(prediction_frame))
        ]
    prediction_frame[f"top_10_{prefix}_scores"] = prediction_frame["top_10_predictions"]
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)
    prediction_frame.to_csv(output_dir / "topk_hit_results.csv", index=False)
    save_json(output_dir / "metrics.json", {
        "completed": True,
        "model": suffix,
        "target": target,
        "source_splits": str(split_directory),
        "keyword_method": keyword_method or ("keybert_modified" if use_keywords else None),
        "selection_split": "validation",
        "selection_metric": config.selection_metric,
        "best_epoch": checkpoint["best_epoch"],
        "best_threshold": checkpoint["threshold"],
        "validation_metrics": checkpoint["validation_metrics"],
        "best_validation_metrics": checkpoint["validation_metrics"],
        "test_metrics": test_metrics,
        "top_k_metrics": top_k,
        f"top_k_hit_real_{prefix}_rows": {
            key: value for key, value in top_k.items()
            if key.startswith("hit@") or key.startswith("evaluated_rows@")
        },
        f"top_k_recall_real_{prefix}_rows": {
            key: value for key, value in top_k.items()
            if key.startswith("recall@") or key.startswith("evaluated_rows@")
        },
        "training_config": asdict(config),
        "elapsed_seconds": time.time() - started,
    })
    print(f"test_micro_f1={test_metrics['micro_f1']:.4f}")
    print(f"saved={output_dir}")
    return output_dir


def main(target: str, use_keywords: bool) -> None:
    run(target, use_keywords)
