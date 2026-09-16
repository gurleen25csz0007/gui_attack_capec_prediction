#!/usr/bin/env python3

"""Fair validation comparison of keyword methods with MiniLM + one MLP.

Only keyword text is used as model input.  Every method uses the same ordered
training/validation CVEs, label space, MiniLM encoder, mean pooling, MLP,
optimizer, threshold grid, seed, and early-stopping rule.  Test files are not
loaded or evaluated by this script.
"""

import ast
import copy
import json
import os
import random
import re
from pathlib import Path


# Set thread pools before importing NumPy/PyTorch.
CPU_THREADS = int(os.environ.get("KEYWORD_EVAL_CPU_THREADS", "16"))
for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
):
    os.environ[variable] = str(CPU_THREADS)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
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


# ---------------------------------------------------------------------------
# Paths and fixed experiment contract
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
METHODOLOGY_DIR = PROJECT_DIR.parent

METHOD_OUTPUT_ROOT_ENV = os.environ.get("KEYWORD_MAKER_OUTPUT_ROOT")
METHOD_OUTPUTS_DIR = (
    Path(METHOD_OUTPUT_ROOT_ENV).expanduser().resolve()
    if METHOD_OUTPUT_ROOT_ENV
    else PROJECT_DIR / "outputs"
)
OUTPUT_DIR = METHOD_OUTPUTS_DIR / "evaluation" / "minilm_nn_validation"
CAPEC_CATALOG_PATH = (
    METHODOLOGY_DIR / "01_cwe_capec_ground_truth" / "3000_capec.csv"
)

METHODS = {
    "qwen_keywords": "Qwen",
    "keybert_modified": "KeyBERT Modified",
    "textrank": "TextRank",
    "phrase_similarity": "Phrase Similarity",
    "occlusion": "Occlusion",
}
SPLIT_FILES = {
    "training": "training_final.csv",
    "validation": "validation_final.csv",
}

ID_COL = "cve_id"
KEYWORD_COL = "keyphrases"
LABEL_COL = "capec_id"
NO_ID_LABEL = "CAPEC-noID"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 256
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5
RANDOM_STATE = 42
THRESHOLDS = (0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98)
TOP_KS = (1, 3, 5, 10)
CONTRACT_VERSION = 1

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = DEVICE.type == "cuda"
torch.set_num_threads(CPU_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------

def is_empty(value):
    if pd.isna(value):
        return True
    return str(value).strip().casefold() in {
        "", "nan", "none", "null", "[]", "{}",
    }


def parse_capec_cell(value):
    """Parse list-like, delimited, or numeric CAPEC label cells."""
    if is_empty(value):
        return []
    text = str(value).strip()
    normalized = text.casefold().replace("_", "").replace(" ", "")
    if normalized in {"capec-noid", "noid", "nocapec", "capecnone"}:
        return []

    raw_items = None
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            raw_items = list(parsed)
        else:
            raw_items = [parsed]
    except (ValueError, SyntaxError):
        pass
    if raw_items is None:
        raw_items = re.split(r"[;,|\n]+", text)

    labels = []
    for item in raw_items:
        item_text = str(item).strip()
        item_normalized = (
            item_text.casefold().replace("_", "").replace(" ", "")
        )
        if item_normalized in {
            "capec-noid", "noid", "nocapec", "capecnone",
        }:
            continue
        matches = re.findall(r"CAPEC-\d+", item_text, flags=re.IGNORECASE)
        labels.extend(match.upper() for match in matches)
        if not matches and re.fullmatch(r"\d+(?:\.0)?", item_text):
            labels.append(f"CAPEC-{int(float(item_text))}")
    return sorted(set(labels), key=capec_sort_key)


def parse_keyphrase_cell(value):
    """Parse all keyword-generator formats with stable de-duplication."""
    if is_empty(value):
        return []
    text = str(value).strip()
    raw_items = None
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            raw_items = list(parsed)
        elif isinstance(parsed, str):
            raw_items = [parsed]
    except (ValueError, SyntaxError):
        pass
    if raw_items is None:
        raw_items = re.split(r"[;,|\n]+", text)

    result = []
    seen = set()
    for item in raw_items:
        if is_empty(item):
            continue
        phrase = re.sub(r"\s+", " ", str(item)).strip(
            " \t\r\n\"'[]{}()"
        )
        key = phrase.casefold()
        if phrase and key not in seen:
            result.append(phrase)
            seen.add(key)
    return result


def capec_sort_key(label):
    match = re.fullmatch(r"CAPEC-(\d+)", label, flags=re.IGNORECASE)
    return int(match.group(1)) if match else float("inf")


def load_method_frames(method_name):
    method_dir = METHOD_OUTPUTS_DIR / method_name
    frames = {}
    for split_name, filename in SPLIT_FILES.items():
        path = method_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {method_name}/{split_name}: {path}")
        frame = pd.read_csv(path)
        frame.columns = frame.columns.str.strip()
        missing = {ID_COL, KEYWORD_COL, LABEL_COL} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame = frame.reset_index(drop=True)
        if frame[ID_COL].isna().any() or frame[ID_COL].astype(str).str.strip().eq("").any():
            raise ValueError(f"{path} contains missing {ID_COL}")
        if frame[ID_COL].astype(str).duplicated().any():
            raise ValueError(f"{path} contains duplicate {ID_COL}")
        frame["keyphrase_list"] = frame[KEYWORD_COL].apply(parse_keyphrase_cell)
        frame["real_capec_list"] = frame[LABEL_COL].apply(parse_capec_cell)
        frame["model_label_list"] = frame["real_capec_list"].apply(
            lambda labels: labels if labels else [NO_ID_LABEL]
        )
        frames[split_name] = frame
        print(f"Loaded {method_name}/{split_name}: {len(frame):,} rows ({path})")
    return frames


def validate_identical_splits(method_frames):
    """Require identical ordered CVEs and truth for every keyword method."""
    baseline_name = next(iter(method_frames))
    baseline = method_frames[baseline_name]
    for method_name, frames in method_frames.items():
        for split_name in SPLIT_FILES:
            expected_ids = baseline[split_name][ID_COL].astype(str).tolist()
            actual_ids = frames[split_name][ID_COL].astype(str).tolist()
            if actual_ids != expected_ids:
                raise ValueError(
                    f"Ordered {split_name} CVEs differ between "
                    f"{baseline_name} and {method_name}"
                )
            if frames[split_name]["real_capec_list"].tolist() != baseline[
                split_name
            ]["real_capec_list"].tolist():
                raise ValueError(
                    f"{split_name} ground truth differs between "
                    f"{baseline_name} and {method_name}"
                )

        train_ids = set(frames["training"][ID_COL].astype(str))
        val_ids = set(frames["validation"][ID_COL].astype(str))
        overlap = train_ids & val_ids
        if overlap:
            raise ValueError(
                f"{method_name} has {len(overlap)} CVEs in both train and validation"
            )


def extract_catalog_labels(path):
    catalog = pd.read_csv(path, dtype=str)
    id_columns = [
        column for column in catalog.columns
        if str(column).strip().casefold() == "id"
    ]
    if len(id_columns) != 1:
        raise ValueError(f"Expected one ID column in {path}; found {id_columns}")
    labels = []
    for value in catalog[id_columns[0]].dropna():
        text = str(value).strip()
        match = re.fullmatch(r"(?:CAPEC-)?(\d+)(?:\.0)?", text, re.I)
        if match is None:
            raise ValueError(f"Invalid CAPEC catalogue ID: {value!r}")
        labels.append(f"CAPEC-{int(match.group(1))}")
    return sorted(set(labels), key=capec_sort_key)


def make_multihot(label_lists, label_to_idx):
    matrix = np.zeros((len(label_lists), len(label_to_idx)), dtype=np.float32)
    for row_index, labels in enumerate(label_lists):
        for label in labels:
            if label not in label_to_idx:
                raise ValueError(f"Label absent from shared label space: {label}")
            matrix[row_index, label_to_idx[label]] = 1.0
    return matrix


# ---------------------------------------------------------------------------
# Keyword-only MiniLM features
# ---------------------------------------------------------------------------

def encode_keyword_means(embedder, keyphrase_lists):
    """Mean-pool individually encoded keywords; empty rows stay all-zero."""
    embedding_dim = embedder.get_sentence_embedding_dimension()
    pooled = np.zeros((len(keyphrase_lists), embedding_dim), dtype=np.float32)
    counts = np.zeros(len(keyphrase_lists), dtype=np.float32)
    flat_phrases = []
    row_ids = []
    for row_id, phrases in enumerate(keyphrase_lists):
        for phrase in phrases:
            flat_phrases.append(phrase)
            row_ids.append(row_id)

    print(
        f"Encoding {len(flat_phrases):,} keyword occurrences for "
        f"{len(keyphrase_lists):,} rows"
    )
    for start in range(0, len(flat_phrases), EMBED_BATCH_SIZE):
        end = min(start + EMBED_BATCH_SIZE, len(flat_phrases))
        embeddings = embedder.encode(
            flat_phrases[start:end],
            batch_size=EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
        batch_rows = np.asarray(row_ids[start:end], dtype=np.int64)
        np.add.at(pooled, batch_rows, embeddings)
        np.add.at(counts, batch_rows, 1.0)
        if end == len(flat_phrases) or start == 0 or start % (EMBED_BATCH_SIZE * 100) == 0:
            print(f"  encoded {end:,}/{len(flat_phrases):,}")

    present = counts > 0
    pooled[present] /= counts[present, None]
    norms = np.linalg.norm(pooled[present], axis=1, keepdims=True)
    pooled[present] /= np.maximum(norms, 1e-12)
    return pooled, present


# ---------------------------------------------------------------------------
# Shared neural network and metrics
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class MultiLabelMLP(nn.Module):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(512, num_labels),
        )

    def forward(self, inputs):
        return self.net(inputs)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def postprocess_predictions(probabilities, threshold, no_id_index):
    predictions = (probabilities >= threshold).astype(np.int8)
    for row_index in range(len(predictions)):
        positives = np.flatnonzero(predictions[row_index])
        if len(positives) == 0:
            predictions[row_index, int(np.argmax(probabilities[row_index]))] = 1
        elif np.any(positives != no_id_index):
            predictions[row_index, no_id_index] = 0
    return predictions


def top_k_metrics(y_true, probabilities, no_id_index):
    real_indices = np.asarray(
        [index for index in range(y_true.shape[1]) if index != no_id_index],
        dtype=np.int64,
    )
    real_probabilities = probabilities[:, real_indices]
    ranking = real_indices[np.argsort(-real_probabilities, axis=1)]
    results = {}
    evaluated_rows = 0
    hit_totals = {k: 0.0 for k in TOP_KS}
    recall_totals = {k: 0.0 for k in TOP_KS}
    for row_index, true_row in enumerate(y_true):
        true_set = set(np.flatnonzero(true_row))
        true_set.discard(no_id_index)
        if not true_set:
            continue
        evaluated_rows += 1
        for k in TOP_KS:
            top_set = set(ranking[row_index, :k])
            recovered = len(true_set & top_set)
            hit_totals[k] += float(recovered > 0)
            recall_totals[k] += recovered / len(true_set)
    for k in TOP_KS:
        denominator = evaluated_rows or 1
        results[f"hit_at_{k}"] = hit_totals[k] / denominator
        results[f"recall_at_{k}"] = recall_totals[k] / denominator
    results["top_k_evaluated_real_capec_rows"] = int(evaluated_rows)
    return results


def calculate_metrics(y_true, probabilities, threshold, no_id_index):
    predictions = postprocess_predictions(
        probabilities, threshold, no_id_index
    )
    metrics = {
        "micro_precision": float(precision_score(
            y_true, predictions, average="micro", zero_division=0
        )),
        "micro_recall": float(recall_score(
            y_true, predictions, average="micro", zero_division=0
        )),
        "micro_f1": float(f1_score(
            y_true, predictions, average="micro", zero_division=0
        )),
        "macro_precision": float(precision_score(
            y_true, predictions, average="macro", zero_division=0
        )),
        "macro_recall": float(recall_score(
            y_true, predictions, average="macro", zero_division=0
        )),
        "macro_f1": float(f1_score(
            y_true, predictions, average="macro", zero_division=0
        )),
        "weighted_f1": float(f1_score(
            y_true, predictions, average="weighted", zero_division=0
        )),
        "sample_f1": float(f1_score(
            y_true, predictions, average="samples", zero_division=0
        )),
        "exact_match": float(accuracy_score(y_true, predictions)),
        "jaccard": float(jaccard_score(
            y_true, predictions, average="samples", zero_division=0
        )),
        "hamming_loss": float(hamming_loss(y_true, predictions)),
        "lrap": float(label_ranking_average_precision_score(
            y_true, probabilities
        )),
    }
    metrics.update(top_k_metrics(y_true, probabilities, no_id_index))
    return metrics, predictions


def select_threshold(y_true, probabilities, no_id_index):
    """Operational checkpoint rule: max validation Micro-F1, then lower t."""
    rows = []
    best = None
    for threshold in THRESHOLDS:
        predictions = postprocess_predictions(
            probabilities, threshold, no_id_index
        )
        true_positive = int(np.logical_and(y_true == 1, predictions == 1).sum())
        false_positive = int(np.logical_and(y_true == 0, predictions == 1).sum())
        false_negative = int(np.logical_and(y_true == 1, predictions == 0).sum())
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        f1_denominator = 2 * true_positive + false_positive + false_negative
        metrics = {
            "micro_precision": (
                true_positive / precision_denominator
                if precision_denominator else 0.0
            ),
            "micro_recall": (
                true_positive / recall_denominator
                if recall_denominator else 0.0
            ),
            "micro_f1": (
                2 * true_positive / f1_denominator
                if f1_denominator else 0.0
            ),
        }
        row = {"threshold": float(threshold), **metrics}
        rows.append(row)
        key = (metrics["micro_f1"], -float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    return best[1], best[2], rows


def make_loaders(x_train, y_train, x_val, y_val):
    generator = torch.Generator()
    generator.manual_seed(RANDOM_STATE)
    train_loader = DataLoader(
        EmbeddingDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(
        EmbeddingDataset(x_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0
    for features, labels in loader:
        features = features.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=AMP_ENABLED,
        ):
            logits = model(features)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * len(features)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_probabilities(model, loader):
    model.eval()
    probabilities = []
    labels = []
    for features, batch_labels in loader:
        logits = model(features.to(DEVICE, non_blocking=True))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.vstack(probabilities), np.vstack(labels)


def train_method(
    method_name,
    display_name,
    frames,
    x_train,
    x_val,
    train_has_keywords,
    val_has_keywords,
    y_train,
    y_val,
    all_labels,
    label_to_idx,
):
    print(f"\n{'=' * 72}\nTraining {display_name}\n{'=' * 72}")
    seed_everything(RANDOM_STATE)
    no_id_index = label_to_idx[NO_ID_LABEL]
    train_loader, val_loader = make_loaders(x_train, y_train, x_val, y_val)
    model = MultiLabelMLP(x_train.shape[1], len(all_labels)).to(DEVICE)

    positive_counts = y_train.sum(axis=0)
    negative_counts = len(y_train) - positive_counts
    positive_weights = np.ones(len(all_labels), dtype=np.float32)
    observed = positive_counts > 0
    positive_weights[observed] = (
        negative_counts[observed] / positive_counts[observed]
    )
    positive_weights = np.clip(positive_weights, 1.0, 50.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy(positive_weights).to(DEVICE)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=AMP_ENABLED)

    method_output_dir = OUTPUT_DIR / method_name
    method_output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler
        )
        val_probabilities, val_truth = predict_probabilities(model, val_loader)
        threshold, metrics, threshold_rows = select_threshold(
            val_truth, val_probabilities, no_id_index
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "selected_threshold": threshold,
            **metrics,
        })
        score_key = (metrics["micro_f1"], -threshold, -epoch)
        improved = best is None or score_key > best["score_key"]
        print(
            f"{display_name} epoch {epoch:02d}: loss={train_loss:.5f}, "
            f"threshold={threshold:.2f}, micro_f1={metrics['micro_f1']:.5f}"
        )
        if improved:
            best = {
                "score_key": score_key,
                "epoch": epoch,
                "threshold": threshold,
                "metrics": metrics,
                "threshold_rows": threshold_rows,
                "state_dict": copy.deepcopy(model.state_dict()),
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping {display_name} after epoch {epoch}")
                break

    if best is None:
        raise RuntimeError(f"No checkpoint selected for {method_name}")

    model.load_state_dict(best["state_dict"])
    final_probabilities, final_truth = predict_probabilities(model, val_loader)
    final_metrics, final_predictions = calculate_metrics(
        final_truth, final_probabilities, best["threshold"], no_id_index
    )
    if not np.isclose(final_metrics["micro_f1"], best["metrics"]["micro_f1"]):
        raise RuntimeError(f"Reloaded checkpoint metrics changed for {method_name}")

    checkpoint = {
        "model_state_dict": best["state_dict"],
        "method": method_name,
        "display_name": display_name,
        "best_epoch": best["epoch"],
        "selected_threshold": best["threshold"],
        "label_to_idx": label_to_idx,
        "all_labels": all_labels,
        "input_dim": int(x_train.shape[1]),
        "feature_representation": "mean-pooled keyword-only MiniLM embeddings",
        "embedding_model": EMBED_MODEL_NAME,
        "random_state": RANDOM_STATE,
    }
    torch.save(checkpoint, method_output_dir / "best_model.pt")
    pd.DataFrame(history).to_csv(
        method_output_dir / "training_history.csv", index=False
    )
    pd.DataFrame(best["threshold_rows"]).to_csv(
        method_output_dir / "best_epoch_threshold_sweep.csv", index=False
    )

    prediction_frame = pd.DataFrame({
        ID_COL: frames["validation"][ID_COL].astype(str),
        "true_labels": [
            json.dumps([all_labels[i] for i in np.flatnonzero(row)])
            for row in final_truth
        ],
        "predicted_labels": [
            json.dumps([all_labels[i] for i in np.flatnonzero(row)])
            for row in final_predictions
        ],
        "top_10_labels": [
            json.dumps([
                all_labels[i]
                for i in np.argsort(-probability_row)
                if i != no_id_index
            ][:10])
            for probability_row in final_probabilities
        ],
    })
    prediction_frame.to_csv(
        method_output_dir / "validation_predictions.csv", index=False
    )

    result = {
        "method": display_name,
        "method_key": method_name,
        **final_metrics,
        "best_epoch": int(best["epoch"]),
        "selected_threshold": float(best["threshold"]),
        "epochs_ran": int(len(history)),
        "training_rows": int(len(x_train)),
        "validation_rows": int(len(x_val)),
        "training_rows_with_keywords": int(train_has_keywords.sum()),
        "validation_rows_with_keywords": int(val_has_keywords.sum()),
    }
    payload = {
        "completed": True,
        "evaluation_contract_version": CONTRACT_VERSION,
        "selection_rule": (
            "Within each method, select threshold and checkpoint by maximum "
            "validation Micro-F1; ties prefer lower threshold then earlier epoch. "
            "This is an operational training rule, not a sole cross-method ranking."
        ),
        "test_evaluated": False,
        "result": result,
    }
    (method_output_dir / "validation_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    del model, optimizer, criterion, scaler, best
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print("Device:", DEVICE)
    print("CPU threads:", CPU_THREADS)
    print("Method inputs:", METHOD_OUTPUTS_DIR)
    print("Validation outputs:", OUTPUT_DIR)
    print("Test data will not be loaded or evaluated.")

    method_frames = {
        method_name: load_method_frames(method_name)
        for method_name in METHODS
    }
    validate_identical_splits(method_frames)

    catalog_labels = extract_catalog_labels(CAPEC_CATALOG_PATH)
    data_labels = {
        label
        for frames in method_frames.values()
        for frame in frames.values()
        for labels in frame["real_capec_list"]
        for label in labels
    }
    all_labels = sorted(
        set(catalog_labels) | data_labels, key=capec_sort_key
    ) + [NO_ID_LABEL]
    label_to_idx = {label: index for index, label in enumerate(all_labels)}

    baseline_frames = method_frames[next(iter(method_frames))]
    y_train = make_multihot(
        baseline_frames["training"]["model_label_list"].tolist(),
        label_to_idx,
    )
    y_val = make_multihot(
        baseline_frames["validation"]["model_label_list"].tolist(),
        label_to_idx,
    )
    print(f"Shared label space: {len(all_labels):,} labels")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    embedder = SentenceTransformer(EMBED_MODEL_NAME, device=str(DEVICE))
    if embedder.get_sentence_embedding_dimension() != 384:
        raise ValueError(
            "Expected all-MiniLM-L6-v2 to produce 384-dimensional embeddings"
        )

    comparison_rows = []
    for method_name, display_name in METHODS.items():
        frames = method_frames[method_name]
        print(f"\nBuilding keyword-only features for {display_name}")
        x_train, train_has_keywords = encode_keyword_means(
            embedder, frames["training"]["keyphrase_list"].tolist()
        )
        x_val, val_has_keywords = encode_keyword_means(
            embedder, frames["validation"]["keyphrase_list"].tolist()
        )
        comparison_rows.append(train_method(
            method_name=method_name,
            display_name=display_name,
            frames=frames,
            x_train=x_train,
            x_val=x_val,
            train_has_keywords=train_has_keywords,
            val_has_keywords=val_has_keywords,
            y_train=y_train,
            y_val=y_val,
            all_labels=all_labels,
            label_to_idx=label_to_idx,
        ))
        del x_train, x_val, train_has_keywords, val_has_keywords

    columns = [
        "method", "method_key",
        "micro_precision", "micro_recall", "micro_f1",
        "macro_precision", "macro_recall", "macro_f1",
        "weighted_f1", "sample_f1", "exact_match", "jaccard",
        "hamming_loss", "lrap",
        "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10",
        "recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10",
        "best_epoch", "selected_threshold", "epochs_ran",
        "top_k_evaluated_real_capec_rows", "training_rows",
        "validation_rows", "training_rows_with_keywords",
        "validation_rows_with_keywords",
    ]
    comparison = pd.DataFrame(comparison_rows)[columns]
    comparison_path = OUTPUT_DIR / "validation_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    config = {
        "evaluation_contract_version": CONTRACT_VERSION,
        "purpose": "keyword-method validation comparison",
        "test_evaluated": False,
        "method_order": list(METHODS),
        "feature_representation": "keywords only; individual MiniLM embeddings mean-pooled per CVE",
        "embedding_model": EMBED_MODEL_NAME,
        "architecture": ["Linear(384,512)", "ReLU", "Dropout(0.30)", "Linear(512,512)", "ReLU", "Dropout(0.30)", f"Linear(512,{len(all_labels)})"],
        "loss": "BCEWithLogitsLoss with train-derived pos_weight clipped to [1,50]",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "maximum_epochs": EPOCHS,
        "patience": PATIENCE,
        "random_state": RANDOM_STATE,
        "threshold_grid": list(THRESHOLDS),
        "checkpoint_and_threshold_rule": "maximum validation Micro-F1; ties prefer lower threshold then earlier epoch",
        "cross_method_rule": "comparison is intentionally unranked; inspect the full validation metric profile before declaring a primary method-selection criterion",
        "top_k_rule": "exclude CAPEC-noID and evaluate only rows with at least one real CAPEC",
        "macro_rule": "macro metrics cover the full shared canonical label space, including labels with zero validation support",
        "label_count": len(all_labels),
        "no_id_label": NO_ID_LABEL,
        "metrics": columns[2:24],
        "device": str(DEVICE),
        "cpu_threads": CPU_THREADS,
    }
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print(f"\nValidation comparison written to: {comparison_path}")
    print(comparison[[
        "method", "micro_f1", "macro_f1", "weighted_f1", "sample_f1",
        "exact_match", "hit_at_5", "best_epoch", "selected_threshold",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
