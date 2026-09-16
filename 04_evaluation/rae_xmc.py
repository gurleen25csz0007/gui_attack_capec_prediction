#!/usr/bin/env python3
"""
Paper-faithful RAE-XMC model for CVE -> CAPEC prediction.

Paper
-----
Y.-S. Wang et al., "Retrieval-augmented Encoders for Extreme Multi-label
Text Classification", arXiv:2502.10615 (2025).

RAE-XMC is a shared dual encoder, not an MLP classifier. It learns normalized
input/label embeddings with the paper's decoupled contrastive objective. Its
knowledge memory is the union of training-instance keys X and label-text keys
Z. The associated values are [lambda * Y_train; (1-lambda) * I]. At inference,
the top-b keys are retrieved from the unified memory, their similarities receive
one joint temperature Softmax, and their sparse values are aggregated:

    p_hat = Softmax(q K^T / tau) V.

Faithfulness and explicit adaptations
--------------------------------------
* The original DistilBERT shared encoder is replaced by the requested
  sentence-transformers/all-MiniLM-L6-v2. Mean pooling and L2 normalization
  are retained exactly as specified by the paper.
* The paper samples one positive label and m mined hard-negative labels per
  input and adds same-tower in-batch input negatives that share no positive
  labels. This file implements that sampled form of Eq. (10).
* The paper refreshes hard negatives by optimization step. This standalone,
  epoch-oriented implementation refreshes them at the configurable epoch
  interval HARD_NEGATIVE_REFRESH_EPOCHS. The approximation is explicit and
  keeps mining bounded and reproducible.
* The prediction vocabulary comes from stage 01's complete 3000_capec.csv
  catalog, supplemented by relationship and fixed-split IDs. CAPEC label text
  uses the local authoritative name when available; otherwise its literal
  identifier is used. The source is recorded in label_info.csv.
* The paper fixes tau=0.04 and uses HNSW top-b=200. Those are defaults here.
  Lambda and the multi-label threshold are selected using validation labels
  only. The held-out test labels are touched only after checkpoint selection
  and calibration are complete.
* RAE-XMC has no classifier head and no BCE loss. HEAD_LR is exposed only to
  satisfy experiment-config compatibility and is deliberately unused.

The public paper and OpenReview submission describe code in supplementary
material, but as of 2026-09-08 no official public source repository is linked
from arXiv/Papers with Code and an exact-title GitHub search returns none. This
file therefore implements the published equations and algorithms directly and
does not claim to be copied from an official repository.

Examples
--------
    python rae_xmc.py
    python rae_xmc.py --smoke-test
    python rae_xmc.py --inference-run outputs/RAE_XMC \
        --text "A remote attacker can inject SQL through the id parameter."
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import gc
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# =============================================================================
# 0. EDITABLE CONFIGURATION
# =============================================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256
EPOCHS = 50
PATIENCE = 5
ENCODER_LR = 2e-5
HEAD_LR = 0.0  # Intentionally unused: published RAE-XMC has no classifier head.
WEIGHT_DECAY = 1e-2
MAX_LENGTH = 160
WARMUP_RATIO = 0.10
RETRIEVAL_K = 200
CONTRASTIVE_TEMPERATURE = 0.04
LOSS_WEIGHTS = {"decoupled_contrastive": 1.0}
MAX_GRAD_NORM = 1.0
USE_AMP = True
SEED = 42

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    label_ranking_average_precision_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from evaluation_data import DATASET_PATH, load_fixed_splits
from evaluation_paths import OUTPUT_ROOT

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


# Data and methodology configuration.
SCRIPT_DIR = Path(__file__).resolve().parent
METHODOLOGY_DIR = SCRIPT_DIR.parent
GROUND_TRUTH_DIR = METHODOLOGY_DIR / "01_cwe_capec_ground_truth"

CAPEC_REL_PATH = GROUND_TRUTH_DIR / "capec_relationships.csv"
CAPEC_CATALOG_PATH = GROUND_TRUTH_DIR / "3000_capec.csv"

# Fixed names make downstream model paths stable across Slurm jobs.
OUTPUT_DIR = OUTPUT_ROOT / "RAE_XMC"
KEYWORD_OUTPUT_DIR = OUTPUT_ROOT / "RAE_XMC_keywords"
SMOKE_OUTPUT_DIR = OUTPUT_ROOT / "RAE_XMC_smoke_test"
KEYWORD_SMOKE_OUTPUT_DIR = OUTPUT_ROOT / "RAE_XMC_keywords_smoke_test"

DESC_COL = "cleaned_description"
KEYPHRASE_COL = "keyphrases"
CAPEC_COL = "capec_id"
CVE_COL = "cve_id"
NO_ID_LABEL = "CAPEC-noID"
USE_KEYPHRASES = False
KEYPHRASE_SEPARATOR = " [SEP] keyphrases: "

NUM_WORKERS = 0
PIN_MEMORY = True
GRADIENT_CHECKPOINTING = False
TOKENIZE_BATCH_SIZE = 4096
EMBED_BATCH_SIZE = 512

HARD_NEGATIVES_PER_INPUT = 2  # paper notation: m
HARD_NEGATIVE_TOPK = 50
HARD_NEGATIVE_REFRESH_EPOCHS = 1
USE_HARD_NEGATIVE_MINING = True

HNSW_M = 64
HNSW_EF_CONSTRUCTION = 500
HNSW_EF_SEARCH = 300
NUMPY_INDEX_KEY_CHUNK = 32768

# The paper fixes tau for train/inference consistency. Add values only for an
# explicitly reported calibration ablation; all choices remain validation-only.
INFERENCE_TEMPERATURE_GRID = (0.04,)
LAMBDA_GRID = tuple(round(step * 0.02, 2) for step in range(51))
THRESHOLD_GRID = np.array([
    0.75,
    0.80,
    0.85,
    0.90,
    0.92,
    0.94,
    0.96,
    0.98,
    0.99,
])
RANKING_KS = (1, 3, 5, 10)


@dataclass(frozen=True)
class RunConfig:
    model_name: str = MODEL_NAME
    batch_size: int = BATCH_SIZE
    eval_batch_size: int = EVAL_BATCH_SIZE
    epochs: int = EPOCHS
    patience: int = PATIENCE
    encoder_lr: float = ENCODER_LR
    head_lr: float = HEAD_LR
    weight_decay: float = WEIGHT_DECAY
    max_length: int = MAX_LENGTH
    warmup_ratio: float = WARMUP_RATIO
    retrieval_k: int = RETRIEVAL_K
    contrastive_temperature: float = CONTRASTIVE_TEMPERATURE
    loss_weights: Optional[Dict[str, float]] = None
    max_grad_norm: float = MAX_GRAD_NORM
    use_amp: bool = USE_AMP
    seed: int = SEED
    use_keyphrases: bool = USE_KEYPHRASES
    hard_negatives_per_input: int = HARD_NEGATIVES_PER_INPUT
    hard_negative_topk: int = HARD_NEGATIVE_TOPK
    hard_negative_refresh_epochs: int = HARD_NEGATIVE_REFRESH_EPOCHS
    use_hard_negative_mining: bool = USE_HARD_NEGATIVE_MINING
    hnsw_m: int = HNSW_M
    hnsw_ef_construction: int = HNSW_EF_CONSTRUCTION
    hnsw_ef_search: int = HNSW_EF_SEARCH
    num_workers: int = NUM_WORKERS
    gradient_checkpointing: bool = GRADIENT_CHECKPOINTING
    paper_arxiv: str = "2502.10615"
    official_public_repository: Optional[str] = None

    def __post_init__(self) -> None:
        if self.loss_weights is None:
            object.__setattr__(self, "loss_weights", dict(LOSS_WEIGHTS))


# =============================================================================
# 1. GENERAL UTILITIES
# =============================================================================


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(value), handle, indent=2, sort_keys=True)


def load_torch_checkpoint(path: Path, map_location: Any) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def package_versions() -> Dict[str, Optional[str]]:
    names = ["torch", "transformers", "numpy", "pandas", "scikit-learn", "faiss-cpu", "faiss-gpu"]
    versions: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    versions["python"] = platform.python_version()
    versions["platform"] = platform.platform()
    versions["cuda_runtime"] = torch.version.cuda
    versions["cudnn"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
    return versions


def autocast_context(enabled: bool):
    if enabled and torch.cuda.is_available():
        return torch.cuda.amp.autocast(enabled=True)
    return contextlib.nullcontext()


def make_grad_scaler(enabled: bool):
    return torch.cuda.amp.GradScaler(enabled=(enabled and torch.cuda.is_available()))


# =============================================================================
# 2. DATA PARSING AND LEAKAGE CHECKS
# =============================================================================


def parse_capec_cell(value: Any) -> List[str]:
    """Robust CAPEC parsing retained from the supplied baseline."""
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return []
    compact = text.lower().replace("_", "").replace(" ", "")
    if compact in {"capec-noid", "noid", "nocapec", "capecnone"}:
        return []

    labels: List[str] = []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            items = list(parsed)
        else:
            items = [parsed]
        for item in items:
            item_text = str(item).strip()
            item_compact = item_text.lower().replace("_", "").replace(" ", "")
            if item_compact in {"capec-noid", "noid", "nocapec", "capecnone"}:
                continue
            labels.extend(x.upper() for x in re.findall(r"CAPEC-\d+", item_text, re.I))
        return sorted(set(labels))
    except Exception:
        return sorted(set(x.upper() for x in re.findall(r"CAPEC-\d+", text, re.I)))


def parse_keyphrase_cell(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return []
    try:
        parsed = ast.literal_eval(text)
        items = list(parsed) if isinstance(parsed, (list, tuple, set)) else [parsed]
    except Exception:
        items = re.split(r"[;|\n,]+", text)
    result: List[str] = []
    seen = set()
    for item in items:
        cleaned = re.sub(r"\s+", " ", str(item)).strip()
        key = cleaned.lower()
        if cleaned and key not in {"nan", "none", "null", "[]"} and key not in seen:
            result.append(cleaned)
            seen.add(key)
    return result


def validate_and_prepare_split(df: pd.DataFrame, split_name: str, use_keyphrases: bool) -> Tuple[pd.DataFrame, int]:
    missing = [column for column in (DESC_COL, CAPEC_COL) if column not in df.columns]
    if missing:
        raise ValueError(f"{split_name} is missing required columns: {missing}")
    df = df.copy().reset_index(drop=True)
    if KEYPHRASE_COL not in df.columns:
        df[KEYPHRASE_COL] = ""
    for column in (DESC_COL, KEYPHRASE_COL, CAPEC_COL):
        df[column] = df[column].fillna("").astype(str)
    empty_description = df[DESC_COL].str.strip().eq("")
    dropped = int(empty_description.sum())
    if dropped:
        print(f"{split_name}: dropping {dropped} rows with empty {DESC_COL}.")
    df = df.loc[~empty_description].reset_index(drop=True)
    df["capec_real_list"] = df[CAPEC_COL].map(parse_capec_cell)
    df["capec_label_list"] = df["capec_real_list"].map(lambda x: x if x else [NO_ID_LABEL])
    df["keyphrase_list"] = df[KEYPHRASE_COL].map(parse_keyphrase_cell)
    if use_keyphrases:
        df["model_text"] = [
            description + (KEYPHRASE_SEPARATOR + " ; ".join(kps) if kps else "")
            for description, kps in zip(df[DESC_COL].astype(str), df["keyphrase_list"])
        ]
    else:
        df["model_text"] = df[DESC_COL].astype(str)
    return df, dropped


def split_id_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    def ids(df: pd.DataFrame) -> set:
        if CVE_COL not in df.columns:
            return set()
        return {x for x in df[CVE_COL].fillna("").astype(str).str.strip() if x}

    train_ids, val_ids, test_ids = ids(train_df), ids(val_df), ids(test_df)
    pairs = {
        "train_validation": sorted(train_ids & val_ids),
        "train_test": sorted(train_ids & test_ids),
        "validation_test": sorted(val_ids & test_ids),
    }
    report = {
        "cve_column_present": all(CVE_COL in df.columns for df in (train_df, val_df, test_df)),
        "unique_ids": {"train": len(train_ids), "validation": len(val_ids), "test": len(test_ids)},
        "overlap_counts": {name: len(values) for name, values in pairs.items()},
        "overlap_ids": pairs,
        "warning": "Overlapping CVE IDs are reported but rows are never moved between fixed splits.",
    }
    if any(report["overlap_counts"].values()):
        print("WARNING: overlapping CVE IDs detected:", report["overlap_counts"])
    else:
        print("No overlapping non-empty CVE IDs detected across fixed splits.")
    return report


def capec_ids_and_names(path: Path) -> Tuple[List[str], Dict[str, str]]:
    if not path.exists():
        print(f"WARNING: {path.name} not found; label space will come from fixed split files only.")
        return [], {}
    rel = pd.read_csv(path)
    lower_to_original = {column.lower(): column for column in rel.columns}
    id_column = next((lower_to_original[k] for k in ("capec", "capec_id", "id") if k in lower_to_original), None)
    if id_column is None:
        raise ValueError(f"Cannot find the CAPEC ID column in {path}")
    capec_ids = set()
    for value in rel[id_column].dropna():
        text = str(value).strip()
        capec_ids.update(
            item.upper() for item in re.findall(r"CAPEC-\d+", text, re.I)
        )
        if re.fullmatch(r"\d+(?:\.0)?", text):
            capec_ids.add(f"CAPEC-{int(float(text))}")

    names: Dict[str, str] = {}
    name_column = next((lower_to_original[k] for k in ("name", "title") if k in lower_to_original), None)
    if id_column and name_column:
        for capec_value, name_value in zip(rel[id_column], rel[name_column]):
            capec_text = str(capec_value).strip()
            found = re.findall(r"CAPEC-\d+", capec_text, re.I)
            if not found and re.fullmatch(r"\d+(?:\.0)?", capec_text):
                found = [f"CAPEC-{int(float(capec_text))}"]
            name = re.sub(r"\s+", " ", str(name_value)).strip()
            if found and name and name.lower() not in {"nan", "none", "null"}:
                names.setdefault(found[0].upper(), name)
    return sorted(capec_ids), names


def make_multihot(label_lists: Sequence[Sequence[str]], label_to_idx: Mapping[str, int]) -> np.ndarray:
    matrix = np.zeros((len(label_lists), len(label_to_idx)), dtype=np.uint8)
    no_id_index = label_to_idx[NO_ID_LABEL]
    for row, labels in enumerate(label_lists):
        indices = [label_to_idx[label] for label in labels if label in label_to_idx]
        if not indices:
            indices = [no_id_index]
        matrix[row, indices] = 1
    return matrix


def build_label_space(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    relationship_path: Path,
    catalog_path: Path,
) -> Tuple[
    List[str], Dict[str, int], Dict[int, str], List[str], List[str], List[str],
    Dict[str, str], Dict[str, str],
]:
    relationship_labels, relationship_names = capec_ids_and_names(relationship_path)
    catalog_labels, catalog_names = capec_ids_and_names(catalog_path)
    data_labels = sorted({
        label
        for frame in (train_df, val_df, test_df)
        for labels in frame["capec_real_list"]
        for label in labels
    })
    real_labels = sorted(
        set(catalog_labels) | set(relationship_labels) | set(data_labels)
    )
    labels = real_labels + [NO_ID_LABEL]
    label_to_idx = {label: index for index, label in enumerate(labels)}
    idx_to_label = {index: label for label, index in label_to_idx.items()}
    authoritative_names = {**catalog_names, **relationship_names}
    name_sources = {
        **{label: "3000_capec.name" for label in catalog_names},
        **{label: "capec_relationships.name" for label in relationship_names},
    }
    return (
        labels, label_to_idx, idx_to_label, relationship_labels, catalog_labels,
        data_labels, authoritative_names, name_sources,
    )


def build_label_texts(
    labels: Sequence[str],
    authoritative_names: Mapping[str, str],
    name_sources: Mapping[str, str],
) -> Tuple[List[str], List[str]]:
    texts, sources = [], []
    for label in labels:
        if label in authoritative_names:
            texts.append(f"{label}: {authoritative_names[label]}")
            sources.append(name_sources[label])
        else:
            texts.append(label)
            sources.append("identifier_fallback")
    return texts, sources


# =============================================================================
# 3. TOKENIZATION, DATASET, AND SHARED ENCODER
# =============================================================================


def tokenize_texts(tokenizer: Any, texts: Sequence[str], max_length: int, name: str) -> Dict[str, torch.Tensor]:
    ids, masks = [], []
    print(f"Tokenizing {name}: {len(texts):,} texts")
    for start in range(0, len(texts), TOKENIZE_BATCH_SIZE):
        batch = [str(x) for x in texts[start:start + TOKENIZE_BATCH_SIZE]]
        encoded = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids.append(encoded["input_ids"])
        masks.append(encoded["attention_mask"])
    if not ids:
        raise ValueError(f"Cannot tokenize empty {name} collection.")
    return {"input_ids": torch.cat(ids), "attention_mask": torch.cat(masks)}


class RAEXMCDataset(Dataset):
    def __init__(
        self,
        tokens: Mapping[str, torch.Tensor],
        labels: np.ndarray,
        known_positive_labels: Optional[np.ndarray] = None,
    ):
        self.input_ids = tokens["input_ids"]
        self.attention_mask = tokens["attention_mask"]
        self.labels = torch.from_numpy(labels.astype(np.uint8, copy=False))
        known = labels if known_positive_labels is None else known_positive_labels
        if known.shape != labels.shape:
            raise ValueError("known_positive_labels must have the same shape as labels")
        self.known_positive_labels = torch.from_numpy(known.astype(np.uint8, copy=False))
        self.hard_negative_ids = torch.empty((len(labels), 0), dtype=torch.long)

    def set_hard_negatives(self, values: np.ndarray) -> None:
        if values.shape[0] != len(self):
            raise ValueError("Hard-negative row count does not match dataset.")
        self.hard_negative_ids = torch.from_numpy(values.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "row_index": torch.tensor(index, dtype=torch.long),
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "labels": self.labels[index],
            "known_positive_labels": self.known_positive_labels[index],
            "hard_negative_ids": self.hard_negative_ids[index],
        }


class TokenOnlyDataset(Dataset):
    def __init__(self, tokens: Mapping[str, torch.Tensor]):
        self.input_ids = tokens["input_ids"]
        self.attention_mask = tokens["attention_mask"]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[index], "attention_mask": self.attention_mask[index]}


class RAEXMCEncoder(nn.Module):
    """Paper's shared Transformer encoder with mean pooling and L2 normalization."""

    def __init__(self, model_name: Optional[str] = None, config_path: Optional[Path] = None):
        super().__init__()
        if config_path is not None:
            encoder_config = AutoConfig.from_pretrained(str(config_path), local_files_only=True)
            self.encoder = AutoModel.from_config(encoder_config)
        elif model_name is not None:
            self.encoder = AutoModel.from_pretrained(model_name)
        else:
            raise ValueError("model_name or config_path is required")

    @property
    def embedding_dim(self) -> int:
        return int(self.encoder.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
        return F.normalize(pooled, p=2, dim=1)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, config: RunConfig) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=(PIN_MEMORY and torch.cuda.is_available()),
        generator=generator if shuffle else None,
    )


@torch.no_grad()
def encode_dataset(
    model: RAEXMCEncoder,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    config: RunConfig,
) -> np.ndarray:
    loader = make_loader(dataset, batch_size, False, config)
    model.eval()
    result = np.empty((len(dataset), model.embedding_dim), dtype=np.float32)
    offset = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with autocast_context(config.use_amp):
            embeddings = model(input_ids, attention_mask)
        values = embeddings.float().cpu().numpy()
        result[offset:offset + len(values)] = values
        offset += len(values)
    return result


# =============================================================================
# 4. PAPER LOSS: DECOUPLED LABEL SOFTMAX + SAME-TOWER NEGATIVES
# =============================================================================


def random_negative_matrix(
    known_positive_y: np.ndarray,
    count: int,
    seed: int,
    eligible_label_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    num_labels = known_positive_y.shape[1]
    negatives = np.empty((len(known_positive_y), count), dtype=np.int64)
    all_ids = np.arange(num_labels)
    eligible = (
        np.ones(num_labels, dtype=bool)
        if eligible_label_mask is None
        else np.asarray(eligible_label_mask, dtype=bool)
    )
    if eligible.shape != (num_labels,):
        raise ValueError("eligible_label_mask has the wrong shape")
    for row in range(len(known_positive_y)):
        valid = all_ids[eligible & (known_positive_y[row] == 0)]
        if len(valid) == 0:
            raise ValueError("A training row has no eligible confirmed-negative label.")
        negatives[row] = rng.choice(valid, size=count, replace=len(valid) < count)
    return negatives


def decoupled_contrastive_loss(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    batch_labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Sampled Eq. (10). candidate_embeddings is [B, 1+m, D], with the sampled
    positive at column zero. Same-tower batch inputs are negatives only when
    their label sets do not overlap; self-pairs are excluded.
    """
    if candidate_embeddings.ndim != 3 or candidate_embeddings.size(1) < 2:
        raise ValueError("At least one positive and one negative label are required.")
    label_logits = torch.einsum("bd,bmd->bm", query_embeddings, candidate_embeddings) / temperature
    positive_logits = label_logits[:, 0]
    negative_label_logits = label_logits[:, 1:]

    instance_logits = query_embeddings @ query_embeddings.T / temperature
    labels_float = batch_labels.float()
    shares_label = (labels_float @ labels_float.T) > 0
    eye = torch.eye(len(query_embeddings), dtype=torch.bool, device=query_embeddings.device)
    valid_instance_negative = ~(shares_label | eye)
    instance_logits = instance_logits.masked_fill(~valid_instance_negative, float("-inf"))

    denominator_terms = torch.cat(
        [positive_logits.unsqueeze(1), negative_label_logits, instance_logits], dim=1
    )
    return (torch.logsumexp(denominator_terms, dim=1) - positive_logits).mean()


def encode_candidate_labels(
    model: RAEXMCEncoder,
    label_tokens: Mapping[str, torch.Tensor],
    candidate_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    flat = candidate_ids.reshape(-1)
    unique_ids, inverse = torch.unique(flat, sorted=True, return_inverse=True)
    cpu_ids = unique_ids.detach().cpu()
    input_ids = label_tokens["input_ids"].index_select(0, cpu_ids).to(device, non_blocking=True)
    attention_mask = label_tokens["attention_mask"].index_select(0, cpu_ids).to(device, non_blocking=True)
    unique_embeddings = model(input_ids, attention_mask)
    return unique_embeddings[inverse].view(candidate_ids.size(0), candidate_ids.size(1), -1)


def train_one_epoch(
    model: RAEXMCEncoder,
    loader: DataLoader,
    label_tokens: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    device: torch.device,
    config: RunConfig,
    positive_sampling_weights: Optional[torch.Tensor] = None,
) -> float:
    model.train()
    total_loss = 0.0
    seen = 0
    weight = float(config.loss_weights["decoupled_contrastive"])
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        known_positive_labels = batch["known_positive_labels"].to(device, non_blocking=True)
        hard_negative_ids = batch["hard_negative_ids"].to(device, non_blocking=True)
        positive_probabilities = labels.float()
        if positive_sampling_weights is not None:
            positive_probabilities = positive_probabilities * positive_sampling_weights.to(
                device=device, dtype=positive_probabilities.dtype
            )
        positive_ids = torch.multinomial(positive_probabilities, num_samples=1)
        candidate_ids = torch.cat([positive_ids, hard_negative_ids], dim=1)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(config.use_amp):
            query_embeddings = model(input_ids, attention_mask)
            candidate_embeddings = encode_candidate_labels(model, label_tokens, candidate_ids, device)
            loss = weight * decoupled_contrastive_loss(
                query_embeddings,
                candidate_embeddings,
                known_positive_labels,
                config.contrastive_temperature,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {step}: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = len(input_ids)
        total_loss += float(loss.item()) * batch_size
        seen += batch_size
        if step % 200 == 0:
            message = f"  step {step:,}/{len(loader):,} loss={loss.item():.5f}"
            if torch.cuda.is_available():
                message += f" peak_vram={torch.cuda.max_memory_allocated() / 1024**3:.2f}GB"
            print(message)
    return total_loss / max(seen, 1)


# =============================================================================
# 5. BOUNDED ANN INDEX AND HARD-NEGATIVE MINING
# =============================================================================


class SearchIndex:
    backend = "abstract"

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError


class FaissHNSWIndex(SearchIndex):
    backend = "faiss_hnsw_cpu"

    def __init__(self, embeddings: np.ndarray, config: RunConfig):
        if faiss is None:
            raise RuntimeError("faiss is unavailable")
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index = faiss.index_factory(
            vectors.shape[1], f"HNSW{config.hnsw_m},Flat", faiss.METRIC_INNER_PRODUCT
        )
        self.index.hnsw.efConstruction = config.hnsw_ef_construction
        self.index.hnsw.efSearch = config.hnsw_ef_search
        self.index.add(vectors)

    @classmethod
    def from_file(cls, path: Path, config: RunConfig) -> "FaissHNSWIndex":
        if faiss is None:
            raise RuntimeError("faiss is unavailable")
        obj = cls.__new__(cls)
        obj.index = faiss.read_index(str(path))
        if hasattr(obj.index, "hnsw"):
            obj.index.hnsw.efSearch = config.hnsw_ef_search
        return obj

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = min(k, int(self.index.ntotal))
        scores, indices = self.index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
        return scores.astype(np.float32), indices.astype(np.int64)

    def save(self, path: Path) -> None:
        faiss.write_index(self.index, str(path))


class NumpyChunkedIPIndex(SearchIndex):
    backend = "numpy_chunked_exact_ip"

    def __init__(self, embeddings: np.ndarray, key_chunk: int = NUMPY_INDEX_KEY_CHUNK):
        self.embeddings = embeddings
        self.key_chunk = key_chunk

    def search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(queries, dtype=np.float32)
        total = len(self.embeddings)
        k = min(k, total)
        best_scores = np.full((len(queries), k), -np.inf, dtype=np.float32)
        best_indices = np.full((len(queries), k), -1, dtype=np.int64)
        for start in range(0, total, self.key_chunk):
            end = min(start + self.key_chunk, total)
            block = np.asarray(self.embeddings[start:end], dtype=np.float32)
            scores = queries @ block.T
            indices = np.broadcast_to(np.arange(start, end, dtype=np.int64), scores.shape)
            combined_scores = np.concatenate([best_scores, scores], axis=1)
            combined_indices = np.concatenate([best_indices, indices], axis=1)
            selected = np.argpartition(combined_scores, -k, axis=1)[:, -k:]
            best_scores = np.take_along_axis(combined_scores, selected, axis=1)
            best_indices = np.take_along_axis(combined_indices, selected, axis=1)
        order = np.argsort(-best_scores, axis=1)
        return (
            np.take_along_axis(best_scores, order, axis=1),
            np.take_along_axis(best_indices, order, axis=1),
        )

    def save(self, path: Path) -> None:
        # Embeddings are always persisted separately; this marker documents the fallback.
        save_json(path.with_suffix(".json"), {"backend": self.backend, "index_file": None})


def build_search_index(embeddings: np.ndarray, config: RunConfig) -> SearchIndex:
    if faiss is not None:
        try:
            return FaissHNSWIndex(embeddings, config)
        except Exception as error:
            print(f"WARNING: FAISS HNSW construction failed ({error}); using bounded NumPy exact search.")
    return NumpyChunkedIPIndex(embeddings)


def mine_hard_negatives(
    model: RAEXMCEncoder,
    train_dataset: RAEXMCDataset,
    label_embeddings: np.ndarray,
    label_embedding_ids: np.ndarray,
    known_positive_y: np.ndarray,
    eligible_label_mask: np.ndarray,
    device: torch.device,
    config: RunConfig,
    epoch: int,
) -> np.ndarray:
    print("Mining hard-negative labels from the current encoder...")
    label_index = build_search_index(label_embeddings, config)
    loader = make_loader(TokenOnlyDataset({
        "input_ids": train_dataset.input_ids,
        "attention_mask": train_dataset.attention_mask,
    }), config.eval_batch_size, False, config)
    result = np.empty((len(train_dataset), config.hard_negatives_per_input), dtype=np.int64)
    rng = np.random.default_rng(config.seed + 1000 + epoch)
    offset = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with autocast_context(config.use_amp):
                query = model(input_ids, attention_mask).float().cpu().numpy()
            _, candidates = label_index.search(
                query, min(config.hard_negative_topk, len(label_embedding_ids))
            )
            for local_row in range(len(query)):
                row = offset + local_row
                global_candidates = label_embedding_ids[candidates[local_row]]
                chosen = [
                    int(idx) for idx in global_candidates.tolist()
                    if eligible_label_mask[idx] and known_positive_y[row, idx] == 0
                ]
                if len(chosen) < config.hard_negatives_per_input:
                    valid = np.flatnonzero(eligible_label_mask & (known_positive_y[row] == 0))
                    if not len(valid):
                        raise ValueError("A training row has no eligible hard-negative label.")
                    extra = rng.choice(
                        valid,
                        size=config.hard_negatives_per_input - len(chosen),
                        replace=len(valid) < config.hard_negatives_per_input - len(chosen),
                    ).tolist()
                    chosen.extend(extra)
                result[row] = chosen[:config.hard_negatives_per_input]
            offset += len(query)
    return result


# =============================================================================
# 6. RAE-XMC UNIFIED MEMORY AND PAPER INFERENCE
# =============================================================================


def label_rows_to_ragged(y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(y_train) + 1, dtype=np.int64)
    rows: List[np.ndarray] = []
    for row, values in enumerate(y_train):
        indices = np.flatnonzero(values).astype(np.int32)
        rows.append(indices)
        offsets[row + 1] = offsets[row] + len(indices)
    flat = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
    return offsets, flat


def build_unified_memory(
    model: RAEXMCEncoder,
    train_dataset: RAEXMCDataset,
    label_dataset: TokenOnlyDataset,
    device: torch.device,
    config: RunConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build K=[X;Z] from the current/final encoder. No validation/test rows enter K."""
    print("Encoding training-instance keys for unified memory...")
    train_embeddings = encode_dataset(model, TokenOnlyDataset({
        "input_ids": train_dataset.input_ids,
        "attention_mask": train_dataset.attention_mask,
    }), device, EMBED_BATCH_SIZE, config)
    print("Encoding label-text keys for unified memory...")
    label_embeddings = encode_dataset(model, label_dataset, device, EMBED_BATCH_SIZE, config)
    memory = np.ascontiguousarray(np.vstack([train_embeddings, label_embeddings]), dtype=np.float32)
    return memory, train_embeddings, label_embeddings


@torch.no_grad()
def retrieve_dataset(
    model: RAEXMCEncoder,
    dataset: TokenOnlyDataset,
    index: SearchIndex,
    device: torch.device,
    config: RunConfig,
    retrieval_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = make_loader(dataset, config.eval_batch_size, False, config)
    all_scores, all_indices = [], []
    model.eval()
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with autocast_context(config.use_amp):
            query = model(input_ids, attention_mask).float().cpu().numpy()
        scores, indices = index.search(query, retrieval_k)
        all_scores.append(scores)
        all_indices.append(indices)
    return np.vstack(all_scores), np.vstack(all_indices)


def stable_softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    scaled = scores.astype(np.float64) / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_scores = np.exp(scaled)
    return (exp_scores / exp_scores.sum(axis=1, keepdims=True)).astype(np.float32)


def aggregate_rae_components(
    neighbor_scores: np.ndarray,
    neighbor_indices: np.ndarray,
    num_train: int,
    num_labels: int,
    train_label_offsets: np.ndarray,
    train_label_indices: np.ndarray,
    temperature: float,
    return_candidates: bool = False,
    label_memory_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Sparse evaluation of Softmax(qK^T/tau)[lambda*Y; (1-lambda)*I].
    Returns the unscaled instance and label contributions so lambda can be
    selected on validation without rerunning ANN search.
    """
    weights = stable_softmax(neighbor_scores, temperature)
    instance_scores = np.zeros((len(neighbor_scores), num_labels), dtype=np.float32)
    label_scores = np.zeros_like(instance_scores)
    candidates = np.zeros_like(instance_scores, dtype=bool) if return_candidates else None
    for row in range(len(neighbor_scores)):
        for weight, memory_index in zip(weights[row], neighbor_indices[row]):
            memory_index = int(memory_index)
            if memory_index < 0:
                continue
            if memory_index < num_train:
                start = int(train_label_offsets[memory_index])
                end = int(train_label_offsets[memory_index + 1])
                labels = train_label_indices[start:end]
                instance_scores[row, labels] += weight
                if candidates is not None:
                    candidates[row, labels] = True
            else:
                local_label_index = memory_index - num_train
                label_index = (
                    int(label_memory_ids[local_label_index])
                    if label_memory_ids is not None and local_label_index < len(label_memory_ids)
                    else local_label_index
                )
                if local_label_index >= 0 and label_index < num_labels:
                    label_scores[row, label_index] += weight
                    if candidates is not None:
                        candidates[row, label_index] = True
    return instance_scores, label_scores, candidates


def combine_rae_scores(instance_scores: np.ndarray, label_scores: np.ndarray, lambda_value: float) -> np.ndarray:
    return lambda_value * instance_scores + (1.0 - lambda_value) * label_scores


# =============================================================================
# 7. THRESHOLDED AND RANKING METRICS
# =============================================================================


def postprocess_predictions(
    scores: np.ndarray,
    threshold: float,
    no_id_index: int,
    active_label_mask: Optional[np.ndarray] = None,
    allow_empty: bool = False,
) -> np.ndarray:
    predictions = (scores >= threshold).astype(np.uint8)
    if active_label_mask is not None:
        predictions[:, ~np.asarray(active_label_mask, dtype=bool)] = 0
    empty = predictions.sum(axis=1) == 0
    if np.any(empty) and not allow_empty:
        fallback_scores = scores.copy()
        if active_label_mask is not None:
            fallback_scores[:, ~np.asarray(active_label_mask, dtype=bool)] = -np.inf
        predictions[np.flatnonzero(empty), np.argmax(fallback_scores[empty], axis=1)] = 1
    real_positive = predictions.sum(axis=1) - predictions[:, no_id_index] > 0
    predictions[real_positive, no_id_index] = 0
    return predictions


def sample_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    intersection = np.logical_and(y_true, y_pred).sum(axis=1).astype(np.float64)
    pred_count = y_pred.sum(axis=1).astype(np.float64)
    true_count = y_true.sum(axis=1).astype(np.float64)
    union = np.logical_or(y_true, y_pred).sum(axis=1).astype(np.float64)
    precision = np.divide(intersection, pred_count, out=np.zeros_like(intersection), where=pred_count > 0)
    recall = np.divide(intersection, true_count, out=np.zeros_like(intersection), where=true_count > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0)
    jaccard = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    return {
        "sample_precision": float(precision.mean()),
        "sample_recall": float(recall.mean()),
        "sample_f1": float(f1.mean()),
        "sample_jaccard": float(jaccard.mean()),
    }


def threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    no_id_index: int,
    active_label_mask: Optional[np.ndarray] = None,
    allow_empty: bool = False,
) -> Tuple[Dict[str, Any], np.ndarray]:
    y_pred = postprocess_predictions(
        scores, threshold, no_id_index, active_label_mask, allow_empty
    )
    metrics: Dict[str, Any] = {
        "threshold": float(threshold),
        "exact_match_accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "average_predicted_labels": float(y_pred.sum(axis=1).mean()),
    }
    metrics.update(sample_metrics(y_true, y_pred))
    return metrics, y_pred


def ranking_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    no_id_index: int,
    ks: Sequence[int] = RANKING_KS,
    active_label_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    active = (
        np.ones(scores.shape[1], dtype=bool)
        if active_label_mask is None else np.asarray(active_label_mask, dtype=bool)
    )
    real_indices = np.array([
        index for index in range(scores.shape[1])
        if index != no_id_index and active[index]
    ])
    result: Dict[str, Any] = {}
    valid_rows = np.flatnonzero(y_true[:, real_indices].sum(axis=1) > 0)
    result["evaluated_real_capec_rows"] = int(len(valid_rows))
    if not len(valid_rows):
        result["label_ranking_average_precision"] = 0.0
        for k in ks:
            result.update({f"hit@{k}": 0.0, f"precision@{k}": 0.0, f"recall@{k}": 0.0, f"ndcg@{k}": 0.0})
        return result

    real_scores = scores[valid_rows][:, real_indices]
    real_true = y_true[valid_rows][:, real_indices]
    try:
        result["label_ranking_average_precision"] = float(
            label_ranking_average_precision_score(real_true, real_scores)
        )
    except Exception:
        result["label_ranking_average_precision"] = None
    order = np.argsort(-real_scores, axis=1)
    ranked_labels = real_indices[order]
    for k in ks:
        top = ranked_labels[:, :min(k, len(real_indices))]
        hits, precisions, recalls, ndcgs = [], [], [], []
        discounts = 1.0 / np.log2(np.arange(2, top.shape[1] + 2))
        for local_row, original_row in enumerate(valid_rows):
            true_set = set(np.flatnonzero(y_true[original_row]).tolist())
            true_set.discard(no_id_index)
            relevance = np.array([int(index in true_set) for index in top[local_row]], dtype=np.float64)
            tp = float(relevance.sum())
            hits.append(float(tp > 0))
            precisions.append(tp / k)
            recalls.append(tp / len(true_set))
            dcg = float((relevance * discounts).sum())
            ideal_count = min(len(true_set), top.shape[1])
            idcg = float(discounts[:ideal_count].sum())
            ndcgs.append(dcg / idcg if idcg else 0.0)
        result[f"hit@{k}"] = float(np.mean(hits))
        result[f"precision@{k}"] = float(np.mean(precisions))
        result[f"recall@{k}"] = float(np.mean(recalls))
        result[f"ndcg@{k}"] = float(np.mean(ndcgs))
    return result


def choose_validation_parameters(
    y_val: np.ndarray,
    components_by_temperature: Mapping[float, Tuple[np.ndarray, np.ndarray]],
    no_id_index: int,
    active_label_mask: Optional[np.ndarray] = None,
    selection_label_mask: Optional[np.ndarray] = None,
    allow_empty: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Any], np.ndarray, List[Dict[str, Any]]]:
    best_key: Optional[Tuple[float, float, float]] = None
    best_parameters: Dict[str, float] = {}
    best_metrics: Dict[str, Any] = {}
    best_scores: Optional[np.ndarray] = None
    search_rows: List[Dict[str, Any]] = []
    for temperature, (instance_scores, label_scores) in components_by_temperature.items():
        for lambda_value in LAMBDA_GRID:
            scores = combine_rae_scores(instance_scores, label_scores, lambda_value)
            for threshold in THRESHOLD_GRID:
                # Compute only selection criteria inside the calibration grid.
                y_pred = postprocess_predictions(
                    scores, threshold, no_id_index, active_label_mask, allow_empty
                )
                metric_mask = (
                    np.ones(y_val.shape[1], dtype=bool)
                    if selection_label_mask is None
                    else np.asarray(selection_label_mask, dtype=bool)
                )
                true_bool = y_val[:, metric_mask].astype(bool)
                pred_bool = y_pred[:, metric_mask].astype(bool)
                tp = float(np.logical_and(true_bool, pred_bool).sum())
                fp = float(np.logical_and(~true_bool, pred_bool).sum())
                fn = float(np.logical_and(true_bool, ~pred_bool).sum())
                denominator = 2.0 * tp + fp + fn
                micro_f1 = 2.0 * tp / denominator if denominator else 0.0
                intersection = np.logical_and(true_bool, pred_bool).sum(axis=1).astype(np.float64)
                row_denominator = true_bool.sum(axis=1) + pred_bool.sum(axis=1)
                row_f1 = np.divide(
                    2.0 * intersection,
                    row_denominator,
                    out=np.zeros_like(intersection),
                    where=row_denominator > 0,
                )
                sample_f1 = float(row_f1.mean())
                exact_match = float(np.all(true_bool == pred_bool, axis=1).mean())
                search_rows.append({
                    "temperature": float(temperature),
                    "lambda": float(lambda_value),
                    "threshold": float(threshold),
                    "validation_micro_f1": micro_f1,
                    "validation_sample_f1": sample_f1,
                    "validation_exact_match_accuracy": exact_match,
                })
                key = (micro_f1, sample_f1, exact_match)
                if best_key is None or key > best_key:
                    best_key = key
                    best_parameters = {
                        "temperature": float(temperature),
                        "lambda": float(lambda_value),
                        "threshold": float(threshold),
                    }
                    best_scores = scores.copy()
    if best_scores is None:
        raise RuntimeError("Validation calibration produced no candidate.")
    best_metrics, _ = threshold_metrics(
        y_val,
        best_scores,
        best_parameters["threshold"],
        no_id_index,
        active_label_mask,
        allow_empty,
    )
    best_metrics.update(ranking_metrics(
        y_val, best_scores, no_id_index, active_label_mask=active_label_mask
    ))
    if best_key is not None:
        best_metrics["selection_micro_f1"] = float(best_key[0])
        best_metrics["selection_sample_f1"] = float(best_key[1])
        best_metrics["selection_exact_match_accuracy"] = float(best_key[2])
        best_metrics["selection_scope"] = (
            "selected label columns" if selection_label_mask is not None else "full label vocabulary"
        )
    for row in search_rows:
        row["selected"] = (
            row["temperature"] == best_parameters["temperature"]
            and row["lambda"] == best_parameters["lambda"]
            and row["threshold"] == best_parameters["threshold"]
        )
    return best_parameters, best_metrics, best_scores, search_rows


# =============================================================================
# 8. ARTIFACTS AND FRESH-PROCESS INFERENCE
# =============================================================================


def save_retrieval_artifacts(
    run_dir: Path,
    memory_embeddings: np.ndarray,
    index: SearchIndex,
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    labels: Sequence[str],
    label_texts: Sequence[str],
    label_text_sources: Sequence[str],
    offsets: np.ndarray,
    flat_labels: np.ndarray,
    label_memory_ids: np.ndarray,
) -> None:
    np.save(run_dir / "retrieval_embeddings.npy", memory_embeddings)
    index_path = run_dir / "retrieval.index"
    if isinstance(index, FaissHNSWIndex):
        index.save(index_path)
        index_file: Optional[str] = index_path.name
    else:
        index_file = None
    np.savez_compressed(
        run_dir / "retrieval_memory_labels.npz",
        train_label_offsets=offsets,
        train_label_indices=flat_labels,
        num_train=np.array([len(train_df)], dtype=np.int64),
        num_labels=np.array([len(labels)], dtype=np.int64),
        label_memory_ids=np.asarray(label_memory_ids, dtype=np.int64),
    )
    metadata_rows = []
    for row in range(len(train_df)):
        label_ids = np.flatnonzero(y_train[row]).tolist()
        metadata_rows.append({
            "memory_index": row,
            "source_type": "training_instance",
            "source_index": row,
            "source_id": train_df.iloc[row][CVE_COL] if CVE_COL in train_df.columns else "",
            "labels": json.dumps([labels[index] for index in label_ids]),
            "label_indices": json.dumps(label_ids),
            "label_text": "",
            "label_text_source": "training_ground_truth",
        })
    base = len(train_df)
    for local_index, label_index in enumerate(label_memory_ids.tolist()):
        label, text, source = (
            labels[label_index], label_texts[label_index], label_text_sources[label_index]
        )
        metadata_rows.append({
            "memory_index": base + local_index,
            "source_type": "label_text",
            "source_index": label_index,
            "source_id": label,
            "labels": json.dumps([label]),
            "label_indices": json.dumps([label_index]),
            "label_text": text,
            "label_text_source": source,
        })
    pd.DataFrame(metadata_rows).to_csv(run_dir / "retrieval_memory_metadata.csv", index=False)
    save_json(run_dir / "retrieval_backend.json", {
        "backend": index.backend,
        "index_file": index_file,
        "embeddings_file": "retrieval_embeddings.npy",
        "similarity": "inner_product_on_l2_normalized_embeddings",
        "memory_order": "training instances followed by label texts",
        "label_memory_ids": label_memory_ids.tolist(),
        "contains_validation_or_test_instances": False,
    })


class LoadedRAEXMC:
    def __init__(self, run_dir: Path, device: Optional[str] = None):
        self.run_dir = Path(run_dir).resolve()
        self.config_dict = json.loads((self.run_dir / "model_config.json").read_text(encoding="utf-8"))
        config_fields = set(RunConfig.__dataclass_fields__)
        saved_run_config = dict(self.config_dict["run_config"])
        self.config = RunConfig(**{k: v for k, v in saved_run_config.items() if k in config_fields})
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.run_dir / "tokenizer"), local_files_only=True)
        self.model = RAEXMCEncoder(config_path=self.run_dir / "encoder_config")
        checkpoint = load_torch_checkpoint(self.run_dir / "best_model.pt", self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()
        self.parameters = json.loads((self.run_dir / "calibration_parameters.json").read_text(encoding="utf-8"))
        self.label_to_idx = json.loads((self.run_dir / "label_to_idx.json").read_text(encoding="utf-8"))
        idx_raw = json.loads((self.run_dir / "idx_to_label.json").read_text(encoding="utf-8"))
        self.idx_to_label = {int(k): v for k, v in idx_raw.items()}
        self.no_id_index = self.label_to_idx[NO_ID_LABEL]
        mapping = np.load(self.run_dir / "retrieval_memory_labels.npz")
        self.offsets = mapping["train_label_offsets"]
        self.flat_labels = mapping["train_label_indices"]
        self.num_train = int(mapping["num_train"][0])
        self.num_labels = int(mapping["num_labels"][0])
        self.label_memory_ids = (
            mapping["label_memory_ids"].astype(np.int64)
            if "label_memory_ids" in mapping.files
            else np.arange(self.num_labels, dtype=np.int64)
        )
        embeddings = np.load(self.run_dir / "retrieval_embeddings.npy", mmap_mode="r")
        index_path = self.run_dir / "retrieval.index"
        if faiss is not None and index_path.exists():
            self.index: SearchIndex = FaissHNSWIndex.from_file(index_path, self.config)
        else:
            self.index = NumpyChunkedIPIndex(embeddings)

    @torch.no_grad()
    def score(
        self,
        description: str,
        keyphrases: Optional[Sequence[str] | str] = None,
    ) -> np.ndarray:
        """Return the complete RAE-XMC score vector for one description."""
        if not str(description).strip():
            raise ValueError("description must be non-empty")
        text = str(description).strip()
        if self.config.use_keyphrases and keyphrases:
            if isinstance(keyphrases, str):
                parsed = parse_keyphrase_cell(keyphrases)
            else:
                parsed = [str(x).strip() for x in keyphrases if str(x).strip()]
            if parsed:
                text += KEYPHRASE_SEPARATOR + " ; ".join(parsed)
        encoded = self.tokenizer(
            [text], padding="max_length", truncation=True,
            max_length=self.config.max_length, return_tensors="pt",
        )
        query = self.model(
            encoded["input_ids"].to(self.device),
            encoded["attention_mask"].to(self.device),
        ).float().cpu().numpy()
        neighbor_scores, neighbor_indices = self.index.search(query, self.config.retrieval_k)
        instance, label, _ = aggregate_rae_components(
            neighbor_scores, neighbor_indices, self.num_train, self.num_labels,
            self.offsets, self.flat_labels, self.parameters["temperature"], False,
            self.label_memory_ids,
        )
        return combine_rae_scores(instance, label, self.parameters["lambda"])[0]

    @torch.no_grad()
    def predict(
        self,
        description: str,
        keyphrases: Optional[Sequence[str] | str] = None,
        top_k: int = 10,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        scores = self.score(description, keyphrases)
        real_order = [
            index for index in np.argsort(-scores)
            if index != self.no_id_index
        ]
        ranked = [
            {"capec_id": self.idx_to_label[int(index)], "score": float(scores[index])}
            for index in real_order[:top_k]
        ]
        chosen_threshold = float(self.parameters["threshold"] if threshold is None else threshold)
        thresholded = postprocess_predictions(
            scores[None, :],
            chosen_threshold,
            self.no_id_index,
        )[0]
        predicted = [self.idx_to_label[index] for index in np.flatnonzero(thresholded)]
        return {
            "ranked_capecs": ranked,
            "threshold": chosen_threshold,
            "thresholded_labels": predicted,
            "retrieval_k": int(self.config.retrieval_k),
            "temperature": float(self.parameters["temperature"]),
            "lambda": float(self.parameters["lambda"]),
            "full_output_shape": [self.num_labels],
        }


def load_rae_xmc(run_dir: str | Path, device: Optional[str] = None) -> LoadedRAEXMC:
    """Load a saved encoder + training-only retrieval memory without retraining."""
    return LoadedRAEXMC(Path(run_dir), device=device)


def predict_cve(
    run_dir: str | Path,
    description: str,
    keyphrases: Optional[Sequence[str] | str] = None,
    top_k: int = 10,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Fresh-process convenience API requested for deployment."""
    return load_rae_xmc(run_dir, device=device).predict(description, keyphrases, top_k)


# =============================================================================
# 10. END-TO-END TRAIN / VALIDATE / INDEX / TEST PIPELINE
# =============================================================================


def save_prediction_tables(
    run_dir: Path,
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    labels: Sequence[str],
    no_id_index: int,
    active_label_mask: Optional[np.ndarray] = None,
) -> None:
    threshold_rows, topk_rows = [], []
    active = (
        np.ones(len(labels), dtype=bool)
        if active_label_mask is None else np.asarray(active_label_mask, dtype=bool)
    )
    real_indices = np.array([
        index for index in range(len(labels)) if index != no_id_index and active[index]
    ])
    ranked_real = real_indices[np.argsort(-scores[:, real_indices], axis=1)]
    for row in range(len(test_df)):
        true_ids = [labels[index] for index in np.flatnonzero(y_true[row])]
        pred_ids = [labels[index] for index in np.flatnonzero(y_pred[row])]
        common = {
            "row_index": row,
            "cve_id": test_df.iloc[row][CVE_COL] if CVE_COL in test_df.columns else None,
            "description": test_df.iloc[row][DESC_COL],
            "keyphrases": test_df.iloc[row][KEYPHRASE_COL],
            "true_labels": json.dumps(true_ids),
        }
        threshold_rows.append({
            **common,
            "predicted_labels": json.dumps(pred_ids),
            "predicted_scores": json.dumps({label: float(scores[row, index]) for index, label in enumerate(labels) if y_pred[row, index]}),
        })
        ranked = [
            {"capec_id": labels[int(index)], "score": float(scores[row, index])}
            for index in ranked_real[row, :max(RANKING_KS)]
        ]
        topk_rows.append({
            **common,
            "ranked_predictions": json.dumps(ranked),
            **{
                f"hit_at_{k}": int(bool(set(true_ids) & {item["capec_id"] for item in ranked[:k]}))
                if any(label != NO_ID_LABEL for label in true_ids) else None
                for k in RANKING_KS
            },
        })
    pd.DataFrame(threshold_rows).to_csv(run_dir / "test_predictions.csv", index=False)
    pd.DataFrame(topk_rows).to_csv(run_dir / "topk_predictions.csv", index=False)


def checkpoint_payload(
    model: RAEXMCEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    config: RunConfig,
    epoch: int,
    parameters: Mapping[str, float],
    validation_metrics: Mapping[str, Any],
    labels: Sequence[str],
) -> Dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "run_config": asdict(config),
        "best_epoch": int(epoch),
        "calibration_parameters": dict(parameters),
        "validation_metrics": dict(validation_metrics),
        "labels": list(labels),
        "paper": "Wang et al., Retrieval-augmented Encoders for Extreme Multi-label Text Classification, arXiv:2502.10615",
    }


def run_pipeline(
    smoke_test: bool = False,
    use_keyphrases: bool = False,
) -> Path:
    config = RunConfig(
        batch_size=min(BATCH_SIZE, 8) if smoke_test else BATCH_SIZE,
        eval_batch_size=min(EVAL_BATCH_SIZE, 32) if smoke_test else EVAL_BATCH_SIZE,
        epochs=1 if smoke_test else EPOCHS,
        patience=1 if smoke_test else PATIENCE,
        retrieval_k=min(RETRIEVAL_K, 16) if smoke_test else RETRIEVAL_K,
        hard_negative_topk=min(HARD_NEGATIVE_TOPK, 16) if smoke_test else HARD_NEGATIVE_TOPK,
        use_keyphrases=use_keyphrases,
    )
    seed_everything(config.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if smoke_test:
        run_dir = (
            KEYWORD_SMOKE_OUTPUT_DIR if use_keyphrases else SMOKE_OUTPUT_DIR
        )
    else:
        run_dir = KEYWORD_OUTPUT_DIR if use_keyphrases else OUTPUT_DIR
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))
    if device.type == "cuda":
        print("Visible CUDA device:", torch.cuda.get_device_name(0))
    print("Run directory:", run_dir)
    print("Mode:", "SMOKE TEST" if smoke_test else "FULL TRAINING")
    print("Input case:", "description+keywords" if use_keyphrases else "description")

    raw_train, raw_val, raw_test = load_fixed_splits(DATASET_PATH, config.seed)
    train_df, dropped_train = validate_and_prepare_split(raw_train, "training", config.use_keyphrases)
    val_df, dropped_val = validate_and_prepare_split(raw_val, "validation", config.use_keyphrases)
    test_df, dropped_test = validate_and_prepare_split(raw_test, "testing", config.use_keyphrases)
    overlap_report = split_id_overlap(train_df, val_df, test_df)
    save_json(run_dir / "split_overlap_report.json", overlap_report)

    (
        labels, label_to_idx, idx_to_label, relationship_labels,
        catalog_labels, data_labels, names, name_sources,
    ) = build_label_space(
        train_df, val_df, test_df, CAPEC_REL_PATH, CAPEC_CATALOG_PATH
    )
    label_texts, label_text_sources = build_label_texts(
        labels, names, name_sources
    )
    missing_catalog_labels = sorted(set(catalog_labels) - set(labels))
    if missing_catalog_labels:
        raise RuntimeError(
            "RAE-XMC label vocabulary is missing canonical CAPECs: "
            f"{missing_catalog_labels[:20]}"
        )
    y_train_full = make_multihot(train_df["capec_label_list"].tolist(), label_to_idx)
    y_val_full = make_multihot(val_df["capec_label_list"].tolist(), label_to_idx)
    y_test_full = make_multihot(test_df["capec_label_list"].tolist(), label_to_idx)

    if smoke_test:
        train_df, val_df, test_df = train_df.iloc[:64].copy(), val_df.iloc[:32].copy(), test_df.iloc[:32].copy()
        y_train, y_val, y_test = y_train_full[:64], y_val_full[:32], y_test_full[:32]
    else:
        y_train, y_val, y_test = y_train_full, y_val_full, y_test_full
    if not len(train_df) or not len(val_df) or not len(test_df):
        raise ValueError("All three fixed splits must contain at least one non-empty-description row.")

    no_id_index = label_to_idx[NO_ID_LABEL]
    active_label_ids = np.arange(len(labels), dtype=np.int64)
    negative_eligible_mask = y_train.sum(axis=0) > 0
    positive_sampling_weights = np.ones(len(labels), dtype=np.float32)
    print(f"Rows: train={len(train_df):,}, validation={len(val_df):,}, test={len(test_df):,}")
    print(f"Labels: {len(labels):,} including {NO_ID_LABEL}")
    print(f"Canonical catalog CAPEC labels: {len(catalog_labels):,}")
    print(f"Authoritative label names: {sum(x != 'identifier_fallback' for x in label_text_sources):,}")
    print(f"Identifier-only label fallbacks: {sum(x == 'identifier_fallback' for x in label_text_sources):,}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    train_tokens = tokenize_texts(tokenizer, train_df["model_text"].tolist(), config.max_length, "train inputs")
    val_tokens = tokenize_texts(tokenizer, val_df["model_text"].tolist(), config.max_length, "validation inputs")
    test_tokens = tokenize_texts(tokenizer, test_df["model_text"].tolist(), config.max_length, "test inputs")
    label_tokens = tokenize_texts(tokenizer, label_texts, config.max_length, "CAPEC label texts")

    train_dataset = RAEXMCDataset(train_tokens, y_train, y_train)
    val_dataset = TokenOnlyDataset(val_tokens)
    test_dataset = TokenOnlyDataset(test_tokens)
    label_dataset = TokenOnlyDataset(label_tokens)
    train_dataset.set_hard_negatives(
        random_negative_matrix(
            y_train,
            config.hard_negatives_per_input,
            config.seed,
            negative_eligible_mask,
        )
    )

    model = RAEXMCEncoder(model_name=config.model_name).to(device)
    if config.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.encoder_lr, weight_decay=config.weight_decay)
    train_loader = make_loader(train_dataset, config.batch_size, True, config)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = make_grad_scaler(config.use_amp)
    print(f"Trainable encoder parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("Classifier head parameters: 0 (paper-faithful non-parametric predictor)")

    best_score = -1.0
    best_epoch = 0
    best_parameters: Dict[str, float] = {}
    best_validation_metrics: Dict[str, Any] = {}
    patience_counter = 0
    history: List[Dict[str, Any]] = []
    checkpoint_path = run_dir / "best_model.pt"

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if config.use_hard_negative_mining and (
            epoch == 1 or (epoch - 1) % config.hard_negative_refresh_epochs == 0
        ):
            label_embeddings_for_mining = encode_dataset(model, label_dataset, device, EMBED_BATCH_SIZE, config)
            mined = mine_hard_negatives(
                model,
                train_dataset,
                label_embeddings_for_mining,
                active_label_ids,
                y_train,
                negative_eligible_mask,
                device,
                config,
                epoch,
            )
            train_dataset.set_hard_negatives(mined)
            del label_embeddings_for_mining, mined
            clear_memory()

        train_loss = train_one_epoch(
            model,
            train_loader,
            label_tokens,
            optimizer,
            scheduler,
            scaler,
            device,
            config,
            torch.from_numpy(positive_sampling_weights),
        )

        # The encoder changed: rebuild the training-only memory before validation.
        memory, _, _ = build_unified_memory(model, train_dataset, label_dataset, device, config)
        index = build_search_index(memory, config)
        val_neighbor_scores, val_neighbor_indices = retrieve_dataset(
            model, val_dataset, index, device, config, config.retrieval_k
        )
        offsets, flat_labels = label_rows_to_ragged(y_train)
        components: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
        for temperature in INFERENCE_TEMPERATURE_GRID:
            instance_scores, label_scores, _ = aggregate_rae_components(
                val_neighbor_scores, val_neighbor_indices, len(train_df), len(labels),
                offsets, flat_labels, float(temperature), False,
                active_label_ids,
            )
            components[float(temperature)] = (instance_scores, label_scores)
        parameters, val_metrics, _, calibration_rows = choose_validation_parameters(
            y_val,
            components,
            no_id_index,
        )
        selection_score = float(val_metrics["selection_micro_f1"])
        elapsed = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_micro_f1": val_metrics["micro_f1"],
            "validation_macro_f1": val_metrics["macro_f1"],
            "validation_sample_f1": val_metrics["sample_f1"],
            "validation_exact_match_accuracy": val_metrics["exact_match_accuracy"],
            "validation_selection_micro_f1": selection_score,
            "lambda": parameters["lambda"],
            "temperature": parameters["temperature"],
            "threshold": parameters["threshold"],
            "seconds": elapsed,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
        print(
            f"Epoch {epoch}/{config.epochs}: loss={train_loss:.5f} "
            f"val_micro_f1={val_metrics['micro_f1']:.5f} "
            f"selection_micro_f1={selection_score:.5f} lambda={parameters['lambda']:.2f} "
            f"threshold={parameters['threshold']:.4f} time={elapsed:.1f}s"
        )

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_parameters = parameters
            best_validation_metrics = val_metrics
            torch.save(
                checkpoint_payload(
                    model, optimizer, scheduler, scaler, config, epoch,
                    parameters, val_metrics, labels,
                ),
                checkpoint_path,
            )
            tokenizer.save_pretrained(run_dir / "tokenizer")
            model.encoder.config.save_pretrained(run_dir / "encoder_config")
            save_json(run_dir / "best_validation_metrics.json", val_metrics)
            save_json(run_dir / "calibration_parameters.json", parameters)
            pd.DataFrame(calibration_rows).to_csv(
                run_dir / "lambda_calibration_results.csv", index=False
            )
            patience_counter = 0
            print("  saved new best validation checkpoint")
        else:
            patience_counter += 1
            print(f"  no improvement ({patience_counter}/{config.patience})")
        del (
            memory, index, val_neighbor_scores, val_neighbor_indices,
            components, calibration_rows,
        )
        clear_memory()
        if patience_counter >= config.patience:
            print("Early stopping on validation micro-F1.")
            break

    # Reload best weights before constructing the persistent training-only memory.
    checkpoint = load_torch_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    best_parameters = dict(checkpoint["calibration_parameters"])
    best_validation_metrics = dict(checkpoint["validation_metrics"])
    final_memory, _, _ = build_unified_memory(model, train_dataset, label_dataset, device, config)
    final_index = build_search_index(final_memory, config)
    offsets, flat_labels = label_rows_to_ragged(y_train)
    save_retrieval_artifacts(
        run_dir,
        final_memory,
        final_index,
        train_df,
        y_train,
        labels,
        label_texts,
        label_text_sources,
        offsets,
        flat_labels,
        active_label_ids,
    )

    # Final held-out test evaluation: all choices above are already fixed.
    test_neighbor_scores, test_neighbor_indices = retrieve_dataset(
        model, test_dataset, final_index, device, config, config.retrieval_k
    )
    instance_scores, label_scores, retrieval_candidates = aggregate_rae_components(
        test_neighbor_scores, test_neighbor_indices, len(train_df), len(labels),
        offsets, flat_labels, best_parameters["temperature"], True,
        active_label_ids,
    )
    test_scores = combine_rae_scores(instance_scores, label_scores, best_parameters["lambda"])
    test_metrics, test_pred = threshold_metrics(
        y_test,
        test_scores,
        best_parameters["threshold"],
        no_id_index,
    )
    test_metrics.update(ranking_metrics(y_test, test_scores, no_id_index))
    test_metrics["calibration_parameters"] = best_parameters
    test_metrics["best_epoch"] = best_epoch
    test_metrics["metric_scope"] = "full label vocabulary"
    save_json(run_dir / "test_metrics.json", test_metrics)

    label_info = pd.DataFrame({
        "label": labels,
        "index": np.arange(len(labels)),
        "label_text": label_texts,
        "label_text_source": label_text_sources,
        "train_count": y_train.sum(axis=0).astype(np.int64),
        "validation_count": y_val.sum(axis=0).astype(np.int64),
        "test_count": y_test.sum(axis=0).astype(np.int64),
    })
    label_info.to_csv(run_dir / "label_info.csv", index=False)
    save_prediction_tables(
        run_dir, test_df, y_test, test_pred, test_scores, labels, no_id_index,
    )

    save_json(run_dir / "label_to_idx.json", label_to_idx)
    save_json(run_dir / "idx_to_label.json", {str(k): v for k, v in idx_to_label.items()})
    save_json(run_dir / "model_config.json", {
        "run_config": asdict(config),
        "data": {
            "script_dir": str(SCRIPT_DIR),
            "split_directory": str(DATASET_PATH),
            "split_source": "fixed KeyBERT Modified 70/15/15 files",
            "capec_relationships_file": CAPEC_REL_PATH.name if CAPEC_REL_PATH.exists() else None,
            "capec_catalog_file": CAPEC_CATALOG_PATH.name if CAPEC_CATALOG_PATH.exists() else None,
            "description_column": DESC_COL,
            "keyphrase_column": KEYPHRASE_COL,
            "label_column": CAPEC_COL,
            "cve_id_column": CVE_COL,
            "fixed_splits_preserved": True,
            "dropped_empty_descriptions": {
                "train": dropped_train, "validation": dropped_val, "test": dropped_test,
            },
            "rows_used": {
                "train": len(train_df),
                "validation": len(val_df),
                "test": len(test_df),
            },
        },
        "label_space": {
            "num_labels": len(labels),
            "relationship_label_count": len(relationship_labels),
            "catalog_label_count": len(catalog_labels),
            "data_label_count": len(data_labels),
            "canonical_catalog_complete": not missing_catalog_labels,
            "missing_catalog_labels": missing_catalog_labels,
            "no_id_label": NO_ID_LABEL,
            "no_id_index": no_id_index,
            "label_text_fallback_count": int(sum(x == "identifier_fallback" for x in label_text_sources)),
            "ordering_preserved": True,
        },
        "method": {
            "name": "RAE-XMC",
            "encoder": "shared MiniLM dual encoder",
            "pooling": "attention-mask-aware mean pooling then L2 normalization",
            "loss": "sampled decoupled contrastive softmax Eq. (10), with non-overlapping same-tower negatives",
            "memory": "K=[training instance embeddings; label text embeddings]",
            "values": "V=[lambda*Y_train; (1-lambda)*I]",
            "prediction": "Softmax(qK^T/tau)V over unified top-b keys",
            "classifier_head": None,
            "bce_loss": None,
        },
        "best_epoch": best_epoch,
        "selected_parameters": best_parameters,
        "package_versions": package_versions(),
        "seed": config.seed,
        "smoke_test": smoke_test,
    })
    save_json(run_dir / "best_validation_metrics.json", best_validation_metrics)
    save_json(run_dir / "calibration_parameters.json", best_parameters)
    save_json(run_dir / "validation_thresholds.json", {
        "selection_split": "validation",
        "candidate_thresholds": list(THRESHOLD_GRID),
        "selected_threshold": best_parameters["threshold"],
        "test_labels_used_for_selection": False,
    })

    # Explicit reload verification, required by smoke-test mode and useful always.
    reloaded = load_rae_xmc(run_dir, device=str(device))
    reload_result = reloaded.predict(
        str(test_df.iloc[0][DESC_COL]),
        test_df.iloc[0][KEYPHRASE_COL] if config.use_keyphrases else None,
        top_k=min(3, len(labels) - 1),
    )
    save_json(run_dir / "checkpoint_reload_test.json", {"passed": True, "example": reload_result})
    del reloaded

    print("\nBest validation metrics:")
    print(json.dumps(json_ready(best_validation_metrics), indent=2))
    print("\nFinal test metrics:")
    print(json.dumps(json_ready(test_metrics), indent=2))
    print("\nSaved complete run to:", run_dir)
    if smoke_test:
        print("SMOKE TEST PASSED: forward, loss/backward, retrieval, save, and reload completed.")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate or load MiniLM RAE-XMC.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny end-to-end integration test.")
    parser.add_argument(
        "--keywords",
        action="store_true",
        help="Train/evaluate the separate description+KeyBERT-keywords model.",
    )
    parser.add_argument("--inference-run", type=Path, help="Existing run directory to load without training.")
    parser.add_argument("--text", type=str, help="New CVE description for --inference-run.")
    parser.add_argument("--keyphrases", type=str, default=None, help="Optional keyphrases for inference.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of ranked real CAPEC IDs to return.")
    parser.add_argument("--device", type=str, default=None, help="Inference device, e.g. cpu or cuda.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inference_run is not None:
        if not args.text:
            raise SystemExit("--text is required with --inference-run")
        predictor = load_rae_xmc(args.inference_run, device=args.device)
        result = predictor.predict(args.text, args.keyphrases, args.top_k)
        print(json.dumps(json_ready(result), indent=2))
        return
    run_pipeline(smoke_test=args.smoke_test, use_keyphrases=args.keywords)


if __name__ == "__main__":
    main()
