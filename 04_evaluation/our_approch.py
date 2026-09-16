#!/usr/bin/env python3
"""Our validation-selected fusion of CGDPF and RAE-XMC.

Two controlled cases are supported: cleaned description only, and cleaned
description plus the selected modified-KeyBERT keywords. RAE-XMC remains the
same description retrieval branch; the keyword case changes the direct CAPEC
and CWE-to-CAPEC branches through their keyword-aware checkpoints.

No encoder or classifier is trained here. The script:

1. loads the Stage-04 frozen-MiniLM CWE and direct-CAPEC MLP checkpoints;
2. maps top-10 CWE probabilities to CAPEC by maximum support;
3. independently calibrates direct, mapping, and RAE scores on validation;
4. selects the old-hybrid alpha, then the RAE fusion beta and threshold using
   validation Micro-F1 only; and
5. freezes every choice before one final test evaluation.

The MiniLM embeddings are L2-normalized exactly as in the Stage-04 baseline
training scripts. This is a calibrated weighted fusion, not a learned neural
fusion network; test labels never select calibration or fusion parameters.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import sys
import time
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    label_ranking_average_precision_score,
    precision_score,
    recall_score,
)
from sentence_transformers import SentenceTransformer

from evaluation_data import DATASET_PATH, load_fixed_splits
from evaluation_paths import OUTPUT_ROOT


# =============================================================================
# 0. EDITABLE CONFIGURATION
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
METHODOLOGY_DIR = SCRIPT_DIR.parent
GROUND_TRUTH_DIR = METHODOLOGY_DIR / "01_cwe_capec_ground_truth"
CAPEC_CATALOG_PATH = GROUND_TRUTH_DIR / "3000_capec.csv"

# All model artifacts are produced inside Stage 04. Source data is restricted
# to Stage 01 ground truth and the canonical Stage 02 dataset.
CWE_MODEL_DIR: Path = OUTPUT_ROOT / "CVE_to_CWE" / "NN_1way"
CAPEC_MODEL_DIR: Path = OUTPUT_ROOT / "NN"
RAE_XMC_RUN_DIR: Optional[Path] = OUTPUT_ROOT / "RAE_XMC"
RAE_XMC_KEYWORDS_RUN_DIR: Optional[Path] = OUTPUT_ROOT / "RAE_XMC_keywords"

LOCAL_MAPPING_PATH = GROUND_TRUTH_DIR / "CWE_MAPPED.csv"
FALLBACK_MAPPING_PATH = GROUND_TRUTH_DIR / "cwe_capec_mapping.csv"
SCORE_CACHE_RUN: Optional[Path] = (
    Path(os.environ["SCORE_CACHE_RUN"]).expanduser()
    if os.environ.get("SCORE_CACHE_RUN")
    else None
)
RAE_SCORE_CACHE_RUN: Optional[Path] = (
    Path(os.environ["RAE_SCORE_CACHE_RUN"]).expanduser()
    if os.environ.get("RAE_SCORE_CACHE_RUN")
    else None
)

RUNS_DIR = OUTPUT_ROOT / "combined"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42
BATCH_SIZE = 256
CWE_TOP_K_FOR_MAPPING = 10
USE_RANK_WEIGHT = False

# Search every fusion weight from 0 through 1 at an inclusive 0.02 interval.
# Integer multiplication avoids the endpoint drift of np.arange with floats.
FUSION_WEIGHT_VALUES = tuple(round(step * 0.02, 2) for step in range(51))
ALPHA_VALUES = FUSION_WEIGHT_VALUES
BETA_VALUES = FUSION_WEIGHT_VALUES
THRESHOLDS = np.array([
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
THRESHOLD_VALUES = tuple(THRESHOLDS.tolist())
CALIBRATION_NEGATIVE_RATIO = 20
SEARCH_OBJECTIVE = "micro_f1"
RANKING_KS = (1, 2, 5, 10)

DESC_COL = "cleaned_description"
KEYPHRASE_COL = "keyphrases"
RAE_KEYPHRASE_SEPARATOR = " [SEP] keyphrases: "
CAPEC_COL = "capec_id"
CVE_COL = "cve_id"
NO_ID_LABEL = "CAPEC-noID"


@dataclass(frozen=True)
class RunConfig:
    seed: int = SEED
    batch_size: int = BATCH_SIZE
    cwe_top_k_for_mapping: int = CWE_TOP_K_FOR_MAPPING
    use_rank_weight: bool = USE_RANK_WEIGHT
    alpha_values: Tuple[float, ...] = ALPHA_VALUES
    beta_values: Tuple[float, ...] = BETA_VALUES
    threshold_values: Tuple[float, ...] = THRESHOLD_VALUES
    calibration_negative_ratio: int = CALIBRATION_NEGATIVE_RATIO
    search_objective: str = SEARCH_OBJECTIVE


# =============================================================================
# 1. GENERAL HELPERS, LABEL PARSING, AND DATA VALIDATION
# =============================================================================


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(value), handle, indent=2, sort_keys=True)


def package_versions() -> Dict[str, Optional[str]]:
    result: Dict[str, Optional[str]] = {}
    for package in ("torch", "transformers", "numpy", "pandas", "scipy", "scikit-learn", "faiss-cpu"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    result.update({"python": platform.python_version(), "platform": platform.platform(), "cuda": torch.version.cuda})
    return result


def safe_torch_load(path: Path, device: Any) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null", "[]"}


def normalize_cwe(value: Any) -> Optional[str]:
    if is_missing(value):
        return None
    text = str(value).strip()
    if "noid" in text.lower() or "no-id" in text.lower():
        return "CWE-noID"
    match = re.search(r"CWE\s*[-_:]?\s*(\d+)", text, flags=re.I)
    if match:
        return f"CWE-{int(match.group(1))}"
    if re.fullmatch(r"\d+(?:\.0)?", text):
        return f"CWE-{int(float(text))}"
    return text


def normalize_capec(value: Any) -> Optional[str]:
    if is_missing(value):
        return None
    text = str(value).strip()
    if "noid" in text.lower() or "no-id" in text.lower():
        return NO_ID_LABEL
    match = re.search(r"CAPEC\s*[-_:]?\s*(\d+)", text, flags=re.I)
    if match:
        return f"CAPEC-{int(match.group(1))}"
    if re.fullmatch(r"\d+(?:\.0)?", text):
        return f"CAPEC-{int(float(text))}"
    return text


def parse_capec_list(value: Any) -> List[str]:
    if is_missing(value):
        return []
    text = str(value).strip()
    matches = re.findall(r"CAPEC\s*[-_:]?\s*\d+", text, flags=re.I)
    if matches:
        return sorted({label for item in matches if (label := normalize_capec(item))})
    try:
        parsed = ast.literal_eval(text)
        items = list(parsed) if isinstance(parsed, (list, tuple, set)) else [parsed]
    except Exception:
        items = re.split(r"[,|;/]+", text)
    return sorted({label for item in items if (label := normalize_capec(item))})


def canonical_capec_vocabulary() -> List[str]:
    """Return the shared direct-NN CAPEC order, including fixed-split IDs."""
    catalog = pd.read_csv(CAPEC_CATALOG_PATH)
    id_column = next(
        (column for column in ("ID", "id", "capec_id") if column in catalog),
        None,
    )
    if id_column is None:
        raise ValueError(f"No CAPEC ID column found in {CAPEC_CATALOG_PATH}")
    canonical = {
        normalized
        for value in catalog[id_column].dropna()
        if (normalized := normalize_capec(value)) not in (None, NO_ID_LABEL)
    }
    for frame in load_fixed_splits(DATASET_PATH, SEED):
        nonempty = frame[
            frame[DESC_COL].fillna("").astype(str).str.strip().ne("")
        ]
        for value in nonempty[CAPEC_COL]:
            canonical.update(parse_capec_list(value))
    canonical.discard(NO_ID_LABEL)
    return sorted(canonical) + [NO_ID_LABEL]


def validate_complete_capec_vocabulary(labels: Sequence[str]) -> Dict[str, Any]:
    """Require every canonical CAPEC and report any non-catalog columns."""
    expected = canonical_capec_vocabulary()
    missing = sorted(set(expected) - set(labels))
    extra = sorted(set(labels) - set(expected))
    if missing:
        raise ValueError(
            "CAPEC vocabulary is not the complete canonical NN vocabulary: "
            f"missing={missing[:20]}, extra={extra[:20]}"
        )
    return {
        "catalog_path": str(CAPEC_CATALOG_PATH.resolve()),
        "canonical_real_capec_count": len(expected) - 1,
        "evaluated_label_count_including_no_id": len(labels),
        "missing_canonical_capecs": missing,
        "extra_data_capecs": extra,
        "matches_canonical_order_exactly": list(labels) == expected,
        "complete": True,
    }


def make_truth(frame: pd.DataFrame, labels: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    label_to_idx = {label: index for index, label in enumerate(labels)}
    no_id = label_to_idx[NO_ID_LABEL]
    output = np.zeros((len(frame), len(labels)), dtype=np.uint8)
    unknown: set[str] = set()
    for row, value in enumerate(frame[CAPEC_COL]):
        parsed = parse_capec_list(value)
        known = [label_to_idx[label] for label in parsed if label in label_to_idx]
        unknown.update(label for label in parsed if label not in label_to_idx)
        output[row, known if known else [no_id]] = 1
    return output, sorted(unknown)


def cve_ids(frame: pd.DataFrame) -> List[str]:
    return frame[CVE_COL].fillna("").astype(str).tolist()


def rae_input_texts(frame: pd.DataFrame, use_keywords: bool) -> List[str]:
    descriptions = frame[DESC_COL].fillna("").astype(str).tolist()
    if not use_keywords:
        return descriptions
    result: List[str] = []
    for description, value in zip(descriptions, frame[KEYPHRASE_COL]):
        if is_missing(value):
            phrases: List[str] = []
        else:
            raw = str(value).strip()
            try:
                parsed = ast.literal_eval(raw)
            except Exception:
                parsed = None
            candidates = (
                list(parsed)
                if isinstance(parsed, (list, tuple, set))
                else re.split(r"[;,|\n]+", raw)
            )
            phrases = [
                " ".join(str(item).strip().split())
                for item in candidates
                if str(item).strip()
            ]
        result.append(
            description
            + (RAE_KEYPHRASE_SEPARATOR + " ; ".join(phrases) if phrases else "")
        )
    return result


def id_digest(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


# =============================================================================
# 2. STAGE-04 FROZEN MINILM EMBEDDING + MLP INFERENCE
# =============================================================================


class StoredMLP(nn.Module):
    """Reconstruct the description-only MLP used by Stage 04."""

    def __init__(self, state: Mapping[str, torch.Tensor]):
        super().__init__()
        linear_layers: Dict[int, Tuple[int, int]] = {}
        for key, value in state.items():
            match = re.fullmatch(r"net\.(\d+)\.weight", key)
            if match and value.ndim == 2:
                linear_layers[int(match.group(1))] = (int(value.shape[1]), int(value.shape[0]))
        if not linear_layers:
            raise ValueError("Cannot infer Stage-04 MLP layers from checkpoint.")

        modules: OrderedDict[str, nn.Module] = OrderedDict()
        last_index = max(linear_layers)
        for index in range(last_index + 1):
            if index in linear_layers:
                modules[str(index)] = nn.Linear(*linear_layers[index])
            elif index in (1, 4):
                modules[str(index)] = nn.ReLU()
            elif index in (2, 5):
                modules[str(index)] = nn.Dropout(0.30)
            else:
                raise ValueError(f"Unsupported Stage-04 MLP layer index: {index}")
        self.net = nn.Sequential(modules)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format.")
    for key in ("model_state_dict", "state_dict", "model", "net", "checkpoint"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    return checkpoint


def normalize_state_keys(state: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    for original_key, value in state.items():
        key = original_key
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        result[key] = value
    return result


def checkpoint_path(model_dir: Path, kind: str) -> Path:
    filename = (
        "best_cwe_multilabel_mlp.pt"
        if kind == "cwe"
        else "best_capec_multilabel_mlp.pt"
    )
    path = model_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing {kind.upper()} checkpoint: {path}")
    return path


def labels_from_checkpoint(checkpoint: Mapping[str, Any], model_dir: Path, kind: str) -> List[str]:
    raw: Any = checkpoint.get("idx_to_label") or checkpoint.get("all_labels")
    if raw is None:
        idx_path = model_dir / "idx_to_label.json"
        if idx_path.exists():
            raw = json.loads(idx_path.read_text(encoding="utf-8"))
        else:
            info = pd.read_csv(model_dir / "label_info.csv")
            index_col = next((column for column in ("index", "idx", "label_idx") if column in info), None)
            label_col = next((column for column in ("label", "capec_id", "cwe_id") if column in info), None)
            if label_col is None:
                raise ValueError(f"Cannot determine labels from {model_dir / 'label_info.csv'}")
            if index_col:
                info = info.sort_values(index_col)
            raw = info[label_col].tolist()
    if isinstance(raw, dict):
        labels = [raw[str(i)] if str(i) in raw else raw[i] for i in range(len(raw))]
    else:
        labels = list(raw)
    normalizer = normalize_cwe if kind == "cwe" else normalize_capec
    normalized = [normalizer(value) for value in labels]
    if any(value is None for value in normalized):
        raise ValueError(f"Missing {kind.upper()} label found in model vocabulary.")
    return [str(value) for value in normalized]


def load_stage04_nn(
    model_dir: Path,
    kind: str,
    device: torch.device,
    sentence_encoder_path: Optional[Path] = None,
) -> Tuple[StoredMLP, SentenceTransformer, List[str], int]:
    path = checkpoint_path(model_dir, kind)
    checkpoint = safe_torch_load(path, "cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected dictionary checkpoint: {path}")
    labels = labels_from_checkpoint(checkpoint, model_dir, kind)
    state = normalize_state_keys(extract_state_dict(checkpoint))
    model = StoredMLP(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint architecture mismatch for {model_dir}: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    model.to(device).eval()
    model_name = str(checkpoint.get("embed_model_name", MODEL_NAME))
    encoder_source = (
        str(Path(sentence_encoder_path).expanduser().resolve())
        if sentence_encoder_path is not None
        else model_name
    )
    embedder = SentenceTransformer(encoder_source, device=str(device))
    embedding_dimension = int(
        embedder.get_embedding_dimension()
        if hasattr(embedder, "get_embedding_dimension")
        else embedder.get_sentence_embedding_dimension()
    )
    input_dim = int(checkpoint.get("input_dim", embedding_dimension))
    if input_dim not in (embedding_dimension, embedding_dimension * 2):
        raise ValueError(
            f"Checkpoint expects {input_dim} features, but {model_name} produces "
            f"{embedding_dimension}; supported Stage-04 widths are description-only "
            f"({embedding_dimension}) and description+keywords "
            f"({embedding_dimension * 2})."
        )
    return model, embedder, labels, input_dim


@torch.inference_mode()
def predict_nn_scores(
    model: StoredMLP,
    embedder: SentenceTransformer,
    texts: Sequence[str],
    input_dim: int,
    device: torch.device,
    batch_size: int,
    keyword_cells: Optional[Sequence[Any]] = None,
) -> np.ndarray:
    description_features = embedder.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    embedding_dimension = description_features.shape[1]
    if input_dim == embedding_dimension:
        features = description_features
    elif input_dim == embedding_dimension * 2:
        if keyword_cells is None or len(keyword_cells) != len(texts):
            raise ValueError(
                "A description+keywords checkpoint requires one keyphrase cell per text."
            )
        parsed_rows: List[List[str]] = []
        for value in keyword_cells:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                parsed_rows.append([])
                continue
            raw = str(value).strip()
            try:
                parsed = ast.literal_eval(raw)
            except Exception:
                parsed = None
            candidates = (
                list(parsed)
                if isinstance(parsed, (list, tuple, set))
                else re.split(r"[;,|\n]+", raw)
            )
            parsed_rows.append([
                " ".join(str(item).strip().split())
                for item in candidates
                if str(item).strip()
            ])
        keyword_features = np.zeros_like(description_features)
        flat_phrases: List[str] = []
        row_indices: List[int] = []
        for row_index, phrases in enumerate(parsed_rows):
            flat_phrases.extend(phrases)
            row_indices.extend([row_index] * len(phrases))
        if flat_phrases:
            phrase_features = embedder.encode(
                flat_phrases,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            counts = np.zeros(len(texts), dtype=np.float32)
            for phrase_feature, row_index in zip(phrase_features, row_indices):
                keyword_features[row_index] += phrase_feature
                counts[row_index] += 1.0
            present = counts > 0
            keyword_features[present] /= counts[present, None]
            norms = np.linalg.norm(keyword_features, axis=1, keepdims=True)
            keyword_features = np.divide(
                keyword_features,
                np.maximum(norms, 1e-12),
                out=np.zeros_like(keyword_features),
            )
        features = np.concatenate(
            [description_features, keyword_features], axis=1
        ).astype(np.float32)
    else:
        raise ValueError(f"Unsupported checkpoint input width: {input_dim}")
    if features.shape[1] != input_dim:
        raise ValueError(f"Embedding width {features.shape[1]} != checkpoint input {input_dim}")
    pieces: List[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        batch = torch.from_numpy(features[start:start + batch_size]).to(device)
        logits = model(batch)
        pieces.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(pieces).astype(np.float32)


# =============================================================================
# 3. SCORE CACHING, LABEL ALIGNMENT, CWE MAPPING, AND RAE INFERENCE
# =============================================================================


def load_labels_from_json(path: Path) -> List[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return [str(raw[str(index)] if str(index) in raw else raw[index]) for index in range(len(raw))]
    return [str(item) for item in raw]


def align_scores(
    scores: np.ndarray,
    source_labels: Sequence[str],
    target_labels: Sequence[str],
    fill_value: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if scores.shape[1] != len(source_labels):
        raise ValueError(f"Score columns {scores.shape[1]} != source labels {len(source_labels)}")
    source = {label: index for index, label in enumerate(source_labels)}
    target_set = set(target_labels)
    output = np.full((len(scores), len(target_labels)), fill_value, dtype=np.float32)
    matched = []
    for target_index, label in enumerate(target_labels):
        if label in source:
            output[:, target_index] = scores[:, source[label]]
            matched.append(label)
    report = {
        "matched_labels": len(matched),
        "missing_from_source": [label for label in target_labels if label not in source],
        "extra_in_source": [label for label in source_labels if label not in target_set],
    }
    return output, report


def align_scores_from_cache(
    scores: np.ndarray,
    source_labels: Sequence[str],
    target_labels: Sequence[str],
    cache_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Align native scores, or accept a cache already saved in target order.

    Completed Our Approach runs save their RAE matrices after alignment to the
    direct-CAPEC vocabulary.  When one of those runs is used as a score cache,
    its ``idx_to_label.json`` therefore describes the cached matrix rather than
    the native RAE vocabulary.
    """
    if cache_dir is not None:
        cached_labels_path = Path(cache_dir) / "idx_to_label.json"
        if cached_labels_path.exists():
            cached_labels = load_labels_from_json(cached_labels_path)
            if (
                scores.shape[1] == len(cached_labels)
                and cached_labels == list(target_labels)
            ):
                _, report = align_scores(
                    np.empty((0, len(source_labels)), dtype=np.float32),
                    source_labels,
                    target_labels,
                )
                report["cache_already_aligned"] = True
                report["cache_vocabulary"] = str(cached_labels_path.resolve())
                return scores.astype(np.float32, copy=False), report

    aligned, report = align_scores(scores, source_labels, target_labels)
    report["cache_already_aligned"] = False
    return aligned, report


def mapping_path() -> Path:
    if LOCAL_MAPPING_PATH.exists():
        return LOCAL_MAPPING_PATH
    if FALLBACK_MAPPING_PATH.exists():
        return FALLBACK_MAPPING_PATH
    raise FileNotFoundError(
        f"CWE_MAPPED.csv not found at {LOCAL_MAPPING_PATH} or {FALLBACK_MAPPING_PATH}"
    )


def load_cwe_mapping(path: Path) -> Dict[str, List[str]]:
    frame = pd.read_csv(path)
    missing = [column for column in ("cwe_id", "capec_ids") if column not in frame]
    if missing:
        raise ValueError(f"{path} is missing mapping columns: {missing}")
    result: Dict[str, set[str]] = defaultdict(set)
    for _, row in frame.iterrows():
        cwe = normalize_cwe(row["cwe_id"])
        if cwe:
            result[cwe].update(parse_capec_list(row["capec_ids"]))
    return {key: sorted(values) for key, values in result.items()}


def compute_mapping_scores(
    cwe_scores: np.ndarray,
    cwe_labels: Sequence[str],
    capec_labels: Sequence[str],
    mapping: Mapping[str, Sequence[str]],
    top_k: int,
    rank_weight: bool,
) -> np.ndarray:
    capec_to_idx = {label: index for index, label in enumerate(capec_labels)}
    output = np.zeros((len(cwe_scores), len(capec_labels)), dtype=np.float32)
    k = min(top_k, cwe_scores.shape[1])
    top = np.argpartition(-cwe_scores, kth=k - 1, axis=1)[:, :k]
    ordered = np.take_along_axis(
        top, np.argsort(-np.take_along_axis(cwe_scores, top, axis=1), axis=1), axis=1
    )
    for row in range(len(cwe_scores)):
        for rank, cwe_index in enumerate(ordered[row], start=1):
            support = float(cwe_scores[row, cwe_index])
            if rank_weight:
                support /= math.log2(rank + 1)
            for capec in mapping.get(cwe_labels[int(cwe_index)], ()):
                index = capec_to_idx.get(capec)
                if index is not None and support > output[row, index]:
                    output[row, index] = support
    return output


def latest_valid_rae_run(explicit: Optional[Path]) -> Path:
    required = {
        "best_model.pt", "model_config.json", "retrieval_embeddings.npy",
        "retrieval_memory_labels.npz", "label_to_idx.json", "idx_to_label.json",
        "calibration_parameters.json", "retrieval_backend.json",
    }
    if explicit is not None:
        candidates = [Path(explicit)]
    else:
        candidates = [OUTPUT_ROOT / "RAE_XMC"]
    for candidate in candidates:
        if (
            candidate.is_dir()
            and all((candidate / name).exists() for name in required)
            and (candidate / "tokenizer").is_dir()
            and (candidate / "encoder_config").is_dir()
        ):
            return candidate.resolve()
    raise FileNotFoundError("No complete RAE-XMC run directory was found.")


def import_rae_module() -> Any:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    return importlib.import_module("rae_xmc")


@torch.inference_mode()
def predict_rae_scores(run_dir: Path, texts: Sequence[str], device: torch.device, batch_size: int) -> Tuple[np.ndarray, List[str], str]:
    """Use the saved encoder and retrieval memory; never rebuild either."""
    rae = import_rae_module()
    loaded = rae.load_rae_xmc(run_dir, device=str(device))
    backend = loaded.index.backend
    all_scores: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        encoded = loaded.tokenizer(
            batch, padding="max_length", truncation=True,
            max_length=loaded.config.max_length, return_tensors="pt",
        )
        queries = loaded.model(
            encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
        ).float().cpu().numpy()
        neighbor_scores, neighbor_indices = loaded.index.search(queries, loaded.config.retrieval_k)
        instance, label, _ = rae.aggregate_rae_components(
            neighbor_scores, neighbor_indices, loaded.num_train, loaded.num_labels,
            loaded.offsets, loaded.flat_labels, loaded.parameters["temperature"], False,
            loaded.label_memory_ids,
        )
        all_scores.append(
            rae.combine_rae_scores(instance, label, loaded.parameters["lambda"]).astype(np.float32)
        )
    labels = [loaded.idx_to_label[index] for index in range(loaded.num_labels)]
    return np.concatenate(all_scores), labels, backend


def cache_ids_path(cache_dir: Path, split: str) -> Path:
    return cache_dir / f"{split}_cve_ids.json"


def load_run_cache(cache_dir: Optional[Path], split: str, branch: str, expected_ids: Sequence[str]) -> Optional[np.ndarray]:
    if cache_dir is None:
        return None
    score_path = Path(cache_dir) / f"{split}_{branch}_scores.npy"
    ids_path = cache_ids_path(Path(cache_dir), split)
    if not score_path.exists() or not ids_path.exists():
        return None
    saved_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if saved_ids != list(expected_ids):
        print(f"Ignoring stale {branch} cache: CVE order differs for {split}.")
        return None
    values = np.load(score_path)
    if values.shape[0] != len(expected_ids):
        return None
    print(f"Reusing {score_path}")
    return values.astype(np.float32, copy=False)


# =============================================================================
# 4. VALIDATION CALIBRATION AND FUSION SEARCH
# =============================================================================


def fit_platt(scores: np.ndarray, truth: np.ndarray, seed: int, negative_ratio: int) -> Dict[str, Any]:
    flat_scores = scores.reshape(-1)
    flat_truth = truth.reshape(-1).astype(np.uint8)
    positive = np.flatnonzero(flat_truth == 1)
    negative = np.flatnonzero(flat_truth == 0)
    rng = np.random.default_rng(seed)
    maximum_negative = min(len(negative), max(1, len(positive) * negative_ratio))
    if len(negative) > maximum_negative:
        negative = rng.choice(negative, maximum_negative, replace=False)
    selected = np.concatenate([positive, negative])
    if len(np.unique(flat_truth[selected])) < 2 or float(np.ptp(flat_scores[selected])) < 1e-12:
        prevalence = float(flat_truth[selected].mean()) if len(selected) else 0.5
        coefficient = 1.0
        intercept = float(np.log(np.clip(prevalence, 1e-6, 1 - 1e-6) / np.clip(1 - prevalence, 1e-6, 1)))
        status = "constant-score fallback"
    else:
        model = LogisticRegression(max_iter=200, random_state=seed)
        model.fit(flat_scores[selected, None], flat_truth[selected])
        coefficient = float(model.coef_[0, 0])
        intercept = float(model.intercept_[0])
        status = "fitted"
    return {
        "method": "global validation-fitted Platt scaling",
        "coefficient": coefficient,
        "intercept": intercept,
        "status": status,
        "sampled_positive_count": len(positive),
        "sampled_negative_count": len(negative),
        "score_is_calibrated_probability": True,
    }


def apply_platt(scores: np.ndarray, calibration: Mapping[str, Any]) -> np.ndarray:
    return expit(
        float(calibration["coefficient"]) * scores + float(calibration["intercept"])
    ).astype(np.float32)


def postprocess(scores: np.ndarray, threshold: float, no_id: int) -> np.ndarray:
    prediction = (scores >= threshold).astype(np.uint8)
    real = np.array([index for index in range(scores.shape[1]) if index != no_id])
    has_real = prediction[:, real].any(axis=1)
    noid_wins = (
        prediction[:, no_id].astype(bool)
        & (scores[:, no_id] >= scores[:, real].max(axis=1))
    )
    prediction[noid_wins] = 0
    prediction[noid_wins, no_id] = 1
    prediction[has_real & ~noid_wins, no_id] = 0
    empty = ~prediction.any(axis=1)
    prediction[empty, no_id] = 1
    return prediction


def micro_f1_fast(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth_bool, pred_bool = truth.astype(bool), prediction.astype(bool)
    tp = np.logical_and(truth_bool, pred_bool).sum(dtype=np.int64)
    fp = np.logical_and(~truth_bool, pred_bool).sum(dtype=np.int64)
    fn = np.logical_and(truth_bool, ~pred_bool).sum(dtype=np.int64)
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def best_threshold(scores: np.ndarray, truth: np.ndarray, no_id: int, thresholds: Sequence[float]) -> Tuple[float, float]:
    best = (-1.0, float(thresholds[0]))
    for threshold in thresholds:
        score = micro_f1_fast(truth, postprocess(scores, threshold, no_id))
        candidate = (score, -float(threshold))
        if candidate > (best[0], -best[1]):
            best = (score, float(threshold))
    return best[1], best[0]


def validation_search_diagnostics(
    truth: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    no_id: int,
) -> Dict[str, float]:
    """Requested compact diagnostics for each alpha/beta search point.

    Hit@K exactly follows NN_BASE/topk.py: exclude CAPEC-noID, skip rows
    without a real CAPEC, and count a hit when any true label is in top K.
    """
    prediction = postprocess(scores, threshold, no_id)
    result = {
        "exact_match_accuracy": float(np.all(truth == prediction, axis=1).mean()),
        "micro_f1": float(f1_score(truth, prediction, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
        "sample_f1": float(f1_score(truth, prediction, average="samples", zero_division=0)),
    }
    real = np.array([index for index in range(scores.shape[1]) if index != no_id])
    rows = np.flatnonzero(truth[:, real].sum(axis=1) > 0)
    result["evaluated_real_capec_rows"] = float(len(rows))
    if len(rows):
        order = np.argsort(-scores[rows][:, real], axis=1)
        selected_truth = truth[rows][:, real]
        for k in RANKING_KS:
            relevance = np.take_along_axis(selected_truth, order[:, :min(k, len(real))], axis=1)
            result[f"hit@{k}"] = float((relevance.sum(axis=1) > 0).mean())
    else:
        for k in RANKING_KS:
            result[f"hit@{k}"] = 0.0
    return result


def search_alpha(
    direct: np.ndarray,
    mapping: np.ndarray,
    truth: np.ndarray,
    no_id: int,
    config: RunConfig,
) -> Tuple[float, float, np.ndarray, List[Dict[str, Any]]]:
    rows, best_key, best_values = [], None, None
    for alpha in config.alpha_values:
        scores = (alpha * direct + (1.0 - alpha) * mapping).astype(np.float32)
        threshold, objective = best_threshold(scores, truth, no_id, config.threshold_values)
        row = {
            "stage": "alpha_old", "alpha_old": alpha, "beta": None,
            "threshold": threshold, "validation_micro_f1": objective,
            **validation_search_diagnostics(truth, scores, threshold, no_id),
        }
        rows.append(row)
        key = (objective, -abs(alpha - 0.5), -alpha)
        if best_key is None or key > best_key:
            best_key, best_values = key, (float(alpha), threshold, scores)
    assert best_values is not None
    return best_values[0], best_values[1], best_values[2], rows


def search_beta(
    rae: np.ndarray,
    hybrid: np.ndarray,
    truth: np.ndarray,
    no_id: int,
    config: RunConfig,
) -> Tuple[float, float, np.ndarray, List[Dict[str, Any]]]:
    rows, best_key, best_values = [], None, None
    for beta in config.beta_values:
        scores = (beta * rae + (1.0 - beta) * hybrid).astype(np.float32)
        threshold, objective = best_threshold(scores, truth, no_id, config.threshold_values)
        row = {
            "stage": "beta", "alpha_old": None, "beta": beta,
            "threshold": threshold, "validation_micro_f1": objective,
            **validation_search_diagnostics(truth, scores, threshold, no_id),
        }
        rows.append(row)
        key = (objective, -abs(beta - 0.5), -beta)
        if best_key is None or key > best_key:
            best_key, best_values = key, (float(beta), threshold, scores)
    assert best_values is not None
    return best_values[0], best_values[1], best_values[2], rows


def fusion_score_diagnostic(
    beta: float,
    validation_final: np.ndarray,
    test_final: np.ndarray,
    validation_rae_calibrated: np.ndarray,
    test_rae_calibrated: np.ndarray,
    validation_cgdpf: np.ndarray,
    test_cgdpf: np.ndarray,
) -> Dict[str, Any]:
    """Verify endpoint fusion semantics and report score reconstruction error."""
    if beta == 1.0:
        reference_name = "RAE_XMC_calibrated"
        validation_reference = validation_rae_calibrated
        test_reference = test_rae_calibrated
        endpoint_assertion_performed = True
    elif beta == 0.0:
        reference_name = "CGDPF"
        validation_reference = validation_cgdpf
        test_reference = test_cgdpf
        endpoint_assertion_performed = True
    else:
        reference_name = "beta_weighted_reconstruction"
        validation_reference = (
            beta * validation_rae_calibrated
            + (1.0 - beta) * validation_cgdpf
        ).astype(np.float32)
        test_reference = (
            beta * test_rae_calibrated
            + (1.0 - beta) * test_cgdpf
        ).astype(np.float32)
        endpoint_assertion_performed = False

    validation_difference = float(
        np.max(np.abs(validation_final - validation_reference))
    )
    test_difference = float(np.max(np.abs(test_final - test_reference)))
    rtol, atol = 1e-7, 1e-8
    validation_equal = bool(
        np.allclose(validation_final, validation_reference, rtol=rtol, atol=atol)
    )
    test_equal = bool(
        np.allclose(test_final, test_reference, rtol=rtol, atol=atol)
    )
    if endpoint_assertion_performed:
        assert validation_equal, (
            f"beta={beta} validation fusion does not equal {reference_name}; "
            f"max absolute difference={validation_difference}"
        )
        assert test_equal, (
            f"beta={beta} test fusion does not equal {reference_name}; "
            f"max absolute difference={test_difference}"
        )
    return {
        "selected_beta": float(beta),
        "reference": reference_name,
        "endpoint_assertion_performed": endpoint_assertion_performed,
        "validation_numerically_equal": validation_equal,
        "test_numerically_equal": test_equal,
        "validation_max_absolute_score_difference": validation_difference,
        "test_max_absolute_score_difference": test_difference,
        "rtol": rtol,
        "atol": atol,
    }


# =============================================================================
# 5. FINAL METRICS AND TABLES
# =============================================================================


def threshold_metrics(truth: np.ndarray, scores: np.ndarray, threshold: float, no_id: int) -> Tuple[Dict[str, Any], np.ndarray]:
    prediction = postprocess(scores, threshold, no_id)
    metrics = {
        "threshold": float(threshold),
        "exact_match_accuracy": float(accuracy_score(truth, prediction)),
        "micro_precision": float(precision_score(truth, prediction, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(truth, prediction, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(truth, prediction, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(truth, prediction, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, prediction, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
        "sample_precision": float(precision_score(truth, prediction, average="samples", zero_division=0)),
        "sample_recall": float(recall_score(truth, prediction, average="samples", zero_division=0)),
        "sample_f1": float(f1_score(truth, prediction, average="samples", zero_division=0)),
        "sample_jaccard": float(jaccard_score(truth, prediction, average="samples", zero_division=0)),
        "hamming_loss": float(hamming_loss(truth, prediction)),
        "label_ranking_average_precision": float(label_ranking_average_precision_score(truth, scores)),
        "average_predicted_labels": float(prediction.sum(axis=1).mean()),
    }
    return metrics, prediction


def ranking_metrics(truth: np.ndarray, scores: np.ndarray, no_id: int) -> Dict[str, Any]:
    real = np.array([index for index in range(scores.shape[1]) if index != no_id])
    rows = np.flatnonzero(truth[:, real].sum(axis=1) > 0)
    result: Dict[str, Any] = {"evaluated_real_capec_rows": int(len(rows))}
    if not len(rows):
        return result
    selected_truth = truth[rows][:, real]
    selected_scores = scores[rows][:, real]
    order = np.argsort(-selected_scores, axis=1)
    for requested_k in RANKING_KS:
        k = min(requested_k, len(real))
        relevance = np.take_along_axis(selected_truth, order[:, :k], axis=1)
        hits = relevance.sum(axis=1)
        result[f"hit@{requested_k}"] = float((hits > 0).mean())
        result[f"precision@{requested_k}"] = float((hits / requested_k).mean())
        result[f"recall@{requested_k}"] = float((hits / selected_truth.sum(axis=1)).mean())
        discount = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = (relevance * discount).sum(axis=1)
        ideal = np.array([discount[:min(k, int(count))].sum() for count in selected_truth.sum(axis=1)])
        result[f"ndcg@{requested_k}"] = float(np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0).mean())
    return result


def evaluate(truth: np.ndarray, scores: np.ndarray, threshold: float, no_id: int) -> Tuple[Dict[str, Any], np.ndarray]:
    metrics, prediction = threshold_metrics(truth, scores, threshold, no_id)
    metrics.update(ranking_metrics(truth, scores, no_id))
    return metrics, prediction


def save_prediction_tables(
    run_dir: Path,
    test: pd.DataFrame,
    truth: np.ndarray,
    prediction: np.ndarray,
    scores: np.ndarray,
    labels: Sequence[str],
    no_id: int,
) -> None:
    real = np.array([index for index in range(len(labels)) if index != no_id])
    ranked = real[np.argsort(-scores[:, real], axis=1)]
    threshold_rows, top_rows = [], []
    for row in range(len(test)):
        common = {
            "row_index": row,
            "cve_id": test.iloc[row][CVE_COL],
            "true_labels": json.dumps([labels[index] for index in np.flatnonzero(truth[row])]),
        }
        selected = np.flatnonzero(prediction[row])
        threshold_rows.append({
            **common,
            "predicted_labels": json.dumps([labels[index] for index in selected]),
            "predicted_scores": json.dumps({labels[index]: float(scores[row, index]) for index in selected}),
        })
        ranked_values = [
            {"capec_id": labels[index], "score": float(scores[row, index])}
            for index in ranked[row, :max(RANKING_KS)]
        ]
        true_real = set(np.flatnonzero(truth[row]).tolist()) - {no_id}
        top_row: Dict[str, Any] = {
            **common,
            "ranked_predictions": json.dumps(ranked_values),
        }
        for k in RANKING_KS:
            top_row[f"top_{k}_predictions"] = json.dumps(ranked_values[:k])
            top_indices = set(ranked[row, :k].tolist())
            top_row[f"hit_at_{k}"] = int(bool(true_real & top_indices)) if true_real else None
        top_rows.append(top_row)
    pd.DataFrame(threshold_rows).to_csv(run_dir / "test_predictions.csv", index=False)
    pd.DataFrame(top_rows).to_csv(run_dir / "topk_predictions.csv", index=False)


# =============================================================================
# 6. END-TO-END FUSION PIPELINE
# =============================================================================


def run_pipeline(use_keywords: bool = False) -> Path:
    started = time.time()
    config = RunConfig()
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    case_name = "our_approach_keywords" if use_keywords else "our_approach"
    cwe_model_dir = (
        OUTPUT_ROOT
        / "CVE_to_CWE"
        / ("NN_keywords" if use_keywords else "NN_1way")
    )
    capec_model_dir = OUTPUT_ROOT / ("NN_keywords" if use_keywords else "NN")
    run_dir = RUNS_DIR / f"{case_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print("Device:", device)
    print("Run directory:", run_dir)

    train, validation, test = load_fixed_splits(DATASET_PATH, config.seed)
    validation_ids, test_ids = cve_ids(validation), cve_ids(test)
    save_json(cache_ids_path(run_dir, "validation"), validation_ids)
    save_json(cache_ids_path(run_dir, "test"), test_ids)

    rae_run = latest_valid_rae_run(
        RAE_XMC_KEYWORDS_RUN_DIR if use_keywords else RAE_XMC_RUN_DIR
    )
    map_path = mapping_path()
    for directory, name in ((cwe_model_dir, "CWE"), (capec_model_dir, "direct CAPEC")):
        if not Path(directory).exists():
            raise FileNotFoundError(f"{name} model directory not found: {directory}")

    # Load label vocabularies before deciding whether cached score matrices can
    # be reused. The direct model's actual vocabulary is the common order.
    cwe_checkpoint = safe_torch_load(checkpoint_path(cwe_model_dir, "cwe"), "cpu")
    capec_checkpoint = safe_torch_load(checkpoint_path(capec_model_dir, "capec"), "cpu")
    cwe_labels = labels_from_checkpoint(cwe_checkpoint, cwe_model_dir, "cwe")
    direct_labels = labels_from_checkpoint(capec_checkpoint, capec_model_dir, "capec")
    if NO_ID_LABEL not in direct_labels:
        raise ValueError(f"Direct CAPEC vocabulary does not contain {NO_ID_LABEL}.")
    source_capec_vocabulary = validate_complete_capec_vocabulary(direct_labels)
    labels = canonical_capec_vocabulary()
    capec_vocabulary = validate_complete_capec_vocabulary(labels)
    no_id = labels.index(NO_ID_LABEL)
    y_train, unknown_train = make_truth(train, labels)
    y_val, unknown_val = make_truth(validation, labels)
    y_test, unknown_test = make_truth(test, labels)

    score_cache = Path(SCORE_CACHE_RUN).resolve() if SCORE_CACHE_RUN is not None else None
    rae_score_cache = (
        Path(RAE_SCORE_CACHE_RUN).resolve()
        if RAE_SCORE_CACHE_RUN is not None else score_cache
    )
    val_cwe = load_run_cache(score_cache, "validation", "cwe", validation_ids)
    test_cwe = load_run_cache(score_cache, "test", "cwe", test_ids)
    val_direct_native = load_run_cache(score_cache, "validation", "direct", validation_ids)
    test_direct_native = load_run_cache(score_cache, "test", "direct", test_ids)
    val_direct_cache = score_cache if val_direct_native is not None else None
    test_direct_cache = score_cache if test_direct_native is not None else None
    if val_cwe is None or test_cwe is None:
        print("Loading supplied CWE model for missing score matrices...")
        model, embedder, loaded_labels, input_dim = load_stage04_nn(
            cwe_model_dir, "cwe", device
        )
        if loaded_labels != cwe_labels:
            raise RuntimeError("CWE label order changed while loading checkpoint.")
        if val_cwe is None:
            val_cwe = predict_nn_scores(
                model, embedder, validation[DESC_COL].tolist(), input_dim,
                device, config.batch_size,
                validation[KEYPHRASE_COL].tolist() if use_keywords else None,
            )
        if test_cwe is None:
            test_cwe = predict_nn_scores(
                model, embedder, test[DESC_COL].tolist(), input_dim,
                device, config.batch_size,
                test[KEYPHRASE_COL].tolist() if use_keywords else None,
            )
        del model, embedder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if val_direct_native is None or test_direct_native is None:
        print("Loading supplied direct CAPEC model for missing score matrices...")
        model, embedder, loaded_direct_labels, input_dim = load_stage04_nn(
            capec_model_dir, "capec", device
        )
        if loaded_direct_labels != direct_labels:
            raise RuntimeError("Direct CAPEC label order changed while loading checkpoint.")
        if val_direct_native is None:
            val_direct_native = predict_nn_scores(
                model, embedder, validation[DESC_COL].tolist(), input_dim,
                device, config.batch_size,
                validation[KEYPHRASE_COL].tolist() if use_keywords else None,
            )
        if test_direct_native is None:
            test_direct_native = predict_nn_scores(
                model, embedder, test[DESC_COL].tolist(), input_dim,
                device, config.batch_size,
                test[KEYPHRASE_COL].tolist() if use_keywords else None,
            )
        del model, embedder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert val_cwe is not None and test_cwe is not None
    assert val_direct_native is not None and test_direct_native is not None
    val_direct, direct_alignment = align_scores_from_cache(
        val_direct_native, direct_labels, labels, val_direct_cache
    )
    test_direct, _ = align_scores_from_cache(
        test_direct_native, direct_labels, labels, test_direct_cache
    )
    mapping = load_cwe_mapping(map_path)
    val_mapping = compute_mapping_scores(
        val_cwe, cwe_labels, labels, mapping,
        config.cwe_top_k_for_mapping, config.use_rank_weight,
    )
    test_mapping = compute_mapping_scores(
        test_cwe, cwe_labels, labels, mapping,
        config.cwe_top_k_for_mapping, config.use_rank_weight,
    )

    val_rae_native = load_run_cache(
        rae_score_cache, "validation", "rae", validation_ids
    )
    test_rae_native = load_run_cache(
        rae_score_cache, "test", "rae", test_ids
    )
    val_rae_cache = rae_score_cache if val_rae_native is not None else None
    test_rae_cache = rae_score_cache if test_rae_native is not None else None
    rae_labels = load_labels_from_json(rae_run / "idx_to_label.json")
    rae_backend = json.loads((rae_run / "retrieval_backend.json").read_text(encoding="utf-8")).get("backend")
    if val_rae_native is None:
        print("Running one validation RAE-XMC retrieval pass...")
        val_rae_native, loaded_rae_labels, rae_backend = predict_rae_scores(
            rae_run, rae_input_texts(validation, use_keywords),
            device, config.batch_size,
        )
        if loaded_rae_labels != rae_labels:
            raise RuntimeError("RAE label order changed during loading.")
    if test_rae_native is None:
        print("Running one test RAE-XMC retrieval pass...")
        test_rae_native, loaded_rae_labels, rae_backend = predict_rae_scores(
            rae_run, rae_input_texts(test, use_keywords),
            device, config.batch_size,
        )
        if loaded_rae_labels != rae_labels:
            raise RuntimeError("RAE label order changed during loading.")
    val_rae, rae_alignment = align_scores_from_cache(
        val_rae_native, rae_labels, labels, val_rae_cache
    )
    test_rae, _ = align_scores_from_cache(
        test_rae_native, rae_labels, labels, test_rae_cache
    )

    # Full raw score matrices are cached immediately after inference.
    raw_matrices = {
        "validation_cwe_scores.npy": val_cwe,
        "test_cwe_scores.npy": test_cwe,
        "validation_direct_scores.npy": val_direct,
        "test_direct_scores.npy": test_direct,
        "validation_mapping_scores.npy": val_mapping,
        "test_mapping_scores.npy": test_mapping,
        "validation_rae_scores.npy": val_rae,
        "test_rae_scores.npy": test_rae,
    }
    for filename, matrix in raw_matrices.items():
        np.save(run_dir / filename, np.asarray(matrix, dtype=np.float32))

    calibrations = {
        "direct": fit_platt(val_direct, y_val, config.seed + 1, config.calibration_negative_ratio),
        "mapping": fit_platt(val_mapping, y_val, config.seed + 2, config.calibration_negative_ratio),
        "rae_xmc": fit_platt(
            val_rae,
            y_val,
            config.seed + 3,
            config.calibration_negative_ratio,
        ),
    }
    val_calibrated = {
        "direct": apply_platt(val_direct, calibrations["direct"]),
        "mapping": apply_platt(val_mapping, calibrations["mapping"]),
        "rae_xmc": apply_platt(val_rae, calibrations["rae_xmc"]),
    }
    test_calibrated = {
        "direct": apply_platt(test_direct, calibrations["direct"]),
        "mapping": apply_platt(test_mapping, calibrations["mapping"]),
        "rae_xmc": apply_platt(test_rae, calibrations["rae_xmc"]),
    }
    alpha, hybrid_threshold, val_hybrid, alpha_rows = search_alpha(
        val_calibrated["direct"], val_calibrated["mapping"], y_val, no_id, config
    )
    test_hybrid = (
        alpha * test_calibrated["direct"] + (1.0 - alpha) * test_calibrated["mapping"]
    ).astype(np.float32)
    beta, final_threshold, val_final, beta_rows = search_beta(
        val_calibrated["rae_xmc"], val_hybrid, y_val, no_id, config
    )
    test_final = (
        beta * test_calibrated["rae_xmc"] + (1.0 - beta) * test_hybrid
    ).astype(np.float32)
    fusion_diagnostic = fusion_score_diagnostic(
        beta,
        val_final,
        test_final,
        val_calibrated["rae_xmc"],
        test_calibrated["rae_xmc"],
        val_hybrid,
        test_hybrid,
    )
    np.save(run_dir / "validation_original_hybrid_scores.npy", val_hybrid)
    np.save(run_dir / "test_original_hybrid_scores.npy", test_hybrid)
    np.save(
        run_dir / "validation_rae_xmc_calibrated_scores.npy",
        val_calibrated["rae_xmc"],
    )
    np.save(
        run_dir / "test_rae_xmc_calibrated_scores.npy",
        test_calibrated["rae_xmc"],
    )
    np.save(run_dir / "validation_final_fusion_scores.npy", val_final)
    np.save(run_dir / "test_final_fusion_scores.npy", test_final)
    calibration_frame = pd.DataFrame(alpha_rows + beta_rows)
    calibration_frame["selected"] = (
        (
            calibration_frame["stage"].eq("alpha_old")
            & np.isclose(calibration_frame["alpha_old"], alpha, equal_nan=False)
            & np.isclose(calibration_frame["threshold"], hybrid_threshold)
        )
        | (
            calibration_frame["stage"].eq("beta")
            & np.isclose(calibration_frame["beta"], beta, equal_nan=False)
            & np.isclose(calibration_frame["threshold"], final_threshold)
        )
    )
    calibration_frame.to_csv(
        run_dir / "fusion_calibration_results.csv", index=False
    )
    # Backward-compatible filename used by earlier analysis notebooks.
    calibration_frame.to_csv(
        run_dir / "validation_search_results.csv", index=False
    )

    branch_thresholds: Dict[str, float] = {}
    for branch in ("direct", "mapping", "rae_xmc"):
        branch_thresholds[branch], _ = best_threshold(
            val_calibrated[branch], y_val, no_id, config.threshold_values
        )
    branch_thresholds.update({"original_hybrid": hybrid_threshold, "hybrid_plus_rae_xmc": final_threshold})
    validation_scores = {
        "direct": val_calibrated["direct"],
        "cwe_to_capec_mapping": val_calibrated["mapping"],
        "original_hybrid": val_hybrid,
        "rae_xmc_calibrated": val_calibrated["rae_xmc"],
        "hybrid_plus_rae_xmc": val_final,
    }
    test_scores = {
        "direct": test_calibrated["direct"],
        "cwe_to_capec_mapping": test_calibrated["mapping"],
        "original_hybrid": test_hybrid,
        "rae_xmc_calibrated": test_calibrated["rae_xmc"],
        "hybrid_plus_rae_xmc": test_final,
    }
    threshold_lookup = {
        "direct": branch_thresholds["direct"],
        "cwe_to_capec_mapping": branch_thresholds["mapping"],
        "original_hybrid": hybrid_threshold,
        "rae_xmc_calibrated": branch_thresholds["rae_xmc"],
        "hybrid_plus_rae_xmc": final_threshold,
    }

    validation_metrics: Dict[str, Any] = {}
    test_metrics: Dict[str, Any] = {}
    test_predictions: Dict[str, np.ndarray] = {}
    comparison_rows = []
    for model_name in validation_scores:
        val_metrics, _ = evaluate(y_val, validation_scores[model_name], threshold_lookup[model_name], no_id)
        metrics, prediction = evaluate(y_test, test_scores[model_name], threshold_lookup[model_name], no_id)
        validation_metrics[model_name] = val_metrics
        test_metrics[model_name] = metrics
        test_predictions[model_name] = prediction
        comparison_row = {"model": model_name, **metrics}
        if model_name == "rae_xmc_calibrated":
            comparison_row.update({
                "rae_matched_labels": rae_alignment["matched_labels"],
                "rae_missing_from_source": json.dumps(
                    rae_alignment["missing_from_source"]
                ),
                "rae_extra_in_source": json.dumps(
                    rae_alignment["extra_in_source"]
                ),
            })
        comparison_rows.append(comparison_row)

    save_json(run_dir / "validation_metrics.json", validation_metrics)
    save_json(run_dir / "test_metrics.json", test_metrics)
    pd.DataFrame(comparison_rows).to_csv(run_dir / "model_comparison.csv", index=False)
    save_json(run_dir / "calibration_parameters.json", {
        "selection_split": "validation",
        "branches": calibrations,
        "thresholds": branch_thresholds,
        "test_labels_used": False,
    })
    save_json(run_dir / "fusion_parameters.json", {
        "selected_alpha_old": alpha,
        "selected_beta": beta,
        "selected_final_threshold": final_threshold,
        "alpha_definition": "weight on calibrated direct CAPEC scores",
        "beta_definition": "weight on calibrated RAE-XMC scores",
        "score_fusion": "validation-Platt-calibrated weighted arithmetic mean over the full label vocabulary",
    })
    save_json(run_dir / "fusion_score_diagnostic.json", fusion_diagnostic)
    save_json(run_dir / "label_to_idx.json", {label: index for index, label in enumerate(labels)})
    save_json(run_dir / "idx_to_label.json", {str(index): label for index, label in enumerate(labels)})
    save_json(run_dir / "cwe_idx_to_label.json", {str(index): label for index, label in enumerate(cwe_labels)})
    save_json(run_dir / "label_alignment_report.json", {
        "common_vocabulary_source": str(capec_model_dir),
        "num_common_labels": len(labels),
        "direct": direct_alignment,
        "rae_xmc": rae_alignment,
        "rae_xmc_calibrated": {
            "source_model": (
                "RAE_XMC_keywords" if use_keywords else "RAE_XMC"
            ),
            "transformation": (
                "common-vocabulary label alignment followed by "
                "validation-fitted Platt calibration"
            ),
            "source_label_count": len(rae_labels),
            "target_label_count": len(labels),
            "matched_labels": rae_alignment["matched_labels"],
            "missing_from_source": rae_alignment["missing_from_source"],
            "extra_in_source": rae_alignment["extra_in_source"],
        },
        "unknown_truth_labels": {"train": unknown_train, "validation": unknown_val, "test": unknown_test},
        "capec_vocabulary": capec_vocabulary,
        "source_capec_vocabulary": source_capec_vocabulary,
        "mapping_capecs_missing_from_common_vocabulary": sorted({
            capec for values in mapping.values() for capec in values if capec not in set(labels)
        }),
    })
    save_prediction_tables(
        run_dir, test, y_test, test_predictions["hybrid_plus_rae_xmc"],
        test_final, labels, no_id,
    )
    save_json(run_dir / "run_config.json", {
        "config": asdict(config),
        "source_models": {
            "cwe_model_dir": str(cwe_model_dir.resolve()),
            "direct_capec_model_dir": str(capec_model_dir.resolve()),
            "rae_xmc_run_dir": str(rae_run),
            "cwe_mapping_csv": str(map_path.resolve()),
            "optional_score_cache_run": str(score_cache) if score_cache else None,
            "optional_rae_score_cache_run": (
                str(rae_score_cache) if rae_score_cache else None
            ),
        },
        "data": {
            "split_directory": str(DATASET_PATH),
            "split_source": "fixed KeyBERT Modified 70/15/15 files",
            "input_case": (
                "cleaned_description+keybert_modified_keywords"
                if use_keywords else "cleaned_description"
            ),
            "training_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
            "validation_cve_id_digest": id_digest(validation_ids),
            "test_cve_id_digest": id_digest(test_ids),
            "fixed_splits_preserved": True,
        },
        "inference": {
            "models_in_eval_mode": True,
            "gradient_updates": False,
            "rae_retrieval_backend": rae_backend,
            "rae_memory_rebuilt": False,
        },
        "package_versions": package_versions(),
        "processing_seconds": time.time() - started,
    })

    # Required persisted-matrix shape check.
    expected_shapes = {
        "validation_direct_scores.npy": (len(validation), len(labels)),
        "test_direct_scores.npy": (len(test), len(labels)),
        "validation_cwe_scores.npy": (len(validation), len(cwe_labels)),
        "test_cwe_scores.npy": (len(test), len(cwe_labels)),
        "validation_final_fusion_scores.npy": (len(validation), len(labels)),
        "test_final_fusion_scores.npy": (len(test), len(labels)),
        "validation_rae_xmc_calibrated_scores.npy": (len(validation), len(labels)),
        "test_rae_xmc_calibrated_scores.npy": (len(test), len(labels)),
    }
    for filename, expected in expected_shapes.items():
        actual = np.load(run_dir / filename, mmap_mode="r").shape
        if actual != expected:
            raise RuntimeError(f"Saved score matrix {filename} has shape {actual}, expected {expected}")
    save_json(run_dir / "score_matrix_check.json", {
        "passed": True,
        "shapes": expected_shapes,
        "fusion_score_diagnostic": fusion_diagnostic,
    })
    pointer_names = (
        ("latest_our_approach_keywords_run.txt",)
        if use_keywords
        else (
            "latest_our_approach_run.txt",
            "latest_our_approch_run.txt",  # Stage-05 compatibility.
        )
    )
    for pointer_name in pointer_names:
        (RUNS_DIR / pointer_name).write_text(
            str(run_dir.resolve()) + "\n", encoding="utf-8"
        )
    print("Selected alpha_old:", alpha)
    print("Selected beta:", beta)
    print("Final validation threshold:", final_threshold)
    print("Fusion score diagnostic:", fusion_diagnostic)
    display_columns = [
        "model", "exact_match_accuracy", "micro_f1", "macro_f1",
        "weighted_f1", "sample_f1", "hit@1", "hit@2", "hit@5", "hit@10",
    ]
    print("\nFinal test comparison:")
    print(pd.DataFrame(comparison_rows)[display_columns].to_string(index=False))
    print("Saved results to:", run_dir)
    return run_dir


# =============================================================================
# 7. SAVED OUR-APPROACH INFERENCE
# =============================================================================


class LoadedOurApproach:
    """Load the three frozen branches and validation-selected fusion settings."""

    def __init__(
        self,
        run_dir: str | Path,
        device: Optional[str] = None,
        sentence_encoder_path: Optional[str | Path] = None,
    ):
        self.run_dir = Path(run_dir).expanduser().resolve()
        required = (
            "calibration_parameters.json",
            "fusion_parameters.json",
            "idx_to_label.json",
            "cwe_idx_to_label.json",
            "run_config.json",
        )
        missing = [name for name in required if not (self.run_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Incomplete our-approach run {self.run_dir}; missing: {', '.join(missing)}"
            )

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fusion = json.loads(
            (self.run_dir / "fusion_parameters.json").read_text(encoding="utf-8")
        )
        calibration_file = json.loads(
            (self.run_dir / "calibration_parameters.json").read_text(encoding="utf-8")
        )
        self.calibrations = calibration_file["branches"]
        run_config = json.loads((self.run_dir / "run_config.json").read_text(encoding="utf-8"))
        sources = run_config["source_models"]
        self.input_case = str(run_config.get("data", {}).get("input_case", ""))
        self.use_keywords = "keywords" in self.input_case.casefold()

        self.labels = load_labels_from_json(self.run_dir / "idx_to_label.json")
        self.cwe_labels = load_labels_from_json(self.run_dir / "cwe_idx_to_label.json")
        if NO_ID_LABEL not in self.labels:
            raise ValueError(f"Saved vocabulary does not contain {NO_ID_LABEL}.")
        self.no_id = self.labels.index(NO_ID_LABEL)

        def source_path(value: str) -> Path:
            path = Path(value).expanduser()
            return path.resolve() if path.is_absolute() else (self.run_dir / path).resolve()

        cwe_dir = source_path(sources["cwe_model_dir"])
        capec_dir = source_path(sources["direct_capec_model_dir"])
        configured_encoder = sentence_encoder_path or sources.get("sentence_encoder_dir")
        encoder_path = (
            source_path(str(configured_encoder)) if configured_encoder else None
        )
        if encoder_path is not None and not encoder_path.is_dir():
            raise FileNotFoundError(f"Sentence encoder directory not found: {encoder_path}")
        cwe_checkpoint = safe_torch_load(checkpoint_path(cwe_dir, "cwe"), "cpu")
        capec_checkpoint = safe_torch_load(checkpoint_path(capec_dir, "capec"), "cpu")
        cwe_encoder_name = str(cwe_checkpoint.get("embed_model_name", MODEL_NAME))
        capec_encoder_name = str(capec_checkpoint.get("embed_model_name", MODEL_NAME))
        if cwe_encoder_name != capec_encoder_name:
            raise ValueError("CWE and CAPEC checkpoints require different embedding models.")

        self.cwe_model, self.embedder, loaded_cwes, self.input_dim = load_stage04_nn(
            cwe_dir, "cwe", self.device, sentence_encoder_path=encoder_path
        )
        self.capec_model, duplicate_embedder, loaded_capecs, capec_input_dim = load_stage04_nn(
            capec_dir, "capec", self.device, sentence_encoder_path=encoder_path
        )
        del duplicate_embedder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if loaded_cwes != self.cwe_labels:
            raise RuntimeError("Saved CWE vocabulary differs from its component checkpoint.")
        validate_complete_capec_vocabulary(loaded_capecs)
        self.direct_labels = loaded_capecs
        if capec_input_dim != self.input_dim:
            raise RuntimeError("CWE and CAPEC checkpoints require different feature widths.")

        self.mapping = load_cwe_mapping(source_path(sources["cwe_mapping_csv"]))
        self.rae_module = import_rae_module()
        self.rae = self.rae_module.load_rae_xmc(
            source_path(sources["rae_xmc_run_dir"]), device=str(self.device)
        )
        if self.use_keywords and not self.rae.config.use_keyphrases:
            raise ValueError(
                "Keyword Our Approach requires a keyword-trained RAE-XMC run."
            )
        self.rae_labels = [
            self.rae.idx_to_label[index] for index in range(self.rae.num_labels)
        ]

    @torch.inference_mode()
    def predict(
        self,
        description: str,
        top_k: int = 10,
        keywords: Optional[Sequence[str] | str] = None,
    ) -> Dict[str, Any]:
        text = " ".join(str(description or "").strip().split())
        if not text:
            raise ValueError("description must be non-empty")
        top_k = max(1, min(int(top_k), len(self.labels) - 1))

        keyword_cells = [keywords] if keywords is not None else None
        cwe_raw = predict_nn_scores(
            self.cwe_model,
            self.embedder,
            [text],
            self.input_dim,
            self.device,
            batch_size=1,
            keyword_cells=keyword_cells,
        )
        direct_native = predict_nn_scores(
            self.capec_model,
            self.embedder,
            [text],
            self.input_dim,
            self.device,
            batch_size=1,
            keyword_cells=keyword_cells,
        )
        direct_raw, _ = align_scores(
            direct_native, self.direct_labels, self.labels
        )
        mapping_raw = compute_mapping_scores(
            cwe_raw,
            self.cwe_labels,
            self.labels,
            self.mapping,
            CWE_TOP_K_FOR_MAPPING,
            USE_RANK_WEIGHT,
        )
        rae_native = self.rae.score(
            text, keywords if self.use_keywords else None
        )[None, :].astype(np.float32)
        rae_raw, _ = align_scores(rae_native, self.rae_labels, self.labels)

        direct = apply_platt(direct_raw, self.calibrations["direct"])
        mapping = apply_platt(mapping_raw, self.calibrations["mapping"])
        rae = apply_platt(rae_raw, self.calibrations["rae_xmc"])
        alpha = float(self.fusion["selected_alpha_old"])
        beta = float(self.fusion["selected_beta"])
        threshold = float(self.fusion["selected_final_threshold"])
        cgdpf = alpha * direct + (1.0 - alpha) * mapping
        final = (beta * rae + (1.0 - beta) * cgdpf).astype(np.float32)

        prediction = postprocess(final, threshold, self.no_id)[0]
        predicted_labels = [self.labels[index] for index in np.flatnonzero(prediction)]
        ranked_indices = [
            int(index) for index in np.argsort(-final[0]) if int(index) != self.no_id
        ][:top_k]
        ranked = [
            {"capec_id": self.labels[index], "score": float(final[0, index])}
            for index in ranked_indices
        ]
        return {
            "ranked_capecs": ranked,
            "thresholded_labels": predicted_labels,
            "threshold": threshold,
            "alpha": alpha,
            "beta": beta,
            "input_case": self.input_case,
            "keywords_used": self.use_keywords,
            "score_type": "validation-calibrated CGDPF + RAE-XMC fusion probability",
        }


def load_our_approach(
    run_dir: str | Path,
    device: Optional[str] = None,
    sentence_encoder_path: Optional[str | Path] = None,
) -> LoadedOurApproach:
    """Load a completed saved fusion run for free-text prediction."""
    return LoadedOurApproach(
        run_dir,
        device=device,
        sentence_encoder_path=sentence_encoder_path,
    )


# =============================================================================
# 8. SYNTHETIC ALIGNMENT/FUSION/SAVE-LOAD CHECK
# =============================================================================


def synthetic_test() -> None:
    source_labels = ["CAPEC-2", "CAPEC-noID", "CAPEC-1"]
    target_labels = ["CAPEC-1", "CAPEC-2", "CAPEC-3", "CAPEC-noID"]
    source_scores = np.array([[0.1, 0.2, 0.9], [0.8, 0.7, 0.1]], dtype=np.float32)
    aligned, report = align_scores(source_scores, source_labels, target_labels)
    expected = np.array([[0.9, 0.1, 0.0, 0.2], [0.1, 0.8, 0.0, 0.7]], dtype=np.float32)
    if not np.allclose(aligned, expected) or report["missing_from_source"] != ["CAPEC-3"]:
        raise AssertionError("Label alignment test failed.")
    with tempfile_directory() as cache_dir:
        save_json(
            cache_dir / "idx_to_label.json",
            {str(index): label for index, label in enumerate(target_labels)},
        )
        cached, cached_report = align_scores_from_cache(
            expected, source_labels, target_labels, cache_dir
        )
        if not np.array_equal(cached, expected) or not cached_report["cache_already_aligned"]:
            raise AssertionError("Already-aligned score-cache test failed.")
    direct = aligned
    mapping = np.array([[0.4, 0.2, 0.0, 0.0], [0.2, 0.6, 0.0, 0.0]], dtype=np.float32)
    rae = np.array([[0.8, 0.3, 0.0, 0.1], [0.2, 0.7, 0.0, 0.4]], dtype=np.float32)
    hybrid = 0.5 * direct + 0.5 * mapping
    fused = 0.75 * rae + 0.25 * hybrid
    if not np.allclose(fused, 0.75 * rae + 0.125 * direct + 0.125 * mapping):
        raise AssertionError("Fusion equation test failed.")
    endpoint_one = fusion_score_diagnostic(
        1.0, rae.copy(), rae.copy(), rae, rae, hybrid, hybrid
    )
    endpoint_zero = fusion_score_diagnostic(
        0.0, hybrid.copy(), hybrid.copy(), rae, rae, hybrid, hybrid
    )
    if (
        endpoint_one["reference"] != "RAE_XMC_calibrated"
        or endpoint_zero["reference"] != "CGDPF"
        or endpoint_one["validation_max_absolute_score_difference"] != 0.0
        or endpoint_zero["test_max_absolute_score_difference"] != 0.0
    ):
        raise AssertionError("Fusion endpoint diagnostic test failed.")
    try:
        fusion_score_diagnostic(
            1.0, rae + 1e-3, rae, rae, rae, hybrid, hybrid
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("Fusion endpoint mismatch was not rejected.")
    truth = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.uint8)
    diagnostics = validation_search_diagnostics(truth, fused, 0.5, 3)
    required_metrics = {
        "exact_match_accuracy", "micro_f1", "macro_f1", "weighted_f1",
        "sample_f1", "hit@1", "hit@2", "hit@5", "hit@10",
    }
    if not required_metrics.issubset(diagnostics):
        raise AssertionError("Requested validation-search metrics are missing.")
    with tempfile_directory() as directory:
        path = directory / "scores.npy"
        np.save(path, fused)
        reloaded = np.load(path)
        if reloaded.shape != fused.shape or not np.array_equal(reloaded, fused):
            raise AssertionError("Saved score matrix check failed.")
    print("SYNTHETIC TEST PASSED: label alignment, fusion, and score-matrix reload")


@contextlib.contextmanager
def tempfile_directory() -> Iterable[Path]:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="our_approch_smoke_") as value:
        yield Path(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse supplied CWE/CAPEC hybrid with trained RAE-XMC")
    parser.add_argument("--synthetic-test", action="store_true", help="Run only alignment/fusion/save-load checks")
    parser.add_argument(
        "--keywords",
        action="store_true",
        help="Use modified-KeyBERT CWE/CAPEC branches and keyphrase inputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.synthetic_test:
        synthetic_test()
        return
    run_pipeline(use_keywords=args.keywords)


if __name__ == "__main__":
    main()
