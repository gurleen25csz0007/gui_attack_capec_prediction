#!/usr/bin/env python3

"""Extract and rerank KeyBERT phrases against all CWE/CAPEC targets."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from keybert import KeyBERT
from keybert.backend import SentenceTransformerBackend
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm


# ============================================================
# PATHS AND FIXED SETTINGS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PAPER_DIR = PROJECT_DIR.parents[1]
OUTPUTS_ROOT = Path(os.environ.get(
    "KEYWORD_MAKER_OUTPUT_ROOT",
    PROJECT_DIR / "outputs"
)).expanduser().resolve()

DATASET_PATH = (
    PAPER_DIR
    / "methodology"
    / "02_preprocessing"
    / "dataset.csv"
)
CWE_PATH = (
    PAPER_DIR
    / "methodology"
    / "01_cwe_capec_ground_truth"
    / "cwe_final.csv"
)
CAPEC_PATH = (
    PAPER_DIR
    / "methodology"
    / "01_cwe_capec_ground_truth"
    / "3000_capec.csv"
)
DEFAULT_OUTPUT_DIR = OUTPUTS_ROOT / "keybert_modified"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RANDOM_SEED = 42
TRAIN_SIZE = 0.70

KEYPHRASE_NGRAM_RANGE = (2, 4)
CANDIDATE_TOP_N = 40
FINAL_TOP_N = 5
KEYPHRASE_SOURCE = "keybert_modified"
TARGET_SIM_THRESHOLD_GRID = [0.35, 0.40, 0.45, 0.50]
TARGET_ID_PATTERN = re.compile(r"(?:CWE|CAPEC)-\d+")
VECTORIZER_TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")

# Edit these values directly to configure a run.
TARGET_SIM_THRESHOLD = 0.40
DOCUMENT_BATCH_SIZE = 256
ENCODE_BATCH_SIZE = 256
DEVICE = "auto"
OUTPUT_DIR = None
EVALUATION_SPLIT = "both"
RUN_HYPERPARAMETER_TUNING = True
OVERWRITE_EXISTING_OUTPUTS = False


CANDIDATE_COLUMNS = [
    "split",
    "cve_id",
    "cleaned_description",
    "candidate",
    "keybert_document_similarity",
    "best_cwe_id",
    "best_cwe_name",
    "best_cwe_target_type",
    "best_cwe_similarity",
    "best_capec_id",
    "best_capec_name",
    "best_capec_target_type",
    "best_capec_similarity",
    "best_target_type",
    "best_target_id",
    "target_similarity",
    "final_score",
    "rank_before_merge",
    "passed_target_threshold",
    "occurrence_spans"
]

INSPECTION_COLUMNS = [
    "split",
    "cve_id",
    "cleaned_description",
    "keyphrases",
    "keyphrase_source",
    "phrase",
    "keybert_document_similarity",
    "best_cwe_id",
    "best_cwe_name",
    "best_cwe_target_type",
    "best_cwe_similarity",
    "best_capec_id",
    "best_capec_name",
    "best_capec_target_type",
    "best_capec_similarity",
    "best_target_type",
    "best_target_id",
    "target_similarity",
    "final_score",
    "rank",
    "source_phrases",
    "was_merged"
]


# ============================================================
# CODE-LEVEL RUN CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RunConfig:
    target_sim_threshold: float
    document_batch_size: int
    encode_batch_size: int
    device: str
    output_dir: Path
    evaluation_split: str
    tune: bool
    overwrite: bool


def make_run_config() -> RunConfig:
    output_dir = (
        Path(OUTPUT_DIR).expanduser()
        if OUTPUT_DIR is not None
        else DEFAULT_OUTPUT_DIR
    )
    if RUN_HYPERPARAMETER_TUNING and OUTPUT_DIR is None:
        output_dir = output_dir / "hyperparameter_tuning"

    config = RunConfig(
        target_sim_threshold=TARGET_SIM_THRESHOLD,
        document_batch_size=DOCUMENT_BATCH_SIZE,
        encode_batch_size=ENCODE_BATCH_SIZE,
        device=DEVICE,
        output_dir=output_dir.resolve(),
        evaluation_split=EVALUATION_SPLIT,
        tune=RUN_HYPERPARAMETER_TUNING,
        overwrite=OVERWRITE_EXISTING_OUTPUTS
    )

    if config.document_batch_size < 1:
        raise ValueError("DOCUMENT_BATCH_SIZE must be positive")
    if config.encode_batch_size < 1:
        raise ValueError("ENCODE_BATCH_SIZE must be positive")
    if not 0.0 <= config.target_sim_threshold <= 1.0:
        raise ValueError("TARGET_SIM_THRESHOLD must be between 0 and 1")
    if config.evaluation_split not in {"validation", "test", "both"}:
        raise ValueError(
            "EVALUATION_SPLIT must be validation, test, or both"
        )

    return config


# ============================================================
# TARGET-THRESHOLD TUNING
# ============================================================

def threshold_run_name(threshold: float) -> str:
    return f"target_similarity_{threshold:.2f}"


def prune_tuning_run(run_dir: Path) -> None:
    """Keep only the small metric file from a tuning configuration."""
    if not run_dir.is_dir():
        return

    for path in run_dir.iterdir():
        if path.name == "evaluation_metrics.csv":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def publish_best_tuning_run(run_dir: Path) -> Path:
    """Copy the winning full-data outputs to the method output root."""
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename in (
        "training_final.csv",
        "validation_final.csv",
        "testing_final.csv",
        "evaluation_metrics.csv",
    ):
        source = run_dir / filename
        if not source.is_file():
            raise FileNotFoundError(
                f"Winning tuning run is missing output: {source}"
            )
        shutil.copy2(source, DEFAULT_OUTPUT_DIR / filename)

    config_path = run_dir / "config.json"
    if config_path.is_file():
        shutil.copy2(config_path, DEFAULT_OUTPUT_DIR / "config.json")
    return DEFAULT_OUTPUT_DIR


def run_threshold_tuning(config: RunConfig) -> None:
    run_threshold_tuning_joint(config)


# ============================================================
# GENERAL HELPERS
# ============================================================

def package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "unknown"


def set_random_seed() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_cwe_id(value: Any) -> str:
    raw = clean_text(value)
    if not raw:
        return ""
    if raw.upper().startswith("CWE-"):
        return raw.upper()
    try:
        return f"CWE-{int(float(raw))}"
    except (TypeError, ValueError):
        return raw


def normalize_capec_id(value: Any) -> str:
    raw = clean_text(value)
    if not raw:
        return ""
    if raw.upper().startswith("CAPEC-"):
        return raw.upper()
    try:
        return f"CAPEC-{int(float(raw))}"
    except (TypeError, ValueError):
        return raw


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def load_embedding_model(device: str) -> SentenceTransformer:
    """Use a cached MiniLM model when available, then allow normal loading."""
    try:
        return SentenceTransformer(
            MODEL_NAME,
            device=device,
            local_files_only=True
        )
    except (OSError, ValueError):
        print("MiniLM is not complete locally; trying normal model loading.")
        return SentenceTransformer(MODEL_NAME, device=device)


def encode_phrases_once(
    model: SentenceTransformer,
    phrases: Sequence[str],
    batch_size: int,
    embedding_cache=None
) -> np.ndarray:
    """Encode every unique merged phrase at most once in a tuning batch."""
    if not phrases:
        return np.empty((0, 0), dtype=np.float32)
    if embedding_cache is None:
        return model.encode(
            list(phrases),
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32, copy=False)

    missing_phrases = list(dict.fromkeys(
        phrase for phrase in phrases if phrase not in embedding_cache
    ))
    if missing_phrases:
        missing_embeddings = model.encode(
            missing_phrases,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32, copy=False)
        embedding_cache.update(zip(missing_phrases, missing_embeddings))

    return np.asarray(
        [embedding_cache[phrase] for phrase in phrases],
        dtype=np.float32
    )


# ============================================================
# DATA SPLIT
# ============================================================

def load_fixed_splits() -> Dict[str, pd.DataFrame]:
    """Create the shared seed-42 CVE-level 70/15/15 split."""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    frame = pd.read_csv(DATASET_PATH).reset_index(drop=True)
    frame.columns = frame.columns.str.strip()
    required = {"cve_id", "cleaned_description"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"dataset.csv is missing columns: {missing}")
    if frame["cve_id"].isna().any():
        raise ValueError("dataset.csv contains missing cve_id values")

    cve_ids = frame["cve_id"].astype(str)
    unique_ids = cve_ids.unique()
    training_ids, held_out_ids = train_test_split(
        unique_ids,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    training = frame[cve_ids.isin(training_ids)].copy().reset_index(drop=True)
    held_out = frame[cve_ids.isin(held_out_ids)].copy().reset_index(drop=True)

    validation_ids, testing_ids = train_test_split(
        held_out["cve_id"].astype(str).unique(),
        test_size=0.50,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    held_ids = held_out["cve_id"].astype(str)
    validation = held_out[
        held_ids.isin(validation_ids)
    ].copy().reset_index(drop=True)
    testing = held_out[
        held_ids.isin(testing_ids)
    ].copy().reset_index(drop=True)

    split_frames = {
        "training": training,
        "validation": validation,
        "testing": testing
    }

    split_id_sets = [
        set(part["cve_id"].astype(str))
        for part in split_frames.values()
    ]
    if any(
        split_id_sets[left] & split_id_sets[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise RuntimeError("CVE leakage detected between generated splits")
    if sum(len(part) for part in split_frames.values()) != len(frame):
        raise RuntimeError("Generated splits do not cover the dataset exactly")

    return split_frames


# ============================================================
# CWE/CAPEC TARGET KNOWLEDGE
# ============================================================

def load_target_knowledge() -> Tuple[
    List[Dict[str, str]],
    List[Dict[str, str]]
]:
    if not CWE_PATH.is_file():
        raise FileNotFoundError(f"CWE file not found: {CWE_PATH}")
    if not CAPEC_PATH.is_file():
        raise FileNotFoundError(f"CAPEC file not found: {CAPEC_PATH}")

    cwe_frame = pd.read_csv(CWE_PATH)
    capec_frame = pd.read_csv(CAPEC_PATH)
    cwe_frame.columns = cwe_frame.columns.str.strip()
    capec_frame.columns = capec_frame.columns.str.strip()

    required_cwe = {"cwe_id", "cwe_name", "description"}
    required_capec = {"ID", "Name", "Description"}
    missing_cwe = sorted(required_cwe - set(cwe_frame.columns))
    missing_capec = sorted(required_capec - set(capec_frame.columns))
    if missing_cwe:
        raise ValueError(f"CWE file is missing columns: {missing_cwe}")
    if missing_capec:
        raise ValueError(f"CAPEC file is missing columns: {missing_capec}")

    cwe_targets = []
    for _, row in cwe_frame.iterrows():
        target_id = normalize_cwe_id(row["cwe_id"])
        if not target_id:
            continue
        name = clean_text(row["cwe_name"])
        description = clean_text(row["description"])
        if name:
            cwe_targets.append({
                "id": target_id,
                "name": name,
                "target_type": "cwe_name",
                "text": name
            })
        if description:
            cwe_targets.append({
                "id": target_id,
                "name": name,
                "target_type": "cwe_description",
                "text": description
            })

    capec_targets = []
    for _, row in capec_frame.iterrows():
        target_id = normalize_capec_id(row["ID"])
        if not target_id:
            continue
        name = clean_text(row["Name"])
        description = clean_text(row["Description"])
        if name:
            capec_targets.append({
                "id": target_id,
                "name": name,
                "target_type": "capec_name",
                "text": name
            })
        if description:
            capec_targets.append({
                "id": target_id,
                "name": name,
                "target_type": "capec_description",
                "text": description
            })

    if not cwe_targets or not capec_targets:
        raise ValueError("CWE and CAPEC target collections must be non-empty")

    return cwe_targets, capec_targets


def target_signature(
    cwe_targets: Sequence[Dict[str, str]],
    capec_targets: Sequence[Dict[str, str]]
) -> str:
    payload = {
        "model": MODEL_NAME,
        "cwe": [
            (
                target["id"],
                target["target_type"],
                target["text"]
            )
            for target in cwe_targets
        ],
        "capec": [
            (
                target["id"],
                target["target_type"],
                target["text"]
            )
            for target in capec_targets
        ]
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def encode_target_knowledge(
    model: SentenceTransformer,
    cwe_targets: Sequence[Dict[str, str]],
    capec_targets: Sequence[Dict[str, str]],
    encode_batch_size: int,
    output_dir: Path
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Encode all targets once and reuse a content-validated local cache."""
    cache_path = output_dir / "target_embeddings.npz"
    signature = target_signature(cwe_targets, capec_targets)

    if cache_path.is_file():
        try:
            with np.load(cache_path, allow_pickle=False) as cache:
                cached_signature = str(cache["signature"].item())
                cwe_embeddings = np.asarray(
                    cache["cwe_embeddings"],
                    dtype=np.float32
                )
                capec_embeddings = np.asarray(
                    cache["capec_embeddings"],
                    dtype=np.float32
                )
            if (
                cached_signature == signature
                and cwe_embeddings.shape[0] == len(cwe_targets)
                and capec_embeddings.shape[0] == len(capec_targets)
            ):
                print("Loaded target embedding cache:", cache_path)
                return cwe_embeddings, capec_embeddings, True
            print("Ignoring stale target embedding cache.")
        except (KeyError, OSError, ValueError) as error:
            print("Ignoring unreadable target embedding cache:", error)

    print(f"Encoding {len(cwe_targets):,} CWE targets...")
    cwe_embeddings = model.encode(
        [target["text"] for target in cwe_targets],
        batch_size=encode_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32, copy=False)

    print(f"Encoding {len(capec_targets):,} CAPEC targets...")
    capec_embeddings = model.encode(
        [target["text"] for target in capec_targets],
        batch_size=encode_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32, copy=False)

    partial_path = output_dir / "target_embeddings.partial.npz"
    np.savez_compressed(
        partial_path,
        signature=np.asarray(signature),
        cwe_embeddings=cwe_embeddings,
        capec_embeddings=capec_embeddings
    )
    os.replace(partial_path, cache_path)
    print("Saved target embedding cache:", cache_path)
    return cwe_embeddings, capec_embeddings, False


# ============================================================
# KEYBERT CANDIDATES
# ============================================================

def extract_keybert_candidates(
    keyword_model: KeyBERT,
    documents: Sequence[str]
) -> Tuple[List[List[str]], np.ndarray, np.ndarray, Dict[str, int]]:
    """Extract the top document-relevant 2-4 grams per description."""
    vectorizer = CountVectorizer(
        ngram_range=KEYPHRASE_NGRAM_RANGE,
        stop_words=None,
        min_df=1
    )

    try:
        document_embeddings, candidate_embeddings = (
            keyword_model.extract_embeddings(
                list(documents),
                vectorizer=vectorizer
            )
        )
    except ValueError:
        empty = np.zeros((len(documents), 0), dtype=np.float32)
        return [[] for _ in documents], empty, empty, {}

    document_embeddings = l2_normalize(document_embeddings)
    candidate_embeddings = l2_normalize(candidate_embeddings)

    keyword_rows = keyword_model.extract_keywords(
        list(documents),
        vectorizer=vectorizer,
        top_n=CANDIDATE_TOP_N,
        use_maxsum=False,
        use_mmr=False,
        doc_embeddings=document_embeddings,
        word_embeddings=candidate_embeddings
    )

    if len(documents) == 1:
        keyword_rows = [keyword_rows]

    candidate_rows = []
    for keyword_row in keyword_rows:
        phrases = []
        seen = set()
        for phrase, _ in keyword_row:
            phrase = str(phrase).strip().lower()
            if phrase and phrase not in seen:
                seen.add(phrase)
                phrases.append(phrase)
        candidate_rows.append(phrases)

    vocabulary = [
        str(candidate)
        for candidate in vectorizer.get_feature_names_out()
    ]
    candidate_to_index = {
        candidate: index
        for index, candidate in enumerate(vocabulary)
    }

    return (
        candidate_rows,
        document_embeddings,
        candidate_embeddings,
        candidate_to_index
    )


# ============================================================
# ALL-TARGET COSINE RERANKING
# ============================================================

def find_candidate_spans(document_tokens: Sequence[str], phrase: str):
    phrase_tokens = phrase.split()
    span_length = len(phrase_tokens)
    if span_length == 0:
        return []

    return [
        (start, start + span_length)
        for start in range(len(document_tokens) - span_length + 1)
        if list(document_tokens[start:start + span_length]) == phrase_tokens
    ]


def group_overlapping_or_adjacent_spans(span_items):
    """Build connected span groups using exclusive end indices."""
    if not span_items:
        return []

    ordered = sorted(
        span_items,
        key=lambda item: (
            item["start"],
            item["end"],
            -item["target_similarity"]
        )
    )
    groups = []

    for item in ordered:
        if not groups or item["start"] > groups[-1]["end"]:
            groups.append({
                "start": item["start"],
                "end": item["end"],
                "items": [item]
            })
            continue

        groups[-1]["end"] = max(groups[-1]["end"], item["end"])
        groups[-1]["items"].append(item)

    return groups


def make_target_score_records(
    phrase_embeddings: np.ndarray,
    cwe_embeddings: np.ndarray,
    capec_embeddings: np.ndarray,
    cwe_targets: Sequence[Dict[str, str]],
    capec_targets: Sequence[Dict[str, str]]
):
    """Return the best CWE, CAPEC, and overall target for each phrase."""
    if len(phrase_embeddings) == 0:
        return []

    cwe_scores = phrase_embeddings @ cwe_embeddings.T
    capec_scores = phrase_embeddings @ capec_embeddings.T
    best_cwe_indices = np.argmax(cwe_scores, axis=1)
    best_capec_indices = np.argmax(capec_scores, axis=1)
    records = []

    for phrase_index in range(len(phrase_embeddings)):
        best_cwe_index = int(best_cwe_indices[phrase_index])
        best_capec_index = int(best_capec_indices[phrase_index])
        best_cwe_similarity = float(
            cwe_scores[phrase_index, best_cwe_index]
        )
        best_capec_similarity = float(
            capec_scores[phrase_index, best_capec_index]
        )

        if best_cwe_similarity >= best_capec_similarity:
            best_target_type = cwe_targets[best_cwe_index]["target_type"]
            best_target_id = cwe_targets[best_cwe_index]["id"]
            target_similarity = best_cwe_similarity
        else:
            best_target_type = capec_targets[
                best_capec_index
            ]["target_type"]
            best_target_id = capec_targets[best_capec_index]["id"]
            target_similarity = best_capec_similarity

        records.append({
            "best_cwe_id": cwe_targets[best_cwe_index]["id"],
            "best_cwe_name": cwe_targets[best_cwe_index]["name"],
            "best_cwe_target_type": cwe_targets[
                best_cwe_index
            ]["target_type"],
            "best_cwe_similarity": best_cwe_similarity,
            "best_capec_id": capec_targets[best_capec_index]["id"],
            "best_capec_name": capec_targets[best_capec_index]["name"],
            "best_capec_target_type": capec_targets[
                best_capec_index
            ]["target_type"],
            "best_capec_similarity": best_capec_similarity,
            "best_target_type": best_target_type,
            "best_target_id": best_target_id,
            "target_similarity": target_similarity,
            "final_score": target_similarity
        })

    return records


def score_candidate_batch(
    candidate_rows: Sequence[Sequence[str]],
    documents: Sequence[str],
    embedding_model: SentenceTransformer,
    document_embeddings: np.ndarray,
    vocabulary_embeddings: np.ndarray,
    candidate_to_index: Dict[str, int],
    cwe_embeddings: np.ndarray,
    capec_embeddings: np.ndarray,
    cwe_targets: Sequence[Dict[str, str]],
    capec_targets: Sequence[Dict[str, str]],
    target_sim_threshold: float,
    encode_batch_size: int,
    merged_embedding_cache=None,
    candidate_target_records=None,
    merged_target_record_cache=None
):
    """Rank/select five phrases, then merge and recheck only that set."""
    flat_embeddings = []
    flat_locations = []

    for document_index, phrases in enumerate(candidate_rows):
        for phrase in phrases:
            candidate_index = candidate_to_index[phrase]
            flat_embeddings.append(vocabulary_embeddings[candidate_index])
            flat_locations.append((document_index, phrase))

    scored_by_document = [[] for _ in candidate_rows]
    if not flat_embeddings:
        empty_rows = [[] for _ in candidate_rows]
        return (
            scored_by_document,
            empty_rows,
            ["" for _ in candidate_rows],
            empty_rows
        )

    candidate_embeddings = np.asarray(flat_embeddings, dtype=np.float32)
    if candidate_target_records is None:
        target_records = make_target_score_records(
            candidate_embeddings,
            cwe_embeddings,
            capec_embeddings,
            cwe_targets,
            capec_targets
        )
    else:
        target_records = candidate_target_records
        if len(target_records) != len(candidate_embeddings):
            raise ValueError(
                "Precomputed candidate target records do not align"
            )

    for flat_index, (document_index, phrase) in enumerate(flat_locations):
        keybert_similarity = float(
            candidate_embeddings[flat_index]
            @ document_embeddings[document_index]
        )
        target_record = target_records[flat_index]
        scored_by_document[document_index].append({
            "candidate": phrase,
            "keybert_document_similarity": keybert_similarity,
            **target_record,
            "passed_target_threshold": (
                target_record["target_similarity"] >= target_sim_threshold
            )
        })

    document_tokens = [
        VECTORIZER_TOKEN_PATTERN.findall(document.lower())
        for document in documents
    ]
    groups_by_document = []
    merge_proposals = []
    merge_locations = []

    for document_index, details in enumerate(scored_by_document):
        details.sort(
            key=lambda item: (
                -item["final_score"],
                -item["keybert_document_similarity"],
                item["candidate"]
            )
        )
        for rank, item in enumerate(details, start=1):
            item["rank_before_merge"] = rank
            spans = find_candidate_spans(
                document_tokens[document_index],
                item["candidate"]
            )
            item["occurrence_spans"] = ";".join(
                f"{start}:{end}" for start, end in spans
            )

        # Selection deliberately happens before merging. Only the five
        # highest-ranked, unique candidates that pass the target threshold
        # are allowed to participate in overlap/adjacency merging.
        selected_items = []
        selected_phrases = set()
        for item in details:
            phrase = item["candidate"]
            if not item["passed_target_threshold"]:
                continue
            if phrase in selected_phrases:
                continue
            selected_phrases.add(phrase)
            selected_items.append(item)
            if len(selected_items) >= FINAL_TOP_N:
                break

        span_items = []
        for item in selected_items:
            for start, end in find_candidate_spans(
                document_tokens[document_index],
                item["candidate"]
            ):
                span_items.append({
                    "phrase": item["candidate"],
                    "start": start,
                    "end": end,
                    "source_phrases": [item["candidate"]],
                    "was_merged": False,
                    **{
                        key: value
                        for key, value in item.items()
                        if key not in {
                            "candidate",
                            "passed_target_threshold",
                            "occurrence_spans",
                            "rank_before_merge"
                        }
                    }
                })

        groups = group_overlapping_or_adjacent_spans(span_items)
        groups_by_document.append(groups)
        for group_index, group in enumerate(groups):
            if len(group["items"]) <= 1:
                continue
            merged_phrase = " ".join(
                document_tokens[document_index][
                    group["start"]:group["end"]
                ]
            )
            merge_proposals.append(merged_phrase)
            merge_locations.append((document_index, group_index))

    merge_results = {}
    if merge_proposals:
        merged_embeddings = encode_phrases_once(
            model=embedding_model,
            phrases=merge_proposals,
            batch_size=encode_batch_size,
            embedding_cache=merged_embedding_cache
        )
        if merged_target_record_cache is None:
            merged_target_records = make_target_score_records(
                merged_embeddings,
                cwe_embeddings,
                capec_embeddings,
                cwe_targets,
                capec_targets
            )
        else:
            missing_indices = [
                index
                for index, phrase in enumerate(merge_proposals)
                if phrase not in merged_target_record_cache
            ]
            if missing_indices:
                missing_embeddings = merged_embeddings[missing_indices]
                missing_records = make_target_score_records(
                    missing_embeddings,
                    cwe_embeddings,
                    capec_embeddings,
                    cwe_targets,
                    capec_targets
                )
                for index, record in zip(
                    missing_indices,
                    missing_records
                ):
                    merged_target_record_cache[
                        merge_proposals[index]
                    ] = record
            merged_target_records = [
                merged_target_record_cache[phrase]
                for phrase in merge_proposals
            ]

        for proposal_index, location in enumerate(merge_locations):
            document_index, group_index = location
            group = groups_by_document[document_index][group_index]
            target_record = merged_target_records[proposal_index]
            document_similarity = float(
                merged_embeddings[proposal_index]
                @ document_embeddings[document_index]
            )
            merge_results[location] = {
                "phrase": merge_proposals[proposal_index],
                "start": group["start"],
                "end": group["end"],
                "source_phrases": list(dict.fromkeys(
                    item["phrase"] for item in group["items"]
                )),
                "was_merged": True,
                "keybert_document_similarity": document_similarity,
                **target_record
            }

    final_items_by_document = []
    final_keyphrases = []
    final_target_ids = []

    for document_index, groups in enumerate(groups_by_document):
        accepted_items = []
        for group_index, group in enumerate(groups):
            if len(group["items"]) == 1:
                accepted_items.extend(group["items"])
                continue

            merged_item = merge_results[(document_index, group_index)]
            if merged_item["target_similarity"] >= target_sim_threshold:
                accepted_items.append(merged_item)
            else:
                # A failed merged phrase leaves its original phrases separate.
                accepted_items.extend(group["items"])

        accepted_items.sort(
            key=lambda item: (
                -item["target_similarity"],
                -item["keybert_document_similarity"],
                item["phrase"]
            )
        )
        unique_items = []
        seen_phrases = set()
        for item in accepted_items:
            if item["phrase"] in seen_phrases:
                continue
            seen_phrases.add(item["phrase"])
            unique_items.append(item)
            if len(unique_items) >= FINAL_TOP_N:
                break

        for rank, item in enumerate(unique_items, start=1):
            item["rank"] = rank

        final_items_by_document.append(unique_items)
        final_keyphrases.append(
            "; ".join(item["phrase"] for item in unique_items)
        )
        final_target_ids.append(sorted({
            item["best_target_id"] for item in unique_items
        }))

    return (
        scored_by_document,
        final_items_by_document,
        final_keyphrases,
        final_target_ids
    )


def _published_target_threshold() -> float:
    """Read the threshold selected for the published KeyBERT outputs."""
    config_path = DEFAULT_OUTPUT_DIR / "config.json"
    if not config_path.is_file():
        return TARGET_SIM_THRESHOLD
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        return float(
            payload.get("target_sim_threshold", TARGET_SIM_THRESHOLD)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return TARGET_SIM_THRESHOLD


@lru_cache(maxsize=2)
def _online_resources(device: str):
    """Load and cache the published extractor resources for UI inference."""
    resolved_device = device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = load_embedding_model(resolved_device)
    backend = SentenceTransformerBackend(
        embedding_model,
        batch_size=ENCODE_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    keyword_model = KeyBERT(model=backend)
    cwe_targets, capec_targets = load_target_knowledge()
    cache_dir = DEFAULT_OUTPUT_DIR / "hyperparameter_tuning"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cwe_embeddings, capec_embeddings, _ = encode_target_knowledge(
        model=embedding_model,
        cwe_targets=cwe_targets,
        capec_targets=capec_targets,
        encode_batch_size=ENCODE_BATCH_SIZE,
        output_dir=cache_dir
    )
    return (
        keyword_model,
        embedding_model,
        cwe_targets,
        capec_targets,
        cwe_embeddings,
        capec_embeddings
    )


def keybert_modified_keywords(
    document: str,
    top_k: int = FINAL_TOP_N,
    device: str = "auto"
) -> str:
    """Extract published modified-KeyBERT phrases for one UI description."""
    text = clean_text(document)
    if not text:
        return ""
    (
        keyword_model,
        embedding_model,
        cwe_targets,
        capec_targets,
        cwe_embeddings,
        capec_embeddings
    ) = _online_resources(device)
    (
        candidate_rows,
        document_embeddings,
        vocabulary_embeddings,
        candidate_to_index
    ) = extract_keybert_candidates(keyword_model, [text])
    if document_embeddings.shape[1] == 0:
        return ""
    _, _, keyphrases, _ = score_candidate_batch(
        candidate_rows=candidate_rows,
        documents=[text],
        embedding_model=embedding_model,
        document_embeddings=document_embeddings,
        vocabulary_embeddings=vocabulary_embeddings,
        candidate_to_index=candidate_to_index,
        cwe_embeddings=cwe_embeddings,
        capec_embeddings=capec_embeddings,
        cwe_targets=cwe_targets,
        capec_targets=capec_targets,
        target_sim_threshold=_published_target_threshold(),
        encode_batch_size=ENCODE_BATCH_SIZE
    )
    phrases = [
        part.strip()
        for part in keyphrases[0].split(";")
        if part.strip()
    ]
    limit = max(1, min(int(top_k), FINAL_TOP_N))
    return "; ".join(phrases[:limit])


# ============================================================
# SPLIT PROCESSING
# ============================================================

def process_split(
    split_name: str,
    frame: pd.DataFrame,
    keyword_model: KeyBERT,
    embedding_model: SentenceTransformer,
    cwe_embeddings: np.ndarray,
    capec_embeddings: np.ndarray,
    cwe_targets: Sequence[Dict[str, str]],
    capec_targets: Sequence[Dict[str, str]],
    candidate_writer: csv.DictWriter,
    inspection_writer: csv.DictWriter,
    document_batch_size: int,
    encode_batch_size: int,
    target_sim_threshold: float,
    output_dir: Path
):
    """Extract phrases without consulting row-level truth-label columns."""
    cve_ids = frame["cve_id"].astype(str).tolist()
    documents = (
        frame["cleaned_description"].fillna("").astype(str).tolist()
    )
    all_keyphrases = ["" for _ in documents]
    predicted_target_ids = [[] for _ in documents]
    candidate_count = 0
    selected_count = 0

    starts = range(0, len(documents), document_batch_size)
    progress = tqdm(
        starts,
        total=(len(documents) + document_batch_size - 1)
        // document_batch_size,
        desc=f"Modified KeyBERT {split_name}"
    )

    for start in progress:
        end = min(start + document_batch_size, len(documents))
        batch_documents = documents[start:end]

        (
            candidate_rows,
            document_embeddings,
            vocabulary_embeddings,
            candidate_to_index
        ) = extract_keybert_candidates(keyword_model, batch_documents)

        if document_embeddings.shape[1] == 0:
            continue

        (
            scored_rows,
            final_rows,
            batch_keyphrases,
            batch_target_ids
        ) = score_candidate_batch(
            candidate_rows=candidate_rows,
            documents=batch_documents,
            embedding_model=embedding_model,
            document_embeddings=document_embeddings,
            vocabulary_embeddings=vocabulary_embeddings,
            candidate_to_index=candidate_to_index,
            cwe_embeddings=cwe_embeddings,
            capec_embeddings=capec_embeddings,
            cwe_targets=cwe_targets,
            capec_targets=capec_targets,
            target_sim_threshold=target_sim_threshold,
            encode_batch_size=encode_batch_size
        )
        all_keyphrases[start:end] = batch_keyphrases
        predicted_target_ids[start:end] = batch_target_ids

        for local_index, details in enumerate(scored_rows):
            absolute_index = start + local_index
            common = {
                "split": split_name,
                "cve_id": cve_ids[absolute_index],
                "cleaned_description": documents[absolute_index]
            }

            for item in details:
                candidate_writer.writerow({**common, **item})
                candidate_count += 1

            for item in final_rows[local_index]:
                inspection_writer.writerow({
                    **common,
                    "keyphrases": batch_keyphrases[local_index],
                    "keyphrase_source": KEYPHRASE_SOURCE,
                    "phrase": item["phrase"],
                    "keybert_document_similarity": item[
                        "keybert_document_similarity"
                    ],
                    "best_cwe_id": item["best_cwe_id"],
                    "best_cwe_name": item["best_cwe_name"],
                    "best_cwe_target_type": item[
                        "best_cwe_target_type"
                    ],
                    "best_cwe_similarity": item["best_cwe_similarity"],
                    "best_capec_id": item["best_capec_id"],
                    "best_capec_name": item["best_capec_name"],
                    "best_capec_target_type": item[
                        "best_capec_target_type"
                    ],
                    "best_capec_similarity": item[
                        "best_capec_similarity"
                    ],
                    "best_target_type": item["best_target_type"],
                    "best_target_id": item["best_target_id"],
                    "target_similarity": item["target_similarity"],
                    "final_score": item["final_score"],
                    "rank": item["rank"],
                    "source_phrases": "; ".join(item["source_phrases"]),
                    "was_merged": item["was_merged"]
                })
                selected_count += 1

    output_frame = frame.copy()
    output_frame["keyphrases"] = all_keyphrases
    output_frame["keyphrase_source"] = np.where(
        output_frame["keyphrases"].eq(""),
        "empty",
        KEYPHRASE_SOURCE
    )

    final_path = output_dir / f"{split_name}_final.csv"
    partial_path = output_dir / f"{split_name}_final.partial.csv"
    output_frame.to_csv(partial_path, index=False)
    os.replace(partial_path, final_path)
    print(f"Saved {final_path} ({len(output_frame):,} rows)")

    statistics = {
        "rows": len(output_frame),
        "candidate_rows": candidate_count,
        "selected_keyphrase_rows": selected_count,
        "rows_with_keyphrases": sum(bool(value) for value in all_keyphrases),
        "empty_rows": sum(not bool(value) for value in all_keyphrases)
    }
    return statistics, predicted_target_ids


# ============================================================
# TARGET-MATCH AND MULTILABEL-F1 EVALUATION
# ============================================================

def extract_target_ids(value: Any) -> List[str]:
    if pd.isna(value):
        return []
    return sorted(set(TARGET_ID_PATTERN.findall(str(value))))


def target_hit_scores(true_labels, predicted_labels):
    """Score 1 when any selected keyword target matches row truth."""
    if len(true_labels) != len(predicted_labels):
        raise ValueError("True and predicted target rows must align")

    return [
        int(bool(set(true_row) & set(predicted_row)))
        for true_row, predicted_row in zip(true_labels, predicted_labels)
    ]


def calculate_f1(true_labels, predicted_labels, prefix=None):
    """Use the same multilabel macro/micro F1 evaluation as Occlusion."""
    if prefix is not None:
        true_labels = [
            [label for label in labels if label.startswith(prefix)]
            for labels in true_labels
        ]
        predicted_labels = [
            [label for label in labels if label.startswith(prefix)]
            for labels in predicted_labels
        ]

    classes = sorted(set(
        label
        for labels in true_labels + predicted_labels
        for label in labels
    ))
    if not classes:
        return 0.0, 0.0

    encoder = MultiLabelBinarizer(classes=classes)
    encoder.fit(true_labels + predicted_labels)
    true_binary = encoder.transform(true_labels)
    predicted_binary = encoder.transform(predicted_labels)
    return (
        float(f1_score(
            true_binary,
            predicted_binary,
            average="macro",
            zero_division=0
        )),
        float(f1_score(
            true_binary,
            predicted_binary,
            average="micro",
            zero_division=0
        ))
    )


def evaluate_target_matches(
    frame: pd.DataFrame,
    predicted_labels,
    split_name: str,
    target_sim_threshold: float
):
    true_labels = [
        sorted(set(
            extract_target_ids(row.get("weakness", ""))
            + extract_target_ids(row.get("capec_id", ""))
        ))
        for _, row in frame.iterrows()
    ]
    hit_scores = target_hit_scores(true_labels, predicted_labels)
    hit_count = int(sum(hit_scores))
    hit_rate = hit_count / len(hit_scores) if hit_scores else 0.0
    macro_f1, micro_f1 = calculate_f1(true_labels, predicted_labels)
    cwe_macro_f1, cwe_micro_f1 = calculate_f1(
        true_labels,
        predicted_labels,
        prefix="CWE-"
    )
    capec_macro_f1, capec_micro_f1 = calculate_f1(
        true_labels,
        predicted_labels,
        prefix="CAPEC-"
    )
    return {
        "split": split_name,
        "rows": len(frame),
        "target_hit_count": hit_count,
        "target_miss_count": len(hit_scores) - hit_count,
        "target_hit_rate": hit_rate,
        "target_hit_percent": hit_rate * 100.0,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "cwe_macro_f1": cwe_macro_f1,
        "cwe_micro_f1": cwe_micro_f1,
        "capec_macro_f1": capec_macro_f1,
        "capec_micro_f1": capec_micro_f1,
        "empty_keyword_rows": sum(not labels for labels in predicted_labels),
        "target_sim_threshold": target_sim_threshold,
        "candidate_top_n": CANDIDATE_TOP_N,
        "final_top_n": FINAL_TOP_N,
        "random_seed": RANDOM_SEED
    }


def validation_summary_row(
    threshold: float,
    metric: Dict[str, Any],
    validation_root: Path
) -> Dict[str, Any]:
    run_name = threshold_run_name(threshold)
    return {
        "run_name": run_name,
        "target_sim_threshold": threshold,
        "rows": metric["rows"],
        "target_hit_count": metric["target_hit_count"],
        "target_miss_count": metric["target_miss_count"],
        "target_hit_rate": metric["target_hit_rate"],
        "target_hit_percent": metric["target_hit_percent"],
        "macro_f1": metric["macro_f1"],
        "micro_f1": metric["micro_f1"],
        "cwe_macro_f1": metric["cwe_macro_f1"],
        "cwe_micro_f1": metric["cwe_micro_f1"],
        "capec_macro_f1": metric["capec_macro_f1"],
        "capec_micro_f1": metric["capec_micro_f1"],
        "empty_keyword_rows": metric["empty_keyword_rows"],
        "output_directory": str(validation_root / run_name)
    }


def run_threshold_tuning_joint(config: RunConfig) -> None:
    """Evaluate all four thresholds from one shared embedding pass."""
    tuning_root = config.output_dir
    validation_root = tuning_root / "validation"
    testing_root = tuning_root / "testing"
    validation_root.mkdir(parents=True, exist_ok=True)
    testing_root.mkdir(parents=True, exist_ok=True)
    set_random_seed()

    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    split_frames = load_fixed_splits()
    validation = split_frames["validation"]
    if validation.empty:
        raise ValueError("Validation split is empty; cannot tune KeyBERT")

    cwe_targets, capec_targets = load_target_knowledge()
    embedding_model = load_embedding_model(device)
    keybert_backend = SentenceTransformerBackend(
        embedding_model,
        batch_size=config.encode_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    keyword_model = KeyBERT(model=keybert_backend)
    cwe_embeddings, capec_embeddings, _ = encode_target_knowledge(
        model=embedding_model,
        cwe_targets=cwe_targets,
        capec_targets=capec_targets,
        encode_batch_size=config.encode_batch_size,
        output_dir=tuning_root
    )

    predictions_by_threshold = {
        threshold: [] for threshold in TARGET_SIM_THRESHOLD_GRID
    }
    documents = (
        validation["cleaned_description"].fillna("").astype(str).tolist()
    )
    starts = range(0, len(documents), config.document_batch_size)
    progress = tqdm(
        starts,
        total=(len(documents) + config.document_batch_size - 1)
        // config.document_batch_size,
        desc="Joint modified-KeyBERT tuning"
    )

    for start in progress:
        end = min(start + config.document_batch_size, len(documents))
        batch_documents = documents[start:end]
        (
            candidate_rows,
            document_embeddings,
            vocabulary_embeddings,
            candidate_to_index
        ) = extract_keybert_candidates(keyword_model, batch_documents)

        flat_candidate_embeddings = np.asarray([
            vocabulary_embeddings[candidate_to_index[phrase]]
            for phrases in candidate_rows
            for phrase in phrases
        ], dtype=np.float32)
        candidate_target_records = make_target_score_records(
            flat_candidate_embeddings,
            cwe_embeddings,
            capec_embeddings,
            cwe_targets,
            capec_targets
        )
        merged_embedding_cache = {}
        merged_target_record_cache = {}
        for threshold in TARGET_SIM_THRESHOLD_GRID:
            _, _, _, batch_predictions = score_candidate_batch(
                candidate_rows=candidate_rows,
                documents=batch_documents,
                embedding_model=embedding_model,
                document_embeddings=document_embeddings,
                vocabulary_embeddings=vocabulary_embeddings,
                candidate_to_index=candidate_to_index,
                cwe_embeddings=cwe_embeddings,
                capec_embeddings=capec_embeddings,
                cwe_targets=cwe_targets,
                capec_targets=capec_targets,
                target_sim_threshold=threshold,
                encode_batch_size=config.encode_batch_size,
                merged_embedding_cache=merged_embedding_cache,
                candidate_target_records=candidate_target_records,
                merged_target_record_cache=merged_target_record_cache
            )
            predictions_by_threshold[threshold].extend(batch_predictions)

        partial_frame = validation.iloc[:end]
        partial_rows = []
        for threshold in TARGET_SIM_THRESHOLD_GRID:
            partial_metric = evaluate_target_matches(
                partial_frame,
                predictions_by_threshold[threshold],
                "validation",
                threshold
            )
            partial_rows.append(validation_summary_row(
                threshold,
                partial_metric,
                validation_root
            ))
        pd.DataFrame(partial_rows).to_csv(
            tuning_root / "validation_summary_partial.csv",
            index=False
        )

    validation_rows = []
    for threshold in TARGET_SIM_THRESHOLD_GRID:
        metric = evaluate_target_matches(
            validation,
            predictions_by_threshold[threshold],
            "validation",
            threshold
        )
        row = validation_summary_row(
            threshold,
            metric,
            validation_root
        )
        validation_rows.append(row)
        run_dir = Path(row["output_directory"])
        run_dir.mkdir(parents=True, exist_ok=True)
        metric["tuning_strategy"] = "joint_thresholds_shared_embeddings"
        pd.DataFrame([metric]).to_csv(
            run_dir / "evaluation_metrics.csv",
            index=False
        )
        prune_tuning_run(run_dir)

    validation_summary = pd.DataFrame(validation_rows).sort_values(
        "macro_f1",
        ascending=False,
        kind="stable"
    ).reset_index(drop=True)
    validation_summary.to_csv(
        tuning_root / "validation_summary.csv",
        index=False
    )

    best = validation_summary.iloc[0]
    best_threshold = float(best["target_sim_threshold"])
    best_run_name = str(best["run_name"])
    best_test_dir = testing_root / best_run_name
    if best_test_dir.exists():
        shutil.rmtree(best_test_dir)
    best_test_dir.mkdir(parents=True)

    source_cache = tuning_root / "target_embeddings.npz"
    if source_cache.is_file():
        shutil.copy2(source_cache, best_test_dir / source_cache.name)

    print("Best validation configuration:", best_run_name)
    print("Validation target hit rate:", best["target_hit_rate"])
    print("Validation target hit percent:", best["target_hit_percent"])
    print("Validation macro F1:", best["macro_f1"])
    print("Validation micro F1:", best["micro_f1"])
    print("Running the winning configuration once on all splits.")

    del keyword_model
    del keybert_backend
    del embedding_model
    torch.cuda.empty_cache()
    run_pipeline(replace(
        config,
        target_sim_threshold=best_threshold,
        output_dir=best_test_dir,
        evaluation_split="test",
        tune=False,
        overwrite=False
    ))

    best_test_metrics_path = best_test_dir / "evaluation_metrics.csv"
    test_metrics = pd.read_csv(best_test_metrics_path)
    test_metrics.to_csv(
        tuning_root / "best_test_metrics.csv",
        index=False
    )
    test_metric = test_metrics.loc[
        test_metrics["split"] == "test"
    ].iloc[0]
    method_output_dir = publish_best_tuning_run(best_test_dir)

    best_parameters = pd.DataFrame([{
        "selection_metric": "validation_macro_f1",
        "tuning_strategy": "joint_thresholds_shared_embeddings",
        "target_sim_threshold": best_threshold,
        "validation_target_hit_rate": float(best["target_hit_rate"]),
        "validation_target_hit_percent": float(
            best["target_hit_percent"]
        ),
        "validation_macro_f1": float(best["macro_f1"]),
        "validation_micro_f1": float(best["micro_f1"]),
        "validation_cwe_macro_f1": float(best["cwe_macro_f1"]),
        "validation_cwe_micro_f1": float(best["cwe_micro_f1"]),
        "validation_capec_macro_f1": float(best["capec_macro_f1"]),
        "validation_capec_micro_f1": float(best["capec_micro_f1"]),
        "test_target_hit_rate": float(test_metric["target_hit_rate"]),
        "test_target_hit_percent": float(
            test_metric["target_hit_percent"]
        ),
        "test_macro_f1": float(test_metric["macro_f1"]),
        "test_micro_f1": float(test_metric["micro_f1"]),
        "test_cwe_macro_f1": float(test_metric["cwe_macro_f1"]),
        "test_cwe_micro_f1": float(test_metric["cwe_micro_f1"]),
        "test_capec_macro_f1": float(test_metric["capec_macro_f1"]),
        "test_capec_micro_f1": float(test_metric["capec_micro_f1"])
    }])
    best_path = method_output_dir / "best_hyperparameters.csv"
    best_parameters.to_csv(best_path, index=False)
    print("Saved best hyperparameters:", best_path)
    prune_tuning_run(best_test_dir)


# ============================================================
# OUTPUT MANAGEMENT
# ============================================================

def ensure_outputs_are_safe(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    managed_outputs = [
        output_dir / "training_final.csv",
        output_dir / "validation_final.csv",
        output_dir / "testing_final.csv",
        output_dir / "keyword_extraction_inspection.csv",
        output_dir / "candidate_target_matches.csv",
        output_dir / "evaluation_metrics.csv",
        output_dir / "config.json"
    ]
    existing = [path for path in managed_outputs if path.exists()]
    if existing and not overwrite:
        joined = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "Modified KeyBERT outputs already exist. "
            "Set OVERWRITE_EXISTING_OUTPUTS = True to replace them:\n"
            f"  {joined}"
        )


# ============================================================
# MAIN
# ============================================================

def run_pipeline(run_config: RunConfig) -> None:
    output_dir = run_config.output_dir
    ensure_outputs_are_safe(output_dir, run_config.overwrite)
    set_random_seed()

    device = run_config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    started_at = datetime.now(timezone.utc)
    started_clock = time.monotonic()

    print("Python:", sys.executable)
    print("Device:", device)
    print("Dataset:", DATASET_PATH)
    print("Output directory:", output_dir)
    print("KeyBERT candidate top N:", CANDIDATE_TOP_N)
    print("Final top N:", FINAL_TOP_N)
    print("N-gram range:", KEYPHRASE_NGRAM_RANGE)
    print("Target similarity threshold:", run_config.target_sim_threshold)
    print(
        "Final score: maximum cosine similarity across CWE name, "
        "CWE description, CAPEC name, and CAPEC description"
    )

    split_frames = load_fixed_splits()
    print(
        "Fixed seed-42 split sizes:",
        ", ".join(
            f"{name}={len(frame):,}"
            for name, frame in split_frames.items()
        )
    )

    cwe_targets, capec_targets = load_target_knowledge()
    print(f"CWE target parts: {len(cwe_targets):,}")
    print(f"CAPEC target parts: {len(capec_targets):,}")

    embedding_model = load_embedding_model(device)
    keybert_backend = SentenceTransformerBackend(
        embedding_model,
        batch_size=run_config.encode_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    keyword_model = KeyBERT(model=keybert_backend)

    (
        cwe_embeddings,
        capec_embeddings,
        target_cache_reused
    ) = encode_target_knowledge(
        model=embedding_model,
        cwe_targets=cwe_targets,
        capec_targets=capec_targets,
        encode_batch_size=run_config.encode_batch_size,
        output_dir=output_dir
    )

    candidate_path = output_dir / "candidate_target_matches.csv"
    inspection_path = output_dir / "keyword_extraction_inspection.csv"
    candidate_partial = output_dir / "candidate_target_matches.partial.csv"
    inspection_partial = (
        output_dir / "keyword_extraction_inspection.partial.csv"
    )
    split_statistics = {}
    split_predictions = {}

    with candidate_partial.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as candidate_file, inspection_partial.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as inspection_file:
        candidate_writer = csv.DictWriter(
            candidate_file,
            fieldnames=CANDIDATE_COLUMNS
        )
        inspection_writer = csv.DictWriter(
            inspection_file,
            fieldnames=INSPECTION_COLUMNS
        )
        candidate_writer.writeheader()
        inspection_writer.writeheader()

        for split_name in ("training", "validation", "testing"):
            statistics, predictions = process_split(
                split_name=split_name,
                frame=split_frames[split_name],
                keyword_model=keyword_model,
                embedding_model=embedding_model,
                cwe_embeddings=cwe_embeddings,
                capec_embeddings=capec_embeddings,
                cwe_targets=cwe_targets,
                capec_targets=capec_targets,
                candidate_writer=candidate_writer,
                inspection_writer=inspection_writer,
                document_batch_size=run_config.document_batch_size,
                encode_batch_size=run_config.encode_batch_size,
                target_sim_threshold=run_config.target_sim_threshold,
                output_dir=output_dir
            )
            split_statistics[split_name] = statistics
            split_predictions[split_name] = predictions

    os.replace(candidate_partial, candidate_path)
    os.replace(inspection_partial, inspection_path)

    metrics_rows = []
    if run_config.evaluation_split in {"validation", "both"}:
        metrics_rows.append(evaluate_target_matches(
            split_frames["validation"],
            split_predictions["validation"],
            "validation",
            run_config.target_sim_threshold
        ))
    if run_config.evaluation_split in {"test", "both"}:
        metrics_rows.append(evaluate_target_matches(
            split_frames["testing"],
            split_predictions["testing"],
            "test",
            run_config.target_sim_threshold
        ))
    metrics_path = output_dir / "evaluation_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

    finished_at = datetime.now(timezone.utc)
    result_config = {
        "method": (
            "KeyBERT with four-part target cosine reranking and "
            "top-five pre-merge selection"
        ),
        "completed": True,
        "model_name": MODEL_NAME,
        "device": device,
        "random_seed": RANDOM_SEED,
        "dataset_path": str(DATASET_PATH),
        "cwe_path": str(CWE_PATH),
        "capec_path": str(CAPEC_PATH),
        "output_directory": str(output_dir),
        "input_text_column": "cleaned_description",
        "identifier_column": "cve_id",
        "ground_truth_used_during_extraction": False,
        "ground_truth_columns_carried_only": ["weakness", "capec_id"],
        "keyphrase_ngram_range": list(KEYPHRASE_NGRAM_RANGE),
        "stop_words": None,
        "candidate_top_n": CANDIDATE_TOP_N,
        "final_top_n": FINAL_TOP_N,
        "target_sim_threshold": run_config.target_sim_threshold,
        "target_sim_threshold_grid": TARGET_SIM_THRESHOLD_GRID,
        "evaluation_split": run_config.evaluation_split,
        "selection_metric": "validation_macro_f1",
        "tuning_strategy": "joint_thresholds_shared_embeddings",
        "target_representation": [
            "cwe_name",
            "cwe_description",
            "capec_name",
            "capec_description"
        ],
        "final_score_formula": (
            "maximum cosine similarity over the four independent "
            "target-part types"
        ),
        "selection_before_merging": "top five unique passing candidates",
        "merging": (
            "overlap_or_adjacency_within_selected_top_five_and_"
            "all-target_recheck"
        ),
        "edge_trimming": False,
        "document_batch_size": run_config.document_batch_size,
        "encode_batch_size": run_config.encode_batch_size,
        "cwe_target_part_count": len(cwe_targets),
        "capec_target_part_count": len(capec_targets),
        "target_embedding_cache": str(
            output_dir / "target_embeddings.npz"
        ),
        "target_embedding_cache_reused": target_cache_reused,
        "split_strategy": "CVE-ID-level 70/15/15 with seed 42",
        "split_statistics": split_statistics,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "elapsed_seconds": time.monotonic() - started_clock,
        "versions": {
            "keybert": package_version("keybert"),
            "sentence_transformers": package_version(
                "sentence-transformers"
            ),
            "scikit_learn": package_version("scikit-learn"),
            "torch": torch.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__
        }
    }

    config_partial = output_dir / "config.partial.json"
    config_partial.write_text(
        json.dumps(result_config, indent=2),
        encoding="utf-8"
    )
    os.replace(config_partial, output_dir / "config.json")

    print("Saved:", candidate_path)
    print("Saved:", inspection_path)
    print("Saved:", metrics_path)
    print("Saved:", output_dir / "config.json")
    print(
        f"Completed in "
        f"{result_config['elapsed_seconds'] / 60.0:.1f} minutes"
    )


def main() -> None:
    run_config = make_run_config()
    if run_config.tune:
        run_threshold_tuning(run_config)
    else:
        run_pipeline(run_config)


if __name__ == "__main__":
    main()
