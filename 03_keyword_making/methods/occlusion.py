#!/usr/bin/env python3

"""Generate and evaluate occlusion keywords using one GPU."""

import itertools
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


def collapse_consecutive_duplicate_tokens(text):
    """Reduce adjacent duplicate tokens before keyword extraction."""
    final_tokens = []
    for token in str(text).split():
        if final_tokens and token.casefold() == final_tokens[-1].casefold():
            continue
        final_tokens.append(token)
    return " ".join(final_tokens)


# ============================================================
# PROJECT PATHS
# ============================================================

# /home/gurleen/paper/methodology/03_keyword_making/methods
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# /home/gurleen/paper/methodology/03_keyword_making
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# /home/gurleen/paper
PAPER_DIR = os.path.dirname(os.path.dirname(PROJECT_DIR))

# A batch job can redirect every method into one self-contained run folder.
# When unset, standalone runs keep using PROJECT_DIR/outputs.
OUTPUTS_ROOT = os.path.abspath(os.environ.get(
    "KEYWORD_MAKER_OUTPUT_ROOT",
    os.path.join(PROJECT_DIR, "outputs")
))


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# HYPERPARAMETER GRID
# ============================================================

# Full grid:



BASE_SIM_THRESHOLD_GRID = [0.30,0.35, 0.40, 0.45, 0.50]
DROP_THRESHOLD_PERCENT_GRID = [2.5, 5.0, 7.5, 10.0]
KEYWORD_SIM_KEEP_RATIO_GRID = [0.80, 0.85, 0.90, 0.95]


# ============================================================
# CODE-LEVEL RUN CONFIGURATION
# ============================================================

BASE_SIM_THRESHOLD = 0.40
DROP_THRESHOLD_PERCENT = 5.0
KEYWORD_SIM_KEEP_RATIO = 0.90
OUTPUT_DIR = None
EVALUATION_SPLIT = "both"
RUN_HYPERPARAMETER_TUNING = True
KEEP_INTERMEDIATE_FILES = False

# After joint tuning, one isolated child runs the winning configuration over
# all splits. This private value passes that final configuration to the child.
_INTERNAL_CONFIG_ENV = "_OCCLUSION_INTERNAL_RUN_CONFIG"
_internal_config = json.loads(
    os.environ.get(_INTERNAL_CONFIG_ENV, "{}")
)

BASE_SIM_THRESHOLD = float(_internal_config.get(
    "base_sim_threshold",
    BASE_SIM_THRESHOLD
))
DROP_THRESHOLD_PERCENT = float(_internal_config.get(
    "drop_threshold_percent",
    DROP_THRESHOLD_PERCENT
))
KEYWORD_SIM_KEEP_RATIO = float(_internal_config.get(
    "keyword_sim_keep_ratio",
    KEYWORD_SIM_KEEP_RATIO
))
EVALUATION_SPLIT = _internal_config.get(
    "evaluation_split",
    EVALUATION_SPLIT
)
RUN_HYPERPARAMETER_TUNING = bool(_internal_config.get(
    "tune",
    RUN_HYPERPARAMETER_TUNING
))
OUTPUT_DIR = _internal_config.get("output_dir", OUTPUT_DIR)

if not 0.0 <= BASE_SIM_THRESHOLD <= 1.0:
    raise ValueError("BASE_SIM_THRESHOLD must be between 0 and 1")
if DROP_THRESHOLD_PERCENT < 0.0:
    raise ValueError("DROP_THRESHOLD_PERCENT must be non-negative")
if not 0.0 <= KEYWORD_SIM_KEEP_RATIO <= 1.0:
    raise ValueError("KEYWORD_SIM_KEEP_RATIO must be between 0 and 1")
if EVALUATION_SPLIT not in {"validation", "test", "both"}:
    raise ValueError(
        "EVALUATION_SPLIT must be validation, test, or both"
    )

if OUTPUT_DIR is None:
    OUTPUT_DIR = os.path.join(
        OUTPUTS_ROOT,
        "occlusion"
    )
    if RUN_HYPERPARAMETER_TUNING:
        OUTPUT_DIR = os.path.join(
            OUTPUT_DIR,
            "hyperparameter_tuning"
        )




def hyperparameter_run_name(base_threshold, drop_threshold, keep_ratio):
    return (
        f"base_{base_threshold:.2f}__"
        f"drop_{drop_threshold:.1f}__"
        f"keep_{keep_ratio:.2f}"
    )


FINAL_OUTPUT_FILENAMES = (
    "training_final.csv",
    "validation_final.csv",
    "testing_final.csv",
)


def prune_tuning_run(run_dir):
    """Keep only the small metric file from a tuning configuration."""
    if not os.path.isdir(run_dir):
        return

    for name in os.listdir(run_dir):
        if name == "evaluation_metrics.csv":
            continue
        path = os.path.join(run_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def publish_best_tuning_run(run_dir):
    """Copy the winning full-data split files to the method output root."""
    method_output_dir = os.path.join(OUTPUTS_ROOT, "occlusion")
    os.makedirs(method_output_dir, exist_ok=True)

    for filename in FINAL_OUTPUT_FILENAMES:
        source = os.path.join(run_dir, filename)
        if not os.path.isfile(source):
            raise FileNotFoundError(
                f"Winning tuning run is missing final output: {source}"
            )
        shutil.copy2(source, os.path.join(method_output_dir, filename))

    metrics_source = os.path.join(run_dir, "evaluation_metrics.csv")
    shutil.copy2(
        metrics_source,
        os.path.join(method_output_dir, "evaluation_metrics.csv")
    )
    return method_output_dir


def run_hyperparameter_tuning():
    """Evaluate the complete grid in one pass over validation CVEs."""
    return run_hyperparameter_tuning_single_pass()
print("Python:", sys.executable)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print(
    "Slurm job:",
    os.environ.get("SLURM_JOB_ID", "not running under Slurm")
)
print("Current folder:", os.getcwd())


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = os.path.join(
    PAPER_DIR,
    "methodology",
    "02_preprocessing",
    "dataset.csv"
)

CWE_PATH = os.path.join(
    PAPER_DIR,
    "methodology",
    "01_cwe_capec_ground_truth",
    "cwe_final.csv"
)

CAPEC_PATH = os.path.join(
    PAPER_DIR,
    "methodology",
    "01_cwe_capec_ground_truth",
    "3000_capec.csv"
)


# ============================================================
# VERIFY PATHS
# ============================================================

for name, path in [
    ("DATASET", DATASET_PATH),
    ("CWE", CWE_PATH),
    ("CAPEC", CAPEC_PATH),
]:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} file not found:\n{path}"
        )





# ============================================================
# OUTPUT PATHS
# ============================================================

OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)




WORK_DIR = os.path.join(
    OUTPUT_DIR,
    "worker"
)

TEMP_DIR = os.path.join(
    OUTPUT_DIR,
    "temp"
)

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
# ============================================================
# SINGLE GPU
# ============================================================



DEBUG_ROWS = 25



# ============================================================
# COLUMNS
# ============================================================

TEXT_COL = "cleaned_description"
CWE_COL = "weakness"
CAPEC_COL = "capec_id"

TRAIN_SIZE = 0.70


# ============================================================
# OCCLUSION SETTINGS
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 256

SAMPLE_N_ENV = os.environ.get("OCCLUSION_SAMPLE_N", "").strip()
SAMPLE_N = int(SAMPLE_N_ENV) if SAMPLE_N_ENV else None
if SAMPLE_N is not None and SAMPLE_N < 3:
    raise ValueError("OCCLUSION_SAMPLE_N must be at least 3")

TUNING_PROGRESS_EVERY_ROWS = 10

HIGH_FREQ_PERCENT_THRESHOLD = 30.0
LOW_FREQ_ROW_THRESHOLD = 2

NGRAM_MIN = 2
NGRAM_MAX = 4

MAX_TOKENS_PER_ROW = 240

TOP_SELECTED_SPANS_BEFORE_MERGE = 40

TOP_FINAL_KEYWORDS = 5

MAX_MERGED_KEYWORD_WORDS = 7

MIN_WORDS_AFTER_TRIM = 2

MAX_TRIM_ROUNDS = 2


FREQ_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "train_word_frequency.csv"
)

CONFIG_PATH = os.path.join(
    WORK_DIR,
    "keyword_config.json"
)

# ============================================================
# HELPERS
# ============================================================

def clean_text(x):
    if pd.isna(x):
        return ""
    return collapse_consecutive_duplicate_tokens(str(x).strip())


def word_tokenize(text):
    text = clean_text(text).lower()
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_+\-]*", text)


def build_train_word_frequency(train_texts):
    counter = Counter()

    for text in tqdm(train_texts, desc="Building train word frequency"):
        words = set(word_tokenize(text))
        counter.update(words)

    return counter


def make_frequency_dataframe(counter, train_n):
    rows = []

    for word, count in counter.most_common():
        percentage = (count / train_n) * 100

        is_too_common = percentage > HIGH_FREQ_PERCENT_THRESHOLD
        is_too_rare = count <= LOW_FREQ_ROW_THRESHOLD

        rows.append({
            "word": word,
            "row_frequency": count,
            "percentage": percentage,
            "is_too_common": is_too_common,
            "is_too_rare": is_too_rare,
            "allowed_for_occlusion_keywords": not is_too_common and not is_too_rare
        })

    return pd.DataFrame(rows)





# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip()
SOURCE_DATASET_ROWS = len(df)

if SAMPLE_N is not None:
    df = df.head(SAMPLE_N).copy()

required_cols = [TEXT_COL, CWE_COL, CAPEC_COL] 

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}. Available columns: {df.columns.tolist()}")

df[TEXT_COL] = (
    df[TEXT_COL]
    .fillna("")
    .astype(str)
    .map(collapse_consecutive_duplicate_tokens)
)

df["_row_id"] = np.arange(len(df))

print("Total rows:", len(df))
print("Columns:", df.columns.tolist())


# ============================================================
# 70/30 SPLIT
# ============================================================

if "cve_id" in df.columns:
    unique_cves = df["cve_id"].dropna().astype(str).unique()
    train_cves, temp_cves = train_test_split(
        unique_cves,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    train_df = df[df["cve_id"].astype(str).isin(train_cves)].copy()
    test_df = df[df["cve_id"].astype(str).isin(temp_cves)].copy()
else:
    train_idx, test_idx = train_test_split(
        df.index,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()

train_df = train_df.reset_index(drop=True).copy()
test_df = test_df.reset_index(drop=True).copy()

train_df["split"] = "train"
test_df["split"] = "test"




train_df["_split_pos"] = np.arange(len(train_df))
test_df["_split_pos"] = np.arange(len(test_df))

print("Train rows:", len(train_df))
print("Test rows:", len(test_df))


# ============================================================
# BUILD TRAIN-ONLY FREQUENCY MODEL
# ============================================================

train_texts = train_df[TEXT_COL].tolist()

train_word_counter = build_train_word_frequency(train_texts)
freq_df = make_frequency_dataframe(train_word_counter, len(train_df))

freq_df.to_csv(FREQ_OUTPUT_PATH, index=False)

print("\nSaved:", FREQ_OUTPUT_PATH)
print("Allowed keyword vocabulary size:", freq_df["allowed_for_occlusion_keywords"].sum())
print("\nTop train word frequencies:")
print(freq_df.head(20))





# ============================================================
# SAVE SINGLE-GPU WORKER INPUT
# ============================================================

train_input_path = os.path.join(
    WORK_DIR,
    "train_input.csv"
)

test_input_path = os.path.join(
    WORK_DIR,
    "test_input.csv"
)

train_df.to_csv(
    train_input_path,
    index=False
)

test_df.to_csv(
    test_input_path,
    index=False
)

print(
    f"Worker input: train={len(train_df)}, "
    f"test={len(test_df)}"
)

print("Saved:", train_input_path)
print("Saved:", test_input_path)


# ============================================================
# SAVE CONFIG FOR WORKERS
# ============================================================

config = {
    "CWE_PATH": CWE_PATH,
    "CAPEC_PATH": CAPEC_PATH,
    "WORK_DIR": WORK_DIR,
    "TEMP_DIR": TEMP_DIR,
    "RANDOM_SEED": RANDOM_SEED,
    "DEBUG_ROWS": DEBUG_ROWS,
    "TEXT_COL": TEXT_COL,
    "CWE_COL": CWE_COL,
    "CAPEC_COL": CAPEC_COL,
    "MODEL_NAME": MODEL_NAME,
    "BATCH_SIZE": BATCH_SIZE,
    "HIGH_FREQ_PERCENT_THRESHOLD": HIGH_FREQ_PERCENT_THRESHOLD,
    "LOW_FREQ_ROW_THRESHOLD": LOW_FREQ_ROW_THRESHOLD,
    "BASE_SIM_THRESHOLD": BASE_SIM_THRESHOLD,
    "DROP_THRESHOLD_PERCENT": DROP_THRESHOLD_PERCENT,
    "NGRAM_MIN": NGRAM_MIN,
    "NGRAM_MAX": NGRAM_MAX,
    "MAX_TOKENS_PER_ROW": MAX_TOKENS_PER_ROW,
    "TOP_SELECTED_SPANS_BEFORE_MERGE": TOP_SELECTED_SPANS_BEFORE_MERGE,
    "TOP_FINAL_KEYWORDS": TOP_FINAL_KEYWORDS,
    "MAX_MERGED_KEYWORD_WORDS": MAX_MERGED_KEYWORD_WORDS,
    "KEYWORD_SIM_KEEP_RATIO": KEYWORD_SIM_KEEP_RATIO,
    "MIN_WORDS_AFTER_TRIM": MIN_WORDS_AFTER_TRIM,
    "MAX_TRIM_ROUNDS": MAX_TRIM_ROUNDS,
    "FREQ_OUTPUT_PATH": FREQ_OUTPUT_PATH
}

with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)

print("\nSaved config:", CONFIG_PATH)


# ============================================================
# STAGE-03 OCCLUSION LOGIC (SINGLE GPU)
# ============================================================

def format_capec_id(value):
    if pd.isna(value):
        return None

    value = str(value).strip()
    return f"CAPEC-{int(value)}" if value else None


def safe_relative_drop(base_similarity, new_similarity):
    if base_similarity <= 1e-8:
        return 0.0
    return (
        (base_similarity - new_similarity)
        / abs(base_similarity)
    ) * 100.0


def is_allowed_keyword_word(word, frequency, train_n):
    count = frequency.get(word, 0)
    percentage = (count / train_n) * 100 if train_n else 0.0

    if percentage > HIGH_FREQ_PERCENT_THRESHOLD:
        return False
    if count <= LOW_FREQ_ROW_THRESHOLD:
        return False
    return True


def make_ngram_candidates(tokens, frequency, train_n, stopwords):
    tokens = tokens[:MAX_TOKENS_PER_ROW]
    candidates = []
    seen = set()

    for ngram_size in range(NGRAM_MIN, NGRAM_MAX + 1):
        for start in range(0, len(tokens) - ngram_size + 1):
            end = start + ngram_size
            ngram = tokens[start:end]

            if any(
                not is_allowed_keyword_word(word, frequency, train_n)
                for word in ngram
            ):
                continue
            if all(word in stopwords for word in ngram):
                continue
            if all(len(word) <= 1 for word in ngram):
                continue

            phrase = " ".join(ngram)
            key = (phrase, start, end)
            if key in seen:
                continue

            seen.add(key)
            candidates.append({
                "phrase": phrase,
                "start": start,
                "end": end
            })

    return candidates


def remove_span(tokens, start, end):
    return " ".join(tokens[:start] + tokens[end:])


def merge_selected_spans(selected_items, tokens):
    if not selected_items:
        return []

    selected_items = sorted(
        selected_items,
        key=lambda item: (item["start"], item["end"])
    )
    groups = []

    for item in selected_items:
        if not groups:
            groups.append({
                "start": item["start"],
                "end": item["end"],
                "items": [item],
                "max_relative_drop_percent": item[
                    "max_relative_drop_percent"
                ],
                "max_absolute_drop": item["max_absolute_drop"]
            })
            continue

        last = groups[-1]

        # End indices are exclusive, so this merges overlap and adjacency.
        if item["start"] <= last["end"]:
            last["end"] = max(last["end"], item["end"])
            last["items"].append(item)
            last["max_relative_drop_percent"] = max(
                last["max_relative_drop_percent"],
                item["max_relative_drop_percent"]
            )
            last["max_absolute_drop"] = max(
                last["max_absolute_drop"],
                item["max_absolute_drop"]
            )
        else:
            groups.append({
                "start": item["start"],
                "end": item["end"],
                "items": [item],
                "max_relative_drop_percent": item[
                    "max_relative_drop_percent"
                ],
                "max_absolute_drop": item["max_absolute_drop"]
            })

    merged = []
    for group in groups:
        merged_tokens = tokens[group["start"]:group["end"]]
        merged_phrase = " ".join(merged_tokens)

        if len(merged_tokens) > MAX_MERGED_KEYWORD_WORDS:
            best_item = max(
                group["items"],
                key=lambda item: (
                    item["max_relative_drop_percent"],
                    item["max_absolute_drop"],
                    len(item["phrase"].split())
                )
            )
            merged_phrase = best_item["phrase"]
            final_start = best_item["start"]
            final_end = best_item["end"]
        else:
            final_start = group["start"]
            final_end = group["end"]

        merged.append({
            "phrase": merged_phrase,
            "start": final_start,
            "end": final_end,
            "max_relative_drop_percent": float(
                group["max_relative_drop_percent"]
            ),
            "max_absolute_drop": float(group["max_absolute_drop"]),
            "source_phrases": [
                item["phrase"] for item in group["items"]
            ],
            "matched_targets": {
                f"{index}:{item['phrase']}@{item['start']}:{item['end']}": (
                    item["matched_targets"]
                )
                for index, item in enumerate(group["items"])
            }
        })

    merged = sorted(
        merged,
        key=lambda item: (
            item["max_relative_drop_percent"],
            item["max_absolute_drop"],
            len(item["phrase"].split())
        ),
        reverse=True
    )

    final = []
    seen_phrases = set()
    for item in merged:
        if item["phrase"] in seen_phrases:
            continue
        seen_phrases.add(item["phrase"])
        final.append(item)
        if len(final) >= TOP_FINAL_KEYWORDS:
            break

    return final


# Load the complete CWE/CAPEC target space used by Stage 03.
cwe_df = pd.read_csv(CWE_PATH)
capec_df = pd.read_csv(CAPEC_PATH)
cwe_df.columns = cwe_df.columns.str.strip()
capec_df.columns = capec_df.columns.str.strip()

required_cwe_cols = {"cwe_id", "cwe_name", "description"}
required_capec_cols = {"ID", "Name", "Description"}
missing_cwe_cols = required_cwe_cols - set(cwe_df.columns)
missing_capec_cols = required_capec_cols - set(capec_df.columns)

if missing_cwe_cols:
    raise ValueError(f"Missing CWE columns: {sorted(missing_cwe_cols)}")
if missing_capec_cols:
    raise ValueError(f"Missing CAPEC columns: {sorted(missing_capec_cols)}")

all_target_texts = []
all_target_meta = []

for _, row in cwe_df.iterrows():
    target_id = clean_text(row["cwe_id"])
    if not target_id:
        continue

    target_name = clean_text(row["cwe_name"])
    target_description = clean_text(row["description"])

    if target_name:
        all_target_texts.append(target_name)
        all_target_meta.append({
            "id": target_id,
            "target_type": "cwe_name",
            "source": "cwe",
            "text": target_name
        })
    if target_description:
        all_target_texts.append(target_description)
        all_target_meta.append({
            "id": target_id,
            "target_type": "cwe_description",
            "source": "cwe",
            "text": target_description
        })

for _, row in capec_df.iterrows():
    target_id = format_capec_id(row["ID"])
    if target_id is None:
        continue

    target_name = clean_text(row["Name"])
    target_description = clean_text(row["Description"])

    if target_name:
        all_target_texts.append(target_name)
        all_target_meta.append({
            "id": target_id,
            "target_type": "capec_name",
            "source": "capec",
            "text": target_name
        })
    if target_description:
        all_target_texts.append(target_description)
        all_target_meta.append({
            "id": target_id,
            "target_type": "capec_description",
            "source": "capec",
            "text": target_description
        })

cwe_target_text_count = sum(
    meta["source"] == "cwe" for meta in all_target_meta
)
capec_target_text_count = sum(
    meta["source"] == "capec" for meta in all_target_meta
)

print("CWE target texts checked per CVE:", cwe_target_text_count)
print("CAPEC target texts checked per CVE:", capec_target_text_count)
print("Total target texts checked per CVE:", len(all_target_texts))
print("CWE records represented:", len(cwe_df))
print("CAPEC records represented:", len(capec_df))

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU is visible. This method requires one GPU.")

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(MODEL_NAME, device="cuda:0")
target_embs = model.encode(
    all_target_texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True
)
target_embs = np.asarray(target_embs, dtype=np.float32)

target_meta_key_to_index = {
    (meta["id"], meta["target_type"]): index
    for index, meta in enumerate(all_target_meta)
}


def get_target_indices_for_selected_item(item):
    target_indices = []
    for target_info in item.get("matched_targets", {}).values():
        key = (
            target_info.get("target_id"),
            target_info.get("target_type")
        )
        if key in target_meta_key_to_index:
            target_indices.append(target_meta_key_to_index[key])
    return sorted(set(target_indices))


def get_target_indices_for_merged_item(item):
    target_indices = []
    for target_dict in item.get("matched_targets", {}).values():
        for target_info in target_dict.values():
            key = (
                target_info.get("target_id"),
                target_info.get("target_type")
            )
            if key in target_meta_key_to_index:
                target_indices.append(target_meta_key_to_index[key])
    return sorted(set(target_indices))


def encode_phrase_embeddings(phrases, embedding_cache=None):
    """Encode each distinct phrase once, optionally reusing a row cache."""
    if not phrases:
        return np.empty((0, target_embs.shape[1]), dtype=np.float32)

    cache = embedding_cache if embedding_cache is not None else {}
    missing_phrases = list(dict.fromkeys(
        phrase for phrase in phrases if phrase not in cache
    ))
    if missing_phrases:
        missing_embeddings = model.encode(
            missing_phrases,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        missing_embeddings = np.asarray(
            missing_embeddings,
            dtype=np.float32
        )
        for phrase, embedding in zip(
            missing_phrases,
            missing_embeddings
        ):
            cache[phrase] = embedding

    return np.asarray([cache[phrase] for phrase in phrases], dtype=np.float32)


def batch_phrase_target_similarities(
    phrases,
    target_index_groups,
    embedding_cache=None
):
    if not phrases:
        return []

    phrase_embs = encode_phrase_embeddings(
        phrases,
        embedding_cache=embedding_cache
    )

    similarities = []
    for phrase_emb, target_indices in zip(
        phrase_embs,
        target_index_groups
    ):
        if not target_indices:
            similarities.append(0.0)
            continue
        target_indices_array = np.asarray(target_indices, dtype=int)
        similarities.append(float(np.max(
            np.dot(target_embs[target_indices_array], phrase_emb)
        )))
    return similarities


def trim_selected_spans_before_merge(
    selected_items,
    tokens,
    base_sim_threshold=None,
    keyword_sim_keep_ratio=None,
    embedding_cache=None
):
    """Trim selected interval edges before overlap/adjacency merging."""
    if base_sim_threshold is None:
        base_sim_threshold = BASE_SIM_THRESHOLD
    if keyword_sim_keep_ratio is None:
        keyword_sim_keep_ratio = KEYWORD_SIM_KEEP_RATIO

    states = []
    similarity_phrases = []
    similarity_target_groups = []
    similarity_state_indices = []

    for state_index, item in enumerate(selected_items):
        start = int(item["start"])
        end = int(item["end"])
        if not 0 <= start < end <= len(tokens):
            raise ValueError(f"Invalid selected span [{start}, {end})")

        target_indices = get_target_indices_for_selected_item(item)
        phrase = " ".join(tokens[start:end])
        state = {
            "item": item,
            "original_phrase": phrase,
            "start": start,
            "end": end,
            "target_indices": target_indices,
            "original_similarity": None,
            "final_similarity": None,
            "minimum_required_similarity": None,
            "trim_rounds": 0,
            "active": (
                (end - start) > MIN_WORDS_AFTER_TRIM
                and bool(target_indices)
            )
        }
        states.append(state)

        if state["active"]:
            similarity_phrases.append(phrase)
            similarity_target_groups.append(target_indices)
            similarity_state_indices.append(state_index)

    original_similarities = batch_phrase_target_similarities(
        similarity_phrases,
        similarity_target_groups,
        embedding_cache=embedding_cache
    )
    for state_index, similarity in zip(
        similarity_state_indices,
        original_similarities
    ):
        state = states[state_index]
        state["original_similarity"] = similarity
        state["final_similarity"] = similarity
        state["minimum_required_similarity"] = max(
            base_sim_threshold,
            keyword_sim_keep_ratio * similarity
        )

    for _ in range(MAX_TRIM_ROUNDS):
        option_phrases = []
        option_target_groups = []
        option_meta = []

        for state_index, state in enumerate(states):
            if not state["active"]:
                continue
            if state["end"] - state["start"] <= MIN_WORDS_AFTER_TRIM:
                state["active"] = False
                continue

            state["trim_rounds"] += 1
            for new_start, new_end in (
                (state["start"] + 1, state["end"]),
                (state["start"], state["end"] - 1)
            ):
                option_phrases.append(" ".join(tokens[new_start:new_end]))
                option_target_groups.append(state["target_indices"])
                option_meta.append((state_index, new_start, new_end))

        option_similarities = batch_phrase_target_similarities(
            option_phrases,
            option_target_groups,
            embedding_cache=embedding_cache
        )
        best_options = {}
        for (state_index, new_start, new_end), similarity in zip(
            option_meta,
            option_similarities
        ):
            state = states[state_index]
            if similarity < state["minimum_required_similarity"]:
                continue
            previous = best_options.get(state_index)
            if previous is None or similarity > previous[2]:
                best_options[state_index] = (
                    new_start,
                    new_end,
                    similarity
                )

        for state_index, state in enumerate(states):
            if not state["active"]:
                continue
            best = best_options.get(state_index)
            if best is None:
                state["active"] = False
                continue
            state["start"], state["end"], state["final_similarity"] = best
            if state["end"] - state["start"] <= MIN_WORDS_AFTER_TRIM:
                state["active"] = False

    results = []
    for state in states:
        trimmed_item = dict(state["item"])
        trimmed_item["start"] = state["start"]
        trimmed_item["end"] = state["end"]
        trimmed_item["phrase"] = " ".join(
            tokens[state["start"]:state["end"]]
        )

        if not state["target_indices"]:
            decision = "no_matched_target"
        elif len(state["original_phrase"].split()) <= MIN_WORDS_AFTER_TRIM:
            decision = "already_at_minimum_words"
        elif trimmed_item["phrase"] != state["original_phrase"]:
            decision = "trimmed_before_merge"
        else:
            decision = "kept_before_merge"

        results.append((trimmed_item, {
            "original_phrase": state["original_phrase"],
            "original_similarity": state["original_similarity"],
            "final_similarity": state["final_similarity"],
            "minimum_required_similarity": state[
                "minimum_required_similarity"
            ],
            "trim_rounds": state["trim_rounds"],
            "decision": decision
        }, state["target_indices"]))

    return results


def build_new_info_for_phrase(phrase, target_indices):
    phrase = phrase.strip().lower()
    if not phrase:
        return ""
    if not target_indices:
        return f"{phrase} -> no_match"

    phrase_emb = model.encode(
        [phrase],
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    phrase_emb = np.asarray(phrase_emb, dtype=np.float32)[0]
    target_indices_array = np.asarray(target_indices, dtype=int)
    similarities = np.dot(
        target_embs[target_indices_array],
        phrase_emb
    )
    best_global = int(
        target_indices_array[int(np.argmax(similarities))]
    )
    best_meta = all_target_meta[best_global]
    best_similarity = float(np.max(similarities))
    return (
        f"{phrase} -> {best_meta['id']} | "
        f"part={best_meta['target_type']} | "
        f"similarity={best_similarity:.4f}"
    )


def make_occlusion_keyword_details(
    final_merged_items,
    embedding_cache=None
):
    phrases = [
        item.get("phrase", "").strip().lower()
        for item in final_merged_items
    ]
    target_groups = [
        get_target_indices_for_merged_item(item)
        for item in final_merged_items
    ]
    if not phrases:
        return ""

    phrase_embs = encode_phrase_embeddings(
        phrases,
        embedding_cache=embedding_cache
    )

    info_rows = []
    for phrase, phrase_emb, target_indices in zip(
        phrases,
        phrase_embs,
        target_groups
    ):
        if not target_indices:
            info_rows.append(f"{phrase} -> no_match")
            continue

        target_indices_array = np.asarray(target_indices, dtype=int)
        similarities = np.dot(
            target_embs[target_indices_array],
            phrase_emb
        )
        best_global = int(
            target_indices_array[int(np.argmax(similarities))]
        )
        best_meta = all_target_meta[best_global]
        best_similarity = float(np.max(similarities))
        info_rows.append(
            f"{phrase} -> {best_meta['id']} | "
            f"part={best_meta['target_type']} | "
            f"similarity={best_similarity:.4f}"
        )

    return "; ".join(info_rows)


ROW_DEBUG_COLUMNS = [
    "row_id", "cve_id", "split", "token_count", "total_target_texts",
    "relevant_target_texts", "candidate_count",
    "passed_drop_threshold_count", "selected_for_merge_count",
    "final_keyword_count", "status", "pretrim_occlusion_keywords",
    "final_occlusion_keywords", "final_occlusion_keyword_details",
    "used_textrank_fallback"
]

CANDIDATE_DEBUG_COLUMNS = [
    "row_id", "cve_id", "split", "phrase", "start", "end",
    "relevant_target_texts", "matched_target_count",
    "matched_target_ids_sample", "max_relative_drop_percent",
    "max_absolute_drop", "passed_drop_threshold", "selected_for_merge",
    "base_sim_threshold", "drop_threshold_percent"
]

TRIMMING_DEBUG_COLUMNS = [
    "row_id", "cve_id", "split", "original_phrase", "final_phrase",
    "changed", "decision", "original_target_similarity",
    "final_target_similarity", "minimum_required_similarity",
    "trim_rounds", "matched_target_count", "final_target_detail",
    "base_sim_threshold", "keyword_sim_keep_ratio"
]


def save_debug_fragment(records, columns, debug_name, split_name):
    path = os.path.join(TEMP_DIR, f"_{debug_name}_{split_name}.csv")
    pd.DataFrame(records, columns=columns).to_csv(path, index=False)
    return path


def process_split(split_name):
    input_path = os.path.join(WORK_DIR, f"{split_name}_input.csv")
    output_path = os.path.join(WORK_DIR, f"output_{split_name}.csv")
    split_df = pd.read_csv(input_path)
    split_df.columns = split_df.columns.str.strip()

    if split_df.empty:
        split_df["occlusion_keywords"] = pd.Series(dtype=str)
        split_df["occlusion_keyword_details"] = pd.Series(dtype=str)
        split_df.to_csv(output_path, index=False)
        save_debug_fragment(
            [], ROW_DEBUG_COLUMNS, "occlusion_row_debug", split_name
        )
        save_debug_fragment(
            [], CANDIDATE_DEBUG_COLUMNS,
            "occlusion_candidate_debug", split_name
        )
        save_debug_fragment(
            [], TRIMMING_DEBUG_COLUMNS, "trimming_debug", split_name
        )
        return

    for column in [TEXT_COL, CWE_COL, CAPEC_COL]:
        if column not in split_df.columns:
            raise ValueError(f"Missing column in worker input: {column}")

    split_df[TEXT_COL] = (
        split_df[TEXT_COL]
        .fillna("")
        .astype(str)
        .map(collapse_consecutive_duplicate_tokens)
    )

    print(f"\nProcessing {split_name} on cuda:0")
    print("Rows:", len(split_df))

    description_embs = model.encode(
        split_df[TEXT_COL].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    description_embs = np.asarray(description_embs, dtype=np.float32)

    stopwords = set(ENGLISH_STOP_WORDS)
    train_n = len(train_df)
    frequency = dict(train_word_counter)
    occlusion_keywords_col = []
    occlusion_keyword_details_col = []
    row_debug_records = []
    candidate_debug_records = []
    trimming_debug_records = []
    changed_count = 0

    for row_position, (index, row) in enumerate(tqdm(
        split_df.iterrows(),
        total=len(split_df),
        desc=f"Occlusion {split_name}"
    )):
        tokens = word_tokenize(row[TEXT_COL])[:MAX_TOKENS_PER_ROW]
        row_id = row.get("_row_id", index)
        cve_id = row.get("cve_id", "")
        collect_candidate_debug = row_position < DEBUG_ROWS
        description_emb = description_embs[row_position]

        # First compare the CVE description with every CWE/CAPEC target.
        base_similarities = np.dot(target_embs, description_emb)
        usable_target_indices = np.where(
            base_similarities >= BASE_SIM_THRESHOLD
        )[0]

        if not tokens or not len(usable_target_indices):
            occlusion_keywords_col.append("")
            occlusion_keyword_details_col.append("")
            row_debug_records.append({
                "row_id": row_id,
                "cve_id": cve_id,
                "split": split_name,
                "token_count": len(tokens),
                "total_target_texts": len(all_target_texts),
                "relevant_target_texts": len(usable_target_indices),
                "candidate_count": 0,
                "passed_drop_threshold_count": 0,
                "selected_for_merge_count": 0,
                "final_keyword_count": 0,
                "status": (
                    "no_tokens" if not tokens else "no_relevant_targets"
                ),
                "pretrim_occlusion_keywords": ""
            })
            continue

        candidates = make_ngram_candidates(
            tokens,
            frequency,
            train_n,
            stopwords
        )
        if not candidates:
            occlusion_keywords_col.append("")
            occlusion_keyword_details_col.append("")
            row_debug_records.append({
                "row_id": row_id,
                "cve_id": cve_id,
                "split": split_name,
                "token_count": len(tokens),
                "total_target_texts": len(all_target_texts),
                "relevant_target_texts": len(usable_target_indices),
                "candidate_count": 0,
                "passed_drop_threshold_count": 0,
                "selected_for_merge_count": 0,
                "final_keyword_count": 0,
                "status": "no_allowed_candidates",
                "pretrim_occlusion_keywords": ""
            })
            continue

        occluded_texts = [
            remove_span(tokens, candidate["start"], candidate["end"])
            for candidate in candidates
        ]
        occluded_embs = model.encode(
            occluded_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        occluded_embs = np.asarray(occluded_embs, dtype=np.float32)

        usable_target_matrix = target_embs[usable_target_indices]
        usable_base_vector = base_similarities[usable_target_indices]
        similarity_matrix = np.dot(
            occluded_embs,
            usable_target_matrix.T
        )

        selected_items = []
        current_candidate_debug = []

        for candidate_index, candidate in enumerate(candidates):
            matched_targets = {}
            max_relative_drop = 0.0
            max_absolute_drop = 0.0

            for target_position, global_target_index in enumerate(
                usable_target_indices
            ):
                base_similarity = float(
                    usable_base_vector[target_position]
                )
                new_similarity = float(
                    similarity_matrix[candidate_index, target_position]
                )
                absolute_drop = base_similarity - new_similarity
                relative_drop = safe_relative_drop(
                    base_similarity,
                    new_similarity
                )

                if relative_drop >= DROP_THRESHOLD_PERCENT:
                    meta = all_target_meta[global_target_index]
                    target_key = f"{meta['id']}::{meta['target_type']}"
                    matched_targets[target_key] = {
                        "target_id": meta["id"],
                        "target_type": meta["target_type"],
                        "target_source": meta["source"],
                        "base_similarity": base_similarity,
                        "after_removal_similarity": new_similarity,
                        "absolute_drop": absolute_drop,
                        "relative_drop_percent": relative_drop
                    }
                    max_relative_drop = max(
                        max_relative_drop,
                        relative_drop
                    )
                    max_absolute_drop = max(
                        max_absolute_drop,
                        absolute_drop
                    )

            if matched_targets:
                selected_items.append({
                    "phrase": candidate["phrase"],
                    "start": candidate["start"],
                    "end": candidate["end"],
                    "max_relative_drop_percent": float(max_relative_drop),
                    "max_absolute_drop": float(max_absolute_drop),
                    "matched_targets": matched_targets
                })

            if collect_candidate_debug:
                current_candidate_debug.append({
                    "row_id": row_id,
                    "cve_id": cve_id,
                    "split": split_name,
                    "phrase": candidate["phrase"],
                    "start": candidate["start"],
                    "end": candidate["end"],
                    "relevant_target_texts": len(usable_target_indices),
                    "matched_target_count": len(matched_targets),
                    "matched_target_ids_sample": ";".join(sorted({
                        target["target_id"]
                        for target in matched_targets.values()
                    })[:20]),
                    "max_relative_drop_percent": max_relative_drop,
                    "max_absolute_drop": max_absolute_drop,
                    "passed_drop_threshold": bool(matched_targets),
                    "selected_for_merge": False,
                    "base_sim_threshold": BASE_SIM_THRESHOLD,
                    "drop_threshold_percent": DROP_THRESHOLD_PERCENT
                })

        selected_items = sorted(
            selected_items,
            key=lambda item: (
                item["max_relative_drop_percent"],
                item["max_absolute_drop"],
                len(item["phrase"].split())
            ),
            reverse=True
        )
        passed_drop_threshold_count = len(selected_items)
        selected_items = selected_items[
            :TOP_SELECTED_SPANS_BEFORE_MERGE
        ]

        selected_keys = {
            (item["phrase"], item["start"], item["end"])
            for item in selected_items
        }
        for debug_record in current_candidate_debug:
            debug_record["selected_for_merge"] = (
                debug_record["phrase"],
                debug_record["start"],
                debug_record["end"]
            ) in selected_keys
        candidate_debug_records.extend(current_candidate_debug)

        pretrim_merged_items = merge_selected_spans(
            selected_items,
            tokens
        )
        pretrim_keywords = [
            item["phrase"] for item in pretrim_merged_items
        ]

        trimming_results = trim_selected_spans_before_merge(
            selected_items,
            tokens
        )
        trimmed_selected_items = [
            result[0] for result in trimming_results
        ]

        if any(
            trimmed_item["phrase"] != details["original_phrase"]
            for trimmed_item, details, _ in trimming_results
        ):
            changed_count += 1

        if collect_candidate_debug:
            for trimmed_item, details, target_indices in trimming_results:
                trimming_debug_records.append({
                    "row_id": row_id,
                    "cve_id": cve_id,
                    "split": split_name,
                    "original_phrase": details["original_phrase"],
                    "final_phrase": trimmed_item["phrase"],
                    "changed": (
                        trimmed_item["phrase"]
                        != details["original_phrase"]
                    ),
                    "decision": details["decision"],
                    "original_target_similarity": details[
                        "original_similarity"
                    ],
                    "final_target_similarity": details[
                        "final_similarity"
                    ],
                    "minimum_required_similarity": details[
                        "minimum_required_similarity"
                    ],
                    "trim_rounds": details["trim_rounds"],
                    "matched_target_count": len(target_indices),
                    "final_target_detail": build_new_info_for_phrase(
                        trimmed_item["phrase"],
                        target_indices
                    ),
                    "base_sim_threshold": BASE_SIM_THRESHOLD,
                    "keyword_sim_keep_ratio": KEYWORD_SIM_KEEP_RATIO
                })

        final_merged_items = merge_selected_spans(
            trimmed_selected_items,
            tokens
        )
        final_keywords = [
            item["phrase"] for item in final_merged_items
        ]
        final_keywords_text = "; ".join(final_keywords)
        final_details = make_occlusion_keyword_details(
            final_merged_items
        )

        occlusion_keywords_col.append(final_keywords_text)
        occlusion_keyword_details_col.append(final_details)
        row_debug_records.append({
            "row_id": row_id,
            "cve_id": cve_id,
            "split": split_name,
            "token_count": len(tokens),
            "total_target_texts": len(all_target_texts),
            "relevant_target_texts": len(usable_target_indices),
            "candidate_count": len(candidates),
            "passed_drop_threshold_count": passed_drop_threshold_count,
            "selected_for_merge_count": len(selected_items),
            "final_keyword_count": len(final_keywords),
            "status": (
                "occlusion_keywords_found"
                if final_keywords
                else "no_keyword_selected"
            ),
            "pretrim_occlusion_keywords": "; ".join(pretrim_keywords)
        })

    output_df = split_df.copy()
    output_df["occlusion_keywords"] = occlusion_keywords_col
    output_df["occlusion_keyword_details"] = (
        occlusion_keyword_details_col
    )
    output_df["occlusion_keywords"] = (
        output_df["occlusion_keywords"].fillna("").astype(str)
    )
    output_df["occlusion_keyword_details"] = (
        output_df["occlusion_keyword_details"].fillna("").astype(str)
    )

    for debug_record, final_keywords, final_details in zip(
        row_debug_records,
        occlusion_keywords_col,
        occlusion_keyword_details_col
    ):
        debug_record["final_occlusion_keywords"] = final_keywords
        debug_record["final_occlusion_keyword_details"] = final_details
        debug_record["used_textrank_fallback"] = (
            final_keywords.strip() == ""
        )

    output_df.to_csv(output_path, index=False)
    row_debug_path = save_debug_fragment(
        row_debug_records,
        ROW_DEBUG_COLUMNS,
        "occlusion_row_debug",
        split_name
    )
    candidate_debug_path = save_debug_fragment(
        candidate_debug_records,
        CANDIDATE_DEBUG_COLUMNS,
        "occlusion_candidate_debug",
        split_name
    )
    trimming_debug_path = save_debug_fragment(
        trimming_debug_records,
        TRIMMING_DEBUG_COLUMNS,
        "trimming_debug",
        split_name
    )

    print("Saved worker output:", output_path)
    print("Rows changed by pre-merge edge trimming:", changed_count)
    print("Saved row debug fragment:", row_debug_path)
    print("Saved candidate debug fragment:", candidate_debug_path)
    print("Saved trimming debug fragment:", trimming_debug_path)


def tuning_true_target_ids(row):
    """Return the exact CWE/CAPEC IDs used by the tuning hit score."""
    pattern = re.compile(r"(?:CWE|CAPEC)-\d+")
    return set(
        pattern.findall(str(row.get(CWE_COL, "")))
        + pattern.findall(str(row.get(CAPEC_COL, "")))
    )


def predicted_target_ids_for_items(final_items, embedding_cache):
    """Get one best matched target ID for every final keyword phrase."""
    phrases = [
        item.get("phrase", "").strip().lower()
        for item in final_items
    ]
    target_groups = [
        get_target_indices_for_merged_item(item)
        for item in final_items
    ]
    if not phrases:
        return set()

    phrase_embeddings = encode_phrase_embeddings(
        phrases,
        embedding_cache=embedding_cache
    )
    predicted_ids = set()
    for phrase_embedding, target_indices in zip(
        phrase_embeddings,
        target_groups
    ):
        if not target_indices:
            continue
        target_indices_array = np.asarray(target_indices, dtype=int)
        similarities = np.dot(
            target_embs[target_indices_array],
            phrase_embedding
        )
        best_global_index = int(
            target_indices_array[int(np.argmax(similarities))]
        )
        predicted_ids.add(all_target_meta[best_global_index]["id"])
    return predicted_ids


def build_selected_items_from_shared_scores(
    candidates,
    target_indices,
    base_similarities,
    after_removal_similarities,
    relative_drops,
    base_threshold,
    drop_threshold
):
    """Select the top spans for one threshold pair without re-encoding."""
    eligible_positions = np.flatnonzero(
        base_similarities >= base_threshold
    )
    if not len(eligible_positions):
        return []

    eligible_relative_drops = relative_drops[:, eligible_positions]
    matched = eligible_relative_drops >= drop_threshold
    passing_candidate_indices = np.flatnonzero(matched.any(axis=1))
    ranked_candidates = []

    for candidate_index in passing_candidate_indices:
        local_matches = np.flatnonzero(matched[candidate_index])
        matched_positions = eligible_positions[local_matches]
        candidate_relative_drops = relative_drops[
            candidate_index,
            matched_positions
        ]
        candidate_absolute_drops = (
            base_similarities[matched_positions]
            - after_removal_similarities[
                candidate_index,
                matched_positions
            ]
        )
        ranked_candidates.append((
            int(candidate_index),
            float(np.max(candidate_relative_drops)),
            float(np.max(candidate_absolute_drops)),
            matched_positions
        ))

    ranked_candidates.sort(
        key=lambda result: (
            result[1],
            result[2],
            len(candidates[result[0]]["phrase"].split())
        ),
        reverse=True
    )

    selected_items = []
    for (
        candidate_index,
        max_relative_drop,
        max_absolute_drop,
        matched_positions
    ) in ranked_candidates[:TOP_SELECTED_SPANS_BEFORE_MERGE]:
        candidate = candidates[candidate_index]
        matched_targets = {}

        for target_position in matched_positions:
            global_target_index = int(target_indices[target_position])
            base_similarity = float(base_similarities[target_position])
            new_similarity = float(
                after_removal_similarities[
                    candidate_index,
                    target_position
                ]
            )
            absolute_drop = base_similarity - new_similarity
            relative_drop = float(
                relative_drops[candidate_index, target_position]
            )
            meta = all_target_meta[global_target_index]
            target_key = f"{meta['id']}::{meta['target_type']}"
            matched_targets[target_key] = {
                "target_id": meta["id"],
                "target_type": meta["target_type"],
                "target_source": meta["source"],
                "base_similarity": base_similarity,
                "after_removal_similarity": new_similarity,
                "absolute_drop": absolute_drop,
                "relative_drop_percent": relative_drop
            }

        selected_items.append({
            "phrase": candidate["phrase"],
            "start": candidate["start"],
            "end": candidate["end"],
            "max_relative_drop_percent": max_relative_drop,
            "max_absolute_drop": max_absolute_drop,
            "matched_targets": matched_targets
        })

    return selected_items


def preload_tuning_phrase_embeddings(
    selected_items_by_threshold,
    tokens,
    embedding_cache
):
    """Batch-encode every phrase that edge trimming may inspect."""
    phrases = []
    seen_spans = set()
    for selected_items in selected_items_by_threshold.values():
        for item in selected_items:
            original_start = int(item["start"])
            original_end = int(item["end"])
            for removed_words in range(MAX_TRIM_ROUNDS + 1):
                for remove_left in range(removed_words + 1):
                    remove_right = removed_words - remove_left
                    start = original_start + remove_left
                    end = original_end - remove_right
                    span = (start, end)
                    if end - start < MIN_WORDS_AFTER_TRIM:
                        continue
                    if span in seen_spans:
                        continue
                    seen_spans.add(span)
                    phrases.append(" ".join(tokens[start:end]))

    encode_phrase_embeddings(
        phrases,
        embedding_cache=embedding_cache
    )


def score_validation_row_for_all_configs(
    row,
    description_embedding,
    combinations,
    frequency,
    train_n,
    stopwords
):
    """Encode one CVE's candidate texts once, then score the full grid."""
    empty_results = {
        hyperparameter_run_name(*combination): (0, 1, set())
        for combination in combinations
    }
    tokens = word_tokenize(row[TEXT_COL])[:MAX_TOKENS_PER_ROW]
    if not tokens:
        return empty_results

    candidates = make_ngram_candidates(
        tokens,
        frequency,
        train_n,
        stopwords
    )
    if not candidates:
        return empty_results

    all_base_similarities = np.dot(target_embs, description_embedding)
    minimum_base_threshold = min(BASE_SIM_THRESHOLD_GRID)
    shared_target_indices = np.flatnonzero(
        all_base_similarities >= minimum_base_threshold
    )
    if not len(shared_target_indices):
        return empty_results

    occluded_texts = [
        remove_span(tokens, candidate["start"], candidate["end"])
        for candidate in candidates
    ]
    occluded_embeddings = model.encode(
        occluded_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    occluded_embeddings = np.asarray(
        occluded_embeddings,
        dtype=np.float32
    )

    shared_base_similarities = all_base_similarities[
        shared_target_indices
    ].astype(np.float32, copy=False)
    after_removal_similarities = np.dot(
        occluded_embeddings,
        target_embs[shared_target_indices].T
    ).astype(np.float32, copy=False)
    relative_drops = (
        (
            shared_base_similarities[None, :]
            - after_removal_similarities
        )
        / np.maximum(
            np.abs(shared_base_similarities[None, :]),
            1e-8
        )
    ) * 100.0

    selected_items_by_threshold = {}
    for base_threshold, drop_threshold in itertools.product(
        BASE_SIM_THRESHOLD_GRID,
        DROP_THRESHOLD_PERCENT_GRID
    ):
        selected_items_by_threshold[(base_threshold, drop_threshold)] = (
            build_selected_items_from_shared_scores(
                candidates=candidates,
                target_indices=shared_target_indices,
                base_similarities=shared_base_similarities,
                after_removal_similarities=after_removal_similarities,
                relative_drops=relative_drops,
                base_threshold=base_threshold,
                drop_threshold=drop_threshold
            )
        )

    true_ids = tuning_true_target_ids(row)
    phrase_embedding_cache = {}
    preload_tuning_phrase_embeddings(
        selected_items_by_threshold,
        tokens,
        phrase_embedding_cache
    )
    final_items_by_run = {}

    for base_threshold, drop_threshold, keep_ratio in combinations:
        selected_items = selected_items_by_threshold[
            (base_threshold, drop_threshold)
        ]
        trimming_results = trim_selected_spans_before_merge(
            selected_items,
            tokens,
            base_sim_threshold=base_threshold,
            keyword_sim_keep_ratio=keep_ratio,
            embedding_cache=phrase_embedding_cache
        )
        trimmed_items = [result[0] for result in trimming_results]
        final_items = merge_selected_spans(trimmed_items, tokens)
        run_name = hyperparameter_run_name(
            base_threshold,
            drop_threshold,
            keep_ratio
        )
        final_items_by_run[run_name] = final_items

    encode_phrase_embeddings(
        [
            item.get("phrase", "").strip().lower()
            for final_items in final_items_by_run.values()
            for item in final_items
        ],
        embedding_cache=phrase_embedding_cache
    )

    results = {}
    for run_name, final_items in final_items_by_run.items():
        predicted_ids = predicted_target_ids_for_items(
            final_items,
            phrase_embedding_cache
        )
        hit_score = int(bool(true_ids & predicted_ids))
        is_empty = int(not final_items)
        results[run_name] = (hit_score, is_empty, predicted_ids)

    return results


def f1_from_target_counts(run_statistics, prefix=None):
    """Calculate macro/micro F1 from accumulated multilabel counts."""
    true_positive = run_statistics["true_positive"]
    false_positive = run_statistics["false_positive"]
    false_negative = run_statistics["false_negative"]
    labels = set(true_positive) | set(false_positive) | set(false_negative)
    if prefix is not None:
        labels = {label for label in labels if label.startswith(prefix)}
    if not labels:
        return 0.0, 0.0

    label_f1_values = []
    total_true_positive = 0
    total_false_positive = 0
    total_false_negative = 0
    for label in labels:
        label_true_positive = true_positive[label]
        label_false_positive = false_positive[label]
        label_false_negative = false_negative[label]
        denominator = (
            2 * label_true_positive
            + label_false_positive
            + label_false_negative
        )
        label_f1_values.append(
            2 * label_true_positive / denominator if denominator else 0.0
        )
        total_true_positive += label_true_positive
        total_false_positive += label_false_positive
        total_false_negative += label_false_negative

    micro_denominator = (
        2 * total_true_positive
        + total_false_positive
        + total_false_negative
    )
    macro_f1 = float(np.mean(label_f1_values))
    micro_f1 = (
        2 * total_true_positive / micro_denominator
        if micro_denominator
        else 0.0
    )
    return macro_f1, float(micro_f1)


def make_joint_tuning_summary_rows(
    combinations,
    statistics,
    processed_rows,
    total_rows,
    validation_root
):
    rows = []
    for base_threshold, drop_threshold, keep_ratio in combinations:
        run_name = hyperparameter_run_name(
            base_threshold,
            drop_threshold,
            keep_ratio
        )
        run_statistics = statistics[run_name]
        hit_count = int(run_statistics["hit_count"])
        hit_rate = (
            hit_count / processed_rows if processed_rows else 0.0
        )
        macro_f1, micro_f1 = f1_from_target_counts(run_statistics)
        cwe_macro_f1, cwe_micro_f1 = f1_from_target_counts(
            run_statistics,
            prefix="CWE-"
        )
        capec_macro_f1, capec_micro_f1 = f1_from_target_counts(
            run_statistics,
            prefix="CAPEC-"
        )
        rows.append({
            "run_name": run_name,
            "base_sim_threshold": base_threshold,
            "drop_threshold_percent": drop_threshold,
            "keyword_sim_keep_ratio": keep_ratio,
            "processed_rows": processed_rows,
            "total_validation_rows": total_rows,
            "target_hit_count": hit_count,
            "target_miss_count": processed_rows - hit_count,
            "target_hit_rate": hit_rate,
            "target_hit_percent": hit_rate * 100.0,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "cwe_macro_f1": cwe_macro_f1,
            "cwe_micro_f1": cwe_micro_f1,
            "capec_macro_f1": capec_macro_f1,
            "capec_micro_f1": capec_micro_f1,
            "empty_occlusion_rows": int(run_statistics["empty_count"]),
            "output_directory": os.path.join(validation_root, run_name)
        })
    return rows


def make_tuning_validation_frame():
    """Reproduce the normal seed-42 validation half of held-out data."""
    if "cve_id" in test_df.columns:
        unique_cves = test_df["cve_id"].dropna().astype(str).unique()
        validation_cves, _ = train_test_split(
            unique_cves,
            test_size=0.5,
            random_state=RANDOM_SEED,
            shuffle=True
        )
        return test_df[
            test_df["cve_id"].astype(str).isin(validation_cves)
        ].reset_index(drop=True)

    validation, _ = train_test_split(
        test_df,
        test_size=0.5,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    return validation.reset_index(drop=True)


def evaluate_hyperparameter_grid_single_pass(
    validation_df,
    combinations,
    tuning_root,
    validation_root
):
    """Evaluate all configurations during one pass over validation CVEs."""
    statistics = {
        hyperparameter_run_name(*combination): {
            "hit_count": 0,
            "empty_count": 0,
            "true_positive": Counter(),
            "false_positive": Counter(),
            "false_negative": Counter()
        }
        for combination in combinations
    }
    descriptions = validation_df[TEXT_COL].fillna("").astype(str).tolist()
    description_embeddings = model.encode(
        descriptions,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    description_embeddings = np.asarray(
        description_embeddings,
        dtype=np.float32
    )
    frequency = dict(train_word_counter)
    stopwords = set(ENGLISH_STOP_WORDS)
    partial_path = os.path.join(
        tuning_root,
        "validation_summary_partial.csv"
    )

    progress = tqdm(
        validation_df.iterrows(),
        total=len(validation_df),
        desc="Joint occlusion tuning"
    )
    for row_position, (_, row) in enumerate(progress, start=1):
        row_results = score_validation_row_for_all_configs(
            row=row,
            description_embedding=description_embeddings[row_position - 1],
            combinations=combinations,
            frequency=frequency,
            train_n=len(train_df),
            stopwords=stopwords
        )
        true_ids = tuning_true_target_ids(row)
        for (
            run_name,
            (hit_score, is_empty, predicted_ids)
        ) in row_results.items():
            run_statistics = statistics[run_name]
            run_statistics["hit_count"] += hit_score
            run_statistics["empty_count"] += is_empty
            run_statistics["true_positive"].update(
                true_ids & predicted_ids
            )
            run_statistics["false_positive"].update(
                predicted_ids - true_ids
            )
            run_statistics["false_negative"].update(
                true_ids - predicted_ids
            )

        if (
            row_position % TUNING_PROGRESS_EVERY_ROWS == 0
            or row_position == len(validation_df)
        ):
            partial_rows = make_joint_tuning_summary_rows(
                combinations=combinations,
                statistics=statistics,
                processed_rows=row_position,
                total_rows=len(validation_df),
                validation_root=validation_root
            )
            pd.DataFrame(partial_rows).to_csv(partial_path, index=False)

    summary_rows = make_joint_tuning_summary_rows(
        combinations=combinations,
        statistics=statistics,
        processed_rows=len(validation_df),
        total_rows=len(validation_df),
        validation_root=validation_root
    )

    for row in summary_rows:
        run_dir = row["output_directory"]
        os.makedirs(run_dir, exist_ok=True)
        metrics_row = {
            "split": "validation",
            "rows": row["processed_rows"],
            "target_hit_count": row["target_hit_count"],
            "target_miss_count": row["target_miss_count"],
            "target_hit_rate": row["target_hit_rate"],
            "target_hit_percent": row["target_hit_percent"],
            "macro_f1": row["macro_f1"],
            "micro_f1": row["micro_f1"],
            "cwe_macro_f1": row["cwe_macro_f1"],
            "cwe_micro_f1": row["cwe_micro_f1"],
            "capec_macro_f1": row["capec_macro_f1"],
            "capec_micro_f1": row["capec_micro_f1"],
            "empty_occlusion_rows": row["empty_occlusion_rows"],
            "base_sim_threshold": row["base_sim_threshold"],
            "drop_threshold_percent": row[
                "drop_threshold_percent"
            ],
            "keyword_sim_keep_ratio": row[
                "keyword_sim_keep_ratio"
            ],
            "random_seed": RANDOM_SEED,
            "tuning_strategy": "single_pass_per_cve"
        }
        pd.DataFrame([metrics_row]).to_csv(
            os.path.join(run_dir, "evaluation_metrics.csv"),
            index=False
        )
        prune_tuning_run(run_dir)

    summary = pd.DataFrame(summary_rows).sort_values(
        "macro_f1",
        ascending=False,
        kind="stable"
    ).reset_index(drop=True)
    summary.to_csv(
        os.path.join(tuning_root, "validation_summary.csv"),
        index=False
    )
    return summary


def run_hyperparameter_tuning_single_pass():
    """Tune jointly, then run only the winning configuration in full."""
    started = time.monotonic()
    tuning_root = os.path.abspath(OUTPUT_DIR)
    validation_root = os.path.join(tuning_root, "validation")
    testing_root = os.path.join(tuning_root, "testing")
    os.makedirs(validation_root, exist_ok=True)
    os.makedirs(testing_root, exist_ok=True)

    combinations = list(itertools.product(
        BASE_SIM_THRESHOLD_GRID,
        DROP_THRESHOLD_PERCENT_GRID,
        KEYWORD_SIM_KEEP_RATIO_GRID
    ))
    validation_df = make_tuning_validation_frame()
    if validation_df.empty:
        raise ValueError("Validation split is empty; cannot tune occlusion")

    print(
        f"Joint tuning {len(combinations)} configurations over "
        f"{len(validation_df)} validation rows."
    )
    validation_summary = evaluate_hyperparameter_grid_single_pass(
        validation_df=validation_df,
        combinations=combinations,
        tuning_root=tuning_root,
        validation_root=validation_root
    )

    best = validation_summary.iloc[0]
    best_run_name = str(best["run_name"])
    best_test_dir = os.path.join(testing_root, best_run_name)
    if os.path.isdir(best_test_dir):
        shutil.rmtree(best_test_dir)

    best_environment = os.environ.copy()
    best_environment[_INTERNAL_CONFIG_ENV] = json.dumps({
        "base_sim_threshold": float(best["base_sim_threshold"]),
        "drop_threshold_percent": float(best["drop_threshold_percent"]),
        "keyword_sim_keep_ratio": float(best["keyword_sim_keep_ratio"]),
        "output_dir": best_test_dir,
        "evaluation_split": "test",
        "tune": False
    })
    print("Best validation configuration:", best_run_name)
    print("Validation target hit rate:", best["target_hit_rate"])
    print("Validation target hit percent:", best["target_hit_percent"])
    print("Validation macro F1:", best["macro_f1"])
    print("Validation micro F1:", best["micro_f1"])
    print("Running the winning configuration once on all splits.")
    subprocess.run(
        [sys.executable, os.path.abspath(__file__)],
        cwd=SCRIPT_DIR,
        check=True,
        env=best_environment
    )

    test_metrics = pd.read_csv(
        os.path.join(best_test_dir, "evaluation_metrics.csv")
    )
    test_metrics.to_csv(
        os.path.join(tuning_root, "best_test_metrics.csv"),
        index=False
    )
    test_metric = test_metrics.loc[
        test_metrics["split"] == "test"
    ].iloc[0]
    method_output_dir = publish_best_tuning_run(best_test_dir)

    best_parameters = pd.DataFrame([{
        "selection_metric": "validation_macro_f1",
        "tuning_strategy": "single_pass_per_cve",
        "base_sim_threshold": float(best["base_sim_threshold"]),
        "drop_threshold_percent": float(best["drop_threshold_percent"]),
        "keyword_sim_keep_ratio": float(best["keyword_sim_keep_ratio"]),
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
    best_parameters.to_csv(
        os.path.join(method_output_dir, "best_hyperparameters.csv"),
        index=False
    )

    elapsed_seconds = time.monotonic() - started
    timing_rows = len(df)
    full_dataset_rows = SOURCE_DATASET_ROWS
    scale = (
        full_dataset_rows / timing_rows if timing_rows else 0.0
    )
    estimated_full_seconds = elapsed_seconds * scale
    timing_path = os.path.join(tuning_root, "timing_estimate.txt")
    with open(timing_path, "w", encoding="utf-8") as timing_file:
        timing_file.write(
            f"sample_rows={timing_rows}\n"
            f"full_dataset_rows={full_dataset_rows}\n"
            f"sample_elapsed_seconds={elapsed_seconds:.3f}\n"
            f"linear_estimated_full_seconds={estimated_full_seconds:.3f}\n"
            f"linear_estimated_full_hours="
            f"{estimated_full_seconds / 3600.0:.3f}\n"
        )
    print("Saved timing estimate:", timing_path)
    print(
        "Linear full-run estimate:",
        f"{estimated_full_seconds / 3600.0:.2f} hours"
    )

    prune_tuning_run(best_test_dir)
    if not KEEP_INTERMEDIATE_FILES:
        if os.path.isdir(WORK_DIR):
            shutil.rmtree(WORK_DIR)
        if os.path.isdir(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)


if RUN_HYPERPARAMETER_TUNING:
    run_hyperparameter_tuning()
    raise SystemExit(0)

process_split("train")
process_split("test")






# ============================================================
# FINAL OUTPUT PATHS
# ============================================================

TRAIN_FINAL_PATH = os.path.join(
    OUTPUT_DIR,
    "training_final.csv"
)

VAL_FINAL_PATH = os.path.join(
    OUTPUT_DIR,
    "validation_final.csv"
)

TEST_FINAL_PATH = os.path.join(
    OUTPUT_DIR,
    "testing_final.csv"
)

INSPECTION_PATH = os.path.join(
    OUTPUT_DIR,
    "keyword_extraction_inspection.csv"
)

METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "evaluation_metrics.csv"
)

HELPER_COLS = [
    "_row_id",
    "_split_pos",
    "split"
]

DROP_COLS = [
    "occlusion_keywords",
    "occlusion_keyword_details"
]
def read_worker_output(split_name):
    path = os.path.join(
        WORK_DIR,
        f"output_{split_name}.csv"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing worker output:\n{path}"
        )

    df = pd.read_csv(path)

    if "_split_pos" in df.columns:
        df = (
            df
            .sort_values("_split_pos")
            .reset_index(drop=True)
        )

    print(
        f"Loaded {split_name}:",
        df.shape
    )

    return df


def combine_debug_fragments(debug_name):
    fragments = []

    for split_name in ["train", "test"]:

        path = os.path.join(
            TEMP_DIR,
            f"_{debug_name}_{split_name}.csv"
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing debug fragment:\n{path}"
            )

        fragments.append(
            pd.read_csv(path)
        )

    merged_debug = pd.concat(
        fragments,
        ignore_index=True
    )

    sort_columns = [
        column
        for column in [
            "row_id",
            "split",
            "start"
        ]
        if column in merged_debug.columns
    ]

    if sort_columns:
        merged_debug = (
            merged_debug
            .sort_values(sort_columns)
            .reset_index(drop=True)
        )

    output_path = os.path.join(
        TEMP_DIR,
        f"{debug_name}.csv"
    )

    merged_debug.to_csv(
        output_path,
        index=False
    )

    # Delete temporary worker fragments.
    for split_name in ["train", "test"]:

        path = os.path.join(
            TEMP_DIR,
            f"_{debug_name}_{split_name}.csv"
        )

        if os.path.exists(path):
            os.remove(path)

    print(
        "Saved merged debug file:",
        output_path
    )

    return merged_debug

def is_empty_value(x):
    if pd.isna(x):
        return True

    x = str(x).strip()

    return x in ["", "[]", "{}", "nan", "None", "none", "NaN"]


def make_final_file(df, output_path):
    final_df = df.copy()

    if "occlusion_keywords" not in final_df.columns:
        raise ValueError("occlusion_keywords column missing")

    # Pure Occlusion:
    # if occlusion_keywords is empty, keyphrases also stays empty.
    final_df["keyphrases"] = (
        final_df["occlusion_keywords"]
        .fillna("")
        .astype(str)
    )

    final_df["keyphrase_source"] = np.where(
        final_df["occlusion_keywords"].apply(is_empty_value),
        "empty",
        "occlusion"
    )

    # Remove internal/debug keyword columns if present.


    final_df = final_df.drop(
        columns=[
            col
            for col in DROP_COLS
            if col in final_df.columns
        ]
    )

    # Remove helper columns.
    final_df = final_df.drop(
        columns=[
            col
            for col in HELPER_COLS
            if col in final_df.columns
        ]
    )

    final_df.to_csv(
        output_path,
        index=False
    )

    empty_count = int(
        (final_df["keyphrase_source"] == "empty").sum()
    )

    print("Saved:", output_path)
    print("Shape:", final_df.shape)
    print("Columns:", final_df.columns.tolist())
    print("Empty Occlusion rows:", empty_count)

    return final_df


TARGET_ID_PATTERN = re.compile(r"(?:CWE|CAPEC)-\d+")


def extract_target_ids(value):
    if pd.isna(value):
        return []
    return sorted(set(TARGET_ID_PATTERN.findall(str(value))))


def true_target_ids(row):
    return sorted(set(
        extract_target_ids(row[CWE_COL])
        + extract_target_ids(row[CAPEC_COL])
    ))


def calculate_f1(true_labels, predicted_labels, prefix=None):
    """Preserve the original macro/micro multilabel F1 evaluation."""
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


def target_hit_scores(true_labels, predicted_labels):
    """Score 1 when any selected keyword target matches row truth."""
    if len(true_labels) != len(predicted_labels):
        raise ValueError("True and predicted target rows must align")

    return [
        int(bool(set(true_row) & set(predicted_row)))
        for true_row, predicted_row in zip(true_labels, predicted_labels)
    ]


def evaluate_split(df, split_name):
    true_labels = [true_target_ids(row) for _, row in df.iterrows()]
    predicted_labels = [
        extract_target_ids(value)
        for value in df["occlusion_keyword_details"]
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
    empty_count = int(df["occlusion_keywords"].apply(is_empty_value).sum())

    return {
        "split": split_name,
        "rows": len(df),
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
        "empty_occlusion_rows": empty_count,
        "base_sim_threshold": BASE_SIM_THRESHOLD,
        "drop_threshold_percent": DROP_THRESHOLD_PERCENT,
        "keyword_sim_keep_ratio": KEYWORD_SIM_KEEP_RATIO,
        "random_seed": RANDOM_SEED
    }


def save_inspection_file(split_frames):
    inspection_parts = []

    for split_name, source_df in split_frames:
        inspection = source_df.copy()

        empty_mask = inspection[
            "occlusion_keywords"
        ].apply(is_empty_value)

        # Pure Occlusion only.
        # Empty Occlusion output remains empty.
        inspection["final_keyphrases"] = (
            inspection["occlusion_keywords"]
            .fillna("")
            .astype(str)
        )

        inspection["keyphrase_source"] = np.where(
            empty_mask,
            "empty",
            "occlusion"
        )

        inspection["empty_reason"] = np.where(
            empty_mask,
            "no occlusion keyword passed the configured thresholds",
            ""
        )

        inspection["true_target_ids"] = inspection.apply(
            lambda row: ";".join(
                true_target_ids(row)
            ),
            axis=1
        )

        inspection["predicted_target_ids"] = inspection[
            "occlusion_keyword_details"
        ].apply(
            lambda value: ";".join(
                extract_target_ids(value)
            )
        )

        inspection["final_split"] = split_name

        inspection["base_sim_threshold"] = (
            BASE_SIM_THRESHOLD
        )

        inspection["drop_threshold_percent"] = (
            DROP_THRESHOLD_PERCENT
        )

        inspection["keyword_sim_keep_ratio"] = (
            KEYWORD_SIM_KEEP_RATIO
        )

        inspection["ngram_min"] = NGRAM_MIN
        inspection["ngram_max"] = NGRAM_MAX

        inspection["max_tokens_per_row"] = (
            MAX_TOKENS_PER_ROW
        )

        inspection[
            "top_selected_spans_before_merge"
        ] = TOP_SELECTED_SPANS_BEFORE_MERGE

        inspection[
            "top_final_keywords"
        ] = TOP_FINAL_KEYWORDS

        inspection[
            "max_merged_keyword_words"
        ] = MAX_MERGED_KEYWORD_WORDS

        inspection[
            "min_words_after_trim"
        ] = MIN_WORDS_AFTER_TRIM

        inspection[
            "max_trim_rounds"
        ] = MAX_TRIM_ROUNDS

        inspection["random_seed"] = RANDOM_SEED

        inspection_parts.append(
            inspection
        )

    inspection_df = pd.concat(
        inspection_parts,
        ignore_index=True
    )

    if "_row_id" in inspection_df.columns:
        inspection_df = (
            inspection_df
            .sort_values("_row_id")
            .reset_index(drop=True)
        )




    empty_rows = int(
        (
            inspection_df["keyphrase_source"]
            == "empty"
        ).sum()
    )

    print(
        "Saved inspection file:",
        INSPECTION_PATH
    )

    print(
        "Total empty Occlusion rows:",
        empty_rows
    )

    for split_name, split_df in (
        inspection_df.groupby("final_split")
    ):

        split_empty = int(
            (
                split_df["keyphrase_source"]
                == "empty"
            ).sum()
        )

        percent = (
            split_empty
            / len(split_df)
            * 100
            if len(split_df)
            else 0.0
        )

        print(
            f"Empty Occlusion {split_name}: "
            f"{split_empty}/{len(split_df)} "
            f"({percent:.2f}%)"
        )

# ============================================================
# MERGE 70% TRAIN AND 30% TEMP/TEST 
# ============================================================

train_merged = read_worker_output("train")
temp_merged = read_worker_output("test")

occlusion_row_debug = combine_debug_fragments("occlusion_row_debug")
occlusion_candidate_debug = combine_debug_fragments("occlusion_candidate_debug")
trimming_debug = combine_debug_fragments("trimming_debug")

print("\nMerged train:", train_merged.shape)
print("Merged temp/test 30%:", temp_merged.shape)


# ============================================================
# SPLIT THE HELD-OUT 30% INTO EQUAL 15% VALIDATION AND 15% TEST
# ============================================================

# Best option: split by CVE ID so same CVE does not appear in both val and test
if "cve_id" in temp_merged.columns:
    unique_cves = temp_merged["cve_id"].dropna().astype(str).unique()

    val_cves, test_cves = train_test_split(
        unique_cves,
        test_size=0.5,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    val_merged = temp_merged[
        temp_merged["cve_id"].astype(str).isin(val_cves)
    ].reset_index(drop=True)

    test_merged = temp_merged[
        temp_merged["cve_id"].astype(str).isin(test_cves)
    ].reset_index(drop=True)

else:
    print("Warning: cve_id not found. Doing row-level split.")

    val_merged, test_merged = train_test_split(
        temp_merged,
        test_size=0.5,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    val_merged = val_merged.reset_index(drop=True)
    test_merged = test_merged.reset_index(drop=True)


print("\nFinal split before cleaning:")
print("Train:", train_merged.shape)
print("Validation:", val_merged.shape)
print("Test:", test_merged.shape)

total_rows = len(train_merged) + len(val_merged) + len(test_merged)

print("\nApprox percentages:")
print("Train %:", round(len(train_merged) / total_rows * 100, 2))
print("Val %:", round(len(val_merged) / total_rows * 100, 2))
print("Test %:", round(len(test_merged) / total_rows * 100, 2))


# ============================================================
# SAVE THE COMBINED INSPECTION REPORT AND EVALUATION METRICS
# ============================================================




metrics_rows = []
if EVALUATION_SPLIT in {"validation", "both"}:
    metrics_rows.append(evaluate_split(val_merged, "validation"))
if EVALUATION_SPLIT in {"test", "both"}:
    metrics_rows.append(evaluate_split(test_merged, "test"))

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(METRICS_PATH, index=False)
print("Saved evaluation metrics:", METRICS_PATH)
print(metrics_df.to_string(index=False))


# ============================================================
# SAVE FINAL FILES WITH OCCLUSION KEYPHRASES
# ============================================================

training_final = make_final_file(train_merged, TRAIN_FINAL_PATH)
validation_final = make_final_file(val_merged, VAL_FINAL_PATH)
testing_final = make_final_file(test_merged, TEST_FINAL_PATH)

print("\nDONE")
print("Final files:")
print(TRAIN_FINAL_PATH)
print(VAL_FINAL_PATH)
print(TEST_FINAL_PATH)
print(INSPECTION_PATH)
print(METRICS_PATH)

print("\nPreview training_final:")
print(training_final.head())

print("\nPreview validation_final:")
print(validation_final.head())

print("\nPreview testing_final:")
print(testing_final.head())


empty_counts = {
    "train": int(train_merged["occlusion_keywords"].apply(is_empty_value).sum()),
    "validation": int(val_merged["occlusion_keywords"].apply(is_empty_value).sum()),
    "test": int(test_merged["occlusion_keywords"].apply(is_empty_value).sum())
}
debug_summary = pd.DataFrame([{
    "dataset_rows": total_rows,
    "train_rows": len(train_merged),
    "validation_rows": len(val_merged),
    "test_rows": len(test_merged),
    "occlusion_row_debug_rows": len(occlusion_row_debug),
    "occlusion_candidate_debug_rows": len(occlusion_candidate_debug),
    "trimming_debug_rows": len(trimming_debug),
    "candidate_debug_rows": DEBUG_ROWS,
    "train_empty_occlusion_rows": empty_counts["train"],
    "validation_empty_occlusion_rows": empty_counts["validation"],
    "test_empty_occlusion_rows": empty_counts["test"],
    "base_sim_threshold": BASE_SIM_THRESHOLD,
    "drop_threshold_percent": DROP_THRESHOLD_PERCENT,
    "keyword_sim_keep_ratio": KEYWORD_SIM_KEEP_RATIO,
    "random_seed": RANDOM_SEED
}])
debug_summary_path = os.path.join(TEMP_DIR, "debug_summary.csv")
debug_summary.to_csv(debug_summary_path, index=False)
print("Saved debug summary:", debug_summary_path)

if not KEEP_INTERMEDIATE_FILES:
    shutil.rmtree(WORK_DIR)
    shutil.rmtree(TEMP_DIR)
    print("Removed intermediate worker and temp files.")
