#!/usr/bin/env python3

"""Generate target-aware keywords using direct phrase similarity on one GPU."""

import itertools
import json
import os
import random
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PAPER_DIR = os.path.dirname(os.path.dirname(PROJECT_DIR))
OUTPUTS_ROOT = os.path.abspath(os.environ.get(
    "KEYWORD_MAKER_OUTPUT_ROOT",
    os.path.join(PROJECT_DIR, "outputs")
))

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
# REPRODUCIBILITY AND HYPERPARAMETERS
# ============================================================

RANDOM_SEED = 42

# Full grid: 4 x 4 x 4 = 64 validation configurations.
BASE_SIM_THRESHOLD_GRID = [0.35, 0.40, 0.45, 0.50]
PHRASE_SIM_THRESHOLD_GRID = [0.35, 0.40, 0.45, 0.50]
KEYWORD_SIM_KEEP_RATIO_GRID = [0.80, 0.85, 0.90, 0.95]

TEXT_COL = "cleaned_description"
CWE_COL = "weakness"
CAPEC_COL = "capec_id"

TRAIN_SIZE = 0.70

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256

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

DEBUG_ROWS = 25
TUNING_PROGRESS_EVERY_ROWS = 10

# Edit these values directly to configure a run.
BASE_SIM_THRESHOLD = 0.40
PHRASE_SIM_THRESHOLD = 0.42
KEYWORD_SIM_KEEP_RATIO = 0.90
OUTPUT_DIR = None
EVALUATION_SPLIT = "both"
RUN_HYPERPARAMETER_TUNING = True
KEEP_INTERMEDIATE_FILES = False


# ============================================================
# CODE-LEVEL RUN CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class RunConfig:
    base_sim_threshold: float
    phrase_sim_threshold: float
    keyword_sim_keep_ratio: float
    output_dir: str
    evaluation_split: str
    tune: bool


def make_run_config():
    output_dir = OUTPUT_DIR
    if output_dir is None:
        output_dir = os.path.join(
            OUTPUTS_ROOT,
            "phrase_similarity"
        )
        if RUN_HYPERPARAMETER_TUNING:
            output_dir = os.path.join(
                output_dir,
                "hyperparameter_tuning"
            )

    config = RunConfig(
        base_sim_threshold=BASE_SIM_THRESHOLD,
        phrase_sim_threshold=PHRASE_SIM_THRESHOLD,
        keyword_sim_keep_ratio=KEYWORD_SIM_KEEP_RATIO,
        output_dir=os.path.abspath(output_dir),
        evaluation_split=EVALUATION_SPLIT,
        tune=RUN_HYPERPARAMETER_TUNING
    )

    if not 0.0 <= config.base_sim_threshold <= 1.0:
        raise ValueError("BASE_SIM_THRESHOLD must be between 0 and 1")
    if not 0.0 <= config.phrase_sim_threshold <= 1.0:
        raise ValueError("PHRASE_SIM_THRESHOLD must be between 0 and 1")
    if not 0.0 <= config.keyword_sim_keep_ratio <= 1.0:
        raise ValueError("KEYWORD_SIM_KEEP_RATIO must be between 0 and 1")
    if config.evaluation_split not in {"validation", "test", "both"}:
        raise ValueError(
            "EVALUATION_SPLIT must be validation, test, or both"
        )

    return config


def set_random_seed():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

def hyperparameter_run_name(base_threshold, phrase_threshold, keep_ratio):
    return (
        f"base_{base_threshold:.2f}__"
        f"phrase_{phrase_threshold:.2f}__"
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
    method_output_dir = os.path.join(OUTPUTS_ROOT, "phrase_similarity")
    os.makedirs(method_output_dir, exist_ok=True)

    for filename in FINAL_OUTPUT_FILENAMES:
        source = os.path.join(run_dir, filename)
        if not os.path.isfile(source):
            raise FileNotFoundError(
                f"Winning tuning run is missing final output: {source}"
            )
        shutil.copy2(source, os.path.join(method_output_dir, filename))

    shutil.copy2(
        os.path.join(run_dir, "evaluation_metrics.csv"),
        os.path.join(method_output_dir, "evaluation_metrics.csv")
    )
    config_path = os.path.join(run_dir, "phrase_similarity_config.json")
    if os.path.isfile(config_path):
        shutil.copy2(
            config_path,
            os.path.join(method_output_dir, "phrase_similarity_config.json")
        )
    return method_output_dir


def run_hyperparameter_tuning(config):
    return run_hyperparameter_tuning_single_pass(config)


# ============================================================
# TEXT AND CANDIDATE HELPERS
# ============================================================

def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value)


def collapse_consecutive_duplicate_tokens(text):
    final_tokens = []
    for token in safe_text(text).split():
        if final_tokens and token.casefold() == final_tokens[-1].casefold():
            continue
        final_tokens.append(token)
    return " ".join(final_tokens)


def clean_text(value):
    return collapse_consecutive_duplicate_tokens(safe_text(value).strip())


def word_tokenize(text):
    return re.findall(
        r"[a-zA-Z][a-zA-Z0-9_+\-]*",
        clean_text(text).lower()
    )


def build_train_word_frequency(train_texts):
    counter = Counter()
    for text in tqdm(train_texts, desc="Building train word frequency"):
        counter.update(set(word_tokenize(text)))
    return counter


def make_frequency_dataframe(counter, train_n):
    rows = []
    for word, count in counter.most_common():
        percentage = (count / train_n) * 100 if train_n else 0.0
        is_too_common = percentage > HIGH_FREQ_PERCENT_THRESHOLD
        is_too_rare = count <= LOW_FREQ_ROW_THRESHOLD
        rows.append({
            "word": word,
            "row_frequency": count,
            "percentage": percentage,
            "is_too_common": is_too_common,
            "is_too_rare": is_too_rare,
            "allowed_for_phrase_similarity_keywords": (
                not is_too_common and not is_too_rare
            )
        })
    return pd.DataFrame(rows)


def is_allowed_keyword_word(word, frequency, train_n):
    count = frequency.get(word, 0)
    percentage = (count / train_n) * 100 if train_n else 0.0
    return not (
        percentage > HIGH_FREQ_PERCENT_THRESHOLD
        or count <= LOW_FREQ_ROW_THRESHOLD
    )


def make_ngram_candidates(tokens, frequency, train_n):
    tokens = tokens[:MAX_TOKENS_PER_ROW]
    stopwords = set(ENGLISH_STOP_WORDS)
    candidates = []
    seen = set()

    for ngram_size in range(NGRAM_MIN, NGRAM_MAX + 1):
        for start in range(len(tokens) - ngram_size + 1):
            end = start + ngram_size
            words = tokens[start:end]

            if any(
                not is_allowed_keyword_word(word, frequency, train_n)
                for word in words
            ):
                continue
            if all(word in stopwords for word in words):
                continue
            if all(len(word) <= 1 for word in words):
                continue

            phrase = " ".join(words)
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


def format_capec_id(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return f"CAPEC-{int(float(value))}"
    except (TypeError, ValueError):
        return f"CAPEC-{value}"


# ============================================================
# DATA AND TARGET LOADING
# ============================================================

def load_dataset():
    for name, path in [
        ("DATASET", DATASET_PATH),
        ("CWE", CWE_PATH),
        ("CAPEC", CAPEC_PATH)
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found:\n{path}")

    frame = pd.read_csv(DATASET_PATH)
    frame.columns = frame.columns.str.strip()
    required = [TEXT_COL, CWE_COL, CAPEC_COL]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Available: {frame.columns.tolist()}"
        )

    frame[TEXT_COL] = (
        frame[TEXT_COL]
        .fillna("")
        .astype(str)
        .map(collapse_consecutive_duplicate_tokens)
    )
    frame["_row_id"] = np.arange(len(frame))
    return frame


def split_dataset(frame):
    if "cve_id" in frame.columns:
        unique_cves = frame["cve_id"].dropna().astype(str).unique()
        train_cves, held_out_cves = train_test_split(
            unique_cves,
            train_size=TRAIN_SIZE,
            random_state=RANDOM_SEED,
            shuffle=True
        )
        cve_ids = frame["cve_id"].astype(str)
        training = frame[cve_ids.isin(train_cves)].copy()
        held_out = frame[cve_ids.isin(held_out_cves)].copy()
    else:
        train_indices, held_out_indices = train_test_split(
            frame.index,
            train_size=TRAIN_SIZE,
            random_state=RANDOM_SEED,
            shuffle=True
        )
        training = frame.loc[train_indices].copy()
        held_out = frame.loc[held_out_indices].copy()

    training = training.reset_index(drop=True)
    held_out = held_out.reset_index(drop=True)
    training["split"] = "train"
    held_out["split"] = "test"
    training["_split_pos"] = np.arange(len(training))
    held_out["_split_pos"] = np.arange(len(held_out))
    return training, held_out


def split_validation_testing(held_out):
    """Create the shared seed-42 validation/test split at CVE-ID level."""
    if "cve_id" in held_out.columns:
        validation_cves, testing_cves = train_test_split(
            held_out["cve_id"].dropna().astype(str).unique(),
            test_size=0.5,
            random_state=RANDOM_SEED,
            shuffle=True
        )
        held_ids = held_out["cve_id"].astype(str)
        validation = held_out[
            held_ids.isin(validation_cves)
        ].reset_index(drop=True)
        testing = held_out[
            held_ids.isin(testing_cves)
        ].reset_index(drop=True)
    else:
        validation, testing = train_test_split(
            held_out,
            test_size=0.5,
            random_state=RANDOM_SEED,
            shuffle=True
        )
        validation = validation.reset_index(drop=True)
        testing = testing.reset_index(drop=True)

    return validation, testing


def load_target_knowledge():
    cwe_df = pd.read_csv(CWE_PATH)
    capec_df = pd.read_csv(CAPEC_PATH)
    cwe_df.columns = cwe_df.columns.str.strip()
    capec_df.columns = capec_df.columns.str.strip()

    required_cwe = {"cwe_id", "cwe_name", "description"}
    required_capec = {"ID", "Name", "Description"}
    if required_cwe - set(cwe_df.columns):
        raise ValueError(
            f"Missing CWE columns: {sorted(required_cwe - set(cwe_df.columns))}"
        )
    if required_capec - set(capec_df.columns):
        raise ValueError(
            "Missing CAPEC columns: "
            f"{sorted(required_capec - set(capec_df.columns))}"
        )

    target_texts = []
    target_meta = []

    for _, row in cwe_df.iterrows():
        target_id = clean_text(row["cwe_id"])
        if not target_id:
            continue
        for target_type, column in (
            ("cwe_name", "cwe_name"),
            ("cwe_description", "description")
        ):
            text = clean_text(row[column])
            if text:
                target_texts.append(text)
                target_meta.append({
                    "id": target_id,
                    "target_type": target_type,
                    "source": "cwe",
                    "text": text
                })

    for _, row in capec_df.iterrows():
        target_id = format_capec_id(row["ID"])
        if target_id is None:
            continue
        for target_type, column in (
            ("capec_name", "Name"),
            ("capec_description", "Description")
        ):
            text = clean_text(row[column])
            if text:
                target_texts.append(text)
                target_meta.append({
                    "id": target_id,
                    "target_type": target_type,
                    "source": "capec",
                    "text": text
                })

    return target_texts, target_meta


# ============================================================
# DIRECT PHRASE-SIMILARITY EXTRACTOR
# ============================================================

class PhraseSimilarityExtractor:
    def __init__(
        self,
        model,
        target_texts,
        target_meta,
        frequency,
        train_n,
        base_threshold,
        phrase_threshold,
        keep_ratio
    ):
        self.model = model
        self.target_texts = target_texts
        self.target_meta = target_meta
        self.frequency = frequency
        self.train_n = train_n
        self.base_threshold = base_threshold
        self.phrase_threshold = phrase_threshold
        self.keep_ratio = keep_ratio

        self.target_embeddings = np.asarray(
            model.encode(
                target_texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=True,
                normalize_embeddings=True
            ),
            dtype=np.float32
        )
        self.target_key_to_index = {
            (meta["id"], meta["target_type"]): index
            for index, meta in enumerate(target_meta)
        }

    def target_indices_for_selected_item(self, item):
        indices = []
        for target_info in item.get("matched_targets", {}).values():
            key = (
                target_info.get("target_id"),
                target_info.get("target_type")
            )
            if key in self.target_key_to_index:
                indices.append(self.target_key_to_index[key])
        return sorted(set(indices))

    def target_indices_for_merged_item(self, item):
        indices = []
        for target_dict in item.get("matched_targets", {}).values():
            for target_info in target_dict.values():
                key = (
                    target_info.get("target_id"),
                    target_info.get("target_type")
                )
                if key in self.target_key_to_index:
                    indices.append(self.target_key_to_index[key])
        return sorted(set(indices))

    def encode_phrases(self, phrases, embedding_cache=None):
        """Encode each unique phrase at most once within a CVE."""
        if not phrases:
            return np.empty(
                (0, self.target_embeddings.shape[1]),
                dtype=np.float32
            )

        if embedding_cache is None:
            return np.asarray(
                self.model.encode(
                    phrases,
                    batch_size=BATCH_SIZE,
                    show_progress_bar=False,
                    normalize_embeddings=True
                ),
                dtype=np.float32
            )

        missing_phrases = list(dict.fromkeys(
            phrase for phrase in phrases if phrase not in embedding_cache
        ))
        if missing_phrases:
            missing_embeddings = np.asarray(
                self.model.encode(
                    missing_phrases,
                    batch_size=BATCH_SIZE,
                    show_progress_bar=False,
                    normalize_embeddings=True
                ),
                dtype=np.float32
            )
            embedding_cache.update(zip(
                missing_phrases,
                missing_embeddings
            ))

        return np.asarray(
            [embedding_cache[phrase] for phrase in phrases],
            dtype=np.float32
        )

    def batch_phrase_target_similarities(
        self,
        phrases,
        target_groups,
        embedding_cache=None
    ):
        if not phrases:
            return []
        phrase_embeddings = self.encode_phrases(
            phrases,
            embedding_cache=embedding_cache
        )
        results = []
        for phrase_embedding, target_indices in zip(
            phrase_embeddings,
            target_groups
        ):
            if not target_indices:
                results.append(0.0)
                continue
            indices = np.asarray(target_indices, dtype=int)
            results.append(float(np.max(
                np.dot(self.target_embeddings[indices], phrase_embedding)
            )))
        return results

    def trim_selected_spans(
        self,
        selected_items,
        tokens,
        phrase_threshold=None,
        keep_ratio=None,
        embedding_cache=None
    ):
        if phrase_threshold is None:
            phrase_threshold = self.phrase_threshold
        if keep_ratio is None:
            keep_ratio = self.keep_ratio

        states = []
        original_phrases = []
        original_target_groups = []
        active_indices = []

        for state_index, item in enumerate(selected_items):
            start = int(item["start"])
            end = int(item["end"])
            if not 0 <= start < end <= len(tokens):
                raise ValueError(f"Invalid selected span [{start}, {end})")

            target_indices = self.target_indices_for_selected_item(item)
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
                    end - start > MIN_WORDS_AFTER_TRIM
                    and bool(target_indices)
                )
            }
            states.append(state)

            if state["active"]:
                original_phrases.append(phrase)
                original_target_groups.append(target_indices)
                active_indices.append(state_index)

        original_similarities = self.batch_phrase_target_similarities(
            original_phrases,
            original_target_groups,
            embedding_cache=embedding_cache
        )
        for state_index, similarity in zip(
            active_indices,
            original_similarities
        ):
            state = states[state_index]
            state["original_similarity"] = similarity
            state["final_similarity"] = similarity
            state["minimum_required_similarity"] = max(
                phrase_threshold,
                keep_ratio * similarity
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
                    option_phrases.append(
                        " ".join(tokens[new_start:new_end])
                    )
                    option_target_groups.append(state["target_indices"])
                    option_meta.append((state_index, new_start, new_end))

            option_similarities = self.batch_phrase_target_similarities(
                option_phrases,
                option_target_groups,
                embedding_cache=embedding_cache
            )
            best_options = {}

            for metadata, similarity in zip(
                option_meta,
                option_similarities
            ):
                state_index, new_start, new_end = metadata
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

    def merge_selected_spans(self, selected_items, tokens):
        if not selected_items:
            return []

        selected_items = sorted(
            selected_items,
            key=lambda item: (item["start"], item["end"])
        )
        groups = []

        for item in selected_items:
            if not groups or item["start"] > groups[-1]["end"]:
                groups.append({
                    "start": item["start"],
                    "end": item["end"],
                    "items": [item],
                    "max_phrase_similarity": item[
                        "max_phrase_similarity"
                    ]
                })
                continue

            last = groups[-1]
            last["end"] = max(last["end"], item["end"])
            last["items"].append(item)
            last["max_phrase_similarity"] = max(
                last["max_phrase_similarity"],
                item["max_phrase_similarity"]
            )

        merged = []
        for group in groups:
            merged_tokens = tokens[group["start"]:group["end"]]
            phrase = " ".join(merged_tokens)

            if len(merged_tokens) > MAX_MERGED_KEYWORD_WORDS:
                best_item = max(
                    group["items"],
                    key=lambda item: (
                        item["max_phrase_similarity"],
                        len(item["phrase"].split())
                    )
                )
                phrase = best_item["phrase"]
                final_start = best_item["start"]
                final_end = best_item["end"]
            else:
                final_start = group["start"]
                final_end = group["end"]

            merged.append({
                "phrase": phrase,
                "start": final_start,
                "end": final_end,
                "max_phrase_similarity": float(
                    group["max_phrase_similarity"]
                ),
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

        merged.sort(
            key=lambda item: (
                item["max_phrase_similarity"],
                len(item["phrase"].split())
            ),
            reverse=True
        )

        final = []
        seen = set()
        for item in merged:
            if item["phrase"] in seen:
                continue
            seen.add(item["phrase"])
            final.append(item)
            if len(final) >= TOP_FINAL_KEYWORDS:
                break
        return final

    def phrase_detail(
        self,
        phrase,
        target_indices,
        embedding_cache=None
    ):
        phrase = phrase.strip().lower()
        if not phrase:
            return ""
        if not target_indices:
            return f"{phrase} -> no_match"

        phrase_embedding = self.encode_phrases(
            [phrase],
            embedding_cache=embedding_cache
        )[0]
        indices = np.asarray(target_indices, dtype=int)
        similarities = np.dot(
            self.target_embeddings[indices],
            phrase_embedding
        )
        best_index = int(indices[int(np.argmax(similarities))])
        best_meta = self.target_meta[best_index]
        return (
            f"{phrase} -> {best_meta['id']} | "
            f"part={best_meta['target_type']} | "
            f"similarity={float(np.max(similarities)):.4f}"
        )

    def merged_details(self, final_items, embedding_cache=None):
        details = []
        for item in final_items:
            details.append(self.phrase_detail(
                item["phrase"],
                self.target_indices_for_merged_item(item),
                embedding_cache=embedding_cache
            ))
        return "; ".join(detail for detail in details if detail)

    def process_frame(self, frame, split_name, temp_dir):
        description_embeddings = np.asarray(
            self.model.encode(
                frame[TEXT_COL].fillna("").astype(str).tolist(),
                batch_size=BATCH_SIZE,
                show_progress_bar=True,
                normalize_embeddings=True
            ),
            dtype=np.float32
        )

        keyword_values = []
        detail_values = []
        row_debug = []
        candidate_debug = []
        trimming_debug = []

        for row_position, (index, row) in enumerate(tqdm(
            frame.iterrows(),
            total=len(frame),
            desc=f"Phrase similarity {split_name}"
        )):
            tokens = word_tokenize(row[TEXT_COL])[:MAX_TOKENS_PER_ROW]
            row_id = row.get("_row_id", index)
            cve_id = row.get("cve_id", "")
            collect_debug = row_position < DEBUG_ROWS
            embedding_cache = {}

            base_similarities = np.dot(
                self.target_embeddings,
                description_embeddings[row_position]
            )
            usable_target_indices = np.where(
                base_similarities >= self.base_threshold
            )[0]

            if not tokens or not len(usable_target_indices):
                keyword_values.append("")
                detail_values.append("")
                row_debug.append({
                    "row_id": row_id,
                    "cve_id": cve_id,
                    "split": split_name,
                    "token_count": len(tokens),
                    "total_target_texts": len(self.target_texts),
                    "relevant_target_texts": len(usable_target_indices),
                    "candidate_count": 0,
                    "passed_phrase_threshold_count": 0,
                    "selected_for_merge_count": 0,
                    "final_keyword_count": 0,
                    "status": (
                        "no_tokens" if not tokens else "no_relevant_targets"
                    ),
                    "pretrim_keywords": ""
                })
                continue

            candidates = make_ngram_candidates(
                tokens,
                self.frequency,
                self.train_n
            )
            if not candidates:
                keyword_values.append("")
                detail_values.append("")
                row_debug.append({
                    "row_id": row_id,
                    "cve_id": cve_id,
                    "split": split_name,
                    "token_count": len(tokens),
                    "total_target_texts": len(self.target_texts),
                    "relevant_target_texts": len(usable_target_indices),
                    "candidate_count": 0,
                    "passed_phrase_threshold_count": 0,
                    "selected_for_merge_count": 0,
                    "final_keyword_count": 0,
                    "status": "no_allowed_candidates",
                    "pretrim_keywords": ""
                })
                continue

            # Core difference from occlusion: encode the candidate phrase itself.
            phrase_embeddings = self.encode_phrases(
                [candidate["phrase"] for candidate in candidates],
                embedding_cache=embedding_cache
            )
            similarity_matrix = np.dot(
                phrase_embeddings,
                self.target_embeddings[usable_target_indices].T
            )

            selected_items = []
            current_candidate_debug = []

            for candidate_index, candidate in enumerate(candidates):
                matched_targets = {}
                max_similarity = -1.0

                for target_position, global_target_index in enumerate(
                    usable_target_indices
                ):
                    phrase_similarity = float(
                        similarity_matrix[candidate_index, target_position]
                    )
                    if phrase_similarity < self.phrase_threshold:
                        continue

                    meta = self.target_meta[global_target_index]
                    target_key = f"{meta['id']}::{meta['target_type']}"
                    matched_targets[target_key] = {
                        "target_id": meta["id"],
                        "target_type": meta["target_type"],
                        "target_source": meta["source"],
                        "description_target_similarity": float(
                            base_similarities[global_target_index]
                        ),
                        "phrase_target_similarity": phrase_similarity
                    }
                    max_similarity = max(max_similarity, phrase_similarity)

                if matched_targets:
                    selected_items.append({
                        "phrase": candidate["phrase"],
                        "start": candidate["start"],
                        "end": candidate["end"],
                        "max_phrase_similarity": max_similarity,
                        "matched_targets": matched_targets
                    })

                if collect_debug:
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
                        "max_phrase_similarity": (
                            max_similarity if matched_targets else np.nan
                        ),
                        "passed_phrase_threshold": bool(matched_targets),
                        "selected_for_merge": False,
                        "base_sim_threshold": self.base_threshold,
                        "phrase_sim_threshold": self.phrase_threshold
                    })

            selected_items.sort(
                key=lambda item: (
                    item["max_phrase_similarity"],
                    len(item["phrase"].split())
                ),
                reverse=True
            )
            passed_count = len(selected_items)
            selected_items = selected_items[
                :TOP_SELECTED_SPANS_BEFORE_MERGE
            ]

            selected_keys = {
                (item["phrase"], item["start"], item["end"])
                for item in selected_items
            }
            for record in current_candidate_debug:
                record["selected_for_merge"] = (
                    record["phrase"], record["start"], record["end"]
                ) in selected_keys
            candidate_debug.extend(current_candidate_debug)

            pretrim_items = self.merge_selected_spans(selected_items, tokens)
            pretrim_keywords = [item["phrase"] for item in pretrim_items]

            trimming_results = self.trim_selected_spans(
                selected_items,
                tokens,
                embedding_cache=embedding_cache
            )
            trimmed_items = [result[0] for result in trimming_results]

            if collect_debug:
                for trimmed_item, trim_info, target_indices in trimming_results:
                    trimming_debug.append({
                        "row_id": row_id,
                        "cve_id": cve_id,
                        "split": split_name,
                        "original_phrase": trim_info["original_phrase"],
                        "final_phrase": trimmed_item["phrase"],
                        "changed": (
                            trimmed_item["phrase"]
                            != trim_info["original_phrase"]
                        ),
                        "decision": trim_info["decision"],
                        "original_target_similarity": trim_info[
                            "original_similarity"
                        ],
                        "final_target_similarity": trim_info[
                            "final_similarity"
                        ],
                        "minimum_required_similarity": trim_info[
                            "minimum_required_similarity"
                        ],
                        "trim_rounds": trim_info["trim_rounds"],
                        "matched_target_count": len(target_indices),
                        "final_target_detail": self.phrase_detail(
                            trimmed_item["phrase"],
                            target_indices,
                            embedding_cache=embedding_cache
                        ),
                        "phrase_sim_threshold": self.phrase_threshold,
                        "keyword_sim_keep_ratio": self.keep_ratio
                    })

            final_items = self.merge_selected_spans(trimmed_items, tokens)
            final_keywords = [item["phrase"] for item in final_items]
            keyword_text = "; ".join(final_keywords)
            detail_text = self.merged_details(
                final_items,
                embedding_cache=embedding_cache
            )
            keyword_values.append(keyword_text)
            detail_values.append(detail_text)
            row_debug.append({
                "row_id": row_id,
                "cve_id": cve_id,
                "split": split_name,
                "token_count": len(tokens),
                "total_target_texts": len(self.target_texts),
                "relevant_target_texts": len(usable_target_indices),
                "candidate_count": len(candidates),
                "passed_phrase_threshold_count": passed_count,
                "selected_for_merge_count": len(selected_items),
                "final_keyword_count": len(final_keywords),
                "status": (
                    "phrase_similarity_keywords_found"
                    if final_keywords
                    else "no_keyword_selected"
                ),
                "pretrim_keywords": "; ".join(pretrim_keywords)
            })

        output = frame.copy()
        output["phrase_similarity_keywords"] = keyword_values
        output["phrase_similarity_keyword_details"] = detail_values

        for record, keywords, details in zip(
            row_debug,
            keyword_values,
            detail_values
        ):
            record["final_keywords"] = keywords
            record["final_keyword_details"] = details

        debug_sets = {
            "phrase_similarity_row_debug": row_debug,
            "phrase_similarity_candidate_debug": candidate_debug,
            "phrase_similarity_trimming_debug": trimming_debug
        }
        for debug_name, records in debug_sets.items():
            pd.DataFrame(records).to_csv(
                os.path.join(temp_dir, f"_{debug_name}_{split_name}.csv"),
                index=False
            )

        return output


# ============================================================
# EVALUATION AND OUTPUT HELPERS
# ============================================================

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


def evaluate_split(frame, split_name, config):
    true_labels = [true_target_ids(row) for _, row in frame.iterrows()]
    predicted_labels = [
        extract_target_ids(value)
        for value in frame["phrase_similarity_keyword_details"]
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
    empty_count = int(
        frame["phrase_similarity_keywords"].fillna("").eq("").sum()
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
        "empty_phrase_similarity_rows": empty_count,
        "base_sim_threshold": config.base_sim_threshold,
        "phrase_sim_threshold": config.phrase_sim_threshold,
        "keyword_sim_keep_ratio": config.keyword_sim_keep_ratio,
        "random_seed": RANDOM_SEED
    }


def build_selected_items_from_shared_scores(
    extractor,
    candidates,
    shared_target_indices,
    shared_base_similarities,
    shared_phrase_similarities,
    base_threshold,
    phrase_threshold
):
    """Select one configuration from shared per-CVE similarity matrices."""
    eligible_positions = np.flatnonzero(
        shared_base_similarities >= base_threshold
    )
    if not len(eligible_positions):
        return []

    eligible_phrase_similarities = shared_phrase_similarities[
        :,
        eligible_positions
    ]
    matched = eligible_phrase_similarities >= phrase_threshold
    passing_candidate_indices = np.flatnonzero(matched.any(axis=1))
    selected_items = []

    for candidate_index in passing_candidate_indices:
        local_matches = np.flatnonzero(matched[candidate_index])
        matched_positions = eligible_positions[local_matches]
        matched_targets = {}

        for target_position in matched_positions:
            global_target_index = int(
                shared_target_indices[target_position]
            )
            phrase_similarity = float(
                shared_phrase_similarities[
                    candidate_index,
                    target_position
                ]
            )
            meta = extractor.target_meta[global_target_index]
            target_key = f"{meta['id']}::{meta['target_type']}"
            matched_targets[target_key] = {
                "target_id": meta["id"],
                "target_type": meta["target_type"],
                "target_source": meta["source"],
                "description_target_similarity": float(
                    shared_base_similarities[target_position]
                ),
                "phrase_target_similarity": phrase_similarity
            }

        candidate = candidates[int(candidate_index)]
        selected_items.append({
            "phrase": candidate["phrase"],
            "start": candidate["start"],
            "end": candidate["end"],
            "max_phrase_similarity": float(np.max(
                shared_phrase_similarities[
                    candidate_index,
                    matched_positions
                ]
            )),
            "matched_targets": matched_targets
        })

    selected_items.sort(
        key=lambda item: (
            item["max_phrase_similarity"],
            len(item["phrase"].split())
        ),
        reverse=True
    )
    return selected_items[:TOP_SELECTED_SPANS_BEFORE_MERGE]


def preload_tuning_phrase_embeddings(
    extractor,
    selected_items_by_threshold,
    tokens,
    embedding_cache
):
    """Batch-encode all phrases that joint edge trimming may inspect."""
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

    extractor.encode_phrases(
        phrases,
        embedding_cache=embedding_cache
    )


def predicted_target_ids_for_items(
    extractor,
    final_items,
    embedding_cache
):
    """Match final phrases exactly as the normal details evaluator does."""
    phrases = [
        item.get("phrase", "").strip().lower()
        for item in final_items
    ]
    target_groups = [
        extractor.target_indices_for_merged_item(item)
        for item in final_items
    ]
    if not phrases:
        return set()

    phrase_embeddings = extractor.encode_phrases(
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
        indices = np.asarray(target_indices, dtype=int)
        similarities = np.dot(
            extractor.target_embeddings[indices],
            phrase_embedding
        )
        best_index = int(indices[int(np.argmax(similarities))])
        predicted_ids.add(extractor.target_meta[best_index]["id"])
    return predicted_ids


def score_validation_row_for_all_configs(
    row,
    description_embedding,
    combinations,
    extractor
):
    """Embed one CVE once and score every phrase-similarity configuration."""
    empty_results = {
        hyperparameter_run_name(*combination): (0, 1, set())
        for combination in combinations
    }
    tokens = word_tokenize(row[TEXT_COL])[:MAX_TOKENS_PER_ROW]
    if not tokens:
        return empty_results

    candidates = make_ngram_candidates(
        tokens,
        extractor.frequency,
        extractor.train_n
    )
    if not candidates:
        return empty_results

    all_base_similarities = np.dot(
        extractor.target_embeddings,
        description_embedding
    )
    shared_target_indices = np.flatnonzero(
        all_base_similarities >= min(BASE_SIM_THRESHOLD_GRID)
    )
    if not len(shared_target_indices):
        return empty_results

    embedding_cache = {}
    candidate_phrases = [
        candidate["phrase"] for candidate in candidates
    ]
    candidate_embeddings = extractor.encode_phrases(
        candidate_phrases,
        embedding_cache=embedding_cache
    )
    shared_base_similarities = all_base_similarities[
        shared_target_indices
    ].astype(np.float32, copy=False)
    shared_phrase_similarities = np.dot(
        candidate_embeddings,
        extractor.target_embeddings[shared_target_indices].T
    ).astype(np.float32, copy=False)

    selected_items_by_threshold = {}
    for base_threshold, phrase_threshold in itertools.product(
        BASE_SIM_THRESHOLD_GRID,
        PHRASE_SIM_THRESHOLD_GRID
    ):
        selected_items_by_threshold[(
            base_threshold,
            phrase_threshold
        )] = build_selected_items_from_shared_scores(
            extractor=extractor,
            candidates=candidates,
            shared_target_indices=shared_target_indices,
            shared_base_similarities=shared_base_similarities,
            shared_phrase_similarities=shared_phrase_similarities,
            base_threshold=base_threshold,
            phrase_threshold=phrase_threshold
        )

    preload_tuning_phrase_embeddings(
        extractor=extractor,
        selected_items_by_threshold=selected_items_by_threshold,
        tokens=tokens,
        embedding_cache=embedding_cache
    )
    final_items_by_run = {}
    for base_threshold, phrase_threshold, keep_ratio in combinations:
        selected_items = selected_items_by_threshold[(
            base_threshold,
            phrase_threshold
        )]
        trimming_results = extractor.trim_selected_spans(
            selected_items,
            tokens,
            phrase_threshold=phrase_threshold,
            keep_ratio=keep_ratio,
            embedding_cache=embedding_cache
        )
        trimmed_items = [result[0] for result in trimming_results]
        run_name = hyperparameter_run_name(
            base_threshold,
            phrase_threshold,
            keep_ratio
        )
        final_items_by_run[run_name] = extractor.merge_selected_spans(
            trimmed_items,
            tokens
        )

    extractor.encode_phrases(
        [
            item.get("phrase", "").strip().lower()
            for final_items in final_items_by_run.values()
            for item in final_items
        ],
        embedding_cache=embedding_cache
    )

    true_ids = set(true_target_ids(row))
    results = {}
    for run_name, final_items in final_items_by_run.items():
        predicted_ids = predicted_target_ids_for_items(
            extractor,
            final_items,
            embedding_cache
        )
        results[run_name] = (
            int(bool(true_ids & predicted_ids)),
            int(not final_items),
            predicted_ids
        )
    return results


def f1_from_target_counts(run_statistics, prefix=None):
    """Calculate macro/micro F1 exactly as the Occlusion joint tuner."""
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
    for base_threshold, phrase_threshold, keep_ratio in combinations:
        run_name = hyperparameter_run_name(
            base_threshold,
            phrase_threshold,
            keep_ratio
        )
        run_statistics = statistics[run_name]
        hit_count = int(run_statistics["hit_count"])
        hit_rate = hit_count / processed_rows if processed_rows else 0.0
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
            "phrase_sim_threshold": phrase_threshold,
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
            "empty_phrase_similarity_rows": int(
                run_statistics["empty_count"]
            ),
            "output_directory": os.path.join(validation_root, run_name)
        })
    return rows


def evaluate_hyperparameter_grid_single_pass(
    validation,
    combinations,
    tuning_root,
    validation_root,
    extractor
):
    """Evaluate the complete grid during one pass over validation CVEs."""
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
    descriptions = validation[TEXT_COL].fillna("").astype(str).tolist()
    description_embeddings = np.asarray(
        extractor.model.encode(
            descriptions,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True
        ),
        dtype=np.float32
    )
    partial_path = os.path.join(
        tuning_root,
        "validation_summary_partial.csv"
    )

    progress = tqdm(
        validation.iterrows(),
        total=len(validation),
        desc="Joint phrase-similarity tuning"
    )
    for row_position, (_, row) in enumerate(progress, start=1):
        row_results = score_validation_row_for_all_configs(
            row=row,
            description_embedding=description_embeddings[row_position - 1],
            combinations=combinations,
            extractor=extractor
        )
        true_ids = set(true_target_ids(row))
        for run_name, (hit_score, is_empty, predicted_ids) in (
            row_results.items()
        ):
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
            or row_position == len(validation)
        ):
            partial_rows = make_joint_tuning_summary_rows(
                combinations=combinations,
                statistics=statistics,
                processed_rows=row_position,
                total_rows=len(validation),
                validation_root=validation_root
            )
            pd.DataFrame(partial_rows).to_csv(partial_path, index=False)

    summary_rows = make_joint_tuning_summary_rows(
        combinations=combinations,
        statistics=statistics,
        processed_rows=len(validation),
        total_rows=len(validation),
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
            "empty_phrase_similarity_rows": row[
                "empty_phrase_similarity_rows"
            ],
            "base_sim_threshold": row["base_sim_threshold"],
            "phrase_sim_threshold": row["phrase_sim_threshold"],
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


def run_hyperparameter_tuning_single_pass(config):
    """Tune jointly by macro F1, then run only the winning configuration."""
    set_random_seed()
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is visible. This method requires one GPU.")

    tuning_root = config.output_dir
    validation_root = os.path.join(tuning_root, "validation")
    testing_root = os.path.join(tuning_root, "testing")
    os.makedirs(validation_root, exist_ok=True)
    os.makedirs(testing_root, exist_ok=True)

    frame = load_dataset()
    training, held_out = split_dataset(frame)
    validation, _ = split_validation_testing(held_out)
    if validation.empty:
        raise ValueError("Validation split is empty; cannot tune phrase similarity")

    frequency = build_train_word_frequency(training[TEXT_COL].tolist())
    target_texts, target_meta = load_target_knowledge()
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device="cuda:0")
    extractor = PhraseSimilarityExtractor(
        model=model,
        target_texts=target_texts,
        target_meta=target_meta,
        frequency=dict(frequency),
        train_n=len(training),
        base_threshold=min(BASE_SIM_THRESHOLD_GRID),
        phrase_threshold=min(PHRASE_SIM_THRESHOLD_GRID),
        keep_ratio=min(KEYWORD_SIM_KEEP_RATIO_GRID)
    )
    combinations = list(itertools.product(
        BASE_SIM_THRESHOLD_GRID,
        PHRASE_SIM_THRESHOLD_GRID,
        KEYWORD_SIM_KEEP_RATIO_GRID
    ))
    print(
        f"Joint tuning {len(combinations)} configurations over "
        f"{len(validation)} validation rows."
    )
    validation_summary = evaluate_hyperparameter_grid_single_pass(
        validation=validation,
        combinations=combinations,
        tuning_root=tuning_root,
        validation_root=validation_root,
        extractor=extractor
    )

    best = validation_summary.iloc[0]
    best_run_name = str(best["run_name"])
    best_test_dir = os.path.join(testing_root, best_run_name)
    if os.path.isdir(best_test_dir):
        shutil.rmtree(best_test_dir)

    print("Best validation configuration:", best_run_name)
    print("Validation target hit rate:", best["target_hit_rate"])
    print("Validation target hit percent:", best["target_hit_percent"])
    print("Validation macro F1:", best["macro_f1"])
    print("Validation micro F1:", best["micro_f1"])
    print("Running the winning configuration once on all splits.")

    del extractor
    del model
    torch.cuda.empty_cache()
    run_pipeline(replace(
        config,
        base_sim_threshold=float(best["base_sim_threshold"]),
        phrase_sim_threshold=float(best["phrase_sim_threshold"]),
        keyword_sim_keep_ratio=float(best["keyword_sim_keep_ratio"]),
        output_dir=best_test_dir,
        evaluation_split="test",
        tune=False
    ))

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
        "phrase_sim_threshold": float(best["phrase_sim_threshold"]),
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
    best_path = os.path.join(
        method_output_dir,
        "best_hyperparameters.csv"
    )
    best_parameters.to_csv(best_path, index=False)
    print("Saved best hyperparameters:", best_path)
    prune_tuning_run(best_test_dir)


def make_final_file(frame, output_path):
    final = frame.copy()
    final["keyphrases"] = (
        final["phrase_similarity_keywords"].fillna("").astype(str)
    )
    final["keyphrase_source"] = np.where(
        final["keyphrases"].eq(""),
        "empty",
        "phrase_similarity"
    )
    final = final.drop(
        columns=[
            column
            for column in [
                "_row_id",
                "_split_pos",
                "split",
                "phrase_similarity_keywords",
                "phrase_similarity_keyword_details"
            ]
            if column in final.columns
        ]
    )
    final.to_csv(output_path, index=False)
    print("Saved:", output_path)
    return final


def combine_debug_fragments(temp_dir, debug_name):
    fragments = []
    for split_name in ["train", "test"]:
        path = os.path.join(temp_dir, f"_{debug_name}_{split_name}.csv")
        try:
            fragment = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            fragment = pd.DataFrame()
        fragments.append(fragment)
        os.remove(path)

    combined = pd.concat(fragments, ignore_index=True)
    sort_columns = [
        column
        for column in ["row_id", "split", "start"]
        if column in combined.columns
    ]
    if sort_columns:
        combined = combined.sort_values(sort_columns).reset_index(drop=True)
    output_path = os.path.join(temp_dir, f"{debug_name}.csv")
    combined.to_csv(output_path, index=False)
    print("Saved merged debug file:", output_path)
    return combined


def save_inspection_file(split_frames, output_path, config):
    parts = []
    for split_name, frame in split_frames:
        inspection = frame.copy()
        inspection["final_keyphrases"] = (
            inspection["phrase_similarity_keywords"]
            .fillna("")
            .astype(str)
        )
        inspection["keyphrase_source"] = np.where(
            inspection["final_keyphrases"].eq(""),
            "empty",
            "phrase_similarity"
        )
        inspection["true_target_ids"] = inspection.apply(
            lambda row: ";".join(true_target_ids(row)),
            axis=1
        )
        inspection["predicted_target_ids"] = inspection[
            "phrase_similarity_keyword_details"
        ].apply(lambda value: ";".join(extract_target_ids(value)))
        inspection["final_split"] = split_name
        inspection["base_sim_threshold"] = config.base_sim_threshold
        inspection["phrase_sim_threshold"] = config.phrase_sim_threshold
        inspection["keyword_sim_keep_ratio"] = (
            config.keyword_sim_keep_ratio
        )
        inspection["ngram_min"] = NGRAM_MIN
        inspection["ngram_max"] = NGRAM_MAX
        inspection["max_tokens_per_row"] = MAX_TOKENS_PER_ROW
        inspection["top_selected_spans_before_merge"] = (
            TOP_SELECTED_SPANS_BEFORE_MERGE
        )
        inspection["top_final_keywords"] = TOP_FINAL_KEYWORDS
        inspection["max_merged_keyword_words"] = MAX_MERGED_KEYWORD_WORDS
        inspection["min_words_after_trim"] = MIN_WORDS_AFTER_TRIM
        inspection["max_trim_rounds"] = MAX_TRIM_ROUNDS
        inspection["random_seed"] = RANDOM_SEED
        parts.append(inspection)

    inspection = pd.concat(parts, ignore_index=True)
    if "_row_id" in inspection.columns:
        inspection = inspection.sort_values("_row_id").reset_index(drop=True)
    inspection.to_csv(output_path, index=False)
    print("Saved inspection file:", output_path)


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(config):
    set_random_seed()

    print("Python:", sys.executable)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    print("Slurm job:", os.environ.get("SLURM_JOB_ID", "not under Slurm"))

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is visible. This method requires one GPU.")

    output_dir = config.output_dir
    worker_dir = os.path.join(output_dir, "worker")
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(worker_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    frame = load_dataset()
    training, held_out = split_dataset(frame)
    print("Total rows:", len(frame))
    print("Train rows:", len(training))
    print("Held-out rows:", len(held_out))

    training.to_csv(os.path.join(worker_dir, "train_input.csv"), index=False)
    held_out.to_csv(os.path.join(worker_dir, "test_input.csv"), index=False)

    frequency = build_train_word_frequency(training[TEXT_COL].tolist())
    frequency_frame = make_frequency_dataframe(frequency, len(training))
    frequency_path = os.path.join(output_dir, "train_word_frequency.csv")
    frequency_frame.to_csv(frequency_path, index=False)

    target_texts, target_meta = load_target_knowledge()
    print("Target texts:", len(target_texts))

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device="cuda:0")
    extractor = PhraseSimilarityExtractor(
        model=model,
        target_texts=target_texts,
        target_meta=target_meta,
        frequency=dict(frequency),
        train_n=len(training),
        base_threshold=config.base_sim_threshold,
        phrase_threshold=config.phrase_sim_threshold,
        keep_ratio=config.keyword_sim_keep_ratio
    )

    train_output = extractor.process_frame(training, "train", temp_dir)
    held_out_output = extractor.process_frame(held_out, "test", temp_dir)
    train_output.to_csv(
        os.path.join(worker_dir, "output_train.csv"),
        index=False
    )
    held_out_output.to_csv(
        os.path.join(worker_dir, "output_test.csv"),
        index=False
    )

    validation, testing = split_validation_testing(held_out_output)

    total_rows = len(train_output) + len(validation) + len(testing)
    print("Final split rows:", len(train_output), len(validation), len(testing))
    print(
        "Final split percentages:",
        round(len(train_output) / total_rows * 100, 2),
        round(len(validation) / total_rows * 100, 2),
        round(len(testing) / total_rows * 100, 2)
    )

    row_debug = combine_debug_fragments(
        temp_dir,
        "phrase_similarity_row_debug"
    )
    candidate_debug = combine_debug_fragments(
        temp_dir,
        "phrase_similarity_candidate_debug"
    )
    trimming_debug = combine_debug_fragments(
        temp_dir,
        "phrase_similarity_trimming_debug"
    )

    inspection_path = os.path.join(
        output_dir,
        "keyword_extraction_inspection.csv"
    )
    save_inspection_file([
        ("train", train_output),
        ("validation", validation),
        ("test", testing)
    ], inspection_path, config)

    metrics_rows = []
    if config.evaluation_split in {"validation", "both"}:
        metrics_rows.append(evaluate_split(
            validation,
            "validation",
            config
        ))
    if config.evaluation_split in {"test", "both"}:
        metrics_rows.append(evaluate_split(testing, "test", config))
    metrics_path = os.path.join(output_dir, "evaluation_metrics.csv")
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

    make_final_file(
        train_output,
        os.path.join(output_dir, "training_final.csv")
    )
    make_final_file(
        validation,
        os.path.join(output_dir, "validation_final.csv")
    )
    make_final_file(
        testing,
        os.path.join(output_dir, "testing_final.csv")
    )

    config_payload = {
        "method": "direct_phrase_similarity",
        "model_name": MODEL_NAME,
        "random_seed": RANDOM_SEED,
        "dataset_path": DATASET_PATH,
        "cwe_path": CWE_PATH,
        "capec_path": CAPEC_PATH,
        "output_directory": output_dir,
        "evaluation_split": config.evaluation_split,
        "train_size": TRAIN_SIZE,
        "base_sim_threshold": config.base_sim_threshold,
        "phrase_sim_threshold": config.phrase_sim_threshold,
        "keyword_sim_keep_ratio": config.keyword_sim_keep_ratio,
        "base_sim_threshold_grid": BASE_SIM_THRESHOLD_GRID,
        "phrase_sim_threshold_grid": PHRASE_SIM_THRESHOLD_GRID,
        "keyword_sim_keep_ratio_grid": KEYWORD_SIM_KEEP_RATIO_GRID,
        "batch_size": BATCH_SIZE,
        "high_freq_percent_threshold": HIGH_FREQ_PERCENT_THRESHOLD,
        "low_freq_row_threshold": LOW_FREQ_ROW_THRESHOLD,
        "ngram_min": NGRAM_MIN,
        "ngram_max": NGRAM_MAX,
        "max_tokens_per_row": MAX_TOKENS_PER_ROW,
        "top_selected_spans_before_merge": TOP_SELECTED_SPANS_BEFORE_MERGE,
        "top_final_keywords": TOP_FINAL_KEYWORDS,
        "max_merged_keyword_words": MAX_MERGED_KEYWORD_WORDS,
        "min_words_after_trim": MIN_WORDS_AFTER_TRIM,
        "max_trim_rounds": MAX_TRIM_ROUNDS,
        "debug_rows": DEBUG_ROWS,
        "keep_intermediate_files": KEEP_INTERMEDIATE_FILES,
        "tuning_strategy": "single_pass_per_cve",
        "selection_metric": "validation_macro_f1",
        "split_strategy": "CVE-ID-level 70/15/15 with seed 42"
    }
    with open(
        os.path.join(output_dir, "phrase_similarity_config.json"),
        "w",
        encoding="utf-8"
    ) as config_file:
        json.dump(config_payload, config_file, indent=2)

    debug_summary = pd.DataFrame([{
        "dataset_rows": total_rows,
        "train_rows": len(train_output),
        "validation_rows": len(validation),
        "test_rows": len(testing),
        "row_debug_rows": len(row_debug),
        "candidate_debug_rows": len(candidate_debug),
        "trimming_debug_rows": len(trimming_debug),
        "base_sim_threshold": config.base_sim_threshold,
        "phrase_sim_threshold": config.phrase_sim_threshold,
        "keyword_sim_keep_ratio": config.keyword_sim_keep_ratio,
        "random_seed": RANDOM_SEED
    }])
    debug_summary.to_csv(
        os.path.join(temp_dir, "debug_summary.csv"),
        index=False
    )

    if not KEEP_INTERMEDIATE_FILES:
        shutil.rmtree(worker_dir)
        shutil.rmtree(temp_dir)
        print("Removed intermediate worker and temp files.")

    print("Saved evaluation metrics:", metrics_path)
    print("DONE")


def main():
    config = make_run_config()
    if config.tune:
        run_hyperparameter_tuning(config)
    else:
        run_pipeline(config)


if __name__ == "__main__":
    main()
