#!/usr/bin/env python3

"""Generate the top five Stage-03 TextRank keywords for each CVE."""

import os
import random
import re
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
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

OUTPUT_DIR = os.path.join(
    OUTPUTS_ROOT,
    "textrank"
)

# ============================================================
# SETTINGS FROM STAGE 03
# ============================================================

TEXT_COL = "cleaned_description"

RANDOM_SEED = 42
TRAIN_SIZE = 0.70

# Fixed unsupervised baseline. TextRank has no target-scored extractor
# threshold, so it is not label-tuned. Stage-03 evaluation compares its
# phrases directly through per-keyphrase MiniLM Top-1 CAPEC retrieval.
TOP_K = 5
WINDOW_SIZE = 4
MAX_PHRASE_WORDS = 6
MIN_WORD_LEN = 2

STOPWORDS = set(ENGLISH_STOP_WORDS)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# STAGE-03 TEXTRANK LOGIC
# ============================================================

def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value)


def collapse_consecutive_duplicate_tokens(text):
    """Remove adjacent duplicate tokens before keyword extraction."""
    final_tokens = []

    for token in safe_text(text).split():
        if final_tokens and token.casefold() == final_tokens[-1].casefold():
            continue
        final_tokens.append(token)

    return " ".join(final_tokens)


def tokenize(text):
    """Keep useful words and cybersecurity-style compound tokens."""
    tokens = re.findall(
        r"[a-z][a-z0-9_+\-/]*",
        collapse_consecutive_duplicate_tokens(text).lower()
    )

    cleaned = []

    for token in tokens:
        token = token.strip("-_/+")

        if len(token) < MIN_WORD_LEN:
            continue
        if token in STOPWORDS:
            continue
        if token.isdigit():
            continue

        cleaned.append(token)

    return cleaned


def split_into_candidate_phrases(text):
    """Split phrases at stopwords and punctuation."""
    prepared = re.sub(
        r"[^a-z0-9_+\-/]+",
        " ",
        collapse_consecutive_duplicate_tokens(text).lower()
    )

    phrases = []
    current = []

    for raw_token in prepared.split():
        token = raw_token.strip("-_/+")

        if not token:
            continue

        if (
            token in STOPWORDS
            or token.isdigit()
            or len(token) < MIN_WORD_LEN
        ):
            if current:
                phrases.append(current)
                current = []
            continue

        current.append(token)

        if len(current) >= MAX_PHRASE_WORDS:
            phrases.append(current)
            current = []

    if current:
        phrases.append(current)

    return phrases


def remove_duplicate_overlap(keywords, top_k=TOP_K):
    """Keep stronger phrases while removing weaker contained phrases."""
    selected = []

    for keyword in keywords:
        keyword = keyword.strip()

        if not keyword:
            continue

        if any(
            keyword == chosen
            or (
                keyword in chosen
                and len(keyword.split()) <= len(chosen.split())
            )
            for chosen in selected
        ):
            continue

        selected.append(keyword)

        if len(selected) >= top_k:
            break

    return selected


def textrank_keywords(
    text,
    top_k=TOP_K,
    window_size=WINDOW_SIZE
):
    """Return semicolon-separated Stage-03 TextRank keyphrases."""
    tokens = tokenize(text)

    if not tokens:
        return ""

    if len(tokens) == 1:
        return tokens[0]

    graph = nx.Graph()
    graph.add_nodes_from(tokens)

    for index in range(len(tokens)):
        window = tokens[index:index + window_size]

        for left in range(len(window)):
            for right in range(left + 1, len(window)):
                first = window[left]
                second = window[right]

                if first == second:
                    continue

                if graph.has_edge(first, second):
                    graph[first][second]["weight"] += 1.0
                else:
                    graph.add_edge(first, second, weight=1.0)

    if graph.number_of_edges() == 0:
        counts = Counter(tokens)
        return "; ".join(
            word
            for word, _ in counts.most_common(top_k)
        )

    scores = nx.pagerank(graph, weight="weight")
    phrase_scores = {}

    for phrase_words in split_into_candidate_phrases(text):
        valid_words = [
            word
            for word in phrase_words
            if word in scores
        ]

        if not valid_words:
            continue

        phrase = " ".join(valid_words).strip()
        score = sum(scores[word] for word in valid_words)
        score *= 1.0 + 0.15 * (len(valid_words) - 1)

        phrase_scores[phrase] = max(
            score,
            phrase_scores.get(phrase, float("-inf"))
        )

    ranked_keywords = [
        phrase
        for phrase, _ in sorted(
            phrase_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )
    ]

    final_keywords = remove_duplicate_overlap(
        ranked_keywords,
        top_k=top_k
    )

    return "; ".join(final_keywords)


def split_dataset(df):
    """Create the shared deterministic 70/15/15 CVE-level split."""
    if "cve_id" in df.columns:
        unique_cves = df["cve_id"].dropna().astype(str).unique()
        training_cves, held_out_cves = train_test_split(
            unique_cves,
            train_size=TRAIN_SIZE,
            random_state=RANDOM_SEED,
            shuffle=True
        )
        training = df[
            df["cve_id"].astype(str).isin(training_cves)
        ].copy()
        held_out = df[
            df["cve_id"].astype(str).isin(held_out_cves)
        ].copy()

        validation_cves, testing_cves = train_test_split(
            held_out["cve_id"].dropna().astype(str).unique(),
            test_size=0.5,
            random_state=RANDOM_SEED,
            shuffle=True
        )
        held_out_ids = held_out["cve_id"].astype(str)
        validation = held_out[
            held_out_ids.isin(validation_cves)
        ].copy()
        testing = held_out[
            held_out_ids.isin(testing_cves)
        ].copy()
    else:
        training, held_out = train_test_split(
            df,
            train_size=TRAIN_SIZE,
            random_state=RANDOM_SEED,
            shuffle=True
        )
        validation, testing = train_test_split(
            held_out,
            test_size=0.5,
            random_state=RANDOM_SEED,
            shuffle=True
        )

    return tuple(
        split.sort_values("_row_id").reset_index(drop=True)
        for split in (training, validation, testing)
    )


def make_final_file(df, output_path):
    final = df.copy()
    final["keyphrases"] = (
        final["textrank_keywords"].fillna("").astype(str)
    )
    final["keyphrase_source"] = np.where(
        final["keyphrases"].str.strip().eq(""),
        "empty",
        "textrank"
    )
    final = final.drop(
        columns=["textrank_keywords", "_row_id"],
        errors="ignore"
    )
    final.to_csv(output_path, index=False)
    print("Saved:", output_path)
    return final


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset file not found:\n{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)
    df.columns = df.columns.str.strip()

    if TEXT_COL not in df.columns:
        raise ValueError(
            f"Column '{TEXT_COL}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    df[TEXT_COL] = (
        df[TEXT_COL]
        .fillna("")
        .astype(str)
        .map(collapse_consecutive_duplicate_tokens)
    )
    df["_row_id"] = np.arange(len(df))

    tqdm.pandas()
    df["textrank_keywords"] = df[TEXT_COL].progress_apply(
        lambda text: textrank_keywords(
            text,
            top_k=TOP_K,
            window_size=WINDOW_SIZE
        )
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    training, validation, testing = split_dataset(df)
    final_frames = {
        "training": make_final_file(
            training,
            os.path.join(OUTPUT_DIR, "training_final.csv")
        ),
        "validation": make_final_file(
            validation,
            os.path.join(OUTPUT_DIR, "validation_final.csv")
        ),
        "testing": make_final_file(
            testing,
            os.path.join(OUTPUT_DIR, "testing_final.csv")
        )
    }

    print("Split sizes:", {
        name: len(frame) for name, frame in final_frames.items()
    })
    print("\nSample:")
    print(final_frames["training"][[TEXT_COL, "keyphrases"]].head(5))


if __name__ == "__main__":
    main()
