#!/usr/bin/env python3

"""Compare keyword extraction methods using MiniLM cosine retrieval.

For each CVE row:
1. Parse the method's extracted keyphrases.
2. Encode keyphrases and CAPEC Name + Description with MiniLM.
3. Compare the normalized embeddings using cosine similarity.
4. Give each CAPEC the MAX cosine similarity obtained from any keyphrase.
5. Rank CAPECs by that score.
6. Use Top-5 CAPECs as the predicted multilabel set for classification metrics.
7. Compute multilabel metrics, exact match, Hit@1, and Hit@5.

MiniLM is used only as a fixed text encoder; there is no trainable classifier,
training, or fine-tuning. The CVE description is never added to the query:
only ``keyphrases`` is used to retrieve CAPECs.
"""

import os

# Keep CPU use explicit and consistent in interactive and Slurm runs.
DEFAULT_CPU_THREADS = "16"
CPU_THREAD_SETTING = os.environ.get(
    "KEYWORD_EVAL_CPU_THREADS",
    DEFAULT_CPU_THREADS,
)
for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
):
    os.environ[variable] = CPU_THREAD_SETTING

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
METHODOLOGY_DIR = PROJECT_DIR.parent
CAPEC_CATALOG_PATH = (
    METHODOLOGY_DIR
    / "01_cwe_capec_ground_truth"
    / "3000_capec.csv"
)

BATCH_OUTPUT_ROOT = os.environ.get("KEYWORD_MAKER_OUTPUT_ROOT")
if BATCH_OUTPUT_ROOT:
    METHOD_OUTPUTS_DIR = Path(BATCH_OUTPUT_ROOT).expanduser().resolve()
else:
    METHOD_OUTPUTS_DIR = PROJECT_DIR / "outputs"

EVALUATION_OUTPUT_DIR = (
    METHOD_OUTPUTS_DIR / "evaluation" / "minilm_cosine_top5"
)


# ============================================================
# SETTINGS
# ============================================================

METHOD_NAMES = [
    "occlusion",
    "phrase_similarity",
    "keybert_modified",
    "textrank",
    "qwen_keywords",
]

FINAL_FILENAMES = {
    "training": "training_final.csv",
    "validation": "validation_final.csv",
    "testing": "testing_final.csv",
}

KEYPHRASE_COL = "keyphrases"
LABEL_COL = "capec_id"
ID_COL = "cve_id"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 256
CPU_THREADS = int(CPU_THREAD_SETTING)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Top-5 is the prediction set used for F1 / exact match / Jaccard / Hamming.
PREDICTION_K = 5

# The complete stored ranking is exactly Top-5.
RANKING_K = 5
HIT_KS = (1, 5)

EVALUATION_CONTRACT_VERSION = 6

torch.set_num_threads(CPU_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ============================================================
# PARSING
# ============================================================

def parse_capec_cell(value):
    """Parse CAPEC IDs from common CSV representations."""
    if pd.isna(value):
        return []

    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null", "[]"}:
        return []

    normalized = text.lower().replace("_", "").replace(" ", "")
    if normalized in {"capec-noid", "noid", "nocapec", "capecnone"}:
        return []

    labels = []
    try:
        parsed = ast.literal_eval(text)
        raw_items = (
            list(parsed)
            if isinstance(parsed, (list, tuple, set))
            else [parsed]
        )

        for item in raw_items:
            item = str(item).strip()
            item_normalized = item.lower().replace("_", "").replace(" ", "")
            if item_normalized in {
                "capec-noid",
                "noid",
                "nocapec",
                "capecnone",
            }:
                continue

            labels.extend(
                match.upper()
                for match in re.findall(
                    r"CAPEC-\d+",
                    item,
                    flags=re.IGNORECASE,
                )
            )
            if re.fullmatch(r"\d+", item):
                labels.append(f"CAPEC-{int(item)}")

        if labels:
            return sorted(set(labels), key=capec_sort_key)
    except (ValueError, SyntaxError):
        pass

    labels.extend(
        match.upper()
        for match in re.findall(
            r"CAPEC-\d+",
            text,
            flags=re.IGNORECASE,
        )
    )
    if not labels:
        labels.extend(
            f"CAPEC-{int(number)}"
            for number in re.findall(r"\b\d+\b", text)
        )

    return sorted(set(labels), key=capec_sort_key)


def is_empty_keyword(value):
    if pd.isna(value):
        return True
    return str(value).strip().lower() in {
        "",
        "nan",
        "none",
        "null",
        "[]",
        "{}",
    }


def parse_keyphrase_cell(value):
    """Return unique keyphrases from list-like or delimited CSV cells."""
    if is_empty_keyword(value):
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
        raw_items = re.split(r"[;|,\n]+", text)

    keyphrases = []
    seen = set()
    for item in raw_items:
        if is_empty_keyword(item):
            continue

        keyphrase = re.sub(r"\s+", " ", str(item)).strip(
            " \t\r\n\"'[]{}()"
        )
        normalized = keyphrase.casefold()
        if keyphrase and normalized not in seen:
            keyphrases.append(keyphrase)
            seen.add(normalized)

    return keyphrases


# ============================================================
# METHOD DISCOVERY AND INPUT VALIDATION
# ============================================================

def discover_method_directories():
    if not METHOD_OUTPUTS_DIR.is_dir():
        raise FileNotFoundError(
            f"Method output directory not found: {METHOD_OUTPUTS_DIR}"
        )

    complete = []
    incomplete = []

    for method_name in METHOD_NAMES:
        directory = METHOD_OUTPUTS_DIR / method_name
        missing = [
            directory / filename
            for filename in FINAL_FILENAMES.values()
            if not (directory / filename).is_file()
        ]

        if missing:
            incomplete.append((method_name, missing))
        else:
            complete.append(directory)

    if incomplete:
        details = "; ".join(
            f"{method_name}: " + ", ".join(str(path) for path in missing)
            for method_name, missing in incomplete
        )
        raise FileNotFoundError(
            "All five keyword methods are required for comparison. "
            f"Missing inputs: {details}"
        )

    return complete


def load_method_splits(method_dir):
    frames = {}

    for split_name, filename in FINAL_FILENAMES.items():
        split_path = method_dir / filename
        print(f"Input path [{method_dir.name}/{split_name}]: {split_path}")

        frame = pd.read_csv(split_path)
        frame.columns = frame.columns.str.strip()

        missing = {KEYPHRASE_COL, LABEL_COL, ID_COL} - set(frame.columns)
        if missing:
            raise ValueError(
                f"{method_dir.name}/{filename} is missing columns: "
                f"{sorted(missing)}"
            )

        frame["keyphrase_list"] = frame[KEYPHRASE_COL].apply(
            parse_keyphrase_cell
        )
        frame["capec_real_list"] = frame[LABEL_COL].apply(
            parse_capec_cell
        )

        frames[split_name] = frame

    return frames


def split_id_signature(frame):
    return tuple(frame[ID_COL].fillna("<missing>").astype(str).tolist())


def validate_shared_splits(method_frames):
    """Ensure all methods are evaluated on identical CVE rows and labels."""
    baseline_name = next(iter(method_frames))
    baseline = method_frames[baseline_name]

    for method_name, frames in method_frames.items():
        split_id_sets = {}

        for split_name in FINAL_FILENAMES:
            ids = frames[split_name][ID_COL]

            if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
                raise ValueError(
                    f"{method_name}/{split_name} contains missing {ID_COL}."
                )

            if ids.astype(str).duplicated().any():
                raise ValueError(
                    f"{method_name}/{split_name} contains duplicate {ID_COL}."
                )

            expected = split_id_signature(baseline[split_name])
            actual = split_id_signature(frames[split_name])
            if actual != expected:
                raise ValueError(
                    f"Split mismatch: {method_name}/{split_name} does not "
                    f"have the same ordered {ID_COL} rows as "
                    f"{baseline_name}/{split_name}."
                )

            expected_truth = baseline[split_name]["capec_real_list"].tolist()
            actual_truth = frames[split_name]["capec_real_list"].tolist()
            if actual_truth != expected_truth:
                raise ValueError(
                    f"Ground-truth mismatch: {method_name}/{split_name} "
                    f"does not match {baseline_name}/{split_name}."
                )

            split_id_sets[split_name] = set(actual)

        for left, right in (
            ("training", "validation"),
            ("training", "testing"),
            ("validation", "testing"),
        ):
            overlap = split_id_sets[left] & split_id_sets[right]
            if overlap:
                raise ValueError(
                    f"{method_name} has {len(overlap)} {ID_COL} values in "
                    f"both {left} and {right}."
                )


# ============================================================
# CAPEC REFERENCES AND COSINE RETRIEVAL
# ============================================================

def resolve_catalog_column(catalog, expected_name):
    matches = [
        column
        for column in catalog.columns
        if str(column).strip().casefold() == expected_name.casefold()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one '{expected_name}' column in "
            f"{CAPEC_CATALOG_PATH}; found {matches}."
        )
    return matches[0]


def canonical_capec_id(value):
    text = "" if pd.isna(value) else str(value).strip()
    match = re.fullmatch(r"(?:CAPEC-)?(\d+)", text, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Invalid CAPEC ID in catalogue: {text!r}")
    return f"CAPEC-{int(match.group(1))}"


def clean_catalog_text(value):
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def capec_sort_key(label):
    return int(label.split("-")[1])


def load_capec_references():
    if not CAPEC_CATALOG_PATH.is_file():
        raise FileNotFoundError(
            f"CAPEC catalogue not found: {CAPEC_CATALOG_PATH}"
        )

    catalog = pd.read_csv(CAPEC_CATALOG_PATH, dtype=str)

    id_column = resolve_catalog_column(catalog, "ID")
    name_column = resolve_catalog_column(catalog, "Name")
    description_column = resolve_catalog_column(catalog, "Description")

    references = pd.DataFrame({
        "capec_id": catalog[id_column].map(canonical_capec_id),
        "capec_name": catalog[name_column].map(clean_catalog_text),
        "capec_description": catalog[description_column].map(clean_catalog_text),
    })

    if references["capec_id"].duplicated().any():
        duplicates = references.loc[
            references["capec_id"].duplicated(keep=False),
            "capec_id",
        ].tolist()
        raise ValueError(f"Duplicate CAPEC catalogue IDs: {duplicates[:10]}")

    references["reference_text"] = [
        " ".join(part for part in (name, description) if part)
        for name, description in zip(
            references["capec_name"],
            references["capec_description"],
        )
    ]

    empty = references["reference_text"].eq("")
    if empty.any():
        raise ValueError(
            "CAPEC entries with neither name nor description: "
            f"{references.loc[empty, 'capec_id'].tolist()[:10]}"
        )

    references["numeric_id"] = (
        references["capec_id"].str.split("-").str[-1].astype(int)
    )
    references = (
        references
        .sort_values("numeric_id")
        .drop(columns="numeric_id")
        .reset_index(drop=True)
    )

    return references


def encode_capec_references(embedder, references):
    """Encode CAPEC Name + Description once with normalized MiniLM vectors."""
    print(f"Encoding {len(references):,} CAPEC reference texts once")
    return embedder.encode(
        references["reference_text"].tolist(),
        batch_size=EMBED_BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)


def collect_unique_keyphrases(method_frames):
    unique = []
    seen = set()
    total_occurrences = 0

    for frames in method_frames.values():
        for frame in frames.values():
            for keyphrases in frame["keyphrase_list"]:
                total_occurrences += len(keyphrases)
                for keyphrase in keyphrases:
                    if keyphrase not in seen:
                        seen.add(keyphrase)
                        unique.append(keyphrase)

    print(
        f"Collected {len(unique):,} unique keyphrases from "
        f"{total_occurrences:,} occurrences"
    )
    return unique


def retrieve_keyphrase_topk(
    embedder,
    keyphrases,
    capec_embeddings,
    capec_ids,
    top_k=RANKING_K,
):
    """Store MiniLM cosine matches for each unique keyphrase.

    This is sufficient for exact CVE-level Top-K ranking when CVE aggregation
    uses MAX across keyphrases: a CAPEC outside a phrase's Top-K cannot become
    global Top-K through that phrase's max score.
    """
    retrieval = {}
    n_capecs = len(capec_ids)
    effective_k = min(top_k, n_capecs)

    for start in range(0, len(keyphrases), EMBED_BATCH_SIZE):
        end = min(start + EMBED_BATCH_SIZE, len(keyphrases))
        batch_phrases = keyphrases[start:end]

        phrase_embeddings = embedder.encode(
            batch_phrases,
            batch_size=EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)

        # Both sides are L2-normalized, so matrix multiplication is cosine
        # similarity. MiniLM is fixed and no classifier is trained.
        similarities = phrase_embeddings @ capec_embeddings.T

        # References are sorted by numeric CAPEC ID. Stable sorting therefore
        # gives deterministic ordering when cosine scores tie.
        top_indices = np.argsort(
            -similarities,
            axis=1,
            kind="stable",
        )[:, :effective_k]

        top_scores = np.take_along_axis(similarities, top_indices, axis=1)

        for row_index, phrase in enumerate(batch_phrases):
            retrieval[phrase] = [
                {
                    "capec_id": capec_ids[int(capec_index)],
                    "cosine_similarity": float(score),
                }
                for capec_index, score in zip(
                    top_indices[row_index],
                    top_scores[row_index],
                )
            ]

        print(f"Retrieved keyphrases: {end:,}/{len(keyphrases):,}")

    return retrieval


def aggregate_cve_ranking(keyphrases, retrieval, ranking_k=RANKING_K):
    """Rank CAPECs for one CVE using max cosine across its keyphrases."""
    if not keyphrases:
        return []

    best_scores = {}
    best_phrases = {}

    for keyphrase in keyphrases:
        for item in retrieval[keyphrase]:
            capec_id = item["capec_id"]
            score = item["cosine_similarity"]

            if capec_id not in best_scores or score > best_scores[capec_id]:
                best_scores[capec_id] = score
                best_phrases[capec_id] = keyphrase

    ranked = sorted(
        best_scores,
        key=lambda capec_id: (
            -best_scores[capec_id],
            capec_sort_key(capec_id),
        ),
    )[:ranking_k]

    return [
        {
            "rank": rank,
            "capec_id": capec_id,
            "cosine_similarity": float(best_scores[capec_id]),
            "best_keyphrase": best_phrases[capec_id],
        }
        for rank, capec_id in enumerate(ranked, start=1)
    ]


def validate_true_labels(method_name, frames, catalog_ids):
    unknown = sorted({
        label
        for frame in frames.values()
        for labels in frame["capec_real_list"]
        for label in labels
        if label not in catalog_ids
    })

    if unknown:
        raise ValueError(
            f"{method_name} contains CAPEC labels absent from "
            f"{CAPEC_CATALOG_PATH}: {unknown}"
        )


# ============================================================
# METRICS
# ============================================================

def safe_metric(metric_fn, y_true, y_pred, average):
    """Call a sklearn multilabel metric with zero_division=0 where supported."""
    return float(
        metric_fn(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        )
    )


def calculate_metrics(true_sets, predicted_top5, ranked_top5, all_classes):
    """Calculate all requested multilabel and ranking metrics."""
    mlb = MultiLabelBinarizer(classes=all_classes)
    mlb.fit([all_classes])

    y_true = mlb.transform(true_sets)
    y_pred = mlb.transform(predicted_top5)

    row_count = len(true_sets)
    exact_match_count = sum(
        set(true_labels) == set(pred_labels)
        for true_labels, pred_labels in zip(true_sets, predicted_top5)
    )

    metrics = {
        "rows": int(row_count),
        "rows_with_true_labels": int(sum(bool(labels) for labels in true_sets)),

        # Micro
        "micro_precision": safe_metric(
            precision_score, y_true, y_pred, "micro"
        ),
        "micro_recall": safe_metric(
            recall_score, y_true, y_pred, "micro"
        ),
        "micro_f1": safe_metric(
            f1_score, y_true, y_pred, "micro"
        ),

        # Macro
        "macro_precision": safe_metric(
            precision_score, y_true, y_pred, "macro"
        ),
        "macro_recall": safe_metric(
            recall_score, y_true, y_pred, "macro"
        ),
        "macro_f1": safe_metric(
            f1_score, y_true, y_pred, "macro"
        ),

        # Weighted
        "weighted_precision": safe_metric(
            precision_score, y_true, y_pred, "weighted"
        ),
        "weighted_recall": safe_metric(
            recall_score, y_true, y_pred, "weighted"
        ),
        "weighted_f1": safe_metric(
            f1_score, y_true, y_pred, "weighted"
        ),

        # Per-sample
        "sample_precision": safe_metric(
            precision_score, y_true, y_pred, "samples"
        ),
        "sample_recall": safe_metric(
            recall_score, y_true, y_pred, "samples"
        ),
        "sample_f1": safe_metric(
            f1_score, y_true, y_pred, "samples"
        ),
        "sample_jaccard": float(
            jaccard_score(
                y_true,
                y_pred,
                average="samples",
                zero_division=0,
            )
        ),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),

        # Exact set equality of truth vs Top-5 prediction set.
        "exact_match": (
            float(exact_match_count / row_count) if row_count else 0.0
        ),
        "exact_match_count": int(exact_match_count),
    }

    # Hit@K: at least one true CAPEC appears in ranked Top-K.
    # Rows with no ground-truth CAPEC are excluded from the Hit@K denominator.
    nonempty_truth_indices = [
        i for i, labels in enumerate(true_sets) if labels
    ]

    for k in HIT_KS:
        hit_count = 0
        for i in nonempty_truth_indices:
            true_set = set(true_sets[i])
            ranked_ids = {
                item["capec_id"] for item in ranked_top5[i][:k]
            }
            hit_count += int(bool(true_set & ranked_ids))

        denominator = len(nonempty_truth_indices)
        metrics[f"hit_at_{k}"] = (
            float(hit_count / denominator) if denominator else 0.0
        )
        metrics[f"hit_at_{k}_count"] = int(hit_count)

    return metrics


# ============================================================
# SPLIT / METHOD EVALUATION
# ============================================================

def evaluate_split(frame, retrieval, all_classes):
    true_sets = frame["capec_real_list"].tolist()

    ranked_top5 = []
    predicted_top5 = []

    row_true_positive = []
    row_false_positive = []
    row_false_negative = []
    row_exact_match = []
    row_hit_1 = []
    row_hit_5 = []

    for keyphrases, true_labels in zip(
        frame["keyphrase_list"],
        true_sets,
    ):
        ranking = aggregate_cve_ranking(
            keyphrases,
            retrieval,
            ranking_k=RANKING_K,
        )
        top5 = [item["capec_id"] for item in ranking[:PREDICTION_K]]

        ranked_top5.append(ranking)
        predicted_top5.append(top5)

        true_set = set(true_labels)
        pred_set = set(top5)

        row_true_positive.append(len(true_set & pred_set))
        row_false_positive.append(len(pred_set - true_set))
        row_false_negative.append(len(true_set - pred_set))
        row_exact_match.append(true_set == pred_set)

        ranked_ids = [item["capec_id"] for item in ranking]
        row_hit_1.append(bool(true_set & set(ranked_ids[:1])) if true_set else False)
        row_hit_5.append(bool(true_set & set(ranked_ids[:5])) if true_set else False)

    metrics = calculate_metrics(
        true_sets,
        predicted_top5,
        ranked_top5,
        all_classes,
    )

    keyword_counts = frame["keyphrase_list"].apply(len)
    metrics.update({
        "rows_with_keywords": int((keyword_counts > 0).sum()),
        "rows_without_keywords": int((keyword_counts == 0).sum()),
        "keyword_coverage": float((keyword_counts > 0).mean()),
        "keyphrase_occurrences": int(keyword_counts.sum()),
        "mean_keyphrases_per_row": float(keyword_counts.mean()),
        "mean_predicted_capecs_per_row": float(
            np.mean([len(labels) for labels in predicted_top5])
        ),
    })

    output = frame.drop(
        columns=["keyphrase_list", "capec_real_list"],
        errors="ignore",
    ).copy()

    output["parsed_keyphrases"] = [
        json.dumps(keyphrases, ensure_ascii=False)
        for keyphrases in frame["keyphrase_list"]
    ]
    output["true_capec_labels"] = [
        json.dumps(labels) for labels in true_sets
    ]
    output["ranked_top5_capecs"] = [
        json.dumps(ranking, ensure_ascii=False)
        for ranking in ranked_top5
    ]
    output["predicted_top5_capec_labels"] = [
        json.dumps(labels) for labels in predicted_top5
    ]
    output["top1_capec"] = [
        ranking[0]["capec_id"] if ranking else ""
        for ranking in ranked_top5
    ]
    output["top5_capecs"] = [
        json.dumps([item["capec_id"] for item in ranking[:5]])
        for ranking in ranked_top5
    ]

    output["prediction_count"] = [len(labels) for labels in predicted_top5]
    output["true_positive_count"] = row_true_positive
    output["false_positive_count"] = row_false_positive
    output["false_negative_count"] = row_false_negative
    output["exact_match"] = row_exact_match
    output["hit_at_1"] = row_hit_1
    output["hit_at_5"] = row_hit_5

    return metrics, output


def evaluate_method(method_name, frames, retrieval, all_classes):
    method_output_dir = EVALUATION_OUTPUT_DIR / method_name
    method_output_dir.mkdir(parents=True, exist_ok=True)

    split_metrics = {}

    for split_name, frame in frames.items():
        metrics, predictions = evaluate_split(
            frame,
            retrieval,
            all_classes,
        )
        split_metrics[split_name] = metrics

        predictions.to_csv(
            method_output_dir / f"{split_name}_predictions.csv",
            index=False,
        )

        print(
            f"{method_name}/{split_name}: "
            f"Micro-F1={metrics['micro_f1']:.4f} "
            f"Macro-F1={metrics['macro_f1']:.4f} "
            f"Weighted-F1={metrics['weighted_f1']:.4f} "
            f"Sample-F1={metrics['sample_f1']:.4f} "
            f"Exact={metrics['exact_match']:.4f} "
            f"Hit@1={metrics['hit_at_1']:.4f} "
            f"Hit@5={metrics['hit_at_5']:.4f}"
        )

    comparison_row = {"method": method_name}
    for split_name, metrics in split_metrics.items():
        comparison_row.update({
            f"{split_name}_{key}": value
            for key, value in metrics.items()
        })

    payload = {
        "completed": True,
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "method": method_name,
        "evaluation_type": "MiniLM cosine CAPEC Top-5 retrieval",
        "embedding_model": EMBED_MODEL_NAME,
        "trainable_classifier_used": False,
        "capec_reference": "CAPEC Name + space + CAPEC Description",
        "cve_score_aggregation": "maximum cosine over that CVE's keyphrases",
        "prediction_rule": "Top-5 ranked CAPECs form predicted multilabel set",
        "ranking_rule": "Top-5 retained for Hit@1/5",
        "empty_keyword_rule": "empty prediction/ranking",
        "hit_denominator_rule": "rows with at least one true CAPEC label",
        "split_metrics": split_metrics,
        "comparison_row": comparison_row,
    }

    (method_output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame([
        {"method": method_name, "split": split_name, **metrics}
        for split_name, metrics in split_metrics.items()
    ]).to_csv(method_output_dir / "metrics.csv", index=False)

    return comparison_row


# ============================================================
# MAIN
# ============================================================

def main():
    print("Evaluator: fixed MiniLM encoder + cosine Top-5")
    print("Device:", DEVICE)
    print("CPU threads:", CPU_THREADS)
    print("Method outputs:", METHOD_OUTPUTS_DIR)
    print("Evaluation outputs:", EVALUATION_OUTPUT_DIR)
    print("CAPEC catalogue:", CAPEC_CATALOG_PATH)
    print("Prediction K:", PREDICTION_K)
    print("Ranking K:", RANKING_K)
    print("Aggregation: max cosine across CVE keyphrases")

    method_directories = discover_method_directories()
    method_frames = {
        directory.name: load_method_splits(directory)
        for directory in method_directories
    }
    validate_shared_splits(method_frames)

    EVALUATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    references = load_capec_references()
    capec_ids = references["capec_id"].tolist()
    catalog_ids = set(capec_ids)

    for method_name, frames in method_frames.items():
        validate_true_labels(method_name, frames, catalog_ids)

    embedder = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
    capec_embeddings = encode_capec_references(embedder, references)

    references.to_csv(
        EVALUATION_OUTPUT_DIR / "capec_references.csv",
        index=False,
    )
    np.save(
        EVALUATION_OUTPUT_DIR / "capec_reference_embeddings.npy",
        capec_embeddings,
    )

    unique_keyphrases = collect_unique_keyphrases(method_frames)
    retrieval = retrieve_keyphrase_topk(
        embedder,
        unique_keyphrases,
        capec_embeddings,
        capec_ids,
        top_k=RANKING_K,
    )

    comparison_rows = []
    comparison_path = EVALUATION_OUTPUT_DIR / "comparison_summary.csv"

    for method_name, frames in method_frames.items():
        print(f"\nEvaluating method: {method_name}")
        comparison_rows.append(
            evaluate_method(
                method_name,
                frames,
                retrieval,
                capec_ids,
            )
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        [
            "testing_micro_f1",
            "testing_hit_at_5",
            "method",
        ],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)

    comparison.insert(
        0,
        "rank",
        np.arange(1, len(comparison) + 1),
    )
    comparison.insert(
        1,
        "best_testing_micro_f1",
        comparison["testing_micro_f1"].eq(
            comparison["testing_micro_f1"].max()
        ),
    )

    comparison.to_csv(comparison_path, index=False)

    config = {
        "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
        "selected_method_names": METHOD_NAMES,
        "evaluated_method_names": list(method_frames),
        "method_outputs_directory": str(METHOD_OUTPUTS_DIR),
        "evaluation_output_directory": str(EVALUATION_OUTPUT_DIR),
        "capec_catalog_path": str(CAPEC_CATALOG_PATH),
        "capec_reference_text": "Name + space + Description",
        "keyphrase_column": KEYPHRASE_COL,
        "query_input": "parsed keyphrases only; CVE description is not used",
        "similarity": "cosine similarity over normalized MiniLM embeddings",
        "embedding_model": EMBED_MODEL_NAME,
        "cve_capec_score": "max cosine similarity over all CVE keyphrases",
        "prediction_k": PREDICTION_K,
        "ranking_k": RANKING_K,
        "prediction_set": "Top-5 CAPEC IDs",
        "hit_metrics": list(HIT_KS),
        "hit_denominator": "only rows with non-empty ground-truth CAPEC labels",
        "empty_keyword_rule": "empty predicted CAPEC set and empty ranking",
        "metrics": [
            "micro precision",
            "micro recall",
            "micro F1",
            "macro precision",
            "macro recall",
            "macro F1",
            "weighted precision",
            "weighted recall",
            "weighted F1",
            "sample precision",
            "sample recall",
            "sample F1",
            "sample Jaccard",
            "Hamming loss",
            "exact match",
            "Hit@1",
            "Hit@5",
        ],
        "trainable_classifier_used": False,
        "embedding_batch_size": EMBED_BATCH_SIZE,
        "device": DEVICE,
        "cpu_threads": CPU_THREADS,
    }

    (EVALUATION_OUTPUT_DIR / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    display_columns = [
        "rank",
        "method",
        "testing_micro_f1",
        "testing_macro_f1",
        "testing_weighted_f1",
        "testing_sample_f1",
        "testing_exact_match",
        "testing_hit_at_1",
        "testing_hit_at_5",
        "testing_keyword_coverage",
    ]

    print("\nComparison complete:", comparison_path)
    print(comparison[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
