#!/usr/bin/env python3

"""Extract base-Qwen keywords and create shared final evaluation splits."""

import json
import os
import random
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


# ==========================================================
# SETTINGS
# ==========================================================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# If the model is already downloaded elsewhere, set MODEL_NAME to that path.

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PAPER_DIR = PROJECT_DIR.parents[1]
OUTPUTS_ROOT = Path(os.environ.get(
    "KEYWORD_MAKER_OUTPUT_ROOT",
    PROJECT_DIR / "outputs",
)).expanduser().resolve()

DATASET_CSV = (
    PAPER_DIR
    / "methodology"
    / "02_preprocessing"
    / "dataset.csv"
)
OUTPUT_DIR = OUTPUTS_ROOT / "qwen_keywords"
CHECKPOINT_CSV = OUTPUT_DIR / "generation_checkpoint.csv"
GENERATION_CACHE_CSV = OUTPUT_DIR / "dataset_qwen_keywords.csv"

TEXT_COLUMN = "uncleaned_description"
GENERATED_KEYWORD_COLUMN = "qwen_keywords_uncleaned"
PROCESSED_COLUMN = "_qwen_processed"
KEYPHRASE_SOURCE = "qwen_keywords"

MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS = 70
BATCH_SIZE = 4
SAVE_EVERY_BATCHES = 20
RANDOM_SEED = 42
TRAIN_SIZE = 0.70

# Fixed generative baseline. Batch size affects throughput, not predictions,
# and the prompt/decoding policy is kept fixed rather than selected using
# target labels. The downstream NN tunes its threshold on validation data.

# Reuse the completed Qwen generation as a validated Stage-03 cache. It is used
# only when its ordered CVE IDs and source descriptions exactly match dataset.csv.
REUSE_VALIDATED_GENERATION_CACHE = True

# The checkpoint is retained after an interruption and removed after all three
# final split files have been written successfully.
KEEP_COMPLETED_CHECKPOINT = False

# None processes the full dataset. Use an integer only for a local smoke test;
# smoke-test split files are not suitable for the final method comparison.
ROW_LIMIT = None

FINAL_FILENAMES = {
    "training": "training_final.csv",
    "validation": "validation_final.csv",
    "testing": "testing_final.csv",
}


# Loaded lazily so a validated completed generation can be converted into the
# shared split files without loading the 7B model into GPU memory again.
tokenizer = None
model = None


# ==========================================================
# REPRODUCIBILITY AND SINGLE-GPU SETUP
# ==========================================================

def seed_everything():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    set_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_qwen_model():
    global tokenizer, model

    if tokenizer is not None and model is not None:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Base Qwen requires one GPU.")

    torch.cuda.set_device(0)
    device = "cuda:0"
    print(
        f"Single-GPU generation on {device}; "
        f"batch size={BATCH_SIZE}; seed={RANDOM_SEED}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()


# ==========================================================
# CLEAN OUTPUT
# ==========================================================

def clean_keyword_output(text):
    if not isinstance(text, str):
        return ""

    text = text.strip().lower()

    text = re.sub(
        r"^(keywords?|key phrases?|output)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize common malformed generations before comma splitting.
    text = text.replace("，", ",")
    text = text.replace("_", " ")
    text = re.sub(
        r"cross\s*,\s*site\s*,\s*scripting",
        "cross site scripting",
        text,
    )
    text = re.sub(
        r"store\s+cross\s+site\s+script(?:ing)?",
        "stored xss",
        text,
    )
    text = re.sub(
        r"stored?\s+cross\s+site\s+scripting",
        "stored xss",
        text,
    )
    text = re.sub(
        r"reflected?\s+cross\s+site\s+scripting",
        "reflected xss",
        text,
    )
    text = re.sub(
        r"persistent\s+cross\s+site\s+scripting",
        "stored xss",
        text,
    )
    text = re.sub(
        r"remote\s*code\s*execution",
        "remote code execution",
        text,
    )
    text = re.sub(
        r"arbitrary\s*code\s*execution",
        "arbitrary code execution",
        text,
    )
    text = re.sub(
        r"improperauthentication",
        "improper authentication",
        text,
    )
    text = re.sub(
        r"authenticationbypass",
        "authentication bypass",
        text,
    )
    text = re.sub(
        r"authorizationbypass",
        "authorization bypass",
        text,
    )
    text = re.sub(r"pathvalidation", "path validation", text)

    text = text.replace("\n", ", ")
    text = re.sub(r"\s*\d+\.\s*", ", ", text)
    text = re.sub(r",\s*,+", ", ", text)
    text = re.sub(r"\s+", " ", text)

    keywords = []
    seen = set()
    banned = {
        "vulnerability",
        "security vulnerability",
        "issue",
        "security issue",
        "attacker",
        "remote attacker",
        "local attacker",
        "affected version",
        "version affected",
        "version range",
        "vendor",
        "product",
        "web application",
        "remote",
        "local",
        "heap",
        "site",
        "scripting",
        "cross",
    }

    for keyword in text.split(","):
        keyword = keyword.strip(" .;:-\\\"'()[]{}")
        keyword = keyword.encode("ascii", "ignore").decode("ascii")
        keyword = re.sub(r"[^a-z0-9*/_.:+ -]", " ", keyword)
        keyword = keyword.replace("_", " ")
        keyword = re.sub(r"\s+", " ", keyword).strip()

        if not keyword:
            continue
        if keyword in banned:
            continue
        if len(keyword) < 3:
            continue
        if len(keyword.split()) > 5:
            continue
        if keyword not in seen:
            keywords.append(keyword)
            seen.add(keyword)

    return ", ".join(keywords[:5])


# ==========================================================
# PROMPT
# ==========================================================

def build_prompt(description):
    return f"""
Extract between 1 and 5 distinct cybersecurity keyphrases from the CVE description.

Prioritize, when explicitly present:

- vulnerability or weakness type
- technical cause
- exploit input, request, parameter, file, packet, or protocol
- affected function, endpoint, component, or interface
- attacker access or required user action
- direct security consequence

Rules:

- Every phrase must be directly supported by the current description.
- Prefer exact technical wording from the description.
- Do not infer facts that are not stated.
- Do not infer authentication requirements.
- Do not infer local or remote access.
- Do not infer user interaction.
- Do not infer a root cause from the vulnerability name.
- Do not infer an impact from the weakness type.
- A crafted SQL statement does not automatically mean SQL injection.
- Command execution does not automatically mean command injection.
- File access does not automatically mean path traversal.
- A URL does not automatically mean SSRF.
- Exclude vendor names, version numbers, CVE IDs, reference IDs,
  and product names unless the product is the only affected component stated.
- Do not include generic words such as vulnerability, issue, attacker,
  product, vendor, affected version, weakness, vector, component, or impact.
- Use lowercase English ASCII.
- Output one comma-separated line only.
- Return fewer than 5 phrases when fewer than 5 useful facts are stated.

CVE description:
{description}
""".strip()


def build_chat_text(description):
    if pd.isna(description):
        description = ""
    description = str(description).strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You extract only cybersecurity information directly stated "
                "in the supplied CVE description. Never infer missing facts."
            ),
        },
        {
            "role": "user",
            "content": build_prompt(description),
        },
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ==========================================================
# BATCH GENERATION
# ==========================================================

def extract_keywords_batch(descriptions):
    chat_texts = [build_chat_text(description) for description in descriptions]

    inputs = tokenizer(
        chat_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to("cuda:0")

    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[:, input_length:]
    outputs = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    return [clean_keyword_output(output) for output in outputs]


# ==========================================================
# CHECKPOINT AND GENERATION-CACHE HANDLING
# ==========================================================

def atomic_csv_write(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def normalized_text_series(frame, column):
    return frame[column].fillna("").astype(str)


def validate_aligned_generation(source_frame, generated_frame, source_name):
    required = {"cve_id", TEXT_COLUMN, GENERATED_KEYWORD_COLUMN}
    missing = sorted(required - set(generated_frame.columns))
    if missing:
        raise ValueError(f"{source_name} is missing columns: {missing}")

    expected_ids = source_frame["cve_id"].astype(str).tolist()
    actual_ids = generated_frame["cve_id"].astype(str).tolist()
    if actual_ids != expected_ids:
        raise ValueError(
            f"Cannot use {source_name}: ordered CVE IDs do not match dataset.csv."
        )

    expected_text = normalized_text_series(source_frame, TEXT_COLUMN)
    actual_text = normalized_text_series(generated_frame, TEXT_COLUMN)
    if not actual_text.equals(expected_text):
        raise ValueError(
            f"Cannot use {source_name}: source descriptions do not match dataset.csv."
        )


def load_or_create_generation_frame(source_frame):
    if CHECKPOINT_CSV.is_file():
        checkpoint = pd.read_csv(CHECKPOINT_CSV)
        validate_aligned_generation(
            source_frame,
            checkpoint,
            str(CHECKPOINT_CSV),
        )
        if PROCESSED_COLUMN not in checkpoint.columns:
            raise ValueError(
                f"Checkpoint is missing {PROCESSED_COLUMN}: {CHECKPOINT_CSV}"
            )
        checkpoint[PROCESSED_COLUMN] = (
            checkpoint[PROCESSED_COLUMN]
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin({"true", "1", "yes"})
        )
        print("Resuming Qwen generation from:", CHECKPOINT_CSV)
        return checkpoint

    if REUSE_VALIDATED_GENERATION_CACHE and GENERATION_CACHE_CSV.is_file():
        cached_generation = pd.read_csv(GENERATION_CACHE_CSV)
        validate_aligned_generation(
            source_frame,
            cached_generation,
            str(GENERATION_CACHE_CSV),
        )
        generation = source_frame[["cve_id", TEXT_COLUMN]].copy()
        generation[GENERATED_KEYWORD_COLUMN] = (
            cached_generation[GENERATED_KEYWORD_COLUMN]
            .fillna("")
            .astype(str)
        )
        # The earlier job generated every row before rejecting its valid empty
        # outputs. Under this project rule, those empty values are completed
        # rows and deliberately become zero predictions during evaluation.
        generation[PROCESSED_COLUMN] = True
        print("Reusing validated completed Qwen generation:", GENERATION_CACHE_CSV)
        return generation

    generation = source_frame[["cve_id", TEXT_COLUMN]].copy()
    generation[GENERATED_KEYWORD_COLUMN] = ""
    generation[PROCESSED_COLUMN] = False
    return generation


# ==========================================================
# SHARED 70/15/15 FINAL SPLITS
# ==========================================================

def split_final_frame(frame):
    unique_cves = frame["cve_id"].dropna().astype(str).unique()
    training_cves, held_out_cves = train_test_split(
        unique_cves,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    cve_ids = frame["cve_id"].astype(str)
    training = frame[cve_ids.isin(training_cves)].copy().reset_index(drop=True)
    held_out = frame[cve_ids.isin(held_out_cves)].copy().reset_index(drop=True)

    validation_cves, testing_cves = train_test_split(
        held_out["cve_id"].astype(str).unique(),
        test_size=0.50,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    held_out_ids = held_out["cve_id"].astype(str)
    validation = held_out[
        held_out_ids.isin(validation_cves)
    ].copy().reset_index(drop=True)
    testing = held_out[
        held_out_ids.isin(testing_cves)
    ].copy().reset_index(drop=True)

    return {
        "training": training,
        "validation": validation,
        "testing": testing,
    }


def create_final_files(source_frame, generation_frame):
    final_frame = source_frame.copy()
    final_frame["keyphrases"] = (
        generation_frame[GENERATED_KEYWORD_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    final_frame["keyphrase_source"] = np.where(
        final_frame["keyphrases"].eq(""),
        "empty",
        KEYPHRASE_SOURCE,
    )

    split_frames = split_final_frame(final_frame)
    for split_name, split_frame in split_frames.items():
        output_path = OUTPUT_DIR / FINAL_FILENAMES[split_name]
        atomic_csv_write(split_frame, output_path)
        empty_count = int(split_frame["keyphrases"].eq("").sum())
        print(
            f"Saved {output_path}: rows={len(split_frame):,}, "
            f"empty keyphrases={empty_count:,}"
        )

    if sum(len(frame) for frame in split_frames.values()) != len(final_frame):
        raise RuntimeError("Qwen final splits do not cover dataset.csv exactly.")

    return split_frames


def final_outputs_exist():
    return all(
        (OUTPUT_DIR / filename).is_file()
        for filename in FINAL_FILENAMES.values()
    )


# ==========================================================
# PROCESS DATASET.CSV
# ==========================================================

def process_dataset():
    if not DATASET_CSV.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_CSV}")

    source_frame = pd.read_csv(DATASET_CSV)
    source_frame.columns = source_frame.columns.str.strip()

    if ROW_LIMIT is not None:
        source_frame = source_frame.head(ROW_LIMIT).copy()
        print(f"Trial mode: using first {len(source_frame)} rows")

    required = {"cve_id", TEXT_COLUMN, "cleaned_description", "capec_id"}
    missing = sorted(required - set(source_frame.columns))
    if missing:
        raise ValueError(f"dataset.csv is missing columns: {missing}")
    if source_frame["cve_id"].isna().any():
        raise ValueError("dataset.csv contains missing cve_id values.")
    if source_frame["cve_id"].astype(str).duplicated().any():
        raise ValueError("dataset.csv contains duplicate cve_id values.")

    if final_outputs_exist():
        print("All Qwen final split files already exist; nothing to do.")
        return

    generation_frame = load_or_create_generation_frame(source_frame)
    pending_indices = np.where(
        ~generation_frame[PROCESSED_COLUMN].to_numpy(dtype=bool)
    )[0].tolist()
    print(f"Pending Qwen rows: {len(pending_indices):,}")

    if pending_indices:
        load_qwen_model()

    for batch_start in tqdm(
        range(0, len(pending_indices), BATCH_SIZE),
        desc=f"Base Qwen {GENERATED_KEYWORD_COLUMN}",
    ):
        batch_indices = pending_indices[
            batch_start:batch_start + BATCH_SIZE
        ]
        descriptions = [
            generation_frame.at[index, TEXT_COLUMN]
            for index in batch_indices
        ]

        try:
            batch_keywords = extract_keywords_batch(descriptions)
            for row_index, keywords in zip(batch_indices, batch_keywords):
                generation_frame.at[
                    row_index,
                    GENERATED_KEYWORD_COLUMN,
                ] = keywords
                generation_frame.at[row_index, PROCESSED_COLUMN] = True
        except torch.cuda.OutOfMemoryError:
            print(f"\nOOM near source row {batch_indices[0]}. Saving progress.")
            atomic_csv_write(generation_frame, CHECKPOINT_CSV)
            torch.cuda.empty_cache()
            raise
        except Exception as error:
            atomic_csv_write(generation_frame, CHECKPOINT_CSV)
            raise RuntimeError(
                f"Generation failed near source row {batch_indices[0]}: {error}"
            ) from error

        completed_batches = batch_start // BATCH_SIZE + 1
        if completed_batches % SAVE_EVERY_BATCHES == 0:
            atomic_csv_write(generation_frame, CHECKPOINT_CSV)

    if pending_indices:
        atomic_csv_write(generation_frame, CHECKPOINT_CSV)

    unprocessed = int((~generation_frame[PROCESSED_COLUMN]).sum())
    if unprocessed:
        raise RuntimeError(
            f"Qwen generation has {unprocessed} unprocessed rows."
        )

    # Empty extracted keyphrases are valid. The shared NN evaluator converts
    # them to explicit all-zero CAPEC predictions.
    empty_count = int(
        generation_frame[GENERATED_KEYWORD_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )
    split_frames = create_final_files(source_frame, generation_frame)

    config = {
        "completed": True,
        "method": "base Qwen keyword extraction",
        "model_name": MODEL_NAME,
        "dataset_path": str(DATASET_CSV),
        "output_directory": str(OUTPUT_DIR),
        "text_column": TEXT_COLUMN,
        "keyphrase_column": "keyphrases",
        "source_keyword_column": GENERATED_KEYWORD_COLUMN,
        "keyphrase_source": KEYPHRASE_SOURCE,
        "empty_keyword_rule": "valid extraction; zero CAPEC prediction",
        "total_rows": len(source_frame),
        "empty_keyword_rows": empty_count,
        "split_rows": {
            name: len(frame) for name, frame in split_frames.items()
        },
        "split_strategy": "CVE-ID-level 70/15/15 with seed 42",
        "random_seed": RANDOM_SEED,
        "batch_size": BATCH_SIZE,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "generation_cache_reuse_enabled": REUSE_VALIDATED_GENERATION_CACHE,
        "generation_cache_path": str(GENERATION_CACHE_CSV),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config_path = OUTPUT_DIR / "config.json"
    temporary_config = OUTPUT_DIR / "config.partial.json"
    temporary_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
    os.replace(temporary_config, config_path)

    if CHECKPOINT_CSV.is_file() and not KEEP_COMPLETED_CHECKPOINT:
        CHECKPOINT_CSV.unlink()
        print("Removed completed generation checkpoint:", CHECKPOINT_CSV)

    error_path = OUTPUT_DIR / "qwen_error.log"
    if error_path.is_file():
        error_path.unlink()

    print(f"Qwen extraction complete: rows={len(source_frame):,}")
    print(f"Empty keyphrase rows retained as zero predictions: {empty_count:,}")


# ==========================================================
# MAIN
# ==========================================================

def main():
    seed_everything()
    print("Dataset:", DATASET_CSV)
    print("Output directory:", OUTPUT_DIR)
    process_dataset()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        error_path = OUTPUT_DIR / "qwen_error.log"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"Base-Qwen generation failed. Traceback saved to {error_path}")
        raise
