#!/usr/bin/env python
# coding: utf-8

# ============================================================
# CVE -> MULTI-LABEL CAPEC CLASSIFICATION
# Final keyword model code
#
# Fixed model: concatenated description and keyphrase MiniLM embeddings.
# ============================================================

import os
import re
import ast
import json
import random
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    jaccard_score,
    hamming_loss,
    accuracy_score,
    label_ranking_average_precision_score,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

from evaluation_paths import OUTPUT_ROOT

try:
    from IPython.display import display
except Exception:
    display = print


# ============================================================
# 0. SETTINGS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KEYWORD_DIR = os.path.abspath(os.environ.get(
    "EVALUATION_KEYWORD_SPLIT_DIR",
    os.path.join(SCRIPT_DIR, "../03_keyword_making/outputs/keybert_modified"),
))
KEYWORD_METHOD = os.environ.get("EVALUATION_KEYWORD_METHOD", "keybert_modified")
OUTPUT_MODEL_NAME = os.environ.get("EVALUATION_MODEL_OUTPUT_NAME", "NN_keywords")
CAPEC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../01_cwe_capec_ground_truth"))

TRAIN_PATH = os.path.join(KEYWORD_DIR, "training_final.csv")
VAL_PATH = os.path.join(KEYWORD_DIR, "validation_final.csv")
TEST_PATH = os.path.join(KEYWORD_DIR, "testing_final.csv")

# Used only to include the canonical CAPEC taxonomy in the label space.
CAPEC_CATALOG_PATH = os.path.join(CAPEC_DIR, "3000_capec.csv")

DESC_COL = "cleaned_description"
KEYPHRASE_COL = "keyphrases"
CAPEC_COL = "capec_id"

NO_ID_LABEL = "CAPEC-noID"
RANDOM_STATE = 42

BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MIN_LABEL_COUNT_REPORT = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)
# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(RANDOM_STATE)


# ============================================================
# 2. HELPERS
# ============================================================

def parse_capec_cell(x):
    """
    Handles:
    - "['CAPEC-1', 'CAPEC-2']"
    - "CAPEC-1, CAPEC-2"
    - "CAPEC-1; CAPEC-2"
    - "CAPEC-1|CAPEC-2"
    - "CAPEC-noID"
    - NaN / empty

    Returns list of CAPEC ids like ["CAPEC-1", "CAPEC-2"].
    """
    if pd.isna(x):
        return []

    x = str(x).strip()

    if x == "" or x.lower() in ["nan", "none", "null", "[]"]:
        return []

    lowered = x.lower().replace("_", "").replace(" ", "")
    if lowered in ["capec-noid", "noid", "nocapec", "capecnone"]:
        return []

    labels = []

    # Case 1: Python-list-like string
    try:
        parsed = ast.literal_eval(x)

        if isinstance(parsed, list):
            raw_items = parsed
        elif isinstance(parsed, tuple):
            raw_items = list(parsed)
        elif isinstance(parsed, set):
            raw_items = list(parsed)
        else:
            raw_items = [parsed]

        for item in raw_items:
            item = str(item).strip()
            item_low = item.lower().replace("_", "").replace(" ", "")

            if item_low in ["capec-noid", "noid", "nocapec", "capecnone"]:
                continue

            found = re.findall(r"CAPEC-\d+", item, flags=re.IGNORECASE)
            labels.extend([f.upper() for f in found])

        return sorted(set(labels))

    except Exception:
        pass

    # Case 2: raw string
    found = re.findall(r"CAPEC-\d+", x, flags=re.IGNORECASE)
    labels = [f.upper() for f in found]

    return sorted(set(labels))



def parse_keyphrase_cell(x):
    """
    Handles:
    - "['sql injection', 'cross site scripting']"
    - "sql injection, cross site scripting"
    - "sql injection; cross site scripting"
    - "sql injection | cross site scripting"
    - empty / NaN

    Returns list of unique clean keyphrases.
    """
    if pd.isna(x):
        return []

    x = str(x).strip()

    if x == "" or x.lower() in ["nan", "none", "null", "[]"]:
        return []

    keyphrases = []

    # Case 1: Python-list-like string
    try:
        parsed = ast.literal_eval(x)

        if isinstance(parsed, list):
            raw_items = parsed
        elif isinstance(parsed, tuple):
            raw_items = list(parsed)
        elif isinstance(parsed, set):
            raw_items = list(parsed)
        else:
            raw_items = [parsed]

        for item in raw_items:
            item = str(item).strip()
            if item and item.lower() not in ["nan", "none", "null", "[]"]:
                keyphrases.append(item)

    except Exception:
        # Case 2: plain separated string
        raw_items = re.split(r"[;|\n,]+", x)

        for item in raw_items:
            item = str(item).strip()
            if item and item.lower() not in ["nan", "none", "null", "[]"]:
                keyphrases.append(item)

    seen = set()
    final = []

    for kp in keyphrases:
        kp_clean = re.sub(r"\s+", " ", kp).strip()
        key = kp_clean.lower()

        if kp_clean and key not in seen:
            final.append(kp_clean)
            seen.add(key)

    return final



def extract_capecs_from_catalog(path):
    """
    Extract all CAPEC IDs from the canonical stage-01 CAPEC catalog.
    Handles both CAPEC-123 and numeric-only id cells.
    """
    capec_rel = pd.read_csv(path)
    id_column = next(
        (column for column in ("ID", "id", "capec_id") if column in capec_rel),
        None,
    )
    if id_column is None:
        raise ValueError(f"Cannot find the CAPEC ID column in {path}")
    capec_ids = set()
    for value in capec_rel[id_column].dropna():
        text = str(value).strip()
        found = re.findall(r"CAPEC-\d+", text, flags=re.IGNORECASE)
        capec_ids.update(item.upper() for item in found)
        if re.fullmatch(r"\d+(?:\.0)?", text):
            capec_ids.add(f"CAPEC-{int(float(text))}")

    return sorted(capec_ids)



def l2_normalize_np(x, eps=1e-12):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)



def make_multihot(label_lists, label_to_idx):
    y = np.zeros((len(label_lists), len(label_to_idx)), dtype=np.float32)

    for i, labels in enumerate(label_lists):
        valid_labels = [lab for lab in labels if lab in label_to_idx]

        # Safety fallback.
        if len(valid_labels) == 0:
            valid_labels = [NO_ID_LABEL]

        for lab in valid_labels:
            y[i, label_to_idx[lab]] = 1.0

    return y



def check_required_columns(split_df, split_name):
    required = [DESC_COL, CAPEC_COL]
    missing = [c for c in required if c not in split_df.columns]

    if missing:
        raise ValueError(f"{split_name} is missing required columns: {missing}")

    if KEYPHRASE_COL not in split_df.columns:
        print(f"Warning: {split_name} missing {KEYPHRASE_COL}. Creating empty keyphrase column.")
        split_df[KEYPHRASE_COL] = ""

    split_df[DESC_COL] = split_df[DESC_COL].fillna("").astype(str)
    split_df[KEYPHRASE_COL] = split_df[KEYPHRASE_COL].fillna("").astype(str)
    split_df[CAPEC_COL] = split_df[CAPEC_COL].fillna("").astype(str)

    return split_df

# ============================================================
# 3. LOAD FIXED TRAIN / VALIDATION / TEST SPLITS
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df = check_required_columns(train_df, "train_df")
val_df = check_required_columns(val_df, "val_df")
test_df = check_required_columns(test_df, "test_df")

# Drop empty descriptions inside each split.
train_df = train_df[train_df[DESC_COL].str.strip() != ""].reset_index(drop=True)
val_df = val_df[val_df[DESC_COL].str.strip() != ""].reset_index(drop=True)
test_df = test_df[test_df[DESC_COL].str.strip() != ""].reset_index(drop=True)

print("\nFixed split sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

total_rows = len(train_df) + len(val_df) + len(test_df)

print("\nSplit percentages:")
print("Train %:", round(len(train_df) / total_rows * 100, 2))
print("Validation %:", round(len(val_df) / total_rows * 100, 2))
print("Test %:", round(len(test_df) / total_rows * 100, 2))
# ============================================================
# 4. PARSE LABELS AND KEYPHRASES
# ============================================================

for split_df in [train_df, val_df, test_df]:
    split_df["capec_real_list"] = split_df[CAPEC_COL].apply(parse_capec_cell)
    split_df["capec_label_list"] = split_df["capec_real_list"].apply(
        lambda labels: labels if len(labels) > 0 else [NO_ID_LABEL]
    )
    split_df["keyphrase_list"] = split_df[KEYPHRASE_COL].apply(parse_keyphrase_cell)

print("\nLabel parsing:")
print("Train real CAPEC rows:", int((train_df["capec_real_list"].apply(len) > 0).sum()))
print("Val real CAPEC rows:", int((val_df["capec_real_list"].apply(len) > 0).sum()))
print("Test real CAPEC rows:", int((test_df["capec_real_list"].apply(len) > 0).sum()))

print("\nKeyphrase parsing:")
print("Train rows with keyphrases:", int((train_df["keyphrase_list"].apply(len) > 0).sum()))
print("Val rows with keyphrases:", int((val_df["keyphrase_list"].apply(len) > 0).sum()))
print("Test rows with keyphrases:", int((test_df["keyphrase_list"].apply(len) > 0).sum()))
print("Example keyphrases:")
display(train_df[[DESC_COL, KEYPHRASE_COL, "keyphrase_list"]].head())


# ============================================================
# 5. BUILD LABEL SPACE AND MULTI-HOT MATRICES
# ============================================================

catalog_capecs = extract_capecs_from_catalog(CAPEC_CATALOG_PATH)

data_capecs = sorted(set(
    capec
    for split_df in [train_df, val_df, test_df]
    for labels in split_df["capec_real_list"]
    for capec in labels
))

all_real_capecs = sorted(set(catalog_capecs) | set(data_capecs))
all_labels = all_real_capecs + [NO_ID_LABEL]

label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

NUM_LABELS = len(all_labels)
NO_ID_INDEX = label_to_idx[NO_ID_LABEL]

Y_train = make_multihot(train_df["capec_label_list"].tolist(), label_to_idx)
Y_val = make_multihot(val_df["capec_label_list"].tolist(), label_to_idx)
Y_test = make_multihot(test_df["capec_label_list"].tolist(), label_to_idx)

print("\nLabel space:")
print("CAPECs from catalog:", len(catalog_capecs))
print("CAPECs found in split files:", len(data_capecs))
print("Final real CAPEC labels:", len(all_real_capecs))
print("Total output labels including CAPEC-noID:", NUM_LABELS)

extra_data_capecs = sorted(set(data_capecs) - set(catalog_capecs))
print("CAPECs in data but not in relationships:", len(extra_data_capecs))

print("\nY shapes:")
print("Y_train:", Y_train.shape)
print("Y_val:", Y_val.shape)
print("Y_test:", Y_test.shape)


# ============================================================
# 6. LABEL FREQUENCY REPORT
# ============================================================

train_label_counts = Y_train.sum(axis=0)

rare_train_labels = [
    all_labels[i]
    for i, count in enumerate(train_label_counts)
    if 0 < count < MIN_LABEL_COUNT_REPORT
]

zero_train_labels = [
    all_labels[i]
    for i, count in enumerate(train_label_counts)
    if count == 0
]

trainable_10_plus = [
    all_labels[i]
    for i, count in enumerate(train_label_counts)
    if count >= MIN_LABEL_COUNT_REPORT
]

print("\nTraining label frequency report:")
print(f"Labels with >= {MIN_LABEL_COUNT_REPORT} train examples:", len(trainable_10_plus))
print(f"Labels with 1-{MIN_LABEL_COUNT_REPORT - 1} train examples:", len(rare_train_labels))
print("Labels with 0 train examples:", len(zero_train_labels))
print("CAPEC-noID train count:", int(train_label_counts[NO_ID_INDEX]))


# ============================================================
# 7. ENCODE DESCRIPTION + KEYPHRASE POOLING USING MINILM
# ============================================================

embedder = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
EMBED_DIM = embedder.get_sentence_embedding_dimension()

print("\nEmbedding model:", EMBED_MODEL_NAME)
print("Embedding dimension:", EMBED_DIM)

if EMBED_DIM != 384:
    print("Warning: expected MiniLM dim 384, got:", EMBED_DIM)



def encode_texts(texts, batch_size=256):
    emb = embedder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)



def encode_keyphrases_mean(keyphrase_lists, batch_size=256):
    """
    For each row:
    - encode every keyphrase using MiniLM
    - pool across keyphrase embeddings

    If no keyphrase exists:
    - return zero vector
    - has_keyphrase[row] = False
    """
    n = len(keyphrase_lists)
    pooled = np.zeros((n, EMBED_DIM), dtype=np.float32)
    has_keyphrase = np.zeros(n, dtype=bool)
    counts = np.zeros(n, dtype=np.float32)

    flat_keyphrases = []
    row_ids = []

    for row_id, kps in enumerate(keyphrase_lists):
        if not isinstance(kps, list):
            kps = []

        for kp in kps:
            kp = str(kp).strip()

            if kp:
                flat_keyphrases.append(kp)
                row_ids.append(row_id)

    print("Total keyphrases to encode:", len(flat_keyphrases))
    print("Pooling method: mean")

    if len(flat_keyphrases) == 0:
        return pooled, has_keyphrase

    for start in range(0, len(flat_keyphrases), batch_size):
        end = min(start + batch_size, len(flat_keyphrases))

        batch_phrases = flat_keyphrases[start:end]
        batch_rows = row_ids[start:end]

        batch_emb = embedder.encode(
            batch_phrases,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)

        for emb, row_id in zip(batch_emb, batch_rows):
            pooled[row_id] += emb
            counts[row_id] += 1.0
            has_keyphrase[row_id] = True

        if start == 0 or end == len(flat_keyphrases) or start % (batch_size * 20) == 0:
            print(f"Encoded keyphrases: {end}/{len(flat_keyphrases)}")

    nonzero = counts > 0
    pooled[nonzero] = pooled[nonzero] / counts[nonzero, None]

    return pooled.astype(np.float32), has_keyphrase



def build_features(split_df, split_name):
    print(f"\nEncoding {split_name} descriptions...")
    desc_emb = encode_texts(split_df[DESC_COL].tolist(), batch_size=BATCH_SIZE)

    print(f"\nEncoding {split_name} keyphrases with mean pooling...")
    keyphrase_emb, has_keyphrase = encode_keyphrases_mean(
        split_df["keyphrase_list"].tolist(),
        batch_size=BATCH_SIZE,
    )

    # Ensure both main blocks are L2-normalized.
    desc_emb = l2_normalize_np(desc_emb).astype(np.float32)
    keyphrase_emb = l2_normalize_np(keyphrase_emb).astype(np.float32)

    X = np.concatenate([desc_emb, keyphrase_emb], axis=1).astype(np.float32)

    print(f"\n{split_name} feature shapes:")
    print("desc_emb:", desc_emb.shape)
    print("keyphrase_emb:", keyphrase_emb.shape)
    print("final X:", X.shape)
    print("rows with keyphrase:", int(has_keyphrase.sum()))
    print("rows without keyphrase:", int((~has_keyphrase).sum()))

    return X


X_train = build_features(train_df, "train")
X_val = build_features(val_df, "val")
X_test = build_features(test_df, "test")

print("\nFinal feature shapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

# Release embedder after encoding.
del embedder
torch.cuda.empty_cache()

if DEVICE == "cuda":
    print("VRAM after clearing embedder:", torch.cuda.memory_allocated() / 1e9, "GB")


# ============================================================
# 9. DATASET / DATALOADER
# ============================================================

class EmbeddingDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


train_loader = DataLoader(
    EmbeddingDataset(X_train, Y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    EmbeddingDataset(X_val, Y_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    EmbeddingDataset(X_test, Y_test),
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# 10. MODEL
# ============================================================

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

    def forward(self, x):
        return self.net(x)


model = MultiLabelMLP(
    input_dim=X_train.shape[1],
    num_labels=NUM_LABELS,
).to(DEVICE)

print("\nModel:")
print("Input dim:", X_train.shape[1])
print("Output labels:", NUM_LABELS)
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


# ============================================================
# 11. LOSS / OPTIMIZER
# ============================================================

pos_counts = Y_train.sum(axis=0)
neg_counts = len(Y_train) - pos_counts

pos_weight = np.ones(NUM_LABELS, dtype=np.float32)
nonzero_mask = pos_counts > 0
pos_weight[nonzero_mask] = neg_counts[nonzero_mask] / pos_counts[nonzero_mask]

# Clip to avoid extreme instability for very rare labels.
pos_weight = np.clip(pos_weight, 1.0, 50.0)
pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# 12. TRAINING / PREDICTION / METRICS
# ============================================================

AMP_ENABLED = DEVICE == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)



def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        Y_batch = Y_batch.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
            logits = model(X_batch)
            loss = criterion(logits, Y_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()

    all_probs = []
    all_true = []

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(DEVICE)

        logits = model(X_batch)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu().numpy())
        all_true.append(Y_batch.numpy())

    probs = np.vstack(all_probs)
    true = np.vstack(all_true)

    return probs, true



def postprocess_predictions(probs, threshold, no_id_index):
    """
    Multi-label thresholding.

    Logic:
    - predict every label whose probability >= threshold
    - CAPEC-noID cannot appear together with real CAPECs
    - if nothing crosses threshold, output the single highest-probability label
    """
    pred = (probs >= threshold).astype(int)

    for i in range(pred.shape[0]):
        positive_indices = np.where(pred[i] == 1)[0].tolist()

        if len(positive_indices) == 0:
            best_idx = int(np.argmax(probs[i]))
            pred[i, best_idx] = 1
            continue

        real_positive_indices = [idx for idx in positive_indices if idx != no_id_index]

        if len(real_positive_indices) > 0:
            pred[i, no_id_index] = 0

    return pred



def sample_average_metrics(y_true, y_pred):
    precisions = []
    recalls = []
    f1s = []
    jaccards = []

    for true_row, pred_row in zip(y_true, y_pred):
        true_set = set(np.where(true_row == 1)[0])
        pred_set = set(np.where(pred_row == 1)[0])

        intersection = len(true_set & pred_set)

        precision = intersection / len(pred_set) if len(pred_set) > 0 else 0.0
        recall = intersection / len(true_set) if len(true_set) > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        union = len(true_set | pred_set)
        jaccard = intersection / union if union > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        jaccards.append(jaccard)

    return {
        "sample_precision": float(np.mean(precisions)),
        "sample_recall": float(np.mean(recalls)),
        "sample_f1": float(np.mean(f1s)),
        "sample_jaccard": float(np.mean(jaccards)),
    }



def evaluate_multilabel(y_true, probs, threshold, no_id_index):
    y_pred = postprocess_predictions(probs, threshold, no_id_index)
    sample_metrics = sample_average_metrics(y_true, y_pred)

    metrics = {
        "threshold": threshold,

        # Strict full-set match. Hard metric for multi-label.
        "exact_match_accuracy": accuracy_score(y_true, y_pred),

        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),

        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),

        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),

        "sample_precision": sample_metrics["sample_precision"],
        "sample_recall": sample_metrics["sample_recall"],
        "sample_f1": sample_metrics["sample_f1"],
        "sample_jaccard": sample_metrics["sample_jaccard"],

        "sklearn_samples_jaccard": jaccard_score(y_true, y_pred, average="samples", zero_division=0),

        # Lower is better.
        "hamming_loss": hamming_loss(y_true, y_pred),
    }

    try:
        metrics["label_ranking_average_precision"] = label_ranking_average_precision_score(y_true, probs)
    except Exception:
        metrics["label_ranking_average_precision"] = None

    return metrics, y_pred



def print_metrics(metrics, title="Metrics"):
    print(f"\n{title}")
    print("-" * len(title))

    for k, v in metrics.items():
        if v is None:
            print(f"{k}: None")
        else:
            print(f"{k}: {v:.4f}")



def find_best_threshold(y_true, probs, no_id_index):
    best_score = -1.0
    best_metrics = None

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
    best_threshold = float(thresholds[0])

    for threshold in thresholds:
        threshold = round(float(threshold), 4)

        metrics, _ = evaluate_multilabel(
            y_true=y_true,
            probs=probs,
            threshold=threshold,
            no_id_index=no_id_index,
        )

        # Primary validation-selection metric.
        score = metrics["micro_f1"]

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics

# ============================================================
# 13. TRAIN LOOP WITH EARLY STOPPING
# ============================================================

SAVE_DIR = os.path.join(str(OUTPUT_ROOT), OUTPUT_MODEL_NAME)

os.makedirs(SAVE_DIR, exist_ok=True)
best_model_path = os.path.join(SAVE_DIR, "best_capec_multilabel_mlp.pt")

best_val_micro_f1 = -1.0
best_state = None
patience_counter = 0
best_threshold = 0.80

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)

    val_probs, val_true = predict_probs(model, val_loader)

    epoch_best_threshold, val_metrics = find_best_threshold(
        y_true=val_true,
        probs=val_probs,
        no_id_index=NO_ID_INDEX,
    )

    val_micro_f1 = val_metrics["micro_f1"]

    print(f"\nEpoch {epoch}/{EPOCHS}")
    print("Train loss:", round(train_loss, 4))
    print("Best val threshold:", epoch_best_threshold)
    print("Val micro F1:", round(val_micro_f1, 4))
    print("Val sample F1:", round(val_metrics["sample_f1"], 4))
    print("Val sample Jaccard:", round(val_metrics["sample_jaccard"], 4))
    print("Val exact match:", round(val_metrics["exact_match_accuracy"], 4))

    if val_micro_f1 > best_val_micro_f1:
        best_val_micro_f1 = val_micro_f1
        best_threshold = epoch_best_threshold

        best_state = {
            "model": model.state_dict(),
            "threshold": best_threshold,
            "val_metrics": val_metrics,
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "all_labels": all_labels,
            "desc_col": DESC_COL,
            "keyphrase_col": KEYPHRASE_COL,
            "capec_col": CAPEC_COL,
            "embed_model_name": EMBED_MODEL_NAME,
            "input_dim": X_train.shape[1],
            "embed_dim": EMBED_DIM,
            "feature_representation": "description_and_mean_pooled_keyphrases",
            "random_state": RANDOM_STATE,
            "batch_size": BATCH_SIZE,
            "epochs_finished": epoch,
            "train_path": TRAIN_PATH,
            "val_path": VAL_PATH,
            "test_path": TEST_PATH,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
        }

        torch.save(best_state, best_model_path)

        patience_counter = 0
        print("Saved best model to:", best_model_path)

    else:
        patience_counter += 1
        print("No improvement. Patience:", patience_counter)

        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break

if best_state is None:
    raise RuntimeError("No best model was saved. Check data, labels, and validation set.")

# ============================================================
# 14. FINAL TEST EVALUATION
# ============================================================

model.load_state_dict(best_state["model"])

test_probs, test_true = predict_probs(model, test_loader)

test_metrics, test_pred = evaluate_multilabel(
    y_true=test_true,
    probs=test_probs,
    threshold=best_threshold,
    no_id_index=NO_ID_INDEX,
)

print_metrics(best_state["val_metrics"], title="Best Validation Metrics")
print_metrics(test_metrics, title="Final Test Metrics")

print("\nBest threshold:", best_threshold)


# ============================================================
# 15. TOP-K HIT / RECALL METRICS ON REAL CAPEC ROWS
# ============================================================
# Hit@K: at least one true real CAPEC appears in top-K predictions.
# Recall@K: fraction of true real CAPECs recovered in top-K predictions.
# CAPEC-noID is excluded by default so real CAPEC ranking is not inflated/deflated.

def sorted_label_indices(prob_row, no_id_index=None, exclude_noid=True):
    sorted_indices = np.argsort(prob_row)[::-1].tolist()

    if exclude_noid and no_id_index is not None:
        sorted_indices = [idx for idx in sorted_indices if idx != no_id_index]

    return sorted_indices


def top_k_hit_metrics(y_true, probs, ks=(1, 2, 3, 5, 10), no_id_index=None, exclude_noid=True):
    results = {}

    for k in ks:
        hits = []
        evaluated_rows = 0

        for true_row, prob_row in zip(y_true, probs):
            true_set = set(np.where(true_row == 1)[0])

            if exclude_noid and no_id_index is not None:
                true_set.discard(no_id_index)

            # Skip rows with no real CAPEC labels.
            if len(true_set) == 0:
                continue

            topk_set = set(sorted_label_indices(prob_row, no_id_index, exclude_noid)[:k])
            hits.append(1 if len(true_set & topk_set) > 0 else 0)
            evaluated_rows += 1

        results[f"hit@{k}"] = float(np.mean(hits)) if hits else 0.0
        results[f"evaluated_rows@{k}"] = evaluated_rows

    return results


def top_k_recall_metrics(y_true, probs, ks=(1, 2, 3, 5, 10, 20, 30, 50), no_id_index=None, exclude_noid=True):
    results = {}

    for k in ks:
        recalls = []
        evaluated_rows = 0

        for true_row, prob_row in zip(y_true, probs):
            true_set = set(np.where(true_row == 1)[0])

            if exclude_noid and no_id_index is not None:
                true_set.discard(no_id_index)

            # Skip rows with no real CAPEC labels.
            if len(true_set) == 0:
                continue

            topk_set = set(sorted_label_indices(prob_row, no_id_index, exclude_noid)[:k])
            recalls.append(len(true_set & topk_set) / len(true_set))
            evaluated_rows += 1

        results[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        results[f"evaluated_rows@{k}"] = evaluated_rows

    return results


def get_top_k_labels(prob_row, k=10, no_id_index=None, exclude_noid=True):
    top_indices = sorted_label_indices(prob_row, no_id_index, exclude_noid)[:k]
    return [(idx_to_label[idx], float(prob_row[idx])) for idx in top_indices]


def true_labels_from_row(true_row, no_id_index=None, exclude_noid=True):
    true_indices = np.where(true_row == 1)[0].tolist()

    if exclude_noid and no_id_index is not None:
        true_indices = [idx for idx in true_indices if idx != no_id_index]

    return [idx_to_label[idx] for idx in true_indices]


def hit_from_top_list(true_row, top_list, no_id_index=None, exclude_noid=True):
    true_set = set(np.where(true_row == 1)[0])

    if exclude_noid and no_id_index is not None:
        true_set.discard(no_id_index)

    if len(true_set) == 0:
        return None

    pred_set = set(label_to_idx[label] for label, _score in top_list)
    return int(len(true_set & pred_set) > 0)


TOPK_HIT_KS = [1, 2, 3, 5, 10]
TOPK_RECALL_KS = [1, 2, 3, 5, 10, 20, 30, 50]

hit_results = top_k_hit_metrics(
    y_true=test_true,
    probs=test_probs,
    ks=TOPK_HIT_KS,
    no_id_index=NO_ID_INDEX,
    exclude_noid=True,
)

recall_results = top_k_recall_metrics(
    y_true=test_true,
    probs=test_probs,
    ks=TOPK_RECALL_KS,
    no_id_index=NO_ID_INDEX,
    exclude_noid=True,
)

print("\nTop-K Hit Results on Real CAPEC Rows")
print("------------------------------------")
for k in TOPK_HIT_KS:
    print(f"Top-{k} Hit@{k}: {hit_results[f'hit@{k}']:.4f}")

print("\nEvaluated real-CAPEC rows:", hit_results.get("evaluated_rows@1", 0))

print("\nTop-K Recall Results on Real CAPEC Rows")
print("---------------------------------------")
for k in TOPK_RECALL_KS:
    print(f"Top-{k} Recall@{k}: {recall_results[f'recall@{k}']:.4f}")


# ============================================================
# 16. CONVERT PREDICTIONS BACK TO LABEL NAMES
# ============================================================

def indices_to_labels(row):
    return [idx_to_label[i] for i in np.where(row == 1)[0]]


test_output = test_df.copy()
test_output["true_capec_labels"] = [indices_to_labels(row) for row in test_true]
test_output["true_real_capec_labels"] = [
    true_labels_from_row(row, NO_ID_INDEX, exclude_noid=True)
    for row in test_true
]
test_output["pred_capec_labels"] = [indices_to_labels(row) for row in test_pred]

# Per-row top-K predictions and Hit@K flags, excluding CAPEC-noID.
for k in TOPK_HIT_KS:
    col = f"top_{k}_predictions"
    test_output[col] = [
        get_top_k_labels(row, k=k, no_id_index=NO_ID_INDEX, exclude_noid=True)
        for row in test_probs
    ]
    test_output[f"hit_at_{k}"] = [
        hit_from_top_list(test_true[i], test_output.iloc[i][col], NO_ID_INDEX, exclude_noid=True)
        for i in range(len(test_output))
    ]

# Keep old convenient top-10 score column name too.
test_output["top_10_capec_scores"] = test_output["top_10_predictions"]

print("\nExample predictions:")
display_cols = [
    DESC_COL,
    KEYPHRASE_COL,
    "true_real_capec_labels",
    "pred_capec_labels",
    "top_10_predictions",
    "hit_at_1",
    "hit_at_5",
    "hit_at_10",
]
display(test_output[display_cols].head())

# ============================================================
# 17. SAVE OUTPUTS
# ============================================================

# Save final best checkpoint again with standard name.
torch.save(best_state, f"{SAVE_DIR}/capec_multilabel_mlp.pt")

# Save test predictions.
test_output.to_csv(f"{SAVE_DIR}/test_predictions.csv", index=False)
test_output.to_csv(f"{SAVE_DIR}/topk_hit_results.csv", index=False)

# Save label info.
label_info = pd.DataFrame({
    "label": all_labels,
    "index": [label_to_idx[label] for label in all_labels],
    "train_count": train_label_counts.astype(int),
})
label_info.to_csv(f"{SAVE_DIR}/label_info.csv", index=False)

# Save metrics.
metrics_out = {
    "best_validation_metrics": best_state["val_metrics"],
    "test_metrics": test_metrics,
    "best_threshold": best_threshold,
    "top_k_hit_real_capec_rows": hit_results,
    "top_k_recall_real_capec_rows": recall_results,
    "config": {
        "keyword_method": KEYWORD_METHOD,
        "train_path": TRAIN_PATH,
        "val_path": VAL_PATH,
        "test_path": TEST_PATH,
        "capec_catalog_path": CAPEC_CATALOG_PATH,
        "desc_col": DESC_COL,
        "keyphrase_col": KEYPHRASE_COL,
        "capec_col": CAPEC_COL,
        "feature_representation": "description_and_mean_pooled_keyphrases",
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "embed_model_name": EMBED_MODEL_NAME,
        "random_state": RANDOM_STATE,
        "num_labels": NUM_LABELS,
        "input_dim": X_train.shape[1],
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
    },
}

with open(f"{SAVE_DIR}/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_out, f, indent=2)

print("\nSaved:")
print(f"{SAVE_DIR}/best_capec_multilabel_mlp.pt")
print(f"{SAVE_DIR}/capec_multilabel_mlp.pt")
print(f"{SAVE_DIR}/test_predictions.csv")
print(f"{SAVE_DIR}/topk_hit_results.csv")
print(f"{SAVE_DIR}/label_info.csv")
print(f"{SAVE_DIR}/metrics.json")
