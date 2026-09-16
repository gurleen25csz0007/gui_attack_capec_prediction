# ============================================================
# FINAL CVE -> MULTI-LABEL CAPEC CLASSIFICATION
# Uses:
# - fixed training/validation/testing files from stage 03
# - the GPU made visible by Slurm
# - MiniLM embeddings + MLP classifier
# - BCEWithLogitsLoss
# - threshold tuning on validation set
# - Top-K Hit@K and Recall@K on testing set
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
    label_ranking_average_precision_score
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer

from evaluation_paths import OUTPUT_ROOT


# ============================================================
# 1. SETTINGS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KEYWORD_DIR = os.path.abspath(os.path.join(
    SCRIPT_DIR, "../03_keyword_making/outputs/keybert_modified"
))
CAPEC_CATALOG_PATH = os.path.abspath(os.path.join(
    SCRIPT_DIR, "../01_cwe_capec_ground_truth/3000_capec.csv"
))

TRAIN_PATH = os.path.join(KEYWORD_DIR, "training_final.csv")
VAL_PATH = os.path.join(KEYWORD_DIR, "validation_final.csv")
TEST_PATH = os.path.join(KEYWORD_DIR, "testing_final.csv")

TEXT_COL = "cleaned_description"
FALLBACK_TEXT_COL = "uncleaned_description"

LABEL_COL = "capec_id"

NO_ID_LABEL = "CAPEC-noID"

RANDOM_STATE = 42

BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OUTPUT_DIR = os.path.join(str(OUTPUT_ROOT), "NN")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. GPU SELECTED BY SLURM
# ============================================================

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Visible CUDA GPUs:", torch.cuda.device_count())
print("Using device:", DEVICE)


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(RANDOM_STATE)


# ============================================================
# 4. CAPEC PARSING
# ============================================================

def parse_capec_cell(x):
    """
    Handles:
    - "['CAPEC-1', 'CAPEC-2']"
    - "CAPEC-1, CAPEC-2"
    - "CAPEC-1; CAPEC-2"
    - "CAPEC-1|CAPEC-2"
    - "1, 2, 3"
    - NaN / empty
    - CAPEC-noID
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

    # Case 1: Python list-like string
    try:
        parsed = ast.literal_eval(x)

        if isinstance(parsed, list):
            raw_items = parsed
        else:
            raw_items = [parsed]

        for item in raw_items:
            item = str(item).strip()

            item_low = item.lower().replace("_", "").replace(" ", "")
            if item_low in ["capec-noid", "noid", "nocapec", "capecnone"]:
                continue

            found = re.findall(r"CAPEC-\d+", item, flags=re.IGNORECASE)
            labels.extend([f.upper() for f in found])

            # Also handle plain numeric values like 123
            if re.fullmatch(r"\d+", item):
                labels.append(f"CAPEC-{int(item)}")

        return sorted(set(labels))

    except Exception:
        pass

    # Case 2: raw string with CAPEC-123
    found = re.findall(r"CAPEC-\d+", x, flags=re.IGNORECASE)
    labels.extend([f.upper() for f in found])

    # Case 3: raw numeric IDs separated by comma / semicolon / pipe / space
    if len(labels) == 0:
        nums = re.findall(r"\b\d+\b", x)
        labels.extend([f"CAPEC-{int(n)}" for n in nums])

    return sorted(set(labels))


def extract_capecs_from_catalog(path):
    """Load the complete canonical CAPEC vocabulary used by every variant."""
    frame = pd.read_csv(path)
    id_column = next(
        (column for column in ("ID", "id", "capec_id") if column in frame),
        None,
    )
    if id_column is None:
        raise ValueError(f"Cannot find the CAPEC ID column in {path}")
    capec_ids = set()
    for value in frame[id_column].dropna():
        text = str(value).strip()
        capec_ids.update(
            item.upper() for item in re.findall(r"CAPEC-\d+", text, flags=re.I)
        )
        if re.fullmatch(r"\d+(?:\.0)?", text):
            capec_ids.add(f"CAPEC-{int(float(text))}")
    return sorted(capec_ids)


# ============================================================
# 5. LOAD TRAIN / VALIDATION / TEST CSV
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print("\nTrain columns:")
print(train_df.columns)

print("\nValidation columns:")
print(val_df.columns)

print("\nTest columns:")
print(test_df.columns)

# Use cleaned_description if present, else uncleaned_description
if TEXT_COL not in train_df.columns:
    print(f"\nWarning: {TEXT_COL} not found in train. Using {FALLBACK_TEXT_COL}.")
    TEXT_COL = FALLBACK_TEXT_COL

for split_name, split_df in [
    ("training CSV", train_df),
    ("validation CSV", val_df),
    ("testing CSV", test_df)
]:
    if TEXT_COL not in split_df.columns:
        raise ValueError(f"{TEXT_COL} not found in {split_name}")

    if LABEL_COL not in split_df.columns:
        raise ValueError(f"{LABEL_COL} not found in {split_name}")

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df[TEXT_COL] = train_df[TEXT_COL].fillna("").astype(str)
val_df[TEXT_COL] = val_df[TEXT_COL].fillna("").astype(str)
test_df[TEXT_COL] = test_df[TEXT_COL].fillna("").astype(str)

train_df = train_df[train_df[TEXT_COL].str.strip() != ""].reset_index(drop=True)
val_df = val_df[val_df[TEXT_COL].str.strip() != ""].reset_index(drop=True)
test_df = test_df[test_df[TEXT_COL].str.strip() != ""].reset_index(drop=True)

print("\nRows after removing empty text:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

# ============================================================
# 6. PARSE LABELS
# ============================================================

for split_df in [train_df, val_df, test_df]:
    split_df["capec_real_list"] = split_df[LABEL_COL].apply(parse_capec_cell)

    split_df["capec_label_list"] = split_df["capec_real_list"].apply(
        lambda labels: labels if len(labels) > 0 else [NO_ID_LABEL]
    )

print("\nLabel parsing:")
print("Train rows with real CAPEC:", (train_df["capec_real_list"].apply(len) > 0).sum())
print("Train rows with CAPEC-noID:", (train_df["capec_real_list"].apply(len) == 0).sum())

print("Validation rows with real CAPEC:", (val_df["capec_real_list"].apply(len) > 0).sum())
print("Validation rows with CAPEC-noID:", (val_df["capec_real_list"].apply(len) == 0).sum())

print("Test rows with real CAPEC:", (test_df["capec_real_list"].apply(len) > 0).sum())
print("Test rows with CAPEC-noID:", (test_df["capec_real_list"].apply(len) == 0).sum())
# ============================================================
# 7. BUILD LABEL SPACE
# ============================================================
# Uses union of train + validation + test labels so evaluation does not crash.

train_labels_set = set(
    lab
    for labels in train_df["capec_label_list"]
    for lab in labels
)

val_labels_set = set(
    lab
    for labels in val_df["capec_label_list"]
    for lab in labels
)

test_labels_set = set(
    lab
    for labels in test_df["capec_label_list"]
    for lab in labels
)

catalog_labels_set = set(extract_capecs_from_catalog(CAPEC_CATALOG_PATH))
all_real_capecs = sorted(
    catalog_labels_set
    | {
        lab for lab in (train_labels_set | val_labels_set | test_labels_set)
        if lab != NO_ID_LABEL
    }
)

all_labels = all_real_capecs + [NO_ID_LABEL]

label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

NUM_LABELS = len(all_labels)
NO_ID_INDEX = label_to_idx[NO_ID_LABEL]

print("\nLabel space:")
print("Real CAPEC labels:", len(all_real_capecs))
print("Canonical catalog CAPEC labels:", len(catalog_labels_set))
print("Total labels including CAPEC-noID:", NUM_LABELS)

val_only_labels = sorted(val_labels_set - train_labels_set)
test_only_labels = sorted(test_labels_set - train_labels_set)

print("Labels present in validation but not train:", len(val_only_labels))
print("Labels present in test but not train:", len(test_only_labels))

if len(val_only_labels) > 0:
    print("Example val-only labels:", val_only_labels[:20])

if len(test_only_labels) > 0:
    print("Example test-only labels:", test_only_labels[:20])
# ============================================================
# 8. MAKE MULTI-HOT FUNCTION
# ============================================================

def make_multihot(label_lists, label_to_idx):
    """
    Converts list of CAPEC labels into multi-hot vectors.

    Example:
    all_labels = ["CAPEC-1", "CAPEC-2", "CAPEC-3"]
    row labels = ["CAPEC-1", "CAPEC-3"]
    output     = [1, 0, 1]
    """

    y = np.zeros((len(label_lists), len(label_to_idx)), dtype=np.float32)

    for i, labels in enumerate(label_lists):
        valid_labels = [lab for lab in labels if lab in label_to_idx]

        if len(valid_labels) == 0:
            valid_labels = [NO_ID_LABEL]

        for lab in valid_labels:
            y[i, label_to_idx[lab]] = 1.0

    return y
# ============================================================
# 9. MAKE MULTI-HOT MATRICES
# ============================================================

Y_train = make_multihot(train_df["capec_label_list"].tolist(), label_to_idx)
Y_val = make_multihot(val_df["capec_label_list"].tolist(), label_to_idx)
Y_test = make_multihot(test_df["capec_label_list"].tolist(), label_to_idx)

print("\nY shapes:")
print("Y_train:", Y_train.shape)
print("Y_val:", Y_val.shape)
print("Y_test:", Y_test.shape)

print("\nFinal split:")
print("Train:", len(train_df), Y_train.shape)
print("Validation:", len(val_df), Y_val.shape)
print("Test:", len(test_df), Y_test.shape)

# ============================================================
# 10. LABEL FREQUENCY REPORT
# ============================================================

train_label_counts = Y_train.sum(axis=0)

zero_train_labels = [
    all_labels[i]
    for i, count in enumerate(train_label_counts)
    if count == 0
]

print("\nTraining label frequency report:")
print("Labels with 0 train examples:", len(zero_train_labels))
print("CAPEC-noID train count:", int(train_label_counts[NO_ID_INDEX]))


# ============================================================
# 11. ENCODE TEXT USING SENTENCE TRANSFORMER
# ============================================================

embedder = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

def encode_texts(texts, batch_size=256, name="texts"):
    print(f"\nEncoding {name}: {len(texts)} rows")

    emb = embedder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return emb.astype(np.float32)


X_train = encode_texts(train_df[TEXT_COL].tolist(), batch_size=BATCH_SIZE, name="train")
X_val = encode_texts(val_df[TEXT_COL].tolist(), batch_size=BATCH_SIZE, name="validation")
X_test = encode_texts(test_df[TEXT_COL].tolist(), batch_size=BATCH_SIZE, name="test")
print("\nEmbedding shapes:")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)


# ============================================================
# 12. DATASET / DATALOADER
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
    shuffle=True
)

val_loader = DataLoader(
    EmbeddingDataset(X_val, Y_val),
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    EmbeddingDataset(X_test, Y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 13. MODEL
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

            nn.Linear(512, num_labels)
        )

    def forward(self, x):
        return self.net(x)


model = MultiLabelMLP(
    input_dim=X_train.shape[1],
    num_labels=NUM_LABELS
).to(DEVICE)


# ============================================================
# 14. LOSS
# ============================================================

pos_counts = Y_train.sum(axis=0)
neg_counts = len(Y_train) - pos_counts

pos_weight = np.ones(NUM_LABELS, dtype=np.float32)

nonzero_mask = pos_counts > 0
pos_weight[nonzero_mask] = neg_counts[nonzero_mask] / pos_counts[nonzero_mask]

# Avoid huge unstable weights for very rare labels
pos_weight = np.clip(pos_weight, 1.0, 50.0)

pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 15. TRAINING / PREDICTION FUNCTIONS
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for X_batch, Y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        Y_batch = Y_batch.to(DEVICE)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, Y_batch)

        loss.backward()
        optimizer.step()

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
    pred = (probs >= threshold).astype(int)

    for i in range(pred.shape[0]):
        positive_indices = np.where(pred[i] == 1)[0].tolist()

        if len(positive_indices) == 0:
            best_idx = int(np.argmax(probs[i]))
            pred[i, best_idx] = 1
            continue

        real_positive_indices = [
            idx for idx in positive_indices
            if idx != no_id_index
        ]

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
        "sample_jaccard": float(np.mean(jaccards))
    }


def evaluate_multilabel(y_true, probs, threshold, no_id_index):
    y_pred = postprocess_predictions(probs, threshold, no_id_index)

    sample_metrics = sample_average_metrics(y_true, y_pred)

    metrics = {
        "threshold": threshold,

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

        "hamming_loss": hamming_loss(y_true, y_pred)
    }

    try:
        metrics["label_ranking_average_precision"] = label_ranking_average_precision_score(y_true, probs)
    except Exception:
        metrics["label_ranking_average_precision"] = None

    return metrics, y_pred


def find_best_threshold(y_true, probs, no_id_index):
    best_score = -1
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
            no_id_index=no_id_index
        )

        score = metrics["micro_f1"]

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def print_metrics(metrics, title="Metrics"):
    print(f"\n{title}")
    print("-" * len(title))

    for k, v in metrics.items():
        if v is None:
            print(f"{k}: None")
        else:
            print(f"{k}: {v:.4f}")


# ============================================================
# 16. TRAIN LOOP WITH EARLY STOPPING
# ============================================================

best_model_path = os.path.join(OUTPUT_DIR, "best_capec_multilabel_mlp.pt")

best_val_micro_f1 = -1
best_state = None
patience_counter = 0
best_threshold = 0.80

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)

    val_probs, val_true = predict_probs(model, val_loader)

    epoch_best_threshold, val_metrics = find_best_threshold(
        y_true=val_true,
        probs=val_probs,
        no_id_index=NO_ID_INDEX
    )

    val_micro_f1 = val_metrics["micro_f1"]

    print(f"\nEpoch {epoch}/{EPOCHS}")
    print("Train loss:", round(train_loss, 4))
    print("Best val threshold:", epoch_best_threshold)
    print("Val micro F1:", round(val_micro_f1, 4))
    print("Val sample F1:", round(val_metrics["sample_f1"], 4))
    print("Val exact match:", round(val_metrics["exact_match_accuracy"], 4))

    if val_micro_f1 > best_val_micro_f1:
        best_val_micro_f1 = val_micro_f1
        best_threshold = epoch_best_threshold

        best_state = {
            "model_state_dict": model.state_dict(),
            "threshold": best_threshold,
            "val_metrics": val_metrics,
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "all_labels": all_labels,
            "no_id_label": NO_ID_LABEL,
            "text_col": TEXT_COL,
            "label_col": LABEL_COL,
            "embed_model_name": EMBED_MODEL_NAME,
            "input_dim": X_train.shape[1],
            "epoch": epoch,
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
    raise RuntimeError("No best model was saved. Check validation data or training loop.")

# ============================================================
# 17. FINAL TEST EVALUATION
# ============================================================

model.load_state_dict(best_state["model_state_dict"])

test_probs, test_true = predict_probs(model, test_loader)

test_metrics, test_pred = evaluate_multilabel(
    y_true=test_true,
    probs=test_probs,
    threshold=best_threshold,
    no_id_index=NO_ID_INDEX
)

print_metrics(best_state["val_metrics"], title="Best Validation Metrics")
print_metrics(test_metrics, title="Final Test Metrics")

print("\nBest threshold:", best_threshold)


# ============================================================
# 18. TOP-K HIT AND RECALL METRICS
# ============================================================

def top_k_hit_metrics(y_true, probs, ks=[1, 2, 3, 5, 10], no_id_index=None, exclude_noid=True):
    results = {}

    for k in ks:
        hits = []
        evaluated_rows = 0

        for true_row, prob_row in zip(y_true, probs):
            true_set = set(np.where(true_row == 1)[0])

            if exclude_noid and no_id_index is not None:
                true_set.discard(no_id_index)

            if len(true_set) == 0:
                continue

            sorted_indices = np.argsort(prob_row)[::-1].tolist()

            if exclude_noid and no_id_index is not None:
                sorted_indices = [
                    idx for idx in sorted_indices
                    if idx != no_id_index
                ]

            topk_set = set(sorted_indices[:k])

            hit = 1 if len(true_set & topk_set) > 0 else 0
            hits.append(hit)
            evaluated_rows += 1

        results[f"hit@{k}"] = float(np.mean(hits)) if len(hits) > 0 else 0.0
        results[f"evaluated_rows@{k}"] = evaluated_rows

    return results


def top_k_recall_metrics(y_true, probs, ks=[1, 2, 3, 5, 10], no_id_index=None, exclude_noid=True):
    results = {}

    for k in ks:
        recalls = []
        evaluated_rows = 0

        for true_row, prob_row in zip(y_true, probs):
            true_set = set(np.where(true_row == 1)[0])

            if exclude_noid and no_id_index is not None:
                true_set.discard(no_id_index)

            if len(true_set) == 0:
                continue

            sorted_indices = np.argsort(prob_row)[::-1].tolist()

            if exclude_noid and no_id_index is not None:
                sorted_indices = [
                    idx for idx in sorted_indices
                    if idx != no_id_index
                ]

            topk_set = set(sorted_indices[:k])

            recall = len(true_set & topk_set) / len(true_set)
            recalls.append(recall)
            evaluated_rows += 1

        results[f"recall@{k}"] = float(np.mean(recalls)) if len(recalls) > 0 else 0.0
        results[f"evaluated_rows@{k}"] = evaluated_rows

    return results


KS = [1, 2, 3, 5, 10, 20, 30, 50]

hit_results = top_k_hit_metrics(
    y_true=test_true,
    probs=test_probs,
    ks=KS,
    no_id_index=NO_ID_INDEX,
    exclude_noid=True
)

recall_results = top_k_recall_metrics(
    y_true=test_true,
    probs=test_probs,
    ks=KS,
    no_id_index=NO_ID_INDEX,
    exclude_noid=True
)

print("\nTop-K Hit Results on Real CAPEC Rows")
print("------------------------------------")
for k in KS:
    print(f"Hit@{k}: {hit_results[f'hit@{k}']:.4f}")

print("\nTop-K Recall Results on Real CAPEC Rows")
print("---------------------------------------")
for k in KS:
    print(f"Recall@{k}: {recall_results[f'recall@{k}']:.4f}")

print("\nEvaluated real-CAPEC rows:", hit_results["evaluated_rows@1"])


# ============================================================
# 19. SAVE PER-ROW PREDICTIONS
# ============================================================

def indices_to_labels(row):
    return [idx_to_label[i] for i in np.where(row == 1)[0]]


def get_top_k_labels(prob_row, k=10, no_id_index=None, exclude_noid=True):
    sorted_indices = np.argsort(prob_row)[::-1].tolist()

    if exclude_noid and no_id_index is not None:
        sorted_indices = [
            idx for idx in sorted_indices
            if idx != no_id_index
        ]

    top_indices = sorted_indices[:k]

    return [
        (idx_to_label[idx], float(prob_row[idx]))
        for idx in top_indices
    ]


def true_labels_from_row(true_row, no_id_index=None, exclude_noid=True):
    true_indices = np.where(true_row == 1)[0].tolist()

    if exclude_noid and no_id_index is not None:
        true_indices = [
            idx for idx in true_indices
            if idx != no_id_index
        ]

    return [idx_to_label[idx] for idx in true_indices]


test_output = test_df.copy()

test_output["true_capec_labels"] = [
    true_labels_from_row(row, NO_ID_INDEX, exclude_noid=False)
    for row in test_true
]

test_output["pred_capec_labels"] = [
    indices_to_labels(row)
    for row in test_pred
]

test_output["top_1_predictions"] = [
    get_top_k_labels(row, k=1, no_id_index=NO_ID_INDEX, exclude_noid=True)
    for row in test_probs
]

test_output["top_3_predictions"] = [
    get_top_k_labels(row, k=3, no_id_index=NO_ID_INDEX, exclude_noid=True)
    for row in test_probs
]

test_output["top_5_predictions"] = [
    get_top_k_labels(row, k=5, no_id_index=NO_ID_INDEX, exclude_noid=True)
    for row in test_probs
]

test_output["top_10_predictions"] = [
    get_top_k_labels(row, k=10, no_id_index=NO_ID_INDEX, exclude_noid=True)
    for row in test_probs
]


def row_hit(true_row, prob_row, k):
    true_set = set(np.where(true_row == 1)[0])
    true_set.discard(NO_ID_INDEX)

    if len(true_set) == 0:
        return None

    sorted_indices = np.argsort(prob_row)[::-1].tolist()
    sorted_indices = [
        idx for idx in sorted_indices
        if idx != NO_ID_INDEX
    ]

    topk_set = set(sorted_indices[:k])

    return int(len(true_set & topk_set) > 0)


for k in [1, 2, 3, 5, 10]:
    test_output[f"hit_at_{k}"] = [
        row_hit(test_true[i], test_probs[i], k)
        for i in range(len(test_df))
    ]


# ============================================================
# 20. SAVE EVERYTHING
# ============================================================

model_path = os.path.join(OUTPUT_DIR, "capec_multilabel_mlp.pt")
pred_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
topk_path = os.path.join(OUTPUT_DIR, "topk_hit_results.csv")
label_info_path = os.path.join(OUTPUT_DIR, "label_info.csv")
metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")

torch.save(best_state, model_path)

test_output.to_csv(pred_path, index=False)
test_output.to_csv(topk_path, index=False)

label_info = pd.DataFrame({
    "label": all_labels,
    "index": [label_to_idx[label] for label in all_labels],
    "train_count": train_label_counts
})

label_info.to_csv(label_info_path, index=False)

metrics_to_save = {
    "best_validation_metrics": best_state["val_metrics"],
    "test_metrics": test_metrics,
    "hit_results": hit_results,
    "recall_results": recall_results,
    "best_threshold": best_threshold,
    "device": DEVICE,
    "train_rows": int(len(train_df)),
    "val_rows": int(len(val_df)),
    "test_rows": int(len(test_df)),
    "num_labels": int(NUM_LABELS),
    "zero_train_labels": zero_train_labels
}

with open(metrics_path, "w") as f:
    json.dump(metrics_to_save, f, indent=4)

print("\nSaved:")
print(model_path)
print(pred_path)
print(topk_path)
print(label_info_path)
print(metrics_path)

print("\nDone.")
