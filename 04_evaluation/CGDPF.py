#!/usr/bin/env python3
"""CGDPF hybrid: direct CAPEC prediction plus CWE-to-CAPEC propagation.

It supports two controlled cases: cleaned description only, and cleaned
description plus the selected KeyBERT keywords. Calibration, alpha, and the
decision threshold are selected only on validation data. Test labels are used
once for final metrics.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation_data import DATASET_PATH, load_fixed_splits
from evaluation_paths import OUTPUT_ROOT
from our_approch import (
    CAPEC_MODEL_DIR,
    CWE_MODEL_DIR,
    DESC_COL,
    GROUND_TRUTH_DIR,
    NO_ID_LABEL,
    RUNS_DIR,
    RunConfig,
    align_scores,
    apply_platt,
    canonical_capec_vocabulary,
    checkpoint_path,
    compute_mapping_scores,
    cve_ids,
    evaluate,
    fit_platt,
    id_digest,
    labels_from_checkpoint,
    load_cwe_mapping,
    load_stage04_nn,
    make_truth,
    mapping_path,
    predict_nn_scores,
    safe_torch_load,
    save_json,
    save_prediction_tables,
    search_alpha,
    seed_everything,
    validate_complete_capec_vocabulary,
)


SCRIPT_DIR = Path(__file__).resolve().parent
KEYPHRASE_COL = "keyphrases"


def run_pipeline(use_keywords: bool = False) -> Path:
    started = time.time()
    config = RunConfig()
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    case_name = "CGDPF_keywords" if use_keywords else "CGDPF"
    cwe_model_dir = (
        OUTPUT_ROOT / "CVE_to_CWE" /
        ("NN_keywords" if use_keywords else "NN_1way")
    )
    capec_model_dir = (
        OUTPUT_ROOT /
        ("NN_keywords" if use_keywords else "NN")
    )
    run_dir = RUNS_DIR / f"{case_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)

    train, validation, test = load_fixed_splits(DATASET_PATH, config.seed)
    validation_ids, test_ids = cve_ids(validation), cve_ids(test)
    save_json(run_dir / "validation_cve_ids.json", validation_ids)
    save_json(run_dir / "test_cve_ids.json", test_ids)

    cwe_checkpoint = safe_torch_load(checkpoint_path(cwe_model_dir, "cwe"), "cpu")
    capec_checkpoint = safe_torch_load(checkpoint_path(capec_model_dir, "capec"), "cpu")
    cwe_labels = labels_from_checkpoint(cwe_checkpoint, cwe_model_dir, "cwe")
    direct_labels = labels_from_checkpoint(capec_checkpoint, capec_model_dir, "capec")
    if NO_ID_LABEL not in direct_labels:
        raise ValueError(f"CAPEC vocabulary does not contain {NO_ID_LABEL}.")
    source_capec_vocabulary = validate_complete_capec_vocabulary(direct_labels)
    labels = canonical_capec_vocabulary()
    capec_vocabulary = validate_complete_capec_vocabulary(labels)
    no_id = labels.index(NO_ID_LABEL)

    y_train, unknown_train = make_truth(train, labels)
    y_val, unknown_val = make_truth(validation, labels)
    y_test, unknown_test = make_truth(test, labels)

    print("Device:", device)
    print("Run directory:", run_dir)
    print("Generating validation/test CWE scores...")
    cwe_model, cwe_embedder, loaded_cwe_labels, cwe_input_dim = load_stage04_nn(
        cwe_model_dir, "cwe", device
    )
    if loaded_cwe_labels != cwe_labels:
        raise RuntimeError("CWE label order changed while loading the checkpoint.")
    val_cwe = predict_nn_scores(
        cwe_model, cwe_embedder, validation[DESC_COL].tolist(),
        cwe_input_dim, device, config.batch_size,
        validation[KEYPHRASE_COL].tolist() if use_keywords else None,
    )
    test_cwe = predict_nn_scores(
        cwe_model, cwe_embedder, test[DESC_COL].tolist(),
        cwe_input_dim, device, config.batch_size,
        test[KEYPHRASE_COL].tolist() if use_keywords else None,
    )
    del cwe_model, cwe_embedder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Generating validation/test direct CAPEC scores...")
    capec_model, capec_embedder, loaded_capec_labels, capec_input_dim = load_stage04_nn(
        capec_model_dir, "capec", device
    )
    if loaded_capec_labels != direct_labels:
        raise RuntimeError("CAPEC label order changed while loading the checkpoint.")
    val_direct_native = predict_nn_scores(
        capec_model, capec_embedder, validation[DESC_COL].tolist(),
        capec_input_dim, device, config.batch_size,
        validation[KEYPHRASE_COL].tolist() if use_keywords else None,
    )
    test_direct_native = predict_nn_scores(
        capec_model, capec_embedder, test[DESC_COL].tolist(),
        capec_input_dim, device, config.batch_size,
        test[KEYPHRASE_COL].tolist() if use_keywords else None,
    )
    del capec_model, capec_embedder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    val_direct, direct_alignment = align_scores(
        val_direct_native, direct_labels, labels
    )
    test_direct, _ = align_scores(test_direct_native, direct_labels, labels)
    map_path = mapping_path()
    mapping = load_cwe_mapping(map_path)
    val_mapping = compute_mapping_scores(
        val_cwe, cwe_labels, labels, mapping,
        config.cwe_top_k_for_mapping, config.use_rank_weight,
    )
    test_mapping = compute_mapping_scores(
        test_cwe, cwe_labels, labels, mapping,
        config.cwe_top_k_for_mapping, config.use_rank_weight,
    )

    for filename, matrix in {
        "validation_cwe_scores.npy": val_cwe,
        "test_cwe_scores.npy": test_cwe,
        "validation_direct_scores.npy": val_direct,
        "test_direct_scores.npy": test_direct,
        "validation_mapping_scores.npy": val_mapping,
        "test_mapping_scores.npy": test_mapping,
    }.items():
        np.save(run_dir / filename, np.asarray(matrix, dtype=np.float32))

    calibrations = {
        "direct": fit_platt(val_direct, y_val, config.seed + 1, config.calibration_negative_ratio),
        "mapping": fit_platt(val_mapping, y_val, config.seed + 2, config.calibration_negative_ratio),
    }
    val_direct_cal = apply_platt(val_direct, calibrations["direct"])
    test_direct_cal = apply_platt(test_direct, calibrations["direct"])
    val_mapping_cal = apply_platt(val_mapping, calibrations["mapping"])
    test_mapping_cal = apply_platt(test_mapping, calibrations["mapping"])

    alpha, threshold, val_hybrid, search_rows = search_alpha(
        val_direct_cal, val_mapping_cal, y_val, no_id, config
    )
    test_hybrid = (
        alpha * test_direct_cal + (1.0 - alpha) * test_mapping_cal
    ).astype(np.float32)
    np.save(run_dir / "validation_CGDPF_scores.npy", val_hybrid)
    np.save(run_dir / "test_CGDPF_scores.npy", test_hybrid)
    calibration_frame = pd.DataFrame(search_rows)
    calibration_frame["selected"] = (
        np.isclose(calibration_frame["alpha_old"], alpha)
        & np.isclose(calibration_frame["threshold"], threshold)
    )
    calibration_frame.to_csv(
        run_dir / "fusion_calibration_results.csv", index=False
    )
    # Backward-compatible filename used by earlier analysis notebooks.
    calibration_frame.to_csv(
        run_dir / "validation_search_results.csv", index=False
    )

    validation_metrics, _ = evaluate(y_val, val_hybrid, threshold, no_id)
    test_metrics, test_prediction = evaluate(y_test, test_hybrid, threshold, no_id)
    save_json(run_dir / "validation_metrics.json", validation_metrics)
    save_json(run_dir / "test_metrics.json", test_metrics)
    save_json(run_dir / "calibration_parameters.json", {
        "selection_split": "validation",
        "branches": calibrations,
        "test_labels_used_for_selection": False,
    })
    save_json(run_dir / "fusion_parameters.json", {
        "selected_alpha": alpha,
        "selected_threshold": threshold,
        "alpha_definition": "weight on calibrated direct CAPEC scores",
        "mapping_definition": "maximum support from the top-10 CWE predictions",
    })
    save_json(run_dir / "label_to_idx.json", {label: i for i, label in enumerate(labels)})
    save_json(run_dir / "idx_to_label.json", {str(i): label for i, label in enumerate(labels)})
    save_json(run_dir / "label_alignment_report.json", {
        "direct": direct_alignment,
        "unknown_truth_labels": {
            "train": unknown_train, "validation": unknown_val, "test": unknown_test,
        },
    })
    save_prediction_tables(
        run_dir, test, y_test, test_prediction, test_hybrid, labels, no_id
    )
    save_json(run_dir / "run_config.json", {
        "config": asdict(config),
        "source_data": {
            "split_directory": str(DATASET_PATH),
            "ground_truth_directory": str(GROUND_TRUTH_DIR),
            "cwe_mapping": str(map_path),
        },
        "source_models": {
            "cwe": str(cwe_model_dir),
            "direct_capec": str(capec_model_dir),
        },
        "input_case": (
            "cleaned_description+keybert_modified_keywords"
            if use_keywords else "cleaned_description"
        ),
        "capec_vocabulary": capec_vocabulary,
        "source_capec_vocabulary": source_capec_vocabulary,
        "split": {
            "seed": config.seed,
            "training_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "validation_cve_id_digest": id_digest(validation_ids),
            "test_cve_id_digest": id_digest(test_ids),
        },
        "processing_seconds": time.time() - started,
    })
    pointer_name = (
        "latest_CGDPF_keywords_run.txt"
        if use_keywords else "latest_CGDPF_run.txt"
    )
    (RUNS_DIR / pointer_name).write_text(
        str(run_dir.resolve()) + "\n", encoding="utf-8"
    )

    print("Selected alpha:", alpha)
    print("Selected validation threshold:", threshold)
    print("Final test Micro-F1:", test_metrics["micro_f1"])
    print("Saved results to:", run_dir)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the validation-selected CGDPF hybrid"
    )
    parser.add_argument(
        "--keywords",
        action="store_true",
        help="Use the NN_keywords CAPEC/CWE checkpoints and KeyBERT features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(use_keywords=args.keywords)


if __name__ == "__main__":
    main()
