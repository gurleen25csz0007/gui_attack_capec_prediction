#!/usr/bin/env python3
"""Collect the completed Stage-04 model metrics into one comparison CSV."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation_paths import OUTPUT_ROOT


ROOT = Path(__file__).resolve().parent
OUTPUTS = OUTPUT_ROOT


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(
    model: str,
    target: str,
    input_case: str,
    training: str,
    validation: dict[str, Any],
    test: dict[str, Any],
    source: Path,
) -> dict[str, Any]:
    return {
        "model": model,
        "target": target,
        "input": input_case,
        "training": training,
        "validation_micro_f1": validation.get("micro_f1"),
        "validation_macro_f1": validation.get("macro_f1"),
        "test_micro_precision": test.get("micro_precision"),
        "test_micro_recall": test.get("micro_recall"),
        "test_micro_f1": test.get("micro_f1"),
        "test_macro_f1": test.get("macro_f1"),
        "test_weighted_f1": test.get("weighted_f1"),
        "test_sample_f1": test.get("sample_f1"),
        "test_exact_match_accuracy": test.get("exact_match_accuracy"),
        "test_hamming_loss": test.get("hamming_loss"),
        "hit@1": test.get("hit@1"),
        "hit@5": test.get("hit@5"),
        "hit@10": test.get("hit@10"),
        "source": str(source),
    }


def nn_row(model: str, target: str, input_case: str, training: str, path: Path) -> dict[str, Any]:
    payload = read_json(path)
    validation = payload.get("best_validation_metrics", payload.get("validation_metrics", {}))
    test = payload["test_metrics"]
    hits = payload.get("hit_results", payload.get("top_k_hit_real_capec_rows", payload.get("top_k_hit_real_cwe_rows", payload.get("top_k_metrics", {}))))
    test = {**test, **{key: value for key, value in hits.items() if key.startswith("hit@")}}
    return metric_row(model, target, input_case, training, validation, test, path)


def pointer_run(pointer_name: str) -> Path:
    pointer = OUTPUTS / "combined" / pointer_name
    if not pointer.is_file():
        raise FileNotFoundError(f"Missing completed-run pointer: {pointer}")
    value = pointer.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError(f"Completed-run pointer is empty: {pointer}")
    run = Path(value).expanduser().resolve()
    if not run.is_dir():
        raise FileNotFoundError(
            f"Completed-run pointer {pointer} targets missing directory: {run}"
        )
    return run


def rae_run_is_complete(run_dir: Path) -> bool:
    """Reject an RAE directory while a new in-place run is still writing it."""
    artifact_names = (
        "best_validation_metrics.json",
        "test_metrics.json",
        "model_config.json",
        "checkpoint_reload_test.json",
    )
    artifacts = [run_dir / name for name in artifact_names]
    if not all(path.is_file() for path in artifacts):
        return False
    # checkpoint_reload_test.json is deliberately the final artifact written by
    # rae_xmc.py. If training has replaced an earlier validation/checkpoint file
    # but not yet reached final test evaluation, the old completion marker is
    # older and the collector must not mix metrics from the two executions.
    completed_at = (run_dir / "checkpoint_reload_test.json").stat().st_mtime_ns
    return all(path.stat().st_mtime_ns <= completed_at for path in artifacts)


def main() -> None:
    specifications = (
        ("NN", "CAPEC", "cleaned_description", "frozen", OUTPUTS / "NN" / "metrics.json"),
        ("NN_keywords", "CAPEC", "cleaned_description+keywords", "frozen", OUTPUTS / "NN_keywords" / "metrics.json"),
        ("NN_fine_tune", "CAPEC", "cleaned_description", "fine_tuned", OUTPUTS / "NN_fine_tune" / "metrics.json"),
        ("NN_fine_tune_keywords", "CAPEC", "cleaned_description+keywords", "fine_tuned", OUTPUTS / "NN_fine_tune_keywords" / "metrics.json"),
        ("qwen_nn", "CAPEC", "cleaned_description+qwen_keywords", "frozen", OUTPUTS / "qwen_nn" / "metrics.json"),
        ("qwen_nnfinetuned", "CAPEC", "cleaned_description+qwen_keywords", "fine_tuned", OUTPUTS / "qwen_nnfinetuned" / "metrics.json"),
        ("NN_1way", "CWE", "cleaned_description", "frozen", OUTPUTS / "CVE_to_CWE" / "NN_1way" / "metrics.json"),
        ("NN_keywords", "CWE", "cleaned_description+keywords", "frozen", OUTPUTS / "CVE_to_CWE" / "NN_keywords" / "metrics.json"),
        ("NN_fine_tune", "CWE", "cleaned_description", "fine_tuned", OUTPUTS / "CVE_to_CWE" / "NN_fine_tune" / "metrics.json"),
        ("NN_fine_tune_keywords", "CWE", "cleaned_description+keywords", "fine_tuned", OUTPUTS / "CVE_to_CWE" / "NN_fine_tune_keywords" / "metrics.json"),
    )
    rows = [nn_row(*specification) for specification in specifications]

    for model_name, directory, input_case in (
        ("RAE_XMC", "RAE_XMC", "cleaned_description"),
        (
            "RAE_XMC_keywords", "RAE_XMC_keywords",
            "cleaned_description+keywords",
        ),
    ):
        rae_test_path = OUTPUTS / directory / "test_metrics.json"
        rae_validation_path = OUTPUTS / directory / "best_validation_metrics.json"
        rae_run_dir = OUTPUTS / directory
        if rae_run_is_complete(rae_run_dir):
            rows.append(metric_row(
                model_name, "CAPEC", input_case, "fine_tuned_retrieval",
                read_json(rae_validation_path), read_json(rae_test_path),
                rae_test_path,
            ))
        else:
            print(
                f"Skipping {model_name}: its output directory does not contain "
                "one complete, internally consistent run."
            )
    for name, pointer, input_case in (
        ("CGDPF", "latest_CGDPF_run.txt", "cleaned_description"),
        ("CGDPF_keywords", "latest_CGDPF_keywords_run.txt", "cleaned_description+keywords"),
    ):
        run = pointer_run(pointer)
        row = metric_row(
            name, "CAPEC", input_case, "score_fusion",
            read_json(run / "validation_metrics.json"),
            read_json(run / "test_metrics.json"), run,
        )
        parameters = read_json(run / "fusion_parameters.json")
        row.update({
            "selected_alpha": parameters.get("selected_alpha"),
            "selected_beta": None,
            "selected_threshold": parameters.get("selected_threshold"),
        })
        rows.append(row)

    for name, pointer, input_case in (
        (
            "our_approach",
            "latest_our_approach_run.txt",
            "cleaned_description",
        ),
        (
            "our_approach_keywords",
            "latest_our_approach_keywords_run.txt",
            "cleaned_description+keywords",
        ),
    ):
        final_run = pointer_run(pointer)
        final_validation = read_json(
            final_run / "validation_metrics.json"
        )["hybrid_plus_rae_xmc"]
        final_test = read_json(
            final_run / "test_metrics.json"
        )["hybrid_plus_rae_xmc"]
        row = metric_row(
            name, "CAPEC", input_case, "score_fusion",
            final_validation, final_test, final_run,
        )
        parameters = read_json(final_run / "fusion_parameters.json")
        row.update({
            "selected_alpha": parameters.get("selected_alpha_old"),
            "selected_beta": parameters.get("selected_beta"),
            "selected_threshold": parameters.get("selected_final_threshold"),
        })
        rows.append(row)

    # The calibrated RAE and original-hybrid branches stored inside an Our
    # Approach run are diagnostics/ablations, not additional final models.
    # Publish exactly the description and keyword variants of Our Approach.
    expected_our_approaches = {"our_approach", "our_approach_keywords"}
    published_our_approaches = [
        row["model"] for row in rows
        if str(row["model"]).startswith("our_approach")
    ]
    if (
        set(published_our_approaches) != expected_our_approaches
        or len(published_our_approaches) != len(expected_our_approaches)
    ):
        raise RuntimeError(
            "Expected exactly two Our Approach summary rows "
            f"{sorted(expected_our_approaches)}, got "
            f"{published_our_approaches}."
        )

    frame = pd.DataFrame(rows)
    duplicate_keys = frame.duplicated(
        subset=["model", "target", "input"], keep=False
    )
    if duplicate_keys.any():
        duplicates = frame.loc[
            duplicate_keys, ["model", "target", "input", "source"]
        ].to_dict(orient="records")
        raise RuntimeError(f"Duplicate model summary rows: {duplicates}")
    frame["validation_rank"] = (
        frame.groupby("target")["validation_micro_f1"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    frame["selected_by_validation"] = frame["validation_rank"].eq(1)
    frame = frame.sort_values(
        ["target", "validation_rank", "test_micro_f1"],
        ascending=[True, True, False],
    )
    frame.to_csv(OUTPUTS / "model_results_summary.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
