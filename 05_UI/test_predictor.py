"""Regression tests for exact CVE lookup and keyword-aware model routing."""

from __future__ import annotations

import json
import csv
import tempfile
import threading
import unittest
from pathlib import Path

from predictor import (
    CAPECPredictor,
    _index_descriptions,
    _load_dataset_index,
    _load_split_index,
    _validate_our_approach_run,
)


class _FakePreprocessor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def clean(self, text: str) -> str:
        self.calls.append(text)
        return "fresh cleaned description"


class _FakeKeyBERT:
    def __init__(self) -> None:
        self.calls = []

    def keybert_modified_keywords(self, document, top_k, device):
        self.calls.append((document, top_k, device))
        return "fresh key phrase; second fresh phrase"


class _FakeOurApproach:
    device = "cpu"
    use_keywords = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str] | None, int]] = []

    def predict(
        self, description: str, keywords: list[str] | None, top_k: int
    ) -> dict:
        self.calls.append((description, keywords, top_k))
        return {
            "ranked_capecs": [{"capec_id": "CAPEC-1", "score": 0.9}],
            "thresholded_labels": ["CAPEC-1"],
            "threshold": 0.8,
            "alpha": 0.4,
            "beta": 0.6,
            "input_case": "cleaned_description+keybert_modified_keywords",
            "keywords_used": True,
            "score_type": "validation-calibrated fusion",
        }


class PredictorPipelineTests(unittest.TestCase):
    def test_empty_dataset_reports_connection_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.csv"
            dataset.write_text("cve_id,uncleaned_description\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no CVE descriptions"):
                _load_dataset_index(dataset)

    def test_dataset_failure_is_not_reported_as_an_unknown_cve(self) -> None:
        predictor = CAPECPredictor.__new__(CAPECPredictor)
        predictor.by_cve = {}
        predictor.dataset_error = "Stage-02 dataset not found: /missing/dataset.csv"
        predictor.model = None
        predictor.load_error = "Model unavailable"
        health = predictor.status()
        self.assertEqual(health["dataset_status"], "unavailable")
        self.assertEqual(health["cve_records"], 0)
        self.assertEqual(health["dataset_error"], predictor.dataset_error)
        with self.assertRaisesRegex(RuntimeError, "CVE dataset lookup is unavailable"):
            predictor._resolve_input("CVE-2024-10665")

    def test_dataset_index_does_not_load_saved_cleaning_or_keywords(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.csv"
            dataset.write_text(
                "cve_id,weakness,capec_id,uncleaned_description,"
                "cleaned_description,keyphrases\n"
                "CVE-2026-1234,CWE-1,CAPEC-1,Raw vulnerability text,"
                "stale cleaned text,stale keywords\n",
                encoding="utf-8",
            )
            record = _load_dataset_index(dataset)["CVE-2026-1234"]

        self.assertEqual(record["description"], "Raw vulnerability text")
        self.assertNotIn("cleaned_description", record)
        self.assertNotIn("keywords", record)

    def test_cve_lookup_preserves_exact_stored_description(self) -> None:
        raw = "  Original  vulnerability text.\nSecond line.\tDetails.  "
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.csv"
            with dataset.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["cve_id", "uncleaned_description"])
                writer.writerow(["CVE-2026-1234", raw])
            predictor = CAPECPredictor.__new__(CAPECPredictor)
            predictor.dataset_error = None
            predictor.by_cve = _load_dataset_index(dataset)
            predictor.by_description = _index_descriptions(predictor.by_cve)

        description, record, input_type = predictor._resolve_input("cve-2026-1234")
        self.assertEqual(description, raw)
        self.assertEqual(record["cve_id"], "CVE-2026-1234")
        self.assertEqual(input_type, "cve_id")
        description, record, input_type = predictor._resolve_input(raw)
        self.assertEqual(description, raw)
        self.assertEqual(record["cve_id"], "CVE-2026-1234")
        self.assertEqual(input_type, "description")
        self.assertIsNone(predictor._resolve_input(raw.lower())[1])
        with self.assertRaises(LookupError):
            predictor._resolve_input("CVE-2026-12345")
        with self.assertRaises(ValueError):
            predictor._resolve_input("CVE-2026-1234 extra text")

    def test_shared_description_does_not_identify_an_arbitrary_cve(self) -> None:
        predictor = CAPECPredictor.__new__(CAPECPredictor)
        predictor.by_description = _index_descriptions({
            cve_id: {"cve_id": cve_id, "description": "Shared vulnerability description."}
            for cve_id in ("CVE-2026-1234", "CVE-2026-5678")
        })
        self.assertIsNone(predictor._resolve_input("Shared vulnerability description.")[1])

    def test_exact_description_can_start_with_cve(self) -> None:
        predictor = CAPECPredictor.__new__(CAPECPredictor)
        raw = "CVE-2026-1234 affects the application login."
        record = {"cve_id": "CVE-2026-1234", "description": raw}
        predictor.by_description = _index_descriptions({"CVE-2026-1234": record})
        self.assertEqual(predictor._resolve_input(raw), (raw, record, "description"))

    def test_split_index_uses_exact_membership_and_rejects_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, cve_id in (
                ("training_final.csv", "CVE-2026-1234"),
                ("validation_final.csv", "CVE-2026-12345"),
                ("testing_final.csv", "CVE-2026-5678"),
            ):
                (root / filename).write_text(f"cve_id\n{cve_id}\n")
            index = _load_split_index(root)
            self.assertEqual(index["CVE-2026-1234"]["split"], "train")
            self.assertEqual(index["CVE-2026-12345"]["split"], "validation")
            self.assertEqual(index["CVE-2026-5678"]["split"], "test")
            with (root / "testing_final.csv").open("a") as handle:
                handle.write("CVE-2026-1234\n")
            with self.assertRaisesRegex(ValueError, "Duplicate CVE ID"):
                _load_split_index(root)

    def test_known_cve_extracts_and_scores_fresh_keywords(self) -> None:
        preprocessor = _FakePreprocessor()
        model = _FakeOurApproach()
        keybert = _FakeKeyBERT()
        predictor = CAPECPredictor.__new__(CAPECPredictor)
        predictor.model_path = Path("/tmp/fake-description-run")
        predictor.model = model
        predictor._lock = threading.Lock()
        predictor.load_error = None
        predictor.dataset_error = None
        predictor.preprocessing_error = None
        predictor.keyword_error = None
        predictor.keyword_module = keybert
        predictor.keyword_contract = {"final_top_n": 5}
        predictor.split_error = None
        predictor.split_index = {
            "CVE-2026-1234": {"split": "test", "source": "/tmp/testing_final.csv"}
        }
        predictor.preprocessor = preprocessor
        predictor.model_contract = {
            "input_case": "cleaned_description+keybert_modified_keywords",
            "use_keywords": True,
        }
        predictor.by_cve = {
            "CVE-2026-1234": {
                "cve_id": "CVE-2026-1234",
                "description": "Raw vulnerability text from Stage 02.",
                "ground_truth_capecs": ["CAPEC-1"],
                "ground_truth_cwes": ["CWE-1"],
                "cleaned_description": "must never be used",
                "keywords": ["must never be used"],
            }
        }
        predictor.by_description = _index_descriptions(predictor.by_cve)

        result = predictor.predict("CVE-2026-1234")

        self.assertEqual(preprocessor.calls, ["Raw vulnerability text from Stage 02."])
        self.assertEqual(
            model.calls,
            [
                (
                    "fresh cleaned description",
                    ["fresh key phrase", "second fresh phrase"],
                    10,
                )
            ],
        )
        self.assertEqual(keybert.calls, [("fresh cleaned description", 5, "cpu")])
        self.assertEqual(result["keywords"], ["fresh key phrase", "second fresh phrase"])
        self.assertEqual(result["dataset_split"], "test")
        self.assertEqual(result["dataset_split_source"], "/tmp/testing_final.csv")
        self.assertTrue(result["keywords_used_for_scoring"])
        self.assertEqual(result["keyword_role"], "model_input")
        self.assertTrue(result["evaluation"]["threshold_exact_match"])
        description_result = predictor.predict("Raw vulnerability text from Stage 02.")
        self.assertEqual(description_result["resolved_cve_id"], "CVE-2026-1234")
        self.assertEqual(description_result["dataset_match_type"], "description")
        self.assertEqual(description_result["dataset_split"], "test")
        self.assertEqual(description_result["predictions"], result["predictions"])
        predictor.split_index["CVE-2026-1234"]["split"] = "train"
        self.assertEqual(predictor.predict("Raw vulnerability text from Stage 02.")["dataset_split"], "train")
        predictor.split_index["CVE-2026-1234"]["split"] = "validation"
        self.assertEqual(predictor.predict("Raw vulnerability text from Stage 02.")["dataset_split"], "validation")

        predictor.split_index = {}
        self.assertEqual(
            predictor.predict("CVE-2026-1234")["dataset_split"],
            "not_in_model_splits",
        )
        predictor.split_error = "Missing split file"
        self.assertEqual(
            predictor.predict("CVE-2026-1234")["dataset_split"], "unavailable"
        )
        free_text = predictor.predict("A vulnerability description supplied by the user.")
        self.assertIsNone(free_text["resolved_cve_id"])
        self.assertIsNone(free_text["dataset_split"])

    def test_model_contract_requires_keyword_branches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "our_approach_run"
            rae = root / "RAE_XMC_keywords"
            run.mkdir()
            rae.mkdir()
            (rae / "model_config.json").write_text(
                json.dumps({"run_config": {"use_keyphrases": True}}),
                encoding="utf-8",
            )
            (run / "run_config.json").write_text(
                json.dumps(
                    {
                        "data": {"input_case": "cleaned_description+keybert_modified_keywords"},
                        "source_models": {
                            "cwe_model_dir": str(root / "NN_keywords"),
                            "direct_capec_model_dir": str(root / "NN_keywords"),
                            "rae_xmc_run_dir": str(rae),
                        },
                    }
                ),
                encoding="utf-8",
            )

            contract = _validate_our_approach_run(run)

        self.assertEqual(contract["input_case"], "cleaned_description+keybert_modified_keywords")
        self.assertTrue(contract["use_keywords"])
        self.assertEqual(Path(contract["rae_xmc_run_dir"]).name, "RAE_XMC_keywords")

    def test_model_contract_rejects_description_only_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "run_config.json").write_text(
                json.dumps(
                    {
                        "data": {
                            "input_case": "cleaned_description"
                        },
                        "source_models": {},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "keyword-aware"):
                _validate_our_approach_run(run)


if __name__ == "__main__":
    unittest.main()
