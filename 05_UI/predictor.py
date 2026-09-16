"""CVE/description UI adapter for the saved Stage-04 our-approach fusion."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

from text_preprocessing import DescriptionPreprocessor, collapse_consecutive_duplicate_tokens


APP_DIR = Path(__file__).resolve().parent
METHODOLOGY_DIR = APP_DIR.parent
EVALUATION_DIR = METHODOLOGY_DIR / "04_evaluation"
STAGE03_DIR = METHODOLOGY_DIR / "03_keyword_making"
SAVED_KEYWORD_DIR = STAGE03_DIR / "outputs" / "keybert_modified"
KEYWORD_MODULE_PATH = STAGE03_DIR / "methods" / "keybert_modified.py"
STAGE02_DATASET_PATH = METHODOLOGY_DIR / "02_preprocessing" / "dataset.csv"
OUR_APPROACH_MODULE_PATH = EVALUATION_DIR / "our_approch.py"
LATEST_MODEL_POINTERS = (
    EVALUATION_DIR / "batch_run" / "outputs" / "combined"
    / "latest_our_approach_keywords_run.txt",
    EVALUATION_DIR / "outputs" / "combined"
    / "latest_our_approach_keywords_run.txt",
)
BUNDLED_MODEL_DIR = APP_DIR / "models" / "our_approach"
BUNDLED_SENTENCE_ENCODER = BUNDLED_MODEL_DIR / "sentence_encoder"

MODEL_DEVICE = os.environ.get("CAPEC_UI_DEVICE") or None
DATASET_PATH = Path(
    os.environ.get("CAPEC_UI_DATASET_PATH", str(STAGE02_DATASET_PATH))
).expanduser().resolve()
MODEL_NAME = "Our Approach with keywords (CGDPF + RAE-XMC)"


def _parse_capecs(value: Any) -> set[str]:
    text = str(value or "").strip()
    labels = {
        f"CAPEC-{int(match)}"
        for match in re.findall(r"CAPEC\s*[-_:]?\s*(\d+)", text, flags=re.IGNORECASE)
    }
    if not labels and ("noid" in text.casefold() or "no-id" in text.casefold()):
        labels.add("CAPEC-noID")
    return labels


def _parse_cwes(value: Any) -> set[str]:
    return {
        f"CWE-{int(match)}"
        for match in re.findall(
            r"CWE\s*[-_:]?\s*(\d+)", str(value or ""), flags=re.IGNORECASE
        )
    }


def _split_keywords(value: str) -> list[str]:
    phrases = (
        collapse_consecutive_duplicate_tokens(" ".join(part.split()))
        for part in str(value or "").split(";")
    )
    result: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        if phrase and phrase.casefold() not in seen:
            result.append(phrase)
            seen.add(phrase.casefold())
    return result


def _validate_keybert_contract(module: Any) -> dict[str, Any]:
    """Keep online extraction aligned with the published training method."""
    config = json.loads((SAVED_KEYWORD_DIR / "config.json").read_text(encoding="utf-8"))
    settings = {
        "model_name": str(module.MODEL_NAME),
        "keyphrase_ngram_range": list(module.KEYPHRASE_NGRAM_RANGE),
        "candidate_top_n": int(module.CANDIDATE_TOP_N),
        "final_top_n": int(module.FINAL_TOP_N),
        "target_sim_threshold": float(module._published_target_threshold()),
    }
    mismatches = {
        key: {"published": config.get(key), "online": value}
        for key, value in settings.items() if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Online KeyBERT settings differ from the published config: {mismatches}")
    return settings


def _load_module(path: Path, name: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required module not found: {path}")
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_model_path(explicit: str | None = None) -> Path:
    """Resolve the completed keyword-aware Our Approach run."""
    configured = explicit or os.environ.get("CAPEC_UI_MODEL_PATH")
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_file():
            path = Path(path.read_text(encoding="utf-8").strip()).expanduser().resolve()
        return path
    for pointer in LATEST_MODEL_POINTERS:
        if pointer.is_file():
            return Path(
                pointer.read_text(encoding="utf-8").strip()
            ).expanduser().resolve()
    raise FileNotFoundError(
        "No completed keyword-aware Our Approach run is registered. Run "
        f"{EVALUATION_DIR / 'batch_run' / 'submit_grid_evaluations.sh'} first, "
        "or set CAPEC_UI_MODEL_PATH to a completed "
        "our_approach_keywords_* run directory."
    )


def _load_dataset_index(dataset_path: Path) -> dict[str, dict[str, Any]]:
    """Index raw descriptions by CVE ID only.

    Stored cleaned descriptions and stored keyphrases are deliberately ignored:
    the UI recalculates both for every request.
    """
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Stage-02 dataset not found: {dataset_path}")
    by_cve: dict[str, dict[str, Any]] = {}
    with dataset_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"cve_id", "uncleaned_description"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(
                f"Stage-02 dataset is missing required columns: {missing}"
            )
        for row in reader:
            cve_id = " ".join(str(row.get("cve_id") or "").split()).upper()
            raw = str(row.get("uncleaned_description") or "")
            if not cve_id or not raw.strip():
                continue
            by_cve[cve_id] = {
                "cve_id": cve_id,
                "description": raw,
                "ground_truth_capecs": sorted(_parse_capecs(row.get("capec_id"))),
                "ground_truth_cwes": sorted(_parse_cwes(row.get("weakness"))),
            }
    if not by_cve:
        raise ValueError(f"Stage-02 dataset has no CVE descriptions: {dataset_path}")
    return by_cve


def _load_split_index(split_dir: Path) -> dict[str, dict[str, str]]:
    """Read exact CVE membership from the model's existing split files."""
    index: dict[str, dict[str, str]] = {}
    for split, filename in (
        ("train", "training_final.csv"),
        ("validation", "validation_final.csv"),
        ("test", "testing_final.csv"),
    ):
        path = split_dir / filename
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if "cve_id" not in (reader.fieldnames or ()):
                raise ValueError(f"Split file is missing cve_id: {path}")
            for row in reader:
                cve_id = str(row.get("cve_id") or "").strip().upper()
                if not cve_id:
                    raise ValueError(f"Split file has an empty CVE ID: {path}")
                if cve_id in index:
                    raise ValueError(f"Duplicate CVE ID in model splits: {cve_id}")
                index[cve_id] = {"split": split, "source": str(path)}
    return index


def _index_descriptions(
    by_cve: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    """Match exact raw text; shared descriptions cannot identify one CVE."""
    index: dict[str, dict[str, Any] | None] = {}
    for record in by_cve.values():
        description = record["description"].strip()
        index[description] = None if description in index else record
    return index


def _validate_our_approach_run(model_path: Path) -> dict[str, Any]:
    """Require the keyword-aware Stage-04 Our Approach fusion."""
    run_config_path = model_path / "run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"Our Approach run config not found: {run_config_path}")
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    input_case = str(run_config.get("data", {}).get("input_case", ""))
    expected_input_case = "cleaned_description+keybert_modified_keywords"
    if input_case != expected_input_case:
        raise ValueError(
            "The UI requires the keyword-aware "
            "Our Approach artifact; "
            f"expected input_case={expected_input_case!r}, found {input_case!r}."
        )

    sources = run_config.get("source_models", {})
    expected_sources = {
        "cwe_model_dir": "NN_keywords",
        "direct_capec_model_dir": "NN_keywords",
        "rae_xmc_run_dir": "RAE_XMC_keywords",
    }
    mismatches = {
        name: {"expected_directory": expected, "actual": sources.get(name)}
        for name, expected in expected_sources.items()
        if Path(str(sources.get(name, ""))).name != expected
    }
    if mismatches:
        raise ValueError(
            "Our Approach does not reference every keyword-aware Stage-04 "
            f"branch: {mismatches}"
        )

    rae_dir = Path(str(sources["rae_xmc_run_dir"])).expanduser()
    if not rae_dir.is_absolute():
        rae_dir = (model_path / rae_dir).resolve()
    rae_config_path = rae_dir / "model_config.json"
    if not rae_config_path.is_file():
        raise FileNotFoundError(
            f"RAE-XMC model config not found: {rae_config_path}"
        )
    rae_config = json.loads(rae_config_path.read_text(encoding="utf-8"))
    if rae_config.get("run_config", {}).get("use_keyphrases") is not True:
        raise ValueError(
            f"RAE-XMC artifact must be trained with keyphrases: {rae_config_path}"
        )
    split_dir = Path(
        run_config.get("data", {}).get("split_directory") or SAVED_KEYWORD_DIR
    ).expanduser()
    if not split_dir.is_absolute():
        split_dir = model_path / split_dir
    return {
        "input_case": input_case,
        "use_keywords": True,
        "cwe_model_dir": str(sources["cwe_model_dir"]),
        "direct_capec_model_dir": str(sources["direct_capec_model_dir"]),
        "rae_xmc_run_dir": str(rae_dir),
        "run_config_path": str(run_config_path),
        "split_directory": str(split_dir.resolve()),
    }


class ModelNotConfiguredError(RuntimeError):
    """Raised when the saved fusion or its preprocessing cannot be loaded."""


class CAPECPredictor:
    """Thread-safe keyword extraction and Our Approach prediction."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path: Path | None = None
        self.model: Any = None
        self._lock = threading.Lock()
        self.load_error: str | None = None
        self.dataset_error: str | None = None
        self.preprocessing_error: str | None = None
        self.keyword_error: str | None = None
        self.split_error: str | None = None
        self.by_cve: dict[str, dict[str, Any]] = {}
        self.by_description: dict[str, dict[str, Any] | None] = {}
        self.preprocessor: DescriptionPreprocessor | None = None
        self.keyword_module: Any = None
        self.keyword_contract: dict[str, Any] = {}
        self.split_index: dict[str, dict[str, str]] = {}
        self.model_contract: dict[str, Any] = {}

        try:
            self.by_cve = _load_dataset_index(DATASET_PATH)
            self.by_description = _index_descriptions(self.by_cve)
        except Exception as exc:
            self.dataset_error = str(exc)

        try:
            self.preprocessor = DescriptionPreprocessor()
        except Exception as exc:
            self.preprocessing_error = str(exc)

        try:
            self.keyword_module = _load_module(
                KEYWORD_MODULE_PATH, "methodology_stage03_keywords_ui"
            )
            self.keyword_contract = _validate_keybert_contract(self.keyword_module)
        except Exception as exc:
            self.keyword_error = str(exc)

        try:
            self.model_path = _resolve_model_path(model_path)
            self.model = self._load_model(self.model_path)
        except Exception as exc:
            self.load_error = str(exc)

        if self.model_path is not None and self.load_error is None:
            try:
                self.split_index = _load_split_index(
                    Path(self.model_contract["split_directory"])
                )
            except Exception as exc:
                self.split_error = str(exc)

    @property
    def ready(self) -> bool:
        return (
            self.model is not None
            and self.load_error is None
            and self.preprocessor is not None
            and self.preprocessing_error is None
            and self.keyword_module is not None
            and self.keyword_error is None
        )

    def _load_model(self, model_path: Path) -> Any:
        if not model_path.is_dir():
            raise FileNotFoundError(f"Our-approach run directory not found: {model_path}")
        self.model_contract = _validate_our_approach_run(model_path)
        module = _load_module(
            OUR_APPROACH_MODULE_PATH, "methodology_our_approach_ui"
        )
        encoder_path = (
            BUNDLED_SENTENCE_ENCODER
            if BUNDLED_SENTENCE_ENCODER.is_dir()
            else None
        )
        model = module.load_our_approach(
            model_path,
            device=MODEL_DEVICE,
            sentence_encoder_path=encoder_path,
        )
        model_uses_keywords = bool(getattr(model, "use_keywords", False))
        if (
            model_uses_keywords != bool(self.model_contract["use_keywords"])
            or getattr(model, "input_case", None) != self.model_contract["input_case"]
        ):
            raise ValueError(
                "Loaded model input mode does not match the validated artifact; "
                f"input_case={getattr(model, 'input_case', None)!r}, "
                f"use_keywords={model_uses_keywords}."
            )
        return model

    def _resolve_input(
        self, user_input: str
    ) -> tuple[str, dict[str, Any] | None, str]:
        value = str(user_input).strip()
        if not value:
            raise ValueError("Enter a CVE ID or vulnerability description.")

        if re.fullmatch(r"CVE-\d{4}-\d{4,}", value, flags=re.IGNORECASE):
            if self.dataset_error:
                raise ModelNotConfiguredError(
                    f"CVE dataset lookup is unavailable: {self.dataset_error}"
                )
            record = self.by_cve.get(value.upper())
            if record is None:
                raise LookupError(
                    f"{value.upper()} is not present in the Stage-02 dataset. "
                    "Paste its vulnerability description instead."
                )
            return record["description"], record, "cve_id"

        record = self.by_description.get(value)
        if record is not None:
            return record["description"], record, "description"

        if value.upper().startswith("CVE"):
            raise ValueError("Enter an exact CVE ID, such as CVE-2024-10665.")

        if len(value) < 10:
            raise ValueError("Please provide a more detailed vulnerability description.")
        return value, None, "description"

    def predict(self, text: str, top_k: int = 10) -> dict[str, Any]:
        if not self.ready:
            raise ModelNotConfiguredError(
                self.load_error
                or self.preprocessing_error
                or self.keyword_error
                or "The CAPEC model is not configured."
            )
        description, record, input_type = self._resolve_input(text)
        model_text = self.preprocessor.clean(description)
        cleaning_source = "stage02_recalculated"
        top_k = max(1, min(int(top_k), 20))
        with self._lock:
            keywords = _split_keywords(
                self.keyword_module.keybert_modified_keywords(
                    model_text,
                    top_k=int(self.keyword_contract["final_top_n"]),
                    device=str(getattr(self.model, "device", MODEL_DEVICE or "auto")),
                )
            )
            result = self.model.predict(
                description=model_text,
                keywords=keywords,
                top_k=top_k,
            )

        predictions = [
            {"capec_id": item["capec_id"], "score": float(item["score"])}
            for item in result["ranked_capecs"]
        ]
        threshold_predictions = [
            label for label in result.get("thresholded_labels", [])
            if label != "CAPEC-noID"
        ]
        ground_truth = [] if record is None else [
            label for label in record["ground_truth_capecs"] if label != "CAPEC-noID"
        ]
        ranked_labels = [item["capec_id"] for item in predictions]
        truth_set, threshold_set = set(ground_truth), set(threshold_predictions)
        evaluation = None
        if ground_truth:
            evaluation = {
                "top_1_correct": bool(ranked_labels and ranked_labels[0] in truth_set),
                "top_5_hit": bool(truth_set.intersection(ranked_labels[:5])),
                "top_10_hit": bool(truth_set.intersection(ranked_labels[:10])),
                "threshold_any_hit": bool(truth_set.intersection(threshold_set)),
                "threshold_exact_match": threshold_set == truth_set,
            }

        return {
            "input_type": input_type,
            "resolved_cve_id": None if record is None else record["cve_id"],
            "dataset_match_type": None if record is None else input_type,
            "description": description,
            "cleaned_description": model_text,
            "model_text": model_text,
            "cleaning_source": cleaning_source,
            "keywords": keywords,
            "keyword_source": "stage03_keybert_modified_recalculated",
            "keywords_used_for_scoring": True,
            "keyword_role": "model_input",
            "keyword_method": dict(self.keyword_contract),
            "dataset_source": str(DATASET_PATH) if record is not None else None,
            "dataset_split": (
                None if record is None else
                "unavailable" if self.split_error else
                self.split_index.get(record["cve_id"], {}).get(
                    "split", "not_in_model_splits"
                )
            ),
            "dataset_split_source": (
                None if record is None else
                self.split_index.get(record["cve_id"], {}).get("source")
            ),
            "dataset_split_error": self.split_error,
            "predictions": predictions,
            "threshold_predictions": threshold_predictions,
            "ground_truth_cwes": [] if record is None else record["ground_truth_cwes"],
            "ground_truth_capecs": ground_truth,
            "evaluation": evaluation,
            "model": {
                "name": MODEL_NAME,
                "threshold": float(result["threshold"]),
                "alpha": float(result["alpha"]),
                "beta": float(result["beta"]),
                "input_case": result.get("input_case"),
                "parameter_source": str(
                    self.model_path / "fusion_parameters.json"
                ),
                "score_type": result["score_type"],
                "device": str(getattr(self.model, "device", MODEL_DEVICE or "auto")),
                "path": str(self.model_path),
            },
            "model_contract": dict(self.model_contract),
        }

    def status(self) -> dict[str, Any]:
        dataset_status = {
            "dataset_status": "connected" if self.by_cve and not self.dataset_error else "unavailable",
            "cve_records": len(self.by_cve),
            "dataset_source": str(DATASET_PATH),
            "dataset_error": self.dataset_error,
        }
        if self.ready:
            return {
                **dataset_status,
                "status": "ready",
                "model": MODEL_NAME,
                "device": str(getattr(self.model, "device", MODEL_DEVICE or "auto")),
                "model_path": str(self.model_path),
                "dataset_usage": "CVE ID to raw description lookup only",
                "preprocessing": (
                    "Fresh Stage 02 cleaning + fresh modified-KeyBERT extraction "
                    "+ saved Stage 04 keyword-aware Our Approach"
                ),
                "model_contract": dict(self.model_contract),
                "dataset_split_error": self.split_error,
                "keywords_used_for_scoring": True,
                "keyword_method": dict(self.keyword_contract),
            }
        return {
            **dataset_status,
            "status": "not_configured",
            "model": MODEL_NAME,
            "message": self.load_error
            or self.preprocessing_error
            or self.keyword_error
            or self.dataset_error
            or "Model path has not been configured.",
        }
