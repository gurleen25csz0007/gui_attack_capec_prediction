"""Standalone copy of the Stage-02 cleaned-description preprocessing."""

from __future__ import annotations

import re
import threading
from typing import Any

import pandas as pd
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS


REMOVE_PURE_NUMBERS = True
USE_LEMMATIZATION = True

STOPWORDS = set(SPACY_STOP_WORDS) | set(ENGLISH_STOP_WORDS)
STOPWORDS -= {"not", "no", "without", "before", "after", "via", "through"}


def split_pipe_terms(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    return sorted(
        {
            part.strip()
            for part in re.split(r"\s*\|\s*", str(value))
            if part.strip()
            and part.strip().lower() not in {"nan", "none", "n/a", "-"}
        },
        key=len,
        reverse=True,
    )


def entity_pattern(term: str) -> str | None:
    parts = [part for part in re.split(r"[\s_\-]+", str(term).strip()) if part]
    if not parts:
        return None
    return r"(?<![A-Za-z0-9])" + r"[\s_\-]*".join(
        re.escape(part) for part in parts
    ) + r"(?![A-Za-z0-9])"


def replace_entities(text: str, terms: list[str], replacement: str) -> str:
    for term in terms:
        pattern = entity_pattern(term)
        if pattern:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def version_aliases(version: str) -> list[str]:
    version = str(version).strip()
    if not version:
        return []
    aliases = {version}
    if not version.lower().startswith("v"):
        aliases.add("v" + version)
    match = re.fullmatch(r"(\d+)\.0", version)
    if match:
        aliases.update({match.group(1), "v" + match.group(1)})
    return sorted(aliases, key=len, reverse=True)


def replace_versions_from_column(text: str, versions: list[str]) -> str:
    aliases = {alias for version in versions for alias in version_aliases(version)}
    for version in sorted(aliases, key=len, reverse=True):
        pattern = r"(?<![A-Za-z0-9])" + re.escape(version) + r"(?![A-Za-z0-9])"
        text = re.sub(pattern, "version", text, flags=re.IGNORECASE)
    return text


def collapse_consecutive_duplicate_tokens(text: Any) -> str:
    """Reduce adjacent duplicate tokens such as ``version version version``."""
    final_tokens: list[str] = []
    for token in str(text or "").split():
        if final_tokens and token.casefold() == final_tokens[-1].casefold():
            continue
        final_tokens.append(token)
    return " ".join(final_tokens)


def clean_masked_text(text: str) -> str:
    text = re.sub(
        r"\b(product|vendor)[\s_\-]+version\b",
        r"\1 version",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(product|vendor)[\s_\-]+v?\d+(?:\.\d+)?\b",
        r"\1 version",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    for pattern, replacement in (
        (r"\bvendor\s+vendor\b", "vendor"),
        (r"\bproduct\s+product\b", "product"),
        (r"\bversion\s+version\b", "version"),
        (r"\bvendor\s+product\b", "vendor"),
        (r"\bproduct\s+vendor\b", "product"),
    ):
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return collapse_consecutive_duplicate_tokens(text.strip())


def mask_description(
    text: str,
    vendor: Any = "",
    product: Any = "",
    version: Any = "",
) -> str:
    """Apply the same CPE-aware entity masking used by Stage 02."""
    masked = replace_entities(str(text), split_pipe_terms(product), "product")
    masked = replace_entities(masked, split_pipe_terms(vendor), "vendor")
    masked = replace_versions_from_column(masked, split_pipe_terms(version))
    return clean_masked_text(masked)


def remove_extra_noise(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", str(text), flags=re.IGNORECASE)
    text = re.sub(r"\b[\w.\-]+@[\w.\-]+\.\w+\b", " ", text)
    text = re.sub(
        r"\b(?:CVE-\d{4}-\d{4,7}|CWE-\d+|CAPEC-\d+|"
        r"GHSA-[A-Za-z0-9]{4}-[A-Za-z0-9]{4}-[A-Za-z0-9]{4})\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b0x[0-9a-fA-F]+\b", " ", text)
    text = re.sub(r"\b[0-9a-fA-F]{16,}\b", " ", text)
    return re.sub(r"[^A-Za-z0-9_\s]", " ", text)


def basic_clean(text: str) -> str:
    return re.sub(r"\s+", " ", remove_extra_noise(text).lower()).strip()


def clean_doc(doc: Any) -> str:
    final_tokens: list[str] = []
    for token in doc:
        raw = token.text.strip()
        if not raw or token.is_space or token.is_punct:
            continue
        if REMOVE_PURE_NUMBERS and raw.isdigit():
            continue
        word = token.lemma_.lower().strip() if USE_LEMMATIZATION else raw.lower()
        if not word or word == "-pron-" or (REMOVE_PURE_NUMBERS and word.isdigit()):
            continue
        if word in STOPWORDS:
            continue
        final_tokens.append(word)
    return collapse_consecutive_duplicate_tokens(" ".join(final_tokens))


class DescriptionPreprocessor:
    """Generate the exact cleaned-description representation used in training."""

    def __init__(self) -> None:
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError as exc:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. Run: "
                "python -m spacy download en_core_web_sm"
            ) from exc
        self._lock = threading.Lock()

    def clean(
        self,
        text: str,
        vendor: Any = "",
        product: Any = "",
        version: Any = "",
    ) -> str:
        value = " ".join(str(text or "").strip().split())
        if not value:
            raise ValueError("The vulnerability description is empty.")
        prepared = basic_clean(mask_description(value, vendor, product, version))
        with self._lock:
            cleaned = clean_doc(self.nlp(prepared))
        cleaned = collapse_consecutive_duplicate_tokens(cleaned)
        if not cleaned:
            raise ValueError(
                "The description became empty after preprocessing. "
                "Provide more vulnerability-specific detail."
            )
        return cleaned
