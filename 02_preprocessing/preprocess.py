from pathlib import Path
import ast
import json
import re

import pandas as pd
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS
from tqdm.auto import tqdm


# ============================================================
# PATHS / SETTINGS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# 6 NVD CSV files are inside this sibling folder
NVD_DIR = PROJECT_DIR / "00_input_data"
GROUND_TRUTH_DIR = PROJECT_DIR / "01_cwe_capec_ground_truth"


# Keep these two support files in the same folder as this script
CWE_FILE = GROUND_TRUTH_DIR / "cwe_final.csv"
CWE_CAPEC_MAP_FILE = GROUND_TRUTH_DIR / "CWE_MAPPED.csv"

# Final output
OUTPUT_FILE = SCRIPT_DIR / "dataset.csv"

REMOVE_PURE_NUMBERS = True
USE_LEMMATIZATION = True

tqdm.pandas()


# ============================================================
# GENERAL PARSER
# ============================================================

def parse_json_like(x):
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        return [x]
    if pd.isna(x):
        return []

    if isinstance(x, str):
        x = x.strip()
        if not x or x.lower() in {"nan", "none", "null"}:
            return []

        try:
            obj = json.loads(x)
        except Exception:
            try:
                obj = ast.literal_eval(x)
            except Exception:
                return []

        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]

    return []


# ============================================================
# 1. LOAD THE 6 NVD CSV FILES
# ============================================================

def load_nvd_data():
    if not NVD_DIR.exists():
        raise FileNotFoundError(f"NVD folder not found: {NVD_DIR}")

    csv_files = sorted(NVD_DIR.glob("*.csv"))

    if not csv_files:
        raise RuntimeError(f"No CSV files found in: {NVD_DIR}")

    frames = []

    for file_path in csv_files:
        print("Reading:", file_path.name)
        temp_df = pd.read_csv(file_path, low_memory=False)
        frames.append(temp_df)

    df = pd.concat(frames, ignore_index=True)

    required = {
        "cve.id",
        "cve.weaknesses",
        "cve.descriptions",
        "cve.configurations",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Required NVD columns are missing: "
            + ", ".join(sorted(missing))
        )

    print("Combined rows:", len(df))
    return df


# ============================================================
# 2. EXTRACT VENDOR / PRODUCT / VERSION FROM CPE
# ============================================================

def split_cpe23(cpe):
    parts = []
    current = []
    escaped = False

    for ch in str(cpe):
        if escaped:
            current.append(ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == ":":
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)

    if escaped:
        current.append("\\")

    parts.append("".join(current))
    return parts


def clean_cpe_value(x):
    if x is None:
        return None

    x = str(x).strip()

    if not x or x.upper() in {"*", "-", "ANY", "NA"}:
        return None

    x = x.replace("\\", "").replace("_", " ").strip()
    return x or None


def walk_nodes(nodes):
    matches = []

    if not isinstance(nodes, list):
        return matches

    for node in nodes:
        if not isinstance(node, dict):
            continue

        cpe_match = node.get("cpeMatch", [])
        if isinstance(cpe_match, list):
            matches.extend(cpe_match)

        matches.extend(walk_nodes(node.get("nodes", [])))

    return matches


def extract_vendor_product_version(config_col):
    vendors = set()
    products = set()
    versions = set()

    for config in parse_json_like(config_col):
        if not isinstance(config, dict):
            continue

        for match in walk_nodes(config.get("nodes", [])):
            if not isinstance(match, dict):
                continue

            criteria = match.get("criteria")

            if criteria:
                parts = split_cpe23(criteria)

                # cpe:2.3:part:vendor:product:version:...
                if len(parts) >= 6:
                    vendor = clean_cpe_value(parts[3])
                    product = clean_cpe_value(parts[4])
                    version = clean_cpe_value(parts[5])

                    if vendor:
                        vendors.add(vendor)
                    if product:
                        products.add(product)
                    if version:
                        versions.add(version)

            for key in (
                "versionStartIncluding",
                "versionStartExcluding",
                "versionEndIncluding",
                "versionEndExcluding",
            ):
                value = clean_cpe_value(match.get(key))
                if value:
                    versions.add(value)

    return (
        " | ".join(sorted(vendors)),
        " | ".join(sorted(products)),
        " | ".join(sorted(versions)),
    )


# ============================================================
# 3. EXTRACT ENGLISH DESCRIPTION + CWE
# ============================================================

def extract_english_description(value):
    for item in parse_json_like(value):
        if isinstance(item, dict) and item.get("lang") == "en":
            text = item.get("value")
            if isinstance(text, str) and text.strip():
                return text.strip()

    return None


def extract_cwes(value):
    cwes = set()

    for weakness in parse_json_like(value):
        if not isinstance(weakness, dict):
            continue

        descriptions = weakness.get("description", [])
        if not isinstance(descriptions, list):
            continue

        for item in descriptions:
            if not isinstance(item, dict):
                continue

            cwe = item.get("value")

            if isinstance(cwe, str):
                cwe = cwe.strip().upper()

                # Automatically excludes NVD-CWE-noinfo and NVD-CWE-Other
                if re.fullmatch(r"CWE-\d+", cwe):
                    cwes.add(cwe)

    if not cwes:
        return None

    return ",".join(sorted(cwes))


# ============================================================
# 4. MASK VENDOR / PRODUCT / VERSION
# ============================================================




def split_pipe_terms(x):
    if pd.isna(x):
        return []

    parts = re.split(r"\s*\|\s*", str(x))

    cleaned = [
        p.strip()
        for p in parts
        if p.strip()
        and p.strip().lower() not in {"nan", "none", "n/a", "-"}
    ]

    return sorted(set(cleaned), key=len, reverse=True)

def entity_pattern(term):
    parts = [
        p for p in re.split(r"[\s_\-]+", str(term).strip())
        if p
    ]

    if not parts:
        return None

    middle = r"[\s_\-]*".join(re.escape(p) for p in parts)

    return (
        r"(?<![A-Za-z0-9])"
        + middle
        + r"(?![A-Za-z0-9])"
    )




def replace_entities(text, terms, replacement):
    for term in terms:
        pattern = entity_pattern(term)

        if pattern:
            text = re.sub(
                pattern,
                replacement,
                text,
                flags=re.IGNORECASE
            )

    return text


def version_aliases(version):
    version = str(version).strip()

    if not version:
        return []

    aliases = {version}

    if not version.lower().startswith("v"):
        aliases.add("v" + version)

    match = re.fullmatch(r"(\d+)\.0", version)

    if match:
        base = match.group(1)
        aliases.add(base)
        aliases.add("v" + base)

    return sorted(aliases, key=len, reverse=True)


def replace_versions_from_column(text, versions):
    aliases = set()

    for version in versions:
        aliases.update(version_aliases(version))

    for version in sorted(aliases, key=len, reverse=True):
        pattern = (
            r"(?<![A-Za-z0-9])"
            + re.escape(version)
            + r"(?![A-Za-z0-9])"
        )

        text = re.sub(
            pattern,
            "version",
            text,
            flags=re.IGNORECASE,
        )

    return text


def collapse_consecutive_duplicate_tokens(text):
    """Reduce adjacent duplicate tokens such as ``vendor vendor vendor``."""
    final_tokens = []

    for token in str(text).split():
        if final_tokens and token.casefold() == final_tokens[-1].casefold():
            continue
        final_tokens.append(token)

    return " ".join(final_tokens)





def clean_masked_text(text):
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

    text = re.sub(
        r"\bvendor\s+vendor\b",
        "vendor",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bproduct\s+product\b",
        "product",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bversion\s+version\b",
        "version",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bvendor\s+product\b",
        "vendor",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\bproduct\s+vendor\b",
        "product",
        text,
        flags=re.IGNORECASE,
    )

    return collapse_consecutive_duplicate_tokens(text.strip())


def mask_description(row):
    text = str(row["description"])

    vendors = split_pipe_terms(row["vendor"])
    products = split_pipe_terms(row["product"])
    versions = split_pipe_terms(row["version"])

    text = replace_entities(text, products, "product")
    text = replace_entities(text, vendors, "vendor")
    text = replace_versions_from_column(text, versions)
    text = clean_masked_text(text)

    return text


# ============================================================
# 5. CLEAN + LEMMATIZE MASKED DESCRIPTION
# ============================================================

stopwords = set(SPACY_STOP_WORDS) | set(ENGLISH_STOP_WORDS)

IMPORTANT_WORDS_TO_KEEP = {
    "not",
    "no",
    "without",
    "before",
    "after",
    "via",
    "through",
}

stopwords -= IMPORTANT_WORDS_TO_KEEP

def remove_extra_noise(text):
    text = str(text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove email addresses
    text = re.sub(
        r"\b[\w.\-]+@[\w.\-]+\.\w+\b",
        " ",
        text,
    )

    # Remove vulnerability / security taxonomy identifiers
    text = re.sub(
        r"\b(?:"
        r"CVE-\d{4}-\d{4,7}"
        r"|CWE-\d+"
        r"|CAPEC-\d+"
        r"|GHSA-[A-Za-z0-9]{4}-[A-Za-z0-9]{4}-[A-Za-z0-9]{4}"
        r")\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove hexadecimal values / long hashes
    text = re.sub(r"\b0x[0-9a-fA-F]+\b", " ", text)
    text = re.sub(r"\b[0-9a-fA-F]{16,}\b", " ", text)

    # Remove punctuation / special characters
    text = re.sub(r"[^A-Za-z0-9_\s]", " ", text)
    return text

def basic_clean(text):
    text = remove_extra_noise(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_doc(doc):
    final_tokens = []

    for token in doc:
        raw = token.text.strip()

        if not raw or token.is_space or token.is_punct:
            continue

        if REMOVE_PURE_NUMBERS and raw.isdigit():
            continue

        if USE_LEMMATIZATION:
            word = token.lemma_.lower().strip()
        else:
            word = raw.lower().strip()

        if not word or word == "-pron-":
            continue

        if REMOVE_PURE_NUMBERS and word.isdigit():
            continue

        if word in stopwords:
            continue

        final_tokens.append(word)

    return collapse_consecutive_duplicate_tokens(" ".join(final_tokens))


def clean_descriptions(df):
    try:
        nlp = spacy.load(
            "en_core_web_sm",
            disable=["parser", "ner"],
        )
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Run: python -m spacy download en_core_web_sm"
        ) from exc

    prepared_texts = [
        basic_clean(text)
        for text in tqdm(
            df["masked_description"].tolist(),
            desc="Basic cleaning",
        )
    ]

    cleaned = []

    for doc in tqdm(
        nlp.pipe(prepared_texts, batch_size=256),
        total=len(prepared_texts),
        desc="Lemmatizing",
    ):
        cleaned.append(clean_doc(doc))

    df["cleaned_description"] = cleaned
    return df


# ============================================================
# 6. CWE VALIDATION + CWE -> CAPEC
# ============================================================

def normalize_cwe_id(x):
    if pd.isna(x):
        return None

    text = str(x).strip().upper()

    match = re.search(r"CWE[-_\s]*(\d+)", text)
    if match:
        return f"CWE-{int(match.group(1))}"

    if re.fullmatch(r"\d+", text):
        return f"CWE-{int(text)}"

    return None


def normalize_capec_id(x):
    if pd.isna(x):
        return None

    text = str(x).strip().upper()

    match = re.search(r"CAPEC[-_\s]*(\d+)", text)
    if match:
        return f"CAPEC-{int(match.group(1))}"

    if re.fullmatch(r"\d+", text):
        return f"CAPEC-{int(text)}"

    return None


def extract_id_list(x, kind):
    if pd.isna(x):
        return []

    text = str(x).strip()

    if text.lower() in {"", "nan", "none", "null", "[]"}:
        return []

    prefix = "CWE" if kind == "cwe" else "CAPEC"
    normalizer = (
        normalize_cwe_id
        if kind == "cwe"
        else normalize_capec_id
    )

    values = []

    if (
        (text.startswith("[") and text.endswith("]"))
        or (text.startswith("(") and text.endswith(")"))
    ):
        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, (list, tuple, set)):
                values = list(parsed)
            else:
                values = [parsed]
        except Exception:
            values = []

    if not values:
        found = re.findall(
            rf"{prefix}[-_\s]*\d+",
            text,
            flags=re.IGNORECASE,
        )

        if found:
            values = found
        else:
            values = re.split(r"[,|;/\s]+", text)

    normalized = []

    for value in values:
        item = normalizer(value)
        if item:
            normalized.append(item)

    return list(dict.fromkeys(normalized))


def add_capec_labels(df):
    if not CWE_FILE.exists():
        raise FileNotFoundError(f"Missing file: {CWE_FILE}")

    if not CWE_CAPEC_MAP_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {CWE_CAPEC_MAP_FILE}"
        )

    cwe_df = pd.read_csv(CWE_FILE, dtype=str)
    map_df = pd.read_csv(CWE_CAPEC_MAP_FILE, dtype=str)

    if "cwe_id" not in cwe_df.columns:
        raise ValueError(
            "cwe_final.csv must contain column: cwe_id"
        )

    required_map_cols = {"cwe_id", "capec_ids"}
    missing = required_map_cols - set(map_df.columns)

    if missing:
        raise ValueError(
            "CWE_MAPPED.csv is missing columns: "
            + ", ".join(sorted(missing))
        )

    valid_cwes = {
        cwe
        for cwe in cwe_df["cwe_id"].map(normalize_cwe_id)
        if cwe
    }

    cwe_to_capecs = {}

    for _, row in map_df.iterrows():
        cwe = normalize_cwe_id(row["cwe_id"])

        if not cwe:
            continue

        capecs = extract_id_list(
            row["capec_ids"],
            "capec",
        )

        cwe_to_capecs.setdefault(cwe, set()).update(capecs)

    def clean_cwes(cell):
        ids = extract_id_list(cell, "cwe")
        return [
            cwe for cwe in ids
            if cwe in valid_cwes
        ]

    def map_capecs(cwe_ids):
        capecs = set()

        for cwe in cwe_ids:
            capecs.update(
                cwe_to_capecs.get(cwe, set())
            )

        return sorted(
            capecs,
            key=lambda x: int(x.split("-")[1]),
        )

    cwe_lists = df["weakness"].apply(clean_cwes)

    df["weakness"] = cwe_lists.apply(
        lambda ids: ",".join(ids)
    )

    df["capec_id"] = cwe_lists.apply(
        lambda ids: ",".join(map_capecs(ids))
    )

    # Remove rows with no valid CWE or no CAPEC mapping
    df = df[
        df["weakness"].str.strip().ne("")
        & df["capec_id"].str.strip().ne("")
    ].copy()

    return df


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("NVD input folder:", NVD_DIR)

    # 1. Combine NVD files
    df = load_nvd_data()

    # 2. Extract vendor/product/version
    print("Extracting vendor/product/version...")

    vpv = df["cve.configurations"].progress_apply(
        extract_vendor_product_version
    )

    df["vendor"] = vpv.str[0]
    df["product"] = vpv.str[1]
    df["version"] = vpv.str[2]

    # 3. English description + CWE
    print("Extracting English descriptions and CWE IDs...")

    df["description"] = df[
        "cve.descriptions"
    ].apply(extract_english_description)

    df["weakness"] = df[
        "cve.weaknesses"
    ].apply(extract_cwes)

    df = df.rename(
        columns={"cve.id": "cve_id"}
    )

    df = df[
        [
            "cve_id",
            "weakness",
            "description",
            "vendor",
            "product",
            "version",
        ]
    ].copy()

    df = df.dropna(
        subset=["cve_id", "weakness", "description"]
    )

    df["cve_id"] = (
        df["cve_id"]
        .astype(str)
        .str.strip()
    )

    df["description"] = (
        df["description"]
        .astype(str)
        .str.replace(r"[\r\n\t]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    for col in ["vendor", "product", "version"]:
        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # One row per CVE
    df = df.drop_duplicates(
        subset=["cve_id"],
        keep="first",
    )

    print("Rows after NVD filtering:", len(df))

    # 4. Mask vendor/product/version
    print("Masking vendor/product/version...")

    df["masked_description"] = df.progress_apply(
        mask_description,
        axis=1,
    )

    # 5. Clean / lemmatize masked text
    print("Cleaning masked descriptions...")
    df = clean_descriptions(df)

    # 6. Validate CWE + map to CAPEC
    print("Mapping CWE -> CAPEC...")
    df = add_capec_labels(df)

    # 7. Final dataset
    df = df.rename(
        columns={
            "description": "uncleaned_description"
        }
    )

    df = df[
        [
            "cve_id",
            "weakness",
            "capec_id",
            "uncleaned_description",
            "cleaned_description",
        ]
    ].copy()

    # Remove empty cleaned descriptions
    df = df[
        df["cleaned_description"].notna()
        & df["cleaned_description"].str.strip().ne("")
    ].copy()

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print("\nSaved:", OUTPUT_FILE)
    print("Final shape:", df.shape)
    print("Final columns:", df.columns.tolist())
    print(df.head())


if __name__ == "__main__":
    main()
