import requests
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
import json
import re
from io import BytesIO
from IPython.display import display
# ============================================================
# CONFIG
# ============================================================

URL = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"

OUTPUT_EMBEDDING = "cwe_for_embedding.csv"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def collapse_consecutive_duplicate_tokens(text):
    """Collapse adjacent repeated tokens without removing later occurrences."""
    value = " ".join(str(text).split())
    return re.sub(
        r"\b([A-Za-z0-9_+\-]+)(?:\s+\1\b)+",
        r"\1",
        value,
        flags=re.IGNORECASE,
    )


def clean_text(text):
    if text is None:
        return ""
    return collapse_consecutive_duplicate_tokens(text)

def get_all_text(elem):
    """
    Extract text from an XML element including nested tags.
    """
    if elem is None:
        return ""
    return clean_text(" ".join(elem.itertext()))

def to_json(obj):
    """
    Store list/dict safely inside CSV.
    """
    return json.dumps(obj, ensure_ascii=False)

# ============================================================
# DOWNLOAD CWE XML ZIP
# ============================================================

print("Downloading CWE data...")

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(URL, headers=headers, timeout=60)
response.raise_for_status()

print("Download complete.")

# ============================================================
# EXTRACT XML
# ============================================================

print("Extracting ZIP...")

zip_file = zipfile.ZipFile(BytesIO(response.content))
xml_files = [name for name in zip_file.namelist() if name.endswith(".xml")]

if not xml_files:
    raise ValueError("No XML file found inside ZIP.")

xml_name = xml_files[0]
xml_data = zip_file.read(xml_name)

print("XML file found:", xml_name)

# ============================================================
# PARSE XML
# ============================================================

print("Parsing XML...")

root = ET.fromstring(xml_data)

namespace = root.tag.split("}")[0].strip("{")
ns = {"cwe": namespace}

rows = []

# ============================================================
# EXTRACT CWE WEAKNESSES
# ============================================================

for weakness in root.findall(".//cwe:Weakness", ns):

    cwe_num = weakness.attrib.get("ID", "")
    cwe_id = f"CWE-{cwe_num}"

    cwe_name = weakness.attrib.get("Name", "")
    abstraction = weakness.attrib.get("Abstraction", "")
    structure = weakness.attrib.get("Structure", "")
    status = weakness.attrib.get("Status", "")

    # --------------------------------------------------------
    # Basic descriptions
    # --------------------------------------------------------

    description = get_all_text(weakness.find("cwe:Description", ns))
    extended_description = get_all_text(weakness.find("cwe:Extended_Description", ns))

    # --------------------------------------------------------
    # Related weaknesses
    # --------------------------------------------------------

    related_weaknesses = []

    for rel in weakness.findall(".//cwe:Related_Weakness", ns):
        related_cwe_num = rel.attrib.get("CWE_ID", "")

        if related_cwe_num:
            related_weaknesses.append({
                "nature": rel.attrib.get("Nature", ""),
                "cwe_id": f"CWE-{related_cwe_num}",
                "view_id": rel.attrib.get("View_ID", ""),
                "ordinal": rel.attrib.get("Ordinal", "")
            })

    related_cwe_ids = sorted(
        set(item["cwe_id"] for item in related_weaknesses if item["cwe_id"])
    )

    # --------------------------------------------------------
    # CAPEC related attack patterns
    # --------------------------------------------------------

    capec_ids = []

    for capec in weakness.findall(".//cwe:Related_Attack_Pattern", ns):
        capec_id = capec.attrib.get("CAPEC_ID", "")

        if capec_id:
            capec_ids.append(f"CAPEC-{capec_id}")

    capec_ids = sorted(set(capec_ids))

    # --------------------------------------------------------
    # Common consequences
    # IMPORTANT:
    # Correct path is Common_Consequences -> Consequence
    # --------------------------------------------------------

    common_consequences = []

    for cons in weakness.findall(".//cwe:Common_Consequences/cwe:Consequence", ns):

        scopes = [
            get_all_text(scope)
            for scope in cons.findall("cwe:Scope", ns)
            if get_all_text(scope)
        ]

        impacts = [
            get_all_text(impact)
            for impact in cons.findall("cwe:Impact", ns)
            if get_all_text(impact)
        ]

        notes = get_all_text(cons.find("cwe:Note", ns))

        common_consequences.append({
            "scopes": scopes,
            "impacts": impacts,
            "notes": notes
        })

    # --------------------------------------------------------
    # Potential mitigations
    # --------------------------------------------------------

    mitigations = []

    for mit in weakness.findall(".//cwe:Potential_Mitigations/cwe:Mitigation", ns):

        phase = get_all_text(mit.find("cwe:Phase", ns))
        strategy = get_all_text(mit.find("cwe:Strategy", ns))
        description_mit = get_all_text(mit.find("cwe:Description", ns))
        effectiveness = get_all_text(mit.find("cwe:Effectiveness", ns))

        mitigations.append({
            "phase": phase,
            "strategy": strategy,
            "description": description_mit,
            "effectiveness": effectiveness
        })

    # --------------------------------------------------------
    # Modes of introduction
    # --------------------------------------------------------

    modes_of_introduction = []

    for mode in weakness.findall(".//cwe:Modes_Of_Introduction/cwe:Introduction", ns):

        phase = get_all_text(mode.find("cwe:Phase", ns))
        note = get_all_text(mode.find("cwe:Note", ns))

        modes_of_introduction.append({
            "phase": phase,
            "note": note
        })

    # --------------------------------------------------------
    # Detection methods
    # --------------------------------------------------------

    detection_methods = []

    for det in weakness.findall(".//cwe:Detection_Methods/cwe:Detection_Method", ns):

        method = get_all_text(det.find("cwe:Method", ns))
        description_det = get_all_text(det.find("cwe:Description", ns))
        effectiveness = get_all_text(det.find("cwe:Effectiveness", ns))

        detection_methods.append({
            "method": method,
            "description": description_det,
            "effectiveness": effectiveness
        })

    # --------------------------------------------------------
    # Observed examples
    # --------------------------------------------------------

    observed_examples = []

    for obs in weakness.findall(".//cwe:Observed_Examples/cwe:Observed_Example", ns):

        reference = get_all_text(obs.find("cwe:Reference", ns))
        description_obs = get_all_text(obs.find("cwe:Description", ns))
        link = get_all_text(obs.find("cwe:Link", ns))

        observed_examples.append({
            "reference": reference,
            "description": description_obs,
            "link": link
        })

    # --------------------------------------------------------
    # Weakness ordinalities
    # --------------------------------------------------------

    weakness_ordinalities = []

    for ord_elem in weakness.findall(".//cwe:Weakness_Ordinalities/cwe:Weakness_Ordinality", ns):

        ordinality = get_all_text(ord_elem.find("cwe:Ordinality", ns))
        description_ord = get_all_text(ord_elem.find("cwe:Description", ns))

        weakness_ordinalities.append({
            "ordinality": ordinality,
            "description": description_ord
        })

    # --------------------------------------------------------
    # Applicable platforms
    # --------------------------------------------------------

    applicable_platforms = []

    for platform in weakness.findall(".//cwe:Applicable_Platforms/*", ns):
        applicable_platforms.append({
            "tag": platform.tag.split("}")[-1],
            "name": platform.attrib.get("Name", ""),
            "prevalence": platform.attrib.get("Prevalence", "")
        })

    # --------------------------------------------------------
    # Full row
    # --------------------------------------------------------

    rows.append({
        "cwe_id": cwe_id,
        "cwe_name": cwe_name,
        "abstraction": abstraction,
        "structure": structure,
        "status": status,

        "description": description,
        "extended_description": extended_description,

        # flattened useful columns
        "related_cwe_ids": "; ".join(related_cwe_ids),
        "capec_ids": ", ".join(capec_ids),

        # JSON full columns
        "related_weaknesses_json": to_json(related_weaknesses),
        "common_consequences_json": to_json(common_consequences),
        "mitigations_json": to_json(mitigations),
        "modes_of_introduction_json": to_json(modes_of_introduction),
        "detection_methods_json": to_json(detection_methods),
        "observed_examples_json": to_json(observed_examples),
        "weakness_ordinalities_json": to_json(weakness_ordinalities),
        "applicable_platforms_json": to_json(applicable_platforms)
    })

# ============================================================
# BUILD FULL CWE DATA IN MEMORY
# ============================================================

source_cwe = pd.DataFrame(rows)
df = source_cwe.copy()

print("\nFull CWE data prepared in memory.")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# ============================================================
# CHECK COMMON CONSEQUENCES
# ============================================================

non_empty_consequences = df["common_consequences_json"].apply(lambda x: x != "[]").sum()

print("\nRows with common consequences:", non_empty_consequences)
print("Rows without common consequences:", len(df) - non_empty_consequences)

# ============================================================
# CREATE EMBEDDING FILE
# ============================================================

df["cwe_text"] = (
    df["cwe_name"].fillna("") + ". " +
    df["description"].fillna("")
)


# ============================================================
# PREVIEW
# ============================================================

print("\nPreview:")
display(df.head())

print("\nColumns:")
print(df.columns.tolist())

import pandas as pd
import re


import pandas as pd
import json

# ============================================================
# CONTINUE WITH IN-MEMORY CWE DATA
# ============================================================


# ============================================================
# HELPERS
# ============================================================

def load_json(x):
    if pd.isna(x) or str(x).strip() == "":
        return []

    try:
        return json.loads(x)
    except:
        return []


def unique_join(values, sep=", "):
    """
    Remove blanks and duplicates while preserving order.
    """
    result = []
    seen = set()

    for value in values:
        if value is None:
            continue

        value = str(value).strip()

        if value and value not in seen:
            seen.add(value)
            result.append(value)

    return sep.join(result)


# ============================================================
# PARENT / CHILD CWE IDS
# ============================================================

def get_parent_cwes(x):
    relations = load_json(x)

    parents = []

    for rel in relations:
        nature = rel.get("nature", "")
        cwe_id = rel.get("cwe_id", "")

        # Current CWE is a child of this CWE
        if nature == "ChildOf" and cwe_id:
            parents.append(cwe_id)

    return unique_join(parents)


def get_child_cwes(x):
    relations = load_json(x)

    children = []

    for rel in relations:
        nature = rel.get("nature", "")
        cwe_id = rel.get("cwe_id", "")

        # Current CWE is a parent of this CWE
        if nature == "ParentOf" and cwe_id:
            children.append(cwe_id)

    return unique_join(children)


df["parent_cwe_ids"] = df["related_weaknesses_json"].apply(get_parent_cwes)
df["child_cwe_ids"] = df["related_weaknesses_json"].apply(get_child_cwes)


# ============================================================
# CONSEQUENCES
# ============================================================

def extract_consequences(x):
    items = load_json(x)

    scopes = []
    impacts = []
    notes = []

    for item in items:

        scopes.extend(item.get("scopes", []))
        impacts.extend(item.get("impacts", []))

        note = item.get("notes", "")
        if note:
            notes.append(note)

    scopes_text = unique_join(scopes)
    impacts_text = unique_join(impacts)
    notes_text = unique_join(notes, sep=" | ")

    combined = []

    if scopes_text:
        combined.append("Scopes: " + scopes_text)

    if impacts_text:
        combined.append("Impacts: " + impacts_text)

    if notes_text:
        combined.append("Notes: " + notes_text)

    return pd.Series({
        "consequence_scopes": scopes_text,
        "consequence_impacts": impacts_text,
        "consequence_notes": notes_text,
        "consequence_text": " | ".join(combined)
    })


consequence_cols = df["common_consequences_json"].apply(
    extract_consequences
)

df = pd.concat([df, consequence_cols], axis=1)


# ============================================================
# MITIGATIONS
# ============================================================

def extract_mitigations(x):
    items = load_json(x)

    phases = []
    strategies = []
    effectiveness = []
    descriptions = []

    for item in items:

        if item.get("phase"):
            phases.append(item["phase"])

        if item.get("strategy"):
            strategies.append(item["strategy"])

        if item.get("effectiveness"):
            effectiveness.append(item["effectiveness"])

        if item.get("description"):
            descriptions.append(item["description"])

    return pd.Series({
        "mitigation_phases": unique_join(phases),
        "mitigation_strategies": unique_join(strategies),
        "mitigation_effectiveness": unique_join(effectiveness),
        "mitigation_text": unique_join(descriptions, sep=" | ")
    })


mitigation_cols = df["mitigations_json"].apply(
    extract_mitigations
)

df = pd.concat([df, mitigation_cols], axis=1)


# ============================================================
# DETECTION METHODS
# ============================================================

def extract_detection(x):
    items = load_json(x)

    methods = []
    effectiveness = []
    descriptions = []

    for item in items:

        if item.get("method"):
            methods.append(item["method"])

        if item.get("effectiveness"):
            effectiveness.append(item["effectiveness"])

        if item.get("description"):
            descriptions.append(item["description"])

    return pd.Series({
        "detection_methods": unique_join(methods),
        "detection_effectiveness": unique_join(effectiveness),
        "detection_text": unique_join(descriptions, sep=" | ")
    })


detection_cols = df["detection_methods_json"].apply(
    extract_detection
)

df = pd.concat([df, detection_cols], axis=1)


# ============================================================
# KEEP ONLY FINAL COLUMNS
# ============================================================

final_columns = [
    "cwe_id",
    "cwe_name",
    "description",
    "extended_description",
    "capec_ids",

    "parent_cwe_ids",
    "child_cwe_ids",

    "consequence_scopes",
    "consequence_impacts",
    "consequence_notes",
    "consequence_text",

    "mitigation_phases",
    "mitigation_strategies",
    "mitigation_effectiveness",
    "mitigation_text",

    "detection_methods",
    "detection_effectiveness",
    "detection_text"
]

cwe_final = df[final_columns].copy()


# ============================================================
# SAVE
# ============================================================

cwe_final.to_csv(
    "cwe_final.csv",
    index=False,
    encoding="utf-8"
)

print("Saved: cwe_final.csv")
print("Shape:", cwe_final.shape)

display(cwe_final.head())


# ============================================================
# LOAD FINAL FILE
# ============================================================

cwe_final = pd.read_csv("cwe_final.csv")

print("Source CWE shape:", source_cwe.shape)
print("cwe_final shape before removing deprecated:", cwe_final.shape)

# ============================================================
# GET DEPRECATED CWE LIST
# ============================================================

deprecated_df = source_cwe[
    source_cwe["status"].fillna("").astype(str).str.lower().str.strip() == "deprecated"
].copy()

deprecated_ids = set(deprecated_df["cwe_id"].astype(str).str.strip())

print("Deprecated CWE count:", len(deprecated_ids))

display(deprecated_df[[
    "cwe_id",
    "cwe_name",
    "status",
    "description"
]].head(30))

# Save deprecated CWE list
deprecated_df.to_csv("deprecated_cwes.csv", index=False, encoding="utf-8")

print("Saved deprecated list as: deprecated_cwes.csv")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def split_ids(x):
    """
    Split comma-separated CWE/CAPEC ids safely.
    """
    if pd.isna(x):
        return []

    x = str(x).strip()

    if x == "" or x.lower() == "nan":
        return []

    parts = re.split(r"\s*,\s*", x)
    return [p.strip() for p in parts if p.strip()]


def clean_relation_ids(x, deprecated_ids):
    """
    Remove deprecated CWE ids from parent_cwe_ids / child_cwe_ids.
    """
    ids = split_ids(x)
    ids = [i for i in ids if i not in deprecated_ids]
    return ", ".join(ids)

# ============================================================
# REMOVE DEPRECATED CWE ROWS FROM CWE_FINAL
# ============================================================

cwe_final_no_deprecated = cwe_final[
    ~cwe_final["cwe_id"].astype(str).str.strip().isin(deprecated_ids)
].copy()

# ============================================================
# REMOVE DEPRECATED IDS FROM PARENT/CHILD COLUMNS ALSO
# ============================================================

for col in ["parent_cwe_ids", "child_cwe_ids"]:
    if col in cwe_final_no_deprecated.columns:
        cwe_final_no_deprecated[col] = cwe_final_no_deprecated[col].apply(
            lambda x: clean_relation_ids(x, deprecated_ids)
        )

# ============================================================
# SAVE FINAL CLEAN FILE
# ============================================================

cwe_final_no_deprecated.to_csv("cwe_final.csv", index=False, encoding="utf-8")

print("\nOverwritten clean final file: cwe_final.csv")
print("Shape after removing deprecated:", cwe_final_no_deprecated.shape)

print("\nRows removed:", cwe_final.shape[0] - cwe_final_no_deprecated.shape[0])

print("\nRemaining deprecated rows check:")
remaining = cwe_final_no_deprecated[
    cwe_final_no_deprecated["cwe_id"].isin(deprecated_ids)
]
print("Remaining deprecated rows:", len(remaining))

display(cwe_final_no_deprecated.head())



import pandas as pd
import re

# ============================================================
# LOAD FINAL CWE FILE
# ============================================================

cwe_final = pd.read_csv("cwe_final.csv")

# ============================================================
# SPLIT CAPEC IDS
# ============================================================

def split_capecs(x):
    if pd.isna(x):
        return []

    x = str(x).strip()

    if x == "" or x.lower() == "nan":
        return []

    # Works for comma-separated or semicolon-separated values
    parts = re.split(r"\s*[,;]\s*", x)

    # Remove empty + duplicates while preserving order
    seen = set()
    capecs = []

    for p in parts:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            capecs.append(p)

    return capecs

# ============================================================
# CREATE ONE ROW PER CWE
# ============================================================

mapped_rows = []

for _, row in cwe_final.iterrows():
    cwe_id = row["cwe_id"]
    capecs = split_capecs(row.get("capec_ids", ""))

    mapped_rows.append({
        "cwe_id": cwe_id,
        "capec_ids": ", ".join(capecs),
        "num_capecs": len(capecs),
        "has_capec_mapping": len(capecs) > 0
    })

CWE_MAPPED = pd.DataFrame(mapped_rows).drop_duplicates(subset=["cwe_id"])

# ============================================================
# COUNTS
# ============================================================

total_cwes = CWE_MAPPED["cwe_id"].nunique()
cwes_with_capec = CWE_MAPPED["has_capec_mapping"].sum()
cwes_without_capec = total_cwes - cwes_with_capec

# Explode only for counting unique mappings and unique CAPECs
exploded = CWE_MAPPED.copy()
exploded["capec_id"] = exploded["capec_ids"].apply(split_capecs)
exploded = exploded.explode("capec_id")
exploded = exploded[exploded["capec_id"].notna() & (exploded["capec_id"] != "")]
exploded = exploded[["cwe_id", "capec_id"]].drop_duplicates()

total_unique_mappings = len(exploded)
unique_capecs_mapped = exploded["capec_id"].nunique()

print("Total CWEs:", total_cwes)
print("CWEs with at least one CAPEC:", cwes_with_capec)
print("CWEs without CAPEC mapping:", cwes_without_capec)
print("Total unique CWE -> CAPEC mappings:", total_unique_mappings)
print("Unique CAPECs mapped:", unique_capecs_mapped)

# ============================================================
# SAVE
# ============================================================
CWE_MAPPED = CWE_MAPPED[CWE_MAPPED["capec_ids"].fillna("").astype(str).str.strip() != ""].copy()
CWE_MAPPED = CWE_MAPPED[["cwe_id", "capec_ids"]].copy()
CWE_MAPPED.to_csv("CWE_MAPPED.csv", index=False, encoding="utf-8")

print("\nSaved as: CWE_MAPPED.csv")
print("Shape:", CWE_MAPPED.shape)

display(CWE_MAPPED.head())



import pandas as pd
import re

# ============================================================
# LOAD FINAL CWE FILE
# ============================================================

cwe_final = pd.read_csv("cwe_final.csv")

# ============================================================
# SPLIT CAPEC IDS
# ============================================================

def split_capecs(x):
    if pd.isna(x):
        return []

    x = str(x).strip()

    if x == "" or x.lower() == "nan":
        return []

    # works for comma-separated or semicolon-separated just in case
    parts = re.split(r"\s*[,;]\s*", x)
    return [p.strip() for p in parts if p.strip()]

# ============================================================
# CREATE CWE-CAPEC MAPPING TABLE
# ============================================================

mapping_rows = []

for _, row in cwe_final.iterrows():
    cwe_id = row["cwe_id"]
    capecs = split_capecs(row.get("capec_ids", ""))

    for capec_id in capecs:
        mapping_rows.append({
            "cwe_id": cwe_id,
            "capec_id": capec_id
        })

cwe_capec_mapping = pd.DataFrame(mapping_rows).drop_duplicates()

# ============================================================
# COUNTS
# ============================================================

total_mappings = len(cwe_capec_mapping)
unique_cwes_with_capec = cwe_capec_mapping["cwe_id"].nunique()
unique_capecs = cwe_capec_mapping["capec_id"].nunique()

total_cwes = cwe_final["cwe_id"].nunique()
cwes_without_capec = total_cwes - unique_cwes_with_capec

print("Total unique CWE -> CAPEC mappings:", total_mappings)
print("Unique CWEs with at least one CAPEC:", unique_cwes_with_capec)
print("Unique CAPECs mapped:", unique_capecs)
print("Total CWEs in cwe_final:", total_cwes)
print("CWEs without CAPEC mapping:", cwes_without_capec)

# Save mapping table
cwe_capec_mapping.to_csv("cwe_capec_mapping.csv", index=False, encoding="utf-8")

print("\nSaved mapping file as: cwe_capec_mapping.csv")

display(cwe_capec_mapping.head())


# ============================================================
# CREATE CAPEC PARENT/CHILD RELATIONSHIPS FOR RAE-XMC
# ============================================================

def build_capec_relationships(
    capec_catalog_path="3000_capec.csv",
    output_path="capec_relationships.csv"
):
    """
    Build the CAPEC hierarchy from the current stage-01 CAPEC catalog.

    A ``ChildOf`` relation gives the current CAPEC's parent. Child lists are
    generated by reversing those same edges, so both directions remain
    consistent. Other relationship natures (for example ``CanPrecede``) are
    intentionally not hierarchy edges.
    """
    capec_df = pd.read_csv(capec_catalog_path)

    required_columns = {"ID", "Name", "Related Attack Patterns"}
    missing_columns = required_columns - set(capec_df.columns)
    if missing_columns:
        raise ValueError(
            "Missing CAPEC catalog columns: " + ", ".join(sorted(missing_columns))
        )

    def normalize_capec_id(value):
        match = re.search(r"(\d+)", str(value))
        return f"CAPEC-{int(match.group(1))}" if match else None

    def capec_sort_key(capec_id):
        match = re.search(r"(\d+)$", str(capec_id))
        return int(match.group(1)) if match else float("inf")

    relation_pattern = re.compile(
        r"NATURE\s*:\s*([^:]+?)\s*:\s*CAPEC ID\s*:\s*(\d+)",
        flags=re.IGNORECASE
    )

    names = {}
    parents = {}
    children = {}

    for _, row in capec_df.iterrows():
        capec_id = normalize_capec_id(row["ID"])
        if capec_id is None:
            continue

        names[capec_id] = clean_text(row["Name"]).lower()
        parents.setdefault(capec_id, set())
        children.setdefault(capec_id, set())

    for _, row in capec_df.iterrows():
        capec_id = normalize_capec_id(row["ID"])
        if capec_id is None:
            continue

        relation_text = "" if pd.isna(row["Related Attack Patterns"]) else str(
            row["Related Attack Patterns"]
        )

        for nature, related_number in relation_pattern.findall(relation_text):
            related_id = f"CAPEC-{int(related_number)}"
            normalized_nature = nature.strip().lower()

            if normalized_nature == "childof":
                parents[capec_id].add(related_id)
                children.setdefault(related_id, set()).add(capec_id)
            elif normalized_nature == "parentof":
                children[capec_id].add(related_id)
                parents.setdefault(related_id, set()).add(capec_id)

    relationship_rows = []
    for capec_id in sorted(names, key=capec_sort_key):
        relationship_rows.append({
            "capec": capec_id,
            "name": names[capec_id],
            "parent": "; ".join(sorted(parents[capec_id], key=capec_sort_key)),
            "child": "; ".join(sorted(children[capec_id], key=capec_sort_key)),
        })

    relationships_df = pd.DataFrame(
        relationship_rows,
        columns=["capec", "name", "parent", "child"]
    )
    relationships_df.to_csv(output_path, index=False, encoding="utf-8")

    print("\nSaved CAPEC relationships as:", output_path)
    print("Shape:", relationships_df.shape)
    print("Rows with a parent:", int((relationships_df["parent"] != "").sum()))
    print("Rows with a child:", int((relationships_df["child"] != "").sum()))

    return relationships_df


capec_relationships = build_capec_relationships()
display(capec_relationships.head())
