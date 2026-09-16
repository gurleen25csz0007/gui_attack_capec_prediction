#!/usr/bin/env python3
"""Generate publication-quality CVE -> CWE -> CAPEC dataset figures.

All reported values are calculated from the configured CSV files.  The default
configuration matches this repository's NVD exports and CWE/CAPEC mapping.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

# A non-interactive backend makes the script safe to run on servers/HPC nodes.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION: change only this section for a different dataset
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# One glob can cover several yearly files.  A single CSV path also works.
CVE_FILES = SCRIPT_DIR.parent / "00_input_data" / "nvdcve-2.0-*.csv"
CWE_CAPEC_FILE = SCRIPT_DIR / "cwe_capec_mapping.csv"
OUTPUT_DIR = SCRIPT_DIR / "image"

# Columns in the CVE CSV file(s).
CVE_ID_COL = "cve.id"
CWE_COL = "cve.weaknesses"

# Columns in the CWE -> CAPEC mapping CSV.
MAPPING_CWE_COL = "cwe_id"
CAPEC_COL = "capec_id"

PNG_DPI = 600
TOP_BOTTOM_N = 30


# =============================================================================
# PARSING AND NORMALIZATION
# =============================================================================

_CWE_PATTERN = re.compile(r"(?i)(?<![A-Z0-9])CWE[\s_:\-]*(\d+)(?!\d)")
_CAPEC_PATTERN = re.compile(r"(?i)(?<![A-Z0-9])CAPEC[\s_:\-]*(\d+)(?!\d)")
_CVE_PATTERN = re.compile(r"(?i)^CVE-(\d{4})-(\d{4,})$")
_NULL_TEXT = {"", "nan", "none", "null", "na", "n/a", "[]", "{}"}


def _is_missing(value: object) -> bool:
    """Return True for scalar missing values without failing on containers."""
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict, np.ndarray)):
        return False
    try:
        result = pd.isna(value)
        return bool(result) if np.isscalar(result) else False
    except (TypeError, ValueError):
        return False


def parse_id_list(value: object) -> list[str]:
    """Safely flatten IDs from scalars or Python-list/dict-like strings.

    ``ast.literal_eval`` handles strings such as ``"['CWE-79', 'CWE-89']"``
    and the nested list/dict representation in the NVD ``cve.weaknesses``
    column.  If a string is not a Python literal, common CSV delimiters are
    used as a fallback.  No ``eval`` is used.
    """
    if _is_missing(value):
        return []

    if isinstance(value, dict):
        flattened: list[str] = []
        for nested_value in value.values():
            flattened.extend(parse_id_list(nested_value))
        return flattened

    if isinstance(value, (list, tuple, set, np.ndarray)):
        flattened = []
        for item in value:
            flattened.extend(parse_id_list(item))
        return flattened

    text = str(value).strip()
    if text.lower() in _NULL_TEXT:
        return []

    # literal_eval is deliberately attempted for every nonempty string.  It is
    # safe and supports list-like, tuple-like, dict-like, and quoted values.
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        parsed = None

    if parsed is not None and not (isinstance(parsed, str) and parsed == text):
        return parse_id_list(parsed)

    pieces = [piece.strip().strip("'\"") for piece in re.split(r"[,;|]", text)]
    return [piece for piece in pieces if piece.lower() not in _NULL_TEXT]


def normalize_cwe(value: object) -> str | None:
    """Return a canonical CWE ID (for example, CWE-79), or None if invalid."""
    if _is_missing(value):
        return None
    text = str(value).strip()
    if text.lower() in {"nvd-cwe-noinfo", "nvd-cwe-other"}:
        return None
    match = _CWE_PATTERN.fullmatch(text)
    return f"CWE-{int(match.group(1))}" if match else None


def normalize_capec(value: object) -> str | None:
    """Return a canonical CAPEC ID (for example, CAPEC-63), or None."""
    if _is_missing(value):
        return None
    match = _CAPEC_PATTERN.fullmatch(str(value).strip())
    return f"CAPEC-{int(match.group(1))}" if match else None


def normalize_cve(value: object) -> str | None:
    """Return a canonical CVE ID, or None for an invalid/empty value."""
    if _is_missing(value):
        return None
    match = _CVE_PATTERN.fullmatch(str(value).strip())
    return f"CVE-{match.group(1)}-{match.group(2)}" if match else None


def _extract_ids(value: object, normalizer: Callable[[object], str | None]) -> list[str]:
    """Extract, normalize, and locally deduplicate identifiers from one cell."""
    return sorted({item for token in parse_id_list(value) if (item := normalizer(token))})


# =============================================================================
# CLEAN INTERMEDIATE TABLES
# =============================================================================

def _resolve_cve_files(path_or_pattern: Path | str | Iterable[Path | str]) -> list[Path]:
    if isinstance(path_or_pattern, (str, Path)):
        candidate = Path(path_or_pattern)
        if any(character in str(candidate) for character in "*?["):
            files = sorted(candidate.parent.glob(candidate.name))
        else:
            files = [candidate]
    else:
        files = sorted(Path(item) for item in path_or_pattern)

    existing_files = [path for path in files if path.is_file()]
    if not existing_files:
        raise FileNotFoundError(f"No CVE CSV files matched: {path_or_pattern}")
    return existing_files


def load_and_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only the configured columns from all source CSV files."""
    cve_frames: list[pd.DataFrame] = []
    for file_path in _resolve_cve_files(CVE_FILES):
        print(f"Reading CVE data: {file_path}")
        frame = pd.read_csv(
            file_path,
            usecols=[CVE_ID_COL, CWE_COL],
            dtype={CVE_ID_COL: "string", CWE_COL: "string"},
            low_memory=False,
        )
        cve_frames.append(frame)

    raw_cve = pd.concat(cve_frames, ignore_index=True)

    print(f"Reading CWE-CAPEC data: {CWE_CAPEC_FILE}")
    raw_mapping = pd.read_csv(
        CWE_CAPEC_FILE,
        usecols=[MAPPING_CWE_COL, CAPEC_COL],
        dtype={MAPPING_CWE_COL: "string", CAPEC_COL: "string"},
        low_memory=False,
    )
    return raw_cve, raw_mapping


def build_cve_cwe_table(raw_cve: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    """Build the deduplicated ``CVE_ID | CWE_ID`` relationship table."""
    rows: list[tuple[str, str]] = []
    all_cves: set[str] = set()

    for cve_value, cwe_value in raw_cve[[CVE_ID_COL, CWE_COL]].itertuples(
        index=False, name=None
    ):
        cve_id = normalize_cve(cve_value)
        if cve_id is None:
            continue
        all_cves.add(cve_id)
        rows.extend((cve_id, cwe_id) for cwe_id in _extract_ids(cwe_value, normalize_cwe))

    cve_cwe = (
        pd.DataFrame(rows, columns=["CVE_ID", "CWE_ID"])
        .dropna()
        .drop_duplicates(["CVE_ID", "CWE_ID"])
        .sort_values(["CVE_ID", "CWE_ID"], kind="stable")
        .reset_index(drop=True)
    )
    return cve_cwe, pd.Index(sorted(all_cves), name="CVE_ID")


def build_cwe_capec_table(raw_mapping: pd.DataFrame) -> pd.DataFrame:
    """Build the deduplicated ``CWE_ID | CAPEC_ID`` relationship table."""
    rows: list[tuple[str, str]] = []
    for cwe_value, capec_value in raw_mapping[[MAPPING_CWE_COL, CAPEC_COL]].itertuples(
        index=False, name=None
    ):
        cwe_ids = _extract_ids(cwe_value, normalize_cwe)
        capec_ids = _extract_ids(capec_value, normalize_capec)
        rows.extend((cwe_id, capec_id) for cwe_id in cwe_ids for capec_id in capec_ids)

    return (
        pd.DataFrame(rows, columns=["CWE_ID", "CAPEC_ID"])
        .dropna()
        .drop_duplicates(["CWE_ID", "CAPEC_ID"])
        .sort_values(["CWE_ID", "CAPEC_ID"], kind="stable")
        .reset_index(drop=True)
    )


def build_cve_capec_table(cve_cwe: pd.DataFrame, cwe_capec: pd.DataFrame) -> pd.DataFrame:
    """Join CVE -> CWE -> CAPEC and deduplicate CVE-CAPEC paths."""
    return (
        cve_cwe.merge(cwe_capec, on="CWE_ID", how="inner", validate="many_to_many")
        [["CVE_ID", "CAPEC_ID"]]
        .drop_duplicates(["CVE_ID", "CAPEC_ID"])
        .sort_values(["CVE_ID", "CAPEC_ID"], kind="stable")
        .reset_index(drop=True)
    )


def validate_tables(
    cve_cwe: pd.DataFrame,
    cwe_capec: pd.DataFrame,
    cve_capec: pd.DataFrame,
    all_cves: pd.Index,
) -> None:
    """Assert relationship uniqueness and print auditable validation totals."""
    assert not cve_cwe.duplicated(["CVE_ID", "CWE_ID"]).any()
    assert not cwe_capec.duplicated(["CWE_ID", "CAPEC_ID"]).any()
    assert not cve_capec.duplicated(["CVE_ID", "CAPEC_ID"]).any()
    assert cve_capec["CVE_ID"].isin(all_cves).all()

    mapped_cves = cve_capec["CVE_ID"].nunique()
    total_cves = all_cves.nunique()

    print("\nValidation statistics")
    print("=" * 72)
    print(f"Total unique CVEs: {total_cves:,}")
    print(f"Total unique valid CWEs in CVE dataset: {cve_cwe['CWE_ID'].nunique():,}")
    print(f"Total unique CWEs having CAPEC mappings: {cwe_capec['CWE_ID'].nunique():,}")
    print(f"Total unique CAPEC IDs in mapping dataset: {cwe_capec['CAPEC_ID'].nunique():,}")
    print(f"Total unique CAPEC IDs reached from CVEs: {cve_capec['CAPEC_ID'].nunique():,}")
    print(f"Total unique CVE-CWE relationships: {len(cve_cwe):,}")
    print(f"Total unique CWE-CAPEC relationships: {len(cwe_capec):,}")
    print(f"Total unique CVE-CAPEC relationships: {len(cve_capec):,}")
    print(f"Number of CVEs having at least one mapped CAPEC: {mapped_cves:,}")
    print(f"Number of CVEs having no mapped CAPEC: {total_cves - mapped_cves:,}")


# =============================================================================
# STATISTICS AND PLOTS
# =============================================================================

def _set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 14,
            "font.weight": "bold",
            "mathtext.default": "bf",
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 16,
            "axes.labelweight": "bold",
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "legend.title_fontsize": 14,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(fig: plt.Figure, stem: str) -> None:
    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path.name} and {pdf_path.name}")


def _bold_tick_values(ax: plt.Axes) -> None:
    """Make all X- and Y-axis tick values bold for paper readability."""
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight("bold")


def _frequency_bin(counts: pd.Series) -> pd.Categorical:
    labels = ["0", "1-10", "11-50", "51-100", "101-500", "501-1000", "1001-5000", "5000+"]
    values = counts.to_numpy()
    assigned = np.select(
        [
            values == 0,
            (values >= 1) & (values <= 10),
            (values >= 11) & (values <= 50),
            (values >= 51) & (values <= 100),
            (values >= 101) & (values <= 500),
            (values >= 501) & (values <= 1000),
            (values >= 1001) & (values <= 5000),
            values > 5000,
        ],
        labels,
        default="0",
    )
    return pd.Categorical(assigned, categories=labels, ordered=True)


def plot_cwe_frequency_mapping_availability(
    cve_cwe: pd.DataFrame, cwe_capec: pd.DataFrame
) -> pd.DataFrame:
    """Create Figure 3: CWE CVE-frequency bins split by mapping availability."""
    cwe_frequency = cve_cwe.groupby("CWE_ID")["CVE_ID"].nunique()

    # The union is necessary for the explicit zero-frequency bin: it includes
    # CWEs known by the mapping dataset but absent from the CVE dataset.
    cwe_universe = pd.Index(
        sorted(set(cve_cwe["CWE_ID"]) | set(cwe_capec["CWE_ID"])), name="CWE_ID"
    )
    details = pd.DataFrame(index=cwe_universe)
    details["CVE_Count"] = cwe_frequency.reindex(cwe_universe, fill_value=0).astype(int)
    mapped_cwes = set(cwe_capec["CWE_ID"])
    details["Mapped"] = details.index.isin(mapped_cwes)
    details["Frequency_Bin"] = _frequency_bin(details["CVE_Count"])

    labels = list(details["Frequency_Bin"].cat.categories)
    grouped = details.groupby(["Frequency_Bin", "Mapped"], observed=False).size().unstack(fill_value=0)
    grouped = grouped.reindex(index=labels, fill_value=0).reindex(columns=[True, False], fill_value=0)
    stats = pd.DataFrame(
        {
            "Frequency_Bin": labels,
            "CWE_Mapped_to_CAPEC": grouped[True].to_numpy(dtype=int),
            "CWE_Not_Mapped_to_CAPEC": grouped[False].to_numpy(dtype=int),
        }
    )
    stats["Total_CWEs"] = stats["CWE_Mapped_to_CAPEC"] + stats["CWE_Not_Mapped_to_CAPEC"]

    print("\nFigure 3 dataframe")
    print("=" * 72)
    print(stats.to_string(index=False))
    stats.to_csv(OUTPUT_DIR / "cwe_frequency_capec_availability.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(stats))
    mapped = stats["CWE_Mapped_to_CAPEC"].to_numpy()
    unmapped = stats["CWE_Not_Mapped_to_CAPEC"].to_numpy()
    total = stats["Total_CWEs"].to_numpy()

    bars_mapped = ax.bar(x, mapped, width=0.72, color="#2F6B9A", label="CWE mapped to CAPEC")
    bars_unmapped = ax.bar(
        x, unmapped, width=0.72, bottom=mapped, color="#D98C3F", label="CWE not mapped to CAPEC"
    )

    small_segment_limit = float(total.max()) * 0.035
    for bar, count in zip(bars_mapped, mapped):
        if count > 0:
            if count < small_segment_limit:
                ax.annotate(
                    f"{count}", (bar.get_x() + bar.get_width(), count / 2),
                    xytext=(6, 6), textcoords="offset points", ha="left", va="center",
                    fontsize=12, fontweight="bold",
                    arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.6},
                )
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, count / 2, f"{count}", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    for bar, base, count in zip(bars_unmapped, mapped, unmapped):
        if count > 0:
            if count < small_segment_limit:
                ax.annotate(
                    f"{count}", (bar.get_x() + bar.get_width(), base + count / 2),
                    xytext=(6, 18), textcoords="offset points", ha="left", va="center",
                    fontsize=12, fontweight="bold",
                    arrowprops={"arrowstyle": "-", "color": "#555555", "linewidth": 0.6},
                )
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, base + count / 2, f"{count}", ha="center", va="center", color="black", fontsize=12, fontweight="bold")

    offset = max(float(total.max()) * 0.025, 0.8)
    for xpos, count in zip(x, total):
        ax.text(xpos, count + offset, f"Total: {count}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_xticks(x, stats["Frequency_Bin"])
    ax.set_xlabel("Number of CVEs Associated with Each CWE")
    ax.set_ylabel("Number of CWE Entries")
    ax.set_title("CWE Frequency in CVE Dataset and Availability of CAPEC Mapping", pad=10)
    ax.set_ylim(0, max(float(total.max()) * 1.16, 1.0))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    _bold_tick_values(ax)
    fig.tight_layout()
    _save_figure(fig, "fig3_cve_cwe_capec_associations")
    return stats


def plot_capec_mapping_distribution(cwe_capec: pd.DataFrame) -> pd.DataFrame:
    """Create Figure 4: unique CAPEC mappings per CWE on a log-scaled Y-axis."""
    capecs_per_cwe = cwe_capec.groupby("CWE_ID")["CAPEC_ID"].nunique()
    capec_distribution = capecs_per_cwe.value_counts().sort_index()
    stats = capec_distribution.rename_axis("Number_of_CAPEC_Mappings").reset_index(name="Number_of_CWEs")
    stats = stats.astype({"Number_of_CAPEC_Mappings": int, "Number_of_CWEs": int})

    print("\nFigure 4 dataframe")
    print("=" * 72)
    print(stats.to_string(index=False))
    stats.to_csv(OUTPUT_DIR / "capec_mappings_per_cwe_distribution.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(stats))
    frequencies = stats["Number_of_CWEs"].to_numpy()
    bars = ax.bar(x, frequencies, width=0.72, color="#3F7D5B")
    ax.set_yscale("log")

    for bar, count in zip(bars, frequencies):
        ax.annotate(
            f"{count}",
            (bar.get_x() + bar.get_width() / 2, count),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xticks(x, stats["Number_of_CAPEC_Mappings"])
    ax.set_xlabel("Number of CAPEC Mappings per CWE")
    ax.set_ylabel("Number of Unique CWEs (Log Scale)")
    ax.set_title("Distribution of CAPEC Mappings Across CWE", pad=10)
    ax.grid(axis="y", which="major", color="#D9D9D9", linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    _bold_tick_values(ax)
    fig.tight_layout()
    _save_figure(fig, "fig4_capec_mapping_distribution_log")
    return stats


def _capec_numeric_id(capec_ids: pd.Series) -> pd.Series:
    return capec_ids.str.extract(r"(\d+)$", expand=False).astype(int)


def _annotate_horizontal_bars(ax: plt.Axes, bars: object, counts: np.ndarray) -> None:
    max_count = max(int(counts.max()), 1)
    padding = max_count * 0.012
    for bar, count in zip(bars, counts):
        ax.text(
            count + padding,
            bar.get_y() + bar.get_height() / 2,
            f"{int(count)}",
            va="center",
            ha="left",
            fontsize=12,
            fontweight="bold",
        )
    ax.set_xlim(0, max_count * 1.12)


def plot_top_capecs(capec_frequency: pd.Series) -> pd.DataFrame:
    """Create Figure 5 using deduplicated unique-CVE CAPEC frequencies."""
    stats = capec_frequency.rename("CVE_Count").reset_index()
    stats["_numeric_id"] = _capec_numeric_id(stats["CAPEC_ID"])
    stats = (
        stats.sort_values(["CVE_Count", "_numeric_id"], ascending=[False, True], kind="stable")
        .head(TOP_BOTTOM_N)
        .drop(columns="_numeric_id")
        .reset_index(drop=True)
    )
    stats["CVE_Count"] = stats["CVE_Count"].astype(int)

    print("\nFigure 5 dataframe")
    print("=" * 72)
    print(stats.to_string(index=False))
    stats.to_csv(OUTPUT_DIR / "top_30_capec_frequency.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 10))
    y = np.arange(len(stats))
    counts = stats["CVE_Count"].to_numpy()
    bars = ax.barh(y, counts, height=0.72, color="#2F6B9A")
    ax.set_yticks(y, stats["CAPEC_ID"])
    ax.invert_yaxis()
    _annotate_horizontal_bars(ax, bars, counts)
    ax.set_xlabel("Number of CVEs / Occurrences")
    ax.set_ylabel("CAPEC ID")
    ax.set_title("Top 30 Most Frequent CAPEC IDs", pad=10)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    _bold_tick_values(ax)
    fig.tight_layout()
    _save_figure(fig, "fig5_top30_capecs")
    return stats


def plot_bottom_capecs(capec_frequency: pd.Series) -> pd.DataFrame:
    """Create Figure 6 with deterministic numeric-ID tie breaking."""
    stats = capec_frequency.rename("CVE_Count").reset_index()
    stats["_numeric_id"] = _capec_numeric_id(stats["CAPEC_ID"])
    stats = (
        stats.loc[stats["CVE_Count"] >= 1]
        .sort_values(["CVE_Count", "_numeric_id"], ascending=[True, True], kind="stable")
        .head(TOP_BOTTOM_N)
        .drop(columns="_numeric_id")
        .reset_index(drop=True)
    )
    stats["CVE_Count"] = stats["CVE_Count"].astype(int)

    print("\nFigure 6 dataframe")
    print("=" * 72)
    print(stats.to_string(index=False))
    stats.to_csv(OUTPUT_DIR / "bottom_30_capec_frequency.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 10))
    y = np.arange(len(stats))
    counts = stats["CVE_Count"].to_numpy()
    bars = ax.barh(y, counts, height=0.72, color="#D98C3F")
    ax.set_yticks(y, stats["CAPEC_ID"])
    ax.invert_yaxis()
    _annotate_horizontal_bars(ax, bars, counts)
    ax.set_xlabel("Number of CVEs / Occurrences")
    ax.set_ylabel("CAPEC ID")
    ax.set_title("Bottom 30 Least Frequent CAPEC IDs", pad=10)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    _bold_tick_values(ax)
    fig.tight_layout()
    _save_figure(fig, "fig6_bottom30_capecs")
    return stats


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _set_publication_style()

    raw_cve, raw_mapping = load_and_clean_data()
    cve_cwe, all_cves = build_cve_cwe_table(raw_cve)
    cwe_capec = build_cwe_capec_table(raw_mapping)
    cve_capec = build_cve_capec_table(cve_cwe, cwe_capec)
    validate_tables(cve_cwe, cwe_capec, cve_capec, all_cves)

    # These are the three clean, inspectable intermediate relationship tables.
    print("\ncve_cwe sample")
    print(cve_cwe.head(10).to_string(index=False))
    print("\ncwe_capec sample")
    print(cwe_capec.head(10).to_string(index=False))
    print("\ncve_capec sample")
    print(cve_capec.head(10).to_string(index=False))

    plot_cwe_frequency_mapping_availability(cve_cwe, cwe_capec)
    plot_capec_mapping_distribution(cwe_capec)

    # One shared frequency series guarantees Figures 5 and 6 use exactly the
    # same deduplicated CVE -> CAPEC calculation.
    capec_frequency = cve_capec.groupby("CAPEC_ID")["CVE_ID"].nunique()
    assert (capec_frequency >= 1).all()
    plot_top_capecs(capec_frequency)
    plot_bottom_capecs(capec_frequency)

    print(f"\nAll outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
