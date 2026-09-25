"""Dataset discovery, loading, schema profiling, missing-value and duplicate analysis."""
import csv
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def count_lines(path: Path) -> int:
    """Count text lines in a file without loading it (a final line without '\\n' counts)."""
    n, last = 0, b"\n"
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            n += chunk.count(b"\n")
            last = chunk[-1:]
    return n + (last != b"\n")


def discover_datasets(data_root: Path) -> dict:
    """Header, size and data-row count of every ``<split>/*.tsv`` under ``data_root``."""
    found = {}
    for path in sorted(data_root.glob("*/*.tsv")):
        with open(path, encoding="utf-8") as f:
            header = f.readline().rstrip("\r\n").split("\t")
        found[f"{path.parent.name}/{path.name}"] = {
            "header": header,
            "bytes": path.stat().st_size,
            "data_rows": count_lines(path) - 1,
        }
    return found


def check_schema(discovered: dict) -> list:
    """Return schema problems: source files must have SOURCE_COLUMNS, ground truth GT_COLUMNS."""
    issues = []
    for key, info in discovered.items():
        expected = GT_COLUMNS if "ground_truth" in key else SOURCE_COLUMNS
        if info["header"] != expected:
            issues.append(f"{key}: header {info['header']} != expected {expected}")
    for split in ("train", "test"):
        for i in (1, 2, 3):
            if f"{split}/{split}_source{i}.tsv" not in discovered:
                issues.append(f"missing {split}/{split}_source{i}.tsv")
    if "train/train_ground_truth.tsv" not in discovered:
        issues.append("missing train/train_ground_truth.tsv")
    return issues


def load_tsv(path: Path, max_rows=None) -> pd.DataFrame:
    """Read a challenge TSV as strings exactly as stored.

    No NA inference (so a literal "NA" stays text) and no quote handling (the
    files are unquoted TSV; quote characters are data). Missing = empty string.
    """
    return pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
        quoting=csv.QUOTE_NONE, nrows=max_rows, encoding="utf-8",
    )


def _length_stats(s: pd.Series) -> dict:
    lengths = s[s != ""].str.len()
    if lengths.empty:
        return {}
    return {
        "min": int(lengths.min()), "max": int(lengths.max()),
        "mean": round(float(lengths.mean()), 2), "median": float(lengths.median()),
        "p95": float(lengths.quantile(0.95)),
    }


def _duplicate_values(df: pd.DataFrame, cols: list) -> dict:
    """Rows whose (non-empty) value combination occurs more than once."""
    sub = df[cols]
    sub = sub[(sub != "").all(axis=1)]
    dup_mask = sub.duplicated(keep=False)
    return {
        "rows_in_duplicate_groups": int(dup_mask.sum()),
        "distinct_duplicated_values": int(sub[dup_mask].drop_duplicates().shape[0]),
    }


def profile_frame(df: pd.DataFrame, id_col: str, id_prefix: str = "") -> dict:
    """Schema, missing-value, uniqueness, length and duplicate profile of one table.

    Duplicates are only reported, never removed.
    """
    n = len(df)
    pct = lambda x: round(100.0 * x / n, 4) if n else 0.0  # noqa: E731
    columns = {}
    for col in df.columns:
        s = df[col]
        stripped = s.str.strip()
        empty = int((s == "").sum()) + int(s.isna().sum())
        whitespace_only = int(((stripped == "") & (s != "")).sum())
        unique = int(s.nunique())
        columns[col] = {
            "dtype": str(s.dtype),
            "empty_string": empty,
            "whitespace_only": whitespace_only,
            "missing": empty + whitespace_only,
            "missing_pct": pct(empty + whitespace_only),
            "unique": unique,
            "unique_pct": pct(unique),
            "leading_or_trailing_whitespace": int(((s != stripped) & (stripped != "")).sum()),
            "length": _length_stats(s),
        }
    ids = df[id_col]
    dup_ids = ids[ids.duplicated(keep=False)]
    profile = {
        "rows": n,
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "column_profile": columns,
        "id": {
            "column": id_col,
            "is_unique": bool(ids.is_unique),
            "duplicate_id_rows": int(len(dup_ids)),
            "duplicate_id_examples": sorted(dup_ids.unique().tolist())[:5],
            "bad_prefix_rows": int((~ids.str.startswith(id_prefix)).sum()) if id_prefix else 0,
        },
        "duplicates": {"exact_duplicate_rows": int(df.duplicated().sum())},
    }
    other = [c for c in df.columns if c != id_col]
    profile["duplicates"]["duplicate_rows_ignoring_id"] = int(df.duplicated(subset=other).sum())
    if "business_name" in df.columns:
        profile["duplicates"]["name"] = _duplicate_values(df, ["business_name"])
        profile["duplicates"]["address"] = _duplicate_values(df, ["business_address"])
        profile["duplicates"]["name_address"] = _duplicate_values(
            df, ["business_name", "business_address"])
    return profile
