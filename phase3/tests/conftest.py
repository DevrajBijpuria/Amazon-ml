"""Shared fixture: a tiny training layout (dataset/train) plus Phase 2-style validation artifacts."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "phase3" / "src"), str(REPO / "phase2" / "src")]


def _tsv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


@pytest.fixture
def mini_run(tmp_path):
    """Validation entities: S1-1 (two true matches), S1-2 (zero-match), S1-3 (one match, no candidates)."""
    data = tmp_path / "dataset"
    src = lambda p, n: pd.DataFrame({"entity_id": [f"{p}-{i}" for i in range(1, n + 1)],  # noqa: E731
                                     "business_name": [f"name {i}" for i in range(1, n + 1)],
                                     "business_address": ["addr"] * n, "country": ["France"] * n})
    _tsv(data / "train" / "train_source1.tsv", src("S1", 4))
    _tsv(data / "train" / "train_source2.tsv", src("S2", 5))
    _tsv(data / "train" / "train_source3.tsv", src("S3", 5))
    art = tmp_path / "run" / "artifacts"
    _tsv(art / "validation_scores.tsv", pd.DataFrame({
        "s1": ["S1-1", "S1-1", "S1-1", "S1-1", "S1-2", "S1-2"],
        "target": ["S2-3", "S2-1", "S3-2", "S3-5", "S2-4", "S3-1"],
        "source": ["s2", "s2", "s3", "s3", "s2", "s3"],
        "label": [1, 0, 1, 0, 0, 0], "score": [0.9, 0.2, 0.85, 0.81, 0.3, 0.1]}))
    _tsv(art / "validation_entities.tsv", pd.DataFrame({"s1": ["S1-1", "S1-2", "S1-3"], "n_true": [2, 0, 1]}))
    (art / "threshold_results.json").write_text(json.dumps({"selected_threshold": 0.8}), encoding="utf-8")
    return {"data": data, "art": art, "run": tmp_path / "run"}
