import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import analysis as A  # noqa: E402
import run_phase1  # noqa: E402
from normalization import normalize_address, normalize_country, normalize_name  # noqa: E402
from profiling import profile_frame  # noqa: E402

TRICKY = [None, float("nan"), pd.NA, "", "   ", "\t\n", "Café  Déjà-Vu, Inc.", "AT&T", "ﬁnance Ⅳ",
          "İSTANBUL", "Straße", "राम मार्केटिंग प्राइवेट लिमिटेड", "क़ानून", "S\ufffdhutons", "B+ Retail"]


# ---------------------------------------------------------------- normalization
@pytest.mark.parametrize("fn", [normalize_name, normalize_address, normalize_country])
def test_missing_and_blank_become_empty(fn):
    for v in (None, float("nan"), pd.NA, "", "   ", "\t\n"):
        assert fn(v) == ""


@pytest.mark.parametrize("fn", [normalize_name, normalize_address, normalize_country])
def test_idempotent(fn):
    for v in TRICKY:
        once = fn(v)
        assert fn(once) == once, repr(v)


def test_name_rules():
    assert normalize_name("  Café  Déjà-Vu, Inc. ") == "cafe deja vu inc"
    assert normalize_name("AT&T") == normalize_name("at and t") == "at and t"
    assert normalize_name("ACME   Pvt. Ltd.") == "acme pvt ltd"  # no abbreviation expansion
    assert normalize_name("ﬁnance") == "finance"  # NFKC ligature


def test_devanagari_marks_preserved():
    raw = "राम मार्केटिंग"
    assert normalize_name(raw) == raw  # vowel signs are not diacritics to strip


def test_address_and_country():
    assert normalize_address("12-B, M.G. Road,\nPune  411001") == "12 b m g road pune 411001"
    assert normalize_country("  US ") == "us"
    assert normalize_country("India") == "india"  # no mapping to a code


# ---------------------------------------------------------------- metrics
def test_f_beta_normal():
    r = A.f_beta(2, 1, 0)
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == 1.0
    assert r["f_beta"] == pytest.approx(1.25 * (2 / 3) / (0.25 * (2 / 3) + 1))


def test_f_beta_edge_cases():
    assert A.f_beta(0, 3, 2)["f_beta"] == 0.0        # zero TP
    assert A.f_beta(4, 0, 4)["precision"] == 1.0     # zero FP
    assert A.f_beta(4, 2, 0)["recall"] == 1.0        # zero FN
    zero = A.f_beta(0, 0, 0)                          # all denominators zero
    assert zero["precision"] == zero["recall"] == zero["f_beta"] == 0.0


def test_entity_and_macro_f05():
    assert A.entity_f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"}) == pytest.approx(0.714, abs=1e-3)
    assert A.entity_f05(set(), set()) == 1.0      # correct singleton
    assert A.entity_f05({"S2-1"}, set()) == 0.0   # false merge on singleton
    assert A.entity_f05(set(), {"S2-1"}) == 0.0   # missed match
    truth = {"a": {"S2-1"}, "b": set()}
    assert A.macro_f05(truth, truth)["macro_f05"] == 1.0
    assert A.macro_f05({}, truth)["macro_f05"] == 0.5


def test_similarity_helpers():
    assert A.levenshtein("kitten", "sitting") == 3
    assert A.levenshtein("", "abc") == 3
    assert math.isnan(A.jaccard(set(), set()))
    assert A.jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


# ---------------------------------------------------------------- profiling
def test_profile_frame_missing_duplicates_ids():
    df = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2", "S1-2", "S1-4"],
        "business_name": ["Acme", "Acme", "Acme", "  "],
        "business_address": ["1 Main St", "", "", "2 Main St"],
        "country": ["US", "US", "US", "US"],
    })
    p = profile_frame(df, "entity_id", "S1-")
    assert p["rows"] == 4 and p["columns"] == 4
    assert not p["id"]["is_unique"] and p["id"]["duplicate_id_rows"] == 2
    assert p["id"]["duplicate_id_examples"] == ["S1-2"]
    assert p["column_profile"]["business_address"]["empty_string"] == 2
    assert p["column_profile"]["business_name"]["whitespace_only"] == 1
    assert p["duplicates"]["exact_duplicate_rows"] == 1
    assert p["duplicates"]["name"]["rows_in_duplicate_groups"] == 3
    assert p["duplicates"]["address"]["rows_in_duplicate_groups"] == 0  # empties ignored


def test_profile_unique_ids():
    df = pd.DataFrame({"entity_id": ["S2-1", "S2-2"], "business_name": ["a", "b"],
                       "business_address": ["x", "y"], "country": ["US", "India"]})
    p = profile_frame(df, "entity_id", "S2-")
    assert p["id"]["is_unique"] and p["id"]["bad_prefix_rows"] == 0


# ---------------------------------------------------------------- reproducibility + end to end
def _write(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join("\t".join(r) for r in [header, *rows]) + "\n", encoding="utf-8")


@pytest.fixture
def tiny_dataset(tmp_path):
    cols = ["entity_id", "business_name", "business_address", "country"]
    s1 = [[f"S1-{i}", f"Acme {i} Pvt Ltd", f"{i} Main Road, Pune, Maharashtra", "India" if i % 2 else "US"]
          for i in range(40)]
    s2 = [[f"S2-{i}", f"ACME {i} PRIVATE LIMITED", f"{i} MAIN RD, PUNE", "India" if i % 2 else "US"]
          for i in range(60)]
    s3 = [[f"S3-{i}", f"acme {i}", "", "India" if i % 2 else "US"] for i in range(60)]
    gt = [[f"S1-{i}", ",".join(x for x in (f"S2-{i}", f"S3-{i}" if i % 3 else "") if x) if i % 5 else ""]
          for i in range(40)]
    for split in ("train", "test"):
        for n, rows in ((1, s1), (2, s2), (3, s3)):
            _write(tmp_path / split / f"{split}_source{n}.tsv", cols, rows)
    _write(tmp_path / "train" / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"], gt)
    return tmp_path


def test_same_seed_same_sample(tiny_dataset):
    ids = pd.Series([f"S1-{i}" for i in range(40)])
    assert A.sample_s1_ids(ids, 10, 7) == A.sample_s1_ids(ids, 10, 7)
    pool = pd.DataFrame({"entity_id": [f"S2-{i}" for i in range(60)], "country": ["US", "India"] * 30})
    records = {f"S1-{i}": ("n", "a", "US" if i % 2 == 0 else "India") for i in range(40)}
    s1 = A.sample_s1_ids(ids, 10, 7)
    assert A.sample_negatives(s1, records, {}, pool, 7) == A.sample_negatives(s1, records, {}, pool, 7)


def test_end_to_end_reproducible_and_in_scope(tiny_dataset, tmp_path):
    outs = []
    for run_id in ("a", "b"):
        out = tmp_path / f"out_{run_id}"
        assert run_phase1.main(["--data-root", str(tiny_dataset), "--output-dir", str(out)]) == 0
        outs.append(out)
    stats = [(o / "artifacts" / "dataset_statistics.json").read_text(encoding="utf-8") for o in outs]
    assert stats[0] == stats[1]  # byte-identical across runs
    s = json.loads(stats[0])
    assert s["ground_truth"]["match_bucket_counts"]["zero"] == 8
    assert s["f05_validation"]["perfect_prediction_on_sample"]["macro_f05"] == 1.0
    assert "Phase 1 does NOT implement" in (outs[0] / "reports" / "phase1_report.md").read_text(encoding="utf-8")
    produced = {p.name for o in outs for p in o.rglob("*") if p.is_file()}
    assert produced == {"phase1_report.md", "dataset_statistics.json"}  # no candidates/predictions
