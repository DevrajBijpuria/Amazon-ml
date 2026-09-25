import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import candidate_generation as cg  # noqa: E402
import feature_engineering as fe  # noqa: E402

CG_CFG = {
    "use_country_block": True, "use_exact_name_block": True, "use_exact_address_block": True,
    "use_numeric_block": True, "use_name_prefix_block": True, "use_address_prefix_block": True,
    "use_name_char_ngram_block": True, "use_name_phonetic_block": True, "use_address_token_block": True,
    "name_prefix_length": 8, "address_prefix_length": 12, "rare_tokens_per_name": 2, "rare_tokens_per_address": 2, "max_block_size": 3, "max_candidates_per_source1": 10,
    "source1_chunk_size": 2,
}


def records(rows):
    df = pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])
    return fe.prepare_records(df)


def candidates(s1, tgt, cfg=CG_CFG):
    blocking = cg.build_blocking(s1, tgt, cfg)
    kept = [k for _, _, _, k, _ in cg.iter_candidates(blocking, s1, tgt, cfg)]
    return pd.concat(kept, ignore_index=True), blocking


def pairs_of(cands, s1, tgt):
    return {(s1["entity_id"][a], tgt["entity_id"][b]) for a, b in zip(cands["s1_idx"], cands["t_idx"])}


# ---------------------------------------------------------------- A. normalization
def test_case_punctuation_whitespace():
    assert fe.normalize_name2("  ACME,   Widgets!! ")[0] == "acme widgets"


def test_legal_suffix_canonicalization():
    norm, _, core, key, legal, legal_set, _, _, _ = fe.normalize_name2("ABC Pvt. Ltd.")
    assert norm == "abc private limited" and core == "abc" and key == "abc"
    assert legal and legal_set == "limited private"
    assert fe.normalize_name2("A.B.C. Corp")[0] == "abc corporation"  # punctuation-separated initials


def test_domain_name_shares_key_with_legal_name():  # hard positive 1
    assert fe.normalize_name2("physicaltherapycenter.com")[3] == fe.normalize_name2("Physical Therapy Center Inc.")[3]


def test_street_abbreviations_and_state_names():  # hard positive 2
    a = fe.normalize_address2("79 Wagoneers Lane, Stafford County, VA")
    b = fe.normalize_address2("79 Wagoneers Ln, Stafford County, Virginia")
    assert a[0] == b[0] == "79 wagoneers ln stafford county va"
    assert a[1] == b[1] == "79" and a[3] == b[3] == "va"


def test_null_address_is_missing():  # hard positive 4
    assert fe.normalize_address2("<NULL>")[0] == ""
    assert "null" not in fe.normalize_address2("11515 Rd 11-J, <NULL>, Ottawa, Ohio")[0].split()
    assert fe.normalize_name2(None)[0] == "" and fe.normalize_address2(None)[0] == ""


def test_unicode_kept_and_romanized():  # hard positive 5
    norm, translit, core, *_ = fe.normalize_name2("राम मार्केटिंग प्राइवेट लिमिटेड")
    assert norm.startswith("राम मार्केटिंग")  # primary representation stays Unicode
    assert translit == "ram marketing private limited" and core == "ram marketing"
    assert fe.normalize_name2("Ram Marketing Pvt Ltd")[3] == fe.normalize_name2("राम मार्केटिंग प्राइवेट लिमिटेड")[3]
    assert fe.normalize_address2("ERNAKULAM, കേരളം")[3] == fe.normalize_address2("Ernakulam, Kerala")[3] == "kl"


def test_postal_not_taken_as_house_number():
    _, house, postal, state, *_ = fe.normalize_address2("12 MG Road, Mumbai 400 021, Maharashtra", "india")
    assert (house, postal, state) == ("12", "400021", "mh")


def test_missing_values_of_any_type():
    for v in (None, float("nan"), pd.NA, 12.5, ""):
        assert fe.normalize_name2(v)[:4] == ("", "", "", "")
        assert fe.normalize_address2(v)[0] == ""
    assert fe.normalize_name2("<NULL> Pvt Ltd")[3] == ""  # suffix-only name gets no key


def test_initials_are_not_legal_suffixes():
    assert fe.normalize_name2("S A Traders")[2] == "sa traders"
    assert fe.normalize_name2("Silver Education P.C.")[2] == "silver education"


def test_more_street_abbreviations():
    a = fe.normalize_address2("1 Main Street, 2 Oak Drive, 3 Elm Avenue, 4 Pine Court, 5 Hill Place, "
                              "6 Sunset Boulevard, 7 State Highway, 8 Park Road")[0]
    assert a == "1 main st 2 oak dr 3 elm ave 4 pine ct 5 hill pl 6 sunset blvd 7 state hwy 8 park rd"


def test_us_state_zip_and_french_postal():
    _, house, postal, state, *_ = fe.normalize_address2("100 200th St, Austin, TX 78701", "us")
    assert (house, postal, state) == ("100", "78701", "tx")
    assert fe.normalize_address2("17560 Ellis Road, Tahlequah, OK", "us")[1:4] == ("17560", "", "ok")
    assert fe.normalize_address2("63 Rue de Dieppe, 59000 Lille", "france")[1:3] == ("63", "59000")


def test_transliteration_signals():
    # romanized Indic spellings agree with English ones on the consonant skeleton
    assert fe.normalize_name2("यूनिक इन्वेस्टमेंट प्राइवेट लिमिटेड")[8] == fe.normalize_name2("Unique Investment Pvt Ltd")[8]
    assert fe.normalize_name2("गोल्डन प्रोड्यूसर")[8] == fe.normalize_name2("Golden Producer")[8]
    assert fe.normalize_name2("ग्रीन प्रोडक्ट्स प्रा. लि.")[2] == "grin prodakts"  # native legal abbreviations dropped
    assert fe.normalize_name2("ਓਮ ਫੂਡਜ਼ ਪ੍ਰਾਈਵੇਟ ਲਿਮਟਿਡ")[5] == "limited private"  # suffixes found by skeleton
    assert fe.normalize_name2("Pravat Enterprises")[2] == "pravat enterprises"  # Latin name never skeleton-canonicalized
    assert fe.normalize_name2("Litt1e Hair Studio")[2] == "little hair studio"  # digit-for-letter typo
    assert fe.normalize_name2("24x7 Services 3rd")[2] == "24x7 services 3rd"   # real alphanumerics kept
    assert fe.normalize_name2("Orthopedic Orthopedic Physicians")[7] == fe.normalize_name2("Physicians Orthopedic")[7]


def test_normalization_deterministic():
    raw = "LLC Moncada Léarning Center (ID: 78977)"
    assert fe.normalize_name2(raw) == fe.normalize_name2(raw)
    assert fe.normalize_name2(raw)[2] == "moncada learning center"


# ---------------------------------------------------------------- A. similarity
def test_similarity_edges():
    assert fe.jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert fe.jaccard(set(), set()) == 0.0  # missing never creates similarity
    assert fe.edit_sim("", "") == 0.0 and fe.edit_sim("abc", "") == 0.0
    assert fe.edit_sim("acme", "acme") == 1.0
    assert fe.edit_sim("abcd", "wxyz") == 0.0
    assert fe.jaccard(fe.char3("acme"), fe.char3("acme")) == 1.0


def test_pair_features_missing_address_and_country():  # hard positive 4 at feature level
    s1 = records([["S1-1", "Allied Cascade Landscaping Corp", "", "France"]])
    tgt = records([["S3-1", "Alliedcascadelandscaping.Com", "<NULL>", "France"]])
    f = fe.compute_features(pd.DataFrame({"s1_idx": [0], "t_idx": [0], "blocks": [1]}), s1, tgt, "s3")
    assert f.loc[0, "address_missing_either"] == 1 and f.loc[0, "address_char3_jaccard"] == 0
    assert f.loc[0, "address_norm_eq"] == 0 and f.loc[0, "name_key_eq"] == 1
    assert f.loc[0, "country_equal"] == 1  # open-set label compared exactly


def test_spelling_corruption_and_noisy_address():  # hard positives 3 and 8
    s1 = records([["S1-1", "Coastal Charities", "8964 Rand Avenue, Unit 6102, AL, Daphne", "US"]])
    tgt = records([["S3-1", "Coasta1 Charitiez", "8964c Rand Ave, Unit 6102, Daphne, Alabama", "US"],
                   ["S3-2", "Pioneer Harmony LLC", "PO BOX 7, NOWHERE", "US"],
                   ["S3-3", "Coastal Charities", "DAPHNE", "US"]])
    f = fe.compute_features(pd.DataFrame({"s1_idx": [0, 0, 0], "t_idx": [0, 1, 2], "blocks": [4, 8, 1]}), s1, tgt, "s3")
    assert f.loc[0, "name_edit_sim"] > 0.8 > f.loc[1, "name_edit_sim"]       # typo still close
    assert f.loc[0, "house_number_equal"] == 1 and f.loc[0, "state_equal"] == 1
    assert f.loc[2, "name_norm_eq"] == 1 and f.loc[2, "address_token_jaccard"] < 0.3  # strong name, weak address


def test_pair_features_hard_positives():
    s1 = records([["S1-1", "Pioneer Harmony LLC", "10140 Pepper Tree Lane, Noblesville, IN", "US"],
                  ["S1-2", "Mason Midwest Biologics", "357 Garfield Street, Lombard, IL", "US"]])
    tgt = records([["S3-1", "harmony pioneer llc", "10140 Pepper Tree Ln, Noblesville, Indiana", "US"],  # token order
                   ["S3-2", "Halonex", "357 Garfield Street, Lombard, Illinois", "US"]])  # noisy name, same address
    pairs = pd.DataFrame({"s1_idx": [0, 1], "t_idx": [0, 1], "blocks": [1 | 2, 2]})
    f = fe.compute_features(pairs, s1, tgt, "s3")
    assert list(f.columns) == fe.FEATURES
    assert f.loc[0, "name_token_jaccard"] == 1.0 and f.loc[0, "name_norm_eq"] == 0.0
    assert f.loc[0, "address_norm_eq"] == 1.0 and f.loc[0, "state_equal"] == 1.0
    assert f.loc[1, "name_token_jaccard"] == 0.0 and f.loc[1, "address_norm_eq"] == 1.0
    assert f.loc[0, "number_of_blocks_that_retrieved_pair"] == 2 and f.loc[1, "matched_by_exact_address_block"] == 1
    assert f["target_source_is_s3"].tolist() == [1.0, 1.0]
    assert fe.compute_features(pairs, s1, tgt, "s3").equals(f)  # deterministic


# ---------------------------------------------------------------- B. candidate generation
S1_ROWS = [["S1-1", "Acme Widgets Pvt Ltd", "12 Park Street, Pune, Maharashtra", "India"],
           ["S1-2", "Zenith Tools", "44 Lake Road, Austin, TX", "US"],
           ["S1-3", "Orion Foods", "9 Hill View, Dallas, TX", "US"]]
T_ROWS = [["S2-1", "ACME WIDGETS PRIVATE LIMITED", "PUNE, MH", "India"],        # exact name key
          ["S2-2", "Totally Different", "44 LAKE RD, AUSTIN, TX", "US"],        # exact address
          ["S2-3", "Acme Widgets", "12 Park Street, Pune", "US"],               # other country
          ["S2-4", "Zenith Tools", "44 Lake Road, Austin, TX", "US"]]           # name + address


def test_exact_name_address_country_union():
    s1, tgt = records(S1_ROWS), records(T_ROWS)
    cands, _ = candidates(s1, tgt)
    got = pairs_of(cands, s1, tgt)
    assert ("S1-1", "S2-1") in got                       # exact-name retrieval
    assert ("S1-2", "S2-2") in got                       # exact-address retrieval
    assert ("S1-1", "S2-3") not in got                   # country block
    assert not cands.duplicated(["s1_idx", "t_idx"]).any()  # union is deduplicated
    zenith = cands[(cands.s1_idx == 1) & (cands.t_idx == 3)]["blocks"].iloc[0]
    assert zenith & fe.RULE_BITS["exact_name"] and zenith & fe.RULE_BITS["exact_address"]


def test_open_set_country_is_kept():
    s1 = records([["S1-9", "Marina Ecole", "63 Rue de Dieppe, Lille", "France"]])
    tgt = records([["S2-9", "Marina Ecole SARL", "63 R. DE DIEPPE, LILLE", "France"]])
    cands, _ = candidates(s1, tgt)
    assert pairs_of(cands, s1, tgt) == {("S1-9", "S2-9")} and s1["country"][0] == "france"


def test_high_frequency_key_is_skipped_and_recorded():
    s1 = records([["S1-1", "Common Name", "1 A St, X, TX", "US"]])
    tgt = records([[f"S2-{i}", "Common Name", f"{i} B St, Y, TX", "US"] for i in range(5)])
    cands, blocking = candidates(s1, tgt)  # max_block_size = 3
    assert cands.empty or not (cands["blocks"] & fe.RULE_BITS["exact_name"]).any()
    assert blocking["exact_name"]["skipped"][["key", "frequency"]].values.tolist() == [["us|commonname", 5]]
    missing = pd.DataFrame({"s1_idx": [0], "t_idx": [2]})
    lost = cg.lost_to_skipped_keys(blocking, missing)
    code = blocking["exact_name"]["skipped"]["code"].iloc[0]
    assert lost["exact_name"]["pairs"] == 1 and lost["exact_name"]["per_code"] == {code: 1}
    assert lost["any_rule"] == 1


def test_per_source1_cap_keeps_strongest_and_reports_drops():
    s1 = records([["S1-1", "Alpha Beta Gamma", "5 Main St, Reno, NV", "US"]])
    tgt = records([[f"S2-{i}", "Alpha Beta Gamma" if i == 0 else f"Alpha Beta Gamma Z{i}", "5 Main St, Reno, NV", "US"]
                   for i in range(3)])
    cfg = {**CG_CFG, "max_block_size": 100, "max_candidates_per_source1": 1}
    blocking = cg.build_blocking(s1, tgt, cfg)
    _, _, precap, kept, dropped = next(cg.iter_candidates(blocking, s1, tgt, cfg))
    assert len(precap) == 3 and len(kept) == 1 and len(dropped) == 2
    assert tgt["entity_id"][kept["t_idx"].iloc[0]] == "S2-0"  # most rules + highest similarity


def test_candidate_summary_counts_empty_s1():
    s = cg.candidate_summary(np.array([3, 1]), 4)
    assert s["candidate_pairs"] == 4 and s["s1_with_zero_candidates"] == 2 and s["max_candidates_per_s1"] == 3


# ---------------------------------------------------------------- labels, split, sampling
import inference as inf  # noqa: E402
import matching_model as mm  # noqa: E402
import threshold_tuning as tt  # noqa: E402


def test_labels_positive_negative_multiple_and_empty():
    cands = pd.DataFrame({"s1_idx": [0, 0, 0, 1, 2], "t_idx": [5, 6, 7, 5, 9]})
    positives = pd.DataFrame({"s1_idx": [0, 0], "t_idx": [5, 7]})  # S1 0 has two matches, S1 1 and 2 none
    assert mm.label_pairs(cands, positives).tolist() == [1, 0, 1, 0, 0]
    assert mm.label_pairs(cands, positives.iloc[:0]).tolist() == [0, 0, 0, 0, 0]  # empty ground truth


def test_entity_split_is_disjoint_and_deterministic():
    ids = [f"S1-{i}" for i in range(100)]
    val = mm.split_entities(ids, 0.2, 42)
    assert len(val) == 20 and val == mm.split_entities(list(reversed(ids)), 0.2, 42)


def test_negative_sampling_keeps_all_positives():
    y = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    feats = pd.DataFrame({"name_char3_jaccard": np.linspace(0, 1, 10), "address_char3_jaccard": np.zeros(10)})
    rows = mm.sample_training_pairs(y, feats, 2, 0.5, 42)
    assert set(np.flatnonzero(y)) <= set(rows) and len(rows) == 6
    assert 9 in rows and 8 in rows  # the two most similar negatives are the hard ones
    assert rows.tolist() == mm.sample_training_pairs(y, feats, 2, 0.5, 42).tolist()


# ---------------------------------------------------------------- metric and threshold
def test_macro_f05_challenge_rules():
    n_true = pd.Series({"A": 2, "B": 0, "C": 0, "D": 1})
    scored = pd.DataFrame({"s1": ["A", "A", "A", "C", "D"], "target": ["x", "y", "z", "w", "v"],
                           "score": [0.9, 0.8, 0.7, 0.9, 0.2], "label": [1, 0, 1, 0, 1]})
    r = tt.evaluate(scored, n_true, 0.5)
    # A: P=2/3 R=1 -> 0.714; B: empty/empty -> 1; C: false merge -> 0; D: missed -> 0
    assert r["macro_f05"] == pytest.approx((1.25 * (2 / 3) / (0.25 * (2 / 3) + 1) + 1 + 0 + 0) / 4)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 2, 1)
    assert r["zero_match_entities"] == 2 and r["zero_match_entities_scored_1"] == 1
    best = tt.sweep(scored, n_true, [0.1, 0.5, 0.95])
    assert best["selected_threshold"] in (0.1, 0.5, 0.95) and len(best["sweep"]) == 3


# ---------------------------------------------------------------- prediction assembly
def test_threshold_multi_match_ordering_and_zero_match():
    scored = pd.DataFrame({"s1": ["S1-2", "S1-2", "S1-2", "S1-1", "S1-3"],
                           "target": ["S3-9", "S2-5", "S2-5", "S2-1", "S2-7"],
                           "score": [0.9, 0.8, 0.8, 0.3, 0.95]})
    out = inf.predictions(["S1-1", "S1-2", "S1-3", "S1-4"], scored, 0.5)
    assert out["source1_entity_id"].tolist() == ["S1-1", "S1-2", "S1-3", "S1-4"]   # input order, all rows
    assert out["matched_entity_ids"].tolist() == ["", "S2-5,S3-9", "S2-7", ""]      # dedup, sorted, empty


def test_streamed_submission_and_checks(tmp_path):
    s1_ids = ["S1-1", "S1-2", "S1-3"]
    # S1 positions 0..2; per-source chunk lines are written independently, then merged
    s2 = inf.chunk_lines(0, 3, np.array([1, 1, 0]), np.array(["S2-9", "S2-5", "S2-1"]), np.array([0.9, 0.2, 0.1]), 0.5)
    s3 = inf.chunk_lines(0, 3, np.array([1, 2]), np.array(["S3-4", "S3-8"]), np.array([0.7, 0.6]), 0.5)
    assert s2 == ["S2-1\t\n", "S2-5,S2-9\tS2-9\n", "\t\n"]
    parts = []
    for name, lines in (("a", s2), ("b", s3)):
        parts.append(tmp_path / name)
        parts[-1].write_text("".join(lines), encoding="utf-8")
    m, c = tmp_path / "matching_results.tsv", tmp_path / "candidate_pairs.tsv"
    res = inf.merge_and_check(s1_ids, parts, m, c, {"S2-1", "S2-5", "S2-9", "S3-4", "S3-8"})
    assert all(res["checks"].values()), res["checks"]
    assert m.read_text(encoding="utf-8").splitlines() == [
        "source1_entity_id\tmatched_entity_ids", "S1-1\t", "S1-2\tS2-9,S3-4", "S1-3\tS3-8"]
    assert c.read_text(encoding="utf-8").splitlines()[2] == "S1-2\tS2-5,S2-9,S3-4"
    assert res["counts"] == {"rows": 3, "with_prediction": 2, "with_multiple": 1, "predictions": 3, "candidate_pairs": 5}
    # a malformed part (unknown id, one row short) must fail the checks
    parts[1].write_text("\t\nS3-4\tS3-4\n", encoding="utf-8")
    bad = inf.merge_and_check(s1_ids, parts, m, c, {"S2-1", "S2-5", "S2-9"})
    assert not bad["checks"]["1_every_s1_represented"] and not bad["checks"]["2_ids_exist_in_s2_s3"]


# ---------------------------------------------------------------- C. candidate recall accounting
def test_candidate_recall_accounting_separates_cap_and_skips():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import run_phase2 as rp
    s1 = records([["S1-1", "Alpha Widgets", "5 Main St, Reno, NV", "US"],
                  ["S1-2", "Common Name", "9 Oak Ln, Waco, TX", "US"],
                  ["S1-3", "Zeta Tools", "1 Elm St, Troy, NY", "US"]])
    tgt = records([["S2-1", "Alpha Widgets", "5 Main St, Reno, NV", "US"],         # found by several rules
                   ["S2-2", "Alpha Widgets Two", "5 Main St, Reno, NV", "US"],     # dropped by the cap (limit 1)
                   *[[f"S2-c{i}", "Common Name", f"{i} Pine Rd, X, CA", "US"] for i in range(5)],  # skipped key
                   ["S2-9", "Unrelated", "77 Far Away, Zz, WA", "US"]])            # never retrieved
    positives = pd.DataFrame({"s1_idx": [0, 0, 1, 2], "t_idx": [0, 1, 2, 7]}).astype(np.int32)
    cfg = {**CG_CFG, "max_block_size": 3, "max_candidates_per_source1": 1, "source1_chunk_size": 2}
    n_true = np.array([2, 1, 1])
    stats, kept = rp.candidates_for_source(s1, tgt, positives, cfg, np.array([True, True, True]), n_true)
    assert stats["positive_pairs_total"] == 4 and stats["positive_pairs_recovered"] == 1
    assert stats["precap_positive_pairs_recovered"] == 2        # S1-1's two positives before the cap
    assert stats["cap"]["positive_pairs_dropped"] == 1           # one lost to the cap ...
    assert stats["positives_lost_to_skipped_keys_any_rule"] == 1  # ... one to the skipped 'commonname' key
    keys = stats["skipped_keys"]["exact_name"]["keys"]
    assert keys == [["us|commonname", 5, 1]]
    assert stats["max_candidates_per_s1"] == 1 and stats["s1_with_zero_candidates"] == 2
    assert set(stats["recall_by_s1_match_bucket"]) == {"one", "multi"}


def test_sharded_validator_passes_good_files_and_catches_misplaced_rows(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    test_dir = tmp_path / "test"
    hdr = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
    test_dir.mkdir()
    (test_dir / "test_source1.tsv").write_text(hdr + "".join(f"S1-{i}\tn\ta\tUS\n" for i in range(5)), encoding="utf-8")
    (test_dir / "test_source2.tsv").write_text(hdr + "S2-1\tn\ta\tUS\nS2-2\tn\ta\tUS\n", encoding="utf-8")
    (test_dir / "test_source3.tsv").write_text(hdr + "S3-1\tn\ta\tUS\n", encoding="utf-8")
    rows = ["S2-1,S3-1", "", "S2-2", "", ""]
    m, c = tmp_path / "matching_results.tsv", tmp_path / "candidate_pairs.tsv"
    m.write_text("source1_entity_id\tmatched_entity_ids\n" + "".join(f"S1-{i}\t{r}\n" for i, r in enumerate(rows)), encoding="utf-8")
    c.write_text("source1_entity_id\tcandidate_entity_ids\n" + "".join(f"S1-{i}\t{r}\n" for i, r in enumerate(rows)), encoding="utf-8")
    ok = inf.run_validator_sharded(repo / "utils" / "validate_submission.py", m, c, test_dir, 2, tmp_path / "shards")
    assert ok["passed"] and ok["shards"] == 3 and not (tmp_path / "shards" / "shard_000").exists()
    swapped = "source1_entity_id\tmatched_entity_ids\nS1-2\tS2-2\nS1-1\t\nS1-0\tS2-1,S3-1\nS1-3\t\nS1-4\t\n"
    m.write_text(swapped, encoding="utf-8")  # S1-2 and S1-0 in the wrong shards
    bad = inf.run_validator_sharded(repo / "utils" / "validate_submission.py", m, c, test_dir, 2, tmp_path / "shards2")
    assert not bad["passed"] and bad["failed_shards"]
