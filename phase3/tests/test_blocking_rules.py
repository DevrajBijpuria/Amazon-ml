import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "phase3" / "src"), str(REPO / "phase2" / "src")]

import blocking_phase3 as bp  # noqa: E402
import candidate_generation as cg  # noqa: E402
import feature_engineering as fe  # noqa: E402

CONFIG = REPO / "phase3" / "config"


def records(rows):
    return fe.prepare_records(pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"]))


S1 = records([["S1-1", "Acme Traders Pvt Ltd", "12 Gandhi Road, Koramangala, Bengaluru, Karnataka 560034", "India"],
              ["S1-2", "Yunik Investament", "4 Rue Lepic, Montmartre, Paris 75018", "Atlantis"]])
TGT = records([["S2-1", "Acme Traders", "12 Gandhi Rd, Koramangala, Bengaluru, KA 560034", "india"],
               ["S2-2", "Unique Investment", "4 Rue Lepic, Montmartre, Paris 75018", "atlantis"]])


def cfg(name="p3_locality_phonetic_cap100.yaml"):
    return yaml.safe_load((CONFIG / name).read_text(encoding="utf-8"))["candidate_generation"]


def keys(rule, rec=S1):
    token_df = {k: cg.rare_token_frequencies([S1, TGT], k) for k in ("addr_norm", "name_core", "name_phon")}
    return cg.rule_keys(rec, rule, cfg(), token_df).values.tolist()


def test_locality_rules_emit_expected_normalized_keys():
    assert keys("house_locality") == [[0, "india|hl|12|bengaluru"], [0, "india|hl|12|gandhi"],
                                      [1, "atlantis|hl|4|lepic"], [1, "atlantis|hl|4|montmartre"]]
    assert keys("postal_locality") == [[0, "india|pl|560034|bengaluru"], [0, "india|pl|560034|gandhi"],
                                       [1, "atlantis|pl|75018|lepic"], [1, "atlantis|pl|75018|montmartre"]]
    # no state extracted for the second record -> no state_locality key for it
    assert keys("state_locality") == [[0, "india|sl|ka|bengaluru"], [0, "india|sl|ka|gandhi"]]
    # house number and "ka 560034" (digits) and the state code are excluded; fewer than two parts -> no key
    assert keys("address_signature") == [[0, "india|sig|bengaluru|koramangala"]]
    for rule in bp.LOCALITY_RULES:  # the variant spellings on the target side produce the same keys
        assert keys(rule) == keys(rule, TGT)


def test_missing_fields_emit_no_key():
    rec = records([["S1-9", "Acme", "", "India"], ["S1-8", "", "<NULL>", "India"]])
    token_df = {"addr_norm": cg.rare_token_frequencies([rec], "addr_norm")}
    for rule in bp.LOCALITY_RULES:
        assert cg.rule_keys(rec, rule, cfg(), token_df).empty


def test_open_set_country_passes_through():
    assert all(k.startswith("atlantis|") for i, k in keys("postal_locality") if i == 1)
    assert S1["country"].tolist() == ["india", "atlantis"]  # never mapped to a fixed list


def test_phonetic_token_rule_differs_from_b0_phonetic_rule():
    assert keys("name_phonetic") == [[0, "india|kmtrdrs"], [1, "atlantis|nknvstmnt"]]
    assert keys("name_phonetic_token") == [[0, "india|pt|trdrs"], [1, "atlantis|pt|nvstmnt"]]


def test_rule_sets_per_experiment():
    assert bp.rule_set(cfg("p3_frozen_baseline.yaml")) == [r for r in cg.RULE_FLAGS if r in bp.FROZEN_RULES]
    p3a = bp.rule_set(cfg("p3_locality_cap100.yaml"))
    assert set(p3a) == set(bp.FROZEN_RULES) | {"postal_locality"} == set(bp.B0_RULES) | set(bp.LOCALITY_RULES)
    for name in ("p3_locality_cap150.yaml", "p3_locality_cap200.yaml"):
        assert bp.rule_set(cfg(name)) == p3a
    assert bp.rule_set(cfg()) == [r for r in cg.RULE_FLAGS if r in set(p3a) | {"name_phonetic_token"}]
    caps = {n: cfg(n)["max_candidates_per_source1"] for n in ("p3_locality_cap100.yaml", "p3_locality_cap150.yaml",
                                                              "p3_locality_cap200.yaml", "p3_locality_phonetic_cap100.yaml")}
    assert list(caps.values()) == [100, 150, 200, 100]
    with pytest.raises(ValueError):
        bp.rule_set({**cfg(), "use_city_block": True})


def test_high_frequency_bucket_is_skipped():
    tgt = records([[f"S2-{i}", f"Shop {i}", "7 Main Street, Springfield, IL 62701", "US"] for i in range(5)])
    s1 = records([["S1-1", "Corner Shop", "7 Main St, Springfield, Illinois 62701", "US"]])
    c = {**cfg(), "max_block_size": 4}
    blocking = cg.apply_cutoff(cg.rule_tables(s1, tgt, c, bp.LOCALITY_RULES), c["max_block_size"])
    for rule in ("house_locality", "postal_locality", "state_locality"):
        assert len(blocking[rule]["t"]) == 0 and (blocking[rule]["skipped"]["frequency"] == 5).all()
    precap = next(cg.iter_candidates(blocking, s1, tgt, c))[2]
    assert precap.empty  # nothing materialized from a skipped bucket
