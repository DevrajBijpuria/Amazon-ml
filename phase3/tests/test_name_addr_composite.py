"""name_addr_composite (Phase 3 intervention): key format, per-rule cutoff, and flag-off equivalence."""
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "phase3" / "src"), str(REPO / "phase2" / "src")]

import blocking_phase3 as bp  # noqa: E402
import candidate_generation as cg  # noqa: E402
import feature_engineering as fe  # noqa: E402

CONFIG = REPO / "phase3" / "config"
RULE = "name_addr_composite"


def records(rows):
    return fe.prepare_records(pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"]))


S1 = records([["S1-1", "Acme Traders Pvt Ltd", "12 Gandhi Road, Koramangala, Bengaluru, Karnataka 560034", "India"],
              ["S1-2", "Yunik Investament", "4 Rue Lepic, Montmartre, Paris 75018", "Atlantis"]])
TGT = records([["S2-1", "Acme Traders", "12 Gandhi Rd, Koramangala, Bengaluru, KA 560034", "india"],
               ["S2-2", "Unique Investment", "4 Rue Lepic, Montmartre, Paris 75018", "atlantis"]])


def cgc(name):
    return yaml.safe_load((CONFIG / name).read_text(encoding="utf-8"))["candidate_generation"]


def keys(rec, cfg):
    token_df = {k: cg.rare_token_frequencies([S1, TGT], k) for k in ("addr_norm", "name_core")}
    return cg.rule_keys(rec, RULE, cfg, token_df).values.tolist()


def test_keys_are_rarest_name_tokens_times_rarest_address_tokens():
    c = cgc("p3_name_addr_composite.yaml")
    assert keys(S1, c) == [[0, "india|na|acme|bengaluru"], [0, "india|na|acme|gandhi"],
                           [0, "india|na|traders|bengaluru"], [0, "india|na|traders|gandhi"]]
    # the address half uses the same rarest-2 locality tokens as house_locality
    token_df = {"addr_norm": cg.rare_token_frequencies([S1, TGT], "addr_norm")}
    hl = [k.split("|")[-1] for i, k in cg.rule_keys(S1, "house_locality", c, token_df).values.tolist() if i == 0]
    assert sorted({k.split("|")[-1] for _, k in keys(S1, c)}) == sorted(hl)
    # a name token seen once (typo/variant) can never pair two records -> no key for S1-2
    assert all(i == 0 for i, _ in keys(S1, c))
    assert keys(TGT, c)[:4] == keys(S1, c)  # variant spellings on the target side give the same keys


def test_missing_name_or_address_emits_no_key():
    rec = records([["S1-9", "Acme Traders", "", "India"], ["S1-8", "", "12 Gandhi Road, Bengaluru", "India"]])
    token_df = {k: cg.rare_token_frequencies([rec, rec], k) for k in ("addr_norm", "name_core")}
    assert cg.rule_keys(rec, RULE, cgc("p3_name_addr_composite.yaml"), token_df).empty


def test_flag_is_off_in_every_existing_config_and_on_only_in_the_intervention():
    for path in sorted(CONFIG.glob("*.yaml")):
        c = yaml.safe_load(path.read_text(encoding="utf-8"))["candidate_generation"]
        on = RULE in bp.rule_set(c)
        assert on == (path.name == "p3_name_addr_composite.yaml"), path.name
        if not on:
            assert "max_block_size_by_rule" not in c, path.name
    c = cgc("p3_name_addr_composite.yaml")
    base = cgc("p3_frozen_baseline.yaml")
    assert bp.rule_set(c) == bp.rule_set(base) + [RULE]  # appended last: existing rule order and bits unchanged
    assert {k: v for k, v in c.items() if k not in ("use_name_addr_composite_block", "max_block_size_by_rule")} == base
    assert c["max_block_size_by_rule"] == {RULE: 25} and c["max_block_size"] == 100 == base["max_block_size"]
    assert c["max_candidates_per_source1"] == 100 == base["max_candidates_per_source1"]


def test_existing_bits_and_flags_unchanged():
    assert fe.RULE_BITS == {"exact_name": 1, "exact_address": 2, "numeric": 4, "name_prefix": 8, "address_prefix": 16,
                            "name_ngram": 32, "name_phonetic": 64, "address_token": 128, "name_phonetic_token": 256,
                            "house_locality": 512, "postal_locality": 1024, "state_locality": 2048,
                            "address_signature": 4096, RULE: 8192}
    assert list(cg.RULE_FLAGS)[-1] == RULE and len(cg.RULE_FLAGS) == 14


def test_per_rule_cutoff_applies_only_to_the_named_rule():
    tgt = records([[f"S2-{i}", f"Acme Traders {i}", "7 Main Street, Springfield, IL 62701", "US"] for i in range(5)])
    s1 = records([["S1-1", "Acme Traders", "7 Main St, Springfield, Illinois 62701", "US"]])
    c = {**cgc("p3_name_addr_composite.yaml"), "max_block_size": 100, "max_block_size_by_rule": {RULE: 4}}
    blocking = cg.build_blocking(s1, tgt, c)
    assert blocking[RULE]["max_block_size"] == 4 and len(blocking[RULE]["t"]) == 0
    assert (blocking[RULE]["skipped"]["frequency"] == 5).all()
    sl = blocking["state_locality"]
    assert sl["max_block_size"] == 100 and sl["skipped"].empty and sl["t"]["t_idx"].nunique() == 5


def test_flag_off_blocking_is_identical_to_the_pre_change_path():
    """Frozen config: build_blocking == apply_cutoff(rule_tables, max_block_size) with no override, and the
    candidates are identical to those of the same config with the composite switched on minus its bit."""
    s1, tgt = S1, TGT
    base = cgc("p3_frozen_baseline.yaml")
    new = cg.build_blocking(s1, tgt, base)
    old = cg.apply_cutoff(cg.rule_tables(s1, tgt, base), base["max_block_size"])
    assert list(new) == list(old) == bp.rule_set(base)
    for r in new:
        for part in ("s1", "t", "t_skipped"):
            pd.testing.assert_frame_equal(new[r][part], old[r][part])
        assert new[r]["max_block_size"] == base["max_block_size"]
    pre_off = next(cg.iter_candidates(new, s1, tgt, base))[2]
    on = cg.build_blocking(s1, tgt, cgc("p3_name_addr_composite.yaml"))
    pre_on = next(cg.iter_candidates({r: b for r, b in on.items() if r != RULE}, s1, tgt, base))[2]
    pd.testing.assert_frame_equal(pre_off.reset_index(drop=True), pre_on.reset_index(drop=True))
