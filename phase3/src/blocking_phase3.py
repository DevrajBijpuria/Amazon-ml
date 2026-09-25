"""Phase 3 blocking: the frozen Phase 2.5 rules plus the Phase 3 address/locality rules.

The rules, the key-frequency cutoff, the union and the per-S1 cap are the unchanged Phase 2 code in
phase2/src/candidate_generation.py (house_locality, postal_locality, state_locality, address_signature
and name_phonetic_token already exist there behind ``use_<rule>_block`` switches). This module only
resolves and checks a Phase 3 rule set from its config switches and states the candidate priority.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "phase2" / "src"))
import candidate_generation as cg  # noqa: E402

B0_RULES = ["exact_name", "exact_address", "numeric", "name_prefix", "address_prefix", "name_ngram",
            "name_phonetic", "address_token"]
FROZEN_RULES = B0_RULES + ["house_locality", "state_locality", "address_signature"]  # phase2_5_frozen.yaml
LOCALITY_RULES = ["house_locality", "postal_locality", "state_locality", "address_signature"]

# B0's own priority (candidate_generation._cap), identical for every experiment. It uses no labels and
# no model scores; t_idx (the target's position in its source file) makes the order total.
CANDIDATE_PRIORITY = ("per S1 and target source, sort by (1) number of blocking rules that retrieved the pair, "
                      "descending; (2) max(name_key edit similarity, addr_norm edit similarity), descending; "
                      "(3) target position in the source file, ascending; keep the first max_candidates_per_source1. "
                      "Groups at or under the cap are kept whole.")


def rule_set(cgc: dict) -> list:
    """Enabled rules in the fixed RULE_FLAGS order; an unknown ``use_*_block`` switch is a config error."""
    known = set(cg.RULE_FLAGS.values()) | {"use_country_block"}
    unknown = sorted(k for k in cgc if k.startswith("use_") and k not in known)
    if unknown:
        raise ValueError(f"unknown blocking switches: {unknown}")
    return cg.enabled_rules(cgc)
