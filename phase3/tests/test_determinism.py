import json

import numpy as np
import pandas as pd
import yaml

import blocking_phase3 as bp
import candidate_generation as cg
import experiment_runner as er
import feature_engineering as fe
import metrics

ROWS = [["Acme Traders", "12 Gandhi Road, Koramangala, Bengaluru, Karnataka 560034", "India"],
        ["Acme Trading Co", "12 Gandhi Rd, Koramangala, Bengaluru 560034", "India"],
        ["Unique Investment", "4 Rue Lepic, Montmartre, Paris 75018", "France"],
        ["Yunik Investament", "4 Rue Lepic, Paris 75018", "France"],
        ["Blue Cafe", "7 Main St, Springfield, IL 62701", "US"]]


def _records(prefix, rows):
    return fe.prepare_records(pd.DataFrame([[f"{prefix}-{i}", *r] for i, r in enumerate(rows)],
                                           columns=["entity_id", "business_name", "business_address", "country"]))


def _candidates(cgc):
    s1, tgt = _records("S1", ROWS), _records("S2", ROWS[::-1] + ROWS)
    blocking = cg.build_blocking(s1, tgt, cgc)
    return pd.concat([k for *_, k, _ in cg.iter_candidates(blocking, s1, tgt, cgc)], ignore_index=True)


def test_repeated_blocking_gives_identical_candidates():
    cgc = {**yaml.safe_load((er.REPO / "phase3" / "config" / "p3_locality_phonetic_cap100.yaml").read_text())
           ["candidate_generation"], "max_candidates_per_source1": 3, "source1_chunk_size": 2}
    a, b = _candidates(cgc), _candidates(cgc)
    assert len(a) and a.equals(b)
    assert a.groupby("s1_idx").size().max() <= 3


def test_cap_ignores_input_row_order():
    s1 = pd.DataFrame({"name_key": ["acme", "blue"], "addr_norm": ["1 main st", ""]})
    tgt = pd.DataFrame({"name_key": ["acme", "acmx", "acm", "blue", "bleu", "blu"] * 10, "addr_norm": ["1 main st"] * 60})
    cands = pd.DataFrame({"s1_idx": np.repeat([0, 1], 60), "t_idx": np.tile(np.arange(60), 2),
                          "blocks": np.tile([1, 3, 1, 7, 1, 1], 20)})
    ref, _ = cg._cap(cands, s1, tgt, 25)
    for seed in range(5):
        kept, _ = cg._cap(cands.sample(frac=1, random_state=seed), s1, tgt, 25)
        assert kept.reset_index(drop=True).equals(ref.reset_index(drop=True))


def test_priority_is_recorded_and_label_free():
    assert "label" not in bp.CANDIDATE_PRIORITY and "score" not in bp.CANDIDATE_PRIORITY


def test_repeat_check_compares_hashes(tmp_path):
    base = {"output_sha256": {"output/candidate_pairs.tsv": "a", "output/matching_results.tsv": "b"},
            "input_sha256": {"x": "1"}, "config_sha256": "c", "code_sha256": "d"}
    runs = []
    for k, cand_hash in ((1, "a"), (2, "a"), (3, "z")):
        d = er.REPO / "phase3" / "output" / "_pytest" / f"run_{k}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(json.dumps({**base, "output_sha256": {**base["output_sha256"],
                                                                               "output/candidate_pairs.tsv": cand_hash}}))
        runs.append(d)
    try:
        assert metrics.repeat_check(runs[:2])["all_outputs_identical"]
        bad = metrics.repeat_check([runs[0], runs[2]])
        assert not bad["all_outputs_identical"] and bad["per_output_file"] == {
            "output/candidate_pairs.tsv": False, "output/matching_results.tsv": True}
    finally:
        for d in runs:
            (d / "manifest.json").unlink()
            d.rmdir()
        runs[0].parent.rmdir()
