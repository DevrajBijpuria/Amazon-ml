import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import feature_engineering as fe  # noqa: E402
import phase2_5_models as pm  # noqa: E402
import threshold_tuning as tt  # noqa: E402


def make_candset(path: Path, n_s1: int = 120, seed: int = 0) -> Path:
    """Tiny synthetic candidate set: positives score high on every feature, negatives low, with noise.
    Includes zero-match entities, entities with no candidates and a positive blocking never retrieved."""
    rng = np.random.default_rng(seed)
    feats, pairs, ents = [], [], []
    for i in range(n_s1):
        s1, val = f"S1-{i:04d}", i % 2
        n_true = i % 3  # 0, 1, 2 true matches
        retrieved = n_true - (1 if i == 5 else 0)  # S1-0005 (validation): one positive never retrieved
        n_neg = 0 if i in (3, 9) else 4  # S1-0003 / S1-0009: zero-match, no candidates
        for j in range(retrieved + n_neg):
            label = int(j < retrieved)
            src = "s2" if j % 2 == 0 else "s3"
            feats.append(np.clip(rng.normal(0.75 if label else 0.3, 0.2, len(fe.FEATURES)), 0, 1))
            pairs.append((s1, f"{src.upper()}-{i}-{j}", src, label, 1 + j % 4, val))
        ents.append((s1, n_true, n_true, 0, val))
    path.mkdir(parents=True)
    np.save(path / "features.npy", np.array(feats, dtype=np.float32))
    pd.DataFrame(pairs, columns=["s1", "target", "source", "label", "blocks", "validation"]).to_csv(
        path / "pairs.tsv", sep="\t", index=False)
    pd.DataFrame(ents, columns=["s1", "n_true", "n_true_s2", "n_true_s3", "validation"]).to_csv(
        path / "entities.tsv", sep="\t", index=False)
    (path / "info.json").write_text(json.dumps({"scale": "synthetic", "features": fe.FEATURES, "seed": 42}))
    return path


def test_bootstrap_is_paired_and_deterministic():
    assert np.array_equal(pm.bootstrap_index(7), pm.bootstrap_index(7))
    f = np.linspace(0.2, 0.6, 7)
    idx = pm.bootstrap_index(7)
    out = pm.bootstrap({"B0": f, "same": f.copy(), "up": f + 0.1, "down": f - 0.1}, "B0", idx)["configurations"]
    assert out["same"]["diff_vs_B0_ci95"] == [0.0, 0.0] and out["same"]["verdict"] == "effectively equivalent"
    # paired: a constant shift gives a zero-width difference CI even though the per-config CIs overlap
    assert np.allclose(out["up"]["diff_vs_B0_ci95"], [0.1, 0.1]) and out["up"]["verdict"] == "better"
    assert out["down"]["verdict"] == "worse"
    assert out == pm.bootstrap({"B0": f, "same": f.copy(), "up": f + 0.1, "down": f - 0.1}, "B0", idx)["configurations"]


def test_feature_sets_are_subsets_and_differ_as_specified():
    sets = {k: v[1] for k, v in pm.feature_sets(fe.FEATURES).items()}
    all_f = set(fe.FEATURES)
    assert sets["B0"] == fe.FEATURES and len(all_f) == 47
    for k, fs in sets.items():
        assert set(fs) <= all_f and len(fs) == len(set(fs)) and fs, k
    diff = {k: all_f - set(fs) for k, fs in sets.items()}
    assert diff["F1"] == {"name_token_jaccard", "name_char3_jaccard", "name_edit_sim"}
    assert diff["F2"] == {"address_char3_jaccard", "address_missing_2"}
    assert diff["F3"] == {f for f in fe.FEATURES if f.startswith("matched_by_")} and len(diff["F3"]) == 5
    assert diff["F4"] == {"number_of_blocks_that_retrieved_pair"}
    assert diff["F6"] == {"name_token_jaccard", "name_char3_jaccard", "name_edit_sim", "name_norm_eq",
                          "name_core_token_jaccard", "name_token_set_ratio"}
    assert "name_edit_sim" not in sets["F5"] and "address_edit_sim" in sets["F5"] and "matched_by_numeric_block" not in sets["F5"]
    assert {f for f in sets["F7"] if f.startswith("name_")} == {"name_key_eq", "name_translit_char3_jaccard"}
    assert "address_token_jaccard" not in sets["F8"] and "address_missing_either" in sets["F8"] and "name_edit_sim" in sets["F8"]
    assert len({tuple(fs) for fs in sets.values()}) == len(sets)


def test_sweep_counts_zero_match_entity_without_candidates():
    scored = pd.DataFrame({"s1": ["a", "a"], "score": [0.9, 0.1], "label": [1, 0]})
    n_true = pd.Series({"a": 1, "z": 0}).sort_index()  # z: no true match, no candidates
    sel = tt.sweep(scored, n_true, [0.5])["selected"]
    assert sel["entities"] == 2 and sel["macro_f05"] == 1.0
    n_pred, tp, t = pm.per_entity(scored, n_true, 0.5)
    assert list(tt.entity_f05(n_pred, t, tp)) == [1.0, 1.0]
    assert tt.sweep(scored, n_true, [0.05])["selected"]["macro_f05"] < 1.0  # extra pred on "a" costs precision


def test_end_to_end_writes_all_three_files(tmp_path):
    cand = make_candset(tmp_path / "cand")
    cs = pm.load_candset(cand)
    assert "S1-0003" in cs["n_true_val"].index and cs["n_true_val"]["S1-0003"] == 0  # zero candidates still evaluated
    out = tmp_path / "art"
    assert pm.main(["--candset", str(cand), "--scale", "dev", "--out", str(out)]) == 0
    assert pm.main(["--candset", str(cand), "--scale", "10k", "--out", str(out)]) == 0  # read-modify-write
    fa = json.loads((out / "feature_ablation.json").read_text())
    mc = json.loads((out / "model_comparison.json").read_text())
    tr = json.loads((out / "threshold_robustness.json").read_text())
    assert set(fa) == set(mc) == set(tr) == {"dev", "10k"}
    d = fa["dev"]
    assert set(d["results"]) == {"B0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"}
    assert d["feature_sets"]["F1"]["features"] == pm.feature_sets(fe.FEATURES)["F1"][1]
    r = d["results"]["B0"]
    for k in ("selected_threshold", "macro_f05", "precision", "recall", "tp", "fp", "fn", "entities_with_prediction",
              "average_predictions_per_entity", "blocking_missed_positives", "classifier_missed_positives"):
        assert k in r
    assert r["blocking_missed_positives"] == 1
    assert r["fn"] == r["blocking_missed_positives"] + r["classifier_missed_positives"]
    assert d["b0_reproduction"]["matched"] is False and d["b0_reproduction"]["discrepancies"]  # synthetic != Phase 2
    assert set(d["bootstrap"]["configurations"]) >= {"B0", "F8", "B0-name_edit_sim", "B0-edit_sim_group"}
    diag = d["name_edit_sim_diagnosis"]
    assert {"correlation", "conditional_model_contribution"} <= set(diag)
    assert set(diag["conditional_model_contribution"]["standardized_coefficient"]) == {
        "B0", "univariate", "B0_minus_abs_r_ge_0.9_correlates"}
    m = mc["dev"]
    assert {"M0/B0", "M1/B0"} <= set(m["results"]) and m["results"]["M1/B0"]["fit_seconds"] >= 0
    assert m["results"]["M0/B0"]["macro_f05"] == r["macro_f05"]
    t = tr["dev"]["configurations"]["M0/B0"]
    ths = [row["threshold"] for row in t["thresholds"]]
    assert 0.52 in ths and 0.05 in ths and ths == sorted(ths)
    st = t["stress_test"]
    assert st["forced_prediction_rule"] is False
    assert st["true_zero_match"] + st["true_singleton"] + st["true_multi_match"] == tr["dev"]["protocol"]["validation_entities"]
    # deterministic: a second run reproduces the same numbers
    def untimed(res):
        return {k: {f: v for f, v in r.items() if not f.endswith("_seconds")} for k, r in res.items()}
    assert untimed(fa["dev"]["results"]) == untimed(fa["10k"]["results"])
    assert fa["dev"]["bootstrap"] == fa["10k"]["bootstrap"]
