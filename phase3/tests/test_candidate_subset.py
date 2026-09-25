import numpy as np
import pandas as pd

import candidate_generation as cg
import experiment_runner as er
import metrics


def _rows(path):
    lines = path.read_text(encoding="utf-8").splitlines()[1:]
    return {s: (ids.split(",") if ids else []) for s, ids in (l.split("\t") for l in lines)}


def test_every_match_is_a_candidate(mini_run):
    sub = er.write_validation_submission(mini_run["art"], mini_run["run"] / "output", mini_run["data"])
    matches, cands = _rows(sub["files"][0]), _rows(sub["files"][1])
    assert all(set(matches[s]) <= set(cands[s]) for s in matches)
    assert sub["section45_checks"]["matches_subset_of_candidates"]
    assert matches == {"S1-1": ["S2-3", "S3-2", "S3-5"], "S1-2": [], "S1-3": []}
    # candidates are exactly the scored pairs, nothing more
    assert cands == {"S1-1": ["S2-1", "S2-3", "S3-2", "S3-5"], "S1-2": ["S2-4", "S3-1"], "S1-3": []}


def test_no_group_exceeds_the_cap():
    rng = np.random.default_rng(0)
    s1 = pd.DataFrame({"name_key": ["acme"] * 3, "addr_norm": ["1 main st"] * 3})
    tgt = pd.DataFrame({"name_key": [f"acme{i}" for i in range(300)], "addr_norm": ["1 main st"] * 300})
    cands = pd.DataFrame({"s1_idx": np.repeat([0, 1, 2], [250, 120, 40]),
                          "t_idx": np.concatenate([rng.permutation(300)[:250], rng.permutation(300)[:120], np.arange(40)]),
                          "blocks": rng.integers(1, 2**13, 410)})
    for cap in (100, 150, 200):
        kept, dropped = cg._cap(cands, s1, tgt, cap)
        sizes = kept.groupby("s1_idx").size()
        assert sizes.max() <= cap and sizes[2] == 40  # groups under the cap are kept whole
        assert len(kept) + len(dropped) == len(cands)
        assert kept.merge(dropped, on=["s1_idx", "t_idx"]).empty


def test_entity_behaviour_counts(mini_run):
    scores, n_true = metrics.validation_frame(mini_run["art"])
    eb = metrics.entity_behaviour(scores, n_true, 0.8)
    assert eb["zero_match_entities"] == 1 and eb["correctly_empty_entities"] == 1
    assert eb["zero_match_false_merge_entities"] == 0
    assert eb["singleton_entities"] == 1 and eb["singleton_missed"] == 1 and eb["singleton_exact_correct"] == 0
    assert eb["extra_predictions_total"] == 1  # S3-5 for S1-1
    assert eb["blocking_missed_positives"] == 1 and eb["classifier_missed_positives"] == 0
    assert eb["validation_candidate_recall"] == round(2 / 3, 5)
