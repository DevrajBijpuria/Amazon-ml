"""Empty-target-address decision policy: one shared helper (threshold_tuning.accept) for validation and production."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "phase3" / "src"), str(REPO / "phase2" / "src")]

import inference as inf  # noqa: E402
import metrics  # noqa: E402
import phase2_5_models as pm  # noqa: E402
import threshold_tuning as tt  # noqa: E402

CONFIG = REPO / "phase3" / "config"
COL = tt.EMPTY_ADDRESS_COLUMN


def test_disabled_policy_is_exactly_score_ge_threshold():
    rng = np.random.default_rng(0)
    s, flag = rng.random(1000), rng.integers(0, 2, 1000)
    for thr in (0.05, 0.8, 0.95):
        want = s >= thr
        assert np.array_equal(tt.accept(s, thr), want)
        assert np.array_equal(tt.accept(s, thr, flag, None), want)  # flag is ignored while disabled


def test_address_present_uses_base_threshold():
    s = np.array([0.79, 0.80, 0.81, 0.94])
    assert tt.accept(s, 0.8, np.zeros(4), 0.95).tolist() == [False, True, True, True]


@pytest.mark.parametrize("score,expected", [(0.94, False), (0.95, True), (0.97, True)])
def test_empty_address_needs_095(score, expected):
    assert tt.accept(np.array([score]), 0.8, np.array([1]), 0.95).tolist() == [expected]


def test_base_threshold_above_095_uses_max():
    s = np.array([0.955, 0.965, 0.975])
    assert tt.accept(s, 0.96, np.array([1, 1, 1]), 0.95).tolist() == [False, True, True]
    assert tt.accept(s, 0.96, np.array([0, 0, 0]), 0.95).tolist() == [False, True, True]


def test_enabled_policy_without_flag_fails_loudly():
    with pytest.raises(ValueError):
        tt.accept(np.array([0.9]), 0.8, None, 0.95)
    with pytest.raises(ValueError):
        inf.chunk_lines(0, 1, np.array([0]), np.array(["S2-1"]), np.array([0.9]), 0.8, None, 0.95)


def frame():
    return pd.DataFrame({"s1": ["S1-1", "S1-1", "S1-1", "S1-2", "S1-2", "S1-3"],
                         "target": ["S2-1", "S2-2", "S3-3", "S2-4", "S3-5", "S2-6"],
                         "source": ["s2", "s2", "s3", "s2", "s3", "s2"],
                         "label": [1, 0, 1, 1, 0, 0],
                         "score": [0.90, 0.94, 0.96, 0.85, 0.99, 0.50],
                         COL: [0, 1, 1, 0, 1, 0]})


N_TRUE = pd.Series([2, 1, 0, 0], index=pd.Index(["S1-1", "S1-2", "S1-3", "S1-4"], name="s1"))


def matches_validation(df, thr, empty):
    n_pred, tp, _ = pm.per_entity(df, N_TRUE, thr, empty)
    return n_pred.tolist(), tp.tolist()


def matches_production(df, thr, empty):
    pos = pd.Series(np.arange(len(N_TRUE)), index=N_TRUE.index)
    lines = inf.chunk_lines(0, len(N_TRUE), pos.loc[df["s1"]].to_numpy(), df["target"].to_numpy(), df["score"].to_numpy(),
                            thr, df[COL].to_numpy(), empty)
    return [ln.rstrip("\n").split("\t") for ln in lines]


def test_validation_and_production_agree_and_candidates_never_change():
    df = frame()
    off, on = matches_production(df, 0.8, None), matches_production(df, 0.8, 0.95)
    assert [c for c, _ in off] == [c for c, _ in on]  # candidate_pairs.tsv content is identical
    assert [m for _, m in off] == ["S2-1,S2-2,S3-3", "S2-4,S3-5", "", ""]
    assert [m for _, m in on] == ["S2-1,S3-3", "S2-4,S3-5", "", ""]  # S2-2: empty address, 0.94 < 0.95
    # validation metrics count exactly the production matches
    n_pred, _ = matches_validation(df, 0.8, 0.95)
    assert n_pred == [len(m.split(",")) if m else 0 for _, m in on]
    ev = tt.evaluate(df, N_TRUE, 0.8, 0.95)
    assert (ev["tp"], ev["fp"]) == (3, 1)
    assert inf.predictions(list(N_TRUE.index), df, 0.8, 0.95)["matched_entity_ids"].tolist() == [m for _, m in on]


def test_every_decision_path_calls_the_shared_helper(monkeypatch):
    calls = []
    real = tt.accept

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(tt, "accept", spy)
    df = frame()
    for fn in (lambda: tt.evaluate(df, N_TRUE, 0.8, 0.95), lambda: pm.per_entity(df, N_TRUE, 0.8, 0.95),
               lambda: metrics.entity_behaviour(df, N_TRUE, 0.8, 0.95), lambda: matches_production(df, 0.8, 0.95),
               lambda: inf.predictions(list(N_TRUE.index), df, 0.8, 0.95)):
        before = len(calls)
        fn()
        assert len(calls) > before


def test_sweep_selects_base_threshold_ignoring_the_policy_column():
    df = frame()
    grid = [0.5, 0.8, 0.9, 0.95]
    assert tt.sweep(df, N_TRUE, grid) == tt.sweep(df.drop(columns=[COL]), N_TRUE, grid)


def test_policy_only_in_the_intervention_config():
    for path in sorted(CONFIG.glob("*.yaml")):
        th = yaml.safe_load(path.read_text(encoding="utf-8"))["threshold"]
        assert ("empty_target_address_threshold" in th) == (path.name == "p3_empty_address_policy.yaml"), path.name
    new = yaml.safe_load((CONFIG / "p3_empty_address_policy.yaml").read_text(encoding="utf-8"))
    base = yaml.safe_load((CONFIG / "p3_frozen_baseline.yaml").read_text(encoding="utf-8"))
    assert new["threshold"].pop("empty_target_address_threshold") == 0.95
    assert {**new, "experiment_id": base["experiment_id"]} == base  # everything else identical


def test_policy_threshold_reads_recorded_policy():
    assert tt.policy_threshold({"selected_threshold": 0.8}) is None
    assert tt.policy_threshold({"empty_target_address_policy": {"empty_target_address_threshold": 0.95}}) == 0.95
