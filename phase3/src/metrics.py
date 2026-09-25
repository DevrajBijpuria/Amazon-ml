"""Section 9/11 metrics of one run, the paired comparison with the baseline, and repeat-run hash checks."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "phase2" / "src"))
import phase2_5_models as pm  # noqa: E402  paired entity bootstrap (B=1000, seed 42)
import threshold_tuning as tt  # noqa: E402

SOURCES = ("s2", "s3")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validation_frame(art: Path) -> tuple:
    scores = pd.read_csv(art / "validation_scores.tsv", sep="\t", dtype={"s1": str, "target": str}, keep_default_na=False)
    n_true = pd.read_csv(art / "validation_entities.tsv", sep="\t", dtype={"s1": str},
                         keep_default_na=False).set_index("s1")["n_true"]
    return scores, n_true


def entity_behaviour(scores: pd.DataFrame, n_true: pd.Series, thr: float, empty_thr=None) -> dict:
    """Zero-match and singleton behaviour at the selected threshold (prediction = tt.accept)."""
    n_pred, tp, nt = pm.per_entity(scores, n_true, thr, empty_thr)
    fp = n_pred - tp
    zero, single = nt == 0, nt == 1
    return {
        "zero_match_entities": int(zero.sum()),
        "zero_match_false_merge_entities": int((zero & (n_pred > 0)).sum()),
        "zero_match_false_merge_predictions": int(fp[zero].sum()),
        "correctly_empty_entities": int((zero & (n_pred == 0)).sum()),
        "singleton_entities": int(single.sum()),
        "singleton_exact_correct": int((single & (tp == 1) & (n_pred == 1)).sum()),
        "singleton_missed": int((single & (tp == 0)).sum()),
        "singleton_with_extra_predictions": int((single & (fp > 0)).sum()),
        "extra_predictions_total": int(fp.sum()),
        "extra_predictions_on_matched_entities": int(fp[~zero].sum()),
        "blocking_missed_positives": int(nt.sum() - scores["label"].sum()),
        "classifier_missed_positives": int(((scores["label"] == 1).to_numpy() & ~tt.accept_frame(scores, thr, empty_thr)).sum()),
        "validation_candidate_recall": round(float(scores["label"].sum() / nt.sum()), 5) if nt.sum() else None,
    }


def blocking_metrics(cs: dict) -> dict:
    n = cs["source1_entities"]
    per = {}
    for s in SOURCES:
        c = cs[s]
        per[s] = {"precap_recall": c["precap_candidate_recall"], "final_recall": c["candidate_recall"],
                  "positives": c["positive_pairs_total"], "precap_recovered": c["precap_positive_pairs_recovered"],
                  "final_recovered": c["positive_pairs_recovered"], "precap_pairs": c["precap_candidate_pairs"],
                  "final_pairs": c["candidate_pairs"], "cap_hit_rate": round(c["cap"]["s1_entities_capped"] / n, 5),
                  "s1_capped": c["cap"]["s1_entities_capped"], "positives_lost_to_cap": c["cap"]["positive_pairs_dropped"],
                  "positives_lost_sharing_skipped_key": c["positives_lost_to_skipped_keys_any_rule"],
                  **{k: c[k] for k in ("mean_candidates_per_s1", "median_candidates_per_s1", "p95_candidates_per_s1",
                                       "p99_candidates_per_s1", "max_candidates_per_s1", "s1_with_zero_candidates")},
                  "recall_by_country": c["recall_by_country"],
                  "skipped_keys_by_rule": {r: v["skipped_key_count"] for r, v in c["skipped_keys"].items()},
                  "recall_by_rule": c["recall_by_rule"]}
    tot = sum(per[s]["positives"] for s in SOURCES)
    comb = {k: sum(per[s][k] for s in SOURCES) for k in ("positives", "precap_recovered", "final_recovered",
                                                          "precap_pairs", "final_pairs", "s1_capped")}
    comb.update({"precap_recall": round(comb["precap_recovered"] / tot, 5), "final_recall": round(comb["final_recovered"] / tot, 5),
                 "blocking_missed_positives": tot - comb["final_recovered"],
                 "cap_hit_rate": round(comb["s1_capped"] / (2 * n), 5),
                 "mean_candidates_per_s1_both_sources": round(sum(per[s]["mean_candidates_per_s1"] for s in SOURCES), 3)})
    return {"by_source": per, "combined": comb}


def runtime_groups(sec: dict) -> dict:
    """Template runtime groups. Index build is timed inside candidate generation and is subtracted from it."""
    g = lambda *keys: round(sum(sec.get(k, 0.0) for k in keys), 1)  # noqa: E731
    idx = g("train_index_s2", "train_index_s3")
    return {"load": g("train_load_s1_gt", "train_load_s2", "train_load_s3"),
            "normalize": g("train_normalize_s1", "train_normalize_s2", "train_normalize_s3"),
            "index": idx, "candidate": round(g("train_candidates_s2", "train_candidates_s3") - idx, 1),
            "features": g("train_features_s2", "train_features_s3"),
            "training": g("train_model"), "scoring": g("scoring"),
            "threshold_and_error_analysis": g("threshold_sweep", "error_analysis"),
            "output": g("output"), "validator": g("validator"), "total": g("total")}


def model_metrics(th: dict) -> dict:
    s = th.get("selected_with_policy", th["selected"])  # the metrics of the decision rule actually applied
    policy = {"empty_target_address_policy": th["empty_target_address_policy"],
              "without_policy": {k: th["selected"][k] for k in ("macro_f05", "precision", "recall", "tp", "fp", "fn")}} \
        if "empty_target_address_policy" in th else {}
    return {"threshold": th["selected_threshold"], "on_grid_edge": th["selected_on_grid_edge"], **policy,
            **{k: s[k] for k in ("macro_f05", "precision", "recall", "tp", "fp", "fn", "entities",
                                 "entities_with_prediction", "zero_match_entities_scored_1")},
            "per_source": {k: {m: v[m] for m in ("macro_f05", "precision", "recall")}
                           for k, v in th["per_source_at_selected"].items()}}


def compare(art: Path, base_art: Path) -> dict:
    """Paired bootstrap of per-entity F0.5 against the baseline on identical validation entities."""
    runs = {}
    for name, a in (("baseline", base_art), ("experiment", art)):
        scores, n_true = validation_frame(a)
        th = _load(a / "threshold_results.json")
        thr, empty = th["selected_threshold"], tt.policy_threshold(th)
        n_pred, tp, nt = pm.per_entity(scores, n_true, thr, empty)
        runs[name] = (n_true.index, tt.entity_f05(n_pred, nt, tp), entity_behaviour(scores, n_true, thr, empty))
    if not runs["baseline"][0].equals(runs["experiment"][0]):
        return {"baseline_artifacts": base_art.relative_to(REPO).as_posix(), "identical_validation_entities": False}
    boot = pm.bootstrap({k: v[1] for k, v in runs.items()}, "baseline", pm.bootstrap_index(len(runs["baseline"][0])))
    base_cs, cs = (blocking_metrics(_load(a / "candidate_statistics.json"))["combined"] for a in (base_art, art))
    eb, ee = runs["baseline"][2], runs["experiment"][2]
    return {"baseline_artifacts": base_art.relative_to(REPO).as_posix(), "identical_validation_entities": True,
            "bootstrap": boot["configurations"]["experiment"], "baseline_macro_f05": boot["configurations"]["baseline"]["macro_f05"],
            "delta_final_recall": round(cs["final_recall"] - base_cs["final_recall"], 5),
            "delta_blocking_missed_positives": cs["blocking_missed_positives"] - base_cs["blocking_missed_positives"],
            "final_pairs_ratio": round(cs["final_pairs"] / base_cs["final_pairs"], 4),
            "entity_behaviour_delta": {k: ee[k] - eb[k] for k in ee if isinstance(ee[k], int)},
            "baseline_entity_behaviour": eb}


def baseline_for(run_dir: Path, frozen_dir: Path) -> Path:
    """The comparison baseline: the immutable frozen_50000 artifacts at 50k, else the Phase 3 reference
    run of the frozen configuration at the same scale."""
    if run_dir.parent.name == "50k":
        return frozen_dir / "artifacts"
    return run_dir.parents[2] / "p3_frozen_baseline" / run_dir.parent.name / "run_1" / "artifacts"


def run_metrics(run_dir: Path, frozen_dir: Path) -> dict:
    art, man = run_dir / "artifacts", _load(run_dir / "manifest.json")
    th = _load(art / "threshold_results.json")
    scores, n_true = validation_frame(art)
    base = baseline_for(run_dir, frozen_dir)
    return {"experiment_id": man["experiment_id"], "scale": man["scale"], "repeat": man["repeat"],
            "blocking": blocking_metrics(_load(art / "candidate_statistics.json")), "model": model_metrics(th),
            "entities": entity_behaviour(scores, n_true, th["selected_threshold"], tt.policy_threshold(th)),
            "runtime": runtime_groups(man["runtime_seconds"]), "memory": man["memory"],
            "checks": {"section45": man["validation_output"]["section45_checks"], "validator_passed": man["validator"]["passed"]},
            "vs_baseline": compare(art, base) if (base / "validation_scores.tsv").exists() and base != art else None}


def repeat_check(run_dirs: list) -> dict:
    """Hash equality of every recorded output (and the inputs, config and code) across repeat runs."""
    mans = [_load(d / "manifest.json") for d in run_dirs]
    keys = ("output_sha256", "input_sha256", "config_sha256", "code_sha256")
    same = {k: all(m[k] == mans[0][k] for m in mans[1:]) for k in keys}
    files = {f: all(m["output_sha256"].get(f) == h for m in mans[1:]) for f, h in mans[0]["output_sha256"].items()}
    return {"runs": [d.relative_to(REPO).as_posix() for d in run_dirs], "identical": same, "per_output_file": files,
            "all_outputs_identical": same["output_sha256"]}
