"""Phase 2.5 feature ablation, name_edit_sim diagnosis, model comparison and threshold robustness
(spec Steps 7-11) on a frozen candidate set.

    python phase2/src/phase2_5_models.py --candset phase2/output/phase2_5/candsets/<scale>/b0 --scale <scale>

Every experiment uses the Phase 2 B0 protocol: same training rows (sampled once on all 47 features),
StandardScaler + LogisticRegression (M0) or HistGradientBoosting (M1), all validation pairs scored,
threshold chosen on validation macro F0.5 per Source-1 entity. Differences between configurations are
judged with a paired entity bootstrap (B=1000, seed 42, one resample matrix shared by every configuration).
Results merge into artifacts/phase2_5/{feature_ablation,model_comparison,threshold_robustness}.json by scale.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matching_model as mm  # noqa: E402
import threshold_tuning as tt  # noqa: E402

log = logging.getLogger("phase2_5_models")
REPO = Path(__file__).resolve().parents[2]
SEED = 42
B = 1000
M0_CFG = {"C": 1.0, "max_iter": 2000}
GRID = tt.threshold_grid({"grid_start": 0.05, "grid_stop": 0.95, "grid_step": 0.05})
FINE_GRID = tt.threshold_grid({"grid_start": 0.50, "grid_stop": 0.90, "grid_step": 0.02})
# Phase 2 report, dev candidate set (5k S1 / 100k targets per source), M0 on B0
DEV_EXPECTED = {"macro_f05": 0.9709, "selected_threshold": 0.70, "fp": 44, "fn": 183,
                "classifier_missed_positives": 111, "blocking_missed_positives": 72}
NAME_SIM = ["name_norm_eq", "name_key_eq", "name_token_jaccard", "name_core_token_jaccard", "name_char3_jaccard",
            "name_translit_char3_jaccard", "name_translit_edit_sim", "name_token_set_ratio", "name_phonetic_eq",
            "name_phonetic_edit_sim"]


# ---------------------------------------------------------------- data
def load_candset(path: Path) -> dict:
    """Candidate set + the fixed B0 protocol pieces (fit rows, validation rows, n_true per validation entity)."""
    info = json.loads((path / "info.json").read_text(encoding="utf-8"))
    names = info["features"]
    X = np.load(path / "features.npy")
    tsv = {"sep": "\t", "dtype": {"s1": str, "target": str, "source": str}, "keep_default_na": False}
    pairs = pd.read_csv(path / "pairs.tsv", **tsv)
    ents = pd.read_csv(path / "entities.tsv", **tsv)
    if X.shape != (len(pairs), len(names)):
        raise ValueError(f"features.npy {X.shape} does not match pairs.tsv ({len(pairs)}) x features ({len(names)})")
    y = pairs["label"].to_numpy(np.int8)
    train_idx = np.flatnonzero(pairs["validation"].to_numpy() == 0)
    # hard negatives are ranked on name_char3 + address_char3 whatever the model's features: same rows everywhere
    fit_idx = train_idx[mm.sample_training_pairs(y[train_idx], pd.DataFrame(X[train_idx], columns=names),
                                                 negative_ratio=5, hard_fraction=0.5, seed=SEED)]
    val_idx = np.flatnonzero(pairs["validation"].to_numpy() == 1)
    val_ents = ents[ents["validation"] == 1]
    n_true_val = pd.Series(val_ents["n_true"].to_numpy(), index=val_ents["s1"].to_numpy()).sort_index()
    return {"info": info, "names": names, "X": X, "y": y, "pairs": pairs, "train_idx": train_idx,
            "fit_idx": fit_idx, "val_idx": val_idx, "n_true_val": n_true_val,
            "val_meta": pairs.iloc[val_idx][["s1", "label"]].reset_index(drop=True)}


def cols(cs: dict, features: list) -> np.ndarray:
    return np.array([cs["names"].index(f) for f in features])


def candidate_set_summary(cs: dict) -> dict:
    """Step 13 fields that are fixed for every configuration on this candidate set."""
    ents = cs["pairs"].groupby("s1").size()
    per_s1 = ents.reindex(cs["n_true_val"].index.union(ents.index), fill_value=0)
    info = cs["info"]
    return {"candidate_pairs": int(len(cs["pairs"])), "validation_candidate_pairs": int(len(cs["val_idx"])),
            "average_candidates_per_s1_with_candidates": round(float(ents.mean()), 3) if len(ents) else 0.0,
            "p95_candidates_per_s1_with_candidates": float(np.percentile(ents, 95)) if len(ents) else 0.0,
            "average_candidates_per_validation_s1": round(float(per_s1.reindex(cs["n_true_val"].index).mean()), 3),
            "candidate_recall": info.get("recall"), "blocking": info.get("blocking"), "seed": info.get("seed")}


# ---------------------------------------------------------------- feature sets
def feature_sets(names: list) -> dict:
    """{name: (description, columns in info.json order)}; raises if a named feature is not in the candidate set."""
    def need(*fs):
        missing = [f for f in fs if f not in names]
        if missing:
            raise KeyError(f"features not in candidate set: {missing}")
        return set(fs)

    def minus(drop):
        return [f for f in names if f not in drop]

    blocks = [f for f in names if f.startswith("matched_by_")]
    ctx = [f for f in names if f.startswith(("target_source_", "matched_by_"))] + list(need("number_of_blocks_that_retrieved_pair"))
    is_name = lambda f: f.startswith(("name_", "legal_suffix_"))  # noqa: E731
    f5 = need("name_norm_eq", "name_key_eq", "name_phonetic_eq", "legal_suffix_present_1", "legal_suffix_present_2",
              "legal_suffix_equal", "name_script_mismatch", "name_missing_1", "name_missing_2",
              "house_number_both_present", "house_number_equal", "postal_both_present", "postal_equal",
              "state_both_present", "state_equal", "country_equal", "target_source_is_s2", "target_source_is_s3")
    f7_extra = need("name_key_eq", "name_translit_char3_jaccard")
    f8_extra = need("address_missing_either")
    return {
        "B0": ("all current features", list(names)),
        "F1": ("remove redundant name similarity (native-script duplicates of the transliterated versions, r >= 0.9)",
               minus(need("name_token_jaccard", "name_char3_jaccard", "name_edit_sim"))),
        "F2": ("remove redundant address similarity", minus(need("address_char3_jaccard", "address_missing_2"))),
        "F3": ("remove block-membership features", minus(set(blocks))),
        "F4": ("remove number_of_blocks_that_retrieved_pair", minus(need("number_of_blocks_that_retrieved_pair"))),
        "F5": ("structural/equality/address evidence only",
               [f for f in names if f in f5 or f.startswith("address_")]),
        "F6": ("transliteration-emphasized (drop native-script name similarity; keep translit/phonetic and name_key_eq)",
               minus(need("name_token_jaccard", "name_char3_jaccard", "name_edit_sim", "name_norm_eq",
                          "name_core_token_jaccard", "name_token_set_ratio"))),
        "F7": ("address-heavy: all non-name features + name_key_eq + name_translit_char3_jaccard",
               [f for f in names if not is_name(f) or f in f7_extra]),
        "F8": ("name-heavy: name_*, legal_suffix_*, context features + address_missing_either",
               [f for f in names if is_name(f) or f in ctx or f in f8_extra]),
    }


# ---------------------------------------------------------------- models and evaluation
def fit(cs: dict, features: list, model: str):
    Xf = cs["X"][np.ix_(cs["fit_idx"], cols(cs, features))]
    yf = cs["y"][cs["fit_idx"]]
    if model == "M0":
        return mm.train_model(pd.DataFrame(Xf, columns=features), yf, M0_CFG, SEED)
    return HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, max_iter=200).fit(
        Xf.astype(np.float64), yf)


def per_entity(scored: pd.DataFrame, n_true: pd.Series, thr: float) -> tuple:
    """n_pred, tp, n_true per validation entity (n_true order). Prediction = score >= thr, nothing else."""
    sel = scored[scored["score"] >= thr]
    n_pred = sel.groupby("s1").size().reindex(n_true.index, fill_value=0).to_numpy()
    tp = sel.groupby("s1")["label"].sum().reindex(n_true.index, fill_value=0).to_numpy()
    return n_pred, tp, n_true.to_numpy()


def run_config(cs: dict, features: list, model: str = "M0") -> dict:
    """Train on the fixed fit rows, score every validation pair, sweep the standard grid."""
    t0 = time.perf_counter()
    est = fit(cs, features, model)
    t1 = time.perf_counter()
    scores = est.predict_proba(cs["X"][np.ix_(cs["val_idx"], cols(cs, features))].astype(np.float64))[:, 1]
    t2 = time.perf_counter()
    scored = cs["val_meta"].assign(score=scores)
    sw = tt.sweep(scored, cs["n_true_val"], GRID)
    thr, s = sw["selected_threshold"], sw["selected"]
    pos_in_cands = int(scored["label"].sum())
    n_pred, tp, n_true = per_entity(scored, cs["n_true_val"], thr)
    res = {"model": model, "n_features": len(features), "selected_threshold": thr,
           "macro_f05": s["macro_f05"], "precision": s["precision"], "recall": s["recall"],
           "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
           "entities_with_prediction": s["entities_with_prediction"],
           "average_predictions_per_entity": s["average_predictions_per_entity"],
           "blocking_missed_positives": int(n_true.sum()) - pos_in_cands,
           "classifier_missed_positives": int(((scored["label"] == 1) & (scored["score"] < thr)).sum()),
           "selected_on_grid_edge": sw["selected_on_grid_edge"],
           "fit_seconds": round(t1 - t0, 3), "scoring_seconds": round(t2 - t1, 3),
           "sweep_macro_f05": {str(r["threshold"]): r["macro_f05"] for r in sw["sweep"]}}
    if model == "M0":
        res["standardized_coefficients"] = mm.coefficients(est, features)
    return {"result": res, "f": tt.entity_f05(n_pred, n_true, tp), "scored": scored}


def bootstrap_index(n: int, b: int = B, seed: int = SEED) -> np.ndarray:
    """One resample matrix (b x n entity positions) shared by all configurations -> paired comparisons."""
    return np.random.default_rng(seed).integers(0, n, size=(b, n))


def bootstrap(f_by_cfg: dict, base: str, idx: np.ndarray) -> dict:
    """95% percentile CI of macro F0.5 per configuration and paired CI of (configuration - base)."""
    boot = {k: f[idx].mean(axis=1) for k, f in f_by_cfg.items()}
    out = {}
    for k, v in boot.items():
        d = v - boot[base]
        lo, hi = np.percentile(d, [2.5, 97.5])
        verdict = "effectively equivalent" if lo <= 0 <= hi else ("better" if lo > 0 else "worse")
        out[k] = {"macro_f05": float(f_by_cfg[k].mean()), "ci95": [float(x) for x in np.percentile(v, [2.5, 97.5])],
                  f"diff_vs_{base}": float(f_by_cfg[k].mean() - f_by_cfg[base].mean()),
                  f"diff_vs_{base}_ci95": [float(lo), float(hi)], "verdict": verdict}
    return {"B": int(idx.shape[0]), "seed": SEED, "entities": int(idx.shape[1]), "base": base,
            "method": "paired percentile bootstrap over validation S1 entities (same resample matrix for all)",
            "configurations": out}


# ---------------------------------------------------------------- Step 8
def pearson(a: np.ndarray, b: np.ndarray):
    a, b = a.astype(np.float64), b.astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def match_rate_table(x: pd.Series, y: np.ndarray) -> list:
    bins = pd.qcut(x, 10, duplicates="drop") if x.nunique() > 1 else pd.Series(["all"] * len(x), index=x.index)
    g = pd.DataFrame({"bin": bins.astype(str), "y": y, "x": x}).groupby("bin", sort=False)
    t = g.agg(n=("y", "size"), positives=("y", "sum"), lo=("x", "min")).sort_values("lo")
    return [{"bin": b, "n": int(r.n), "positives": int(r.positives), "match_rate": float(r.positives / r.n)}
            for b, r in t.iterrows()]


def edit_sim_diagnosis(cs: dict, runs: dict, f_by_cfg: dict) -> dict:
    """Correlation/multicollinearity (training rows) kept separate from conditional model contribution."""
    names, tr = cs["names"], cs["train_idx"]
    X = pd.DataFrame(cs["X"][tr], columns=names)
    y = cs["y"][tr]
    e = X["name_edit_sim"].to_numpy()
    corr_all = {f: pearson(e, X[f].to_numpy()) for f in names if f != "name_edit_sim"}
    group = sorted(f for f, r in corr_all.items() if r is not None and abs(r) >= 0.9)
    others = [f for f in names if f != "name_edit_sim"]
    r2 = float(LinearRegression().fit(X[others], e).score(X[others], e))
    vif = 1.0 / (1.0 - r2) if r2 < 1 else float("inf")

    cfgs = {"B0-name_edit_sim": [f for f in names if f != "name_edit_sim"],
            "B0-edit_sim_group": [f for f in names if f not in group and f != "name_edit_sim"],
            "B0-edit_sim_correlates(keep name_edit_sim)": [f for f in names if f not in group],
            "name_edit_sim_only": ["name_edit_sim"]}
    for k, fs in cfgs.items():
        if k not in runs:
            runs[k] = run_config(cs, fs)
            f_by_cfg[k] = runs[k]["f"]
    coef = lambda k: runs[k]["result"]["standardized_coefficients"]["name_edit_sim"]  # noqa: E731

    tb = pd.cut(X["name_translit_edit_sim"], [-np.inf, 0.5, 0.8, np.inf], right=False, labels=["<0.5", "0.5-0.8", ">=0.8"])
    within = {}
    for lab in ["<0.5", "0.5-0.8", ">=0.8"]:
        m = (tb == lab).to_numpy()
        within[lab] = {"n": int(m.sum()), "positives": int(y[m].sum()),
                       "corr_name_edit_sim_label": pearson(e[m], y[m]) if m.sum() > 1 else None,
                       "by_name_edit_sim_decile": match_rate_table(X.loc[m, "name_edit_sim"], y[m]) if m.any() else []}
    return {
        "rows": "training rows (validation == 0), all candidate pairs",
        "correlation": {
            "with_label": pearson(e, y),
            "with_name_similarity_features": {f: corr_all[f] for f in NAME_SIM if f in corr_all},
            "abs_r_ge_0.9_group": group,
            "all_features": corr_all,
            "mean_positive": float(e[y == 1].mean()) if (y == 1).any() else None,
            "mean_negative": float(e[y == 0].mean()) if (y == 0).any() else None,
            "raw_std": float(e.std()),
            "vif": {"r2_on_other_b0_features": r2, "vif": vif},
        },
        "conditional_model_contribution": {
            "note": "M0 pipeline standardizes inputs, so coefficients are per 1 SD of the feature",
            "standardized_coefficient": {"B0": coef("B0"), "univariate": coef("name_edit_sim_only"),
                                         "B0_minus_abs_r_ge_0.9_correlates": coef("B0-edit_sim_correlates(keep name_edit_sim)")},
            "match_rate_by_name_edit_sim_decile": match_rate_table(X["name_edit_sim"], y),
            "within_name_translit_edit_sim_bins": within,
            "models": {k: {**runs[k]["result"], "features_removed": [f for f in names if f not in fs]}
                       for k, fs in cfgs.items()},
        },
    }


# ---------------------------------------------------------------- Steps 10-11
def threshold_row(scored: pd.DataFrame, n_true: pd.Series, thr: float) -> dict:
    r = tt.evaluate(scored, n_true, thr)
    n_pred, tp, t = per_entity(scored, n_true, thr)
    zero, single = t == 0, t == 1
    return {**{k: r[k] for k in ("threshold", "macro_f05", "precision", "recall", "tp", "fp", "fn",
                                 "entities_with_prediction", "average_predictions_per_entity")},
            "predictions": int(n_pred.sum()),
            "zero_match_predicted_empty": int((zero & (n_pred == 0)).sum()),
            "zero_match_false_merges": int((zero & (n_pred > 0)).sum()),
            "singleton_exact_correct": int((single & (n_pred == 1) & (tp == 1)).sum()),
            "singleton_missed": int((single & (tp == 0)).sum()),
            "singleton_extra_predictions": int((single & (tp == 1) & (n_pred > 1)).sum())}


def stress_test(scored: pd.DataFrame, n_true: pd.Series, thr: float) -> dict:
    n_pred, tp, t = per_entity(scored, n_true, thr)
    # no forcing rule: an entity is predicted non-empty iff one of its candidates scores >= thr
    has_hit = (scored.groupby("s1")["score"].max() >= thr).reindex(n_true.index, fill_value=False).to_numpy()
    assert np.array_equal(n_pred > 0, has_hit), "a rule forced predictions beyond score >= threshold"
    zero, single, multi = t == 0, t == 1, t > 1
    return {"threshold": thr, "true_zero_match": int(zero.sum()), "true_singleton": int(single.sum()),
            "true_multi_match": int(multi.sum()),
            "zero_match_correctly_empty": int((zero & (n_pred == 0)).sum()),
            "zero_match_false_merges": int((zero & (n_pred > 0)).sum()),
            "singleton_correct_match_rate": float((single & (n_pred == 1) & (tp == 1)).sum() / single.sum()) if single.any() else None,
            "singleton_misses": int((single & (tp == 0)).sum()),
            "multi_match": {"mean_true": float(t[multi].mean()) if multi.any() else 0.0,
                            "mean_predicted": float(n_pred[multi].mean()) if multi.any() else 0.0,
                            "predicted_at_least_two": int((n_pred[multi] >= 2).sum())},
            "average_predictions_per_s1": float(n_pred.mean()) if len(n_pred) else 0.0,
            "entities_predicted_empty": int((n_pred == 0).sum()),
            "forced_prediction_rule": False}


def robustness(run: dict, n_true: pd.Series) -> dict:
    grid = sorted(set(GRID) | set(FINE_GRID))
    rows = [threshold_row(run["scored"], n_true, t) for t in grid]
    thr = run["result"]["selected_threshold"]
    near = [r["macro_f05"] for r in rows if abs(r["threshold"] - thr) <= 0.1 + 1e-9]
    return {"selected_threshold_standard_protocol": thr,
            "best_on_combined_grid": max(rows, key=lambda r: (round(r["macro_f05"], 12), r["threshold"]))["threshold"],
            "macro_f05_range_within_0.1_of_selected": [min(near), max(near)],
            "thresholds": rows, "stress_test": stress_test(run["scored"], n_true, thr)}


# ---------------------------------------------------------------- driver
def merge_json(path: Path, scale: str, payload: dict) -> None:
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[scale] = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                    encoding="utf-8")


def reproduction_check(scale: str, r: dict) -> dict:
    observed = {"macro_f05": round(r["macro_f05"], 4), "selected_threshold": r["selected_threshold"], "fp": r["fp"],
                "fn": r["fn"], "classifier_missed_positives": r["classifier_missed_positives"],
                "blocking_missed_positives": r["blocking_missed_positives"]}
    if scale != "dev":
        return {"expected": None, "observed": observed, "note": "Phase 2 reference numbers exist only for dev"}
    diff = {k: {"expected": v, "observed": observed[k]} for k, v in DEV_EXPECTED.items() if observed[k] != v}
    return {"expected": DEV_EXPECTED, "observed": observed, "matched": not diff, "discrepancies": diff}


def select_feature_set(boot: dict, sets: dict) -> tuple:
    """Spec 23: a set significantly better than B0 wins (highest F0.5); else the smallest set that is
    effectively equivalent to B0 (ties -> higher F0.5)."""
    c = {k: v for k, v in boot["configurations"].items() if k in sets}
    better = [k for k, v in c.items() if v["verdict"] == "better"]
    if better:
        return max(better, key=lambda k: c[k]["macro_f05"]), "significantly better than B0 (paired CI > 0)"
    eq = [k for k, v in c.items() if v["verdict"] == "effectively equivalent"]
    return min(eq, key=lambda k: (len(sets[k][1]), -c[k]["macro_f05"])), "fewest features among sets equivalent to B0"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candset", required=True, type=Path)
    ap.add_argument("--scale", required=True)
    ap.add_argument("--out", type=Path, default=REPO / "phase2" / "artifacts" / "phase2_5")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t_start = time.perf_counter()
    cs = load_candset(args.candset)
    n_true = cs["n_true_val"]
    idx = bootstrap_index(len(n_true))
    protocol = {"training_rows": "validation == 0", "fit_rows": int(len(cs["fit_idx"])),
                "sampling": "matching_model.sample_training_pairs(negative_ratio=5, hard_fraction=0.5, seed=42), "
                            "hardness on name_char3_jaccard + address_char3_jaccard for every experiment",
                "M0": "StandardScaler + LogisticRegression(C=1.0, max_iter=2000, random_state=42)",
                "M1": "HistGradientBoostingClassifier(random_state=42, early_stopping=False, max_iter=200)",
                "threshold_grid": GRID, "selection": "max validation macro F0.5 per S1 entity (ties -> higher threshold)",
                "validation_entities": int(len(n_true)), "validation_pairs": int(len(cs["val_idx"])),
                "candset": str(args.candset), "seed": SEED}
    header = {"scale": args.scale, "protocol": protocol, "candidate_set": candidate_set_summary(cs)}

    # Step 7
    sets = feature_sets(cs["names"])
    runs, f_by_cfg = {}, {}
    for k, (_, fs) in sets.items():
        log.info("M0 on %s (%d features)", k, len(fs))
        runs[k] = run_config(cs, fs)
        f_by_cfg[k] = runs[k]["f"]
    # Step 8 (adds its configurations to runs/f_by_cfg)
    diag = edit_sim_diagnosis(cs, runs, f_by_cfg)
    boot = bootstrap(f_by_cfg, "B0", idx)
    best, why = select_feature_set(boot, sets)
    merge_json(args.out / "feature_ablation.json", args.scale, {
        **header, "b0_reproduction": reproduction_check(args.scale, runs["B0"]["result"]),
        "feature_sets": {k: {"description": d, "n_features": len(fs), "features": fs} for k, (d, fs) in sets.items()},
        "results": {k: runs[k]["result"] for k in sets}, "bootstrap": boot,
        "selected_feature_set": best, "selection_reason": why, "name_edit_sim_diagnosis": diag})

    # Step 9
    mruns = {"M0/B0": runs["B0"]}
    log.info("M1 on B0")
    mruns["M1/B0"] = run_config(cs, sets["B0"][1], "M1")
    if best != "B0":
        log.info("M1 on %s", best)
        mruns[f"M1/{best}"] = run_config(cs, sets[best][1], "M1")
    mboot = bootstrap({k: v["f"] for k, v in mruns.items()}, "M0/B0", idx)
    m1_better = mboot["configurations"]["M1/B0"]["verdict"] == "better"
    merge_json(args.out / "model_comparison.json", args.scale, {
        **header, "results": {k: v["result"] for k, v in mruns.items()}, "bootstrap": mboot,
        "best_feature_set_from_ablation": best, "m1_better_than_m0": m1_better,
        "hyperparameter_search": "none"})

    # Steps 10-11
    rob = {"M0/B0": robustness(mruns["M0/B0"], n_true)}
    if m1_better:
        rob["M1/B0"] = robustness(mruns["M1/B0"], n_true)
    merge_json(args.out / "threshold_robustness.json", args.scale, {
        **header, "grid": sorted(set(GRID) | set(FINE_GRID)), "configurations": rob,
        "never_force_prediction": "empty prediction is valid; predictions are exactly score >= threshold"})
    log.info("done in %.0fs; B0 macro F0.5 %.4f @ %.2f", time.perf_counter() - t_start,
             runs["B0"]["result"]["macro_f05"], runs["B0"]["result"]["selected_threshold"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
