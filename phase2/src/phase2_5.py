"""Phase 2.5 hardening harness: blocking experiments and frozen candidate sets.

    python phase2/src/phase2_5.py blocking --scale dev|10k

One pass per target source at the given scale:
  Step 2  blocking-rule ablation (individual, cumulative, exact Shapley unique contribution)
  Step 3  candidate-cap sensitivity          Step 4  high-frequency-key cutoff sensitivity
  Step 5  transliteration blocking           Step 6  address-first blocking
  Step 1  every validation positive missed by the B0 candidate set, with per-rule reasons
and freezes the B0 candidate set (features, labels, fixed entity split) under
output/phase2_5/candsets/<scale>/b0/ for the feature and model experiments.

It reuses the Phase 2 modules unchanged; results merge into artifacts/phase2_5/*.json by scale.
"""
import argparse
import itertools
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import candidate_generation as cg
import feature_engineering as fe
import matching_model as mm
import run_phase2 as rp

log = logging.getLogger("phase2_5")
REPO = Path(__file__).resolve().parents[2]
ART = REPO / "phase2" / "artifacts" / "phase2_5"
OUT = REPO / "phase2" / "output" / "phase2_5"
BASE_RULES = ["exact_name", "exact_address", "numeric", "name_prefix", "address_prefix", "name_ngram",
              "name_phonetic", "address_token"]
TRANSLIT_RULES = ["name_phonetic_token"]
ADDRESS_RULES = ["house_locality", "postal_locality", "state_locality", "address_signature"]
SCALES = {"dev": (5000, 100000), "10k": (10000, None), "25k": (25000, None), "50k": (50000, None)}
CUTOFFS = [50, 100, 150, 250, 500]
CAPS = [30, 60, 80, 100, 150, 200]
NO_CAP = 10 ** 9


# ---------------------------------------------------------------- helpers
def merge_json(name: str, scale: str, payload: dict) -> None:
    """artifacts/phase2_5/<name>: one section per scale, so dev and full-target results sit side by side."""
    path = ART / name
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[scale] = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=rp._default), encoding="utf-8")


def scale_config(scale: str) -> dict:
    cfg = rp.load_config(REPO / "phase2" / "config" / "phase2_config.yaml", "dev")
    cfg["development"]["max_source1_rows"], cfg["development"]["max_target_rows"] = SCALES[scale]
    return cfg


def per_s1(pairs: pd.DataFrame, n_s1: int) -> dict:
    sizes = np.bincount(pairs["s1_idx"].to_numpy(), minlength=n_s1) if len(pairs) else np.zeros(n_s1, int)
    return {"pairs": int(sizes.sum()), "mean_per_s1": round(float(sizes.mean()), 3),
            "p95_per_s1": float(np.percentile(sizes, 95)), "max_per_s1": int(sizes.max())}


def found(pairs: pd.DataFrame, positives: pd.DataFrame) -> np.ndarray:
    """Boolean per positive: is it among ``pairs``?"""
    m = positives.merge(pairs[["s1_idx", "t_idx"]].assign(_f=1), how="left", on=["s1_idx", "t_idx"])
    return m["_f"].notna().to_numpy()


def precap(blocking: dict, s1, tgt, cgc) -> tuple:
    """Union of rule candidates without the per-S1 cap, and the seconds it took."""
    t0 = time.perf_counter()
    parts = [p for _, _, p, _, _ in cg.iter_candidates(blocking, s1, tgt, {**cgc, "max_candidates_per_source1": NO_CAP})]
    return pd.concat(parts, ignore_index=True), round(time.perf_counter() - t0, 2)


def bucket_stats(table: dict, cutoff) -> dict:
    freq = table["freq"]
    kept = freq[freq <= cutoff] if cutoff is not None else freq
    return {"target_keys": int(len(freq)), "skipped_keys": int(len(freq) - len(kept)),
            "largest_kept_bucket": int(kept.max()) if len(kept) else 0,
            "p99_kept_bucket": float(kept.quantile(0.99)) if len(kept) else 0.0,
            "largest_bucket_overall": int(freq.max()) if len(freq) else 0}


# ---------------------------------------------------------------- experiments per source
def cutoff_and_cap(tables, s1, tgt, positives, cgc) -> tuple:
    """Steps 3 and 4. Returns (cutoff results, cap results at every cutoff, B0 precap at cutoff 100)."""
    base = {r: tables[r] for r in BASE_RULES}
    n_pos = len(positives)
    cut_res, cap_res, b0_pre, b0_lost = {}, {}, None, None
    for c in CUTOFFS:
        blk = cg.apply_cutoff(base, c)
        pre, secs = precap(blk, s1, tgt, cgc)
        f_pre = found(pre, positives)
        missing = positives[~f_pre]
        lost_all = cg.lost_to_skipped_keys(blk, missing)
        lost = lost_all["any_rule"]
        cut_res[str(c)] = {"precap": per_s1(pre, len(s1)), "precap_recall": round(f_pre.mean(), 5) if n_pos else None,
                           "positives_recovered_precap": int(f_pre.sum()), "blocking_seconds": secs,
                           "positives_lost_sharing_skipped_key": lost,
                           "buckets": {r: bucket_stats(base[r], c) for r in BASE_RULES}}
        caps = {}
        for cap in CAPS:
            t0 = time.perf_counter()
            kept, dropped = cg._cap(pre, s1, tgt, cap)
            secs_cap = round(time.perf_counter() - t0, 2)
            f_kept = found(kept, positives)
            caps[str(cap)] = {"recall": round(f_kept.mean(), 5) if n_pos else None, "after_cap": per_s1(kept, len(s1)),
                              "pairs_before_cap": int(len(pre)), "pairs_dropped": int(len(dropped)),
                              "positives_lost_to_cap": int(f_pre.sum() - f_kept.sum()),
                              "recall_lost_to_cap": round((f_pre.sum() - f_kept.sum()) / n_pos, 5) if n_pos else None,
                              "cap_seconds": secs_cap}
        cap_res[str(c)] = caps
        if c == 100:
            b0_pre, b0_lost = pre, lost_all
        log.info("  cutoff %s: precap recall %.4f, %d pairs, %.1fs", c, f_pre.mean(), len(pre), secs)
    # no cutoff: diagnostic only, never materialized. Recall = positives sharing any key of any rule;
    # the pair count is an upper bound (sum over keys of S1 records x target records, before de-duplication).
    shared, bound = np.zeros(n_pos, bool), 0
    for r in BASE_RULES:
        tb = tables[r]
        m = positives.reset_index().merge(tb["s1"], on="s1_idx").merge(tb["t"], on=["t_idx", "code"])
        shared[m["index"].unique()] = True
        n_s1 = tb["s1"]["code"].value_counts()
        bound += int((n_s1 * tb["freq"].reindex(n_s1.index, fill_value=0)).sum())
    cut_res["no_cutoff_diagnostic"] = {"recall_upper_bound": round(shared.mean(), 5) if n_pos else None,
                                       "pair_count_upper_bound": bound,
                                       "largest_bucket": max(int(tables[r]["freq"].max()) for r in BASE_RULES),
                                       "note": "not materialized; never deployable"}
    return cut_res, cap_res, b0_pre, b0_lost


def ablation(pre: pd.DataFrame, positives: pd.DataFrame, tables: dict, n_s1: int, lost: dict) -> dict:
    """Step 2 on the B0 precap set (cutoff 100): each rule alone, cumulative in the listed order,
    unique-only positives, and exact Shapley values (average marginal gain over all rule orders)."""
    lab = positives.merge(pre, how="left", on=["s1_idx", "t_idx"])
    pos_bits = lab["blocks"].fillna(0).astype(np.int64).to_numpy()
    pair_bits = pre["blocks"].to_numpy().astype(np.int64)
    bits = [fe.RULE_BITS[r] for r in BASE_RULES]
    n_pos = len(positives)
    rules = {}
    for r, b in zip(BASE_RULES, bits):
        sel = pre[(pair_bits & b) > 0]
        rules[r] = {**per_s1(sel, n_s1), "true_positives": int(((pos_bits & b) > 0).sum()),
                    "recall": round(((pos_bits & b) > 0).mean(), 5),
                    "unique_only_positives": int((pos_bits == b).sum()),
                    "unique_only_pairs": int((pair_bits == b).sum()),
                    "true_positives_lost_to_skipped_keys": lost[r]["pairs"], **bucket_stats(tables[r], 100)}
    cum, mask = [], 0
    for r, b in zip(BASE_RULES, bits):
        mask |= b
        cum.append({"rules_added_so_far": BASE_RULES[:BASE_RULES.index(r) + 1], "recall": round(((pos_bits & mask) > 0).mean(), 5),
                    "pairs": int(((pair_bits & mask) > 0).sum())})
    # exact Shapley over the 8 rules: value(S) = recall / pairs of the union of rules in S
    k = len(bits)
    value_r, value_p = {}, {}
    for size in range(k + 1):
        for subset in itertools.combinations(range(k), size):
            m = sum(bits[i] for i in subset)
            value_r[subset] = ((pos_bits & m) > 0).sum() / n_pos if n_pos else 0.0
            value_p[subset] = int(((pair_bits & m) > 0).sum())
    shap_r, shap_p = np.zeros(k), np.zeros(k)
    for subset in value_r:
        for i in range(k):
            if i in subset:
                continue
            with_i = tuple(sorted(subset + (i,)))
            w = math.factorial(len(subset)) * math.factorial(k - len(subset) - 1) / math.factorial(k)
            shap_r[i] += w * (value_r[with_i] - value_r[subset])
            shap_p[i] += w * (value_p[with_i] - value_p[subset])
    for i, r in enumerate(BASE_RULES):
        rules[r]["shapley_recall"] = round(float(shap_r[i]), 5)
        rules[r]["shapley_pairs"] = round(float(shap_p[i]), 1)
    return {"rules": rules, "cumulative_in_listed_order": cum,
            "union_precap_recall": round((pos_bits > 0).mean(), 5), "union_precap_pairs": int(len(pre))}


def experimental(tables, s1, tgt, positives, cgc, b0_pre, b0_kept) -> dict:
    """Steps 5 and 6: each experimental rule alone at cutoff 100 and added to B0 (then capped at 60)."""
    n_pos = len(positives)
    f_b0 = found(b0_kept, positives)
    out = {}
    groups = [[r] for r in TRANSLIT_RULES + ADDRESS_RULES] + [ADDRESS_RULES, TRANSLIT_RULES + ADDRESS_RULES]
    for group in groups:
        name = "+".join(group) if len(group) > 1 else group[0]
        blk = cg.apply_cutoff({r: tables[r] for r in group}, 100)
        new, secs = precap(blk, s1, tgt, cgc)
        f_new = found(new, positives)
        union = pd.concat([b0_pre, new]).groupby(["s1_idx", "t_idx"], as_index=False)["blocks"].agg(
            lambda b: np.bitwise_or.reduce(b.to_numpy()))
        kept, _ = cg._cap(union, s1, tgt, cgc["max_candidates_per_source1"])
        f_kept = found(kept, positives)
        added = union.merge(b0_pre[["s1_idx", "t_idx"]].assign(_b=1), how="left", on=["s1_idx", "t_idx"])
        out[name] = {
            "rule_alone": {**per_s1(new, len(s1)), "recall": round(f_new.mean(), 5) if n_pos else None,
                           "blocking_seconds": secs},
            "buckets": {r: bucket_stats(tables[r], 100) for r in group},
            "new_pairs_vs_b0_precap": int(added["_b"].isna().sum()),
            "b0_missed_positives_retrieved": int((f_new & ~f_b0).sum()),
            "with_b0_after_cap": {"recall": round(f_kept.mean(), 5) if n_pos else None, **per_s1(kept, len(s1))},
            "b0_after_cap_recall": round(f_b0.mean(), 5) if n_pos else None,
        }
    return out


def rule_reasons(tables: dict, rules: list, pairs: pd.DataFrame, cutoff: int) -> list:
    """For each (s1_idx, t_idx): why each rule did or did not retrieve it."""
    out = [dict() for _ in range(len(pairs))]
    idx = pairs.reset_index(drop=True)
    for r in rules:
        tb = tables[r]
        s1k = idx.reset_index().merge(tb["s1"], on="s1_idx")
        tk = idx.reset_index().merge(tb["t"], on="t_idx")
        shared = s1k.merge(tk[["index", "code"]], on=["index", "code"])
        has_s1, has_t = set(s1k["index"]), set(tk["index"])
        best = shared.assign(f=tb["freq"].reindex(shared["code"]).to_numpy()).groupby("index")["f"].min()
        for i in range(len(idx)):
            if i in best.index:
                f = int(best[i])
                out[i][r] = "retrieved" if f <= cutoff else f"shared key skipped (bucket {f} > {cutoff})"
            elif i not in has_s1 and i not in has_t:
                out[i][r] = "no key on either side"
            elif i not in has_s1:
                out[i][r] = "no key on S1 side"
            elif i not in has_t:
                out[i][r] = "no key on target side"
            else:
                out[i][r] = "keys differ"
    return out


CATEGORY_ORDER = ["Per-S1 candidate cap", "High-frequency-key suppression", "Transliteration spelling variation",
                  "Cross-script name", "Token reordering", "Legal suffix/business-type variation",
                  "Weak/missing name but useful address", "Address corruption with strong name",
                  "Address truncation/component omission", "Numeric/address variation", "Severe name corruption",
                  "Unknown", "No existing block overlap"]


def categorize(f: dict, reasons: dict, in_precap: bool, rec_a, rec_b) -> list:
    """All applicable categories, in CATEGORY_ORDER (the first is the primary one).

    Mechanism categories (cap, skipped key) and content categories are evidence-based; when none
    applies the miss is "Unknown". "No existing block overlap" is a factual tag added whenever no
    rule shares any key for the pair; it is primary only if nothing else describes the miss."""
    cats = []
    if in_precap:
        cats.append("Per-S1 candidate cap")
    if any(v.startswith("shared key skipped") for v in reasons.values()):
        cats.append("High-frequency-key suppression")
    if f["name_script_mismatch"]:
        cats.append("Cross-script name")
        if f["name_phonetic_edit_sim"] >= 0.6:
            cats.append("Transliteration spelling variation")
    elif f["name_phonetic_edit_sim"] >= 0.8 and f["name_translit_char3_jaccard"] < 0.6:
        cats.append("Transliteration spelling variation")  # same script, spelling differs, sound agrees
    if rec_a["name_sorted"] == rec_b["name_sorted"] and rec_a["name_key"] != rec_b["name_key"]:
        cats.append("Token reordering")
    if rec_a["name_core"] == rec_b["name_core"] and rec_a["name_norm"] != rec_b["name_norm"]:
        cats.append("Legal suffix/business-type variation")
    name_strong = f["name_translit_char3_jaccard"] >= 0.6 or f["name_phonetic_eq"]
    addr_strong = f["address_token_jaccard"] >= 0.5
    if not name_strong and addr_strong:
        cats.append("Weak/missing name but useful address")
    if name_strong and not addr_strong and not f["address_missing_either"]:
        cats.append("Address corruption with strong name")
    ta, tb_ = len(rec_a["addr_norm"].split()), len(rec_b["addr_norm"].split())
    if f["address_missing_either"] or (max(ta, tb_) and min(ta, tb_) / max(ta, tb_) < 0.5):
        cats.append("Address truncation/component omission")
    if f["house_number_both_present"] and not f["house_number_equal"]:
        cats.append("Numeric/address variation")
    if not f["name_script_mismatch"] and f["name_translit_char3_jaccard"] < 0.3:
        cats.append("Severe name corruption")
    if not cats:
        cats.append("Unknown")
    if all(v != "retrieved" and not v.startswith("shared") for v in reasons.values()):
        cats.append("No existing block overlap")
    return [c for c in CATEGORY_ORDER if c in cats]


def miss_analysis(tables, s1, tgt, src, positives, val_mask_s1, b0_pre, b0_kept, raw_s1, raw_t, nj) -> list:
    """Step 1: every validation positive absent from the B0 candidate set."""
    pos_val = positives[val_mask_s1[positives["s1_idx"].to_numpy()]].reset_index(drop=True)
    missed = pos_val[~found(b0_kept, pos_val)].reset_index(drop=True)
    if missed.empty:
        return []
    in_pre = found(b0_pre, missed)
    feats = fe.compute_features(missed.assign(blocks=0), s1, tgt, src, nj)
    reasons = rule_reasons(tables, BASE_RULES, missed, 100)
    proposed = rule_reasons(tables, TRANSLIT_RULES + ADDRESS_RULES, missed, 100)
    rows = []
    cols = ["name_norm", "name_translit", "name_core", "name_key", "name_sorted", "name_phonetic", "addr_norm",
            "house", "postal", "state", "country"]
    for i, (a, b) in enumerate(zip(missed["s1_idx"], missed["t_idx"])):
        f = feats.iloc[i].to_dict()
        ra, rb = s1.iloc[a], tgt.iloc[b]
        cats = categorize(f, reasons[i], bool(in_pre[i]), ra, rb)
        ea, eb = s1["entity_id"].iat[a], tgt["entity_id"].iat[b]
        rows.append({
            "s1": ea, "target": eb, "source": src, "country": ra["country"],
            "raw": {"s1_name": raw_s1.get(ea, ("", ""))[0], "target_name": raw_t.get(eb, ("", ""))[0],
                    "s1_address": raw_s1.get(ea, ("", ""))[1], "target_address": raw_t.get(eb, ("", ""))[1]},
            "normalized": {c: [ra[c], rb[c]] for c in cols},
            "script_mismatch": bool(f["name_script_mismatch"]),
            "similarity": {k: round(float(f[k]), 4) for k in (
                "name_translit_char3_jaccard", "name_translit_edit_sim", "name_phonetic_edit_sim", "name_edit_sim",
                "name_token_jaccard", "name_core_token_jaccard", "address_token_jaccard", "address_edit_sim",
                "address_char3_jaccard", "address_component_jaccard")},
            "structured_evidence": {k: bool(f[k]) for k in (
                "house_number_both_present", "house_number_equal", "postal_both_present", "postal_equal",
                "state_both_present", "state_equal", "address_missing_either")},
            "in_b0_precap_but_capped": bool(in_pre[i]),
            "b0_rule_outcomes": reasons[i],
            "proposed_rule_outcomes": proposed[i],
            "retrieved_by_proposed_rule": [r for r, v in proposed[i].items() if v == "retrieved"],
            "categories": cats, "primary_category": cats[0],
        })
    return rows


def freeze_candset(scale, src, s1, tgt, kept, positives, val_ids, nj) -> tuple:
    """Features + labels of the B0 candidate set (cap 60, cutoff 100) for the model experiments."""
    feats = fe.compute_features(kept, s1, tgt, src, nj)
    meta = pd.DataFrame({"s1": s1["entity_id"].to_numpy()[kept["s1_idx"].to_numpy()],
                         "target": tgt["entity_id"].to_numpy()[kept["t_idx"].to_numpy()],
                         "source": src, "label": mm.label_pairs(kept, positives), "blocks": kept["blocks"].to_numpy()})
    meta["validation"] = meta["s1"].isin(val_ids).astype(np.int8)
    return feats, meta


# ---------------------------------------------------------------- driver
def blocking_run(scale: str) -> None:
    cfg = scale_config(scale)
    seed, nj, data_root = cfg["seed"], cfg["n_jobs"], REPO / cfg["paths"]["data_root"]
    cgc_all = {**cfg["candidate_generation"], **{cg.RULE_FLAGS[r]: True for r in TRANSLIT_RULES + ADDRESS_RULES}}
    t_start = time.perf_counter()
    s1_raw = rp.load_source1(data_root, "train", cfg)
    gt = rp.load_tsv(data_root / "train" / "train_ground_truth.tsv")
    pairs = rp.gt_pairs(gt[gt["source1_entity_id"].isin(s1_raw["entity_id"])])
    del gt
    s1 = fe.prepare_records(s1_raw, nj, keep_raw=False)
    raw_s1 = dict(zip(s1_raw["entity_id"], zip(s1_raw["business_name"], s1_raw["business_address"])))
    ids = s1["entity_id"]
    val_ids = mm.split_entities(ids.tolist(), cfg["validation"]["validation_fraction"], seed)
    val_mask = ids.isin(val_ids).to_numpy()
    n_true = pairs.groupby("s1").size().reindex(ids, fill_value=0)
    results = {k: {} for k in ("ablation", "cutoff", "cap", "experimental", "miss")}
    cand_dir = OUT / "candsets" / scale / "b0"
    cand_dir.mkdir(parents=True, exist_ok=True)
    feats_all, meta_all, recall_by_src = [], [], {}
    for src in rp.SOURCES:
        log.info("[%s] %s: load + normalize", scale, src)
        tgt_raw = rp.load_target(data_root, "train", src, cfg, set(pairs["target"]))
        tgt = fe.prepare_records(tgt_raw, nj, keep_raw=False)
        del tgt_raw
        positives = rp.positional(pairs[pairs["target"].str.startswith(src.upper() + "-")], s1, tgt)
        t0 = time.perf_counter()
        tables = cg.rule_tables(s1, tgt, cgc_all, BASE_RULES + TRANSLIT_RULES + ADDRESS_RULES)
        key_seconds = round(time.perf_counter() - t0, 1)
        cut, cap, b0_pre, b0_lost = cutoff_and_cap(tables, s1, tgt, positives, cfg["candidate_generation"])
        b0_kept, _ = cg._cap(b0_pre, s1, tgt, cfg["candidate_generation"]["max_candidates_per_source1"])
        results["cutoff"][src], results["cap"][src] = cut, cap
        results["ablation"][src] = {**ablation(b0_pre, positives, tables, len(s1), b0_lost),
                                    "key_build_seconds_all_rules": key_seconds}
        results["experimental"][src] = experimental(tables, s1, tgt, positives, cfg["candidate_generation"], b0_pre, b0_kept)
        example_t = set(tgt["entity_id"].to_numpy()[positives["t_idx"].to_numpy()])
        raw_t = rp.raw_lookup(data_root / "train" / f"train_source{src[1]}.tsv", example_t)
        results["miss"][src] = miss_analysis(tables, s1, tgt, src, positives, val_mask, b0_pre, b0_kept, raw_s1, raw_t, nj)
        f_b0 = found(b0_kept, positives)
        recall_by_src[src] = {"positives": int(len(positives)), "b0_recall": round(f_b0.mean(), 5),
                              "validation_positives": int(val_mask[positives["s1_idx"].to_numpy()].sum()),
                              "validation_b0_recall": round(f_b0[val_mask[positives["s1_idx"].to_numpy()]].mean(), 5)}
        feats, meta = freeze_candset(scale, src, s1, tgt, b0_kept, positives, val_ids, nj)
        feats_all.append(feats)
        meta_all.append(meta)
        del tables, tgt, b0_pre, b0_kept
    feats = pd.concat(feats_all, ignore_index=True)
    meta = pd.concat(meta_all, ignore_index=True)
    np.save(cand_dir / "features.npy", feats[fe.FEATURES].to_numpy(np.float32))
    meta.to_csv(cand_dir / "pairs.tsv", sep="\t", index=False)
    per_src = {s: pairs[pairs["target"].str.startswith(s.upper() + "-")].groupby("s1").size().reindex(ids, fill_value=0)
               for s in rp.SOURCES}
    pd.DataFrame({"s1": ids, "n_true": n_true.to_numpy(), "n_true_s2": per_src["s2"].to_numpy(),
                  "n_true_s3": per_src["s3"].to_numpy(), "validation": val_mask.astype(np.int8)}).to_csv(
        cand_dir / "entities.tsv", sep="\t", index=False)
    info = {"scale": scale, "source1_entities": len(ids), "validation_entities": len(val_ids),
            "target_rows_per_source": SCALES[scale][1] or "full", "features": fe.FEATURES,
            "candidate_pairs": int(len(meta)), "validation_candidate_pairs": int(meta["validation"].sum()),
            "positives_in_candidates": int(meta["label"].sum()), "recall": recall_by_src,
            "blocking": {"cutoff": 100, "cap": cfg["candidate_generation"]["max_candidates_per_source1"],
                         "rules": BASE_RULES}, "seed": seed}
    (cand_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    common = {"scale": scale, "source1_entities": len(ids), "validation_entities": len(val_ids),
              "target_rows_per_source": SCALES[scale][1] or "full", "b0_recall": recall_by_src}
    merge_json("blocking_ablation.json", scale, {**common, "by_source": results["ablation"],
                                                  "address_blocking": {s: {k: v for k, v in results["experimental"][s].items()
                                                                           if k in ADDRESS_RULES or "+" in k}
                                                                       for s in rp.SOURCES}})
    merge_json("key_cutoff_sensitivity.json", scale, {**common, "cap_during_test": "none (pre-cap)",
                                                       "by_source": results["cutoff"]})
    merge_json("cap_sensitivity.json", scale, {**common, "by_source_by_cutoff": results["cap"]})
    merge_json("transliteration_blocking.json", scale, {**common, "by_source": {
        s: {k: v for k, v in results["experimental"][s].items() if k in TRANSLIT_RULES or "+" in k} for s in rp.SOURCES}})
    misses = results["miss"]["s2"] + results["miss"]["s3"]
    counts = pd.Series([m["primary_category"] for m in misses]).value_counts().to_dict() if misses else {}
    all_cats = pd.Series([c for m in misses for c in m["categories"]]).value_counts().to_dict() if misses else {}
    merge_json("blocking_miss_analysis.json", scale, {
        **common, "validation_positives_missed": len(misses), "primary_category_counts": counts,
        "any_category_counts": all_cats, "category_order": CATEGORY_ORDER,
        "recoverable_by_proposed_rule": pd.Series([r for m in misses for r in m["retrieved_by_proposed_rule"]]
                                                  ).value_counts().to_dict() if misses else {},
        "misses": misses})
    log.info("[%s] done in %.0fs", scale, time.perf_counter() - t_start)


SPEC_B0 = {"s2_candidate_recall": 0.98103, "s3_candidate_recall": 0.98156, "combined_candidate_recall": 0.9813,
           "train_entities": 4000, "validation_entities": 1000, "train_candidate_pairs": 319829,
           "train_pairs_used": 81972, "selected_threshold": 0.7, "macro_f05": 0.9709, "false_positives": 44,
           "false_negatives_below_threshold": 111, "blocking_missed_validation_positives": 72,
           "zero_match_validation_entities": 54, "zero_match_predicted_empty": 49}


def baseline_report() -> None:
    """Step 0: compare the original B0 artifacts with the reproductions (before and after the
    behavior-preserving blocking refactor) and record runtime and memory."""
    def load(d, n):
        return json.loads((d / n).read_text(encoding="utf-8"))

    def metrics(d):
        th, tr, cs, ea = (load(d, n) for n in ("threshold_results.json", "training_summary.json",
                                               "candidate_statistics.json", "error_analysis.json"))
        return {"s2_candidate_recall": cs["s2"]["candidate_recall"], "s3_candidate_recall": cs["s3"]["candidate_recall"],
                "combined_candidate_recall": cs["combined"]["candidate_recall"],
                "train_entities": tr["train_entities"], "validation_entities": tr["validation_entities"],
                "train_candidate_pairs": tr["train_candidate_pairs"], "train_pairs_used": tr["train_pairs_used"],
                "selected_threshold": th["selected_threshold"], "macro_f05": round(th["selected"]["macro_f05"], 4),
                "false_positives": ea["false_positives"]["count"],
                "false_negatives_below_threshold": ea["false_negatives_scored_below_threshold"]["count"],
                "blocking_missed_validation_positives": ea["false_negatives_never_retrieved"]["count"],
                "zero_match_validation_entities": ea["zero_match_entities"]["count"],
                "zero_match_predicted_empty": ea["zero_match_entities"]["predicted_empty"],
                "train_validation_entity_overlap": tr["entity_overlap_train_validation"]}

    b0_dir = REPO / "phase2" / "artifacts"
    out = {"spec_reported_b0": SPEC_B0, "original_b0_artifacts": metrics(b0_dir)}
    for name in ("b0_reproduction", "b0_reproduction_after_refactor", "b0_reproduction_final_code"):
        d = OUT / name / "artifacts"
        same = {}
        for n in ("threshold_results.json", "training_summary.json", "candidate_statistics.json",
                  "feature_statistics.json", "error_analysis.json"):
            a, b = load(b0_dir, n), load(d, n)
            a.pop("test_inference", None), b.pop("test_inference", None)
            same[n] = a == b
        inf_a, inf_b = load(b0_dir, "inference_summary.json"), load(d, "inference_summary.json")
        run = json.loads((OUT / "runs" / f"{name}.summary.json").read_text(encoding="utf-8-sig"))
        out[name] = {
            "metrics": metrics(d), "artifacts_identical_to_b0": same,
            "output_sha256_identical": inf_a["sha256"] == inf_b["sha256"], "output_sha256": inf_b["sha256"],
            "model_file_identical": (b0_dir / "model" / "matcher.joblib").read_bytes()
                                    == (d / "model" / "matcher.joblib").read_bytes(),
            "validator": inf_b["validator"], "section45_checks": inf_b["section45_checks"],
            "runtime_seconds_by_stage": load(d, "runtime_dev.json")["seconds"],
            "wall_seconds": run["wall_seconds"], "main_process_peak_working_set_mb": run["main_process_peak_working_set_mb"],
            "all_python_processes_peak_mb": run["all_python_processes_peak_mb"],
            "memory_method": "external: Windows PeakWorkingSet64 sampled every 2 s (phase2/tools/memwatch.ps1)",
        }
    rep = out["b0_reproduction"]["metrics"]
    out["matches_spec_b0"] = {k: rep[k] == v for k, v in SPEC_B0.items()}
    out["reproduced"] = all(out["matches_spec_b0"].values()) and all(
        all(out[n]["artifacts_identical_to_b0"].values()) and out[n]["output_sha256_identical"]
        for n in ("b0_reproduction", "b0_reproduction_after_refactor", "b0_reproduction_final_code"))
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "baseline_reproduction.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("baseline reproduced: %s", out["reproduced"])


FROZEN_RULES = BASE_RULES + ["house_locality", "state_locality", "address_signature"]


def capsweep_run(scale: str) -> None:
    """Cap and cutoff sensitivity on the exact frozen rule set (B0 rules + 3 address rules), so the
    frozen cap is chosen on full-target evidence for the combination actually deployed."""
    cfg = scale_config(scale)
    nj, data_root = cfg["n_jobs"], REPO / cfg["paths"]["data_root"]
    cgc = {**cfg["candidate_generation"], **{cg.RULE_FLAGS[r]: True for r in FROZEN_RULES}}
    s1_raw = rp.load_source1(data_root, "train", cfg)
    gt = rp.load_tsv(data_root / "train" / "train_ground_truth.tsv")
    pairs = rp.gt_pairs(gt[gt["source1_entity_id"].isin(s1_raw["entity_id"])])
    del gt
    s1 = fe.prepare_records(s1_raw, nj, keep_raw=False)
    out = {}
    for src in rp.SOURCES:
        tgt = fe.prepare_records(rp.load_target(data_root, "train", src, cfg, set(pairs["target"])), nj, keep_raw=False)
        positives = rp.positional(pairs[pairs["target"].str.startswith(src.upper() + "-")], s1, tgt)
        tables = cg.rule_tables(s1, tgt, cgc, FROZEN_RULES)
        res = {}
        for c in (100, 150):
            pre, secs = precap(cg.apply_cutoff(tables, c), s1, tgt, cgc)
            f_pre = found(pre, positives)
            caps = {}
            for cap in (60, 80, 100, 150):
                t0 = time.perf_counter()
                kept, dropped = cg._cap(pre, s1, tgt, cap)
                f_kept = found(kept, positives)
                caps[str(cap)] = {"recall": round(f_kept.mean(), 5), "after_cap": per_s1(kept, len(s1)),
                                  "pairs_dropped": int(len(dropped)), "positives_lost_to_cap": int(f_pre.sum() - f_kept.sum()),
                                  "cap_seconds": round(time.perf_counter() - t0, 2)}
            res[str(c)] = {"precap_recall": round(f_pre.mean(), 5), "precap": per_s1(pre, len(s1)),
                           "blocking_seconds": secs, "caps": caps}
            log.info("  %s cutoff %d: precap %.4f, cap100 %.4f", src, c, f_pre.mean(), caps["100"]["recall"])
        out[src] = res
        del tables, tgt
    data = json.loads((ART / "cap_sensitivity.json").read_text(encoding="utf-8"))
    data.setdefault("frozen_rule_set", {})[scale] = {"rules": FROZEN_RULES, "source1_entities": len(s1),
                                                     "by_source_by_cutoff": out}
    (ART / "cap_sensitivity.json").write_text(json.dumps(data, indent=2, default=rp._default), encoding="utf-8")


def compare_runs(names: list) -> None:
    """Before/after on identical validation entities: paired bootstrap of macro F0.5 between
    run_phase2 runs under output/phase2_5/<name>/ (the first name is the reference)."""
    import phase2_5_models as pm
    import threshold_tuning as tt
    runs, ents = {}, None
    for name in names:
        art = OUT / name / "artifacts"
        scored = pd.read_csv(art / "validation_scores.tsv", sep="\t", dtype={"s1": str, "target": str},
                             keep_default_na=False)
        n_true = pd.read_csv(art / "validation_entities.tsv", sep="\t", dtype={"s1": str},
                             keep_default_na=False).set_index("s1")["n_true"]
        if ents is None:
            ents = n_true.index
        if not n_true.index.equals(ents):
            raise SystemExit(f"{name}: validation entities differ from {names[0]}; no paired comparison possible")
        th = json.loads((art / "threshold_results.json").read_text(encoding="utf-8"))
        cs = json.loads((art / "candidate_statistics.json").read_text(encoding="utf-8"))
        thr = th["selected_threshold"]
        n_pred, tp, nt = pm.per_entity(scored, n_true, thr)
        sel = th["selected"]
        runs[name] = {"f": tt.entity_f05(n_pred, nt, tp), "summary": {
            "selected_threshold": thr, "macro_f05": sel["macro_f05"], "precision": sel["precision"],
            "recall": sel["recall"], "tp": sel["tp"], "fp": sel["fp"], "fn": sel["fn"],
            "candidate_recall": cs["combined"]["candidate_recall"],
            "candidate_recall_s2": cs["s2"]["candidate_recall"], "candidate_recall_s3": cs["s3"]["candidate_recall"],
            "mean_candidates_per_s1": round(cs["s2"]["mean_candidates_per_s1"] + cs["s3"]["mean_candidates_per_s1"], 3),
            "p95_candidates_per_s1_s2": cs["s2"]["p95_candidates_per_s1"], "p95_candidates_per_s1_s3": cs["s3"]["p95_candidates_per_s1"],
            "blocking_missed_positives": int(nt.sum() - scored["label"].sum()),
            "classifier_missed_positives": int(((scored["label"] == 1) & (scored["score"] < thr)).sum()),
            "validation_entities": int(len(nt)), "sweep": th["sweep"],
            "zero_match_entities": sel["zero_match_entities"], "zero_match_scored_1": sel["zero_match_entities_scored_1"]}}
    boot = pm.bootstrap({k: v["f"] for k, v in runs.items()}, names[0], pm.bootstrap_index(len(ents)))
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "final_comparison.json").write_text(json.dumps(
        {"reference": names[0], "identical_validation_entities": True, "runs": {k: v["summary"] for k, v in runs.items()},
         "bootstrap": boot}, indent=2, default=rp._default), encoding="utf-8")
    for k, v in boot["configurations"].items():
        log.info("%s: macro F0.5 %.4f CI %s diff %s %s", k, v["macro_f05"], v["ci95"], v[f"diff_vs_{names[0]}_ci95"], v["verdict"])


def resource_report() -> None:
    """Step 12: measured runs at increasing Source-1 size, plus a projection to the full run."""
    rows = {}
    for name, n_s1, targets in (("b0_reproduction", 5000, "100k per source"), ("blocking_10k", 10000, "full"),
                                ("resource_b0_25000", 25000, "full"), ("resource_b0_50000", 50000, "full")):
        summ = OUT / "runs" / f"{name}.summary.json"
        if not summ.exists():
            continue
        run = json.loads(summ.read_text(encoding="utf-8-sig"))
        row = {"source1_entities": n_s1, "targets": targets, "exit_code": run["exit_code"],
               "wall_seconds": run["wall_seconds"], "main_process_peak_working_set_mb": run["main_process_peak_working_set_mb"],
               "all_python_processes_peak_mb": run["all_python_processes_peak_mb"]}
        art = OUT / name / "artifacts"
        if (art / f"runtime_dev.json").exists():
            rt = json.loads((art / "runtime_dev.json").read_text(encoding="utf-8"))
            cs = json.loads((art / "candidate_statistics.json").read_text(encoding="utf-8"))
            row.update({"stage_seconds": rt["seconds"], "record_table_mb": rt["memory"], "workers": rt["n_jobs"],
                        "source1_chunk_size": rt["source1_chunk_size"],
                        "candidate_pairs": {s: cs[s]["candidate_pairs"] for s in rp.SOURCES},
                        "precap_candidate_pairs": {s: cs[s]["precap_candidate_pairs"] for s in rp.SOURCES},
                        "candidate_recall": {s: cs[s]["candidate_recall"] for s in rp.SOURCES},
                        "largest_skipped_bucket": {s: max((v["keys"][0][1] for v in cs[s]["skipped_keys"].values() if v["keys"]),
                                                          default=0) for s in rp.SOURCES},
                        "max_candidates_per_s1": {s: cs[s]["max_candidates_per_s1"] for s in rp.SOURCES}})
        rows[name] = row
    out = {"runs": rows, "cartesian_product": "never built: candidates are equi-joins on blocking keys; target keys "
                                              "above max_block_size are skipped; each S1 keeps at most "
                                              "max_candidates_per_source1 candidates per source",
           "memory_method": "external: Windows PeakWorkingSet64 sampled every 2 s (phase2/tools/memwatch.ps1)"}
    for name in ("blocking_lr_50000", "frozen_50000", "capsweep_frozen_10k"):
        summ = OUT / "runs" / f"{name}.summary.json"
        if summ.exists():
            run = json.loads(summ.read_text(encoding="utf-8-sig"))
            rows[name] = {"source1_entities": 10000 if "10k" in name else 50000, "targets": "full",
                          "exit_code": run["exit_code"], "wall_seconds": run["wall_seconds"],
                          "main_process_peak_working_set_mb": run["main_process_peak_working_set_mb"],
                          "all_python_processes_peak_mb": run["all_python_processes_peak_mb"]}
            if (OUT / name / "artifacts" / "runtime_dev.json").exists():
                rows[name]["stage_seconds"] = json.loads((OUT / name / "artifacts" / "runtime_dev.json").read_text())["seconds"]
    frozen = rows.get("frozen_50000")
    sweep = json.loads((ART / "cap_sensitivity.json").read_text(encoding="utf-8")).get("frozen_rule_set", {}).get("10k")
    if frozen and sweep:
        st = frozen["stage_seconds"]
        cs = json.loads((OUT / "frozen_50000" / "artifacts" / "candidate_statistics.json").read_text(encoding="utf-8"))
        n_test, proj = 1732544, {}
        for src in rp.SOURCES:
            sw = sweep["by_source_by_cutoff"][src]["100"]
            join_per_s1 = sw["blocking_seconds"] / 10000          # candidate join/union, measured without index build
            cap_per_s1 = sw["caps"]["100"]["cap_seconds"] / 10000
            pairs_per_s1 = cs[src]["mean_candidates_per_s1"]
            feat_rate = cs[src]["candidate_pairs"] / st[f"train_features_{src}"]   # pairs per second, 50k run
            index = st[f"train_candidates_{src}"] - 50000 * (join_per_s1 + cap_per_s1)
            proj[src] = {"load_normalize_s": round(st[f"train_load_{src}"] + st[f"train_normalize_{src}"], 0),
                         "index_build_s": round(index, 0), "join_and_cap_s": round(n_test * (join_per_s1 + cap_per_s1), 0),
                         "test_candidate_pairs": int(n_test * pairs_per_s1),
                         "features_s": round(n_test * pairs_per_s1 / feat_rate, 0), "feature_pairs_per_second": round(feat_rate)}
            proj[src]["total_s"] = sum(v for k, v in proj[src].items() if k.endswith("_s"))
        out["projection_full_run"] = {
            "method": "stage-based, frozen configuration: target load/normalize and index build are fixed per source "
                      "(measured at 50k); candidate join and cap scale with Source-1 count (measured at 10k on the "
                      "frozen rule set); features scale with candidate pairs (throughput measured at 50k). Model "
                      "scoring, submission merge and validation are extra and not included.",
            "test_inference_by_source": proj,
            "test_inference_total_seconds_estimate": round(sum(p["total_s"] for p in proj.values()), 0),
            "test_candidate_pairs_estimate": sum(p["test_candidate_pairs"] for p in proj.values()),
            "peak_memory_note": "The frozen 50k run peaked at %d MB (main process); the target-side key tables and "
                                "normalized target records dominate and do not grow with Source-1 size. Feature "
                                "buffers grow with the S1 chunk: at 100k S1 per chunk about 6.2M pairs per source "
                                "(~1.2 GB of float32 features plus copies), so a 25k chunk is advised for the full run."
                                % frozen["main_process_peak_working_set_mb"]}
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "full_scale_resource_test.json").write_text(json.dumps(out, indent=2, default=rp._default), encoding="utf-8")
    log.info("resource report written with %d runs", len(rows))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["baseline", "blocking", "capsweep", "compare", "resource"])
    ap.add_argument("--scale", choices=list(SCALES), default="dev")
    ap.add_argument("--runs", nargs="*", default=[])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "baseline":
        baseline_report()
    elif args.command == "compare":
        compare_runs(args.runs)
    elif args.command == "resource":
        resource_report()
    elif args.command == "capsweep":
        capsweep_run(args.scale)
    else:
        blocking_run(args.scale)
    return 0


if __name__ == "__main__":
    sys.exit(main())
