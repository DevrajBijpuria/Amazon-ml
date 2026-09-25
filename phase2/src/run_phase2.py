"""Phase 2 entry point: baseline entity-resolution pipeline.

    python phase2/src/run_phase2.py [--config ...] [--mode dev|full] [--until candidates|model|all]

Order (spec 34): config -> load -> normalize -> index -> candidates -> candidate recall ->
features -> labels -> entity split -> train -> pair metrics -> threshold sweep -> save ->
test inference -> submission files + checks + validator -> report.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import candidate_generation as cg
import feature_engineering as fe
import inference as inf
import matching_model as mm
import threshold_tuning as tt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "phase1" / "src"))
from profiling import load_tsv  # noqa: E402  frozen Phase 1 loader (exact strings, no NA inference)

log = logging.getLogger("phase2")
REPO = Path(__file__).resolve().parents[2]
SOURCES = ("s2", "s3")


# ---------------------------------------------------------------- helpers
def _json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_default), encoding="utf-8")


def _default(o):
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    raise TypeError(type(o))


def _mb(df: pd.DataFrame) -> float:
    return round(df.memory_usage(deep=True).sum() / 1e6, 1)


class Timer(dict):
    def __call__(self, name):
        timer = self

        class _T:
            def __enter__(self):
                self.t = time.perf_counter()
                log.info("[%s] start", name)

            def __exit__(self, *exc):
                timer[name] = round(time.perf_counter() - self.t, 1)
                log.info("[%s] %.1fs", name, timer[name])
        return _T()


def load_config(path: Path, mode: str) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if mode:
        cfg["development"]["enabled"] = mode == "dev"
    cfg["mode"] = "dev" if cfg["development"]["enabled"] else "full"
    return cfg


# ---------------------------------------------------------------- data
def load_source1(data_root: Path, split: str, cfg: dict) -> pd.DataFrame:
    """Source 1 of a split; in dev mode a deterministic sample."""
    df = load_tsv(data_root / split / f"{split}_source1.tsv")
    if cfg["mode"] == "dev":
        df = df.sample(n=min(cfg["development"]["max_source1_rows"], len(df)), random_state=cfg["seed"]).sort_index()
    return df.reset_index(drop=True)  # input file order is kept and becomes the submission row order


def load_target(data_root: Path, split: str, src: str, cfg: dict, needed: set) -> pd.DataFrame:
    """One target source, loaded only when it is processed (memory). Dev mode keeps every id in
    ``needed`` (GT matches of the sampled S1) plus a deterministic random fill; a null
    ``max_target_rows`` keeps the complete source (full-target probe)."""
    df = load_tsv(data_root / split / f"{split}_source{src[1]}.tsv")
    limit = cfg["development"]["max_target_rows"]
    if cfg["mode"] == "full" or limit is None:
        return df
    keep = df["entity_id"].isin(needed)
    fill = df[~keep].sample(n=max(0, min(limit - int(keep.sum()), int((~keep).sum()))), random_state=cfg["seed"])
    return pd.concat([df[keep], fill]).sort_index().reset_index(drop=True)


def raw_lookup(path: Path, ids: set) -> dict:
    """id -> (raw name, raw address) for a few ids, re-read from disk (keeps raw text out of memory)."""
    df = load_tsv(path)
    df = df[df["entity_id"].isin(ids)]
    return dict(zip(df["entity_id"], zip(df["business_name"], df["business_address"])))


def gt_pairs(gt: pd.DataFrame) -> pd.DataFrame:
    """Ground truth as one row per (s1, target) positive pair."""
    pairs = gt.assign(target=gt["matched_entity_ids"].str.split(",")).explode("target")
    pairs["target"] = pairs["target"].fillna("").str.strip()
    pairs = pairs[pairs["target"] != ""][["source1_entity_id", "target"]].drop_duplicates()
    return pairs.rename(columns={"source1_entity_id": "s1"}).reset_index(drop=True)


def positional(pairs: pd.DataFrame, s1: pd.DataFrame, tgt: pd.DataFrame) -> pd.DataFrame:
    """Map (s1 id, target id) to positional (s1_idx, t_idx); pairs outside the tables are dropped."""
    s1_idx = pd.Index(s1["entity_id"]).get_indexer(pairs["s1"])
    t_idx = pd.Index(tgt["entity_id"]).get_indexer(pairs["target"])
    ok = (s1_idx >= 0) & (t_idx >= 0)
    return pd.DataFrame({"s1_idx": s1_idx[ok].astype(np.int32), "t_idx": t_idx[ok].astype(np.int32)})


# ---------------------------------------------------------------- candidate stage
def candidates_for_source(s1, tgt, positives, cg_cfg, keep_s1_mask, n_s1_true):
    """Generate candidates for every S1 in chunks; return (stats, kept candidates of keep_s1_mask S1)."""
    blocking = cg.build_blocking(s1, tgt, cg_cfg)
    pos = positives.assign(_pos=np.int8(1))
    rule_found = {r: 0 for r in blocking}
    rule_pairs = {r: 0 for r in blocking}
    rule_unique = {r: 0 for r in blocking}
    found_pre, found_post, dropped_pairs, dropped_pos, capped_s1 = [], [], 0, 0, 0
    sizes = np.zeros(len(s1), dtype=np.int64)
    precap_total, keep = 0, []
    for lo, hi, precap, kept, dropped in cg.iter_candidates(blocking, s1, tgt, cg_cfg):
        precap_total += len(precap)
        lab = precap.merge(pos, how="left", on=["s1_idx", "t_idx"])
        lab_pos = lab[lab["_pos"] == 1]
        for r in blocking:
            bit = fe.RULE_BITS[r]
            rule_pairs[r] += int(((precap["blocks"] & bit) > 0).sum())
            rule_found[r] += int(((lab_pos["blocks"] & bit) > 0).sum())
            rule_unique[r] += int((lab_pos["blocks"] == bit).sum())
        found_pre.append(lab_pos[["s1_idx", "t_idx"]])
        kept_pos = kept.merge(pos, on=["s1_idx", "t_idx"])
        found_post.append(kept_pos[["s1_idx", "t_idx"]])
        dropped_pairs += len(dropped)
        dropped_pos += int(len(dropped.merge(pos, on=["s1_idx", "t_idx"])))
        capped_s1 += int(dropped["s1_idx"].nunique())
        counts = kept["s1_idx"].value_counts()
        sizes[counts.index.to_numpy()] = counts.to_numpy()
        keep.append(kept[keep_s1_mask[kept["s1_idx"].to_numpy()]])
    found_pre = pd.concat(found_pre, ignore_index=True)
    found_post = pd.concat(found_post, ignore_index=True)
    # skip cost = positives no rule retrieved that share a skipped key (cap drops are counted separately)
    missing = positives.merge(found_pre.assign(_f=1), how="left", on=["s1_idx", "t_idx"])
    missing = missing[missing["_f"].isna()][["s1_idx", "t_idx"]]
    lost_skip = cg.lost_to_skipped_keys(blocking, missing)

    n_pos = len(positives)
    rec = lambda k: round(k / n_pos, 5) if n_pos else None  # noqa: E731
    found_flag = positives.merge(found_post.assign(found=1), how="left", on=["s1_idx", "t_idx"])["found"].fillna(0)
    country = s1["country"].to_numpy()[positives["s1_idx"].to_numpy()]
    bucket = np.where(n_s1_true[positives["s1_idx"].to_numpy()] == 1, "one", "multi")
    by = lambda keys: {str(k): {"positives": int(len(g)), "recovered": int(g.sum()),  # noqa: E731
                                "recall": round(float(g.mean()), 5)}
                       for k, g in found_flag.groupby(keys)}
    zero_match_s1 = n_s1_true == 0
    stats = {
        **cg.candidate_summary(sizes, len(s1)),
        "positive_pairs_total": n_pos,
        "positive_pairs_recovered": int(len(found_post)),
        "candidate_recall": rec(len(found_post)),
        "precap_candidate_pairs": int(precap_total),
        "precap_positive_pairs_recovered": int(len(found_pre)),
        "precap_candidate_recall": rec(len(found_pre)),
        "recall_by_rule": {r: {"candidate_pairs_precap": rule_pairs[r], "positives_recovered": rule_found[r],
                               "recall": rec(rule_found[r]), "positives_found_only_by_this_rule": rule_unique[r]}
                           for r in blocking},
        "recall_by_country": by(country),
        "recall_by_s1_match_bucket": by(bucket),
        "zero_match_s1": {"entities": int(zero_match_s1.sum()),
                          "mean_candidates": round(float(sizes[zero_match_s1].mean()), 3) if zero_match_s1.any() else 0.0},
        "cap": {"max_candidates_per_source1": cg_cfg["max_candidates_per_source1"],
                "s1_entities_capped": capped_s1, "pairs_dropped": dropped_pairs,
                "positive_pairs_dropped": dropped_pos, "recall_cost": rec(dropped_pos)},
        # every skipped key: [key, target frequency, positives lost that shared this key]
        "skipped_keys": {r: {"max_block_size": b["max_block_size"], "skipped_key_count": len(b["skipped"]),
                             "target_records_in_skipped_keys": int(b["t_skipped"]["t_idx"].nunique()),
                             "positives_lost_sharing_a_skipped_key": lost_skip[r]["pairs"],
                             "recall_cost": rec(lost_skip[r]["pairs"]),
                             "keys": [[k, int(f), int(lost_skip[r]["per_code"].get(c, 0))]
                                      for c, k, f in zip(b["skipped"]["code"], b["skipped"]["key"], b["skipped"]["frequency"])]}
                         for r, b in blocking.items()},
        "positives_lost_to_skipped_keys_any_rule": lost_skip["any_rule"],
        "recall_cost_of_skipped_keys_any_rule": rec(lost_skip["any_rule"]),
    }
    return stats, pd.concat(keep, ignore_index=True)


# ---------------------------------------------------------------- training stage
def train_stage(cfg: dict, timer: Timer, out: dict, until: str) -> None:
    seed, cgc, nj = cfg["seed"], cfg["candidate_generation"], cfg["n_jobs"]
    data_root = REPO / cfg["paths"]["data_root"]
    art = out["artifacts"]
    with timer("train_load_s1_gt"):
        s1_raw = load_source1(data_root, "train", cfg)
        gt = load_tsv(data_root / "train" / "train_ground_truth.tsv")
        gt = gt[gt["source1_entity_id"].isin(s1_raw["entity_id"])].reset_index(drop=True)
        pairs = gt_pairs(gt)
        del gt
    with timer("train_normalize_s1"):
        s1 = fe.prepare_records(s1_raw, nj, keep_raw=False)
    s1_raw_text = dict(zip(s1_raw["entity_id"], zip(s1_raw["business_name"], s1_raw["business_address"])))
    del s1_raw
    ids = s1["entity_id"]
    n_true = pd.Series(0, index=pd.Index(ids))
    counts = pairs.groupby("s1").size()
    n_true.loc[counts.index] = counts.to_numpy()
    n_true_arr = n_true.to_numpy()

    if cfg["mode"] == "dev":
        model_ids = ids.tolist()
    else:
        model_ids = ids.sample(n=min(cfg["training"]["source1_sample"], len(ids)), random_state=seed).tolist()
    val_ids = mm.split_entities(model_ids, cfg["validation"]["validation_fraction"], seed)
    keep_mask = ids.isin(set(model_ids)).to_numpy()

    cand_stats = {"mode": cfg["mode"], "source1_entities": len(s1), "candidate_generation_config": cgc,
                  "max_candidates_per_source1": cgc["max_candidates_per_source1"]}
    memory = {"s1_records_mb": _mb(s1)}
    feats_all, meta_all, example_ids = [], [], {}
    needed = set(pairs["target"])
    for src in SOURCES:
        with timer(f"train_load_{src}"):
            tgt_raw = load_target(data_root, "train", src, cfg, needed)
        with timer(f"train_normalize_{src}"):
            tgt = fe.prepare_records(tgt_raw, nj, keep_raw=False)
            del tgt_raw
        memory[f"{src}_records_mb"] = _mb(tgt)
        src_pairs = pairs[pairs["target"].str.startswith(src.upper() + "-")]
        positives = positional(src_pairs, s1, tgt)
        with timer(f"train_candidates_{src}"):
            stats, kept = candidates_for_source(s1, tgt, positives, cgc, keep_mask, n_true_arr)
        cand_stats[src] = stats
        log.info("%s: recall %.4f, %d pairs, mean %.1f per S1", src, stats["candidate_recall"],
                 stats["candidate_pairs"], stats["mean_candidates_per_s1"])
        if until == "candidates":
            continue
        with timer(f"train_features_{src}"):
            feats = fe.compute_features(kept, s1, tgt, src, nj)
        labels = mm.label_pairs(kept, positives)
        meta = pd.DataFrame({"s1": ids.to_numpy()[kept["s1_idx"].to_numpy()],
                             "target": tgt["entity_id"].to_numpy()[kept["t_idx"].to_numpy()],
                             "label": labels, "source": src})
        feats_all.append(feats)
        meta_all.append(meta)
        example_ids[src] = set(meta.loc[meta["s1"].isin(val_ids), "target"]) | set(
            src_pairs.loc[src_pairs["s1"].isin(val_ids), "target"])
        del tgt, kept, feats
    tot = sum(cand_stats[s]["positive_pairs_total"] for s in SOURCES)
    cand_stats["combined"] = {
        "positive_pairs_total": tot,
        "positive_pairs_recovered": sum(cand_stats[s]["positive_pairs_recovered"] for s in SOURCES),
        "candidate_recall": round(sum(cand_stats[s]["positive_pairs_recovered"] for s in SOURCES) / tot, 5) if tot else None,
        "candidate_pairs": sum(cand_stats[s]["candidate_pairs"] for s in SOURCES),
    }
    _json(art / "candidate_statistics.json", cand_stats)
    out["memory"] = memory
    if until == "candidates":
        return

    feats = pd.concat(feats_all, ignore_index=True)
    meta = pd.concat(meta_all, ignore_index=True)
    del feats_all, meta_all
    features = fe.active_features(cfg["features"])
    is_val = meta["s1"].isin(val_ids).to_numpy()
    y = meta["label"].to_numpy()
    _json(art / "feature_statistics.json", feature_statistics(feats, y, features))

    with timer("train_model"):
        train_rows = np.flatnonzero(~is_val)
        chosen = train_rows[mm.sample_training_pairs(y[train_rows], feats.iloc[train_rows],
                                                     cfg["validation"]["negative_ratio"],
                                                     cfg["validation"]["hard_negative_fraction"], seed)]
        model = mm.train_model(feats.iloc[chosen][features], y[chosen], cfg["model"], seed)
    val = meta[is_val].copy()
    val["score"] = inf.score(model, feats[is_val], features)
    empty_thr = cfg["threshold"].get("empty_target_address_threshold")  # None/absent = policy disabled
    score_cols = ["s1", "target", "source", "label", "score"]
    if empty_thr is not None:  # the decision needs the target's empty-normalized-address flag (address_missing_2)
        val[tt.EMPTY_ADDRESS_COLUMN] = feats["address_missing_2"].to_numpy()[is_val].astype(np.int8)
        score_cols.append(tt.EMPTY_ADDRESS_COLUMN)
    n_true_val = n_true.loc[sorted(val_ids)]
    # per-pair validation scores and per-entity truth counts, for paired comparisons across runs
    val[score_cols].to_csv(art / "validation_scores.tsv", sep="\t", index=False)
    n_true.loc[sorted(val_ids)].rename("n_true").rename_axis("s1").to_csv(art / "validation_entities.tsv", sep="\t")
    grid = tt.threshold_grid(cfg["threshold"])
    with timer("threshold_sweep"):
        th = tt.sweep(val, n_true_val, grid)
    thr = th["selected_threshold"]  # base threshold: selected exactly as without the policy
    th["per_source_at_selected"] = {s: tt.evaluate(val[val["source"] == s], _true_by_source(pairs, n_true_val, s), thr,
                                                   empty_thr) for s in SOURCES}
    th["validation_entities"] = len(val_ids)
    th["validation_candidate_pairs"] = int(len(val))
    th["pair_level_on_candidates"] = {
        "at_0.5": mm.pair_metrics(val["label"].to_numpy(), (val["score"] >= 0.5).astype(int)),
        "at_selected": mm.pair_metrics(val["label"].to_numpy(), tt.accept_frame(val, thr, empty_thr).astype(int)),
    }
    if empty_thr is not None:  # recorded here so every later decision (metrics, submissions, production) applies it
        th["empty_target_address_policy"] = {
            "empty_target_address_threshold": empty_thr, "base_threshold": thr,
            "effective_threshold_empty_target_address": max(thr, empty_thr),
            "rule": "target normalized address empty (address_missing_2): score >= max(base, empty_target_address_threshold); "
                    "otherwise score >= base"}
        th["selected_with_policy"] = tt.evaluate(val, n_true_val, thr, empty_thr)
    _json(art / "threshold_results.json", th)

    train_info = {
        "train_entities": len(model_ids) - len(val_ids), "validation_entities": len(val_ids),
        "train_candidate_pairs": int(len(train_rows)), "train_pairs_used": int(len(chosen)),
        "train_positives": int(y[chosen].sum()), "train_negatives": int(len(chosen) - y[chosen].sum()),
        "train_candidate_positives": int(y[train_rows].sum()),
        "negative_ratio": cfg["validation"]["negative_ratio"],
        "hard_negative_fraction": cfg["validation"]["hard_negative_fraction"],
        "coefficients": mm.coefficients(model, features),
        "entity_overlap_train_validation": len(set(meta.loc[~is_val, "s1"]) & set(val_ids)),
    }
    mm.save_model(model, features, art / "model", {"threshold": thr, "mode": cfg["mode"], "seed": seed})
    _json(art / "training_summary.json", train_info)
    with timer("error_analysis"):
        raw = {s: s1_raw_text[s] for s in val_ids}
        for src in SOURCES:  # raw target text is re-read for the few ids that can appear as examples
            raw.update(raw_lookup(data_root / "train" / f"train_source{src[1]}.tsv", example_ids[src]))
        _json(art / "error_analysis.json", error_analysis(val, pairs[pairs["s1"].isin(val_ids)], thr, raw, n_true_val,
                                                          empty_thr))


def _true_by_source(pairs, n_true_val, src):
    counts = pairs[pairs["target"].str.startswith(src.upper() + "-")].groupby("s1").size()
    return counts.reindex(n_true_val.index, fill_value=0)


def feature_statistics(feats: pd.DataFrame, y: np.ndarray, features: list) -> dict:
    pos, neg = feats[y == 1], feats[y == 0]
    corr = feats[features].corr().fillna(0.0)
    high = [[a, b, round(float(corr.loc[a, b]), 3)] for i, a in enumerate(features) for b in features[i + 1:]
            if abs(corr.loc[a, b]) >= 0.9]
    return {
        "feature_order": features, "rows": int(len(feats)), "positives": int((y == 1).sum()),
        "negatives": int((y == 0).sum()),
        "features": {f: {"dtype": str(feats[f].dtype), "missing_rate": float(feats[f].isna().mean()),
                         "positive_mean": round(float(pos[f].mean()), 4) if len(pos) else None,
                         "negative_mean": round(float(neg[f].mean()), 4) if len(neg) else None,
                         "quantiles": {q: round(float(feats[f].quantile(p)), 4)
                                       for q, p in (("p05", .05), ("p50", .5), ("p95", .95))},
                         "correlation_with_label": round(float(np.corrcoef(feats[f], y)[0, 1]), 4)
                         if feats[f].std() > 0 and y.std() > 0 else 0.0}
                     for f in feats.columns},
        "highly_correlated_pairs": high,
    }


def error_analysis(val: pd.DataFrame, val_pairs: pd.DataFrame, thr: float, raw: dict, n_true_val: pd.Series,
                   empty_thr=None) -> dict:
    def ex(rows, k=10):
        return [{"s1": r.s1, "target": r.target, "score": round(float(getattr(r, "score", float("nan"))), 4),
                 "s1_name": raw[r.s1][0], "target_name": raw.get(r.target, ("?", "?"))[0],
                 "s1_address": raw[r.s1][1], "target_address": raw.get(r.target, ("?", "?"))[1]}
                for r in rows.head(k).itertuples()]
    acc = tt.accept_frame(val, thr, empty_thr)
    fp = val[acc & (val["label"] == 0).to_numpy()].sort_values(["score", "s1", "target"], ascending=[False, True, True])
    fn_scored = val[~acc & (val["label"] == 1).to_numpy()].sort_values(["score", "s1", "target"])
    in_cands = val_pairs.merge(val[["s1", "target"]], how="left", on=["s1", "target"], indicator=True)
    never = in_cands[in_cands["_merge"] == "left_only"].sort_values(["s1", "target"])
    pos = val[val["label"] == 1]
    hard = pos.sort_values(["score", "s1", "target"])
    pred = val[acc].groupby("s1").size().reindex(n_true_val.index, fill_value=0)
    zero = n_true_val[n_true_val == 0].index
    multi = n_true_val[n_true_val > 1].index
    return {
        "threshold": thr,
        **({"empty_target_address_threshold": empty_thr} if empty_thr is not None else {}),
        "false_positives": {"count": int(len(fp)), "examples": ex(fp)},
        "false_negatives_scored_below_threshold": {"count": int(len(fn_scored)), "examples": ex(fn_scored)},
        "false_negatives_never_retrieved": {"count": int(len(never)), "examples": ex(never)},
        "hardest_positives_lowest_score": ex(hard),
        "zero_match_entities": {"count": int(len(zero)), "predicted_empty": int((pred.loc[zero] == 0).sum()),
                                "false_merge_examples": ex(fp[fp["s1"].isin(zero)])},
        "multi_match_entities": {"count": int(len(multi)),
                                 "mean_true_matches": round(float(n_true_val.loc[multi].mean()), 3) if len(multi) else 0,
                                 "mean_predicted": round(float(pred.loc[multi].mean()), 3) if len(multi) else 0,
                                 "predicted_at_least_two": int((pred.loc[multi] >= 2).sum())},
    }


# ---------------------------------------------------------------- test inference
def test_stage(cfg: dict, timer: Timer, out: dict) -> None:
    """Score every test S1 against both target sources and stream the submission files."""
    cgc, nj = cfg["candidate_generation"], cfg["n_jobs"]
    data_root = REPO / cfg["paths"]["data_root"]
    art, out_dir = out["artifacts"], out["output"]
    out_dir.mkdir(parents=True, exist_ok=True)
    model, features = mm.load_model(art / "model")
    th_res = json.loads((art / "threshold_results.json").read_text(encoding="utf-8"))
    thr, empty_thr = th_res["selected_threshold"], tt.policy_threshold(th_res)  # the same rule validation used
    with timer("test_load_s1"):
        s1_raw = load_source1(data_root, "test", cfg)
    test_dir, dev = data_root / "test", cfg["mode"] == "dev"
    if dev:  # the validator needs test files that match the dev subset
        test_dir = out_dir / "test_subset"
        inf.write_tsv(s1_raw, test_dir / "test_source1.tsv")
    with timer("test_normalize_s1"):
        s1 = fe.prepare_records(s1_raw, nj, keep_raw=False)
        del s1_raw
    s1_ids = s1["entity_id"].tolist()
    cand_info, parts, valid_ids = {}, [], (set() if dev else None)
    for src in SOURCES:
        with timer(f"test_load_{src}"):
            tgt_raw = load_target(data_root, "test", src, cfg, set())
        if dev:
            inf.write_tsv(tgt_raw, test_dir / f"test_source{src[1]}.tsv")
            valid_ids |= set(tgt_raw["entity_id"])
        with timer(f"test_normalize_{src}"):
            tgt = fe.prepare_records(tgt_raw, nj, keep_raw=False)
            del tgt_raw
        part = out_dir / f"_part_{src}.tsv"
        parts.append(part)
        with timer(f"test_candidates_features_scores_{src}"):
            blocking = cg.build_blocking(s1, tgt, cgc)
            n_pairs = n_dropped = n_capped = 0
            t_ids = tgt["entity_id"].to_numpy()
            with open(part, "w", encoding="utf-8", newline="\n") as f:
                for lo, hi, precap, kept, dropped in cg.iter_candidates(blocking, s1, tgt, cgc):
                    feats = fe.compute_features(kept, s1, tgt, src, nj)
                    scores = inf.score(model, feats, features)
                    f.writelines(inf.chunk_lines(lo, hi, kept["s1_idx"].to_numpy(), t_ids[kept["t_idx"].to_numpy()],
                                                 scores, thr, feats["address_missing_2"].to_numpy(), empty_thr))
                    del feats
                    n_pairs += len(kept)
                    n_dropped += len(dropped)
                    n_capped += int(dropped["s1_idx"].nunique())
                    log.info("  %s S1 %d-%d: %d candidates", src, lo, hi, len(kept))
        cand_info[src] = {"candidate_pairs": n_pairs, "pairs_dropped_by_cap": n_dropped, "s1_entities_capped": n_capped,
                          "max_candidates_per_source1": cgc["max_candidates_per_source1"],
                          "recall_cost": "unknown on test (no labels); see the train-side figures",
                          "skipped_keys": {r: {"skipped_key_count": len(b["skipped"]),
                                               "keys": b["skipped"][["key", "frequency"]].values.tolist()}
                                           for r, b in blocking.items()}}
        del tgt, blocking
    m_path, c_path = out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv"
    with timer("test_write_outputs"):
        merged = inf.merge_and_check(s1_ids, parts, m_path, c_path, valid_ids)
    for part in parts:
        part.unlink()
    with timer("validator"):
        if dev:
            validators = {"with_candidates": inf.run_validator(REPO / cfg["paths"]["validator"], m_path, c_path,
                                                               test_dir, check_ids=True)}
        else:  # the validator holds every candidate id in memory; full-size files are validated in S1 shards
            validators = {"sharded_with_candidates_check_ids": inf.run_validator_sharded(
                REPO / cfg["paths"]["validator"], m_path, c_path, test_dir, cfg["paths"]["validator_shard_rows"],
                out_dir / "_validator_shards")}
    # correction 4: test-side cap and every skipped key, too; kept next to the outputs when artifacts are frozen
    info_dir = out_dir if cfg["paths"].get("artifacts_read_only") else art
    stats_path = art / "candidate_statistics.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["test_inference"] = cand_info
    _json(info_dir / "candidate_statistics.json", stats)
    c = merged["counts"]
    _json(info_dir / "inference_summary.json", {
        "mode": cfg["mode"], "threshold": thr,
        **({"empty_target_address_policy": th_res["empty_target_address_policy"]} if empty_thr is not None else {}),
        "test_source1_entities": len(s1_ids),
        "candidates": {s: {k: v for k, v in d.items() if k != "skipped_keys"} for s, d in cand_info.items()},
        "candidate_pairs_written": c["candidate_pairs"], "entities_with_prediction": c["with_prediction"],
        "entities_with_multiple": c["with_multiple"], "mean_predictions_per_entity": round(c["predictions"] / len(s1_ids), 4),
        "matching_results": m_path.relative_to(REPO).as_posix(), "candidate_pairs": c_path.relative_to(REPO).as_posix(),
        "section45_checks": merged["checks"], "validator": validators,
        "sha256": {p.name: _sha(p) for p in (m_path, c_path)},
    })


def _sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- report
def _table(headers, rows) -> str:
    out = ["| " + " | ".join(map(str, headers)) + " |", "|" + "---|" * len(headers)]
    return "\n".join(out + ["| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in r) + " |"
                            for r in rows])


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.2f}%"


FEATURE_PURPOSE = {
    "name_norm_eq": "normalized names identical",
    "name_key_eq": "compact core names identical (suffix-, domain- and space-insensitive)",
    "name_token_jaccard": "token overlap of normalized names",
    "name_core_token_jaccard": "token overlap without legal suffixes",
    "name_char3_jaccard": "character 3-gram overlap (typo-tolerant)",
    "name_edit_sim": "1 - Levenshtein / max length",
    "name_translit_char3_jaccard": "3-gram overlap after Indic romanization",
    "name_translit_edit_sim": "edit similarity after romanization",
    "name_token_set_ratio": "order-insensitive token-set similarity of core names",
    "name_phonetic_eq": "consonant skeletons identical",
    "name_phonetic_edit_sim": "edit similarity of consonant skeletons (transliteration spelling)",
    "name_length_1": "S1 name length", "name_length_2": "target name length",
    "name_length_ratio": "shorter / longer name length", "name_length_difference": "absolute length difference",
    "name_token_count_1": "S1 token count", "name_token_count_2": "target token count",
    "name_token_count_difference": "token count difference",
    "legal_suffix_present_1": "S1 name has a legal suffix", "legal_suffix_present_2": "target name has a legal suffix",
    "legal_suffix_equal": "both have the same legal-suffix set",
    "name_script_mismatch": "one name in Indic script, the other Latin",
    "name_missing_1": "S1 name empty", "name_missing_2": "target name empty",
    "address_norm_eq": "normalized addresses identical", "address_token_jaccard": "address token overlap",
    "address_char3_jaccard": "address 3-gram overlap", "address_edit_sim": "address edit similarity",
    "address_component_jaccard": "overlap of comma components (order-insensitive street/city/state parts)",
    "address_missing_1": "S1 address empty", "address_missing_2": "target address empty",
    "address_missing_either": "either address empty",
    "house_number_both_present": "both house numbers extracted", "house_number_equal": "house numbers equal",
    "postal_both_present": "both postal codes extracted", "postal_equal": "postal codes equal",
    "state_both_present": "both states extracted",
    "state_equal": "states equal (codes, English names and native scripts unified)",
    "country_equal": "exact normalized country equal (open set)",
    "target_source_is_s2": "target from Source 2", "target_source_is_s3": "target from Source 3",
    "matched_by_exact_name_block": "retrieved by the exact-name block",
    "matched_by_exact_address_block": "retrieved by the exact-address block",
    "matched_by_numeric_block": "retrieved by the house-number block",
    "matched_by_prefix_block": "retrieved by a name or address prefix block",
    "matched_by_ngram_block": "retrieved by a rare-token or phonetic signature block",
    "number_of_blocks_that_retrieved_pair": "how many independent blocking rules retrieved the pair",
}

RULE_KEYS = {
    "exact_name": "compact core name; sorted unique core tokens",
    "exact_address": "sorted unique normalized address tokens",
    "numeric": "house number + first street token; house number + postal code",
    "name_prefix": "first N characters of the compact core name",
    "address_prefix": "first N characters of the compact normalized address",
    "name_ngram": "each of the 2 rarest core-name tokens (seen at least twice)",
    "name_phonetic": "consonant skeleton of the compact core name",
    "address_token": "each of the 2 rarest locality tokens of the address (seen at least twice)",
    "house_locality": "house number + each of the 2 rarest locality tokens (Phase 2.5)",
    "state_locality": "state + each of the 2 rarest locality tokens (Phase 2.5)",
    "address_signature": "sorted non-numeric, non-state address components (Phase 2.5)",
    "postal_locality": "postal code + each of the 2 rarest locality tokens (Phase 2.5)",
    "name_phonetic_token": "each of the 2 rarest per-token consonant skeletons (Phase 2.5)",
}


def _examples(rows) -> str:
    if not rows:
        return "_none_"
    return _table(["s1", "target", "score", "s1 name", "target name", "s1 address", "target address"],
                  [[e["s1"], e["target"], e["score"], e["s1_name"], e["target_name"], e["s1_address"],
                    e["target_address"]] for e in rows])


def write_report(cfg: dict, out: dict) -> None:
    """phase2_report.md, generated only from this run's artifacts."""
    art = out["artifacts"]

    def load(name):
        return json.loads((art / name).read_text(encoding="utf-8"))
    cs, fs, th, tr, ea, isum = (load(n) for n in (
        "candidate_statistics.json", "feature_statistics.json", "threshold_results.json",
        "training_summary.json", "error_analysis.json", "inference_summary.json"))
    rt = load(f"runtime_{cfg['mode']}.json")
    cgc = cs["candidate_generation_config"]
    L = [f"# Phase 2 Report: Baseline Entity Matching ({cfg['mode']} mode)", "",
         "Generated automatically by `phase2/src/run_phase2.py` from the JSON artifacts in `phase2/artifacts/`.", ""]
    if cfg["mode"] == "dev":
        d = cfg["development"]
        L += [f"> **Development mode.** {d['max_source1_rows']:,} sampled Source-1 entities per split and "
              f"{d['max_target_rows'] or 'all'} target records per source. The target pools are much smaller than "
              "the real sources, so candidate counts and precision here are not representative of full scale.", ""]

    L += ["## 1. Objective", "",
          "A baseline pipeline: normalization, blocking (S1 to S2 and S1 to S3 separately), candidate recall "
          "measurement, pair features, a logistic-regression scorer, a threshold chosen on validation macro F0.5 "
          "per Source-1 entity, zero/one/many-match assembly, and the two submission files.", ""]

    rules = list(cs["s2"]["recall_by_rule"])
    L += ["## 2. Candidate generation", "",
          "Every key is prefixed with the exact normalized country (trimmed and casefolded, an open set that is "
          f"never mapped). Target keys holding more than {cgc['max_block_size']} records are skipped, and each one "
          "is recorded with its frequency and the positives it cost. Each S1 entity keeps at most "
          f"{cs['max_candidates_per_source1']} candidates per target source (most rules first, then the best "
          "name or address edit similarity); the dropped pairs and their recall cost are recorded.", "",
          _table(["rule", "key"], [[r, RULE_KEYS[r]] for r in rules]), "",
          _table(["metric", "S2", "S3"],
                 [[k, cs["s2"][k], cs["s3"][k]] for k in (
                     "candidate_recall", "precap_candidate_recall", "positive_pairs_total",
                     "positive_pairs_recovered", "candidate_pairs", "mean_candidates_per_s1",
                     "median_candidates_per_s1", "p95_candidates_per_s1", "p99_candidates_per_s1",
                     "max_candidates_per_s1", "s1_with_zero_candidates")]
                 + [["recall cost of the per-S1 cap", cs["s2"]["cap"]["recall_cost"], cs["s3"]["cap"]["recall_cost"]],
                    ["positives lost that shared a skipped key", cs["s2"]["positives_lost_to_skipped_keys_any_rule"],
                     cs["s3"]["positives_lost_to_skipped_keys_any_rule"]]]), "",
          f"Combined candidate recall (S2 and S3): **{_pct(cs['combined']['candidate_recall'])}**. This is the "
          "ceiling on what the model can recover.", "",
          "Recall by rule, before the cap (\"only\" = positives that no other rule found):", "",
          _table(["rule", "S2 recall", "S2 only", "S3 recall", "S3 only", "S2 pairs", "S3 pairs",
                  "S2 skipped keys", "S3 skipped keys"],
                 [[r, cs["s2"]["recall_by_rule"][r]["recall"],
                   cs["s2"]["recall_by_rule"][r]["positives_found_only_by_this_rule"],
                   cs["s3"]["recall_by_rule"][r]["recall"],
                   cs["s3"]["recall_by_rule"][r]["positives_found_only_by_this_rule"],
                   cs["s2"]["recall_by_rule"][r]["candidate_pairs_precap"],
                   cs["s3"]["recall_by_rule"][r]["candidate_pairs_precap"],
                   cs["s2"]["skipped_keys"][r]["skipped_key_count"], cs["s3"]["skipped_keys"][r]["skipped_key_count"]]
                  for r in rules]), "",
          "Recall by S1 country and S1 match bucket:", "",
          _table(["group", "S2 recall", "S3 recall"],
                 [[f"country = {k}", cs["s2"]["recall_by_country"][k]["recall"],
                   cs["s3"]["recall_by_country"].get(k, {}).get("recall")] for k in cs["s2"]["recall_by_country"]]
                 + [[f"bucket = {k}", cs["s2"]["recall_by_s1_match_bucket"][k]["recall"],
                     cs["s3"]["recall_by_s1_match_bucket"].get(k, {}).get("recall")]
                    for k in cs["s2"]["recall_by_s1_match_bucket"]]), "",
          "Every skipped key with its target frequency and lost positives is listed in "
          "`candidate_statistics.json` under `<source>.skipped_keys.<rule>.keys`.", ""]

    L += ["## 3. Features", "",
          f"{len(fs['feature_order'])} features, in the exact order stored in `artifacts/model/feature_schema.json`. "
          "A missing value never creates similarity (every similarity is 0 when either side is empty), and "
          "missingness has explicit indicator features.", "",
          _table(["feature", "purpose", "positive mean", "negative mean", "correlation with label"],
                 [[f, FEATURE_PURPOSE.get(f, ""), fs["features"][f]["positive_mean"],
                   fs["features"][f]["negative_mean"], fs["features"][f]["correlation_with_label"]]
                  for f in fs["feature_order"]]), "",
          "Highly correlated feature pairs (|r| >= 0.9): "
          + (", ".join(f"{a} ~ {b} ({r})" for a, b, r in fs["highly_correlated_pairs"]) or "none") + ".", ""]

    L += ["## 4. Training", "",
          _table(["item", "value"], [
              ["model", f"StandardScaler + LogisticRegression (C = {cfg['model']['C']}, lbfgs)"],
              ["split", f"entity level: {tr['train_entities']:,} training and {tr['validation_entities']:,} "
                        f"validation Source-1 entities; entities in both = {tr['entity_overlap_train_validation']}"],
              ["training candidate pairs", f"{tr['train_candidate_pairs']:,} ({tr['train_candidate_positives']:,} positive)"],
              ["pairs used to fit", f"{tr['train_pairs_used']:,} = {tr['train_positives']:,} positives + "
                                    f"{tr['train_negatives']:,} negatives"],
              ["negative sampling", f"{tr['negative_ratio']} negatives per positive, drawn from the training "
                                    f"candidates; {int(100 * tr['hard_negative_fraction'])}% are the most "
                                    "name-and-address-similar non-matches (hard negatives), the rest uniform "
                                    f"(seed {cfg['seed']})"]]), "",
          "Largest standardized coefficients:", "",
          _table(["feature", "coefficient"],
                 sorted(([k, v] for k, v in tr["coefficients"].items() if k != "intercept"),
                        key=lambda kv: -abs(kv[1]))[:12]), ""]

    L += ["## 5. Validation", "",
          ("**Warning: the selected threshold is on the edge of the grid.** " if th["selected_on_grid_edge"] else "")
          + f"Selected threshold **{th['selected_threshold']}** by {th['selection_metric']}, on "
          f"{th['validation_entities']:,} validation entities ({th['validation_candidate_pairs']:,} candidate "
          "pairs). Every validation entity counts, including zero-match entities and entities with no candidates. "
          "Pooled FN include positives that blocking never retrieved.", "",
          _table(["threshold", "macro F0.5", "TP", "FP", "FN", "precision", "recall", "pooled F0.5",
                  "entities with prediction", "avg predictions per entity", "zero-match entities scored 1"],
                 [[r["threshold"], round(r["macro_f05"], 4), r["tp"], r["fp"], r["fn"], round(r["precision"], 4),
                   round(r["recall"], 4), round(r["pooled_f05"], 4), r["entities_with_prediction"],
                   round(r["average_predictions_per_entity"], 3),
                   f"{r['zero_match_entities_scored_1']}/{r['zero_match_entities']}"] for r in th["sweep"]]), "",
          "At the selected threshold, by target source. Diagnostic only: an entity with no match in that source "
          "and no prediction scores 1.0 there, so these are not comparable to the challenge metric.", "",
          _table(["source", "macro F0.5 (that source only)", "precision", "recall"],
                 [[s, round(v["macro_f05"], 4), round(v["precision"], 4), round(v["recall"], 4)]
                  for s, v in th["per_source_at_selected"].items()]), "",
          "Pair level on validation candidates only (this recall excludes pairs blocking never produced):", "",
          _table(["threshold", "TP", "FP", "FN", "TN", "precision", "recall", "F0.5"],
                 [[k, v["tp"], v["fp"], v["fn"], v["tn"], round(v["precision"], 4), round(v["recall"], 4),
                   round(v["f05"], 4)] for k, v in th["pair_level_on_candidates"].items()]), ""]

    mmx = ea["multi_match_entities"]
    L += ["## 6. Error analysis (validation)", "",
          f"- False positives: {ea['false_positives']['count']:,}.",
          f"- False negatives scored below the threshold: {ea['false_negatives_scored_below_threshold']['count']:,}.",
          f"- False negatives never retrieved by blocking: {ea['false_negatives_never_retrieved']['count']:,}.",
          f"- Zero-match entities: {ea['zero_match_entities']['count']:,}, of which "
          f"{ea['zero_match_entities']['predicted_empty']:,} were correctly predicted empty.",
          f"- Multi-match entities: {mmx['count']:,}; mean true matches {mmx['mean_true_matches']}, mean "
          f"predicted {mmx['mean_predicted']}, entities with at least two predictions {mmx['predicted_at_least_two']:,}.", "",
          "### False positives (highest scores)", "", _examples(ea["false_positives"]["examples"]), "",
          "### False negatives scored below the threshold (lowest scores)", "",
          _examples(ea["false_negatives_scored_below_threshold"]["examples"]), "",
          "### False negatives never retrieved by blocking", "", _examples(ea["false_negatives_never_retrieved"]["examples"]), "",
          "### Hardest retrieved positives (lowest scores)", "", _examples(ea["hardest_positives_lowest_score"]), "",
          "### False merges on zero-match entities", "", _examples(ea["zero_match_entities"]["false_merge_examples"]), ""]

    L += ["## 7. Test inference, submission and scalability", "",
          _table(["item", "value"], [
              ["test Source-1 entities", f"{isum['test_source1_entities']:,}"],
              ["candidate pairs S2 / S3", f"{isum['candidates']['s2']['candidate_pairs']:,} / "
                                          f"{isum['candidates']['s3']['candidate_pairs']:,}"],
              ["pairs dropped by the per-S1 cap S2 / S3", f"{isum['candidates']['s2']['pairs_dropped_by_cap']:,} / "
                                                          f"{isum['candidates']['s3']['pairs_dropped_by_cap']:,}"],
              ["threshold used", isum["threshold"]],
              ["entities with at least one / at least two predictions",
               f"{isum['entities_with_prediction']:,} / {isum['entities_with_multiple']:,}"],
              ["files", f"`{isum['matching_results']}`, `{isum['candidate_pairs']}`"],
              ["section 45 checks", ", ".join(
                  f"{k} = {'pass' if v else ('not checked in-process' if v is None else 'FAIL')}"
                  for k, v in isum["section45_checks"].items())]]
             + [[f"challenge validator ({name}: `{v['command']}`)",
                 ("PASS" if v["passed"] else "FAIL") + ": " + " / ".join(v["output"][-2:])]
                for name, v in isum["validator"].items()] + [
              ["sha256", ", ".join(f"{k} {v[:16]}..." for k, v in isum["sha256"].items())]]), "",
          "Runtime per stage (seconds):", "", _table(["stage", "seconds"], list(rt["seconds"].items())), "",
          f"Memory of the normalized record tables (MB): {rt['memory']}. Worker processes: {rt['n_jobs']}; "
          f"Source-1 chunk size {rt['source1_chunk_size']:,}; features are computed in chunks of 50,000 pairs. "
          "No step builds an S1-by-target product: candidates are equi-joins on blocking keys, and oversized keys "
          "are skipped.", ""]

    L += ["## 8. Limitations", "",
          "- Candidate recall is the ceiling. Positives that no rule retrieves (usually a heavily corrupted name "
          "together with a differently written address) cannot be predicted.",
          "- Skipping high-frequency keys and capping candidates per S1 trade recall for bounded cost; both costs "
          "are measured and recorded.",
          "- Indic romanization is a deterministic approximation. The consonant skeleton recovers many, but not "
          "all, transliterated spellings.",
          "- One logistic-regression model and one threshold serve both target sources and all countries. France "
          "appears only in test; it is handled by the same country-agnostic rules but has no training examples.",
          "- The spec's `city_equal` (16.8) is replaced by comma-component overlap, because the position of the "
          "city is not deterministic across sources.", ""]
    path = out["reports"] / "phase2_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")

# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 2 baseline pipeline")
    ap.add_argument("--config", default=str(REPO / "phase2" / "config" / "phase2_config.yaml"))
    ap.add_argument("--mode", choices=["dev", "full"])
    ap.add_argument("--until", choices=["candidates", "model", "all"], default="all")
    ap.add_argument("--test-only", action="store_true",
                    help="skip training; score the test split with the model and threshold already in <phase2_dir>/artifacts")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(Path(args.config), args.mode)
    base = REPO / cfg["paths"]["phase2_dir"]
    out = {"artifacts": base / "artifacts", "output": base / "output" / cfg["mode"], "reports": base / "reports"}
    timer = Timer()
    log.info("Phase 2 mode=%s until=%s test_only=%s", cfg["mode"], args.until, args.test_only)
    if cfg["paths"].get("artifacts_read_only") and not args.test_only:
        raise SystemExit(f"{out['artifacts']} holds a frozen model (paths.artifacts_read_only); use --test-only")
    if not args.test_only:
        train_stage(cfg, timer, out, args.until)
    if args.until == "all" or args.test_only:
        test_stage(cfg, timer, out)
        if args.test_only:  # the frozen training artifacts are not rewritten
            _json(out["output"] / "runtime_test_only.json", {"seconds": dict(timer), "n_jobs": cfg["n_jobs"],
                  "source1_chunk_size": cfg["candidate_generation"]["source1_chunk_size"]})
            return 0
    _json(out["artifacts"] / f"runtime_{cfg['mode']}.json", {"seconds": dict(timer), "memory": out.get("memory", {}),
                                                              "n_jobs": cfg["n_jobs"], "until": args.until,
                                                              "source1_chunk_size": cfg["candidate_generation"]["source1_chunk_size"]})
    if args.until == "all":
        write_report(cfg, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
