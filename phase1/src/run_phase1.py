"""Phase 1 entry point: data understanding only.

    python phase1/src/run_phase1.py [--config ...] [--data-root ...] [--max-rows N]

Reads TRAIN data (test files: headers and row counts only), writes
reports/phase1_report.md and artifacts/dataset_statistics.json. It never
generates candidates, matches, predictions or submission files.
"""
import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import yaml

import analysis as A
from profiling import check_schema, discover_datasets, load_tsv, profile_frame

log = logging.getLogger("phase1")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "phase1" / "config" / "phase1_config.yaml"


# ---------------------------------------------------------------- config / AWS
def load_config(path: Path, overrides: dict) -> dict:
    """YAML config, then PHASE1_* environment variables, then CLI overrides."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("data_root", "output_dir", "output_s3_uri"):
        if os.environ.get(f"PHASE1_{key.upper()}"):
            cfg[key] = os.environ[f"PHASE1_{key.upper()}"]
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def _local(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _split_s3(uri: str):
    bucket, _, prefix = uri[len("s3://"):].partition("/")
    prefix = prefix.strip("/")
    return bucket, f"{prefix}/" if prefix else ""


def stage_from_s3(uri: str, cache: Path) -> Path:
    """Download every .tsv under an s3:// prefix into ``cache`` (credentials from the IAM role/env)."""
    import boto3  # only needed when data lives in S3
    bucket, prefix = _split_s3(uri)
    s3 = boto3.client("s3")
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith(".tsv"):
                continue
            dest = cache / obj["Key"][len(prefix):]
            if dest.exists() and dest.stat().st_size == obj["Size"]:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            log.info("downloading s3://%s/%s", bucket, obj["Key"])
            s3.download_file(bucket, obj["Key"], str(dest))
    return cache


def upload_to_s3(files: list, base: Path, uri: str) -> None:
    """Upload Phase 1 outputs, keeping their paths relative to ``base``."""
    import boto3
    bucket, prefix = _split_s3(uri)
    s3 = boto3.client("s3")
    for f in files:
        key = prefix + f.relative_to(base).as_posix()
        log.info("uploading %s -> s3://%s/%s", f, bucket, key)
        s3.upload_file(str(f), bucket, key)


# ---------------------------------------------------------------- pipeline
def run(data_root: Path, cfg: dict) -> dict:
    """Execute every Phase 1 analysis and return the statistics dictionary."""
    seed, max_rows = cfg["seed"], cfg.get("max_rows")
    train = data_root / "train"
    # paths are left out so the artifact is identical wherever it runs
    paths = ("data_root", "output_dir", "output_s3_uri", "local_cache_dir")
    stats = {"config": {k: cfg[k] for k in sorted(cfg) if k not in paths},
             "profiles": {}, "variation": {}, "country": {}, "gt_coverage": {}}

    log.info("discovering datasets under %s", data_root)
    stats["discovery"] = discover_datasets(data_root)
    stats["schema_issues"] = check_schema(stats["discovery"])
    if any("train" in issue for issue in stats["schema_issues"]):
        raise SystemExit(f"train schema problems: {stats['schema_issues']}")

    log.info("ground truth")
    gt = load_tsv(train / "train_ground_truth.tsv", max_rows)
    stats["profiles"]["train_ground_truth"] = profile_frame(gt, "source1_entity_id", "S1-")
    pairs = A.explode_ground_truth(gt)
    counts = A.match_counts(gt)
    stats["ground_truth"] = A.analyze_ground_truth(gt, pairs, counts)
    sample_ids = A.sample_s1_ids(gt["source1_entity_id"], cfg["pair_sample_s1"], seed)
    del gt

    pos = pairs[pairs["source1_entity_id"].isin(sample_ids)]
    positives = sorted(zip(pos["source1_entity_id"], pos["target_id"]))
    truth = {s1: set() for s1 in sample_ids}
    for s1, tgt in positives:
        truth[s1].add(tgt)
    needed = set(sample_ids) | {t for _, t in positives}

    records, pools = {}, []
    for i in (1, 2, 3):
        name, prefix = f"train_source{i}", f"S{i}-"
        log.info("loading %s", name)
        df = load_tsv(train / f"{name}.tsv", max_rows)
        log.info("profiling %s (%d rows)", name, len(df))
        stats["profiles"][name] = profile_frame(df, "entity_id", prefix)
        stats["variation"][name] = A.analyze_text_variation(df, cfg["variation_sample_rows"], seed)
        stats["country"][name] = A.analyze_country(df)
        keep = df["entity_id"].isin(needed)
        if i == 1:
            stats["singletons"] = A.analyze_singletons(df, counts, cfg["singleton_sample_rows"], seed)
        else:
            stats["gt_coverage"][name] = A.target_coverage(df["entity_id"], pairs, prefix)
            pool = df.sample(n=min(cfg["negative_pool_rows"], len(df)), random_state=seed)
            pools.append(pool[["entity_id", "country"]])
            keep |= df["entity_id"].isin(pool["entity_id"])
        sub = df[keep]
        records.update(zip(sub["entity_id"], zip(sub["business_name"], sub["business_address"], sub["country"])))
        del df, sub

    log.info("pair-level analysis")
    stats["pairs"] = _pair_analysis(sample_ids, positives, truth, records, pools, pairs, counts, seed)
    stats["f05_validation"] = _f05_validation(truth)
    return stats


def _pair_analysis(sample_ids, positives, truth, records, pools, gt_pairs, counts, seed) -> dict:
    import pandas as pd
    usable_pos = [(a, b) for a, b in positives if a in records and b in records]
    usable_s1 = [s for s in sample_ids if s in records]
    negatives = A.sample_negatives(usable_s1, records, truth, pd.concat(pools), seed)
    labeled = [(a, b, "positive") for a, b in usable_pos] + [(a, b, "sampled_negative") for a, b in negatives]
    feats = A.pair_features(labeled, records)
    bucket = dict(zip(counts["source1_entity_id"], counts["n_matches"].map(A._bucket)))
    feats["s1_bucket"] = feats["source1_entity_id"].map(bucket)
    is_pos = feats["label"] == "positive"

    neg_targets = {b for _, b in negatives}
    signal = {}
    for rule, mask in {
        "name_norm_eq": feats["name_norm_eq"],
        "address_norm_eq": feats["address_norm_eq"],
        "name_norm_eq_and_address_norm_eq": feats["name_norm_eq"] & feats["address_norm_eq"],
    }.items():
        signal[rule] = A.f_beta(int((mask & is_pos).sum()), int((mask & ~is_pos).sum()),
                                int((~mask & is_pos).sum()))
    pos_feats = feats[is_pos]
    hard = pos_feats[pos_feats["name_token_jaccard"].fillna(0) == 0].sort_values(
        ["source1_entity_id", "target_id"]).head(10)
    return {
        "sampling": {
            "seed": seed,
            "s1_entities_sampled": len(sample_ids),
            "s1_entities_with_records": len(usable_s1),
            "ground_truth_positive_pairs": len(positives),
            "positive_pairs_with_both_records": len(usable_pos),
            "sampled_negative_pairs": len(negatives),
            "negative_targets_matched_to_another_s1": int(gt_pairs["target_id"].isin(neg_targets).sum()),
        },
        "by_label": A.summarize_pairs(feats.drop(columns="s1_bucket"), ["label"]),
        "positives_by_target_source": A.summarize_pairs(pos_feats.drop(columns="s1_bucket"), ["target_source"]),
        "positives_by_s1_bucket": A.summarize_pairs(pos_feats, ["s1_bucket"]),
        "positives_by_country": A.summarize_pairs(
            pos_feats.drop(columns="s1_bucket").assign(country=[records[s][2] for s in pos_feats["source1_entity_id"]]),
            ["country"]),
        "exact_equality_signal_on_sample": signal,
        "name_tokens_on_one_side_of_positive": A.token_differences(usable_pos, records, 0, A.normalize_name),
        "address_tokens_on_one_side_of_positive": A.token_differences(usable_pos, records, 1, A.normalize_address),
        "hard_positive_examples": [
            {"s1": r.source1_entity_id, "target": r.target_id,
             "s1_name": records[r.source1_entity_id][0], "target_name": records[r.target_id][0],
             "s1_address": records[r.source1_entity_id][1], "target_address": records[r.target_id][1]}
            for r in hard.itertuples()],
    }


def _f05_validation(truth: dict) -> dict:
    """Exercise the metric on known cases; this is a check of the scorer, not a model score."""
    example = A.entity_f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    return {
        "readme_example_expected_0_714": round(example, 4),
        "perfect_prediction_on_sample": A.macro_f05(truth, truth),
        "all_empty_prediction_on_sample": A.macro_f05({}, truth),
    }


# ---------------------------------------------------------------- outputs
def _clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean_json(v) for v in (sorted(obj) if isinstance(obj, set) else obj)]
    if hasattr(obj, "item"):  # numpy scalar
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def _table(headers: list, rows: list) -> str:
    lines = ["| " + " | ".join(map(str, headers)) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return "\n".join(lines)


def render_report(s: dict) -> str:
    """Markdown report generated purely from the statistics dictionary."""
    srcs = ["train_source1", "train_source2", "train_source3"]
    prof, gt, pr = s["profiles"], s["ground_truth"], s["pairs"]
    out = ["# Phase 1 Report — Data Understanding (TRAIN)", "",
           "Generated by `phase1/src/run_phase1.py`. All numbers come from "
           "`phase1/artifacts/dataset_statistics.json`. Test files were inspected for "
           "header/row count only.", ""]

    out += ["## 1. Dataset discovery and schema", "", _table(
        ["file", "columns", "data rows", "MB"],
        [[k, ", ".join(v["header"]), f"{v['data_rows']:,}", round(v["bytes"] / 1e6, 1)]
         for k, v in s["discovery"].items()]), "",
        f"Schema issues: {s['schema_issues'] or 'none'}", ""]

    out += ["## 2. Profiling, missing values and duplicates", "", _table(
        ["table", "rows", "IDs unique", "dup ID rows", "bad ID prefix", "exact dup rows",
         "dup rows ignoring ID", "rows w/ dup name", "rows w/ dup address", "rows w/ dup name+address"],
        [[t, f"{p['rows']:,}", p["id"]["is_unique"], p["id"]["duplicate_id_rows"], p["id"]["bad_prefix_rows"],
          p["duplicates"]["exact_duplicate_rows"], p["duplicates"]["duplicate_rows_ignoring_id"],
          *[f"{p['duplicates'][k]['rows_in_duplicate_groups']:,}" if k in p["duplicates"] else "-"
            for k in ("name", "address", "name_address")]]
         for t, p in prof.items()]), "",
        "Duplicate name/address counts consider non-empty values only. Nothing was removed.", "",
        _table(["table", "column", "empty", "whitespace-only", "missing %", "unique %",
                "lead/trail space", "len min/median/p95/max"],
               [[t, c, v["empty_string"], v["whitespace_only"], v["missing_pct"], v["unique_pct"],
                 v["leading_or_trailing_whitespace"],
                 "/".join(str(v["length"].get(k, "-")) for k in ("min", "median", "p95", "max"))]
                for t, p in prof.items() for c, v in p["column_profile"].items()]), ""]

    def rate_table(kind):
        keys = list(s["variation"][srcs[0]][kind])
        return _table(["pattern"] + srcs, [[k] + [s["variation"][x][kind][k] for x in srcs] for k in keys])

    out += ["## 3. Name variation", "",
            f"Share of rows matching each pattern (fixed-seed sample of up to "
            f"{s['config']['variation_sample_rows']:,} rows per source).", "", rate_table("name_rates"), "",
            "Most frequent raw legal-suffix spellings:", ""]
    out += [f"- **{x}**: " + ", ".join(f"`{k}` {n:,}" for k, n in s["variation"][x]["legal_suffix_forms"][:15])
            for x in srcs]
    out += ["", "Share of names changed by `normalize_name` (anything beyond identity): " +
            ", ".join(f"{x} {s['variation'][x]['names_changed_by_normalization']}" for x in srcs), ""]

    out += ["## 4. Address variation", "", rate_table("address_rates"), "",
            "Per-country address rates:", ""]
    for x in srcs:
        by = s["variation"][x]["address_rates_by_country"]
        keys = ["empty", "starts_with_digit", "starts_with_state_code", "ends_with_state_code",
                "us_zip_at_end", "six_digit_pin", "landmark", "house_number_marker", "all_upper_latin"]
        out += [f"**{x}**", "", _table(["pattern"] + list(by), [[k] + [by[c][k] for c in by] for k in keys]), ""]
    out += ["Street-type spellings:", ""]
    out += [f"- **{x}**: " + ", ".join(f"`{k}` {n:,}" for k, n in s["variation"][x]["street_type_forms"][:15])
            for x in srcs]
    out += ["", "Commas per address (count of rows):", ""]
    out += [f"- **{x}**: " + ", ".join(f"{k}:{n:,}" for k, n in s["variation"][x]["address_comma_count"][:10])
            for x in srcs]

    out += ["", "## 5. Country", "", _table(
        ["source", "distinct raw", "distinct normalized", "missing", "distribution", "variants", "unexpected"],
        [[x, c["distinct_raw"], c["distinct_normalized"], c["missing"],
          ", ".join(f"{k}: {n:,}" for k, n in c["distribution"]), c["normalized_variants"] or "none",
          c["unexpected_values"] or "none"] for x, c in s["country"].items()]), "",
        "No code/name mapping is applied (the README states test adds `France`; country is an open set).", ""]

    out += ["## 6. Ground truth", "", _table(["metric", "value"], [
        ["Source-1 rows / unique", f"{gt['source1_rows']:,} / {gt['source1_ids_unique']}"],
        ["positive pairs", f"{gt['positive_pairs']:,}"],
        ["pairs by target source", gt["pairs_by_target_source"]],
        ["S1 with zero / one / multiple matches",
         " / ".join(f"{gt['match_bucket_counts'][b]:,} ({gt['match_bucket_pct'][b]}%)" for b in ("zero", "one", "multi"))],
        ["composition", gt["composition"]],
        ["duplicate pairs", gt["duplicate_pairs"]],
        ["targets linked to >1 S1", gt["targets_linked_to_multiple_s1"]],
        ["targets with bad prefix", gt["targets_with_bad_prefix"]],
    ]), "", "Matches per Source-1 entity: " + ", ".join(f"{k}:{v:,}" for k, v in gt["matches_per_s1"].items()), "",
        "S2 matches per S1: " + ", ".join(f"{k}:{v:,}" for k, v in gt["s2_matches_per_s1"].items()), "",
        "S3 matches per S1: " + ", ".join(f"{k}:{v:,}" for k, v in gt["s3_matches_per_s1"].items()), "",
        _table(["source", "records", "referenced by GT", "not referenced (%)", "GT ids missing from source"],
               [[x, f"{c['records']:,}", f"{c['referenced_by_ground_truth']:,}",
                 f"{c['not_referenced']:,} ({c['not_referenced_pct']}%)", c["gt_ids_missing_from_source"]]
                for x, c in s["gt_coverage"].items()]), ""]

    sg = s["singletons"]
    buckets = [b for b in ("zero", "one", "multi") if b in sg]
    out += ["## 7. Singleton analysis (Source 1 by ground-truth match count)", "", _table(
        ["metric"] + buckets,
        [["records"] + [f"{sg[b]['records']:,}" for b in buckets],
         ["country %"] + [sg[b]["country_pct"] for b in buckets],
         ["address missing %"] + [sg[b]["address_missing_pct"] for b in buckets],
         ["name length mean"] + [sg[b]["name_length_mean"] for b in buckets],
         ["address length mean"] + [sg[b]["address_length_mean"] for b in buckets]]
        + [[f"name {k}"] + [sg[b]["name_rates"][k] for b in buckets] for k in sg[buckets[0]]["name_rates"]]
        + [[f"address {k}"] + [sg[b]["address_rates"][k] for b in buckets] for k in sg[buckets[0]]["address_rates"]]),
        "", f"Zero-match % by country: {sg['zero_match_pct_by_country']}",
        f"Exactly-one-match records, source of the single match: {sg.get('one', {}).get('single_match_source')}", ""]

    sm = pr["sampling"]
    feat_rows = ["name_raw_eq", "name_norm_eq", "name_token_order_only", "address_missing_either",
                 "address_raw_eq", "address_norm_eq", "address_token_order_only", "country_raw_eq"]
    num_rows = ["name_token_jaccard", "name_char3_jaccard", "name_edit_sim", "name_edit_distance",
                "address_token_jaccard", "address_char3_jaccard", "address_edit_sim"]

    def pair_table(block):
        cols = list(block)
        rows = [["pairs"] + [block[c]["pairs"] for c in cols]]
        rows += [[f] + [block[c][f] for c in cols] for f in feat_rows]
        rows += [[f"{f} (median / mean)"] + [f"{block[c][f]['median']} / {block[c][f]['mean']}" for c in cols]
                 for f in num_rows]
        return _table(["feature"] + cols, rows)

    out += ["## 8. Match difficulty and positive vs negative pairs", "",
            f"- **Ground-truth positives**: all {sm['ground_truth_positive_pairs']:,} GT pairs of "
            f"{sm['s1_entities_sampled']:,} Source-1 entities sampled with seed {sm['seed']} "
            f"({sm['positive_pairs_with_both_records']:,} with both records loaded).",
            f"- **Sampled comparison pairs (negatives)**: {sm['sampled_negative_pairs']:,}; per sampled S1 entity one "
            "uniformly random S2/S3 record of the same country from a fixed-seed pool, excluding its GT matches. "
            f"{sm['negative_targets_matched_to_another_s1']:,} of them are GT matches of a *different* S1 entity.",
            "- **Assumption**: train GT is complete for Source 1 (every S1 id is listed, no S2/S3 id is linked "
            "twice), so a pair absent from GT is treated as a non-match.",
            "- **Limitation**: random negatives are *easy*; they bound the separability from above and say nothing "
            "about near-duplicate businesses. Hard negatives need candidate generation, which is Phase 2.", "",
            pair_table(pr["by_label"]), "", "Positives by target source:", "", pair_table(pr["positives_by_target_source"]),
            "", "Positives by S1 match bucket:", "", pair_table(pr["positives_by_s1_bucket"]),
            "", "Positives by country:", "", pair_table(pr["positives_by_country"]), "",
            "Normalized name tokens present on only one side of a positive pair (abbreviations, suffixes, typos):", "",
            ", ".join(f"`{k}` {n}" for k, n in pr["name_tokens_on_one_side_of_positive"]), "",
            "Normalized address tokens present on only one side of a positive pair:", "",
            ", ".join(f"`{k}` {n}" for k, n in pr["address_tokens_on_one_side_of_positive"]), "",
            "Hard positives (no shared normalized name token):", "",
            _table(["s1", "target", "s1 name", "target name", "s1 address", "target address"],
                   [[h["s1"], h["target"], h["s1_name"], h["target_name"], h["s1_address"], h["target_address"]]
                    for h in pr["hard_positive_examples"]]), "",
            "Exact-equality signal on the labeled sample (pair level, ~1 negative per S1; precision is NOT "
            "representative of real prevalence — this measures signal strength, it is not a matcher):", "",
            _table(["rule", "TP", "FP", "FN", "precision", "recall", "F0.5"],
                   [[k, v["tp"], v["fp"], v["fn"], round(v["precision"], 4), round(v["recall"], 4),
                     round(v["f_beta"], 4)] for k, v in pr["exact_equality_signal_on_sample"].items()]), ""]

    fv = s["f05_validation"]
    out += ["## 9. F0.5 validation infrastructure", "",
            "`analysis.f_beta` (β=0.5, zero denominators → 0.0), `entity_f05` (empty-vs-empty = 1.0, the "
            "challenge's singleton rule) and `macro_f05` (average over every S1 entity in the truth set).", "",
            _table(["check", "macro F0.5", "micro TP/FP/FN"], [
                ["README example (expected 0.714)", fv["readme_example_expected_0_714"], "-"],
                ["perfect prediction on sampled entities", round(fv["perfect_prediction_on_sample"]["macro_f05"], 4),
                 "/".join(str(fv["perfect_prediction_on_sample"]["micro"][k]) for k in ("tp", "fp", "fn"))],
                ["all-empty prediction on sampled entities", round(fv["all_empty_prediction_on_sample"]["macro_f05"], 4),
                 "/".join(str(fv["all_empty_prediction_on_sample"]["micro"][k]) for k in ("tp", "fp", "fn"))],
            ]), "", "The all-empty score equals the zero-match share: the floor any matcher must beat.", "",
            "## 10. Normalization (phase1/src/normalization.py)", "",
            "- `normalize_name`: NFKC, casefold, strip Latin diacritics only, `&`→`and`, punctuation→space, "
            "collapse whitespace.",
            "- `normalize_address`: same without the `&` rule; line breaks become spaces.",
            "- `normalize_country`: NFKC, trim, collapse whitespace, casefold; no code/name mapping.",
            "- Not applied: abbreviation expansion, legal-suffix removal, transliteration (Phase 2 decisions).", "",
            "## 11. Scope", "",
            "Phase 1 does NOT implement candidate generation, final matching, model training, test prediction, "
            "or submission generation.", ""]
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, {"data_root": args.data_root, "output_dir": args.output_dir,
                                    "max_rows": args.max_rows})

    data_root = (stage_from_s3(cfg["data_root"], _local(cfg["local_cache_dir"]))
                 if str(cfg["data_root"]).startswith("s3://") else _local(cfg["data_root"]))
    stats = _clean_json(run(data_root, cfg))

    out_dir = _local(cfg["output_dir"])
    report, artifact = out_dir / "reports" / "phase1_report.md", out_dir / "artifacts" / "dataset_statistics.json"
    for path in (report, artifact):
        path.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(stats, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False),
                        encoding="utf-8")
    report.write_text(render_report(stats), encoding="utf-8")
    log.info("wrote %s and %s", report, artifact)
    if cfg.get("output_s3_uri"):
        upload_to_s3([report, artifact], out_dir, cfg["output_s3_uri"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
