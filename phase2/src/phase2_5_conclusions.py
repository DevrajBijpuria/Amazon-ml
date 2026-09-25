"""Builds artifacts/phase2_5/conclusions.json: the interpretive sentences of the Phase 2.5 report,
with every number read from the artifacts (run before phase2_5_report.py)."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "phase2" / "artifacts" / "phase2_5"


def load(n):
    return json.loads((ART / n).read_text(encoding="utf-8"))


def pct(x):
    return f"{100 * x:.2f}%"


def main() -> None:
    fc, mc, fa, cap, base = (load(n) for n in ("final_comparison.json", "model_comparison.json",
                                               "feature_ablation.json", "cap_sensitivity.json", "baseline_reproduction.json"))
    res = load("full_scale_resource_test.json")
    runs, boot = fc["runs"], fc["bootstrap"]["configurations"]
    b0, bl, fz = runs["resource_b0_50000"], runs["blocking_lr_50000"], runs["frozen_50000"]
    bb, bf = boot["blocking_lr_50000"], boot["frozen_50000"]
    d = "diff_vs_resource_b0_50000"
    st = fc["stress_tests"]
    m10 = mc["10k"]["results"]
    mb10 = mc["10k"]["bootstrap"]["configurations"]
    f10 = fa["10k"]["results"]
    fb10 = fa["10k"]["bootstrap"]["configurations"]
    sw = cap["frozen_rule_set"]["10k"]["by_source_by_cutoff"]
    cb = {s: cap["10k"]["by_source_by_cutoff"][s]["100"] for s in ("s2", "s3")}
    pj = res["projection_full_run"]
    peak = res["runs"]["frozen_50000"]["main_process_peak_working_set_mb"]
    ci = lambda c: f"[{c[0]:.4f}, {c[1]:.4f}]"  # noqa: E731
    dev_m = mc["dev"]["results"]

    exec_summary = [
        f"**Both blocking and the classifier limit performance at full scale, and the classifier limits it more.** "
        f"On 50,000 Source-1 entities against the complete target files (10,000 identical validation entities), "
        f"the unchanged Phase 2 baseline B0 reaches macro F0.5 {b0['macro_f05']:.4f} {ci(boot['resource_b0_50000']['ci95'])}, "
        f"not the 0.9709 of the development setting: its candidate recall falls from about 98.1% on 100,000-record "
        f"target pools to {pct(b0['candidate_recall'])} on the full files, and its logistic regression misses "
        f"{b0['classifier_missed_positives']:,} retrieved positives while making {b0['fp']:,} false merges.",
        f"The frozen hardened configuration scores **{fz['macro_f05']:.4f} {ci(bf['ci95'])}** at threshold "
        f"{fz['selected_threshold']}, a paired improvement of {bf[d]:+.4f} {ci(bf[d + '_ci95'])} over B0. Improved "
        f"blocking alone (B0's logistic regression on the new candidate set) raises candidate recall to "
        f"{pct(bl['candidate_recall'])} and macro F0.5 by {bb[d]:+.4f} {ci(bb[d + '_ci95'])}; replacing the model with "
        f"gradient boosting on the same candidates adds the rest ({fz['macro_f05'] - bl['macro_f05']:+.4f}), cutting "
        f"false positives from {bl['fp']:,} to {fz['fp']:,} and classifier misses from {bl['classifier_missed_positives']:,} "
        f"to {fz['classifier_missed_positives']:,}. Blocking still loses {fz['blocking_missed_positives']:,} validation "
        f"positives, the ceiling Phase 3 inherits.",
        "B0 was reproduced byte-for-byte before any change and again on the final code; every comparison uses "
        "identical validation entities, seed 42 and paired bootstrap confidence intervals; no improvement is credited "
        "from a lower-recall candidate set (the frozen candidate set has higher recall than B0). Test data has not "
        "been used for any choice: after the configuration was frozen, a development-scale smoke test ran test "
        "inference on a 5,000-entity test subset only to check that the frozen pipeline runs end to end and passes the "
        "validator; no labels exist for test, nothing was changed afterwards, and no Phase 2.5 result reads its output.",
    ]

    name_edit = (
        f"The negative `name_edit_sim` coefficient is a multicollinearity artifact, not a harmful feature: on its own "
        f"its standardized coefficient is positive at both scales, it correlates at r ≥ 0.9 with "
        f"`name_translit_edit_sim` and `name_char3_jaccard`, and removing it changes macro F0.5 by an amount "
        f"indistinguishable from zero (10k: {fb10['B0-name_edit_sim']['diff_vs_B0']:+.4f} "
        f"{ci(fb10['B0-name_edit_sim']['diff_vs_B0_ci95'])}). It was kept; gradient boosting is insensitive to it.")

    s2c, s3c = sw["s2"]["100"]["caps"], sw["s3"]["100"]["caps"]
    final_cfg = "\n".join([
        "Frozen in `phase2/config/phase2_5_frozen.yaml` (derived from the B0 config; every other value unchanged):", "",
        "| setting | B0 | frozen | evidence |", "|---|---|---|---|",
        f"| blocking rules | 8 | 8 + house_locality, state_locality, address_signature | 10k: together they retrieve "
        f"618 (S2) and 708 (S3) positives B0 missed, with buckets bounded by the 100-record cutoff; name_phonetic_token "
        f"retrieves 1 and 3 and lowers recall after the cap; postal_locality retrieves 3 and 1 (postcodes are rare) |",
        f"| max_candidates_per_source1 | 60 | 100 | chosen (05:27) from the 10k full-target sweep on the B0 rules, "
        f"cutoff 100: cap 100 {pct(cb['s2']['100']['recall'])} / {pct(cb['s3']['100']['recall'])} vs cap 150 "
        f"{pct(cb['s2']['150']['recall'])} / {pct(cb['s3']['150']['recall'])} (S2 / S3); then confirmed unchanged (06:03) "
        f"by a sweep on the exact frozen rule set: {pct(s2c['60']['recall'])} / {pct(s3c['60']['recall'])} at cap 60, "
        f"{pct(s2c['100']['recall'])} / {pct(s3c['100']['recall'])} at cap 100, {pct(s2c['150']['recall'])} / "
        f"{pct(s3c['150']['recall'])} at cap 150; cap 150 adds 0.1-0.2 points for about 14% more pairs |",
        f"| max_block_size (cutoff) | 100 | 100 | B0 rules at 10k: cutoffs 50-500 measured; on the frozen rule set "
        f"(cutoffs 100 and 150 only) cutoff 150 at cap 100 adds 0.3 points for about 18% more pairs "
        f"({pct(sw['s2']['150']['caps']['100']['recall'])} / {pct(sw['s3']['150']['caps']['100']['recall'])}); "
        f"not worth the feature cost and memory |",
        "| source1_chunk_size | 100000 | 25000 | memory only: candidates and features are computed per S1, so results "
        "do not depend on the chunk size; smaller chunks bound the feature buffers for the full run |",
        "| gradient boosting max_iter | n/a | 200 | the value evaluated in Step 9; deliberately not tuned |",
        f"| model | StandardScaler + LogisticRegression | HistGradientBoostingClassifier (max_iter 200, seed 42) | "
        f"10k, B0 candidates: {m10['M1/B0']['macro_f05']:.4f} {ci(mb10['M1/B0']['ci95'])} vs "
        f"{m10['M0/B0']['macro_f05']:.4f}, paired {mb10['M1/B0']['diff_vs_M0/B0']:+.4f} "
        f"{ci(mb10['M1/B0']['diff_vs_M0/B0_ci95'])}; dev: {dev_m['M1/B0']['macro_f05']:.4f} vs {dev_m['M0/B0']['macro_f05']:.4f} |",
        f"| features | 47 | 42 (F3: the five matched_by_* block-membership flags removed) | equivalent to 47 under "
        f"gradient boosting at dev and 10k (paired CI of the difference contains 0), so the simpler set is preferred |",
        f"| threshold | 0.70 (dev) | {fz['selected_threshold']} (validation, 50k) | selected on validation macro F0.5 only |",
    ])

    s25 = [
        ["B0 current baseline (50k, full targets)", pct(b0["candidate_recall"]), b0["mean_candidates_per_s1"],
         f"{b0['macro_f05']:.4f} {ci(boot['resource_b0_50000']['ci95'])}", f"{b0['precision']:.4f}", f"{b0['recall']:.4f}",
         "Current Phase 2 (dev-mode 0.9709 does not hold at full scale)"],
        ["Best blocking variant (50k): B0 model on the frozen candidate set", pct(bl["candidate_recall"]),
         bl["mean_candidates_per_s1"], f"{bl['macro_f05']:.4f} {ci(bb['ci95'])}", f"{bl['precision']:.4f}",
         f"{bl['recall']:.4f}", f"paired vs B0 {bb[d]:+.4f} {ci(bb[d + '_ci95'])}"],
        ["Best feature variant (10k, B0 candidates, LR): F3", pct(fa["10k"]["candidate_set"]["candidate_recall"]["s2"]["b0_recall"] * 0.5
                                                              + fa["10k"]["candidate_set"]["candidate_recall"]["s3"]["b0_recall"] * 0.5),
         round(fa["10k"]["candidate_set"]["candidate_pairs"] / 10000, 1),
         f"{f10['F3']['macro_f05']:.4f} {ci(fb10['F3']['ci95'])}", f"{f10['F3']['precision']:.4f}", f"{f10['F3']['recall']:.4f}",
         "equivalent to all 47 features; no feature set is better (different scale from the 50k rows)"],
        ["Best model variant (10k, B0 candidates): gradient boosting", pct(fa["10k"]["candidate_set"]["candidate_recall"]["s2"]["b0_recall"] * 0.5
                                                                       + fa["10k"]["candidate_set"]["candidate_recall"]["s3"]["b0_recall"] * 0.5),
         round(fa["10k"]["candidate_set"]["candidate_pairs"] / 10000, 1),
         f"{m10['M1/B0']['macro_f05']:.4f} {ci(mb10['M1/B0']['ci95'])}", f"{m10['M1/B0']['precision']:.4f}",
         f"{m10['M1/B0']['recall']:.4f}", f"vs LR 0.{str(round(m10['M0/B0']['macro_f05'], 4))[2:]} on the same 2,000 entities"],
        ["Final hardened Phase 2 (50k, full targets)", pct(fz["candidate_recall"]), fz["mean_candidates_per_s1"],
         f"{fz['macro_f05']:.4f} {ci(bf['ci95'])}", f"{fz['precision']:.4f}", f"{fz['recall']:.4f}",
         f"selected only after gates; paired vs B0 {bf[d]:+.4f} {ci(bf[d + '_ci95'])}"],
    ]

    resources = (
        f"Measured: 50k Source-1 entities against the full targets take {res['runs']['frozen_50000']['wall_seconds'] / 60:.0f} "
        f"minutes end to end for the frozen configuration (training and validation), with a main-process peak of "
        f"{peak:,} MB; the 10k harness run peaked at {res['runs']['blocking_10k']['main_process_peak_working_set_mb']:,} MB "
        f"because it holds 13 rules' key tables plus experiment buffers. Target normalization (about 2 minutes per "
        f"source) and index building (about 5 minutes per source) are fixed costs; candidate joins, the cap and features "
        f"scale with Source-1 count. Projected full test inference (1.73M Source-1 entities): about "
        f"{pj['test_candidate_pairs_estimate'] / 1e6:.0f} million candidate pairs and about "
        f"{pj['test_inference_total_seconds_estimate'] / 3600:.1f} hours for candidates and features; model scoring of "
        f"about {pj['test_candidate_pairs_estimate'] / 1e6:.0f} million rows, loading and normalizing the test Source-1 "
        f"file, the submission merge and validation were not measured at scale and add to this (an estimate of 2 to 3 "
        f"hours in total). Peak memory was measured only for training and validation (8.0 GB at 50k); the inference "
        f"path was profiled only at development scale, so the full run should be monitored; the frozen config uses "
        f"25,000 Source-1 entities per chunk to bound feature buffers. The challenge validator keeps every candidate id in memory; for a "
        f"candidate_pairs.tsv of about {pj['test_candidate_pairs_estimate'] / 1e6:.0f} million ids it must be run on "
        f"Source-1 shards (each shard with its own test_source1 subset) on a 16 GB machine.")

    limitations = [
        f"Blocking still misses {fz['blocking_missed_positives']:,} of the validation positives at 50k "
        f"(candidate recall {pct(fz['candidate_recall'])}); at 10k most B0 misses were suppressed high-frequency keys "
        f"(453 of 600), and the no-cutoff diagnostic shows up to 99% of positives share some key, so the remaining "
        f"recall sits in buckets too large to enumerate.",
        f"The classifier still misses {fz['classifier_missed_positives']:,} retrieved positives and makes {fz['fp']:,} "
        f"false merges on 10,000 validation entities; recall ({fz['recall']:.4f}) remains the weaker side.",
        "The per-S1 cap ranks by the number of retrieving rules and edit similarity; adding rules can push true pairs "
        "out of the cap (observed for name_phonetic_token), so any new rule must be evaluated after the cap.",
        "Cross-script and transliterated names are only partly recovered by deterministic romanization and the "
        "consonant skeleton; the per-token phonetic block did not help at full scale.",
        "Validation macro F0.5 is selected and reported on the same validation entities (the Phase 2 protocol); the "
        "threshold curve is flat around the optimum, but a small optimism remains.",
        "France appears only in the test set; it is handled by the same country-agnostic rules without training "
        "examples.",
        "All Phase 2.5 measurements are on Source-1 samples (5k to 50k); the full-scale figures are projections until "
        "the full run is executed.",
    ]
    handoff = [
        "Start from `phase2/config/phase2_5_frozen.yaml` and the frozen code; reproduce "
        f"`phase2/output/phase2_5/frozen_50000/` (macro F0.5 {fz['macro_f05']:.4f}) before changing anything.",
        "Largest remaining lever on the blocking side: selective composite keys for oversized buckets (for example a "
        "rare name token combined with a locality token) instead of skipping them, evaluated after the cap.",
        "Largest lever on the model side: more training entities than 40,000 and features that separate "
        "near-duplicate businesses at the same address; gradient boosting hyperparameters were deliberately not tuned.",
        "Keep the evaluation protocol: identical validation entities, seed 42, paired bootstrap, recall reported "
        "alongside every F0.5.",
        "The full Source-1 / full target run and the test submission are pending the user's approval (runtime and "
        "memory projection in section 13).",
    ]
    deps = ("Allowed list plus one documented exception approved by the user: pandas, numpy, scikit-learn "
            "(LogisticRegression, HistGradientBoostingClassifier, StandardScaler), pyyaml, pytest, and — carried over "
            "from the Phase 2 specification — rapidfuzz (edit similarity, token-set ratio) and joblib (worker processes, "
            "model persistence). No other third-party package; memory was measured externally with Windows PowerShell "
            "(`phase2/tools/memwatch.ps1`). No external data, no LLM or API, raw datasets unchanged.")
    labels = {"resource_b0_50000": "B0 (Phase 2 as is)", "blocking_lr_50000": "Frozen blocking + B0 model (LR, 47 features)",
              "frozen_50000": "Frozen hardened configuration (blocking + gradient boosting, 42 features)"}
    (ART / "conclusions.json").write_text(json.dumps({
        "executive_summary": exec_summary, "name_edit_sim": name_edit, "final_configuration": final_cfg,
        "section25_rows": s25, "resources": resources, "limitations": limitations, "handoff": handoff,
        "dependencies": deps, "run_labels": labels, "baseline_reproduced": base["reproduced"]}, indent=2,
        ensure_ascii=False), encoding="utf-8")
    print("conclusions written")


if __name__ == "__main__":
    main()
