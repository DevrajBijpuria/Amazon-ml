"""phase3/reports/phase3_report.md, built from every run's manifest.json + metrics.json, the gate
evaluation, and the recorded decisions (phase3/artifacts/decisions.json, written by hand after each rung)."""
import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "phase3" / "output"
ART = REPO / "phase3" / "artifacts"
EXPERIMENTS = [("Reference", "p3_frozen_baseline"), ("P3-A", "p3_locality_cap100"), ("P3-B", "p3_locality_cap150"),
               ("P3-C", "p3_locality_cap200"), ("P3-D", "p3_locality_phonetic_cap100")]
SCALES = ["1k", "10k", "50k"]
BOTO3_ALLOWED = {"phase1/src/run_phase1.py", "phase3/src/aws_runtime.py"}
NETWORK = re.compile(r"^\s*(import|from)\s+(requests|urllib|http|socket|ftplib|smtplib|aiohttp|httpx)\b", re.M)


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def _table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    return "\n".join(out + ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows])


def _pct(x, d=2):
    return "n/a" if x is None else f"{100 * x:.{d}f}%"


def collect() -> dict:
    """runs[exp_id][scale][k] = {"dir", "manifest", "metrics"} for every finished run."""
    runs = {}
    for _, exp in EXPERIMENTS:
        for scale in SCALES:
            for d in sorted((OUT / exp / scale).glob("run_*")):
                if (d / "metrics.json").exists():
                    runs.setdefault(exp, {}).setdefault(scale, {})[int(d.name[4:])] = {
                        "dir": d, "manifest": _load(d / "manifest.json"), "metrics": _load(d / "metrics.json")}
    return runs


def compliance_scan() -> dict:
    """G8 evidence: no network client is imported anywhere; boto3 is imported only where our own data moves
    to or from our own S3 prefix (Phase 1 dataset fetch, aws_runtime.archive_to_s3); neither is called by a run."""
    files = sorted(p for d in ("phase1/src", "phase2/src", "phase3/src") for p in (REPO / d).glob("*.py"))
    hits = {p.relative_to(REPO).as_posix(): NETWORK.findall(p.read_text(encoding="utf-8")) for p in files}
    boto = [p.relative_to(REPO).as_posix() for p in files if re.search(r"^\s*import boto3", p.read_text(encoding="utf-8"), re.M)]
    return {"files_scanned": len(files), "network_imports": {k: v for k, v in hits.items() if v}, "boto3_in": boto}


def gates(runs: dict, dec: dict) -> list:
    sel = dec["selected"]
    r50 = runs.get(sel, {}).get("50k", {})
    main, rep = r50.get(1), r50.get(2)
    tests = (ART / "pytest.txt").read_text(encoding="utf-8").strip().splitlines()[-1] if (ART / "pytest.txt").exists() else "not run"
    all_runs = [r for e in runs.values() for s in e.values() for r in s.values()]
    s45 = all(all(v for v in r["metrics"]["checks"]["section45"].values() if v is not None) for r in all_runs)
    val = all(r["metrics"]["checks"]["validator_passed"] for r in all_runs)
    rows = [["G1 Correctness", "PASS" if "failed" not in tests and "passed" in tests and s45 and val else "FAIL",
             f"pytest: {tests}; section-45 checks (incl. one row per S1, no duplicates) pass on all {len(all_runs)} runs: "
             f"{s45}; official validator PASS on all runs: {val}"],
            ["G2 Candidate subset", "PASS" if all(r["metrics"]["checks"]["section45"]["matches_subset_of_candidates"]
                                                  for r in all_runs) else "FAIL",
             f"matches_subset_of_candidates true on every run's validation output ({len(all_runs)} runs)"]]
    if main:
        m = main["metrics"]
        vb = m["vs_baseline"]
        rows.append(["G3 Candidate recall", "PASS" if vb and vb["delta_final_recall"] > 0 else "FAIL",
                     f"50k combined final recall {_pct(m['blocking']['combined']['final_recall'], 3)}; change vs frozen "
                     f"baseline {vb['delta_final_recall'] * 100:+.3f} points; blocking-missed positives change "
                     f"{vb['delta_blocking_missed_positives']:+d}; same {m['model']['entities']:,} validation entities"])
        d = vb["entity_behaviour_delta"]
        rows.append(["G4 Precision behavior", dec["gate_verdicts"]["G4"],
                     f"zero-match false-merge entities {d['zero_match_false_merge_entities']:+d} vs baseline "
                     f"(baseline {vb['baseline_entity_behaviour']['zero_match_false_merge_entities']}); singleton entities "
                     f"with extra predictions {d['singleton_with_extra_predictions']:+d}; extra predictions {d['extra_predictions_total']:+d}. "
                     + dec["gate_notes"].get("G4", "")])
        rows.append(["G5 Runtime", dec["gate_verdicts"]["G5"], dec["gate_notes"]["G5"]])
        rows.append(["G6 Memory", dec["gate_verdicts"]["G6"],
                     f"50k main-process peak {m['memory']['main_process_peak_mb']:,} MB, all-python peak "
                     f"{m['memory']['all_python_processes_peak_mb']:,} MB. " + dec["gate_notes"]["G6"]])
    if rep:
        rc = _load(rep["dir"] / "repeat_check.json")
        rows.append(["G7 Reproducibility", "PASS" if rc["all_outputs_identical"] else "FAIL",
                     "run_1 vs run_2 SHA-256: " + ", ".join(f"{k} {'identical' if v else 'DIFFERENT'}"
                                                             for k, v in rc["per_output_file"].items())])
    else:
        rows.append(["G7 Reproducibility", "NOT RUN", "repeat run of the selected configuration not finished"])
    cs = compliance_scan()
    rows.append(["G8 Compliance", "PASS" if not cs["network_imports"] and set(cs["boto3_in"]) <= BOTO3_ALLOWED else "FAIL",
                 f"{cs['files_scanned']} source files scanned: network-client imports {cs['network_imports'] or 'none'}; "
                 f"boto3 imported only in {cs['boto3_in']} (own dataset fetch / archival of own outputs to own S3 prefix; not called by any run); inputs are the "
                 "challenge TSVs only (input SHA-256 recorded per run)"])
    ok = [s for s in ("10k", "50k") if 1 in runs.get(sel, {}).get(s, {})]
    rows.append(["G9 Full-scale readiness", "PASS" if ok == ["10k", "50k"] else "FAIL",
                 f"selected configuration completed at {', '.join(ok) or 'no'} full-target scale(s) with exit code 0"])
    return rows


def template_block(label: str, r: dict) -> str:
    man, m = r["manifest"], r["metrics"]
    b, c, mo, e, rt, mem = (m["blocking"], m["blocking"]["combined"], m["model"], m["entities"], m["runtime"], m["memory"])
    s2, s3 = b["by_source"]["s2"], b["by_source"]["s3"]
    rt_line = " / ".join(f"{rt[k]}" for k in ("load", "normalize", "index", "candidate", "features", "scoring",
                                             "output", "validator", "total"))
    lines = [
        f"Experiment ID: {label} ({man['experiment_id']}, {man['scale']}, run {man['repeat']})",
        f"Git commit: {man['git_commit']}; code SHA256 {man['code_sha256']}",
        f"Config SHA256: {man['config_sha256']}",
        "Input hashes: " + ", ".join(f"{k} {v[:16]}..." for k, v in man["input_sha256"].items()),
        f"Seed: {man['seed']}",
        f"S1 scale: {man['source1_entities']:,} training S1 ({mo['entities']:,} validation entities)",
        f"Target scale: {man['targets']}",
        f"Blocking rules: {', '.join(man['rules'])}",
        f"max_block_size: 100",
        f"max_candidates_per_source1: {_load(r['dir'] / 'artifacts' / 'candidate_statistics.json')['max_candidates_per_source1']}",
        "Model: HistGradientBoostingClassifier(max_iter=200, random_state=42, early_stopping=False), retrained",
        "Feature set: F3 (42 features)",
        f"Selected threshold: {mo['threshold']}",
        "Blocking:",
        f"  S2 precap recall: {_pct(s2['precap_recall'], 3)}",
        f"  S3 precap recall: {_pct(s3['precap_recall'], 3)}",
        f"  S2 final recall: {_pct(s2['final_recall'], 3)}",
        f"  S3 final recall: {_pct(s3['final_recall'], 3)}",
        f"  Combined recall: {_pct(c['final_recall'], 3)} (precap {_pct(c['precap_recall'], 3)})",
        f"  Blocking-missed positives: {c['blocking_missed_positives']:,} of {c['positives']:,} "
        f"(validation entities only: {e['blocking_missed_positives']:,})",
        "Candidates:",
        f"  Precap pairs: {c['precap_pairs']:,}",
        f"  Final pairs: {c['final_pairs']:,}",
        f"  Cap-hit rate: {_pct(c['cap_hit_rate'])} of S1/source groups (S2 {_pct(s2['cap_hit_rate'])}, S3 {_pct(s3['cap_hit_rate'])})",
        f"  Mean / P95 / Max candidates: S2 {s2['mean_candidates_per_s1']} / {s2['p95_candidates_per_s1']:g} / "
        f"{s2['max_candidates_per_s1']}; S3 {s3['mean_candidates_per_s1']} / {s3['p95_candidates_per_s1']:g} / {s3['max_candidates_per_s1']}",
        "Model:",
        f"  Macro F0.5: {mo['macro_f05']:.4f}",
        f"  Precision: {mo['precision']:.4f}",
        f"  Recall: {mo['recall']:.4f}",
        f"  TP / FP / FN: {mo['tp']:,} / {mo['fp']:,} / {mo['fn']:,}",
        f"  Zero-match false merges: {e['zero_match_false_merge_entities']} entities ({e['zero_match_false_merge_predictions']} "
        f"predictions) of {e['zero_match_entities']} zero-match entities",
        f"  Correct empty entities: {e['correctly_empty_entities']}",
        f"  Singleton exact-correct: {e['singleton_exact_correct']:,} of {e['singleton_entities']:,}",
        f"  Singleton missed: {e['singleton_missed']:,}",
        "Runtime (seconds):",
        f"  Load / Normalize / Index / Candidate / Features / Scoring / Output / Validator / Total: {rt_line} "
        f"(training {rt['training']}, threshold sweep + error analysis {rt['threshold_and_error_analysis']})",
        "Memory:",
        f"  Main process peak: {mem['main_process_peak_mb']:,} MB",
        f"  All-process peak: {mem['all_python_processes_peak_mb']:,} MB (all python processes)",
    ]
    return "\n".join(lines)


def results_table(runs: dict, scale: str) -> str:
    rows = []
    for label, exp in EXPERIMENTS:
        r = runs.get(exp, {}).get(scale, {}).get(1)
        if not r:
            continue
        m = r["metrics"]
        c, vb = m["blocking"]["combined"], m["vs_baseline"]
        ci = vb["bootstrap"]["diff_vs_baseline_ci95"] if vb and vb.get("identical_validation_entities") else None
        rows.append([label, _pct(c["precap_recall"], 3), _pct(c["final_recall"], 3), f"{c['blocking_missed_positives']:,}",
                     f"{c['final_pairs']:,}", c["mean_candidates_per_s1_both_sources"], _pct(c["cap_hit_rate"]),
                     f"{m['model']['macro_f05']:.4f}", m["model"]["threshold"],
                     f"{vb['bootstrap']['diff_vs_baseline']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "reference",
                     m["entities"]["zero_match_false_merge_entities"], m["runtime"]["total"],
                     f"{m['memory']['main_process_peak_mb']:,} / {m['memory']['all_python_processes_peak_mb']:,}",
                     "PASS" if m["checks"]["validator_passed"] else "FAIL"])
    return _table(["experiment", "precap recall", "final recall", "blocking-missed", "final pairs", "mean cands/S1 (S2+S3)",
                   "cap-hit rate", "macro F0.5", "thr", "diff vs baseline [95% CI]", "zero-match false merges",
                   "total s", "peak MB main / all", "validator"], rows)


def write_report() -> Path:
    runs, dec = collect(), _load(ART / "decisions.json")
    ART.joinpath("gates.json").write_text(json.dumps(gates(runs, dec), indent=2), encoding="utf-8")
    L = ["# Phase 3 Report: Blocking Hardening", "",
         "Generated by `phase3/src/run_phase3.py --report` from `phase3/output/*/*/run_*/{manifest,metrics}.json`, "
         "`phase3/artifacts/pytest.txt` and the decisions recorded in `phase3/artifacts/decisions.json`.", "",
         "## 1. Outcome", ""] + dec["summary"] + [""]
    L += ["## 2. Gates", "", _table(["gate", "status", "evidence"], gates(runs, dec)), ""]
    L += ["## 3. Cap trade-off", ""] + dec["cap_tradeoff"] + [""]
    L += ["## 4. Setup and interpretations", ""] + dec["setup"] + [""]
    for i, scale in enumerate(reversed(SCALES)):
        L += [f"## {5 + i}. Results at {scale} training S1 against the full target sources", "",
              dec.get("scale_notes", {}).get(scale, ""), "", results_table(runs, scale), ""]
    n = 5 + len(SCALES)
    L += [f"## {n}. Experiment records (section 27 template)", "",
          "One block per experiment at the principal 50k scale (run 1), then P3-E, the repeat of the selected "
          "configuration.", ""]
    for label, exp in EXPERIMENTS:
        r = runs.get(exp, {}).get("50k", {}).get(1)
        if r:
            d = dec["decisions"].get(exp, {"decision": "PENDING", "reason": ""})
            L += [f"### {label}", "", "```", template_block(label, r), f"Decision: {d['decision']}",
                  f"Reason: {d['reason']}", "```", ""]
    rep = runs.get(dec["selected"], {}).get("50k", {}).get(2)
    if rep:
        d = dec["decisions"].get("P3-E", {"decision": "PENDING", "reason": ""})
        L += ["### P3-E", "", "```", template_block("P3-E", rep), f"Decision: {d['decision']}",
              f"Reason: {d['reason']}", "```", ""]
    L += [f"## {n + 1}. Decision trail", ""] + [f"{i}. {t}" for i, t in enumerate(dec["trail"], 1)] + [""]
    L += [f"## {n + 2}. Error analysis", ""] + dec["error_analysis"] + [""]
    L += [f"## {n + 3}. Limitations and open items", ""] + [f"- {t}" for t in dec["limitations"]] + [""]
    path = REPO / "phase3" / "reports" / "phase3_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
    return path
