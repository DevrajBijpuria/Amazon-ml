"""One Phase 3 experiment at one scale, isolated under phase3/output/<experiment_id>/<scale>/run_<k>/.

Training S1 sample (seed 42) against the FULL target sources -> blocking -> F3 features -> HGB ->
threshold on validation macro F0.5. That stage is the unchanged Phase 2 ``run_phase2.train_stage``
(the one that produced phase2/output/phase2_5/frozen_50000). Then the validation entities' candidate
and match lists are written in submission format and checked (section 45 checks + the unchanged
challenge validator), and hashes, runtime, memory and metrics are recorded. Test data is never read.
"""
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import aws_runtime
import blocking_phase3
import metrics

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "phase2" / "src"))
import candidate_generation as cg  # noqa: E402
import inference as inf  # noqa: E402
import run_phase2 as rp  # noqa: E402

log = logging.getLogger("phase3")
MODES = {"scale10k": 10000, "scale50k": 50000}  # dev takes --s1-limit
INPUTS = ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv", "train_ground_truth.tsv"]


def scale_label(n: int) -> str:
    return f"{n // 1000}k" if n % 1000 == 0 else str(n)


def run_dir_for(cfg: dict, n_s1: int, repeat: int) -> Path:
    base = REPO / cfg["paths"]["phase3_dir"] / cfg["experiment_id"] / scale_label(n_s1)
    if "model_seed" in cfg:  # P3-N: one directory per (negative_sample_seed, model_seed) pair
        ns, ms = cfg["negative_sample_seed"], cfg["model_seed"]
        return base / (f"seed_{ms}" if ns == ms else f"seed_n{ns}_m{ms}")
    return base / f"run_{repeat}"


def _with_seed(fn, seed):
    """P3-N: call ``fn`` with its trailing ``seed`` argument replaced; S1 sample and validation split keep cfg["seed"]."""
    def wrapped(*a, **k):
        if "seed" in k:
            k["seed"] = seed
        else:
            a = a[:-1] + (seed,)
        return fn(*a, **k)
    return wrapped


def code_sha256() -> str:
    """One hash over every source file a run can execute (Phase 1, Phase 2, Phase 3 except the report writer)."""
    files = sorted(p for d in ("phase1/src", "phase2/src", "phase3/src") for p in (REPO / d).glob("*.py")
                   if p.name != "reports.py")
    return hashlib.sha256("".join(p.relative_to(REPO).as_posix() + aws_runtime.sha256(p) for p in files).encode()).hexdigest()


def git_commit() -> str:
    p = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else f"unavailable ({p.stderr.strip().splitlines()[0]})"


def _timed_per_source(fn, timer, stage):
    """Time each call of ``fn`` as <stage>_<source>; the training stage calls it once per source, S2 then S3."""
    sources = iter(rp.SOURCES)

    def wrapped(*a, **k):
        with timer(f"{stage}_{next(sources)}"):
            return fn(*a, **k)
    return wrapped


def write_validation_submission(art: Path, out_dir: Path, data_root: Path) -> dict:
    """matching_results.tsv + candidate_pairs.tsv for the validation entities: candidates are exactly
    the scored pairs (validation_scores.tsv), matches those scoring >= the selected threshold."""
    th = json.loads((art / "threshold_results.json").read_text(encoding="utf-8"))
    thr, empty_thr = th["selected_threshold"], rp.tt.policy_threshold(th)  # the rule the metrics used
    scores = pd.read_csv(art / "validation_scores.tsv", sep="\t", dtype={"s1": str, "target": str}, keep_default_na=False)
    s1_ids = pd.read_csv(art / "validation_entities.tsv", sep="\t", dtype={"s1": str}, keep_default_na=False)["s1"].tolist()
    pos = pd.Series(np.arange(len(s1_ids)), index=s1_ids)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for src in rp.SOURCES:
        sub = scores[scores["source"] == src]
        part = out_dir / f"_part_{src}.tsv"
        flag = sub[rp.tt.EMPTY_ADDRESS_COLUMN].to_numpy() if empty_thr is not None else None
        part.write_text("".join(inf.chunk_lines(0, len(s1_ids), pos.loc[sub["s1"]].to_numpy(), sub["target"].to_numpy(),
                                                sub["score"].to_numpy(), thr, flag, empty_thr)), encoding="utf-8", newline="\n")
        parts.append(part)
    m_path, c_path = out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv"
    merged = inf.merge_and_check(s1_ids, parts, m_path, c_path)
    for part in parts:
        part.unlink()
    # validator input: the validation S1 rows as test_source1.tsv, the train target sources linked in
    vdir = out_dir / "validator_input"
    s1 = rp.load_tsv(data_root / "train" / "train_source1.tsv")
    inf.write_tsv(s1[s1["entity_id"].isin(set(s1_ids))], vdir / "test_source1.tsv")
    for k in (2, 3):
        os.link(data_root / "train" / f"train_source{k}.tsv", vdir / f"test_source{k}.tsv")
    return {"threshold": thr, "validation_entities": len(s1_ids), "section45_checks": merged["checks"],
            "counts": merged["counts"], "files": [m_path, c_path], "validator_dir": vdir}


def run(config_path: Path, mode: str, s1_limit: int, repeat: int) -> Path:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    n_s1 = s1_limit if mode == "dev" else MODES[mode]
    run_dir = run_dir_for(cfg, n_s1, repeat)
    if (run_dir / "manifest.json").exists():
        raise SystemExit(f"{run_dir} already holds a finished run; experiments are never overwritten")
    run_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    rules = blocking_phase3.rule_set(cfg["candidate_generation"])
    log.info("experiment %s, %d S1 vs full targets, run %d, rules %s", cfg["experiment_id"], n_s1, repeat, rules)
    (run_dir / "config.yaml").write_bytes(config_path.read_bytes())
    code_hash = code_sha256()  # at start: the code that actually runs
    data_root = REPO / cfg["paths"]["data_root"]

    t_start = time.perf_counter()
    sampler = aws_runtime.MemorySampler()
    sampler.start()
    timer = rp.Timer()
    rp_cfg = {**cfg, "mode": "dev", "development": {"enabled": True, "max_source1_rows": n_s1, "max_target_rows": None}}
    out = {"artifacts": run_dir / "artifacts"}
    originals = cg.build_blocking, inf.score
    cg.build_blocking = _timed_per_source(cg.build_blocking, timer, "train_index")

    def timed_score(*a, **k):
        with timer("scoring"):
            return originals[1](*a, **k)
    inf.score = timed_score
    mm_originals = rp.mm.sample_training_pairs, rp.mm.train_model
    if "model_seed" in cfg:  # P3-N amendment B: easy-negative draw and HGB get their own seeds
        rp.mm.sample_training_pairs = _with_seed(mm_originals[0], cfg["negative_sample_seed"])
        rp.mm.train_model = _with_seed(mm_originals[1], cfg["model_seed"])
    try:
        rp.train_stage(rp_cfg, timer, out, "model")
    finally:
        cg.build_blocking, inf.score = originals
        rp.mm.sample_training_pairs, rp.mm.train_model = mm_originals
    with timer("output"):
        sub = write_validation_submission(out["artifacts"], run_dir / "output", data_root)
    with timer("validator"):
        validator = inf.run_validator(REPO / cfg["paths"]["validator"], *sub["files"], sub["validator_dir"], check_ids=True)
    for k in (2, 3):  # drop the links to the multi-GB train files (archives must not copy them)
        (sub["validator_dir"] / f"test_source{k}.tsv").unlink()
    memory = sampler.stop()
    timer["total"] = round(time.perf_counter() - t_start, 1)

    outputs = sub["files"] + [out["artifacts"] / n for n in ("validation_scores.tsv", "candidate_statistics.json")] + [
        out["artifacts"] / "model" / "matcher.joblib"]
    manifest = {
        "experiment_id": cfg["experiment_id"], "scale": scale_label(n_s1), "source1_entities": n_s1,
        "targets": "full train Source 2 and Source 3", "repeat": repeat, "seed": cfg["seed"],
        "negative_sample_seed": cfg.get("negative_sample_seed", cfg["seed"]), "model_seed": cfg.get("model_seed", cfg["seed"]),
        "rules": rules, "candidate_priority": blocking_phase3.CANDIDATE_PRIORITY,
        "git_commit": git_commit(), "code_sha256": code_hash,
        "config_sha256": aws_runtime.sha256(config_path),
        "input_sha256": {n: aws_runtime.sha256(data_root / "train" / n) for n in INPUTS},
        "output_sha256": {p.relative_to(run_dir).as_posix(): aws_runtime.sha256(p) for p in outputs},
        "runtime_seconds": dict(timer), "memory": memory, "environment": aws_runtime.environment(),
        "validation_output": {"threshold": sub["threshold"], "validation_entities": sub["validation_entities"],
                              "section45_checks": sub["section45_checks"], "counts": sub["counts"]},
        "validator": validator,
    }
    rp._json(run_dir / "manifest.json", manifest)
    rp._json(run_dir / "metrics.json", metrics.run_metrics(run_dir, REPO / cfg["paths"]["baseline_dir"]))
    logging.getLogger().removeHandler(handler)
    handler.close()
    return run_dir
