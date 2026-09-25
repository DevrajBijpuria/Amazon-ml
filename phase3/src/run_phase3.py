"""Phase 3 entry point.

    python phase3/src/run_phase3.py --config phase3/config/p3_locality_cap100.yaml --mode dev --s1-limit 1000
    python phase3/src/run_phase3.py --config <config> --mode scale10k | scale50k [--repeat 2]
    python phase3/src/run_phase3.py --refresh-metrics              # recompute metrics.json of every finished run
    python phase3/src/run_phase3.py --report                       # rebuild phase3/reports/phase3_report.md
    python phase3/src/run_phase3.py --release <50k run dir>        # freeze the selected run into phase3/release/
    python phase3/src/run_phase3.py --mode production              # full test inference from phase3/release/

Every experiment/scale/repeat writes only phase3/output/<experiment_id>/<scale>/run_<k>/. The
production mode is the only one that reads test data; run it once, after every gate has passed.
"""
import argparse
import json
import logging
import shutil
import stat
import sys
import time
from pathlib import Path

import yaml

import aws_runtime
import experiment_runner as er
import metrics

REPO = er.REPO
RELEASE = REPO / "phase3" / "release"
rp = er.rp


def release(run_dir: Path) -> None:
    """Copy the selected run's config, model, threshold and statistics into phase3/release/ and make it read-only."""
    if RELEASE.exists():
        raise SystemExit(f"{RELEASE} exists; a release is frozen once")
    art = RELEASE / "artifacts"
    shutil.copytree(run_dir / "artifacts" / "model", art / "model")
    for name in ("threshold_results.json", "candidate_statistics.json"):
        shutil.copy2(run_dir / "artifacts" / name, art / name)
    shutil.copy2(run_dir / "config.yaml", RELEASE / "config.yaml")
    shutil.copy2(run_dir / "manifest.json", RELEASE / "source_run_manifest.json")
    files = sorted(p for p in RELEASE.rglob("*") if p.is_file())
    rp._json(RELEASE / "release_hashes.json", {"source_run": run_dir.relative_to(REPO).as_posix(),
                                               "sha256": {p.relative_to(RELEASE).as_posix(): aws_runtime.sha256(p) for p in files}})
    for p in files + [RELEASE / "release_hashes.json"]:
        p.chmod(stat.S_IREAD)


def production() -> None:
    """Full test inference with the released model and threshold (unchanged Phase 2 test stage)."""
    cfg = yaml.safe_load((RELEASE / "config.yaml").read_text(encoding="utf-8"))
    cfg.update(mode="full", development={"enabled": False, "max_source1_rows": None, "max_target_rows": None})
    cfg["paths"] = {**cfg["paths"], "phase2_dir": RELEASE.relative_to(REPO).as_posix(), "artifacts_read_only": True}
    out = {"artifacts": RELEASE / "artifacts", "output": RELEASE / "output" / "full"}
    timer, sampler, t0 = rp.Timer(), aws_runtime.MemorySampler(), time.perf_counter()
    sampler.start()
    rp.test_stage(cfg, timer, out)
    timer["total"] = round(time.perf_counter() - t0, 1)
    rp._json(out["output"] / "production_run.json", {"runtime_seconds": dict(timer), "memory": sampler.stop(),
                                                     "environment": aws_runtime.environment(),
                                                     "code_sha256": er.code_sha256(), "git_commit": er.git_commit()})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 3 blocking experiments")
    ap.add_argument("--config")
    ap.add_argument("--mode", choices=["dev", "scale10k", "scale50k", "production"])
    ap.add_argument("--s1-limit", type=int, default=1000, help="dev mode: training S1 sample size")
    ap.add_argument("--repeat", type=int, default=1, help="run index; k > 1 is compared by hash with runs 1..k-1")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--release")
    ap.add_argument("--refresh-metrics", action="store_true",
                    help="recompute every finished run's metrics.json from its artifacts (e.g. after its baseline finished)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")
    if args.refresh_metrics:
        for man in sorted((REPO / "phase3" / "output").glob("*/*/run_*/manifest.json")):
            cfg = yaml.safe_load((man.parent / "config.yaml").read_text(encoding="utf-8"))
            rp._json(man.parent / "metrics.json", metrics.run_metrics(man.parent, REPO / cfg["paths"]["baseline_dir"]))
    elif args.report:
        import reports
        reports.write_report()
    elif args.release:
        release(REPO / args.release)
    elif args.mode == "production":
        production()
    else:
        run_dir = er.run(REPO / args.config, args.mode, args.s1_limit, args.repeat)
        if args.repeat > 1:
            check = metrics.repeat_check([run_dir.parent / f"run_{k}" for k in range(1, args.repeat + 1)])
            rp._json(run_dir / "repeat_check.json", check)
            logging.info("repeat outputs identical by hash: %s", check["all_outputs_identical"])
        m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        logging.info("done %s: combined recall %.5f, macro F0.5 %.4f, validator %s", run_dir.relative_to(REPO),
                     m["blocking"]["combined"]["final_recall"], m["model"]["macro_f05"], m["checks"]["validator_passed"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
