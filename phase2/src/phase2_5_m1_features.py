"""Phase 2.5 follow-up: paired comparison of the feature sets under M1 (gradient boosting).

    python phase2/src/phase2_5_m1_features.py --scale dev|10k

feature_ablation.json compares feature sets under M0 only; before freezing M1 with a feature set,
this compares M1/F3 (the simpler set selected there) against M1/B0 on the same frozen candidate set,
fit rows and validation entities (paired bootstrap), and stores it in model_comparison.json.
"""
import argparse
from pathlib import Path

import phase2_5_models as pm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", required=True)
    args = ap.parse_args()
    cs = pm.load_candset(pm.REPO / "phase2" / "output" / "phase2_5" / "candsets" / args.scale / "b0")
    sets = pm.feature_sets(cs["names"])
    runs = {name: pm.run_config(cs, sets[fs][1], "M1") for name, fs in (("M1/B0", "B0"), ("M1/F3", "F3"))}
    boot = pm.bootstrap({k: v["f"] for k, v in runs.items()}, "M1/B0", pm.bootstrap_index(len(cs["n_true_val"])))
    path = Path(pm.REPO / "phase2" / "artifacts" / "phase2_5" / "model_comparison.json")
    data = pm.json.loads(path.read_text(encoding="utf-8"))
    data[args.scale]["m1_feature_set_comparison"] = {
        "results": {k: v["result"] for k, v in runs.items()}, "bootstrap": boot,
        "features_F3_drops": sorted(set(sets["B0"][1]) - set(sets["F3"][1]))}
    path.write_text(pm.json.dumps(data, indent=2), encoding="utf-8")
    for k, v in boot["configurations"].items():
        print(args.scale, k, round(v["macro_f05"], 4), v["ci95"], v["diff_vs_M1/B0_ci95"], v["verdict"])


if __name__ == "__main__":
    main()
