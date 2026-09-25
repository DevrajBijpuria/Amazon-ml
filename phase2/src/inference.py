"""Scoring, multi-match assembly, submission writing and submission checks (spec 25, 45)."""
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import threshold_tuning as tt

ID_RE = re.compile(r"^S[23]-\d+$")
LIST_RE = re.compile(r"^S[23]-\d+(,S[23]-\d+)*$")


def score(model, feats: pd.DataFrame, features: list) -> np.ndarray:
    """P(match | features), using the exact training column order."""
    if len(feats) == 0:  # a chunk whose S1 entities got no candidates
        return np.zeros(0)
    return model.predict_proba(feats[features].to_numpy(np.float64))[:, 1]


def assemble(s1_ids: list, pairs: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """One row per S1 id (input order kept); target ids deduplicated, sorted, comma-joined, '' if none."""
    grouped = (pairs.drop_duplicates(["s1", "target"]).sort_values(["s1", "target"])
               .groupby("s1")["target"].agg(",".join))
    return pd.DataFrame({"source1_entity_id": s1_ids,
                         id_col: [grouped.get(s, "") for s in s1_ids]})


def predictions(s1_ids: list, scored: pd.DataFrame, threshold: float, empty_target_address_threshold=None) -> pd.DataFrame:
    """Every candidate accepted by ``tt.accept`` becomes a match (zero, one or many per S1)."""
    return assemble(s1_ids, scored[tt.accept_frame(scored, threshold, empty_target_address_threshold)],
                    "matched_entity_ids")


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    """Plain TSV: header + one line per row, no quoting (ids never contain tabs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(df.columns) + "\n")
        for row in df.itertuples(index=False):
            f.write("\t".join(row) + "\n")


def chunk_lines(lo: int, hi: int, s1_idx: np.ndarray, targets: np.ndarray, scores: np.ndarray,
                threshold: float, target_address_missing: np.ndarray = None, empty_target_address_threshold=None) -> list:
    """For S1 positions lo..hi-1: 'candidate_ids<TAB>matched_ids' with sorted, comma-joined ids
    (every candidate is listed; matches are the candidates accepted by ``tt.accept``)."""
    cols = {"i": s1_idx, "t": targets, "s": scores}
    if empty_target_address_threshold is not None and target_address_missing is not None:
        cols["m"] = target_address_missing  # absent while the policy is enabled -> tt.accept raises
    df = pd.DataFrame(cols).sort_values(["i", "t"])
    cand = df.groupby("i")["t"].agg(",".join)  # the candidate list never depends on the decision rule
    ok = tt.accept(df["s"].to_numpy(), threshold, df["m"].to_numpy() if "m" in df else None, empty_target_address_threshold)
    match = df[ok].groupby("i")["t"].agg(",".join)
    return [f"{cand.get(i, '')}\t{match.get(i, '')}\n" for i in range(lo, hi)]


def merge_and_check(s1_ids: list, part_paths: list, m_path: Path, c_path: Path, valid_ids: set = None) -> dict:
    """Stream the per-source part files (one line per S1, same order) into matching_results.tsv and
    candidate_pairs.tsv, checking section 45 and correction 3 on every row.

    S2 ids sort before S3 ids, so concatenating each source's sorted list keeps lists sorted."""
    ok = dict.fromkeys(["1_every_s1_represented", "3_no_malformed_ids", "4_no_duplicate_ids_in_a_list",
                        "5_multi_match_serialization", "6_zero_match_serialization",
                        "7_deterministic_sorted_lists", "matches_subset_of_candidates"], True)
    ok["2_ids_exist_in_s2_s3"] = True if valid_ids is not None else None  # None = left to the validator
    n = {"rows": 0, "with_prediction": 0, "with_multiple": 0, "predictions": 0, "candidate_pairs": 0}
    parts = [open(p, encoding="utf-8") for p in part_paths]
    with open(m_path, "w", encoding="utf-8", newline="\n") as fm, open(c_path, "w", encoding="utf-8", newline="\n") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in s1_ids:
            fields = [f.readline().rstrip("\n").split("\t") for f in parts]
            if any(len(x) != 2 for x in fields):
                ok["1_every_s1_represented"] = False
                break
            cand = ",".join(x[0] for x in fields if x[0])
            match = ",".join(x[1] for x in fields if x[1])
            fm.write(f"{s1}\t{match}\n")
            fc.write(f"{s1}\t{cand}\n")
            c_list, m_list = cand.split(",") if cand else [], match.split(",") if match else []
            n["rows"] += 1
            n["candidate_pairs"] += len(c_list)
            n["predictions"] += len(m_list)
            n["with_prediction"] += bool(m_list)
            n["with_multiple"] += len(m_list) > 1
            if match and not LIST_RE.match(match):
                ok["5_multi_match_serialization"] = ok["3_no_malformed_ids"] = False
            if not match and match != "":
                ok["6_zero_match_serialization"] = False
            if cand and not LIST_RE.match(cand):
                ok["3_no_malformed_ids"] = False
            if len(set(c_list)) != len(c_list) or len(set(m_list)) != len(m_list):
                ok["4_no_duplicate_ids_in_a_list"] = False
            if c_list != sorted(c_list) or m_list != sorted(m_list):
                ok["7_deterministic_sorted_lists"] = False
            if not set(m_list) <= set(c_list):
                ok["matches_subset_of_candidates"] = False
            if valid_ids is not None and not all(i in valid_ids for i in c_list):
                ok["2_ids_exist_in_s2_s3"] = False
        if any(f.readline() for f in parts):  # a part file with more rows than S1 entities
            ok["1_every_s1_represented"] = False
    for f in parts:
        f.close()
    ok["8_row_count"] = n["rows"] == len(s1_ids)
    with open(m_path, encoding="utf-8") as fm, open(c_path, encoding="utf-8") as fc:
        ok["9_header"] = (fm.readline() == "source1_entity_id\tmatched_entity_ids\n"
                          and fc.readline() == "source1_entity_id\tcandidate_entity_ids\n")
    return {"checks": ok, "counts": n}


def run_validator_sharded(validator: Path, matching: Path, candidate: Path, test_dir: Path, shard_rows: int,
                          work_dir: Path) -> dict:
    """Run the unchanged challenge validator (with --check-ids) on consecutive Source-1 shards.

    Both output files list the Source-1 ids in test_source1.tsv order, so shard k of each file is
    validated against shard k of test_source1.tsv; test_source2/3.tsv are hard-linked into every
    shard directory. Any row in the wrong shard fails that shard ("missing" / "not in the test set").
    """
    files = [open(p, encoding="utf-8") for p in (test_dir / "test_source1.tsv", matching, candidate)]
    headers = [f.readline() for f in files]
    shards, k = [], 0
    while True:
        lines = [[f.readline() for _ in range(shard_rows)] for f in files]
        lines = [[x for x in part if x] for part in lines]
        if not any(lines):
            break
        d = work_dir / f"shard_{k:03d}"
        d.mkdir(parents=True, exist_ok=True)
        for head, part, name in zip(headers, lines, ("test_source1.tsv", "matching_results.tsv", "candidate_pairs.tsv")):
            (d / name).write_text(head + "".join(part), encoding="utf-8", newline="\n")
        for name in ("test_source2.tsv", "test_source3.tsv"):
            if not (d / name).exists():
                os.link(test_dir / name, d / name)
        res = run_validator(validator, d / "matching_results.tsv", d / "candidate_pairs.tsv", d, check_ids=True)
        shards.append({"shard": k, "rows": [len(x) for x in lines], "passed": res["passed"], "output": res["output"]})
        for p in d.iterdir():
            p.unlink()
        d.rmdir()
        k += 1
    for f in files:
        f.close()
    return {"command": "validate_submission.py --check-ids per shard", "shard_rows": shard_rows,
            "shards": len(shards), "passed": bool(shards) and all(s["passed"] for s in shards),
            "failed_shards": [s for s in shards if not s["passed"]],
            "output": shards[-1]["output"] if shards else ["no shards"]}


def run_validator(validator: Path, matching: Path, candidate, test_dir: Path, check_ids: bool) -> dict:
    """Run the challenge's utils/validate_submission.py unchanged."""
    cmd = [sys.executable, str(validator), "--matching", str(matching), "--test-dir", str(test_dir)]
    cmd += ["--candidate", str(candidate)] if candidate else ["--candidate", str(matching.parent / "_none_.tsv")]
    cmd += ["--check-ids"] if check_ids else []
    # the validator prints non-ASCII (em dashes); force UTF-8 so Windows' cp1252 console encoding can't break decoding
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return {"command": " ".join(Path(c).name if i < 2 else c for i, c in enumerate(cmd)),
            "returncode": proc.returncode, "passed": proc.returncode == 0,
            "output": proc.stdout.strip().splitlines()[-12:]}
