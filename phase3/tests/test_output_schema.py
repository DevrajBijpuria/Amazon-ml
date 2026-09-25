import re

import experiment_runner as er
import inference as inf

ID = re.compile(r"^S[23]-\d+$")


def _lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def test_schema_completeness_prefixes_and_duplicates(mini_run):
    sub = er.write_validation_submission(mini_run["art"], mini_run["run"] / "output", mini_run["data"])
    m, c = (_lines(p) for p in sub["files"])
    assert m[0] == "source1_entity_id\tmatched_entity_ids" and c[0] == "source1_entity_id\tcandidate_entity_ids"
    for lines in (m, c):
        s1 = [l.split("\t")[0] for l in lines[1:]]
        assert s1 == ["S1-1", "S1-2", "S1-3"]  # every validation S1 exactly once
        for l in lines[1:]:
            ids = [x for x in l.split("\t")[1].split(",") if x]
            assert all(ID.match(x) for x in ids) and len(ids) == len(set(ids)) and ids == sorted(ids)
    assert "S1-3\t" in m and "S1-3\t" in c  # no-candidate entity -> empty lists
    assert all(sub["section45_checks"][k] for k in sub["section45_checks"] if sub["section45_checks"][k] is not None)


def test_official_validator_passes_on_smoke_output(mini_run):
    sub = er.write_validation_submission(mini_run["art"], mini_run["run"] / "output", mini_run["data"])
    res = inf.run_validator(er.REPO / "utils" / "validate_submission.py", *sub["files"], sub["validator_dir"], check_ids=True)
    assert res["passed"], res["output"]
    # only the validation S1 rows are required by the validator input
    assert _lines(sub["validator_dir"] / "test_source1.tsv")[1:] == [
        "S1-1\tname 1\taddr\tFrance", "S1-2\tname 2\taddr\tFrance", "S1-3\tname 3\taddr\tFrance"]


def test_validator_rejects_unknown_ids(mini_run):
    sub = er.write_validation_submission(mini_run["art"], mini_run["run"] / "output", mini_run["data"])
    m = sub["files"][0]
    m.write_text(m.read_text(encoding="utf-8").replace("S2-3", "S2-99"), encoding="utf-8")
    assert not inf.run_validator(er.REPO / "utils" / "validate_submission.py", m, None, sub["validator_dir"], check_ids=True)["passed"]
