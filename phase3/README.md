# Phase 3: blocking and inference hardening

Specification: `docs/phase3_spec.docx`. Report: `reports/phase3_report.md`.

Phase 3 changes only blocking. The rules, the key-frequency cutoff, the union and the per-S1 cap are the
unchanged Phase 2 code (`phase2/src/candidate_generation.py`). The features, the HGB model and the threshold
protocol are the frozen Phase 2.5 ones (`phase2/src/run_phase2.py` `train_stage`). HGB is retrained on each
experiment's own training candidates, and its threshold is re-selected on the same validation entities, so any
difference between experiments comes from blocking. The rollback baseline is
`phase2/output/phase2_5/frozen_50000/`, which Phase 3 only reads.

| config | experiment | rules | cap |
|---|---|---|---|
| `p3_frozen_baseline.yaml` | reference | frozen Phase 2.5 (B0 + house_locality, state_locality, address_signature) | 100 |
| `p3_locality_cap100.yaml` | P3-A | B0 + house_locality, postal_locality, state_locality, address_signature | 100 |
| `p3_locality_cap150.yaml` | P3-B | as P3-A | 150 |
| `p3_locality_cap200.yaml` | P3-C | as P3-A | 200 |
| `p3_locality_phonetic_cap100.yaml` | P3-D | P3-A + name_phonetic_token | 100 |

P3-E is the selected configuration run a second time (`--repeat 2`).

## Run

```
pip install -r phase3/requirements.txt
python -m pytest phase3/tests -q
python phase3/src/run_phase3.py --config phase3/config/p3_locality_cap100.yaml --mode dev --s1-limit 1000   # smoke
python phase3/src/run_phase3.py --config phase3/config/p3_locality_cap100.yaml --mode scale10k
python phase3/src/run_phase3.py --config phase3/config/p3_locality_cap100.yaml --mode scale50k
python phase3/src/run_phase3.py --config <selected config> --mode scale50k --repeat 2    # hash comparison with run_1
python phase3/src/run_phase3.py --report
python phase3/src/run_phase3.py --release phase3/output/<experiment>/50k/run_1         # freeze the selection
python phase3/src/run_phase3.py --mode production                                     # only after every gate + go-ahead
```

Set `PYTHONIOENCODING=utf-8` on Windows. Run one job at a time: a full-target run peaks at about 8 to 10 GB.

## Layout of one run

`output/<experiment_id>/<1k|10k|50k>/run_<k>/`:
- `config.yaml` is the exact config used.
- `run.log` is the log.
- `manifest.json` holds the rules, candidate priority, git/code/config/input/output SHA-256, stage runtimes,
  memory peaks, environment, section-45 checks and the validator result.
- `metrics.json` holds the blocking, candidate-volume, model, entity-behaviour, runtime and memory metrics,
  and the paired bootstrap against the baseline.
- `artifacts/` holds the model, the threshold sweep, the candidate statistics, the validation scores and the
  error analysis.
- `output/` holds `matching_results.tsv` and `candidate_pairs.tsv` for the validation entities, plus the
  validator input.

Only training data is used. The validator runs on the validation entities, with the train target sources
standing in for the test ones. Test data is read only by `--mode production`.

## AWS

`src/aws_runtime.py` identifies the host (EC2 or local), measures memory and hashes files. Its
`archive_to_s3` uploads a run directory to its own S3 prefix. It is never called automatically: a bucket and
IAM role must be approved and created first.
