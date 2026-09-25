# Things done so far (handoff for a new chat)

Last updated: 2026-09-26 (Phase 3 done, final submission produced). Project root: `C:\Users\dbijp\OneDrive\Desktop\amazon_aws`
(Windows 11, Python 3.14, 16 GB RAM, usually only ~5 GB free). GitHub (public):
https://github.com/DevrajBijpuria/Amazon-ml (branch `main`).

## 1. The task

Amazon ML Challenge 2026, business entity resolution. For every Source-1 (S1) record find all
matching records in Source 2 (S2) and Source 3 (S3). Metric: **macro F0.5 per S1 entity**; an S1
with no true match scores 1.0 for an empty prediction, 0.0 for any prediction. Submission = two
TSVs: `matching_results.tsv` (source1_entity_id, matched_entity_ids) and `candidate_pairs.tsv`
(source1_entity_id, candidate_entity_ids — exactly the candidates the model scored; every match
must be a candidate). Validate with `utils/validate_submission.py` (challenge-provided, unchanged).

Data (`dataset/`, never modified; SHA-256 checked): train S1 2,206,821 / S2 5,034,616 /
S3 5,285,603, ground truth 7,638,365 positive pairs; test S1 1,732,544 / S2 4,887,273 /
S3 5,082,316. Countries: train US + India; test adds France (country is an open set — never map it
to a fixed list). S2/S3 contain Indic scripts (Devanagari, Tamil, Malayalam, …).

## 2. Layout

```
amazon_aws/
├── README.md, Documentation_template.md   challenge files
├── thing_done.md                          this file
├── docs/phase2_spec.md, docs/phase2_5_spec.docx
├── utils/validate_submission.py           challenge validator (unchanged)
├── dataset/                               challenge data (NOT in git)
├── phase1/                                done, frozen
│   ├── src/{run_phase1,profiling,analysis,normalization}.py, tests/, config/
│   ├── reports/phase1_report.md, artifacts/dataset_statistics.json
└── phase2/
    ├── src/  candidate_generation.py  feature_engineering.py  matching_model.py
    │         threshold_tuning.py  inference.py  run_phase2.py        (Phase 2 pipeline)
    │         phase2_5.py  phase2_5_models.py  phase2_5_m1_features.py
    │         phase2_5_conclusions.py  phase2_5_report.py             (Phase 2.5 harness)
    ├── config/phase2_config.yaml          B0 (Phase 2 baseline) config
    ├── config/phase2_5_frozen.yaml        FROZEN hardened config (use this)
    ├── tests/test_phase2.py, test_phase2_5_models.py   (34 tests, all pass)
    ├── tools/  memwatch.ps1, run_measured.ps1 (runtime + peak memory), run_p25_*.ps1
    ├── artifacts/                         B0 dev artifacts (do not overwrite)
    ├── artifacts/phase2_5/                ALL Phase 2.5 experiment JSONs
    ├── reports/phase2_report.md, reports/phase2_5_report.md
    └── output/phase2_5/<run>/             per-run artifacts; frozen model in frozen_50000/
└── phase3/
    ├── src/  run_phase3.py (entry point)  experiment_runner.py  blocking_phase3.py  metrics.py
    │         reports.py  aws_runtime.py
    ├── config/  p3_frozen_baseline.yaml (reference)  p3_locality_cap{100,150,200}.yaml
    │            p3_locality_phonetic_cap100.yaml  p3_noise_seed{42,1,2,3,4}.yaml
    │            p3_name_addr_composite.yaml  p3_empty_address_policy.yaml (SELECTED)
    ├── tests/   (34 tests; 68 with phase2/tests, all pass)
    ├── output/<experiment>/<scale>/run_1/   per-run artifacts; _invalid/ = contaminated/aborted runs
    ├── output/p3_noise/                   retraining-noise runs + partial report
    └── release/                           FROZEN final release (read-only) + output/full/ submission
```

## 3. Phase 1 (done, frozen — do not modify)

Data profiling, quality, name/address/country variation, ground-truth analysis, match difficulty,
F0.5 helper, conservative normalization (`phase1/src/normalization.py`, reused by Phase 2).
Dependency rule for Phase 1: stdlib + pandas/numpy/scipy/sklearn/pytest/pyyaml/boto3 only.

## 4. Phase 2 baseline "B0" (done)

Pipeline: normalization (Phase 1 rules + legal-suffix canonicalization, Indic romanization,
consonant-skeleton "phonetic" key, US/Indian state names in English and native script → codes,
house/postal/state extraction) → blocking S1→S2 and S1→S3 separately (8 rules, every key prefixed
by exact normalized country, keys with > 100 target records skipped and recorded, per-S1 cap 60)
→ 47 pair features → StandardScaler + LogisticRegression → threshold chosen on validation macro
F0.5 → streamed multi-match output + section-45 checks + validator.
User's 4 corrections: open-set country; threshold chosen on macro F0.5 per S1; two-TSV submission
with candidates = exactly what was scored; cap and every skipped key recorded with recall cost.

B0 dev result (5,000 S1 vs 100,000 targets per source): candidate recall S2 98.10% / S3 98.16%,
validation macro F0.5 **0.9709** at threshold 0.70. Not representative of full scale (see 5).

## 5. Phase 2.5 hardening (done) — report: `phase2/reports/phase2_5_report.md`

- B0 reproduced byte-identically 3 times (incl. on the final code).
- At full target scale B0 is much worse: 50k S1 vs full targets → macro F0.5 **0.8772**
  [0.8727, 0.8813], candidate recall 91.25%. Both blocking and the classifier limit performance;
  the classifier more.
- Frozen configuration (`phase2/config/phase2_5_frozen.yaml`):
  - blocking: B0's 8 rules + `house_locality`, `state_locality`, `address_signature`
    (11 rules); cutoff 100; cap **100** per S1 per source; S1 chunk 25,000 (memory only)
  - model: HistGradientBoostingClassifier(max_iter 200, random_state 42, early_stopping False)
  - features: F3 = 42 features (the 5 `matched_by_*` flags excluded)
  - threshold **0.8** (chosen on validation)
- Frozen result, 50k S1 vs full targets, same 10,000 validation entities as B0:
  macro F0.5 **0.9309** [0.9279, 0.9340], paired gain vs B0 +0.0537 [0.0497, 0.0577];
  precision 0.9705, recall 0.8706, candidate recall 95.25%, 124.5 candidates per S1.
  Blocking-only (B0 model on the new candidates): 0.8839.
- Experiment files: `phase2/artifacts/phase2_5/` (baseline_reproduction, blocking_miss_analysis,
  blocking_ablation, cap_sensitivity, key_cutoff_sensitivity, transliteration_blocking,
  feature_ablation, model_comparison, threshold_robustness, full_scale_resource_test,
  final_comparison, conclusions). Most have "dev" and "10k" sections.
- Dependency exception approved by the user: **rapidfuzz** and **joblib** are allowed (inherited
  from the Phase 2 spec). Otherwise only pandas, numpy, scipy, scikit-learn, pytest, pyyaml, boto3.

## 6. Full test inference (done by the user, 09:26)

`python phase2/src/run_phase2.py --config phase2/config/phase2_5_frozen.yaml --mode full --test-only`
- Uses the frozen model + threshold 0.8 from `phase2/output/phase2_5/frozen_50000/artifacts/`
  (`artifacts_read_only: true` refuses any training run with this config).
- Outputs: `phase2/output/phase2_5/frozen_50000/output/full/matching_results.tsv` (96 MB) and
  `candidate_pairs.tsv` (2.85 GB). 1,732,544 S1 rows; 219,681,953 candidate pairs;
  1,625,744 S1 with ≥1 match. All section-45 checks pass; challenge validator PASS on all
  18 shards (sharded because the validator needs ~20 GB for the whole candidate file).

## 7. Phase 3 (done) — final submission produced 2026-09-26 01:52

Phase 3 runner: `python phase3/src/run_phase3.py --config <cfg> --mode dev --s1-limit 1000 | scale10k | scale50k`.
Each run writes `phase3/output/<experiment_id>/<scale>/run_1/` (never overwrites a finished run).
Reference `p3_frozen_baseline` at 50k reproduces Phase 2.5 `frozen_50000` byte-for-byte
(model, scores, thresholds; chunk-size independent).

Experiments and verdicts (50k = 10,000 validation entities, paired bootstrap vs frozen 0.9309):
- **Locality blocking (P3-A cap100, = frozen + postal_locality):** +0.0012 [−0.0002, +0.0027];
  pair-level: 0 of 768 validation flips came from new candidates (all retraining) → not pursued.
- **Cap 150 (P3-B)**: −0.0029 (worse; run contaminated by a duplicate queue → `_invalid/`).
  Cap 200 at 10k significantly worse. Raising the cap is not supported.
- **Error analysis:** oversized-bucket keys cause ~80% of blocking misses; empty target
  address = 32% of classifier misses and 37% of FPs (FP rate 20× base; not separable by score).
- **P3-N retraining noise (partial, stopped by user):** 50k seed gate reproduced frozen exactly;
  1k (n=3) combined negative-sampling + HGB noise SD 0.0055. 50k noise floor NOT measured.
  Adds optional `negative_sample_seed` / `model_seed` config keys (experiment_runner wrapper).
- **name_addr_composite blocking** (rarest-2 name × rarest-2 address tokens, bucket limit 25):
  10k only: candidate recall 0.9528 → 0.9649, +35 TP / +12 FP, F0.5 +0.0014 [−0.0028, +0.0052];
  peak memory 10.3 GB → 50k not run (memory). Code kept, flag off by default.
- **Empty-target-address policy (SELECTED):** pairs whose target normalized address is empty
  (`address_missing_2`) need score ≥ max(base 0.8, 0.95); others unchanged. One shared decision
  helper `threshold_tuning.accept()` for metrics AND submissions; `sweep()` unchanged.
  50k: **0.9349249447970085** (TP 29858 / FP 599 / FN 4743), +0.0040 [+0.0028, +0.0052].
  Caveat: 0.95 chosen on the same validation set; other models showed +0.001–0.002 (one negative).
  Model byte-identical to frozen; candidate_pairs byte-identical.
- Run `p3_empty_address_policy/50k/run_1` was killed by a (faulty, withdrawn) paging-rate watchdog
  right after its manifest; `metrics.json` was completed afterwards — see its `run_note.json`.

**Final submission** (release of `p3_empty_address_policy/50k/run_1`, `--mode production`, 2 h 10 min,
peak 10.4 GB, validator PASS on 18 shards, all section-45 checks pass):
- `phase3/release/output/full/matching_results.tsv` — sha256 `d67093a1…4707b`; 1,732,544 rows,
  5,621,048 matches, 1,622,504 S1 with ≥ 1 match (a strict subset of the Phase 2.5 matches: −84,056).
- `phase3/release/output/full/candidate_pairs.tsv` (2.85 GB) — sha256 `1cb3b677…2f746`; 219,681,953
  candidates; **byte-identical to the Phase 2.5 full candidate file**.
- Full hashes: `phase3/release/output/full/inference_summary.json`.

## 8. GitHub

Repo: https://github.com/DevrajBijpuria/Amazon-ml (public). Commits `fe60343` (code, configs,
reports, artifacts, frozen model) and `017d5fa` (all run outputs < 100 MB, incl. the full
`matching_results.tsv`), then the Phase 3 commit (code, configs, tests, run outputs, release,
final `matching_results.tsv`). The final 2.85 GB `candidate_pairs.tsv` is attached gzip-compressed
to the GitHub Release `phase3-final-submission` (with SHA256SUMS of the uncompressed files).
Not in git: `dataset/`, `test_subset/` copies of test data, the two full `candidate_pairs.tsv`
files and the 10k `features.npy` (> 100 MB limit).
**Important:** the local git repository is rooted at `C:\Users\dbijp` (home folder), not the
project. Push by cloning the GitHub repo into a temp folder, copying files, committing there.

## 9. How to run

```
python -m pytest phase2/tests -q                                   # 34 tests
python phase2/src/run_phase2.py --mode dev                         # B0 dev pipeline
python phase2/src/run_phase2.py --config phase2/config/phase2_5_frozen.yaml --mode dev --test-only
python phase2/src/run_phase2.py --config phase2/config/phase2_5_frozen.yaml --mode full --test-only
python phase2/src/phase2_5_conclusions.py && python phase2/src/phase2_5_report.py   # rebuild report
powershell -File phase2/tools/run_measured.ps1 -Name <label> -PyArgs "<python args>"  # time + memory
python -m pytest phase3/tests phase2/tests -q                      # 68 tests
python phase3/src/run_phase3.py --config phase3/config/p3_empty_address_policy.yaml --mode scale50k
python phase3/src/run_phase3.py --release phase3/output/p3_empty_address_policy/50k/run_1   # once only
python phase3/src/run_phase3.py --mode production                  # full test inference from release/
```
Set `PYTHONIOENCODING=utf-8` on Windows when printing non-ASCII.

## 10. Gotchas learned

- Memory: full-target runs peak at 7–10 GB; run heavy jobs one at a time; killing a run leaves
  orphan joblib/loky worker processes (kill them).
- Dev-mode numbers (100k-target pools) are optimistic; always check at full target scale.
- pandas 3 routes `str.contains` through pyarrow RE2 — the code uses Python `re` on lists instead.
- The validator prints an em dash; subprocess output must be decoded as UTF-8.
- The per-S1 cap ranks by number of retrieving rules, so adding a blocking rule can push true
  pairs out of the cap — always measure recall after the cap.
- **Never run two Phase 3 jobs at once, and never leave an auto-chaining queue/watcher**: a duplicate
  queue contaminated P3-B and auto-started P3-C. One manual launch at a time; check
  `run_phase3` process count = 0 first.
- Close Chrome/VS Code before 50k or production runs: production peaked at 10.4 GB and free RAM
  briefly hit 63 MB on this 16 GB machine.
- Memory watchdog: use only sustained free-RAM / commit-headroom limits. "Pages Input/sec" counts
  memory-mapped file reads (validator) and falsely killed a finished run.
- HGB `random_state` only affects bin subsampling above 200k training rows (no effect at 1k/10k).
- Changing `seed` also changes the S1 sample and validation entities; use `model_seed` /
  `negative_sample_seed` to vary only training randomness.

## 11. Not done / next

- Leaderboard upload of `phase3/release/output/full/matching_results.tsv` (+ candidate file) is the
  user's action.
- Final submission zip (`output/` + `code/business_entity_resolution/` + filled
  `Documentation_template.md`) not assembled yet.
- Open: 50k retraining-noise floor; 50k test of name_addr_composite (needs a lower-memory index);
  cross-script blocking (Indic names = 25% of blocking misses); leading-zero house numbers.
- `phase3/reports/phase3_report.md` sections 1/3/5/8–11 still "Pending" (generated before the 50k runs).
