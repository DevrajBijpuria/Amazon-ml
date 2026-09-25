# Phase 1 — Data Understanding

Profiles the TRAIN data of the Business Entity Resolution challenge, analyses name /
address / country variation, ground truth, match difficulty and singletons, provides
the F0.5 metric and conservative normalization functions, and writes a report.

**Phase 1 does NOT implement candidate generation, final matching, model training,
test prediction, or submission generation.** Test files are read for header and row
count only.

## Layout

```
phase1/
├── README.md
├── requirements.txt
├── config/phase1_config.yaml   seed, sample sizes, data/output locations
├── src/
│   ├── run_phase1.py           orchestration, report rendering, optional S3 in/out
│   ├── profiling.py            discovery, schema check, loading, missing/unique/length/duplicate profile
│   ├── analysis.py             name/address/country variation, ground truth, singletons,
│   │                           pair-level match difficulty, similarity helpers, F0.5
│   └── normalization.py        normalize_name / normalize_address / normalize_country
├── tests/test_phase1.py
├── reports/phase1_report.md            (generated)
└── artifacts/dataset_statistics.json   (generated, byte-identical across runs)
```

## Run

From the repository root (Python ≥ 3.11):

```bash
pip install -r phase1/requirements.txt
python phase1/src/run_phase1.py                # full TRAIN data, ~15 min, ~3 GB RAM peak
python phase1/src/run_phase1.py --max-rows 100000   # quick look
python -m pytest phase1/tests -q
```

Options: `--config`, `--data-root` (local path or `s3://bucket/prefix`), `--output-dir`,
`--max-rows`. Environment variables `PHASE1_DATA_ROOT`, `PHASE1_OUTPUT_DIR`,
`PHASE1_OUTPUT_S3_URI` override the YAML.

## What is computed

| Area | Where | Data used |
|---|---|---|
| discovery, schema check, row counts | `profiling.discover_datasets/check_schema` | train + test headers |
| rows/cols/dtypes, missing, whitespace-only, unique %, lengths, ID uniqueness, exact/ID-less/name/address/name+address duplicates | `profiling.profile_frame` | full train files |
| name & address pattern rates overall and per country, raw legal-suffix and street-type spellings, comma counts | `analysis.analyze_text_variation` | 200k-row fixed-seed sample per source |
| country distribution, variants, odd values | `analysis.analyze_country` | full train files |
| ground-truth match distributions, S2/S3 composition, duplicate/ambiguous links, unreferenced S2/S3 records | `analysis.analyze_ground_truth`, `target_coverage` | full |
| zero / one / multi-match comparison | `analysis.analyze_singletons` | full S1 (+50k sample per bucket for regex rates) |
| match difficulty: raw/normalized equality, token order, token & char-trigram Jaccard, edit distance | `analysis.pair_features` | all GT pairs of 5,000 sampled S1 entities |
| positive vs negative | `analysis.sample_negatives`, `summarize_pairs` | 1 same-country random negative per sampled S1 |
| F0.5 (β = 0.5), entity-level challenge rule, macro average | `analysis.f_beta/entity_f05/macro_f05` | metric checks |

Sampling uses `seed` from the config everywhere (`pandas.sample(random_state=seed)`,
`random.Random(seed)`).

**Negative-pair assumption.** Train ground truth lists every S1 id exactly once and
links each S2/S3 id to at most one S1 id, so a pair absent from it is treated as a
non-match. Negatives are uniform random (easy) — they show separability, not the
hard-negative problem, which needs candidate generation (Phase 2).

**Normalization** is deliberately conservative: NFKC, casefold, Latin-only diacritic
stripping (Devanagari marks kept), punctuation → space, whitespace collapse, and
`&` → `and` for names. No abbreviation expansion, suffix removal, transliteration or
country mapping.

## Dependencies

| Package | Why |
|---|---|
| pandas | reading the multi-GB TSVs, profiling, group/duplicate statistics |
| PyYAML | reading `config/phase1_config.yaml` |
| boto3 | S3 input staging / output upload; imported only when an `s3://` URI is configured |
| pytest | tests |

Everything else is the standard library (`re`, `unicodedata`, `collections`, `random`,
`json`, `csv`, `math`, `logging`, `argparse`, `pathlib`, `os`, `sys`).
numpy is only an indirect dependency of pandas and is not imported. pandas 3 uses
pyarrow for its string dtype *if it is installed* (lower memory); it is not imported
here and not required. scipy / scikit-learn are not needed in Phase 1.

## AWS mapping

| Concern | Service |
|---|---|
| raw data + outputs | S3: `s3://<bucket>/raw/{train,test}/*.tsv`, outputs to `s3://<bucket>/phase1/` via `output_s3_uri` |
| execution | SageMaker Processing job (scikit-learn/pandas container, `ml.m5.2xlarge`, 32 GB) with the S3 input mounted at `/opt/ml/processing/input` → `PHASE1_DATA_ROOT=/opt/ml/processing/input`; or an AWS Glue Python-shell job with `PHASE1_DATA_ROOT=s3://...` (boto3 staging) |
| logging | stdout logging → CloudWatch Logs automatically in both runtimes |
| ad-hoc SQL | optional Glue Data Catalog table + Athena over the raw TSVs (`ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'`, `skip.header.line.count=1`) |
| credentials | IAM execution role only; nothing hard-coded. Least privilege: `s3:GetObject`/`s3:ListBucket` on the raw prefix, `s3:PutObject` on the output prefix, CloudWatch `logs:CreateLogStream`/`PutLogEvents` |

No infrastructure is created by the code. The same code runs locally unchanged.
