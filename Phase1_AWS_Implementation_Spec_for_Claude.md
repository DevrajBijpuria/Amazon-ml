# Amazon ML Challenge 2026 — Phase 1 Implementation Specification
## Claude Code / Claude Implementation Brief

> **Purpose:** Give Claude a concrete, execution-ready specification for implementing Phase 1 of the entity-matching pipeline.
>
> **Infrastructure constraint:** This is an AWS ML Challenge hackathon. All cloud/storage/compute/ML components must use AWS services. Local execution may be used for development/testing, but the intended implementation and deployable architecture must be AWS-native.

---

## 1. Objective

Implement **Phase 1: Data Understanding, EDA, Data Quality Analysis, and Validation Setup** for the entity-resolution/entity-matching challenge.

Phase 1 must produce a reproducible understanding of the three source datasets and establish the evaluation/validation foundation required before candidate generation and ML matching.

Do **not** jump directly to the final matching model.

The Phase 1 implementation must answer:

1. What data do we have?
2. What are the schema, cardinality, and missingness characteristics?
3. What entity/name/address variations and corruption patterns exist?
4. What does the existing ground truth look like?
5. How difficult is the matching problem?
6. What normalization and blocking decisions are justified by the data?
7. How will precision-heavy F0.5 validation be performed?
8. What artifacts should Phase 2 consume?

---

# 2. Non-negotiable implementation rules

### AWS-first architecture

Use AWS services for the actual pipeline:

- **Amazon S3** — raw, processed, reports, artifacts
- **AWS Glue Data Catalog / Glue jobs** — dataset cataloging and scalable preprocessing where appropriate
- **Amazon Athena** — SQL-based profiling and exploratory analysis where appropriate
- **Amazon SageMaker** — ML experimentation/training/evaluation when ML is introduced
- **Amazon CloudWatch** — logs/monitoring
- **AWS IAM** — least-privilege permissions
- **AWS Step Functions** — orchestration if orchestration is required
- **AWS Lambda** — lightweight orchestration/utility tasks only
- **Amazon ECR** — container images if custom processing containers are needed

Do not introduce GCP, Azure, or non-AWS managed infrastructure.

For libraries, standard Python packages such as pandas, numpy, scikit-learn, rapidfuzz, pyarrow, etc. may be used inside AWS jobs/containers where appropriate.

### Reproducibility

Every Phase 1 result must be reproducible from the raw input.

Do not manually edit datasets.

Record:
- input S3 locations
- processing timestamp
- code version/commit if available
- configuration
- output locations
- dataset row counts
- schema/version information

### Data safety

Do not modify the raw data.

Recommended S3 layout:

```text
s3://<bucket>/
  raw/
    source1/
    source2/
    source3/

  phase1/
    profiling/
    normalized_samples/
    validation/
    reports/
    logs/

  phase2/
  phase3/
  submissions/
```

Use configuration/environment variables for bucket names and paths. Never hard-code credentials.

---

# 3. Inputs

There are three source datasets referred to throughout the challenge:

- **Source 1**
- **Source 2**
- **Source 3**

The exact filenames/columns must be discovered from the supplied challenge files/configuration. Do not invent schema fields.

Before processing, inspect:

- file format
- delimiter
- encoding
- headers
- column names
- data types
- primary/entity identifiers
- name fields
- address fields
- country fields
- ground-truth/match fields if present

If the actual schema differs from assumptions, adapt the implementation to the real schema and document the mapping.

---

# 4. Phase 1 deliverables

Create:

```text
phase1/
├── README.md
├── requirements.txt
├── config/
│   └── phase1_config.yaml
├── src/
│   ├── profile_datasets.py
│   ├── analyze_quality.py
│   ├── analyze_ground_truth.py
│   ├── analyze_variations.py
│   ├── validation.py
│   └── run_phase1.py
├── tests/
│   ├── test_profiling.py
│   ├── test_normalization.py
│   └── test_validation.py
├── reports/
│   └── phase1_report.md
└── infrastructure/
    └── README.md
```

If using AWS CDK/Terraform/CloudFormation, put the infrastructure implementation under:

```text
infrastructure/
```

and document deployment.

---

# 5. Step 1 — Dataset discovery

Implement an automated discovery step.

For every source dataset report:

```text
dataset_name
file_count
row_count
column_count
column_names
data_types
file_size
encoding
null_count_by_column
unique_count_by_column
sample_rows
```

Do not dump sensitive/full datasets into logs.

The report should use representative samples.

---

# 6. Step 2 — Schema profiling

For every relevant column calculate:

### General

- dtype
- number of records
- non-null count
- null count
- null percentage
- unique count
- unique percentage

### String fields

- minimum length
- maximum length
- average length
- median length
- whitespace-only count
- punctuation-heavy count
- numeric-only count
- alphanumeric count

### Entity identifiers

Check:

- uniqueness
- nulls
- duplicate IDs
- ID format consistency

Do not assume an ID is unique simply because it is named `id`.

---

# 7. Step 3 — Data quality analysis

Analyze the following explicitly.

## Missing data

For every matching-relevant field:

```text
field
missing_count
missing_percentage
```

Identify records missing:

- entity name
- address
- country
- other matching-relevant attributes

Also report combinations, e.g.:

```text
name missing + address present
name present + address missing
name missing + address missing
```

## Duplicates

Identify:

- exact duplicate rows
- duplicate entity IDs
- exact duplicate names
- exact duplicate addresses
- repeated name/address combinations

Do not automatically delete duplicates.

The purpose of Phase 1 is analysis.

## Formatting inconsistencies

Measure examples such as:

```text
Ltd
Limited
LTD.
Pvt Ltd
Private Limited
Inc
Incorporated
Corp
Corporation
```

Only report variants actually observed in the data.

---

# 8. Step 4 — Name variation analysis

Analyze how entity names vary across the sources.

Examples to detect if present:

```text
case differences
punctuation differences
whitespace differences
legal suffix differences
abbreviations
token reordering
duplicate tokens
Unicode differences
accent/diacritic differences
transliteration
OCR-like corruption
minor spelling differences
```

Create summary statistics such as:

```text
raw_name
normalized_name
transformation_type
frequency
```

Do not over-normalize at this stage.

Normalization must preserve enough information for later matching.

Implement normalization as a deterministic function so that:

```text
raw -> normalized
```

is reproducible.

Keep both versions.

---

# 9. Step 5 — Address variation analysis

Perform the same analysis for addresses.

Investigate, where present:

- punctuation differences
- whitespace differences
- line-break differences
- abbreviations
- postal-code formatting
- state/city variations
- missing components
- component reordering
- numeric formatting
- Unicode/transliteration differences

Do not assume a particular country-specific address format unless it is present in the supplied data.

Keep:

```text
raw_address
normalized_address
```

separately.

---

# 10. Step 6 — Country analysis

For each source calculate:

```text
country
record_count
percentage
```

Identify:

- missing country
- country-code inconsistencies
- spelling variants
- unexpected values

Do not silently map values to a canonical country without documenting the mapping.

Country may later be used as a blocking feature, so the quality of this field must be quantified.

---

# 11. Step 7 — Ground-truth analysis

If the supplied training data contains known matches, analyze the existing ground truth.

For Source 1 entities calculate:

```text
number with zero known matches
number with exactly one match
number with multiple matches
```

Where applicable, also calculate the distribution of:

```text
Source 1 -> Source 2 matches
Source 1 -> Source 3 matches
```

Report match-density statistics.

Important:

- Do not fabricate labels.
- Do not infer a ground-truth match solely from string similarity.
- Use only the challenge-provided labels/relationships for ground-truth statistics.

If the data format does not expose ground truth directly, document that fact and identify the actual available supervision signal.

---

# 12. Step 8 — Match difficulty analysis

Use the ground truth, where available, to quantify how difficult matching is.

For known matched pairs, calculate descriptive distributions for:

### Name

- exact match rate
- normalized exact match rate
- character similarity
- token similarity
- edit distance

### Address

- normalized exact match rate
- character similarity
- token similarity
- edit distance

### Country

- exact agreement rate

Use these only as analysis/features.

Do not build the final matcher in Phase 1.

The objective is to understand which signals are informative.

---

# 13. Step 9 — Negative/random-pair analysis

Where computationally feasible, sample non-matching pairs from the available entities.

Compare positive and negative pair distributions.

For example:

```text
feature                  positive mean    negative mean
--------------------------------------------------------
name_similarity
address_similarity
country_match
name_length_difference
address_length_difference
```

Do not create an artificially misleading negative set.

Document the sampling strategy and sample size.

The purpose is to understand whether candidate-generation/matching features appear separable.

---

# 14. Step 10 — F0.5 validation implementation

Implement the challenge evaluation metric as a reusable function.

The evaluation must explicitly calculate:

```text
precision
recall
F0.5
```

with:

```text
F0.5 = (1 + 0.5^2) * precision * recall /
       ((0.5^2 * precision) + recall)
```

Handle zero-denominator cases safely.

The implementation must also report:

```text
true_positives
false_positives
false_negatives
precision
recall
f0.5
```

Because the challenge metric is precision-heavy, validation must make false-positive behavior visible.

Do not optimize anything yet.

---

# 15. Step 11 — Singleton analysis

Analyze singleton behavior explicitly.

For Source 1 records with exactly one known match:

- count them
- calculate their proportion
- examine whether their names/addresses are systematically different
- compare their similarity distributions with multi-match records

The purpose is to avoid building a strategy that performs well only on obvious repeated entities.

---

# 16. Step 12 — Initial normalization module

Create a conservative normalization module.

It should expose functions similar to:

```python
normalize_name(value)
normalize_address(value)
normalize_country(value)
```

Requirements:

- deterministic
- null-safe
- Unicode-aware
- idempotent where practical
- preserve raw values
- unit tested

Example expectation:

```python
normalize_name(normalize_name(x)) == normalize_name(x)
```

where applicable.

Do not make aggressive semantic substitutions without evidence from the dataset.

---

# 17. Step 13 — Phase 1 report

Generate:

```text
phase1_report.md
```

Structure:

```markdown
# Phase 1 Report

## 1. Executive Summary

## 2. Dataset Overview

## 3. Schema

## 4. Missing Values

## 5. Duplicate Analysis

## 6. Name Variation Analysis

## 7. Address Variation Analysis

## 8. Country Analysis

## 9. Ground Truth Analysis

## 10. Match Difficulty Analysis

## 11. Positive vs Negative Pair Analysis

## 12. Validation / F0.5

## 13. Key Observations

## 14. Implications for Phase 2

## 15. Reproducibility Information
```

The report must contain actual generated numbers, not placeholders.

---

# 18. Phase 1 output JSON

Also generate:

```text
dataset_statistics.json
```

Suggested structure:

```json
{
  "run_metadata": {
    "timestamp": "...",
    "code_version": "...",
    "input_locations": {}
  },
  "datasets": {},
  "missingness": {},
  "duplicates": {},
  "ground_truth": {},
  "variation_analysis": {},
  "validation": {}
}
```

Keep it machine-readable.

---

# 19. AWS architecture

The intended AWS flow is:

```text
Challenge Data
     |
     v
Amazon S3 - raw/
     |
     +----------------------+
     |                      |
     v                      v
AWS Glue Catalog       Athena
     |                      |
     +----------+-----------+
                |
                v
        Phase 1 Processing
       Glue / SageMaker Job
                |
                v
         S3 phase1/
        /    |     \
       /     |      \
      v      v       v
 profiling validation reports
                |
                v
          Phase 2 input
```

Use SageMaker when the work benefits from managed ML experimentation or processing.

Do not force SageMaker into every non-ML task if Glue/Athena/S3 is more appropriate.

---

# 20. IAM requirements

Create/use least-privilege IAM permissions.

The Phase 1 execution role should only access required:

```text
S3 bucket/prefixes
Glue catalog/database
Athena workgroup/output
CloudWatch logs
SageMaker resources if used
```

Do not use:

```text
AdministratorAccess
```

unless explicitly required by the challenge environment and documented.

Never commit AWS access keys.

---

# 21. Configuration

Use:

```yaml
aws:
  region: <configured-region>

s3:
  bucket: <configured-bucket>
  raw_prefix: raw/
  phase1_prefix: phase1/

datasets:
  source1: ...
  source2: ...
  source3: ...

validation:
  random_seed: 42
  negative_sample_size: ...
```

Do not hard-code environment-specific paths in Python.

---

# 22. Logging

Every processing job should log:

```text
job start
input paths
dataset names
row counts
schema
processing stages
output paths
validation metrics
job completion
errors
```

Avoid logging full raw records.

Use structured logging where practical.

---

# 23. Tests

At minimum test:

### Normalization

```text
None/null handling
empty strings
whitespace
Unicode
punctuation
idempotence
```

### Metrics

Test known examples for:

```text
perfect precision/recall
zero true positives
all false positives
all false negatives
mixed results
```

### Profiling

Test that row/column/null counts are correct on a small fixture.

### Reproducibility

Given the same fixture + seed, the same summary statistics should be generated.

---

# 24. Acceptance criteria

Phase 1 is complete only when all of the following are true:

- [ ] All three datasets can be discovered/read from configured locations.
- [ ] Schema profiling is generated automatically.
- [ ] Row counts and column statistics are generated.
- [ ] Missingness analysis is complete.
- [ ] Duplicate analysis is complete.
- [ ] Name variation analysis is complete.
- [ ] Address variation analysis is complete.
- [ ] Country analysis is complete.
- [ ] Ground-truth analysis is complete when labels are available.
- [ ] Positive/negative pair analysis is documented where applicable.
- [ ] F0.5 calculation is implemented and unit-tested.
- [ ] Singleton analysis is included.
- [ ] Conservative normalization functions exist and are tested.
- [ ] `phase1_report.md` is generated.
- [ ] `dataset_statistics.json` is generated.
- [ ] AWS storage/processing architecture is documented.
- [ ] No credentials are committed.
- [ ] Raw data remains unchanged.
- [ ] The implementation can be rerun with configuration only.
- [ ] Phase 2 receives a clear list of recommended candidate-generation/blocking inputs.

---

# 25. Phase 2 handoff

At the end of the report, create a section:

```text
## Phase 2 Handoff
```

It must contain evidence-based observations such as:

```text
1. Which fields appear reliable for blocking?
2. Which fields require normalization?
3. Which fields have high missingness?
4. Which fields appear most discriminative?
5. Which corruption patterns are common?
6. What candidate-generation strategies should be tested?
7. What should NOT be used as a blocking key because of poor quality?
```

Do not choose a final model in Phase 1.

The output should give Phase 2 enough evidence to implement:

```text
candidate generation
blocking
pair construction
feature engineering
matching model
threshold tuning
```

---

# 26. Claude execution instructions

Claude should work in this order:

### Step A
Inspect the repository and identify:

- existing code
- challenge files
- dataset locations
- AWS configuration
- current outputs
- existing Phase 1 work

### Step B
Do not overwrite existing work blindly.

If existing Phase 1 code exists:

1. inspect it
2. identify gaps
3. reuse working components
4. improve them
5. preserve useful outputs

### Step C
Inspect the actual datasets before writing schema-dependent code.

Never invent column names.

### Step D
Implement the profiling/analysis modules.

### Step E
Run them against the real challenge data.

### Step F
Generate the actual Phase 1 report and JSON statistics.

### Step G
Run tests.

### Step H
Review AWS compatibility.

### Step I
Give the user a concise completion report containing:

```text
Implemented:
...

AWS resources:
...

Generated artifacts:
...

Dataset statistics:
...

Important findings:
...

Phase 2 recommendations:
...

Known limitations:
...
```

---

# 27. Important constraint: do not prematurely solve Phase 2

Do NOT implement a final fuzzy matcher, neural model, LLM matcher, or production thresholding strategy as part of Phase 1 unless existing project code already contains it and it is necessary to run the requested analysis.

Phase 1 is for understanding the data and establishing reliable validation.

The next stage should use Phase 1 evidence to design candidate generation and matching.

---

# 28. Definition of done

The implementation is considered successful when a fresh engineer can run the documented command/configuration and obtain:

```text
phase1_report.md
dataset_statistics.json
validation metrics
normalization analysis
data-quality analysis
AWS execution logs/artifacts
```

without manually editing the source data.

The implementation must be suitable as the foundation for the subsequent AWS-native entity-matching pipeline.
