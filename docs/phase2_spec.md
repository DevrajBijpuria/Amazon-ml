# Phase 2 --- Entity Matching & Candidate Generation

## Amazon ML Challenge 2026 --- Detailed Implementation Specification

**Status:** Implementation specification\
**Phase:** 2\
**Prerequisite:** Phase 1 completed and frozen\
**Primary objective:** Build a scalable entity-resolution system that
generates candidate matches between Source 1 and Sources 2/3, computes
robust similarity features, trains/calibrates a match scorer, validates
it with the competition's F0.5 metric, and produces a submission-ready
prediction pipeline.

------------------------------------------------------------------------

## 1. Purpose of Phase 2

Phase 1 established the structure, quality, noise patterns, and matching
difficulty of the data. Phase 2 converts those findings into an actual
entity-matching system.

The Phase 2 system must:

1.  Read Source 1, Source 2, Source 3, and training ground truth.
2.  Normalize names and addresses consistently.
3.  Generate a high-recall candidate set without comparing every
    possible pair.
4.  Generate candidate pairs separately for S1→S2 and S1→S3.
5.  Compute deterministic similarity features for each candidate.
6.  Train a binary match-scoring model using the provided ground truth.
7.  Calibrate/select thresholds for the competition's F0.5 objective.
8.  Support zero, one, or multiple target matches per Source-1 entity.
9.  Validate candidate recall before trusting the classifier.
10. Run the same pipeline on test data.
11. Produce the exact submission format required by the challenge.

Phase 2 is therefore **not** a simple fuzzy-matching script. It is a
staged entity-resolution pipeline:

`Normalization → Blocking → Candidate Generation → Feature Engineering → Scoring → Thresholding → Multi-match Assembly → Submission`

------------------------------------------------------------------------

# 2. Phase 1 findings that Phase 2 must use

The Phase 1 output reports the following important properties.

### Dataset sizes

-   Train Source 1: **2,206,821**
-   Train Source 2: **5,034,616**
-   Train Source 3: **5,285,603**
-   Ground-truth Source-1 entities: **2,206,821**
-   Ground-truth positive pairs: **7,638,365**

### Ground-truth match composition

  Category            S1 entities
  ----------------- -------------
  No target match         123,247
  S2 only                 143,029
  S3 only                 164,498
  S2 + S3               1,776,047

The system must therefore support multiple target IDs rather than
forcing a single match.

### Match multiplicity

-   Zero matches: 123,247
-   Exactly one match: 119,157
-   Multiple matches: 1,964,417

The final inference logic must not assume one-to-one matching.

### Country

The only observed countries are:

-   US
-   India

Country equality is present for the sampled positive pairs at 100%.

Country must therefore be used as a strong blocking constraint and
feature.

### Exact matching is insufficient

On the Phase 1 positive sample:

-   normalized name equality recall ≈ **25.6%**
-   normalized address equality recall ≈ **8.4%**
-   normalized name AND address equality recall ≈ **1.5%**

Therefore Phase 2 must use fuzzy/approximate signals.

### Positive-pair similarity

Positive pairs have substantially higher:

-   name character n-gram similarity
-   name token similarity
-   name edit similarity
-   address character n-gram similarity
-   address token similarity
-   address edit similarity

than sampled negatives.

### Source differences

S2 and S3 contain different formatting/noise patterns.

S2 has substantial uppercase and non-ASCII variation, including
Devanagari.

S3 also contains non-ASCII and Devanagari variation but has different
address formatting characteristics.

Consequently:

-   maintain shared normalization principles;
-   allow source-specific feature behavior;
-   train/evaluate S2 and S3 separately or include target-source
    indicators.

### Hard positives

Phase 1 contains examples where:

-   legal names become domains;
-   spelling is corrupted;
-   address abbreviations differ;
-   addresses contain `<NULL>`;
-   tokens are reordered;
-   Indian-language forms appear in target records;
-   names can be nearly unrelated lexically while the address is highly
    informative.

The feature system must therefore be multi-signal rather than relying on
one field.

------------------------------------------------------------------------

# 3. Required Phase 2 architecture

The implementation should be organized as follows:

``` text
phase2/
├── artifacts/
│   ├── candidate_statistics.json
│   ├── feature_statistics.json
│   ├── threshold_results.json
│   └── model/
│
├── config/
│   └── phase2_config.yaml
│
├── reports/
│   └── phase2_report.md
│
├── src/
│   ├── candidate_generation.py
│   ├── feature_engineering.py
│   ├── matching_model.py
│   ├── threshold_tuning.py
│   ├── inference.py
│   └── run_phase2.py
│
└── tests/
    └── test_phase2.py
```

Do not create unrelated helper modules unless they are genuinely
required.

Phase 1 files remain unchanged.

------------------------------------------------------------------------

# 4. Allowed libraries

Use a deliberately restricted dependency set.

Recommended:

``` text
pandas
numpy
scikit-learn
scipy
rapidfuzz
pyyaml
joblib
```

Optional only if already approved by the project environment:

``` text
boto3
```

Do not introduce:

-   deep-learning frameworks;
-   transformer models;
-   external APIs;
-   web scraping;
-   third-party entity-resolution services;
-   paid matching APIs;
-   undocumented packages.

The baseline must remain reproducible and AWS-compatible.

------------------------------------------------------------------------

# 5. Configuration

Create:

``` text
phase2/config/phase2_config.yaml
```

The configuration must contain:

``` yaml
seed: 42

candidate_generation:
  use_country_block: true
  use_exact_name_block: true
  use_exact_address_block: true
  use_name_prefix_block: true
  use_address_prefix_block: true
  use_name_char_ngram_block: true
  max_candidates_per_source1: <configured_limit>

features:
  name:
    char_ngram: true
    token_jaccard: true
    edit_similarity: true
    normalized_exact: true

  address:
    char_ngram: true
    token_jaccard: true
    edit_similarity: true
    normalized_exact: true

  structural:
    country_equal: true
    name_missing: true
    address_missing: true
    house_number_equal: true
    postal_code_equal: true

model:
  type: logistic_regression

threshold:
  default: <validated_value>

validation:
  negative_ratio: <configured_value>
  validation_fraction: 0.2
```

All tunable values must live in YAML rather than being hard-coded.

------------------------------------------------------------------------

# 6. Normalization layer

Phase 2 must use the normalization logic established in Phase 1.

Normalization should be deterministic.

## 6.1 Name normalization

For each business name:

1.  convert to Unicode-safe representation;
2.  normalize Unicode;
3.  lowercase;
4.  replace punctuation with spaces;
5.  normalize whitespace;
6.  normalize obvious legal suffix variants;
7.  normalize common punctuation-separated abbreviations;
8.  preserve meaningful alphanumeric content;
9.  create a token representation;
10. create a compact character representation for n-gram similarity.

Do not aggressively delete words.

For example:

``` text
"ABC Pvt. Ltd."
→
"abc private limited"
```

The original raw name must remain available.

Maintain:

``` text
name_raw
name_norm
name_tokens
name_compact
```

------------------------------------------------------------------------

# 7. Address normalization

Maintain:

``` text
address_raw
address_norm
address_tokens
address_compact
```

Normalization should:

1.  Unicode-normalize;
2.  lowercase;
3.  replace `<NULL>` with missing representation;
4.  normalize punctuation;
5.  normalize whitespace;
6.  standardize common street abbreviations;
7.  preserve numbers;
8.  preserve postal/PIN information;
9.  preserve state/city information;
10. preserve unit/suite information where possible.

Examples of street normalization:

``` text
road → rd
street → st
drive → dr
avenue → ave
lane → ln
court → ct
place → pl
boulevard → blvd
highway → hwy
```

Do not discard house numbers.

------------------------------------------------------------------------

# 8. Country handling

Country must be normalized to a canonical representation:

``` text
US
India
```

Because Phase 1 found country equality on positive pairs to be 100%,
country should be used as:

1.  a candidate-generation blocking key;
2.  a model feature.

A candidate with different countries should normally be excluded from
the candidate set.

Do not use country as the only matching criterion.

------------------------------------------------------------------------

# 9. Candidate generation

This is the most important scalability component.

A full Cartesian comparison would be infeasible:

``` text
2.2M × 5.0M+
```

Therefore candidate generation must reduce the search space.

Candidate generation must optimize **recall first**.

A candidate generator should be considered successful only after
measuring how many known ground-truth positive pairs it recovers.

------------------------------------------------------------------------

# 10. Blocking strategy

Use multiple blocking rules.

Each rule independently proposes candidates.

The final candidate set is the union of candidates generated by all
rules.

## Block A --- Country + exact normalized name

Key:

``` text
country + name_norm
```

This catches strong exact-name matches.

------------------------------------------------------------------------

## Block B --- Country + exact normalized address

Key:

``` text
country + address_norm
```

This catches cases where the address is highly reliable but names
differ.

------------------------------------------------------------------------

## Block C --- Country + name prefix

Use a conservative prefix of the normalized name.

Example:

``` text
country + first N normalized characters
```

Do not use a very short prefix because common names would explode the
candidate count.

------------------------------------------------------------------------

## Block D --- Country + address prefix

Use an address prefix after normalization.

The address prefix should retain useful numeric/address information.

------------------------------------------------------------------------

## Block E --- Character n-gram blocking

Create character n-gram signatures for names.

Use a limited number of representative n-grams or a deterministic
prefix/signature scheme.

The purpose is not to calculate the final similarity here. It is to
retrieve plausible candidates.

------------------------------------------------------------------------

## Block F --- Numeric/address key

Where possible, extract:

-   house number;
-   postal/PIN code.

Use combinations such as:

``` text
country + house_number
country + postal_code
country + house_number + postal_code
```

Only use sufficiently selective keys.

------------------------------------------------------------------------

# 11. Candidate-generation implementation

For each target source:

``` text
S1 → S2
S1 → S3
```

build indexes once.

Do not repeatedly scan the complete target DataFrame for every S1 row.

Recommended conceptual structure:

``` python
index_name[(country, name_norm)] -> target IDs
index_address[(country, address_norm)] -> target IDs
index_house_number[(country, house_number)] -> target IDs
index_postal[(country, postal_code)] -> target IDs
```

Then:

``` python
candidates = union(
    exact_name_candidates,
    exact_address_candidates,
    prefix_candidates,
    numeric_candidates,
    ngram_candidates
)
```

Deduplicate candidate IDs per S1 entity.

------------------------------------------------------------------------

# 12. Candidate recall validation

Before training a classifier, calculate:

``` text
candidate_recall =
ground_truth_positive_pairs_found_in_candidate_set
/
total_ground_truth_positive_pairs
```

Measure this separately for:

-   S2;
-   S3;
-   US;
-   India;
-   one-match S1 entities;
-   multi-match S1 entities.

Also measure:

``` text
mean candidates per S1
median candidates per S1
p95 candidates per S1
p99 candidates per S1
maximum candidates per S1
```

Candidate generation should be judged primarily on whether it loses
known positives.

If a blocking strategy reduces candidate count but loses a significant
number of positives, revise the blocking strategy rather than
immediately accepting the reduction.

------------------------------------------------------------------------

# 13. Candidate explosion protection

Some common names/addresses will generate very large candidate lists.

For every blocking key, record:

``` text
key
frequency
```

If a key occurs excessively often, it must be treated as an
uninformative block and skipped or capped.

Never create millions of candidate pairs because of a generic key such
as:

``` text
private
limited
india
road
```

Candidate generation must use complete normalized fields or selective
composite keys rather than individual common tokens.

------------------------------------------------------------------------

# 14. Feature engineering

For every candidate pair create a deterministic feature vector.

The feature vector should contain four groups:

``` text
name features
address features
structural features
source/context features
```

------------------------------------------------------------------------

# 15. Name features

Required:

### 15.1 Exact normalized equality

``` text
name_norm_eq
```

Binary.

### 15.2 Token Jaccard

``` text
|tokens1 ∩ tokens2|
--------------------
|tokens1 ∪ tokens2|
```

### 15.3 Character n-gram Jaccard

Use character 3-grams or equivalent.

### 15.4 Edit similarity

Use normalized edit similarity.

A useful form is:

``` text
1 - edit_distance / max(len(a), len(b))
```

with safe handling of empty strings.

### 15.5 Length features

Include:

``` text
name_length_1
name_length_2
name_length_ratio
name_length_difference
```

### 15.6 Token-count features

Include:

``` text
name_token_count_1
name_token_count_2
name_token_count_difference
```

### 15.7 Legal suffix behavior

Create indicators for:

``` text
legal_suffix_present_1
legal_suffix_present_2
legal_suffix_equal
```

The system must not treat:

``` text
LLC
Ltd
Limited
Private
Pvt
Inc
Corp
```

as equally informative as the actual business tokens.

------------------------------------------------------------------------

# 16. Address features

Required:

### 16.1 Exact normalized address equality

``` text
address_norm_eq
```

### 16.2 Token Jaccard

``` text
address_token_jaccard
```

### 16.3 Character n-gram Jaccard

``` text
address_char3_jaccard
```

### 16.4 Edit similarity

``` text
address_edit_sim
```

### 16.5 Missing indicators

``` text
address_missing_1
address_missing_2
address_missing_either
```

### 16.6 House-number agreement

Extract numeric house number when available.

``` text
house_number_equal
```

### 16.7 Postal/PIN agreement

``` text
postal_equal
```

### 16.8 State/city agreement

Where deterministic extraction is possible:

``` text
state_equal
city_equal
```

These should be treated as supporting signals, not absolute proof.

------------------------------------------------------------------------

# 17. Structural/context features

Include:

``` text
country_equal
same_source_family
target_source_is_s2
target_source_is_s3
```

Also include candidate provenance:

``` text
matched_by_exact_name_block
matched_by_exact_address_block
matched_by_numeric_block
matched_by_prefix_block
matched_by_ngram_block
number_of_blocks_that_retrieved_pair
```

The last feature is especially useful because a candidate retrieved by
multiple independent blocking rules is often structurally different from
one retrieved by only one weak rule.

------------------------------------------------------------------------

# 18. Missing-data handling

Never replace missing values with misleading strings that can
accidentally create similarity.

For example:

``` text
missing → ""
```

is preferable to treating the literal word:

``` text
"null"
```

as an actual business token.

However, preserve a separate missing indicator.

------------------------------------------------------------------------

# 19. Training-label construction

Ground truth has:

``` text
source1_entity_id
matched_entity_ids
```

Parse `matched_entity_ids` into individual target IDs.

For every generated candidate:

``` text
label = 1
```

if its target ID is in the ground-truth match set.

Otherwise:

``` text
label = 0
```

This creates pair-level training examples.

------------------------------------------------------------------------

# 20. Negative sampling

The candidate generator may produce many more negatives than positives.

Do not train on an enormous arbitrary negative set.

Use controlled negative sampling.

Recommended negative categories:

1.  random same-country candidates;
2.  hard negatives from the same name block;
3.  hard negatives from the same address block;
4.  candidates with high fuzzy similarity but known non-match.

Hard negatives are especially important because the competition metric
strongly penalizes false positives.

Maintain the positive class completely.

Sample negatives deterministically with:

``` text
seed = 42
```

------------------------------------------------------------------------

# 21. Train/validation split

Avoid random row-level leakage where possible.

The split should be at the Source-1 entity level.

For example:

``` text
80% S1 entities → train
20% S1 entities → validation
```

All candidate pairs belonging to a Source-1 entity must remain in the
same partition.

This prevents the same S1 entity from appearing in both train and
validation.

------------------------------------------------------------------------

# 22. Model

Start with a simple, interpretable baseline:

``` text
LogisticRegression
```

with feature scaling if required.

The first model must be reproducible and fast.

Do not immediately introduce a neural network.

The feature engineering and candidate generation are more important at
this stage.

The model output should be:

``` text
P(match | candidate features)
```

For each candidate pair.

------------------------------------------------------------------------

# 23. Model evaluation

Evaluate at two levels.

## Pair level

Calculate:

-   precision;
-   recall;
-   F0.5;
-   confusion matrix.

Because the competition uses F0.5, precision is weighted more strongly
than recall.

## Entity/submission level

For each Source-1 entity:

1.  gather all candidates;
2.  score all candidates;
3.  apply threshold;
4.  emit zero, one, or multiple target IDs.

Then evaluate the final assembled prediction using the challenge metric.

The entity-level evaluation is the important one for deciding the
inference threshold.

------------------------------------------------------------------------

# 24. Threshold tuning

Do not blindly use:

``` text
0.5
```

as the final threshold.

Evaluate a threshold grid, for example:

``` text
0.10
0.15
0.20
...
0.90
```

For each threshold calculate the actual challenge-style F0.5.

Record:

``` text
threshold
TP
FP
FN
precision
recall
F0.5
entities_with_prediction
average_predictions_per_entity
```

The selected threshold must be determined from validation data and saved
to:

``` text
phase2/artifacts/threshold_results.json
```

Do not tune the threshold on the test set.

------------------------------------------------------------------------

# 25. Multi-match assembly

For every S1 entity:

``` text
candidate scores
      ↓
threshold
      ↓
all candidates >= threshold
      ↓
deduplicate target IDs
      ↓
sort deterministically
      ↓
serialize IDs
```

Do not force exactly one prediction.

For example:

``` text
S1-123 → S2-111,S2-222,S3-333
```

if all three are above threshold and supported by the model.

------------------------------------------------------------------------

# 26. S2/S3 handling

Treat S2 and S3 as distinct target populations.

The pipeline should conceptually run:

``` text
S1 → S2
S1 → S3
```

independently.

This allows:

-   different candidate statistics;
-   different score distributions;
-   different thresholds if validation demonstrates the need;
-   source-specific error analysis.

At minimum, include a target-source indicator in the model.

If separate models are trained, save:

``` text
model_s2
model_s3
```

and their corresponding thresholds.

------------------------------------------------------------------------

# 27. Special handling for India

Phase 1 shows meaningful non-ASCII and Devanagari content, especially in
S2/S3.

The system must:

-   remain Unicode-safe;
-   never ASCII-strip names/addresses as the primary representation;
-   retain Unicode-normalized text;
-   use character-level similarity;
-   preserve transliterated/Unicode forms as separate signals.

Do not assume English-only text.

------------------------------------------------------------------------

# 28. Special handling for US

US records contain strong address structure such as:

-   state codes;
-   street types;
-   house numbers.

These should be exploited through structured features.

However, state equality alone is not sufficient for matching.

------------------------------------------------------------------------

# 29. Hard-positive testing

The Phase 1 hard-positive examples should become regression tests.

At minimum test cases should cover:

1.  `.com` business name versus legal business name;
2.  street abbreviation differences;
3.  spelling corruption;
4.  `<NULL>` in one address;
5.  Unicode/Indian-language representation;
6.  token order changes;
7.  noisy names with highly similar addresses;
8.  noisy addresses with strong name evidence.

The test should verify that normalization and feature computation behave
deterministically.

------------------------------------------------------------------------

# 30. Unit tests

`phase2/tests/test_phase2.py` must test:

### Normalization

-   case normalization;
-   punctuation normalization;
-   whitespace;
-   legal suffix normalization;
-   street abbreviations;
-   Unicode handling;
-   missing values.

### Similarity

-   exact equality;
-   Jaccard;
-   edit similarity;
-   empty strings;
-   identical strings;
-   completely different strings.

### Candidate generation

-   exact-name retrieval;
-   exact-address retrieval;
-   country blocking;
-   union/deduplication;
-   high-frequency block protection.

### Label creation

-   positive target;
-   negative target;
-   multiple targets;
-   empty ground truth.

### Prediction

-   threshold application;
-   multi-match output;
-   deterministic ordering;
-   zero-match handling.

------------------------------------------------------------------------

# 31. Determinism

Every stochastic operation must use the configured seed.

The same input must produce the same:

-   candidates;
-   features;
-   train/validation split;
-   model;
-   threshold evaluation;
-   final predictions.

The output ordering must also be deterministic.

------------------------------------------------------------------------

# 32. Memory and scalability

The dataset is large enough that careless Python object usage can become
a problem.

Avoid:

``` python
for every S1 row:
    scan all S2 rows
```

Avoid constructing the complete Cartesian product.

Prefer:

-   dictionary indexes;
-   grouped DataFrames;
-   vectorized operations;
-   chunked processing;
-   compact candidate tables;
-   explicit column selection.

Candidate generation and feature computation should be chunkable.

The pipeline should be capable of processing data substantially larger
than the development sample.

------------------------------------------------------------------------

# 33. Development mode

Before running the complete dataset, support a development
configuration.

For example:

``` yaml
development:
  enabled: true
  max_source1_rows: 5000
  max_target_rows: 100000
```

The development mode must use deterministic subsets.

Once correctness is established:

``` yaml
development:
  enabled: false
```

for the complete run.

------------------------------------------------------------------------

# 34. Phase 2 pipeline entry point

Create:

``` text
phase2/src/run_phase2.py
```

It should orchestrate:

``` text
1. load configuration
2. load data
3. normalize
4. build indexes
5. generate candidates
6. evaluate candidate recall
7. construct features
8. build labels
9. split train/validation
10. train model
11. evaluate model
12. tune threshold
13. save model/artifacts
14. run inference
15. write prediction output
16. write report
```

It must not contain the implementation of every component itself.

------------------------------------------------------------------------

# 35. Candidate statistics artifact

Create:

``` text
phase2/artifacts/candidate_statistics.json
```

Include:

``` json
{
  "s2": {
    "candidate_pairs": 0,
    "positive_pairs_recovered": 0,
    "candidate_recall": 0.0,
    "mean_candidates_per_s1": 0.0,
    "median_candidates_per_s1": 0.0,
    "p95_candidates_per_s1": 0.0,
    "p99_candidates_per_s1": 0.0
  },
  "s3": {}
}
```

Also record recall by country and S1 match bucket.

------------------------------------------------------------------------

# 36. Feature statistics artifact

Create:

``` text
phase2/artifacts/feature_statistics.json
```

Record:

-   feature names;
-   data types;
-   missing rates;
-   positive means;
-   negative means;
-   basic distributions;
-   feature correlations where useful.

This will help identify features that provide little separation.

------------------------------------------------------------------------

# 37. Threshold artifact

Create:

``` text
phase2/artifacts/threshold_results.json
```

It must contain every tested threshold and the corresponding validation
metrics.

Also store:

``` text
selected_threshold
selection_metric = F0.5
```

If separate S2/S3 thresholds are used, store both.

------------------------------------------------------------------------

# 38. Model artifact

Save the trained model with `joblib`.

Example conceptual structure:

``` text
phase2/artifacts/model/
├── matcher.joblib
└── feature_schema.json
```

`feature_schema.json` must record the exact feature order used during
training.

This prevents training/inference column-order mistakes.

------------------------------------------------------------------------

# 39. Phase 2 report

Create:

``` text
phase2/reports/phase2_report.md
```

The report must contain:

## 39.1 Objective

What Phase 2 implements.

## 39.2 Candidate generation

-   blocking rules;
-   candidate counts;
-   candidate recall;
-   S2/S3 differences.

## 39.3 Features

List every feature and explain its purpose.

## 39.4 Training

-   model;
-   train/validation split;
-   positive/negative counts;
-   negative sampling strategy.

## 39.5 Validation

Report:

-   precision;
-   recall;
-   F0.5;
-   threshold sweep;
-   final selected threshold.

## 39.6 Error analysis

Show examples of:

-   false positives;
-   false negatives;
-   hard positives;
-   zero-match entities;
-   multi-match entities.

## 39.7 Scalability

Document:

-   candidate counts;
-   runtime;
-   memory observations;
-   chunk sizes.

## 39.8 Limitations

Explicitly state remaining failure modes.

------------------------------------------------------------------------

# 40. Required validation sequence

The implementation should not jump directly to final inference.

Run validation in this order:

``` text
A. normalization tests
        ↓
B. candidate-generation unit tests
        ↓
C. candidate recall evaluation
        ↓
D. feature correctness checks
        ↓
E. model training
        ↓
F. pair-level evaluation
        ↓
G. entity-level F0.5 evaluation
        ↓
H. threshold sweep
        ↓
I. error analysis
        ↓
J. full-scale inference
```

If candidate recall is poor, do not compensate by changing the
classifier. Fix candidate generation first.

------------------------------------------------------------------------

# 41. Important failure modes to avoid

## Failure 1 --- Exact matching only

Not acceptable.

Phase 1 demonstrates insufficient recall.

## Failure 2 --- One match per S1

Not acceptable.

Most matched S1 entities have multiple target matches.

## Failure 3 --- Full Cartesian comparison

Not acceptable for the complete dataset.

## Failure 4 --- Country ignored

Wasteful and inconsistent with Phase 1 evidence.

## Failure 5 --- Country used as the only signal

Not acceptable because both countries contain many businesses.

## Failure 6 --- Threshold = 0.5 without validation

Not acceptable.

## Failure 7 --- Test-set threshold tuning

Not acceptable.

## Failure 8 --- Random row split

Can leak the same S1 entity across train and validation.

## Failure 9 --- ASCII-only normalization

Will damage Indian-language matching.

## Failure 10 --- Deleting all legal suffixes blindly

Can remove useful information and create collisions.

## Failure 11 --- Huge common-token blocks

Can create candidate explosions.

## Failure 12 --- Treating candidate recall as model recall

A classifier cannot recover a positive pair that candidate generation
never produced.

------------------------------------------------------------------------

# 42. AWS compatibility

The Phase 2 implementation must be designed for AWS execution.

The code must not depend on:

-   a developer's local absolute paths;
-   local-only services;
-   interactive notebooks;
-   manually maintained state.

Use configurable paths.

The architecture should permit:

``` text
Amazon S3
   ↓
data loading
   ↓
candidate generation
   ↓
feature computation
   ↓
model training
   ↓
model artifact
   ↓
batch inference
   ↓
submission artifact
```

AWS integration should be kept separate from core matching logic.

For example, core code should accept paths/configuration rather than
directly hard-coding S3 operations everywhere.

------------------------------------------------------------------------

# 43. AWS-oriented execution model

The eventual execution should support:

### Storage

Amazon S3 for:

-   input datasets;
-   artifacts;
-   models;
-   reports;
-   predictions.

### Compute

A suitable AWS compute environment can execute the Python pipeline.

The exact service selection should follow the challenge's permitted AWS
environment and resource limits.

### Model artifact

Store the trained model and feature schema in S3.

### Inference

Run batch inference against the test sources.

The same feature-generation code must be used during training and
inference.

------------------------------------------------------------------------

# 44. Training/inference separation

Never duplicate feature logic.

Training:

``` text
raw records
→ normalization
→ candidate generation
→ features
→ labels
→ model
```

Inference:

``` text
raw records
→ normalization
→ candidate generation
→ features
→ model
→ threshold
→ predictions
```

The feature code must be shared.

------------------------------------------------------------------------

# 45. Submission generation

The final output must preserve the challenge's expected schema.

Before writing the final submission:

1.  verify every S1 ID is represented;
2.  verify IDs belong to valid S2/S3 sources;
3.  verify no malformed IDs;
4.  verify duplicate target IDs are removed;
5.  verify multi-match serialization;
6.  verify zero-match serialization;
7.  verify deterministic ordering;
8.  verify row count;
9.  verify header.

Run a final submission validator.

------------------------------------------------------------------------

# 46. Quality gates

Phase 2 is considered complete only when all of these are true:

### Gate 1

All Phase 2 unit tests pass.

### Gate 2

Candidate generation produces measurable recall against ground truth.

### Gate 3

Candidate recall is reported separately for S2 and S3.

### Gate 4

The model trains successfully from generated candidates.

### Gate 5

F0.5 is evaluated at the entity/submission level.

### Gate 6

Threshold selection is performed only on validation data.

### Gate 7

Multiple matches per S1 are supported.

### Gate 8

Zero matches are supported.

### Gate 9

Full inference runs without a Cartesian product.

### Gate 10

Final prediction schema is validated.

### Gate 11

The entire pipeline can be rerun deterministically.

### Gate 12

All Phase 2 artifacts and documentation are generated automatically.

------------------------------------------------------------------------

# 47. Recommended implementation order

Implement Phase 2 in exactly this sequence:

``` text
STEP 1
Create phase2 directory structure.

STEP 2
Implement configuration loading.

STEP 3
Reuse/finalize Phase 1 normalization behavior.

STEP 4
Implement target-source indexes.

STEP 5
Implement exact-name blocking.

STEP 6
Implement exact-address blocking.

STEP 7
Implement numeric/address blocking.

STEP 8
Implement prefix/ngram blocking.

STEP 9
Union and deduplicate candidates.

STEP 10
Measure candidate recall.

STEP 11
Implement name features.

STEP 12
Implement address features.

STEP 13
Implement structural features.

STEP 14
Generate training labels.

STEP 15
Create entity-level train/validation split.

STEP 16
Implement controlled negative sampling.

STEP 17
Train baseline logistic-regression matcher.

STEP 18
Evaluate pair-level metrics.

STEP 19
Evaluate entity-level F0.5.

STEP 20
Run threshold sweep.

STEP 21
Perform error analysis.

STEP 22
Freeze model + threshold + feature schema.

STEP 23
Run complete test inference.

STEP 24
Validate submission.

STEP 25
Generate phase2_report.md.
```

------------------------------------------------------------------------

# 48. Definition of Done

Phase 2 should produce the following final artifacts:

``` text
phase2/
├── artifacts/
│   ├── candidate_statistics.json
│   ├── feature_statistics.json
│   ├── threshold_results.json
│   └── model/
│       ├── matcher.joblib
│       └── feature_schema.json
│
├── config/
│   └── phase2_config.yaml
│
├── reports/
│   └── phase2_report.md
│
├── src/
│   ├── candidate_generation.py
│   ├── feature_engineering.py
│   ├── matching_model.py
│   ├── threshold_tuning.py
│   ├── inference.py
│   └── run_phase2.py
│
└── tests/
    └── test_phase2.py
```

The pipeline must demonstrate:

``` text
high-recall candidate generation
        +
robust similarity features
        +
trained match scorer
        +
F0.5-aware thresholding
        +
zero/one/multiple match support
        +
deterministic scalable inference
        =
Phase 2 entity-resolution system
```

------------------------------------------------------------------------

# 49. Relationship to Phase 3

Phase 2 should stop at a validated, scalable matching engine and
submission-generation pipeline.

Do not prematurely add Phase 3 ideas such as:

-   advanced ensemble systems;
-   extensive hyperparameter searches;
-   complex neural architectures;
-   post-hoc manual corrections;
-   large-scale external data enrichment.

Those can be evaluated later only after the Phase 2 baseline is
measurable.

The purpose of Phase 2 is to establish a **strong, reproducible,
explainable baseline** whose candidate recall, feature quality, model
performance, threshold behavior, and runtime are all measurable.

------------------------------------------------------------------------

## Final Phase 2 principle

The most important design rule is:

> **Candidate generation determines what the model can possibly
> discover; the scorer determines which discovered candidates become
> predictions.**

Therefore, Phase 2 must first maximize reliable candidate recall under
computational constraints, and only then optimize precision/F0.5 through
feature engineering, scoring, and thresholding.
