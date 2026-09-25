# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Ninja Hattori]

**Team Members:** [Devraj Bijpuria , Shashwat]

**Submission Date:** 25 September 2026

---

## 1. Executive Summary

We developed a large-scale, precision-oriented business entity resolution pipeline using multi-rule candidate generation, engineered name/address similarity features, and supervised binary matching. The pipeline progressively evolved from a Logistic Regression baseline to a hardened HistGradientBoosting model, with threshold selection based on macro F_0.5 on held-out validation data.

The final pipeline performs multi-stage blocking before model inference, combines complementary name, address, geographic-structure, and exact-match features, and generates both the final matching results and the exact candidate set used immediately before inference.

---

## 2. Methodology

### 2.1 Problem Analysis

The task is to identify all Source 2 and Source 3 records corresponding to each Source 1 business entity. A Source 1 entity may have zero, one, or multiple matches.

Exploratory analysis showed substantial variation in business names and addresses across sources. Important noise patterns included:

- Abbreviations and legal-suffix variations such as Corp/Corporation, Pvt/Private, and Ltd/Limited.
- Punctuation and formatting differences.
- Word-order changes and minor spelling errors.
- Business-name variants and transliteration differences.
- Address abbreviations such as Rd/Road and St/Street.
- Reordering or omission of address components.
- Missing postal codes, states, or other address components.
- Landmark-based address descriptions.
- Differences in municipal numbering and address formatting.
- Missing business names or addresses.

Because the test set contains a country not present in training, country was treated as an open-set categorical/string attribute rather than hard-coding the known training countries.

The evaluation metric is macro-averaged F_0.5, which places greater emphasis on precision than recall. This makes avoiding incorrect business merges particularly important.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Supervised Classifier

**Core Innovation:** A multi-rule blocking and matching pipeline combining complementary exact, structural, lexical, character-level, phonetic, and address-based signals. The blocking stage was hardened through miss analysis, rule ablations, candidate-cap analysis, transliteration-aware blocking, and address-focused blocking before freezing the final configuration.

The overall pipeline consists of:

1. Data loading and validation.
2. Field normalization.
3. Candidate generation using multiple blocking rules.
4. Candidate-level feature engineering.
5. Supervised binary classification.
6. Threshold selection using validation F_0.5.
7. Inference over the final candidate set.
8. Generation and validation of `matching_results.tsv` and `candidate_pairs.tsv`.

No external business databases, geocoding services, commercial entity-resolution APIs, or external business-data augmentation were used.

---

## 3. Candidate Generation (Blocking)

The candidate-generation stage reduces the comparison space before machine-learning inference while attempting to preserve true matches.

### Blocking keys used:

The frozen Phase 2.5 configuration uses complementary blocking rules based on normalized business and address information:

- Country block.
- Exact normalized business-name block.
- Exact normalized address block.
- Numeric/house/postal-information block.
- Business-name prefix block.
- Address prefix block.
- Business-name character n-gram block.
- Business-name phonetic block.
- Address-token block.
- House-number/locality block.
- State/locality block.
- Address-signature block.

The configuration limits very large blocks and caps the number of candidates considered per Source 1 entity.

The frozen settings include:

- Maximum block size: 100.
- Maximum candidates per Source 1 entity per target source: 100.
- Source 1 processing chunk size: 25,000.

### Candidate pairs generated:

The final full-test candidate file contains one candidate list for each of the 1,732,544 Source 1 test entities.

Of these:

- 1,731,720 Source 1 entities have at least one candidate.
- 824 Source 1 entities have an empty candidate list.

The candidate file represents the final candidate set immediately before model scoring, rather than an intermediate blocking result.

### How true matches were protected from being lost:

Multiple overlapping blocking rules were used rather than relying on a single key. Phase 2 analysis measured candidate recall and identified blocking misses.

The baseline blocking configuration achieved approximately:

- Source 2 candidate recall: 98.103%.
- Source 3 candidate recall: 98.156%.
- Combined candidate recall: approximately 98.13%.

The blocking miss analysis identified difficult cases including near-duplicate businesses, same-address variants, and transliteration-related name/address differences. Additional locality, address-signature, and transliteration-aware blocking rules were subsequently incorporated into the hardened Phase 2.5 configuration.

---

## 4. Matching Model

### Features used:

Candidate pairs are represented using complementary name, address, and structural features.

**Name features:**

- Character n-gram similarity.
- Token Jaccard similarity.
- Edit-based similarity.
- Exact equality of normalized names.

**Address features:**

- Character n-gram similarity.
- Token Jaccard similarity.
- Edit-based similarity.
- Exact equality of normalized addresses.

**Other features:**

- Country equality.
- Name-missing indicator.
- Address-missing indicator.
- House-number equality.
- Postal-code equality.

The final frozen Phase 2.5 configuration uses 42 candidate-level features.

Blocking-rule indicators that directly encode how a candidate was retrieved were excluded from the final matching feature set to reduce dependence on the blocking mechanism itself.

### Model type:

The initial Phase 2 baseline used Logistic Regression as a transparent supervised matching model.

During Phase 2.5 hardening, alternative model configurations were evaluated and the frozen configuration uses:

**HistGradientBoosting**

with:

- `max_iter = 200`
- Training sample: 50,000 Source 1 entities.

The model is trained as a binary classifier over candidate pairs.

### Threshold selection method:

The decision threshold was selected using held-out validation data and macro F_0.5 rather than using the default probability threshold of 0.5.

The frozen Phase 2.5 configuration uses a decision threshold of:

**0.80**

This reflects the precision-heavy nature of the challenge metric and the need to avoid false business merges.

---

## 5. Results & Error Analysis

### Validation results

The best documented validation result from the Phase 2 development evaluation was:

- **F_0.5 Score (macro): 0.9709**
- **Precision: 0.9868**
- **Recall: 0.9472**
- **True Positives: 3,285**
- **False Positives: 44**
- **False Negatives: 183**

The validation result above corresponds to the documented Phase 2 baseline evaluation and should not be interpreted as the hidden-test or leaderboard score. The final frozen Phase 2.5 pipeline subsequently changed the matching model and threshold.

### Common false positives (wrong merges):

Observed false-positive risks primarily came from records with strong partial similarity but insufficient distinguishing information, particularly:

- Businesses with similar or identical names.
- Common business-name patterns combined with incomplete addresses.
- Address similarities where distinguishing components were missing.
- Multiple businesses sharing similar locality or structural address information.

The precision-heavy F_0.5 objective was therefore used when selecting the matching threshold.

### Common false negatives (missed matches):

Blocking and error analysis identified several difficult match patterns:

- Near-duplicate businesses with formatting differences.
- Same-address businesses with different name representations.
- Transliteration differences.
- Address variants with missing components.
- Name variations that were not captured by a single normalization rule.

These observations motivated the additional blocking and address/locality rules introduced during Phase 2.5 hardening.

### Zero-match / singleton handling:

The pipeline explicitly supports Source 1 entities with no corresponding Source 2 or Source 3 records. In the documented validation evaluation, 49 of 54 zero-match entities were correctly identified as having no matches.

This behavior is important because correctly predicting an empty match list receives full credit for a true zero-match entity under the challenge's macro F_0.5 evaluation.

---

## 6. Conclusion

The final solution uses a scalable blocking-plus-classification architecture designed specifically for the precision-heavy entity-resolution objective. The main improvements came from combining multiple complementary blocking rules with name/address similarity features and systematically hardening the pipeline against blocking misses and false merges.

The complete test-set inference was executed for all 1,732,544 Source 1 entities, producing the required `matching_results.tsv` and `candidate_pairs.tsv` files. The generated submission outputs passed the provided submission-format validation checks.

---

## Appendix

### A. Code Artefacts

The complete runnable implementation is provided in:

`code/business_entity_resolution/`

The source code is organized under:

`code/business_entity_resolution/src/`

The main pipeline components cover:

- Data loading and normalization.
- Candidate generation/blocking.
- Feature engineering.
- Matching-model training and inference.
- Threshold tuning.
- Output generation.

The final outputs are:

`output/matching_results.tsv`

and

`output/candidate_pairs.tsv`

The accompanying `README.md` provides the reproduction instructions and `requirements.txt` specifies the Python dependencies required by the implementation.

### B. Additional Results

The Phase 2.5 development process included additional experiments covering:

- Baseline reproduction.
- Blocking-rule ablation.
- Blocking miss analysis.
- Candidate-cap sensitivity.
- High-frequency/key cutoff sensitivity.
- Transliteration-aware blocking.
- Feature ablation.
- Model comparison.
- Threshold robustness.
- Full-scale resource testing.

The final frozen configuration was selected after these analyses and used for the full test-set inference.

---