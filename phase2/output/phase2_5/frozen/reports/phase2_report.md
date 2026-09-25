# Phase 2 Report: Baseline Entity Matching (dev mode)

Generated automatically by `phase2/src/run_phase2.py` from the JSON artifacts in `phase2/artifacts/`.

> **Development mode.** 5,000 sampled Source-1 entities per split and 100000 target records per source. The target pools are much smaller than the real sources, so candidate counts and precision here are not representative of full scale.

## 1. Objective

A baseline pipeline: normalization, blocking (S1 to S2 and S1 to S3 separately), candidate recall measurement, pair features, a logistic-regression scorer, a threshold chosen on validation macro F0.5 per Source-1 entity, zero/one/many-match assembly, and the two submission files.

## 2. Candidate generation

Every key is prefixed with the exact normalized country (trimmed and casefolded, an open set that is never mapped). Target keys holding more than 100 records are skipped, and each one is recorded with its frequency and the positives it cost. Each S1 entity keeps at most 100 candidates per target source (most rules first, then the best name or address edit similarity); the dropped pairs and their recall cost are recorded.

| rule | key |
|---|---|
| exact_name | compact core name; sorted unique core tokens |
| exact_address | sorted unique normalized address tokens |
| numeric | house number + first street token; house number + postal code |
| name_prefix | first N characters of the compact core name |
| address_prefix | first N characters of the compact normalized address |
| name_ngram | each of the 2 rarest core-name tokens (seen at least twice) |
| name_phonetic | consonant skeleton of the compact core name |
| address_token | each of the 2 rarest locality tokens of the address (seen at least twice) |
| house_locality | house number + each of the 2 rarest locality tokens (Phase 2.5) |
| state_locality | state + each of the 2 rarest locality tokens (Phase 2.5) |
| address_signature | sorted non-numeric, non-state address components (Phase 2.5) |

| metric | S2 | S3 |
|---|---|---|
| candidate_recall | 0.98487 | 0.98597 |
| precap_candidate_recall | 0.98547 | 0.98609 |
| positive_pairs_total | 8328 | 9055 |
| positive_pairs_recovered | 8202 | 8928 |
| candidate_pairs | 285004 | 271797 |
| mean_candidates_per_s1 | 57.001 | 54.359 |
| median_candidates_per_s1 | 57.0 | 50.0 |
| p95_candidates_per_s1 | 100.0 | 100.0 |
| p99_candidates_per_s1 | 100.0 | 100.0 |
| max_candidates_per_s1 | 100 | 100 |
| s1_with_zero_candidates | 2 | 2 |
| recall cost of the per-S1 cap | 0.0006 | 0.00011 |
| positives lost that shared a skipped key | 50 | 64 |

Combined candidate recall (S2 and S3): **98.55%**. This is the ceiling on what the model can recover.

Recall by rule, before the cap ("only" = positives that no other rule found):

| rule | S2 recall | S2 only | S3 recall | S3 only | S2 pairs | S3 pairs | S2 skipped keys | S3 skipped keys |
|---|---|---|---|---|---|---|---|---|
| exact_name | 0.59654 | 6 | 0.58034 | 9 | 7347 | 7754 | 0 | 0 |
| exact_address | 0.31088 | 0 | 0.38145 | 0 | 2597 | 3472 | 0 | 0 |
| numeric | 0.56976 | 5 | 0.54158 | 8 | 6654 | 6747 | 4 | 4 |
| name_prefix | 0.63509 | 19 | 0.62529 | 41 | 24858 | 27698 | 55 | 53 |
| address_prefix | 0.38713 | 4 | 0.38277 | 10 | 3600 | 4065 | 0 | 0 |
| name_ngram | 0.66559 | 58 | 0.65555 | 110 | 143250 | 121575 | 381 | 385 |
| name_phonetic | 0.61972 | 77 | 0.57637 | 35 | 8903 | 8919 | 0 | 0 |
| address_token | 0.86011 | 1 | 0.8201 | 1 | 159541 | 159913 | 167 | 182 |
| house_locality | 0.62776 | 10 | 0.60486 | 18 | 5809 | 6240 | 0 | 0 |
| state_locality | 0.87776 | 2 | 0.84495 | 4 | 89524 | 94470 | 81 | 88 |
| address_signature | 0.12764 | 7 | 0.16345 | 1 | 5260 | 3026 | 3 | 12 |

Recall by S1 country and S1 match bucket:

| group | S2 recall | S3 recall |
|---|---|---|
| country = india | 0.972 | 0.97084 |
| country = us | 0.99356 | 0.99581 |
| bucket = multi | 0.98474 | 0.98622 |
| bucket = one | 0.9927 | 0.96825 |

Every skipped key with its target frequency and lost positives is listed in `candidate_statistics.json` under `<source>.skipped_keys.<rule>.keys`.

## 3. Features

42 features, in the exact order stored in `artifacts/model/feature_schema.json`. A missing value never creates similarity (every similarity is 0 when either side is empty), and missingness has explicit indicator features.

| feature | purpose | positive mean | negative mean | correlation with label |
|---|---|---|---|---|
| name_norm_eq | normalized names identical | 0.2962 | 0.0023 | 0.4784 |
| name_key_eq | compact core names identical (suffix-, domain- and space-insensitive) | 0.572 | 0.0087 | 0.6103 |
| name_token_jaccard | token overlap of normalized names | 0.6814 | 0.1416 | 0.4745 |
| name_core_token_jaccard | token overlap without legal suffixes | 0.7279 | 0.1247 | 0.5072 |
| name_char3_jaccard | character 3-gram overlap (typo-tolerant) | 0.6864 | 0.1416 | 0.4888 |
| name_edit_sim | 1 - Levenshtein / max length | 0.7309 | 0.3104 | 0.3795 |
| name_translit_char3_jaccard | 3-gram overlap after Indic romanization | 0.7129 | 0.1454 | 0.5041 |
| name_translit_edit_sim | edit similarity after romanization | 0.7657 | 0.3182 | 0.4024 |
| name_token_set_ratio | order-insensitive token-set similarity of core names | 0.9202 | 0.4641 | 0.3581 |
| name_phonetic_eq | consonant skeletons identical | 0.6279 | 0.0145 | 0.5894 |
| name_phonetic_edit_sim | edit similarity of consonant skeletons (transliteration spelling) | 0.8648 | 0.3148 | 0.4375 |
| name_length_1 | S1 name length | 25.2936 | 25.0451 | 0.0052 |
| name_length_2 | target name length | 24.6125 | 26.1233 | -0.0274 |
| name_length_ratio | shorter / longer name length | 0.8617 | 0.7051 | 0.1386 |
| name_length_difference | absolute length difference | 3.9548 | 9.3415 | -0.1272 |
| name_token_count_1 | S1 token count | 3.5666 | 3.5383 | 0.005 |
| name_token_count_2 | target token count | 3.4242 | 3.5905 | -0.0243 |
| name_token_count_difference | token count difference | 0.5214 | 1.0974 | -0.1018 |
| legal_suffix_present_1 | S1 name has a legal suffix | 0.6649 | 0.6389 | 0.0093 |
| legal_suffix_present_2 | target name has a legal suffix | 0.5696 | 0.5875 | -0.0063 |
| legal_suffix_equal | both have the same legal-suffix set | 0.448 | 0.1514 | 0.1395 |
| name_script_mismatch | one name in Indic script, the other Latin | 0.097 | 0.0653 | 0.022 |
| name_missing_1 | S1 name empty | 0.0 | 0.0 | 0.0 |
| name_missing_2 | target name empty | 0.0 | 0.0 | 0.0 |
| address_norm_eq | normalized addresses identical | 0.2288 | 0.0 | 0.4714 |
| address_token_jaccard | address token overlap | 0.7688 | 0.1201 | 0.6611 |
| address_char3_jaccard | address 3-gram overlap | 0.7374 | 0.1242 | 0.6496 |
| address_edit_sim | address edit similarity | 0.7066 | 0.3025 | 0.4226 |
| address_component_jaccard | overlap of comma components (order-insensitive street/city/state parts) | 0.6752 | 0.1221 | 0.4611 |
| address_missing_1 | S1 address empty | 0.0 | 0.0 | 0.0 |
| address_missing_2 | target address empty | 0.0413 | 0.0162 | 0.0336 |
| address_missing_either | either address empty | 0.0413 | 0.0162 | 0.0336 |
| house_number_both_present | both house numbers extracted | 0.8722 | 0.9104 | -0.023 |
| house_number_equal | house numbers equal | 0.6653 | 0.0089 | 0.6739 |
| postal_both_present | both postal codes extracted | 0.002 | 0.0001 | 0.0318 |
| postal_equal | postal codes equal | 0.002 | 0.0 | 0.0438 |
| state_both_present | both states extracted | 0.9554 | 0.982 | -0.0339 |
| state_equal | states equal (codes, English names and native scripts unified) | 0.948 | 0.3471 | 0.2154 |
| country_equal | exact normalized country equal (open set) | 1.0 | 1.0 | 0.0 |
| target_source_is_s2 | target from Source 2 | 0.4788 | 0.5129 | -0.0118 |
| target_source_is_s3 | target from Source 3 | 0.5212 | 0.4871 | 0.0118 |
| number_of_blocks_that_retrieved_pair | how many independent blocking rules retrieved the pair | 6.3168 | 1.3 | 0.8055 |

Highly correlated feature pairs (|r| >= 0.9): name_token_jaccard ~ name_char3_jaccard (0.939), name_token_jaccard ~ name_translit_char3_jaccard (0.92), name_char3_jaccard ~ name_edit_sim (0.901), name_char3_jaccard ~ name_translit_char3_jaccard (0.983), name_edit_sim ~ name_translit_edit_sim (0.958), name_translit_char3_jaccard ~ name_translit_edit_sim (0.902), name_length_ratio ~ name_length_difference (-0.916), address_token_jaccard ~ address_char3_jaccard (0.945), address_missing_2 ~ address_missing_either (1.0), address_missing_2 ~ state_both_present (-0.95), address_missing_either ~ state_both_present (-0.95), target_source_is_s2 ~ target_source_is_s3 (-1.0).

## 4. Training

| item | value |
|---|---|
| model | StandardScaler + LogisticRegression (C = 1.0, lbfgs) |
| split | entity level: 4,000 training and 1,000 validation Source-1 entities; entities in both = 0 |
| training candidate pairs | 441,517 (13,713 positive) |
| pairs used to fit | 82,278 = 13,713 positives + 68,565 negatives |
| negative sampling | 5 negatives per positive, drawn from the training candidates; 50% are the most name-and-address-similar non-matches (hard negatives), the rest uniform (seed 42) |

Largest standardized coefficients:

| feature | coefficient |
|---|---|

## 5. Validation

Selected threshold **0.8** by macro F0.5 per Source-1 entity (challenge metric), on 1,000 validation entities (115,284 candidate pairs). Every validation entity counts, including zero-match entities and entities with no candidates. Pooled FN include positives that blocking never retrieved.

| threshold | macro F0.5 | TP | FP | FN | precision | recall | pooled F0.5 | entities with prediction | avg predictions per entity | zero-match entities scored 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.9495 | 3401 | 214 | 67 | 0.9408 | 0.9807 | 0.9485 | 954 | 3.615 | 45/54 |
| 0.1 | 0.9607 | 3398 | 157 | 70 | 0.9558 | 0.9798 | 0.9605 | 951 | 3.555 | 48/54 |
| 0.15 | 0.9644 | 3398 | 133 | 70 | 0.9623 | 0.9798 | 0.9658 | 951 | 3.531 | 48/54 |
| 0.2 | 0.9662 | 3397 | 121 | 71 | 0.9656 | 0.9795 | 0.9684 | 951 | 3.518 | 48/54 |
| 0.25 | 0.9672 | 3393 | 115 | 75 | 0.9672 | 0.9784 | 0.9694 | 950 | 3.508 | 49/54 |
| 0.3 | 0.9702 | 3391 | 97 | 77 | 0.9722 | 0.9778 | 0.9733 | 950 | 3.488 | 49/54 |
| 0.35 | 0.971 | 3388 | 88 | 80 | 0.9747 | 0.9769 | 0.9751 | 950 | 3.476 | 49/54 |
| 0.4 | 0.9725 | 3386 | 80 | 82 | 0.9769 | 0.9764 | 0.9768 | 950 | 3.466 | 49/54 |
| 0.45 | 0.9735 | 3380 | 77 | 88 | 0.9777 | 0.9746 | 0.9771 | 949 | 3.457 | 50/54 |
| 0.5 | 0.9739 | 3379 | 74 | 89 | 0.9786 | 0.9743 | 0.9777 | 949 | 3.453 | 50/54 |
| 0.55 | 0.9744 | 3378 | 71 | 90 | 0.9794 | 0.974 | 0.9783 | 949 | 3.449 | 50/54 |
| 0.6 | 0.9752 | 3376 | 67 | 92 | 0.9805 | 0.9735 | 0.9791 | 949 | 3.443 | 50/54 |
| 0.65 | 0.9756 | 3372 | 64 | 96 | 0.9814 | 0.9723 | 0.9795 | 949 | 3.436 | 50/54 |
| 0.7 | 0.9761 | 3367 | 59 | 101 | 0.9828 | 0.9709 | 0.9804 | 949 | 3.426 | 50/54 |
| 0.75 | 0.9768 | 3363 | 55 | 105 | 0.9839 | 0.9697 | 0.981 | 949 | 3.418 | 50/54 |
| 0.8 | 0.977 | 3353 | 49 | 115 | 0.9856 | 0.9668 | 0.9818 | 949 | 3.402 | 50/54 |
| 0.85 | 0.9762 | 3340 | 43 | 128 | 0.9873 | 0.9631 | 0.9824 | 948 | 3.383 | 50/54 |
| 0.9 | 0.976 | 3313 | 34 | 155 | 0.9898 | 0.9553 | 0.9827 | 946 | 3.347 | 51/54 |
| 0.95 | 0.9747 | 3279 | 27 | 189 | 0.9918 | 0.9455 | 0.9822 | 946 | 3.306 | 51/54 |

At the selected threshold, by target source. Diagnostic only: an entity with no match in that source and no prediction scores 1.0 there, so these are not comparable to the challenge metric.

| source | macro F0.5 (that source only) | precision | recall |
|---|---|---|---|
| s2 | 0.9673 | 0.9836 | 0.9695 |
| s3 | 0.971 | 0.9875 | 0.9644 |

Pair level on validation candidates only (this recall excludes pairs blocking never produced):

| threshold | TP | FP | FN | TN | precision | recall | F0.5 |
|---|---|---|---|---|---|---|---|
| at_0.5 | 3379 | 74 | 38 | 111793 | 0.9786 | 0.9889 | 0.9806 |
| at_selected | 3353 | 49 | 64 | 111818 | 0.9856 | 0.9813 | 0.9847 |

## 6. Error analysis (validation)

- False positives: 49.
- False negatives scored below the threshold: 64.
- False negatives never retrieved by blocking: 51.
- Zero-match entities: 54, of which 50 were correctly predicted empty.
- Multi-match entities: 898; mean true matches 3.808, mean predicted 3.727, entities with at least two predictions 879.

### False positives (highest scores)

| s1 | target | score | s1 name | target name | s1 address | target address |
|---|---|---|---|---|---|---|
| S1-116338782 | S2-402815945 | 0.9999 | Rocky Ministries Corp | Rocky Ministries Harbor Corp | 1026 1/2 Bridge Street, Unit 1/2, Columbia, TN | NULL, TN, 1026 1/15 BRIDGE STREET, COLUMBIA |
| S1-557022307 | S2-229162590 | 0.9997 | DTM Solutions Limited | DTV Solutions Limited | Plot No.699, Flat No.Gm-1, Meenal Apt. Shalimar Garden Extension - 1, Sahibabad, Ghaziabad, Uttar Pradesh | 702/2 , FLAT NO.GM-1, MEENAL APT. SHALIMAR GARTEN EXTENSION - 1, SAHIBABAD, Uttar Pradesh |
| S1-680997139 | S2-784387477 | 0.9992 | Consultants Godbole (india) Pvt Ltd | Consultants Godbole (india) Group Pvt Ltd | C/O- Manoj Kumar Singh, Bhagwanpur, Mushahari, Muzaffarpur, Bihar | H.NO 69/5 C/O- MANOJ KUMAR SINGH, BHAGWANPUR, MUSHAHARI, Bihar |
| S1-772206302 | S3-28141358 | 0.9961 | Accurate Vidyalaya | Sreeram Vidyalaya | Ganpati Indigo, Near Gaayodhya By Pass, H No - 10, Madhya Pradesh, Bhopal | House No. 10, Bhopal, MP |
| S1-393259793 | S3-506660295 | 0.9945 | Innova Care Ltd | Innova  Marketing | 34, Pelican Ind Estate, Road No 5 Kathwada Gidc, Daskroi, Ahmedabad, Gujarat | 55, Pelican Ind Estate, Road No 5 Kathwada Gidc, Daskroi, Ahmedabad, GJ |
| S1-607139509 | S2-346699671 | 0.9935 | Apex Elite Fox | APEX ELITE FOX GROUP | 137 Chesterfield Way, Folsom, CA | 141 CHESTERFIELD WAY, FOLSOM, CA |
| S1-573427830 | S3-476424970 | 0.9925 | Wildlife Network | [LLC] Wildlife Network | 5587 Fox Chase Drive, Hokah, MN | 5596 Fox Chase Dr, Hokah, Minnesota |
| S1-214131574 | S3-349805237 | 0.9882 | Vertex Esports, L.L.C. | Vertex, LLC | 7767 White Chapel Road, OH, Licking Twp |  |
| S1-163249957 | S3-395565336 | 0.9854 | Empire Investments, LLC | Door LLC Investments | 5145 45, Auburn, IN |  |
| S1-825411169 | S2-529566185 | 0.9854 | Redstone Traders Private Limited | REDSTONE TRADERS ENTERPRISES PRIVATE LTD | 4 Hari Nivas C Road Churchgate Mumbai, Mumbai, Mumbai City, Maharashtra | NO 5 HARI NIVAS C ROAD CHURCHGATE MUMBAI, MUMBAI, Maharashtra |

### False negatives scored below the threshold (lowest scores)

| s1 | target | score | s1 name | target name | s1 address | target address |
|---|---|---|---|---|---|---|
| S1-342344172 | S2-130834849 | 0.0002 | Panvel Sugar Limited | Iridova | Shop No. 4, Ground Floor, Swami Smarth Chs, Plot Number - E1/C, Sector Number - 12, Panvel, Raigarh(Mh), Maharashtra | DOOR NO 3, Maharashtra, RAIGARH(MH) |
| S1-690151152 | S2-228559678 | 0.0003 | Masco Foundation | Shri Masco Service | 167 Ground Floor Abhay Khand - 1, Indrapuram, Ghaziabad, Uttar Pradesh |  |
| S1-51145721 | S3-384554073 | 0.0004 | Baptist Services | Baptist-Center Enterprises | 217 Springbrook Lane, Chesapeake City, VA |  |
| S1-539406463 | S3-519560115 | 0.0009 | Genetic Heights | Shri -- 6enetic Heights | Flat No 302, Orange Homes, Adorn, K V R Rainbowcolony, Qutubullapur, K.V.Rangareddy, Telangana | 298, Qutubullapur, K.v.rangareddy, TG |
| S1-324704727 | S2-4097610 | 0.0009 | OQ Purpose Co | Dovabelo | D-7, Phase Iii Garden Homes Alkapuri, Gwalior, Madhya Pradesh | H.NO 87 D-7, GWALIOR, Madhya Pradesh |
| S1-900817625 | S3-772786324 | 0.001 | Cameron Allied Priority | Cameron  Priority Partners | N4231 C, Town Of Freedom, WI | N423 C, PMB 8248, Kaukauuna CDP, Wisconsin |
| S1-685048543 | S3-975683365 | 0.0022 | Ghanshyam Milk | Ghanshyam Me | 5Th Floor, 502, Meraki Arena, Cts No. 619/14 619/15 619/21A 21B, V N Purav Marg, Opp R K Studio, Chembur East, Mumbai, Maharashtra | Floor, Mumbai (SUBURBAN), MH |
| S1-977292711 | S2-256007340 | 0.0029 | Sai Innovative Exports Private Limited | ସାଇ ଇନୋଭେଟିଭ୍ ଏକ୍ସପୋର୍ଟସ୍ ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍ | Plot No-355, Rohini Villa, Taraboi, Baniatangi, Jatni, Khordha, Orissa | PCOT NO-00355, JATNI, JATANI, Orissa |
| S1-667264941 | S3-810666500 | 0.0038 | Bright Bright Corporate, Inc | Bright  Bright Cofeforate, Inc | 35142 Lead Line Road, Virden, IL |  |
| S1-916679619 | S3-677979213 | 0.0047 | Tramel Holding Company PLLC | Cal0evo | 300 Charles Street, Unit APT 201, Baltimore, MD | 0300 Charles St, Baltimore, Maryland |

### False negatives never retrieved by blocking

| s1 | target | score | s1 name | target name | s1 address | target address |
|---|---|---|---|---|---|---|
| S1-119033930 | S3-45249084 | nan | Royal Technology Private Limited | रॉयल टेक्नोलॉजी प्राइवेट लिमिटेड | G-8-A, No. 1, Bhagirath Palace, North East, Delhi, 1737 | 737, North East, Delhi, दिल्ली |
| S1-119033930 | S3-745743078 | nan | Royal Technology Private Limited | रॉयल टेक्नोलॉजी प्राइवेट लिमिटेड | G-8-A, No. 1, Bhagirath Palace, North East, Delhi, 1737 | 737, North East, Delhi, दिल्ली |
| S1-119187241 | S2-865100548 | nan | Silver Classic Industries Private Limited | சில்வர் கிளாசிக் இண்டஸ்ட்ரீஸ் பிரைவேட் லிமிடெட் | Dr.No.299, Anandha Nagar, Gudalur (N) And (S), Coimbatore North, Coimbatore, Tamil Nadu | COIMBATORE, COIMBATORE NORTH, DR.NO.299, தமிழ்நாடு |
| S1-202025302 | S3-285987646 | nan | Oncology Associates LLC | Oncology Associates L.L.C. Trading | 2310 Iris Drive, Sierra Vista, AZ |  |
| S1-255217115 | S2-798805961 | nan | My Investments LLP | మై ఇన్వెస్ట్‌మెంట్స్ ఎల్‌ఎల్‌పీ | 22-6-270/3/8, Balala Shopping Mall, Machli Kaman, Charminar, Hyderabad, Telangana | 22-6-270/3/8, HYDERABAD, Telangana |
| S1-261169281 | S2-927345730 | nan | U 3 K Steel | U 3 K | 5200 Clairemont Drive, Town Of Grand Chute, WI |  |
| S1-263753520 | S2-171637668 | nan | Silver Industries Private Limited | सिल्वर इंडस्ट्रीज प्राइवेट लिमिटेड | House No.65 (Basement), Furniture Block, Kirti Nagar, New Delhi, West Delhi, Delhi | NO. 208 HOUSE NO.65 (BASEMENT), NEW DELHI, Delhi |
| S1-270293020 | S2-336292518 | nan | Northern Advanced Lincoln LP | Northem Advanced | 3250 Lake Road, Lancaster, OH |  |
| S1-309089131 | S3-447370721 | nan | Future Impex Limited | M/s Future  Impex | Orissa, Flat No-324, Northern Heights Kalarahanga, Khordha, Bhubaneswar |  |
| S1-309089131 | S3-994619470 | nan | Future Impex Limited | Quoavi | Orissa, Flat No-324, Northern Heights Kalarahanga, Khordha, Bhubaneswar | Door No 621 Flat No-324, Bhubaneswar, Khordha, OD |

### Hardest retrieved positives (lowest scores)

| s1 | target | score | s1 name | target name | s1 address | target address |
|---|---|---|---|---|---|---|
| S1-342344172 | S2-130834849 | 0.0002 | Panvel Sugar Limited | Iridova | Shop No. 4, Ground Floor, Swami Smarth Chs, Plot Number - E1/C, Sector Number - 12, Panvel, Raigarh(Mh), Maharashtra | DOOR NO 3, Maharashtra, RAIGARH(MH) |
| S1-690151152 | S2-228559678 | 0.0003 | Masco Foundation | Shri Masco Service | 167 Ground Floor Abhay Khand - 1, Indrapuram, Ghaziabad, Uttar Pradesh |  |
| S1-51145721 | S3-384554073 | 0.0004 | Baptist Services | Baptist-Center Enterprises | 217 Springbrook Lane, Chesapeake City, VA |  |
| S1-539406463 | S3-519560115 | 0.0009 | Genetic Heights | Shri -- 6enetic Heights | Flat No 302, Orange Homes, Adorn, K V R Rainbowcolony, Qutubullapur, K.V.Rangareddy, Telangana | 298, Qutubullapur, K.v.rangareddy, TG |
| S1-324704727 | S2-4097610 | 0.0009 | OQ Purpose Co | Dovabelo | D-7, Phase Iii Garden Homes Alkapuri, Gwalior, Madhya Pradesh | H.NO 87 D-7, GWALIOR, Madhya Pradesh |
| S1-900817625 | S3-772786324 | 0.001 | Cameron Allied Priority | Cameron  Priority Partners | N4231 C, Town Of Freedom, WI | N423 C, PMB 8248, Kaukauuna CDP, Wisconsin |
| S1-685048543 | S3-975683365 | 0.0022 | Ghanshyam Milk | Ghanshyam Me | 5Th Floor, 502, Meraki Arena, Cts No. 619/14 619/15 619/21A 21B, V N Purav Marg, Opp R K Studio, Chembur East, Mumbai, Maharashtra | Floor, Mumbai (SUBURBAN), MH |
| S1-977292711 | S2-256007340 | 0.0029 | Sai Innovative Exports Private Limited | ସାଇ ଇନୋଭେଟିଭ୍ ଏକ୍ସପୋର୍ଟସ୍ ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍ | Plot No-355, Rohini Villa, Taraboi, Baniatangi, Jatni, Khordha, Orissa | PCOT NO-00355, JATNI, JATANI, Orissa |
| S1-667264941 | S3-810666500 | 0.0038 | Bright Bright Corporate, Inc | Bright  Bright Cofeforate, Inc | 35142 Lead Line Road, Virden, IL |  |
| S1-916679619 | S3-677979213 | 0.0047 | Tramel Holding Company PLLC | Cal0evo | 300 Charles Street, Unit APT 201, Baltimore, MD | 0300 Charles St, Baltimore, Maryland |

### False merges on zero-match entities

| s1 | target | score | s1 name | target name | s1 address | target address |
|---|---|---|---|---|---|---|
| S1-116338782 | S2-402815945 | 0.9999 | Rocky Ministries Corp | Rocky Ministries Harbor Corp | 1026 1/2 Bridge Street, Unit 1/2, Columbia, TN | NULL, TN, 1026 1/15 BRIDGE STREET, COLUMBIA |
| S1-393259793 | S3-506660295 | 0.9945 | Innova Care Ltd | Innova  Marketing | 34, Pelican Ind Estate, Road No 5 Kathwada Gidc, Daskroi, Ahmedabad, Gujarat | 55, Pelican Ind Estate, Road No 5 Kathwada Gidc, Daskroi, Ahmedabad, GJ |
| S1-360795448 | S3-140801565 | 0.9519 | Department of Health | Department Housing-of | 464 Howard Hill Road, Newark Valley, NY |  |
| S1-360795448 | S2-283683612 | 0.9478 | Department of Health | Department of Health (Inc) | 464 Howard Hill Road, Newark Valley, NY |  |
| S1-268651992 | S2-648003261 | 0.8636 | Davidson Vulcan | Owens Vulcan | 501 Main Street, Loretto, TN | TN, LORETTO, 51 MAIN ST |

## 7. Test inference, submission and scalability

| item | value |
|---|---|
| test Source-1 entities | 5,000 |
| candidate pairs S2 / S3 | 302,270 / 278,157 |
| pairs dropped by the per-S1 cap S2 / S3 | 61,424 / 52,521 |
| threshold used | 0.8 |
| entities with at least one / at least two predictions | 799 / 163 |
| files | `phase2/output/phase2_5/frozen/output/dev/matching_results.tsv`, `phase2/output/phase2_5/frozen/output/dev/candidate_pairs.tsv` |
| section 45 checks | 1_every_s1_represented = pass, 3_no_malformed_ids = pass, 4_no_duplicate_ids_in_a_list = pass, 5_multi_match_serialization = pass, 6_zero_match_serialization = pass, 7_deterministic_sorted_lists = pass, matches_subset_of_candidates = pass, 2_ids_exist_in_s2_s3 = pass, 8_row_count = pass, 9_header = pass |
| challenge validator (with_candidates: `python.exe validate_submission.py --matching C:\Users\dbijp\OneDrive\Desktop\amazon_aws\phase2\output\phase2_5\frozen\output\dev\matching_results.tsv --test-dir C:\Users\dbijp\OneDrive\Desktop\amazon_aws\phase2\output\phase2_5\frozen\output\dev\test_subset --candidate C:\Users\dbijp\OneDrive\Desktop\amazon_aws\phase2\output\phase2_5\frozen\output\dev\candidate_pairs.tsv --check-ids`) | PASS:  / PASS — no blocking issues found. Safe to submit. |
| sha256 | matching_results.tsv b45ba669f8280d18..., candidate_pairs.tsv 001b58e1b5a93f76... |

Runtime per stage (seconds):

| stage | seconds |
|---|---|
| train_load_s1_gt | 7.0 |
| train_normalize_s1 | 1.7 |
| train_load_s2 | 10.8 |
| train_normalize_s2 | 6.2 |
| train_candidates_s2 | 6.6 |
| train_features_s2 | 5.0 |
| train_load_s3 | 11.2 |
| train_normalize_s3 | 5.4 |
| train_candidates_s3 | 6.3 |
| train_features_s3 | 4.4 |
| train_model | 3.8 |
| threshold_sweep | 0.1 |
| error_analysis | 21.0 |
| test_load_s1 | 3.4 |
| test_normalize_s1 | 0.5 |
| test_load_s2 | 10.1 |
| test_normalize_s2 | 5.7 |
| test_candidates_features_scores_s2 | 12.6 |
| test_load_s3 | 10.3 |
| test_normalize_s3 | 5.3 |
| test_candidates_features_scores_s3 | 11.9 |
| test_write_outputs | 0.3 |
| validator | 0.6 |

Memory of the normalized record tables (MB): {'s1_records_mb': 1.8, 's2_records_mb': 35.0, 's3_records_mb': 34.8}. Worker processes: 6; Source-1 chunk size 100,000; features are computed in chunks of 50,000 pairs. No step builds an S1-by-target product: candidates are equi-joins on blocking keys, and oversized keys are skipped.

## 8. Limitations

- Candidate recall is the ceiling. Positives that no rule retrieves (usually a heavily corrupted name together with a differently written address) cannot be predicted.
- Skipping high-frequency keys and capping candidates per S1 trade recall for bounded cost; both costs are measured and recorded.
- Indic romanization is a deterministic approximation. The consonant skeleton recovers many, but not all, transliterated spellings.
- One logistic-regression model and one threshold serve both target sources and all countries. France appears only in test; it is handled by the same country-agnostic rules but has no training examples.
- The spec's `city_equal` (16.8) is replaced by comma-component overlap, because the position of the city is not deterministic across sources.
