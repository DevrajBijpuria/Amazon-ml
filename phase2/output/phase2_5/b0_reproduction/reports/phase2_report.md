# Phase 2 Report: Baseline Entity Matching (dev mode)

Generated automatically by `phase2/src/run_phase2.py` from the JSON artifacts in `phase2/artifacts/`.

> **Development mode.** 5,000 sampled Source-1 entities per split and 100000 target records per source. The target pools are much smaller than the real sources, so candidate counts and precision here are not representative of full scale.

## 1. Objective

A baseline pipeline: normalization, blocking (S1 to S2 and S1 to S3 separately), candidate recall measurement, pair features, a logistic-regression scorer, a threshold chosen on validation macro F0.5 per Source-1 entity, zero/one/many-match assembly, and the two submission files.

## 2. Candidate generation

Every key is prefixed with the exact normalized country (trimmed and casefolded, an open set that is never mapped). Target keys holding more than 100 records are skipped, and each one is recorded with its frequency and the positives it cost. Each S1 entity keeps at most 60 candidates per target source (most rules first, then the best name or address edit similarity); the dropped pairs and their recall cost are recorded.

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

| metric | S2 | S3 |
|---|---|---|
| candidate_recall | 0.98103 | 0.98156 |
| precap_candidate_recall | 0.98235 | 0.98299 |
| positive_pairs_total | 8328 | 9055 |
| positive_pairs_recovered | 8170 | 8888 |
| candidate_pairs | 207436 | 196200 |
| mean_candidates_per_s1 | 41.487 | 39.24 |
| median_candidates_per_s1 | 52.0 | 45.0 |
| p95_candidates_per_s1 | 60.0 | 60.0 |
| p99_candidates_per_s1 | 60.0 | 60.0 |
| max_candidates_per_s1 | 60 | 60 |
| s1_with_zero_candidates | 3 | 5 |
| recall cost of the per-S1 cap | 0.00132 | 0.00144 |
| positives lost that shared a skipped key | 72 | 91 |

Combined candidate recall (S2 and S3): **98.13%**. This is the ceiling on what the model can recover.

Recall by rule, before the cap ("only" = positives that no other rule found):

| rule | S2 recall | S2 only | S3 recall | S3 only | S2 pairs | S3 pairs | S2 skipped keys | S3 skipped keys |
|---|---|---|---|---|---|---|---|---|
| exact_name | 0.59654 | 6 | 0.58034 | 10 | 7347 | 7754 | 0 | 0 |
| exact_address | 0.31088 | 2 | 0.38145 | 4 | 2597 | 3472 | 0 | 0 |
| numeric | 0.56976 | 11 | 0.54158 | 17 | 6654 | 6747 | 4 | 4 |
| name_prefix | 0.63509 | 24 | 0.62529 | 47 | 24858 | 27698 | 55 | 53 |
| address_prefix | 0.38713 | 4 | 0.38277 | 11 | 3600 | 4065 | 0 | 0 |
| name_ngram | 0.66559 | 74 | 0.65555 | 129 | 143250 | 121575 | 381 | 385 |
| name_phonetic | 0.61972 | 86 | 0.57637 | 45 | 8903 | 8919 | 0 | 0 |
| address_token | 0.86011 | 264 | 0.8201 | 268 | 159541 | 159913 | 167 | 182 |

Recall by S1 country and S1 match bucket:

| group | S2 recall | S3 recall |
|---|---|---|
| country = india | 0.96753 | 0.96468 |
| country = us | 0.99014 | 0.99253 |
| bucket = multi | 0.98083 | 0.98186 |
| bucket = one | 0.9927 | 0.96032 |

Every skipped key with its target frequency and lost positives is listed in `candidate_statistics.json` under `<source>.skipped_keys.<rule>.keys`.

## 3. Features

47 features, in the exact order stored in `artifacts/model/feature_schema.json`. A missing value never creates similarity (every similarity is 0 when either side is empty), and missingness has explicit indicator features.

| feature | purpose | positive mean | negative mean | correlation with label |
|---|---|---|---|---|
| name_norm_eq | normalized names identical | 0.2975 | 0.0033 | 0.4761 |
| name_key_eq | compact core names identical (suffix-, domain- and space-insensitive) | 0.5744 | 0.0122 | 0.6073 |
| name_token_jaccard | token overlap of normalized names | 0.6832 | 0.1589 | 0.4986 |
| name_core_token_jaccard | token overlap without legal suffixes | 0.73 | 0.1453 | 0.5236 |
| name_char3_jaccard | character 3-gram overlap (typo-tolerant) | 0.6882 | 0.1603 | 0.5091 |
| name_edit_sim | 1 - Levenshtein / max length | 0.7325 | 0.3309 | 0.3971 |
| name_translit_char3_jaccard | 3-gram overlap after Indic romanization | 0.7144 | 0.1643 | 0.5256 |
| name_translit_edit_sim | edit similarity after romanization | 0.7667 | 0.3385 | 0.4223 |
| name_token_set_ratio | order-insensitive token-set similarity of core names | 0.9214 | 0.4897 | 0.3777 |
| name_phonetic_eq | consonant skeletons identical | 0.6308 | 0.0203 | 0.5859 |
| name_phonetic_edit_sim | edit similarity of consonant skeletons (transliteration spelling) | 0.8662 | 0.3435 | 0.4505 |
| name_length_1 | S1 name length | 25.2857 | 25.0949 | 0.0047 |
| name_length_2 | target name length | 24.6139 | 25.9671 | -0.029 |
| name_length_ratio | shorter / longer name length | 0.8621 | 0.7107 | 0.1572 |
| name_length_difference | absolute length difference | 3.9416 | 9.1085 | -0.1448 |
| name_token_count_1 | S1 token count | 3.5658 | 3.5423 | 0.0049 |
| name_token_count_2 | target token count | 3.4251 | 3.5652 | -0.0241 |
| name_token_count_difference | token count difference | 0.5201 | 1.0699 | -0.1146 |
| legal_suffix_present_1 | S1 name has a legal suffix | 0.6644 | 0.646 | 0.0077 |
| legal_suffix_present_2 | target name has a legal suffix | 0.5696 | 0.5963 | -0.0109 |
| legal_suffix_equal | both have the same legal-suffix set | 0.4481 | 0.1556 | 0.1574 |
| name_script_mismatch | one name in Indic script, the other Latin | 0.0959 | 0.0622 | 0.0278 |
| name_missing_1 | S1 name empty | 0.0 | 0.0 | 0.0 |
| name_missing_2 | target name empty | 0.0 | 0.0 | 0.0 |
| address_norm_eq | normalized addresses identical | 0.2297 | 0.0 | 0.4702 |
| address_token_jaccard | address token overlap | 0.7697 | 0.1105 | 0.7189 |
| address_char3_jaccard | address 3-gram overlap | 0.7383 | 0.1184 | 0.6948 |
| address_edit_sim | address edit similarity | 0.7077 | 0.3065 | 0.4595 |
| address_component_jaccard | overlap of comma components (order-insensitive street/city/state parts) | 0.6759 | 0.1108 | 0.5308 |
| address_missing_1 | S1 address empty | 0.0 | 0.0 | 0.0 |
| address_missing_2 | target address empty | 0.0414 | 0.0184 | 0.0337 |
| address_missing_either | either address empty | 0.0414 | 0.0184 | 0.0337 |
| house_number_both_present | both house numbers extracted | 0.8719 | 0.9043 | -0.022 |
| house_number_equal | house numbers equal | 0.6656 | 0.0089 | 0.7036 |
| postal_both_present | both postal codes extracted | 0.0021 | 0.0001 | 0.0319 |
| postal_equal | postal codes equal | 0.0021 | 0.0 | 0.0437 |
| state_both_present | both states extracted | 0.9553 | 0.9797 | -0.0339 |
| state_equal | states equal (codes, English names and native scripts unified) | 0.9479 | 0.3049 | 0.2747 |
| country_equal | exact normalized country equal (open set) | 1.0 | 1.0 | 0.0 |
| target_source_is_s2 | target from Source 2 | 0.479 | 0.5155 | -0.0147 |
| target_source_is_s3 | target from Source 3 | 0.521 | 0.4845 | 0.0147 |
| matched_by_exact_name_block | retrieved by the exact-name block | 0.5993 | 0.0126 | 0.6221 |
| matched_by_exact_address_block | retrieved by the exact-address block | 0.3542 | 0.0001 | 0.5855 |
| matched_by_numeric_block | retrieved by the house-number block | 0.5654 | 0.0072 | 0.6501 |
| matched_by_prefix_block | retrieved by a name or address prefix block | 0.7876 | 0.0915 | 0.4295 |
| matched_by_ngram_block | retrieved by a rare-token or phonetic signature block | 0.9879 | 0.9484 | 0.0365 |
| number_of_blocks_that_retrieved_pair | how many independent blocking rules retrieved the pair | 4.6887 | 1.0648 | 0.8314 |

Highly correlated feature pairs (|r| >= 0.9): name_key_eq ~ matched_by_exact_name_block (0.98), name_token_jaccard ~ name_char3_jaccard (0.94), name_token_jaccard ~ name_translit_char3_jaccard (0.922), name_char3_jaccard ~ name_edit_sim (0.907), name_char3_jaccard ~ name_translit_char3_jaccard (0.983), name_edit_sim ~ name_translit_edit_sim (0.96), name_translit_char3_jaccard ~ name_translit_edit_sim (0.907), name_length_ratio ~ name_length_difference (-0.917), address_token_jaccard ~ address_char3_jaccard (0.957), address_missing_2 ~ address_missing_either (1.0), address_missing_2 ~ state_both_present (-0.951), address_missing_either ~ state_both_present (-0.951), house_number_equal ~ matched_by_numeric_block (0.914), target_source_is_s2 ~ target_source_is_s3 (-1.0).

## 4. Training

| item | value |
|---|---|
| model | StandardScaler + LogisticRegression (C = 1.0, lbfgs) |
| split | entity level: 4,000 training and 1,000 validation Source-1 entities; entities in both = 0 |
| training candidate pairs | 319,829 (13,662 positive) |
| pairs used to fit | 81,972 = 13,662 positives + 68,310 negatives |
| negative sampling | 5 negatives per positive, drawn from the training candidates; 50% are the most name-and-address-similar non-matches (hard negatives), the rest uniform (seed 42) |

Largest standardized coefficients:

| feature | coefficient |
|---|---|
| address_token_jaccard | 2.6442 |
| address_char3_jaccard | 2.2487 |
| name_edit_sim | -1.9156 |
| number_of_blocks_that_retrieved_pair | 1.8612 |
| state_equal | 1.8602 |
| name_translit_edit_sim | 1.5672 |
| house_number_equal | 1.5097 |
| name_phonetic_edit_sim | 1.4372 |
| name_token_set_ratio | 1.422 |
| legal_suffix_equal | 1.1338 |
| name_translit_char3_jaccard | 1.0347 |
| address_edit_sim | -1.0242 |

## 5. Validation

Selected threshold **0.7** by macro F0.5 per Source-1 entity (challenge metric), on 1,000 validation entities (83,807 candidate pairs). Every validation entity counts, including zero-match entities and entities with no candidates. Pooled FN include positives that blocking never retrieved.

| threshold | macro F0.5 | TP | FP | FN | precision | recall | pooled F0.5 | entities with prediction | avg predictions per entity | zero-match entities scored 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.8893 | 3384 | 535 | 84 | 0.8635 | 0.9758 | 0.8838 | 963 | 3.919 | 36/54 |
| 0.1 | 0.9248 | 3374 | 319 | 94 | 0.9136 | 0.9729 | 0.9249 | 957 | 3.693 | 42/54 |
| 0.15 | 0.9372 | 3370 | 237 | 98 | 0.9343 | 0.9717 | 0.9416 | 955 | 3.607 | 43/54 |
| 0.2 | 0.9477 | 3362 | 178 | 106 | 0.9497 | 0.9694 | 0.9536 | 954 | 3.54 | 44/54 |
| 0.25 | 0.9537 | 3358 | 143 | 110 | 0.9592 | 0.9683 | 0.961 | 953 | 3.501 | 45/54 |
| 0.3 | 0.9561 | 3353 | 126 | 115 | 0.9638 | 0.9668 | 0.9644 | 953 | 3.479 | 45/54 |
| 0.35 | 0.9592 | 3345 | 108 | 123 | 0.9687 | 0.9645 | 0.9679 | 953 | 3.453 | 45/54 |
| 0.4 | 0.9614 | 3337 | 98 | 131 | 0.9715 | 0.9622 | 0.9696 | 952 | 3.435 | 46/54 |
| 0.45 | 0.9625 | 3330 | 90 | 138 | 0.9737 | 0.9602 | 0.971 | 952 | 3.42 | 46/54 |
| 0.5 | 0.9645 | 3322 | 79 | 146 | 0.9768 | 0.9579 | 0.9729 | 952 | 3.401 | 46/54 |
| 0.55 | 0.9663 | 3310 | 70 | 158 | 0.9793 | 0.9544 | 0.9742 | 951 | 3.38 | 47/54 |
| 0.6 | 0.9684 | 3300 | 59 | 168 | 0.9824 | 0.9516 | 0.9761 | 950 | 3.359 | 48/54 |
| 0.65 | 0.9696 | 3294 | 51 | 174 | 0.9848 | 0.9498 | 0.9776 | 950 | 3.345 | 48/54 |
| 0.7 | 0.9709 | 3285 | 44 | 183 | 0.9868 | 0.9472 | 0.9786 | 949 | 3.329 | 49/54 |
| 0.75 | 0.9698 | 3274 | 43 | 194 | 0.987 | 0.9441 | 0.9781 | 949 | 3.317 | 49/54 |
| 0.8 | 0.9697 | 3258 | 37 | 210 | 0.9888 | 0.9394 | 0.9785 | 949 | 3.295 | 49/54 |
| 0.85 | 0.9693 | 3233 | 30 | 235 | 0.9908 | 0.9322 | 0.9785 | 945 | 3.263 | 51/54 |
| 0.9 | 0.9663 | 3193 | 23 | 275 | 0.9928 | 0.9207 | 0.9775 | 943 | 3.216 | 51/54 |
| 0.95 | 0.9603 | 3117 | 18 | 351 | 0.9943 | 0.8988 | 0.9736 | 942 | 3.135 | 51/54 |

At the selected threshold, by target source. Diagnostic only: an entity with no match in that source and no prediction scores 1.0 there, so these are not comparable to the challenge metric.

| source | macro F0.5 (that source only) | precision | recall |
|---|---|---|---|
| s2 | 0.9604 | 0.9863 | 0.9455 |
| s3 | 0.9615 | 0.9873 | 0.9488 |

Pair level on validation candidates only (this recall excludes pairs blocking never produced):

| threshold | TP | FP | FN | TN | precision | recall | F0.5 |
|---|---|---|---|---|---|---|---|
| at_0.5 | 3322 | 79 | 74 | 80332 | 0.9768 | 0.9782 | 0.9771 |
| at_selected | 3285 | 44 | 111 | 80367 | 0.9868 | 0.9673 | 0.9828 |

## 6. Error analysis (validation)

- False positives: 44.
- False negatives scored below the threshold: 111.
- False negatives never retrieved by blocking: 72.
- Zero-match entities: 54, of which 49 were correctly predicted empty.
- Multi-match entities: 898; mean true matches 3.808, mean predicted 3.649, entities with at least two predictions 872.

### False positives (highest scores)

| s1 | target | score | s1 name | target name | s1 address | target address |
|---|---|---|---|---|---|---|
| S1-116338782 | S2-402815945 | 0.9991 | Rocky Ministries Corp | Rocky Ministries Harbor Corp | 1026 1/2 Bridge Street, Unit 1/2, Columbia, TN | NULL, TN, 1026 1/15 BRIDGE STREET, COLUMBIA |
| S1-573427830 | S3-476424970 | 0.9987 | Wildlife Network | [LLC] Wildlife Network | 5587 Fox Chase Drive, Hokah, MN | 5596 Fox Chase Dr, Hokah, Minnesota |
| S1-792705717 | S2-555931962 | 0.9987 | Anderson Floating | Anderson  Floating Co | 866 30th Avenue, Santa Cruz, CA | 879 30TH AVE, SANTA CRUZ, CA |
| S1-680997139 | S2-784387477 | 0.9985 | Consultants Godbole (india) Pvt Ltd | Consultants Godbole (india) Group Pvt Ltd | C/O- Manoj Kumar Singh, Bhagwanpur, Mushahari, Muzaffarpur, Bihar | H.NO 69/5 C/O- MANOJ KUMAR SINGH, BHAGWANPUR, MUSHAHARI, Bihar |
| S1-607139509 | S2-346699671 | 0.9977 | Apex Elite Fox | APEX ELITE FOX GROUP | 137 Chesterfield Way, Folsom, CA | 141 CHESTERFIELD WAY, FOLSOM, CA |
| S1-503260283 | S3-551584731 | 0.9975 | Kai Shorey Rapid Microelectronics LLC | Kai Shorey Rapid  Microelectronics Greater LLC | 7 Cappo Lane, Collinsville, IL | Illinois, 9 Cappo Lane, Collinsville |
| S1-622183692 | S2-41584172 | 0.9951 | Technical & Brothers Limited | Technical & Brothers Infratech Limited | No.3, Gandhi Nagar 3Rd Street, Abdul Ameer Street Guduvanchery, Chengalpet, Chennai, Tamil Nadu | H.NO G-10 , GANDHI NAGAR 3RD STREET, ABDUL AMEER STREET GUDUVANCHERY, CHENGALPET, Tamil Nadu |
| S1-825411169 | S2-529566185 | 0.995 | Redstone Traders Private Limited | REDSTONE TRADERS ENTERPRISES PRIVATE LTD | 4 Hari Nivas C Road Churchgate Mumbai, Mumbai, Mumbai City, Maharashtra | NO 5 HARI NIVAS C ROAD CHURCHGATE MUMBAI, MUMBAI, Maharashtra |
| S1-904474730 | S2-326401761 | 0.9939 | Vivyan Ruiz Aluminum P.C. | Vivyan Ruiz Aluminum Ltd | 2815 Finwood Drive, Rosenberg, TX | 2818 FINWOOD DR, ROSENBERG, TX |
| S1-517433608 | S2-899475214 | 0.9917 | Lopez's Solutions | HOLDINGS LOPEZ'S SOLUTIONS | 9510 Rocky River Road, Harrisburg, NC | 9512 ROCKY RIVER RD, HARRISBURG, NC |

### False negatives scored below the threshold (lowest scores)

| s1 | target | score | s1 name | target name | s1 address | target address |
|---|---|---|---|---|---|---|
| S1-324704727 | S2-4097610 | 0.0003 | OQ Purpose Co | Dovabelo | D-7, Phase Iii Garden Homes Alkapuri, Gwalior, Madhya Pradesh | H.NO 87 D-7, GWALIOR, Madhya Pradesh |
| S1-690151152 | S2-228559678 | 0.0036 | Masco Foundation | Shri Masco Service | 167 Ground Floor Abhay Khand - 1, Indrapuram, Ghaziabad, Uttar Pradesh |  |
| S1-900817625 | S3-772786324 | 0.0077 | Cameron Allied Priority | Cameron  Priority Partners | N4231 C, Town Of Freedom, WI | N423 C, PMB 8248, Kaukauuna CDP, Wisconsin |
| S1-161559404 | S3-324979106 | 0.0101 | Talbott and Conaway Petroleum LLC | Viojax | 131 Gilliland Drive, Danville City, VA | Gilliland Drive, Danvlle Township, Virginia |
| S1-916679619 | S3-677979213 | 0.0131 | Tramel Holding Company PLLC | Cal0evo | 300 Charles Street, Unit APT 201, Baltimore, MD | 0300 Charles St, Baltimore, Maryland |
| S1-72130895 | S2-358057926 | 0.0173 | Pipefitters Union Local No 685 | Pipefitters Union Local No 685 Co \| www.pipefitter.com | 302 Euclid Avenue, Prospect Heights, IL |  |
| S1-878533471 | S3-521488495 | 0.0256 | Kolkata Child Limited | Zephorbixylo | Riaz Manzil, 139/1A Tiljala Road, Kolkata, Kolkata, Howrah, West Bengal | Riaz Manzil, Kolkata, পশ্চিমবঙ্গ |
| S1-407546990 | S2-668479738 | 0.0339 | Mogal Consultancy | Smt Mogal Consmlntcy | Covaitechpark, Fl.1, Tw.1, 758/2, 759/2, Kovaithirungr, Coimbatore South, Coimbatore, Tamil Nadu |  |
| S1-585930773 | S2-495281210 | 0.0348 | Oceanic Welfare Society | Oceanic Society (Partners) | Sr. No. 16, 19, 21, 83(P) &91, 98, Forest Trails Pebbles A Bldg, Flat No. A, -103, Paud, Pune, Maharashtra |  |
| S1-832809894 | S2-517737342 | 0.0354 | Deese Charter LLC | Deese LLC Services | 184 Windstone Crossing Trail, Troutman, NC |  |

### False negatives never retrieved by blocking

| s1 | target | score | s1 name | target name | s1 address | target address |
|---|---|---|---|---|---|---|
| S1-119033930 | S3-45249084 | nan | Royal Technology Private Limited | रॉयल टेक्नोलॉजी प्राइवेट लिमिटेड | G-8-A, No. 1, Bhagirath Palace, North East, Delhi, 1737 | 737, North East, Delhi, दिल्ली |
| S1-119033930 | S3-745743078 | nan | Royal Technology Private Limited | रॉयल टेक्नोलॉजी प्राइवेट लिमिटेड | G-8-A, No. 1, Bhagirath Palace, North East, Delhi, 1737 | 737, North East, Delhi, दिल्ली |
| S1-119187241 | S2-865100548 | nan | Silver Classic Industries Private Limited | சில்வர் கிளாசிக் இண்டஸ்ட்ரீஸ் பிரைவேட் லிமிடெட் | Dr.No.299, Anandha Nagar, Gudalur (N) And (S), Coimbatore North, Coimbatore, Tamil Nadu | COIMBATORE, COIMBATORE NORTH, DR.NO.299, தமிழ்நாடு |
| S1-174657 | S3-218448202 | nan | Elite Telecom Union | Avidova | 355 Russ Court, OR, Mcminnville | Oregon, Mcminnville, 355-357 Russ Ct |
| S1-189665260 | S2-775867191 | nan | Green Software Private Limited | கிரீன் சாஃப்ட்வேர் பிரைவேட் லிமிடெட் | 268/1, 2Nd Street Advocate Ramanathan Nagar, Tirupattur, Vellore, Tamil Nadu | தமிழ்நாடு, TIRUPATTUR, 268/1, VELLORE |
| S1-189665260 | S3-858920318 | nan | Green Software Private Limited | கிரீன் சாஃப்ட்வேர் பிரைவேட் லிமிடெட் | 268/1, 2Nd Street Advocate Ramanathan Nagar, Tirupattur, Vellore, Tamil Nadu | ##268/1, Tirupattur, Vellore, TN |
| S1-202025302 | S3-285987646 | nan | Oncology Associates LLC | Oncology Associates L.L.C. Trading | 2310 Iris Drive, Sierra Vista, AZ |  |
| S1-255217115 | S2-798805961 | nan | My Investments LLP | మై ఇన్వెస్ట్‌మెంట్స్ ఎల్‌ఎల్‌పీ | 22-6-270/3/8, Balala Shopping Mall, Machli Kaman, Charminar, Hyderabad, Telangana | 22-6-270/3/8, HYDERABAD, Telangana |
| S1-255555674 | S2-479065796 | nan | Ear Nose & Throat Associates LLC | Ear Nose & Throat | 5419 9th Street, Albuquerque, NM | 5419 NINTH SAINT, ALBUQUERQUE, NM |
| S1-261169281 | S2-927345730 | nan | U 3 K Steel | U 3 K | 5200 Clairemont Drive, Town Of Grand Chute, WI |  |

### Hardest retrieved positives (lowest scores)

| s1 | target | score | s1 name | target name | s1 address | target address |
|---|---|---|---|---|---|---|
| S1-324704727 | S2-4097610 | 0.0003 | OQ Purpose Co | Dovabelo | D-7, Phase Iii Garden Homes Alkapuri, Gwalior, Madhya Pradesh | H.NO 87 D-7, GWALIOR, Madhya Pradesh |
| S1-690151152 | S2-228559678 | 0.0036 | Masco Foundation | Shri Masco Service | 167 Ground Floor Abhay Khand - 1, Indrapuram, Ghaziabad, Uttar Pradesh |  |
| S1-900817625 | S3-772786324 | 0.0077 | Cameron Allied Priority | Cameron  Priority Partners | N4231 C, Town Of Freedom, WI | N423 C, PMB 8248, Kaukauuna CDP, Wisconsin |
| S1-161559404 | S3-324979106 | 0.0101 | Talbott and Conaway Petroleum LLC | Viojax | 131 Gilliland Drive, Danville City, VA | Gilliland Drive, Danvlle Township, Virginia |
| S1-916679619 | S3-677979213 | 0.0131 | Tramel Holding Company PLLC | Cal0evo | 300 Charles Street, Unit APT 201, Baltimore, MD | 0300 Charles St, Baltimore, Maryland |
| S1-72130895 | S2-358057926 | 0.0173 | Pipefitters Union Local No 685 | Pipefitters Union Local No 685 Co \| www.pipefitter.com | 302 Euclid Avenue, Prospect Heights, IL |  |
| S1-878533471 | S3-521488495 | 0.0256 | Kolkata Child Limited | Zephorbixylo | Riaz Manzil, 139/1A Tiljala Road, Kolkata, Kolkata, Howrah, West Bengal | Riaz Manzil, Kolkata, পশ্চিমবঙ্গ |
| S1-407546990 | S2-668479738 | 0.0339 | Mogal Consultancy | Smt Mogal Consmlntcy | Covaitechpark, Fl.1, Tw.1, 758/2, 759/2, Kovaithirungr, Coimbatore South, Coimbatore, Tamil Nadu |  |
| S1-585930773 | S2-495281210 | 0.0348 | Oceanic Welfare Society | Oceanic Society (Partners) | Sr. No. 16, 19, 21, 83(P) &91, 98, Forest Trails Pebbles A Bldg, Flat No. A, -103, Paud, Pune, Maharashtra |  |
| S1-832809894 | S2-517737342 | 0.0354 | Deese Charter LLC | Deese LLC Services | 184 Windstone Crossing Trail, Troutman, NC |  |

### False merges on zero-match entities

| s1 | target | score | s1 name | target name | s1 address | target address |
|---|---|---|---|---|---|---|
| S1-116338782 | S2-402815945 | 0.9991 | Rocky Ministries Corp | Rocky Ministries Harbor Corp | 1026 1/2 Bridge Street, Unit 1/2, Columbia, TN | NULL, TN, 1026 1/15 BRIDGE STREET, COLUMBIA |
| S1-268651992 | S2-648003261 | 0.9863 | Davidson Vulcan | Owens Vulcan | 501 Main Street, Loretto, TN | TN, LORETTO, 51 MAIN ST |
| S1-393259793 | S3-506660295 | 0.9852 | Innova Care Ltd | Innova  Marketing | 34, Pelican Ind Estate, Road No 5 Kathwada Gidc, Daskroi, Ahmedabad, Gujarat | 55, Pelican Ind Estate, Road No 5 Kathwada Gidc, Daskroi, Ahmedabad, GJ |
| S1-109597244 | S3-579085008 | 0.8384 | Al Properties Private Limited | 5un Properties | 8-2-624/A/B1, 1St Floor, Road No.10, Banjara Hills, Hyderabad, Telangana | Hyderabad, తెలంగాణ, 8-2-293/82/A/265J/1 Road No 10 Jubilee Hills |
| S1-714642946 | S2-367164368 | 0.818 | U 8 Clear Telecommunication LLC | Solon Ashford L.L.C. | 501 Rebecca Smith Way, Cecilton, MD | REBECCA SMITH WAY, CECILTON, MD |

## 7. Test inference, submission and scalability

| item | value |
|---|---|
| test Source-1 entities | 5,000 |
| candidate pairs S2 / S3 | 215,387 / 201,678 |
| pairs dropped by the per-S1 cap S2 / S3 | 130,063 / 108,490 |
| threshold used | 0.7 |
| entities with at least one / at least two predictions | 789 / 132 |
| files | `phase2/output/phase2_5/b0_reproduction/output/dev/matching_results.tsv`, `phase2/output/phase2_5/b0_reproduction/output/dev/candidate_pairs.tsv` |
| section 45 checks | 1_every_s1_represented = pass, 3_no_malformed_ids = pass, 4_no_duplicate_ids_in_a_list = pass, 5_multi_match_serialization = pass, 6_zero_match_serialization = pass, 7_deterministic_sorted_lists = pass, matches_subset_of_candidates = pass, 2_ids_exist_in_s2_s3 = pass, 8_row_count = pass, 9_header = pass |
| challenge validator (with_candidates: `python.exe validate_submission.py --matching C:\Users\dbijp\OneDrive\Desktop\amazon_aws\phase2\output\phase2_5\b0_reproduction\output\dev\matching_results.tsv --test-dir C:\Users\dbijp\OneDrive\Desktop\amazon_aws\phase2\output\phase2_5\b0_reproduction\output\dev\test_subset --candidate C:\Users\dbijp\OneDrive\Desktop\amazon_aws\phase2\output\phase2_5\b0_reproduction\output\dev\candidate_pairs.tsv --check-ids`) | PASS:  / PASS — no blocking issues found. Safe to submit. |
| sha256 | matching_results.tsv f82d3596ba8f2f23..., candidate_pairs.tsv 0957d652d05bc208... |

Runtime per stage (seconds):

| stage | seconds |
|---|---|
| train_load_s1_gt | 7.4 |
| train_normalize_s1 | 1.9 |
| train_load_s2 | 11.9 |
| train_normalize_s2 | 7.2 |
| train_candidates_s2 | 7.1 |
| train_features_s2 | 6.6 |
| train_load_s3 | 17.4 |
| train_normalize_s3 | 8.4 |
| train_candidates_s3 | 8.2 |
| train_features_s3 | 5.5 |
| train_model | 0.5 |
| threshold_sweep | 0.1 |
| error_analysis | 29.3 |
| test_load_s1 | 4.2 |
| test_normalize_s1 | 0.6 |
| test_load_s2 | 11.8 |
| test_normalize_s2 | 9.1 |
| test_candidates_features_scores_s2 | 15.2 |
| test_load_s3 | 13.2 |
| test_normalize_s3 | 6.8 |
| test_candidates_features_scores_s3 | 11.7 |
| test_write_outputs | 0.4 |
| validator | 0.7 |

Memory of the normalized record tables (MB): {'s1_records_mb': 1.8, 's2_records_mb': 35.0, 's3_records_mb': 34.8}. Worker processes: 6; Source-1 chunk size 100,000; features are computed in chunks of 50,000 pairs. No step builds an S1-by-target product: candidates are equi-joins on blocking keys, and oversized keys are skipped.

## 8. Limitations

- Candidate recall is the ceiling. Positives that no rule retrieves (usually a heavily corrupted name together with a differently written address) cannot be predicted.
- Skipping high-frequency keys and capping candidates per S1 trade recall for bounded cost; both costs are measured and recorded.
- Indic romanization is a deterministic approximation. The consonant skeleton recovers many, but not all, transliterated spellings.
- One logistic-regression model and one threshold serve both target sources and all countries. France appears only in test; it is handled by the same country-agnostic rules but has no training examples.
- The spec's `city_equal` (16.8) is replaced by comma-component overlap, because the position of the city is not deterministic across sources.
