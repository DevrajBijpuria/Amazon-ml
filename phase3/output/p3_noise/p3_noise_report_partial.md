# P3-N Retraining Noise: partial report (stopped after 4 of 13 runs)

Status: **PARTIAL.** Stopped by the user on 2026-09-25 after the 50k gate and the three 1k runs. The 10k runs (5)
and the 50k non-42 runs (4) were not run. Machine-readable results: `p3_noise_summary_partial.json`.

## Design (spec amendment B)
- Experiment seed 42 fixed: S1 sample, validation split. Blocking and hard negatives are deterministic.
- `negative_sample_seed` controls the easy-negative draw only. `model_seed` controls HGB `random_state` only.
  Both are varied together (k/k).
- Statistic label: **combined retraining noise from easy-negative sampling + model-training randomness.** It is not
  pure model-seed noise, and the two contributions are not separately identified.
- Code change: `phase3/src/experiment_runner.py` only (seed wrappers restored in `finally`, `seed_<k>` run
  directory, manifest fields). Configs: `phase3/config/p3_noise_seed{42,1,2,3,4}.yaml`. No Phase 2 file changed.

## Seed-42 gate (50k, 42/42): PASSED
Official metric (`threshold_tuning.evaluate` / `entity_f05`): macro F0.5 **0.9309412798962662**, TP 30125,
FP 916, FN 4476, 10,000 entities, threshold 0.8. 12/12 artifacts byte-identical to
`phase3/output/p3_frozen_baseline/50k/run_1` (validation scores, entities, model, thresholds, candidate statistics,
matching/candidate files). `metrics.json` diff vs frozen: 0.0 [0.0, 0.0].

## Per-run results (all valid)
| scale | seeds | macro F0.5 | precision | recall | TP | FP | FN | thr | runtime s | validator | 4.5 | fixed structure |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50k | 42/42 | 0.9309412799 | 0.9705 | 0.8706 | 30125 | 916 | 4476 | 0.80 | 1207 | PASS | PASS | = frozen (byte) |
| 1k | 42/42 | 0.9169000842 | 0.9588 | 0.8755 | 605 | 26 | 86 | 0.70 | 910 | PASS | PASS | = frozen 1k (byte) |
| 1k | 1/1 | 0.9111497177 | 0.9700 | 0.8437 | 583 | 18 | 108 | 0.85 | 1072 | PASS | PASS | = 1k 42/42 |
| 1k | 2/2 | 0.9220757789 | 0.9879 | 0.8263 | 571 | 7 | 120 | 0.90 | 1046 | PASS | PASS | = 1k 42/42 |

"Fixed structure" means validation entities, candidate statistics, feature statistics, candidate_pairs.tsv, the
(s1, target, source, label) structure of the scored pairs, and training-candidate counts are identical. Only the
model and scores differ. The 1k runs have 200 validation entities.

## 1k combined retraining noise (n = 3)
| metric | mean | SD | min | max | range | Δ 1/1 vs 42 | Δ 2/2 vs 42 |
|---|---|---|---|---|---|---|---|
| macro F0.5 | 0.91671 | **0.00547** | 0.91115 | 0.92208 | 0.01093 | −0.00575 | +0.00518 |
| precision | 0.97224 | 0.01467 | 0.95880 | 0.98789 | 0.02909 | +0.01125 | +0.02909 |
| recall | 0.84853 | 0.02495 | 0.82634 | 0.87554 | 0.04920 | −0.03184 | −0.04920 |
| TP / FP / FN | 586.3 / 17.0 / 104.7 | 17.2 / 9.5 / 17.2 | | | 34 / 19 / 34 | −22 / −8 / +22 | −34 / −19 / +34 |

## Comparison with the empty-address validation counterfactuals (not leaderboard gains)
Reference effects (50k, validation-selected): +0.00398 (empty address ≥ 0.95) and +0.00432 (never predict).
- **Measured:** at 1k, combined retraining noise has SD 0.0055 and range 0.0109 in macro F0.5. Both are larger
  than the counterfactual effects. The selected threshold moved 0.70 → 0.85 → 0.90 across seeds.
- **Not comparable at the same scale:** the counterfactuals are 50k effects on 10,000 entities. The 1k noise is
  measured on 200 entities with far fewer training rows, so it is expected to exceed 50k noise. The 50k noise
  floor (σ at 50k) was **not measured**, because the 50k seeds 1–4 were not run.
- **Inference:** P3-N cannot confirm or refute that a ~+0.004 effect exceeds 50k retraining noise.

## Validation, safety and remaining uncertainty
- All 4 runs: exit 0, validator PASS, Section 4.5 all pass (`2_ids_exist_in_s2_s3` is null by design, left to the
  validator). Runs were launched one at a time with no concurrency. 285 protected files (frozen baseline,
  Phase 2.5, all other Phase 3 outputs and configs, Phase 1/2 source, validator) unchanged by SHA-256 after every run.
- Uncertainty: n = 3 at 1k only. No 10k or 50k spread. Negative-sampling and HGB contributions are not separated.
  The SD from n = 3 is itself very uncertain.
- **P3-N is not complete** (Definition of Done: 10k and 50k matrix rows not met).
