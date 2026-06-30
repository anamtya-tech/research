# Roadmap — What's Done and What Remains

Status of the Chatak acoustic-monitoring effort against `plan.md` (ML warm-up), `experiments.pdf`
(the production program), and the follow-on tracks our findings opened. See [`REPORT.md`](REPORT.md)
for the narrative.

## ✅ Done

| Area | Outcome | Doc |
|---|---|---|
| Data inventory | 450 clean clips, classes separable | [01](01_data_prep.md) |
| YAMNet sanity (plan.md L1–L4) | embeddings excellent; head retrain enough | [L0](L0_plan_results.md) |
| ODAS firmware running | built + smoke-tested in arm64 Docker | [ODAS](ODAS_BRINGUP.md) |
| EXP-A1 (post-ODAS vs clean) | post-ODAS training wins (0.48 vs 0.36) | [A1·B](P1_exp_a1_trackB.md) |
| FP root cause | array geometry (not Kalman); ⅓ null artifact | [FP](P1_fp_cause_analysis.md) |
| ODAS config tuning (SST, probMin, votes) | hygiene only; can't hit FP target without recall loss | [SST](P1_sst_sweep.md) |
| EXP-B4 hard negatives | reject ghosts by spectrum; testbed-mismatch caught | [B4](P2_exp_b4.md) |
| EXP-A5 real-ambient testbed | 51 FP/min real; hard negs **−81 %** (48.5→9.2) | [A5](P1_exp_a5.md) |
| Persistent corpus + growth | 7,251 samples, 4 envs, 3 SNRs, balanced; 1-cmd train | [corpus](P2_corpus_multisnr.md) |
| Data-growth ceiling found | linear head capacity-saturated → need backbone FT | [corpus](P2_corpus_multisnr.md) |
| Backbone fine-tuning unblocked | rare classes lifted; needed regularization | [P3](P3_backbone_finetune.md) |
| **Roadmap A — broke the accuracy ceiling** | FT + **SpecAugment**: test **0.48→0.62**, gap 0.33→0.13 (Lion 0.90, Frog 0.68) | [P3](P3_backbone_finetune.md) |

## 🔜 Remaining — prioritized

### A. Break the accuracy ceiling — ✅ DONE (test 0.48 → 0.62)
1. ✅ **Backbone fine-tune + SpecAugment broke the ceiling** ([P3](P3_backbone_finetune.md)):
   overall 0.62, gap closed to 0.13. SpecAugment (not capacity/weight-decay) was the lever — the
   constraint was cross-render generalization. *Remaining within A:*
   - **drone_binary** still weak (0.19, n=32) → more distinct source variety + targeted renders.
   - **Leakage-free render-level val** (current early-stop val is random patches from train renders).
   - **Larger balanced real-ambient test + more seeds** (635-sample holdout is noisy ±0.02–0.06).
   - Tune SpecAugment strength + unfreeze depth now that it clearly helps.

### B. False-positive handling (mechanism solved; finalize the operating point)
4. **Per-class + background-confidence thresholds** to set the FP↔recall operating point per use case.
5. **Pool hard negatives across many real sites (EXP-B5)** for cross-environment robustness; keep a
   small structural-silence set for true quiet.
6. **In-pipeline null/zero-activity filter** (≈⅓ of raw FPs are free to drop) + Low-FP SST preset.

### C. The rest of the experiments.pdf program (not yet run)
7. **Phase 1 A2–A4** (synthetic Rain/Wind/Bird ambient) — needs Rain/Bird source audio (we have Wind).
   *Lower priority:* real-ambient (A5) is the deployment-representative case and is done.
8. **Phase 2 B1/B2/B3/B5** (GT-only / ODAS-only / hybrid / pooled) — B2-style (post-ODAS) is largely
   covered; B3 hybrid (GT pretrain → ODAS fine-tune) and B5 pooled remain.
9. **Phase 3 C1/C2/C3** — C1 threshold sweep (done on A5), C2 per-class thresholds, C3 ODAS-preset
   comparison on the best model.
10. **Phase 4** — real-world Pi validation against natural ambient.

### D. Deployment & infrastructure
11. ✅ **Shipped (on branches):** exported the FT+SpecAugment model → TFLite (96×64→7-class,
    verified 0.635 on test patches), registered in `yamnet` `registry.json` (branch
    `ship/specaug-ft-registry`), and deployed into `chatak-odas/models/` replacing the stale
    4-class model (branch `ship/specaug-ft-model`; previous kept as `*_prev` for rollback).
    *Remaining:* push / open PRs after review; flip registry `deployed/active_model` once validated.
12. **Fold the backbone-FT fix back into `yamnet/training/train_yamnet.py`** (pin the
    `tensorflow/models` commit + run under `TF_USE_LEGACY_KERAS=1`) — once the separate track matures.
13. **Fix the `odaslive` shutdown segfault** (harmless for batch, matters for long-running capture).
14. **Standardize on a labeled-corpus convention** (the `experiments/corpus/` store) so all future
    runs accumulate and retrain collectively.

## Guard-rails / lessons to carry forward
- **Always measure FPs on real ambient**, never silent renders (A1 numbers were artifacts).
- **Data volume isn't the lever past balance** — difficulty diversity (SNR) and model capacity are.
- **FP and recall trade off**; report both, set thresholds deliberately.
- **Define a detection-range / SNR envelope** rather than chasing 100 % at −5 dB SNR.

## Notes on separation of work
- The backbone fine-tuning lives entirely in `experiments/scripts/finetune_backbone.py` — **no repo
  code modified**. All experiment scripts/data/figures are under `research/experiments/`; all docs
  under `research/docs/`. The chatak-odas / yamnet / simulator repos remain unchanged (built/populated
  only).
