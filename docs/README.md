# Chatak Pipeline — Step Docs

Stepwise documentation of the Chatak wildlife acoustic-monitoring pipeline, derived from the
actual code in `chatak-odas/`, `simulator/`, `yamnet/`, and the `data_prep/` work in this repo.
Each file documents one step: goal, inputs/outputs, key code references, commands, and gotchas.

## 📄 Read this first — [`REPORT.md`](REPORT.md)
**The single end-to-end, plain-language report** — what we built, every experiment, findings,
figures, and conclusions. Start there. The docs below are the detailed step/experiment references.

**What's left:** [`ROADMAP.md`](ROADMAP.md) — done vs remaining, prioritized.

---

Pipeline internals start at [`00_overview.md`](00_overview.md). For the `plan.md` experiment
results (Level-0 warm-up: baseline → embeddings → linear probe → MLP), see
[`L0_plan_results.md`](L0_plan_results.md).

| # | Doc | Subsystem |
|---|---|---|
| 00 | [overview](00_overview.md) | architecture, the closed loop, hardware/signal params, glossary |
| 01 | [data prep](01_data_prep.md) | annotated positive/background clip inventory (`data_prep/`) |
| 02 | [scene configuration](02_scene_configuration.md) | `label.txt` source library + scene JSON (`configurator.py`) |
| 03 | [audio rendering](03_audio_rendering.md) | pyroomacoustics → 6-ch raw + GT sidecars (`renderer.py`) |
| 04 | [ODAS processing](04_odas_processing.md) | C firmware SSL→SST→YAMNet + sim orchestration |
| 05 | [analysis & GT matching](05_analysis_and_gt_matching.md) | detection ↔ ground-truth (`analyzer.py`, timing) |
| 06 | [dataset curation](06_yamnet_dataset_curation.md) | `.bin` → WAV + `labels.csv` (`yamnet_dataset_curator.py`) |
| 07 | [YAMNet training](07_yamnet_training.md) | export base → 2-phase fine-tune → TFLite (`yamnet/`) |
| 08 | [deployment & loop](08_deployment_and_loop.md) | deploy to ODAS + retraining loop |

### plan.md experiment legs (Level-0 warm-up, results in `experiments/outputs/`)

| Leg | Doc | Finding |
|---|---|---|
| L0 | [results summary](L0_plan_results.md) | embedding space is excellent; only the head needs adapting |
| L1 | [YAMNet baseline](L1_yamnet_baseline.md) | raw 521-class head doesn't recognize our classes |
| L2 | [embedding viz](L2_embedding_viz.md) | UMAP/t-SNE — 4/5 classes cluster cleanly (kNN purity 0.80) |
| L3 | [linear probe](L3_linear_probe.md) | LogReg AUC 0.90–1.00, all PASS |
| L4 | [MLP head](L4_mlp_head.md) | no gain over linear → bottleneck is features, not the head |

### experiments.pdf production program (results as completed)

| Exp | Doc | Finding |
|---|---|---|
| infra | [ODAS bring-up](ODAS_BRINGUP.md) | odaslive built + smoke-tested in arm64 Linux Docker — unblocks Track B / FP-min |
| A1 | [EXP-A1 no-ambient baseline (Track A)](P1_exp_a1.md) | GT model: internal clip-acc 0.71–0.81, holdout 0.60 (clean clips) |
| A1·B | [EXP-A1 Track B — post-ODAS](P1_exp_a1_trackB.md) | **FP/min ~20** (binding constraint); post-ODAS training beats GT on deployment holdout **0.48 vs 0.36** (hypothesis confirmed) |
| A1·SST | [EXP-A1 SST preset sweep](P1_sst_sweep.md) | SST tuning: Low-FP best at **14 FP/min** (still 7× over ≤2 target) → tuning alone insufficient, confirms structural floor; hard negatives needed |
| A1·FP | [EXP-A1 FP cause analysis](P1_fp_cause_analysis.md) | FPs are SSL/array-geometry (not Kalman): ~⅓ null-DOA artifact (free to filter), rest = mic-baseline hotspots (±90 Y, 180, ±120); votes≥2 → 0.7 FP/min but recall 0.65; probMin ineffective |
| B4 | [EXP-B4 hard negatives](P2_exp_b4.md) | structural ghosts→`background`: FP/min **8.1→4.4**. **Key finding:** real-ambient negatives *don't* transfer to our no-ambient holdout → FP testbed wasn't deployment-representative; **run A5** |
| A5 | [EXP-A5 real-ambient testbed + B4](P1_exp_a5.md) | real ambient: **51 FP/min**, recall 0.62 (A1 numbers were artifacts). On it, hard negatives **cut FP 48.5→9.2 (−81%)** — B4 validated; new constraint = classification accuracy in ambient (0.33) |
| corpus | [Persistent corpus + multi-SNR + growth](P2_corpus_multisnr.md) | one growing labeled corpus (**7,251** samples, 4 envs, 3 SNRs, balanced) + 1-command collective train. **Decisive finding:** data growth exhausted — linear head is capacity-saturated; **next lever = backbone fine-tuning** |
| FT | [Backbone fine-tuning + SpecAugment (separate track)](P3_backbone_finetune.md) | standalone script (no repo edits). **Broke the ceiling: test 0.48→0.62, gap 0.33→0.13** (Lion 0.90, Frog 0.68, Elephant/Bear/drone_bebop ~0.54). SpecAugment — not capacity/weight-decay — was the lever (constraint was cross-render generalization). drone_binary still lags |

## Notes on accuracy

These docs flag several places where the code diverges from the repos' own READMEs (verified
while reading the source):
- the source library is a `label.txt` filesystem scan, **not** `config/sources.csv` ([02](02_scene_configuration.md))
- render output is `{scene}_{timestamp}.raw`, not `_ChatakX_sim.raw` ([03](03_audio_rendering.md))
- ODAS emits events **every 48 ms while a track is alive**, not only at track-end ([04](04_odas_processing.md))
- there is **no** `raw.class_map_path` cfg key; `raw.model_path` is a directory ([04](04_odas_processing.md), [08](08_deployment_and_loop.md))
- `registry.json.val_accuracy` actually stores **test** accuracy ([07](07_yamnet_training.md))
