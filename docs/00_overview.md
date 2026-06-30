# 00 — System Overview

Chatak is a wildlife acoustic-monitoring system: a **ReSpeaker USB 4-mic array** runs
**ODAS** (Open embeddeD Audition System) on-device to localise, track, separate and
**classify** sound events with an embedded **YAMNet** TFLite model, emitting JSON events
(animal/drone/threat + direction) to a GUI/edge service.

This repo collects the three components plus a data-prep stage:

| Dir | Repo | Role |
|---|---|---|
| `chatak-odas/` | C firmware (fork of introlab/odas) | On-device SSL → SST → embedded YAMNet → event JSON + `.bin` sidecars |
| `simulator/` | Python / Streamlit | Generate **labelled training data**: scene → render → ODAS sim → analyze → curate |
| `yamnet/` | Python (TF) | Export base YAMNet → fine-tune on curated data → versioned TFLite → deploy to ODAS |
| `data_prep/` | Python (this work) | Annotated positive/background clip inventory (see [`01_data_prep.md`](01_data_prep.md)) |
| `backup/` | — | Prior outputs (renders, runs, analyses, models), reference audio caches |

## The closed loop

```
                ┌─────────────────────── simulator/ (Python) ───────────────────────┐
 sources.csv ─► configurator ─► renderer ─► 6-ch .raw ─► ODAS sim ─► analyzer ─► curator
   (audio       (scene JSON)   (pyroom-      (16 kHz,    (odaslive   (match to   (16 kHz
    library)                    acoustics)    S16_LE)     replay)     ground      mono WAV
                                                                      truth)      + labels.csv)
                                                                                      │
                                                                                      ▼
                ┌──────────────────────── yamnet/ (TF) ─────────────────────────┐
   TF-Hub ─► export_yamnet_core ─► base SavedModel/TFLite ─► train_yamnet ─► export_finetuned
                                                            (2-phase FT)    (versioned .tflite
                                                                             + class map + registry)
                                                                                      │
                                                                                      ▼
                ┌──────────────────── chatak-odas/ (C firmware) ────────────────┐
   Mic 6-ch ─► SSL ─► SST (Kalman) ─► YAMNet TFLite (every 48 ms, top-K vote) ─► event JSON
   16 kHz                              + .bin sidecars (raw spectra) ──────────────┘
                                       │
                                       └─► Unix/TCP socket ─► ChatakGUI / edge service
```

The `.bin` sidecars written on-device feed back into the simulator's curator, so field
recordings can become new training data — an incremental retraining loop.

## Hardware & fixed signal parameters

| Parameter | Value |
|---|---|
| Mic array | ReSpeaker USB 4-Mic Array — 4 mics, circular, 64 mm diameter |
| Channels | 6 (S16_LE): **ch 0 & 5 empty**, ch 1–4 = the 4 mics |
| Sample rate | 16 kHz (fixed end-to-end) |
| ODAS frame / hop | 256 / 128 samples (~8 ms frames) |
| Spectrum bins | **257** (512-pt FFT, half spectrum) — do **not** truncate to 128 |
| YAMNet input | 96 × 64 log-mel patch (25 ms frame, 10 ms hop) |
| YAMNet output | 521 AudioSet classes (sigmoid); fine-tuned head replaces this with Chatak classes |

## The plan vs the pipeline

[`plan.md`](../plan.md) is the **ML experiment plan** (data prep → YAMNet baseline → embedding
viz → linear probe → MLP head). It currently runs on the clean clip inventory in `data_prep/`
(easier than field audio) before moving to simulator-mixed scenes. These step docs document the
**production pipeline** the experiments feed into.

## Step docs

| # | Doc | Subsystem |
|---|---|---|
| 00 | this file | architecture & glossary |
| 01 | [`01_data_prep.md`](01_data_prep.md) | annotated clip inventory |
| 02 | [`02_scene_configuration.md`](02_scene_configuration.md) | sources library + scene config |
| 03 | [`03_audio_rendering.md`](03_audio_rendering.md) | pyroomacoustics render → 6-ch raw |
| 04 | [`04_odas_processing.md`](04_odas_processing.md) | ODAS firmware + simulator orchestration |
| 05 | [`05_analysis_and_gt_matching.md`](05_analysis_and_gt_matching.md) | detection ↔ ground-truth matching |
| 06 | [`06_yamnet_dataset_curation.md`](06_yamnet_dataset_curation.md) | curate WAV + labels.csv |
| 07 | [`07_yamnet_training.md`](07_yamnet_training.md) | export base → fine-tune → TFLite |
| 08 | [`08_deployment_and_loop.md`](08_deployment_and_loop.md) | deploy to ODAS + retraining loop |

## Glossary

- **SSL** — Sound Source Localisation (direction of arrival).
- **SST** — Sound Source Tracking (Kalman-tracked source tracks over time).
- **DOA** — Direction Of Arrival (azimuth/elevation of a source).
- **GT** — Ground Truth (the known scene source positions/labels the simulator placed).
- **`.bin` sidecar** — raw per-detection spectra dumped by ODAS for offline reconstruction/retraining.
- **Curator** — component that turns matched detections into a YAMNet training dataset.
