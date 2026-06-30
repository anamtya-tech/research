# 08 — Deployment & the Retraining Loop

**Goal:** get a fine-tuned model running inside ODAS, and keep improving it from field data.

## Deploying a release to ODAS

There are **two deployment paths** — know which you're using:

### (a) Manual (what `export_finetuned.py` prints)
Targets `~/sodas/` with the release filenames, then edit the cfg:
```bash
cp model_store/releases/v1.0.0/chatak_yamnet_v1.0.0.tflite ~/sodas/
cp model_store/releases/v1.0.0/custom_class_map.csv        ~/sodas/
# update the cfg:
raw.model_path = .../chatak_yamnet_v1.0.0.tflite
```
> ⚠️ **Caveat:** the firmware does **not** read a `chatak_yamnet_*.tflite` file or a
> `raw.class_map_path` key. `raw.model_path` is a **directory**, and the firmware loads the
> fixed names `yamnet_core.tflite` + `yamnet_class_map.csv` from it
> ([`04`](04_odas_processing.md), `mod_sst.c:377`). So this manual path only works if you rename
> the files to those fixed names inside the directory `raw.model_path` points at.

### (b) Programmatic (what the UI actually does) — recommended
`simulator/yamnet_finetuner.py:deploy_to_odas()` (`:810`) copies the active model into
`ODAS_MODELS_DIR = /home/azureuser/z_odas_newbeamform/models`, **renaming to the fixed names**
`yamnet_core.tflite` + `yamnet_class_map.csv` — so **no cfg edit is needed**. It backs up the
originals once to `yamnet_core_base.tflite` / `yamnet_class_map_base.csv`;
`restore_base_odas_model()` (`:839`) reverts. Source files come from the registry's active model
(`get_active_model_paths()`).

**Bottom line:** prefer path (b). If you deploy by hand, put `yamnet_core.tflite` +
`yamnet_class_map.csv` (the fixed names) into the `raw.model_path` directory.

## Verifying a deployment

```bash
# Standalone C++ classifier test (yamnet repo)
cd integration && bash build_api_test.sh && bash run_api_test.sh wavs/miaow_16k.wav
# Or run ODAS in sim mode on a known render and inspect the emitted classes
build/bin/odaslive -c ~/sodas/local_socket.cfg          # T1
python3 scripts/vm_socket_emit.py --audio known.raw --port 10000   # T2
```
Confirm: class names in the JSON `event_class_name` / `topk_history` match
`yamnet_class_map.csv`, and `num_classes` in the loaded model matches the class map length.

## The closed retraining loop

```
field/sim audio ─► ODAS (sim_mode=1) ─► .bin sidecars + session JSON
                                              │
   analyzer (match to GT) ◄────────────────────┘
        │
        ▼
   curator: clean+during_gt → training bucket;  ambiguous → unknown bucket (manual label)
        │                                              │
        └──────────► labels.csv (active dataset) ◄─────┘
                          │
                          ▼
   train_yamnet (2-phase) ─► export_finetuned ─► releases/<version>/*.tflite + class map + registry
                          │
                          ▼
   deploy_to_odas() ─► z_odas_newbeamform/models/{yamnet_core.tflite, yamnet_class_map.csv}
                          │
                          └────────────► back to ODAS (next round)
```

Each iteration: render or capture new scenes → run ODAS → analyze → curate (growing the active
dataset and clearing the unknown backlog) → retrain → export → redeploy. The `.bin` sidecars are
what make field recordings reusable as training data without re-recording.

## Operational guidance

- **Class set:** the active model is currently 4-class (Elephant/Frog/background/drone_bebop).
  Adding a class means new `label.txt` sources → scenes → curated samples → retrain (the head's
  `Dense(N)` changes, so warm-start auto-skips the old head).
- **`min_event_votes`:** run `=1` (collect-all) while building data; raise to `4` (4-of-6) once
  the model is good, to suppress false positives at the edge.
- **`sim_mode`:** `1` in the lab (writes `.bin` for curation), `0` on the Pi (no `.bin`,
  classification rides in `topk_history`).
- **Version discipline:** give each export a unique `--version`; the registry currently lets
  reused versions overwrite the deployed binary ([`07`](07_yamnet_training.md) gotchas).
- **Keep front-ends in sync:** STFT/mel constants in `data_loader.py`, the C wrapper
  (`yamnet_classifier.cpp`), and the Python helper must match, or train/inference drift.

## The full step set

[`00`](00_overview.md) overview · [`01`](01_data_prep.md) data prep ·
[`02`](02_scene_configuration.md) scene config · [`03`](03_audio_rendering.md) rendering ·
[`04`](04_odas_processing.md) ODAS · [`05`](05_analysis_and_gt_matching.md) analysis/GT ·
[`06`](06_yamnet_dataset_curation.md) curation · [`07`](07_yamnet_training.md) training ·
**08 deployment & loop**.
