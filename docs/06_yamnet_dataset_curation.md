# 06 — YAMNet Dataset Curation

**Component:** `simulator/yamnet_dataset_curator.py` (`YAMNetDatasetCurator`), with
`audio_reconstructor.py`. Alternative clean path: `gt_dataset_builder.py`.
**Input:** analyzer matches + the ODAS `.bin` spectra sidecars.
**Output:** `outputs/yamnet_datasets/<name>/` — `audio/*.wav`, `bins/*.bin`, `labels.csv`.
**Goal:** turn matched detections into a YAMNet-ready training set, routing ambiguous ones to a
manual-review "unknown" bucket.

## Two datasets: active vs unknown

`curator_config.json` tracks an `active_dataset` (training samples) and an `unknown_dataset`
(needs manual verification), plus per-dataset `sample_count`, `samples_by_label`,
`runs_processed`, `curation_log`, and the `curation_criteria` / `audio_params`. Helpers
`get/set_active_dataset` and `get/set_unknown_dataset` (`:138`).

## Keep / route decision

`curate_from_analysis` (`:193`) applies, per match, criteria from `curator_config.json`
(defaults: `confidence_threshold=0.75`, `direction_threshold_deg=15.0`, `min_activity=0.01`,
`min_spectral_bins=1`, `save_unknown=True`, `allow_render_fallback=False`,
`echo_pad_seconds=0.5`):

1. **Skip** if `activity < min_activity` and no `.bin` exists (`:239`).
2. A **clean match** requires *all* of: spatially aligned (`angular_error ≤
   direction_threshold_deg`), temporally aligned (`match_type=='ground_truth'` **and**
   `patch_quality=='during_gt'`), enough spectral frames (`spectral_count ≥ min_spectral_bins`),
   and `ground_truth != 'unknown'` (`:245`).
3. **Not clean → unknown bucket** with a `curation_reason` (`direction_error_Xdeg`, `pre_gt`,
   `post_gt_reverb`, `sparse_spectra_Nbins`, `no_ground_truth`) for manual labeling (`:280`).
4. **Clean *and* has a learnable issue** (unclassified, `yamnet_conf < threshold`, or label
   mismatch) → **training bucket** (`:304`). Clean with no issue → skipped (already correct).

`curate_ambient_as_background` (`:1079`) forces every detection from an ambient-only run to
`label='background'` — cheap **hard negatives**.

## Audio reconstruction from `.bin` spectra

The `.bin` sidecar is raw `float32`, shape `n_frames × 257` (257 = 512-pt FFT half-spectrum +1,
linear magnitude @ 16 kHz). The firmware writes a 96-frame (~480 ms) patch per YAMNet hop and
fires every 48 frames (**50 % overlap**).

`_save_samples` (`:391`) groups samples by **GT window** `(label, gt_start, gt_end)` to merge
flickering Kalman track-ids, then stitches that window's `.bin` files:

- **Priority 1** (deployment-faithful): `.bin` → Griffin-Lim via
  `AudioReconstructor.reconstruct_from_spectra_files` (`audio_reconstructor.py:231`).
- **Priority 2** (opt-in `allow_render_fallback`): extract from the raw render PCM.

`AudioReconstructor` (`audio_reconstructor.py:19`) → **16 kHz mono** WAV via Griffin-Lim
(`_griffin_lim_multi_frame`, `:135`; `n_fft=512`, `hop_length=128`). It **dedups the 50 %
overlap**: keep all 96 frames of the first `.bin`, append only the new 48 of each subsequent
one (`:303`) — otherwise the clip is time-stretched 3–6×. The curator pins `n_fft=512,
hop_length=128` (NOT 1024/512) for the same reason (`:51`).

## Output layout

Per dataset under `outputs/yamnet_datasets/<name>/`:

| Path | Contents |
|---|---|
| `audio/*.wav` | reconstructed 16 kHz mono clips |
| `bins/*.bin` | stitched `(N×96)×257` float32 — **the actual YAMNet training input** |
| `spectrograms/*.png` | visual QA |
| `metadata/<run>_<ts>.csv` | per-run record |
| **`labels.csv`** | master label file (`_save_samples`, `:647`) |

`labels.csv` columns (`:603`): `filename, spectra_file, n_frames, run_id, timestamp, label,
yamnet_class, yamnet_confidence, yamnet_votes, yamnet_ambiguous, top_k_candidates,
ground_truth, curation_reason, activity, spectral_count, patch_quality, n_stitched_bins,
stitched_duration_s, position{x,y,z}, confidence, angular_error, dataset_type, clean_match,
manual_verification_needed, fold` (default `train`).

## Deployment metrics (precision/recall/F1)

These live in the curator, not the analyzer — `compute_deployment_metrics` (`:1136`):
- event-level **P / R / F1** (a GT event is a TP if ≥1 detection matched it),
- `fp_per_min` (FP detections per quiet minute),
- `correct_class_and_direction` (% of GT matches where `yamnet_class==label` **and**
  `angular_error ≤ threshold`),
- a `confusion` matrix and `per_label` detected/total.
Surfaced in the analyzer UI via `_render_deployment_eval` (`analyzer.py:3216`).

## Alternative: `gt_dataset_builder.py` (ODAS-free clean path)

`GTDatasetBuilder` bypasses ODAS entirely and builds a dataset **directly from the renderer's
per-source sidecars** (isolated RIR-processed mic signals + ambient background):

- chunks into **3.0 s windows at 1.5 s hop**, 16 kHz mono (`YAMNET_WINDOW_S/HOP_S`, `:53`),
- `_chunk_and_save` (`:242`) emits `src{NN}_clip{NNNN}.wav`,
- `_save_manifest` (`:329`) writes `manifest.csv` (`wav_path, label, source_idx, scene, fold`)
  with **source-level** train/val/test splitting (seed 42, no leakage) + `dataset_info.json`.

Use this when you want clean, perfectly-labelled training data without ODAS reconstruction
noise; use the curator when you want data that matches the on-device `.bin` → reconstruction
path (deployment-faithful, including reconstruction artifacts the model should learn to handle).

## Next

[`07_yamnet_training.md`](07_yamnet_training.md) — fine-tune YAMNet on this `labels.csv`
dataset and export a deployable TFLite.
