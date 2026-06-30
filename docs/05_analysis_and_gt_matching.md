# 05 — Analysis & Ground-Truth Matching

**Component:** `simulator/analyzer.py` (`ResultAnalyzer`), with `timing_compensator.py`.
**Input:** a run manifest (`outputs/runs/<run>.json`) + the scene file + ODAS session JSON.
**Output:** `outputs/analysis/<run_id>_{analysis.json, dataset.csv, report.html}`.
**Goal:** match each ODAS detection back to the known scene source that produced it, score the
spatial/temporal accuracy, and stage the matches for curation.

## Main flow

`ResultAnalyzer.render()` (`analyzer.py:71`) lists `outputs/runs/*.json` and calls
`_analyze_run` (`:436`):

1. Load `session_live_file`, `scene_file`, `warmup_seconds` from the run manifest (`:441`).
2. `_parse_odas_output` (`:512`) → detection records (deduped on `(track_id, hop_id)`, last wins).
3. `_match_detections_to_sources` (`:705`) → matched + unmatched records.
4. `_calculate_statistics` (`:1242`).
5. `_apply_yamnet_classifications` → `_save_analysis` → `_generate_html_report`
   → `_create_dataset` → optional `curator.curate_from_analysis()`.

## Reading the ODAS session JSON

`sst_session_live.json_*.json` is JSON-lines, one object per emitted frame
(`_parse_odas_output`, `:512`):

- `timeStamp` — a **cumulative 8 ms hop counter, not wall time**. Converted to scene time as
  `t = timeStamp * 0.008 - warmup_seconds` (`:540`).
- `src[]` — per tracked source: `id` (track id), `x/y/z` (unit **direction** vector),
  `activity` (Kalman 0–1), `frame_count`, `tag`, `type`.
- Event fields (new firmware top-K voting): `event_class_id/name/votes/avg_confidence/
  max_confidence/event_candidates[]`, plus `topk_history[]` (≤6 hops × top-5).
- `spectra_file` → the `.bin` sidecar path; `spectral_count` → number of real (non-zero)
  spectral frames in the 96-slot buffer (`:545`).

> `sst_classify_events_*.json` is recorded in the manifest but **not** separately parsed —
> classification is read from the per-`src` event fields embedded in the session JSON.

## Angular matching algorithm

`_match_detections_to_sources` (`:705`) — two-pass, **first-match-wins**:

- **Direction:** `_cartesian_to_spherical` (`:643`): `azimuth = atan2(y, x)`,
  `elevation = arcsin(z/r)`.
- **Time gate:** each GT source gets an asymmetric window `[start − pre, end + post]`
  (defaults `pre=2.0 s`, `post=3.0 s`), then `timing_compensator.check_temporal_overlap`
  (`:806`) confirms overlap using the *reconstructed capture interval* (absorbs YAMNet's
  ~960 ms latency).
- **Distance** — `CONFIG['use_azimuth_only_matching']` defaults **True** (`:828`):
  - `_azimuth_distance` (`:673`) — horizontal-plane only (the planar 4-mic array **cannot
    resolve elevation**), wrapped to [0, π] → degrees. ← default
  - `_angular_distance` (`:652`) — full 3-D great-circle via `arccos(dot(û₁, û₂))`.
- **Threshold:** `CONFIG['angle_threshold_deg'] = 15.0` (UI 1–45°); match if `diff ≤ threshold`.
- **Validity pre-filter** (new firmware, `:735`): drop detections with `event_class_id == -1`,
  `event_votes < 1`, or `event_avg_confidence < 0.1`.
- **Confidence** = mean of spatial (`cos(error)`, `_calculate_confidence` `:692`) and temporal
  overlap confidence.
- **Pass 2** (`:857`): leftover valid detections → records with `label='unknown'`,
  `match_type='unmatched'`.

## The timing problem (and the fix)

ODAS reports a *hop counter*, and a detection's reported time lags the real sound by a
**variable** stack of latencies: YAMNet needs **96 frames (~960 ms)** for its first
classification, then **48-frame (~480 ms)** hops; the 6-hop rolling vote adds up to ~288 ms;
Kalman warm-up keeps `activity` near zero early and keeps tracks alive after the sound ends.
A naive "detection time == sound time" comparison mis-aligns badly.

`TimingCompensator` (`timing_compensator.py:95`) reconstructs the true capture interval:

- Constants (`:101`): `ODAS_HOP=0.008`, `YAMNET_FRAME=0.010`, `PATCH=96`, `PATCH_HOP=48`,
  `ROLLING_HOPS=6`, `HOP_INTERVAL=0.048`.
- `normalize_odas_timestamp` (`:119`): `(ts − first_ts)·0.008 + offset` → seconds.
- `analyze_detection_timing` (`:138`): per detection back-computes latency (first
  classification assumes ~960 ms ±200; later ~480 ms ±100) → `estimated_sound_start`.
- `check_temporal_overlap` (`:244`): the gate the analyzer uses for matching.

## Statistics & outputs

`_calculate_statistics` (`:1242`) produces **spatial** stats only:
- `summary`: `total_detections, matched, unmatched, match_rate, avg_angular_error,
  avg_confidence, time_span_seconds, unique_sources`.
- `by_source`: per-label detection count and angular-error avg/min/max/std.

> **Precision / Recall / F1 / confusion / FP-per-min are NOT in analyzer.py** — they are
> computed by `YAMNetDatasetCurator.compute_deployment_metrics` (see
> [`06_yamnet_dataset_curation.md`](06_yamnet_dataset_curation.md)) and surfaced via
> `analyzer._render_deployment_eval` (`:3216`).

Files written under `outputs/analysis/<run_id>_*`:

| File | Function | Contents |
|---|---|---|
| `_analysis.json` | `_save_analysis` (`:1753`, atomic) | `analysis_id, render_id, run_id, scene_name, config, summary, by_source, model_stats, matches[], unmatched[]` |
| `_dataset.csv` | `_create_dataset` (`:1812`) | `[bin_0…bin_1023] + label, confidence, angular_error, timestamp` — **legacy 1024-bin** path, usually empty on new firmware |
| `_report.html` | `_generate_html_report` (`:1943`) | interactive Plotly dashboard |

Each match record (`_build_match_record`, `:1320`) carries: timestamp, detected & GT position,
`gt_start/end`, `detection_latency`, `patch_gt_overlap`, `patch_quality`
(`pre_gt`/`during_gt`/`post_gt`), `angular_error`, `confidence`, `spectral_count`,
`spectra_file`, event fields, `match_type`.

## Next

[`06_yamnet_dataset_curation.md`](06_yamnet_dataset_curation.md) — turn these matches into a
YAMNet training dataset (or route ambiguous ones to the unknown bucket).
