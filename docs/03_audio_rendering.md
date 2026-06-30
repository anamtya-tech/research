# 03 — Audio Rendering

**Component:** `simulator/renderer.py` (`AudioRenderer`), Streamlit page "Audio Renderer".
**Input:** a scene JSON ([`02`](02_scene_configuration.md)).
**Output:** `outputs/renders/{scene}_{YYYYMMDD_HHMMSS}.raw` (6-ch S16_LE 16 kHz) + metadata
JSON + per-source ground-truth `.f32` sidecars.
**Goal:** synthesize realistic multi-channel mic-array audio from the scene's known source
positions, using room-acoustics simulation, so the rest of the pipeline has a signal with a
known right answer.

## Mic array geometry (ReSpeaker USB 4-Mic, 64 mm)

4 mics on the axes at radius 0.032 m, array at room center, **z = 1.5 m** (`renderer.py:39`, `:335`):

```
Mic 1 (→ Ch2) [-0.032, 0, 0]  Left
Mic 2 (→ Ch3) [ 0,-0.032, 0]  Back
Mic 3 (→ Ch4) [ 0.032, 0, 0]  Right
Mic 4 (→ Ch5) [ 0, 0.032, 0]  Front
```

## Room simulation (pyroomacoustics)

- **One `ShoeBox` room per source**, simulated then freed → bounded memory (`:340`, `:470`).
- Dims default `max_radius*2.5` (X,Y), `max(20, max_height*1.5)` (Z), overridable (`:223`).
- `absorption` default **0.7** (forest = open/absorptive), single scalar for all 6 walls via
  `pra.Material` (`:227`) — no frequency dependence.
- `max_order` default **3** (reflection order, `:229`).
- Source position shifted by room/2 then clipped inside the room (`:436`).

## Source handling

- **Directional:** clip → `librosa.load` 16 kHz mono → trim/loop/pad to `[start,end]` window
  (`:380`) → convolved through the room → written into the mic accumulator at `start_sample`
  (`:447`).
- **Ambient (synthetic):** looped to full duration, summed, added **identically to all 4 mics**
  — no spatialization (`:556`).
- **Ambient (capture):** real 6-ch `.raw`; channels 1–4 used as mics, de-spiked at ±6σ,
  per-channel RMS-normalized, mixed onto the mics (`:478`).

## Level normalization

`_apply_level_normalization` (`:71`) / `_resolve_source_level` (`:50`):
- Uses each source's `spl_db_1m`; scene baseline = **median** of known SPLs (fallback 70 dB).
- `target_dbfs = -24 + (spl − baseline)`; clip RMS scaled to that target.
- Per-source **`volume`** applied *after* SPL normalization as a multiplicative gain
  (default 1.0 directional, 0.5 ambient).
- Final master normalize to peak 0.95 across all mics (`:594`).

## Warmup / tail

`WARMUP_SECONDS=10`, `TAIL_SECONDS=10` of silence are prepended/appended (`:600`) — **every
render is 20 s longer than `duration`.** Downstream timestamps must subtract `warmup_seconds`
(the analyzer does this). Sidecar `start_sample`/`end_sample` are relative to the **content
block** (pre-warmup), not the final file.

## Output format

- **File:** `{scene}_{timestamp}.raw` in `outputs/renders/`. (The README's `_ChatakX_sim.raw`
  name is stale.)
- **6-channel interleaved, S16_LE, 16 kHz**, written in 30 s chunks via memmap streaming.
- **Mic data in channels 1–4 (0-indexed) = Ch2–Ch5; Ch1 (idx 0) and Ch6 (idx 5) are zeros**
  (`:627`) — matches the physical ReSpeaker / ODAS channel map (`mapping.map = (2,3,4,5)`).
- `_save_as_wav` (`:687`) can export a 4-channel (mics-only) WAV on demand.

## Ground-truth sidecars (the key to clean labels)

Metadata JSON (same stem, `.json`) fields: `scene_name, timestamp, render_id, duration,
sample_rate, n_channels(6), format("S16_LE"), room_dimensions, absorption, max_order,
scene_file, output_file, warmup_seconds, tail_silence_seconds, source_sidecars[],
ambient_sidecar_path`.

- **Per-source GT:** `{stem}_src{NN}_{label}.f32` — float32 `(4, n_frames)`, the isolated
  RIR-processed mic signal for that one source (`:454`). Each entry records `source_idx, label,
  start_time, end_time, start_sample, end_sample, n_frames, audio_active_samples`.
- **Ambient GT:** `{stem}.ambient.f32` — float32 `(4, n_samples)`, background only.
- **GT reconstruction:** `clean_mic_for_source_i = src_i[mic] + ambient[mic, start:end]`.

These sidecars are what [`gt_dataset_builder.py`](06_yamnet_dataset_curation.md) uses for the
clean, ODAS-free training path.

## How to run

`streamlit run app.py` → **"🔊 Audio Renderer"** → pick scene → Render. No CLI; to call
`_render_scene(scene_dict, room_x, room_y, room_z, absorption, max_order, …)` (`:301`)
programmatically you must supply (or stub) a Streamlit `st` context.

## Gotchas

- **20 s warmup+tail** offset — the single most common source of timestamp confusion downstream.
- **No RNG seeding** — `"Random"` clip picks are frozen at scene-save, but room/placement
  randomness is not reproducible.
- RAM: UI warns >8 GB, hard `MemoryError` >16 GB (`:248`, `:328`); per-source `.f32` sidecars
  and the `.ambient.f32` persist next to the `.raw`.

## Next

[`04_odas_processing.md`](04_odas_processing.md) — feed this `.raw` to ODAS for
localisation, tracking, and on-device YAMNet classification.
