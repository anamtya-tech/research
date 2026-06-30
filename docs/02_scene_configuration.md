# 02 — Scene Configuration

**Component:** `simulator/configurator.py` (`SceneConfigurator`), Streamlit page "Scene Configurator".
**Output:** a scene JSON in `config/scenes/<name>.json` describing known source positions/timing.
**Goal:** define a synthetic scene with **ground-truth** source placement so the renderer can
produce multi-channel audio and the analyzer later knows the right answer.

> **Correction to the README:** there is **no `config/sources.csv`** anymore. The source
> library is built by scanning a sounds directory tree for `.wav` files governed by `label.txt`
> files. Treat README mentions of `sources.csv` as stale.

## Source library — `label.txt` scan

- **Root:** `DEFAULT_SOUNDS_DIR = /home/azureuser/sounds` (`configurator.py:28`), overridable in
  the UI. Scanned by `_scan_library` (`:78`) / `_load_library` (`:145`).
- Each folder subtree carries a **`label.txt`** (deeper overrides ancestor):
  - **Line 1** — label name (e.g. `Lion`)
  - **Line 2** — `directional` | `ambient` (default `directional`)
  - **Line 3** *(optional)* — average **SPL @ 1 m in dB**; missing → `DEFAULT_SPL_DB_1M = 80.0`,
    flagged `spl_defaulted=True` (`:30, :102`)
  - Examples: `Lion\ndirectional\n115`, `Frog\ndirectional\n90`, `drone_binary\ndirectional`
- Per-label structure: `{source_type, files:[wav paths], spl_db_1m, spl_defaulted}` (`:130`).
  Same label across branches → files merged; explicit SPL beats a defaulted one.
- **directional** = positioned point source (rendered through pyroomacoustics);
  **ambient** = omnidirectional background mixed flat into all mics.
- Real ambient captures (`ambient_mode='capture'`): 6-ch `.raw` in
  `CAPTURES_DIR = /home/azureuser/audio_cache/ambient_captures` (`:29`).

## Scene JSON schema

Produced by `_create_default_scene` (`:45`) + editors; saved by `_save_scene` (`:1457`):

| Field | Type | Meaning |
|---|---|---|
| `name` | str | scene name (file = `{name}.json`) |
| `duration` | float s | content length, 1.0–3600 |
| `max_radius` | float m | placement bound, 1–500 |
| `max_height` / `min_height` | float m | 0–100 / −10–0 |
| `directional_sources` | list | positioned point sources (below) |
| `ambient_sources` | list | omnidirectional backgrounds (below) |
| `ambient_mode` | str | `'synthetic'` (default) or `'capture'` |
| `ambient_capture` | obj | capture mode only: `{path, start_offset, volume}` |
| `created_at`, `version` | str | added on save (`'1.0'`) |

**Directional source** (`:1051`): `label`, `wav_path` (abs path or `"Random"`, resolved & frozen
at save), `x/y/z` (m, cartesian), `start_time/end_time` (s), `repeat` (loop to fill window),
`spl_db_1m`, `spl_defaulted`, `volume` (default 1.0).

**Ambient source** (`:1081`): `label`, `wav_path`, `volume` (0.0–1.0, default 0.5), `spl_db_1m`,
`spl_defaulted`.

## Coordinate convention (read this)

- Stored **cartesian**; edited in the UI as **azimuth / distance / height**.
- `_azimuth_elevation_to_cartesian` (`:57`): `x = d·cos(az)`, `y = d·sin(az)`, `z = height`.
- **Azimuth is in the XY plane, degrees, −180…180, from +X.** The name says "elevation" but the
  third value is literally **height (z)**, not an elevation angle. The planar array can't resolve
  elevation anyway (see [`05`](05_analysis_and_gt_matching.md) azimuth-only matching).

## Two ways to build a scene

- **Manual** — add/edit one source at a time. `_add_directional_source(randomize=False)` (`:1038`)
  defaults to az=0, dist=`max_radius/2`, height=0, full-duration span; editor at `:548`.
- **Rich Scene Generator** (auto) — `_generate_rich_scene` (`:1183`), batch-schedules many
  directional instances under constraints:
  - counts: X unique clips/label, Y max simultaneous, Z total insertions (`:322`)
  - angular separation: hard min angle + near/mid/far buckets `[0–30]/[30–120]/[120–180]`,
    enforced by `_pick_azimuths` (`:1127`, circular distance, 300 attempts + fallback)
  - temporal: min gap between groups, ±start jitter, min occurrences/clip (`:404`)
  - spatial: min/max distance & height (`:449`)
  - under-represented labels/clips preferred via inverse-usage weighting (`:1238`)

`_visualize_scene` (`:1315`) renders a top-down map + Gantt timeline.

## How to run

`streamlit run app.py` → **"🎨 Scene Configurator"**. Runtime dirs are **hardcoded**
(`/home/azureuser/...`, `app.py:31`) — edit in source to run elsewhere. No CLI; the
scan/generate/save helpers assume a Streamlit `session_state` context.

## Gotchas

- **Non-deterministic:** the Rich generator and random placement use global `random` with **no
  seed** — runs differ. `"Random"` wav choices are frozen only at save.
- **Moving sources are not implemented** (TODO `:12`) — all sources are static.
- Hardcoded `/home/azureuser/*` paths; won't run as-is off that host.

## Next

[`03_audio_rendering.md`](03_audio_rendering.md) — turn this scene JSON into 6-channel raw audio
+ per-source ground-truth sidecars.
