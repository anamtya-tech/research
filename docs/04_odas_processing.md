# 04 — ODAS Processing (firmware + simulator orchestration)

**Components:** `chatak-odas/` (the C firmware = the real engine) and
`simulator/simulator.py` (`SimulationRunner`, the orchestration that feeds it a render).
**Input:** a 6-ch `.raw` (live mic on-device, or a render replayed over a socket in sim).
**Output:** `sst_session_live.json_*.json` (per-hop detections), `.bin` spectra sidecars
(sim only), and a run manifest in `outputs/runs/<run>.json`.
**Goal:** localise, track, and classify sound events on the mic array — the same code path on
the Raspberry Pi and in simulation.

## On-device event pipeline (the C firmware)

Audio math: 16 kHz, `hopSize=128` → 8 ms/frame; `frameSize=512` → **257** half-spectrum bins.
Per-track ring buffer = **96 frames × 257 floats** (~768 ms), `nFramesPerTrack=96`
(`mod_sst.c:55`).

```
Mic 6-ch ─► SSL (≤4 pots, GCC-PHAT/SRP) ─► SST (Kalman tracks)
              │  pot→track association (mod_sst.c:920), with a 1.2× "semantic boost"
              │  for tracks whose recent top-K hits a target animal class (mod_sst.c:931)
              ▼
       push_pot_to_track_buffer (mod_sst.c:1786): write half-spectrum at count%96
              │  fire a YAMNet hop when count≥96 and (count-96)%48==0  → 50% overlap
              ▼
       classify_track_hop (mod_sst.c:1640): build 96×257 patch → (sim) write .bin →
              yamnet_classify_patch_topk(TOP_K=5) → push topk_hop into per-track history
              ▼
       compute_event (mod_sst.c:1941): pool top-K across ≤6 stored hops (ROLLING_HOPS=6),
              winner = most distinct-hop votes, tie-break avg confidence
              ▼
       dump_track_buffers_to_json (mod_sst.c:2026): emit one JSON line per live track,
              gated to once per 6 hops = every 48 ms (mod_sst.c:1099)
              ▼
       Unix/TCP socket → ChatakGUI / simulator
```

### Two corrections vs the READMEs
- **Events are emitted continuously every 48 ms while a track is alive**, *not* only at
  track-end. The gate (`:2070`) fires whenever `topk_count ≥ 1` and `event_class_id ≥ 0`, and
  deliberately emits even when `votes < min_event_votes` ("emit all data, analyzer filters").
  Track-end only triggers `reset_track_slot` (`:1338`). There is **no** separate
  `sst_classify_events.json`-on-shutdown writer in this source.
- **There is no `raw.class_map_path` cfg key.** The class-map CSV name is hard-coded relative
  to the `raw.model_path` *directory* (see below).

### Emitted JSON per track (`dump_track_buffers_to_json`, `:2049`)
`timeStamp` (top level); per `src`: `id, tag, x/y/z` (direction), `activity`, `type`,
`frame_count`, `spectral_count`; event fields `event_class_id/name/votes/avg_confidence/
max_confidence`; `event_candidates[]` (ranked `{class_id, class_name, hop_votes,
avg_confidence}`); `topk_history[]` (per hop: `timestamp, class_ids[5], class_names[5],
confidences[5]`); and `spectra_file` (abs `.bin` path in sim, else `""`). Legacy
`bins[257]`/`fingerprint`/`class_*` fields are removed.

## YAMNet C integration (`src/yamnet/`)

- **TFLite C API** (`yamnet_classifier.cpp`): `TfLiteModelCreateFromFile` →
  `TfLiteInterpreterCreate` → `AllocateTensors` (`:312`). Reads the model's actual output class
  count at load (`:343`) so fine-tuned non-521 models work. `extract_scores` (`:23`) handles
  float32/int8/uint8 outputs.
- **Input is NOT raw 257 bins:** each of the 96 frames is a 257-bin magnitude spectrum that the
  wrapper converts to a **64-bin log-mel** (`SpectrumToMel`, `:126`; Hann, 125–7500 Hz,
  `LOG_OFFSET=0.001`) → model input is **96×64 log-mel**.
- **`ClassifyPatchTopK`** (`:188`): mel-convert → invoke → `partial_sort` → top-5 class ids +
  confidences. C wrapper `yamnet_classify_patch_topk` (`yamnet_wrapper.cpp:89`).
- **Model files** built from `raw.model_path` directory (`mod_sst.c:377`):
  `<model_path>/yamnet_core.tflite` + `<model_path>/yamnet_class_map.csv`. `LoadClassNames`
  (`:363`) reads the 3rd CSV column (`display_name`) or 2-col fallback.

## `.bin` spectra sidecars

- **Only in `sim_mode == 1`**, written in `classify_track_hop` *before* inference (`:1689`).
- Path `<classifier_log_dir>/patch_<trackID>_<timeStamp>.bin`.
- **Format:** flat `fwrite(patch, float, 96*257)` → exactly **96 × 257 float32** (≈96 KB), raw
  STFT magnitude (pre-mel), frame-major. Read in Python as
  `np.fromfile(p, np.float32).reshape(96, 257)`.
- **Purpose:** offline reconstruction/retraining (see [`06`](06_yamnet_dataset_curation.md)).
  On the Pi (`sim_mode=0`) nothing is written and consumers fall back to `topk_history`.

## Build (chatak-odas)

```bash
git submodule update --init --recursive   # cJSON
git lfs install && git lfs pull            # models/yamnet_core.tflite (~14 MB)
sudo apt install -y cmake libfftw3-dev libconfig-dev libasound2-dev \
                    libpulse-dev libjson-c-dev pkg-config
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)             # → build/bin/odaslive, build/lib/libodas.so
```
TFLite: aarch64 (Pi) uses the bundled `third_party/tflite/aarch64/libtensorflowlite_c.so` (TF
2.17, Git LFS); x86-64 installs the TFLite C lib system-wide (`-DTFLITE_LIB=...` to override).

## Key config keys

Parsed in `demo/odaslive/parameters.c`; defaults in `mod_sst.c:1546`.

| Key | Meaning |
|---|---|
| `sst.enable_classifier_output` | `"enabled"`/`"disabled"` — master switch for JSON/`.bin` |
| `sst.classifier_log_dir` | output dir (relative → absolute via `getcwd()`, auto-created) |
| `sst.sim_mode` | **int 0/1** (not bool): 0=Pi (no `.bin`), 1=simulator (write `.bin`) |
| `sst.min_event_votes` | int, default 4, range 1–6 (intended 4-of-6 gate; **not hard-enforced** in current emit path) |
| `raw.model_path` | **directory** holding `yamnet_core.tflite` + `yamnet_class_map.csv` |
| `raw.interface` | `{type="socket"; ip="127.0.0.1"; port=10000;}` selects the audio source |
| `general.mics` / `mapping.map` | `(2,3,4,5)` — selects ReSpeaker mic channels 2–5 of 6 |

Representative `config/runtime/local_socket.cfg.template`:
```cfg
raw: {
    fS = 16000; hopSize = 128; nBits = 16; nChannels = 6;
    interface: { type = "socket"; ip = "127.0.0.1"; port = 10000; }
    model_path = "${ODAS_DIR}/models";
}
mapping: { map: (2, 3, 4, 5); }
sst: {
    enable_classifier_output = "enabled";
    classifier_log_dir       = "./ClassifierLogs";
    sim_mode                 = 1;   # 0=Pi, 1=simulator
    min_event_votes          = 1;   # 1=collect-all, 4+ after fine-tune
}
```
`scripts/setup_runtime.sh` substitutes `${ODAS_DIR}`/`${CHATAK_GUI_DIR}` into `~/sodas/*.cfg`
and creates `~/sodas/ClassifierLogs/`.

## Simulation: replaying a render

`sim_mode` does **not** itself switch the audio source — `raw.interface.type` does. The runtime
template ships `type="socket"` + `sim_mode=1` together, so they travel as a pair. To replay a
`.raw`:

```bash
# T1 — start the engine (connects to the socket)
build/bin/odaslive -c ~/sodas/local_socket.cfg
# T2 — stream the render to it
python3 scripts/vm_socket_emit.py --audio /path/render.raw --port 10000
```

`vm_socket_emit.py` is a TCP **server** (binds `0.0.0.0:port`, `listen(1)`), waits for odaslive
to connect, then sends fixed 1536-byte chunks (`128 hop × 6 ch × 2 bytes`) paced ~10 ms/frame
(or by a `--timestamps` file) until EOF.

## Simulator orchestration (`simulator.py`)

`SimulationRunner._run_simulation` (`simulator.py:361`) wires the above for the Streamlit
"ODAS Simulator" page:

1. **Apply an SST preset** by regex-patching 7 cfg params (`Pnew, theta_new, N_prob,
   theta_prob, Pfalse, gainMin, theta_inactive`) — presets *Balanced / High-Recall / Low-FP*
   (`_apply_sst_preset`, `:335`; `SST_PRESETS`, `:35`).
2. **Free the port** (`fuser -k`), default 10000 (range 10000–20000).
3. **Start the socket server** `vm_socket_emit.py` (`:398`).
4. **Start odaslive** `-v -c <cfg>`, logging to `outputs/runs/odas_log_<ts>.txt` (`:420`).
5. **Daemon monitor thread** `_monitor_background` (`:470`) survives Streamlit reruns; ODAS
   runs ~0.79× real-time, so it waits up to 90 s to drain after the stream ends.
6. **Write the run manifest** `outputs/runs/<run>.json` (`:547`): `run_id, render_id,
   scene_name, raw_audio_file, scene_metadata, scene_file, odas_log_file,
   classify_events_file, session_live_file, port, odas_config, warmup_seconds, odas_preset,
   experiment_tag` — this is what the analyzer consumes.

> **Pure-Python alternative:** `odas_simulator.py` (`ODASSimulator`) + `odas_optimized.py` run
> SSL/SST entirely in Python (no socket/odaslive, ~28× real-time) and emit an
> analyzer-compatible `session_live` file. Useful for fast DOA experiments; `odas.py` is the
> readable reference implementation (`ssl_process:364`, `sst_process:566`).

## Next

[`05_analysis_and_gt_matching.md`](05_analysis_and_gt_matching.md) — match these detections back
to the scene's ground-truth sources.
