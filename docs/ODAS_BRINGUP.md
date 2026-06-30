# ODAS Bring-up — odaslive in an arm64 Linux container (unblocks Track B)

**Goal:** get the `chatak-odas` firmware running so the **post-ODAS** half of the
`experiments.pdf` program (post-beamformed `.bin` spectra, event detections, **FP/min**) is
measurable on this Mac. **Status:** ✅ built and smoke-tested.

## Why a container (and why arm64)

`chatak-odas` targets Linux (CMake hard-requires ALSA + PulseAudio + libconfig + fftw + json-c),
and the bundled TFLite C lib is **aarch64**. This is an **Apple-Silicon (arm64) Mac**, so a native
**arm64 Linux container reports `uname -m = aarch64`** → CMake uses the bundled
`third_party/tflite/aarch64/libtensorflowlite_c.so` directly (no TFLite build) and the env matches
the Raspberry-Pi deployment target. No native-macOS build, no ALSA hacks, no emulation.

## Prerequisites resolved

| Missing piece | Fix |
|---|---|
| cJSON submodule (empty) | `git submodule update --init --recursive` |
| `libtensorflowlite_c.so` was a 132-byte **Git LFS pointer** | `brew install git-lfs && git lfs pull` → real 3.6 MB aarch64 ELF |
| `yamnet_core.tflite` model | already a real 14 MB binary after `git lfs pull` |

## Build

Deps image — `experiments/odas/Dockerfile` (arm64 debian + cmake/fftw/libconfig/alsa/pulse/json-c):

```bash
cd experiments/odas
docker build --platform linux/arm64 -t chatak-odas-build .
```

Compile (repo bind-mounted so artifacts land in `chatak-odas/build/`):

```bash
docker run --rm --platform linux/arm64 \
  -v /Users/abhinav/research/chatak-odas:/work -w /work chatak-odas-build \
  bash -c "cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)"
# → build/bin/odaslive  (arm64 ELF), build/lib/libodas.so
```

## Runtime config

`experiments/odas/local_socket_container.cfg` — generated from
`chatak-odas/config/runtime/local_socket.cfg.template` with container paths substituted:

- `raw.model_path = /work/models` (holds `yamnet_core.tflite` + `yamnet_class_map.csv`)
- `raw.interface = { type=socket; ip=127.0.0.1; port=10000 }`
- `bandpass = /exp/odas/bandpass.cfg` (copied from `backup/ChatakGUI/config/bandpass.cfg`)
- `sst.classifier_log_dir = /exp/odas/logs`, `sst.sim_mode = 1`,
  `enable_classifier_output = enabled`, `min_event_votes = 1`
- mic geometry: ReSpeaker 4-mic, `mapping.map = (2,3,4,5)`

## Run (sim mode = replay a render over the socket)

```bash
docker run --rm --platform linux/arm64 \
  -v /Users/abhinav/research/chatak-odas:/work \
  -v /Users/abhinav/research/experiments:/exp -w /work chatak-odas-build bash -c '
    export LD_LIBRARY_PATH=/work/third_party/tflite/aarch64:$LD_LIBRARY_PATH
    python3 /work/scripts/vm_socket_emit.py --audio /exp/sim/renders/renders/<render>.raw --port 10000 &
    sleep 1
    ./build/bin/odaslive -v -c /exp/odas/local_socket_container.cfg
  '
```

Outputs land in `experiments/odas/logs/`:
- `sst_session_live.json_<ts>.json` — JSON-lines, one object per 48 ms gate
- `patch_<track>_<ts>.bin` — 96×257 float32 (98 688 B) raw spectra patches
- `sst_session_live_fingerprint_<ts>.json`

## Smoke-test result (30 s no-ambient render)

| Signal | Value |
|---|---|
| Gated frames (48 ms) | 1041 |
| Active detections / distinct tracks | 1328 / 67 |
| Detections carrying a `.bin` | 926 |
| `.bin` shape check | 96×257 float32 ✓ (98 688 B) |
| YAMNet event classes emitted | Elephant 553, Frog 373 |

The full chain Mic-stream → SSL → SST → YAMNet → JSON + `.bin` runs and produces exactly the
artifacts the analyzer/curator consume.

## Known issue

`odaslive` **segfaults at teardown** (rc=139) *after* all outputs are flushed — a cleanup-time
crash, not a data problem (session JSON + all `.bin` files are complete and valid). Harmless for
offline dataset generation; worth fixing before it matters for long-running live capture. Run under
`timeout` in batch jobs so the exit code doesn't abort a pipeline.

## What this unblocks

Track B for the whole `experiments.pdf` program:
1. `analyzer.py` on a run manifest → match detections to GT ([`05`](05_analysis_and_gt_matching.md))
2. `yamnet_dataset_curator.py` → post-ODAS dataset from `.bin` ([`06`](06_yamnet_dataset_curation.md))
3. train `model_*_odas` and compute **FP/min** — the real Phase-1/2 metrics.

Next: drive a full EXP-A1 600 s render through this to produce `model_a1_odas` and the GT-vs-ODAS
comparison ([`P1_exp_a1.md`](P1_exp_a1.md)).
