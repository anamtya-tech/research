# Data Prep — Annotated Clip Inventory (plan.md, Stage 1)

Deliverable for the first block of `plan.md` ("Data Prep"). Builds a standardized,
annotated inventory of positive (target) and background (negative) clips, with
SNR/RMS/quality analysis and listen-candidate spectrograms.

## What was built

| Artifact | Path |
|---|---|
| Positive clips (50/class, 16 kHz mono, 1.0 s) | `clips/positive/<Class>/<class>_NNN.wav` |
| Background clips (200, 16 kHz mono, 1.0 s) | `clips/background/bg_NNN.wav` |
| **Annotated inventory** | `inventory.csv` |
| Per-class stats + listen candidates | `summary.json` |
| Distribution / separability / spectrogram figures | `figures/` |
| Build / analysis scripts | `scripts/build_dataset.py`, `scripts/visualize.py` |

**Target classes:** Elephant, Drone, Lion, Monkey, Frog. **Total: 450 clips** (250 positive + 200 background).

## Source data

- **Elephant** — `audio_cache/elephant_samples_new` (232 clean 6 s / 48 k mono files; `*_aug` excluded).
- **Drone** — `audio_cache/yes_drone_binary` (1332 files, already 16 k mono ~1 s).
- **Lion** — `audio_cache/wild_animals/.../Aslan` (43 unique after dedup; 48 k stereo).
- **Monkey** — `audio_cache/wild_animals/.../Monkey` (36 unique; 48 k stereo).
- **Frog** — `audio_cache/wild_animals/.../Frog` (44 unique; 11025 Hz stereo).
- **Background** — `audio_cache/ambient_captures/*.raw` (6-ch S16 16 k mic-array; **mic channel 2** extracted; ch 0 & 5 are empty). Negatives spread across 4 captures (Eco10, two Mar-30, Eco_Park).

> `jungle_animals/{Lion,Monkey,Frog}` are **byte-identical duplicates** of the `wild_animals`
> copies (verified by MD5) — used only one tree to avoid train/test leakage.

## Pipeline decisions

- **Standardization:** all clips → 16 kHz mono, 1.0 s (matches the curator config:
  `sample_rate=16000, target_duration=1.0`). Resampled with librosa, stereo mixed to mono.
- **Smart splitting:** for each source file the most-energetic 1.0 s window is chosen
  (sliding sum of `y²`). Classes with few unique sources (Lion/Monkey/Frog) reach 50 clips
  by taking additional non-overlapping windows from longer files, round-robin across files.
  Files shorter than 1.0 s are centre zero-padded (29 clips, flagged `padded`).
- **Leakage control:** an 80/20 **source-level** split — every window from one source file
  stays in the same split.
- **Labels:** single-source library clips → label = class, prob 1.0. (The plan's
  distance-weighted probability labelling applies to *simulator-mixed* scenes, which are a
  later track — the simulator source code is not present in this repo, only its outputs.)

## Inventory columns (`inventory.csv`)

| Column | Meaning |
|---|---|
| `clip_id`, `label`, `split` | id, class, train/test |
| `source_file`, `source_start_sec` | provenance (path relative to `backup/`, window offset) |
| `rms_db`, `peak_db` | clip level / peak (dBFS) |
| `dr_db` | within-clip dynamic range (P90/P10 of frame RMS, capped 60). **High = impulsive** (frog/monkey), **low = steady** (drone/elephant rumble). |
| `snr_vs_ambient_db` | clip RMS minus the dataset's **median background RMS**. A robust "how far above the ambient floor" measure that works for steady sounds (unlike a within-clip SNR). |
| `active_frac` | fraction of frames > floor +10 dB (how much of the clip is event vs gap) |
| `quality` | 1–3, from `snr_vs_ambient_db` (≥25→3, ≥12→2, else 1); demoted if near-silent or truly clipped |
| `notes` | `clipping` (>1% railed samples), `near-silent`, `padded` |

## Data distribution (the Pass criterion: describe it confidently)

```
class       n    train/test   ΔdB above ambient (min/mean/max)   dr_db   active   Q1/Q2/Q3
Elephant    50    38/12         7.4 / 28.2 / 40.0                 8.3     0.11     2/13/35
Drone       50    38/12        19.9 / 23.2 / 30.6                 4.9     0.02     0/40/10
Lion        50    40/10        19.5 / 29.8 / 37.5                11.7     0.22     0/15/35
Monkey      50    40/10        14.6 / 27.4 / 40.4                20.5     0.50     0/23/27
Frog        50    40/10         9.6 / 25.2 / 33.4                25.4     0.45     1/20/29
background  200  160/40        -7.7 /  0.5 / 31.1                 6.2     0.01     197/0/3
```

**Findings**
- **Clean level separation:** every target class sits ~20–40 dB above the ambient floor;
  background centres at 0 dB. See `figures/dist_snr_vs_ambient.png`.
- **`dr_db` separates source types:** steady (Drone 4.9, Elephant 8.3) vs impulsive
  (Frog 25.4, Monkey 20.5). Even these two hand-crafted features visibly separate background
  from targets (`figures/scatter_dr_vs_snr.png`) — encouraging for the YAMNet embedding work.
- **Spectrograms confirm targets are present** (`figures/spectrograms_<class>.png`): elephant
  low-freq rumble + harmonic trumpet (`elephant_017`); drone steady tonal bands + broadband
  motor; frog/monkey impulsive calls.

## Caveats / things to know

- **Drone `active_frac` is low (0.02)** despite loud clips — the metric counts frames *above
  the clip's own floor*, and a steady tone has little internal variation. For steady sources
  read `snr_vs_ambient_db`, not `active_frac`.
- **Background outliers:** `bg_000/050/100` are the window at the t=12 s warmup boundary and
  catch a start-up transient (flagged `clipping`). A few other bg clips (Δ10–11 dB) are real
  ambient events — useful as *hard negatives*, but worth a listen before trusting as "silence."
- The 5 source-libraries are **clean, isolated recordings**, not field captures — so positives
  are easier than real deployment audio. Real-world difficulty will come from the simulator /
  ambient-mixed scenes (next track).

## How to regenerate

```bash
.venv/bin/python data_prep/scripts/build_dataset.py   # clips + inventory.csv + summary.json
.venv/bin/python data_prep/scripts/visualize.py       # figures/
```

## Next per plan.md

YAMNet baseline scores → extract 1024-d embeddings → UMAP → logistic-regression linear probe.
(The prior analyzer already emitted YAMNet 1024-d embedding CSVs, so there is precedent here.)
Requires adding `tensorflow` + `tensorflow-hub` (YAMNet) to the venv.
