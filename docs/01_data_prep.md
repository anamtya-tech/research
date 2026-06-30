# 01 — Data Prep: Annotated Clip Inventory

**Goal (plan.md, Stage 1):** understand exactly what data we have before touching a model.
**Deliverable:** an annotated clip inventory you can describe confidently.
**Status:** ✅ done — see [`../data_prep/`](../data_prep/) and its [`README.md`](../data_prep/README.md).

## What was produced

| Artifact | Path |
|---|---|
| Positive clips (50/class, 16 kHz mono, 1.0 s) | `data_prep/clips/positive/<Class>/<class>_NNN.wav` |
| Background clips (200) | `data_prep/clips/background/bg_NNN.wav` |
| **Annotated inventory** | `data_prep/inventory.csv` |
| Per-class stats + listen candidates | `data_prep/summary.json` |
| Figures (distribution, separability, spectrograms) | `data_prep/figures/` |
| Build + viz scripts | `data_prep/scripts/build_dataset.py`, `visualize.py` |

**450 clips total** — 50 each for **Elephant, Drone, Lion, Monkey, Frog** + 200 background,
with an 80/20 **source-level** train/test split (every window from one source file stays in
one split, so no leakage).

## Sources

- Elephant — `backup/audio_cache/elephant_samples_new` (232 clean files; `*_aug` excluded)
- Drone — `backup/audio_cache/yes_drone_binary` (1332, already 16 k mono ~1 s)
- Lion / Monkey / Frog — `backup/audio_cache/wild_animals/.../{Aslan,Monkey,Frog}`
  (43 / 36 / 44 unique after MD5 dedup; the `jungle_animals` copies are byte-identical)
- Background — `backup/audio_cache/ambient_captures/*.raw` (6-ch S16 16 k mic-array;
  **mic channel 2** extracted; spread across 4 captures)

## Key decisions

- **Standardize** everything to 16 kHz mono, 1.0 s (matches the curator config
  `sample_rate=16000, target_duration=1.0`).
- **Smart split:** pick the most-energetic 1.0 s window per source (sliding sum of `y²`);
  reach 50/class on thin classes by taking extra non-overlapping windows from longer files.
- **`.raw` are actually RIFF/WAV** (6-ch, 16 k) but libsndfile won't auto-detect them via the
  `.raw` extension → read mic channel directly via memmap past the 44-byte header.
- **SNR, reframed honestly:** a within-clip P90/P10 ratio mislabels steady sounds (a loud
  drone looked "low SNR"). The inventory therefore reports:
  - `dr_db` — within-clip dynamic range (high = impulsive: frog/monkey; low = steady: drone/elephant rumble)
  - `snr_vs_ambient_db` — clip RMS minus the dataset's median background RMS (robust for steady sounds); **quality(1–3) is based on this.**

## inventory.csv columns

`clip_id, label, split, source_file, source_start_sec, duration_sec, sr, channel,
rms_db, peak_db, dr_db, snr_vs_ambient_db, active_frac, quality, notes`
(`notes` ∈ {`clipping` (>1 % railed samples), `near-silent`, `padded`})

## Distribution (the pass criterion)

```
class       n    train/test   ΔdB above ambient (min/mean/max)   dr_db   active   Q1/Q2/Q3
Elephant    50    38/12         7.4 / 28.2 / 40.0                 8.3     0.11     2/13/35
Drone       50    38/12        19.9 / 23.2 / 30.6                 4.9     0.02     0/40/10
Lion        50    40/10        19.5 / 29.8 / 37.5                11.7     0.22     0/15/35
Monkey      50    40/10        14.6 / 27.4 / 40.4                20.5     0.50     0/23/27
Frog        50    40/10         9.6 / 25.2 / 33.4                25.4     0.45     1/20/29
background  200  160/40        -7.7 /  0.5 / 31.1                 6.2     0.01     197/0/3
```

All targets sit ~20–40 dB above the ambient floor; background centres at 0 dB. `dr_db`
separates steady (Drone, Elephant) from impulsive (Frog, Monkey) sources, and even these two
hand-crafted features visibly separate targets from background. Spectrograms confirm the
target is genuinely present in each clip.

## Caveats

- These are **clean isolated library recordings**, not field captures — positives here are
  easier than real deployment audio. Real difficulty comes from the simulator's ambient-mixed
  scenes (steps 02–05).
- A few background clips are real ambient events (Δ10–31 dB) — good **hard negatives**; the
  t=12 s clips catch a recording-start transient.

## Reproduce

```bash
.venv/bin/python data_prep/scripts/build_dataset.py   # clips + inventory.csv + summary.json
.venv/bin/python data_prep/scripts/visualize.py       # figures/
```

## Next

Step [`07_yamnet_training.md`](07_yamnet_training.md) and the rest of `plan.md` (YAMNet baseline
→ embeddings → UMAP → linear probe → MLP head) consume this inventory. Steps 02–06 document how
the **simulator** produces the harder, spatially-mixed training data that supersedes these clean
clips for production.
