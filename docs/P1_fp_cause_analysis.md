# Phase 1 · EXP-A1 — What causes the false positives? (is it Kalman?)

**Question:** the no-ambient baseline emits ~20 FP/min. Is the Kalman tracker (SST) creating them,
or something else? **Answer: primarily SSL / array-geometry structural hotspots — *not* Kalman.**
Kalman is a secondary contributor; the open vote-gate and a classifier with no learned rejection
do the rest.

## Evidence

### 1. FPs cluster at fixed array-axis directions (decisive)

Azimuth of every quiet-period FP vs where GT sources were actually placed (600 s + holdout pooled):

![FP azimuths cluster at array hotspots](figures/fp_azimuth_cause.png)

- GT sources are spread fairly uniformly across all azimuths; FPs are **not** — they lock to the
  mic-baseline axes. A random tracker would not produce this geometry-locked pattern → the cause
  is upstream of Kalman.

Per-axis breakdown of the 209 quiet FPs (within ±20° of each mic-baseline direction):

| direction | mic baseline | share of FPs | note |
|---|---|---|---|
| **0° (+X, Right)** | Left–Right | **47%** | **but 33% of *all* FPs are at *exactly* 0.000°** → see below |
| **±90° (+Y/−Y, Front/Back)** | Front–Back | **30%** | genuine structural hotspot (the "Y axis / 90–270°" cluster) |
| 180° (−X, Left) | Left–Right | 14% | genuine hotspot |
| ±120° | — | 6% | minor |

**Two distinct mechanisms hide in that 0° spike:**

1. **Degenerate / null DOA (~33% of all FPs).** 70/209 detections report position **exactly
   `(0,0,0)` with `activity = 0`** — a placeholder emitted when SSL has no real peak in a quiet
   window. `atan2(0,0)=0` bins them at 0°. **This is a software artifact, not array physics, and
   is trivially filterable** (drop `activity ≈ 0` / zero-vector detections — the curator's
   `min_activity = 0.01` already would). So **our raw 18.8 FP/min over-counts by ~a third**;
   downstream curation discards these.
2. **Genuine GCC-PHAT structural hotspots (the rest).** Concentrated on the mic baselines —
   **±90° (Y-axis, Front/Back), 180° (−X), ±120°**, plus a smaller real +X lobe. These match the
   hotspots the PDF named and come from spatial aliasing / steering-vector ambiguity of the 64 mm
   square array (spacing < λ/2 only below ~2.7 kHz). **This is the real residual FP source**, and
   it is created in SSL, not Kalman.

### 2. Tightening Kalman barely helps (corroborating)

The [SST preset sweep](P1_sst_sweep.md): driving the Kalman new-track params hard
(`Pnew 0.6→0.03`, `N_prob 3→8`, Low-FP preset) only moved FP/min 18.7 → 14.0. If Kalman were the
source, aggressive tightening would have collapsed it. It didn't — the spurious SSL peaks persist
and the tracker keeps re-confirming them.

### 3. The ghosts are single-hop flickers that YAMNet labels confidently

- **96% of ghosts carry just 1 vote** (holdout: 49/51 are 1-vote) — momentary, single-hop
  detections, not sustained tracks.
- Yet **YAMNet is confident on them**: median `event_avg_confidence ≈ 0.9`, up to 1.0. The model
  has no "this is just noise/structure" response — it maps hotspot-noise spectra to a real class
  with high confidence. **So plain confidence thresholding will not filter them.**
- We ran `min_event_votes = 1` (collect-all), so every single-hop ghost passed the gate.

### 4. Vote-gate sweep (offline, no ODAS re-run) — the trade-off this exposes

Applying a vote threshold post-hoc to the holdout (FPs and recall together):

| `min_event_votes ≥` | FP/min | event recall |
|---|---|---|
| 1 (what we ran) | 18.8 | 0.967 |
| **2** | **0.7** | 0.650 |
| 3 | 0.4 | 0.500 |
| 4 | 0.4 | 0.383 |

Requiring just **2 votes removes 96% of FPs** (18.8 → 0.7, under the ≤2 target) — confirming the
ghosts are 1-hop flickers. But recall drops to 0.65, because **many real events are also brief**
(short/quiet animals — Bear, Frog — sustain only 1–2 classified hops). So a blunt vote gate trades
FPs for missed real events — the same short-event problem the PDF flags.

## Conclusion — the causal chain

```
(A) SSL has no real peak in quiet  →  null DOA (0,0,0), activity 0  →  ~33% of FPs   [ARTIFACT, free to fix]
(B) square 4-mic array + GCC-PHAT  →  hotspots on baselines: ±90 (Y), 180 (-X), ±120 [ROOT, real residual]
        ↓                              (spatial aliasing; reverb tail minor ~10–20%)
SST/Kalman                         →  promotes/sustains some; tightening helps only ~25% [secondary]
        ↓
YAMNet (no learned rejection)      →  confidently labels noise/hotspot as a target class  [amplifier]
        ↓
min_event_votes = 1 (collect-all)  →  every 1-hop ghost passes the gate                    [open gate]
        ↓
                                      ~20 classified FP/min  (≈⅓ removable for free via (A))
```

**It is not Kalman.** Kalman tightening (#2) and the geometry-locked azimuth pattern (#1) rule it
out. The FPs split into a removable **null-DOA artifact (A)** and a genuine **SSL-hotspot residual
(B)**; the classifier and open vote gate let both through.

## Fixes, ranked

0. **Drop null/zero-activity detections (free, lossless).** Filter `activity ≈ 0` / `(0,0,0)`
   position. Measured effect: **FP/min 18.7 → 8.1 (holdout), 22.0 → 11.1 (600 s), recall
   unchanged** (0.967 / 0.889). The curator's `min_activity=0.01` already does this offline, so the
   honest **deployed baseline is ~8–11 FP/min**, not ~20 — but still ~4–5× over the ≤2 target.
1. **Hard negatives (EXP-B4) — surgical, for residual (B), and direction-agnostic.** Train on
   hotspot/ambient ghost spectra labeled `background`. Discriminates by **spectral content, not
   direction**: a real source at a hotspot bearing keeps its animal spectra → classified correctly;
   a structural ghost there is aliased noise → `background`. Because the genuine ghosts are
   *high-confidence* (#3), this is the only fix that removes them **without** harming real events.
   Top priority.
2. **`min_event_votes = 2`** — cheap config change, FP/min → 0.7, but recall → 0.65. Stopgap /
   combine, knowing the short-event recall cost.
3. **Azimuth hotspot masking — DON'T (here).** Blanket-ignoring ±90/180/±120 (±20°) would also
   drop real sources at those bearings: **54–58% of GT source instances and ~50–55% of real
   matched detections lie in those bands** (sources are placed uniformly; the hotspot bands cover
   ~55% of the circle). So masking ≈ losing half of real detection — the reason we discriminate by
   spectrum (#1), not direction. Only viable as a narrow, frequency-aware SSL de-weighting, not a
   hard azimuth gate.
4. **Per-class confidence thresholds (EXP-C2)** — limited, since ghost confidence is high.

The combination that should work: **Low-FP SST preset + hard negatives + a modest vote gate**, so
each lever removes a different slice without over-cutting recall.

## Config-lever sweep results (empirical — holdout, null-activity filtered)

Tested the two config-only ODAS levers (probMin needs ODAS re-runs; votes applied offline),
SST held constant:

| probMin | FP/min (votes≥1) | recall | FP/min (votes≥2) | recall |
|---|---|---|---|---|
| **0.5** (default) | 8.1 | 0.967 | **0.7** | 0.650 |
| 0.7 | 9.2 | 0.933 | 1.1 | 0.600 |
| 0.8 | 9.2 | 0.933 | 1.1 | 0.600 |

- **`ssl.probMin` is ineffective (even counterproductive).** Raising it 0.5→0.7/0.8 slightly
  *worsened* FP/min and *cut* recall (0.7≡0.8 → no further effect). Reason: the structural
  hotspots are **strong** GCC-PHAT peaks (aliasing = high correlation), so a confidence floor
  can't separate them from real sources — it only discards weak/distant *real* ones.
- **`min_event_votes ≥ 2` is the only effective config lever** — FP/min 8.1 → **0.7** (meets the
  ≤2 target) but recall 0.97 → **0.65** (kills 1-hop flickers *and* short/quiet real animals).
  Stacking probMin on top is strictly worse (1.1 / 0.60).

**Takeaway:** config-only tuning *can* reach the FP target, but only by sacrificing ~⅓ of real
events. This is the empirical proof that **hard negatives (EXP-B4) are necessary, not optional** —
they reject ghosts by *spectrum* and recover the recall a blunt vote gate destroys. ODAS config
(probMin, SST preset, votes) is a complement, not a substitute.

## Artifacts / reproduce

`figures/fp_azimuth_cause.png`; azimuth + vote analysis from `experiments/outputs/a1_analysis.pkl`
and `a1_hold_analysis.pkl`. FP definition: see
[`P1_exp_a1_trackB.md` § What counts as a False Positive](P1_exp_a1_trackB.md#what-counts-as-a-false-positive-precise-definition).
