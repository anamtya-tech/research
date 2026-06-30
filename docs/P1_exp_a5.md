# Phase 1 · EXP-A5 — Real-Capture Ambient (the deployment-realistic testbed) + B4 on it

**experiments.pdf tag:** `exp_a5_capture_ambient`. This fixes the methodological gap found in
[`P2_exp_b4.md`](P2_exp_b4.md): every prior FP measurement was on *no-ambient* renders whose
"quiet" was literal digital silence (−97 dBFS), so the false positives were structural artifacts,
not deployment-representative. Here scenes carry **real-capture ambient**, so FPs are real.

## Setup

- Rendered A5 train (600 s) + holdout (300 s) = directional sources **+ real Mar-30 capture
  ambient** (`ambient_mode='capture'`, volume 1.0 → ambient ≈ −36 dBFS, ~5 dB under sources),
  different capture segments for train vs holdout.
- Verified the mix: A5 holdout mid-content RMS −36.6 dBFS vs A1 (no-ambient) **−96.9 dBFS** —
  A1's "quiet" really was silence.
- Ran both through ODAS (same default SST as A1).

## Result 1 — real ambient makes everything much harder

| testbed | FP/min | event recall (detected) |
|---|---|---|
| A1 (no ambient) | 18.7 | 0.89–0.97 |
| **A5 train (real ambient)** | **36.8** | 0.64 |
| **A5 holdout (real ambient)** | **51.1** | 0.62 |

Real ambient **2–3× the false positives** *and* **drops recall ~30 pts**. The no-ambient A1
numbers were optimistic artifacts; **51 FP/min is the honest deployment-realistic baseline.**

## Result 2 — on the realistic testbed, hard negatives WORK (the opposite of the silent testbed)

Both heads evaluated on the A5 real-ambient holdout. (`recall` here = GT event detected **and**
classified as the correct animal; baseline 0.33 vs detected 0.62 means the 6-class model
mis-classifies ~half of detected events in heavy ambient.)

| model | FP/min | recall | holdout ghosts → background |
|---|---|---|---|
| 6-class baseline | 48.5 | 0.333 | 0 / 132 |
| **+ same-env negs (Mar-30, 245)** | **9.2** | 0.283 | **107 / 132 (81%)** |
| + cross-env negs (Eco_Park, 773) | 15.1 | 0.250 | 91 / 132 (69%) |

![EXP-A5+B4 real-ambient holdout](figures/expA5_b4.png)

- **Hard negatives cut FP/min 48.5 → 9.2 (−81%)** by sending 107/132 real-ambient ghosts to
  `background`. On the *silent* A1 holdout the same idea barely moved FP (8.1 → 7.4) — because
  those ghosts were structural artifacts, not ambient. **This is the direct confirmation that the
  testbed, not the method, was the problem.**
- **Same-environment negatives transfer best** (9.2 vs 15.1), but **cross-environment still helps
  a lot** (48.5 → 15.1) — real-ambient negatives partially generalize across recording sites.
- Small recall cost (0.33 → 0.28): the `background` class steals a few real events; a
  background-confidence threshold should recover most of it.

## Result 3 — background-confidence threshold can't lift recall (the wall)

Using the 7-class (+same-env) head, we suppress a detection only when `P(background) ≥ τ` and sweep
τ on the real-ambient holdout:

| τ | FP/min | recall (detect + correct class) |
|---|---|---|
| 0.4 | 8.5 | 0.267 |
| 0.9 | 15.8 | 0.283 |
| never suppress | 48.5 | 0.300 |

![Background-threshold trade-off — recall plateaus ~0.30](figures/expA5_bgthresh.png)

**Recall is capped at ~0.30 regardless of τ** — no threshold reaches FP ≤ 2 with usable recall. In
~5 dB-SNR ambient the post-ODAS spectra are intrinsically hard to classify; a threshold only trades
FP against recall along a flat ceiling. **Classification accuracy in ambient is the binding
constraint now**, and it's a data/SNR problem, not a threshold one.

## What this tells us about the goal

1. **The real deployment problem is now correctly measured: ~51 FP/min and ~62% detection / ~33%
   correct-class on real ambient.** Far harder than the silent renders implied.
2. **Hard negatives are highly effective against the real FP problem** (−81%), and *direction-
   agnostic*, validating the entire B4 direction — once measured on the right distribution.
3. **The new binding constraint is recall / classification accuracy in ambient** (0.33 correct-
   class). Heavy ambient corrupts the post-ODAS spectra; this needs **ambient-mixed training
   positives** (train on A5-style post-ODAS clips, not clean ones) + more data + a background-
   confidence threshold so FP suppression doesn't eat real events.

## Recommended next

1. **Train positives from A5 (real-ambient) post-ODAS clips**, not the no-ambient A1 set — so the
   model sees deployment-distribution animal spectra (should lift the 0.33).
2. **Background-confidence threshold** — only route to `background` when confident → recover recall.
3. **Per-environment / pooled negatives** (EXP-B5) — combine ambient sites for robustness.
4. Re-run with these to push toward usable (FP ≤ a few /min at recall ≥ ~0.7).

## Artifacts / reproduce

`experiments/outputs/{a5,a5_hold}_deploy_metrics.json`, `expA5_b4_results.json`,
`figures/expA5_b4.png`. ODAS logs: `experiments/odas/logs_a5train/`, `logs_a5hold/`.
```bash
.venv/bin/python experiments/scripts/expA1_render.py --duration 600 --instances 15 --seed 1 \
   --name exp_a5_real --capture <cap.raw> --cap-volume 1.0 --cap-offset 0
# (+ holdout at a different offset/seed) → ODAS → analyze → :
.venv/bin/python experiments/scripts/expA5_b4.py
```
