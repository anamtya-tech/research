# Phase 2 · Persistent Corpus + Multi-SNR Training

**Goal:** stop training throwaway heads per experiment — accumulate all post-ODAS data into one
growing, labeled corpus so we can train **collectively**, and use it to attack the in-noise
accuracy wall with **multi-SNR** positives.

## The corpus (`experiments/corpus/`)

Every ODAS run contributes post-ODAS `.bin` spectra, embedded once and appended with provenance
(`corpus.py`):
- **ambient / silence** runs → every peak = `background`
- **clean / mixed** runs → GT-matched peaks = animal label; quiet peaks = `background`

Store: `embeddings.npy` (N×1024) + `meta.csv`
(`sample_id,label,source,env,snr_db,run_tag,bin_path,fold`). De-duped by `bin_path` (idempotent).
Train collectively any time: `python corpus.py --train`.

| run | source | samples |
|---|---|---|
| a1 / a1_hold | clean | 496 / 354 |
| a5 / a5_hold | mixed (Mar-30, ~5 dB) | 490 / 281 |
| a5_snrhi / a5_snrlo | mixed (~15 / ~−5 dB) | 522 / 493 |
| amb_ecopark | ambient (Eco_Park) | 773 |
| sil_a1struct | silence (structural ghosts) | 127 |
| **total** | | **3,536** (2,901 train / 635 test) |

This realizes the intended three-stream recipe: **raw ambient → background**, **clean → GT**,
**animal+ambient → GT + background**, all pooled.

## Multi-SNR result

Recall degrades monotonically with ambient level (one render per SNR, ODAS, GT-matched):

| SNR | event recall | FP/min |
|---|---|---|
| ~15 dB | 0.689 | 35 |
| ~5 dB | 0.62 | 51 |
| ~−5 dB | 0.433 | 46 |

Adding the easier + harder SNR positives to the corpus and retraining the pooled head:

| test holdout | single-SNR corpus | **multi-SNR corpus** |
|---|---|---|
| mixed (deployment-realistic) | 0.566 | **0.619** (+5.3) |
| clean | 0.376 | **0.452** (+7.6) |
| overall | 0.460 | **0.526** (+6.6) |

**Spanning the SNR range lifts deployment accuracy ~5 pts and clean ~8 pts** — direct evidence that
"more real-noise training across SNRs" is the right lever for the in-noise accuracy wall.

## Takeaways

- **Infrastructure in place:** one growing labeled corpus + one-command collective retraining.
  Every future run just appends.
- **The lever works but isn't saturated:** 0.62 on the deployment holdout is up but not yet
  deployment-grade. The path forward is *more of the same*: more positives per class, more SNR
  levels, more ambient environments — and likely accepting a defined detection-range/SNR envelope
  rather than chasing 100% at −5 dB.

## Corpus growth round 2 — and what it revealed

We then nearly doubled the corpus: **5,924 samples** across **4 ambient environments** (Eco_Park,
First, Second, Eco10 + structural silence), **3 SNRs**, **2 extra mixed sites** (g1/g2), and richer
per-class source pools (elephant_samples_new + wild_animals merged in).

**Result: accuracy plateaued** (overall 0.51, mixed 0.59, clean 0.44 — flat vs the smaller
multi-SNR corpus, within training noise). Doubling the data did **not** help. Per-class diagnosis
on the test set shows why — it's **class imbalance**, not volume:

| class | train count | test acc | → predicted background |
|---|---|---|---|
| Lion | 631 | **0.75** | 0.01 |
| Frog | 184 | 0.50 | 0.18 |
| Elephant | 473 | 0.36 | 0.27 |
| drone_bebop | 133 | 0.34 | 0.17 |
| Bear | 231 | 0.17 | 0.14 |
| **drone_binary** | **90** | **0.00** | 0.59 |
| background | **3547** | 0.69 | — |

Accuracy tracks per-class training count; **background (3547) swamps the rare classes**, which get
predicted `background`. The growth added mostly *background* (already plentiful), so it plateaued.

**Rebalancing test (cap background, no new data):**

| bg cap | overall | mixed | drone_binary acc | background acc |
|---|---|---|---|---|
| 3547 | 0.520 | 0.577 | 0.00 | 0.70 |
| 600 | 0.496 | 0.520 | 0.12 | 0.49 |
| 300 | 0.472 | 0.463 | 0.22 | 0.42 |

Capping background **recovers rare classes but raises false positives** — it slides the FP↔recall
operating point, it does **not** lift the frontier.

### Conclusions from growing the corpus

1. **Infrastructure scales** — 5,924 samples, 4 environments, 3 SNRs, one-command collective train.
2. **Volume isn't the lever; balance + difficulty-diversity are.** Multi-SNR helped (+5 pts); raw
   volume (more background) plateaued.
3. **The binding constraints are now explicit:** (a) too few positives for rare classes
   (drone_binary, drone_bebop, Bear), and (b) the fundamental FP↔recall tension at ~5 dB SNR that a
   frozen-backbone + linear head can't break.

## Corpus growth round 3 — targeted rare-class boost (the decisive test)

We rendered rare-class-only scenes (drone_binary, drone_bebop, Bear, Frog) at high instance counts
across two SNRs → **+867 balanced rare-class positives**. Corpus now **7,251 samples**; train
balance evened out (drone_binary 90→264, Bear 231→508, Frog 184→452). Retrained (3 seeds,
variance ±0.02–0.06 — stable):

| class | pre-boost | post-boost | train n |
|---|---|---|---|
| Bear | 0.17 | **0.25** ↑ | 508 |
| drone_binary | 0.00 | **0.08** ↑ | 264 |
| Frog | 0.50 | 0.49 → | 452 |
| Elephant | 0.36 | **0.26** ↓ | 473 |
| Lion | 0.75 | **0.63** ↓ | 631 |
| background | 0.69 | 0.63 ↓ | 4007 |

**The boost helped the targeted rare classes but degraded the previously-good ones** (Elephant,
Lion, background). Overall test accuracy did *not* improve (~0.48). Background-capping (3209 train)
behaved the same — it slid FP↔recall, no net gain.

### Decisive conclusion: the linear head is capacity-saturated

This is a **zero-sum reshuffle, not a net gain** — the hallmark of a fixed-capacity classifier. The
frozen-backbone + linear head **cannot fit all 7 classes well simultaneously in heavy ambient**;
adding rare-class data just reallocates its limited capacity from common classes to rare ones.
**Data growth as a lever is now exhausted** — we've scaled volume (7,251), environments (4), SNRs
(3), and balance, and the deployment-holdout accuracy is stuck ~0.5.

## Next — the lever has shifted from data to model capacity

1. **Fine-tune the YAMNet backbone (Phase-2 unfreeze)** — *the* next lever. The linear head is at
   its ceiling; only more model capacity can lift all classes together. Requires resolving the
   Keras/layer-def version pin that blocked the upstream 2-phase trainer ([07](07_yamnet_training.md)).
   The corpus (7,251 labeled, balanced, multi-env, multi-SNR) is ready to drive it.
2. Per-class + background-confidence thresholds to set the deployment operating point.
3. Export + deploy a current (6-class + background) model to ODAS.

**Bottom line of the data-growth arc:** infrastructure works and scales; data volume/balance is no
longer the bottleneck — **model capacity (backbone fine-tuning) is.**

## Reproduce
```bash
python experiments/scripts/corpus.py --build    # (re)populate from all runs
python experiments/scripts/corpus.py --train    # pooled head + per-source eval
```
