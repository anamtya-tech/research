# Phase 3 (separate track) · YAMNet Backbone Fine-Tuning

**Status:** ✅ unblocked and working, as a **standalone script** that does **not** modify any of the
chatak-odas / yamnet / simulator repos. `experiments/scripts/finetune_backbone.py`.

This is the lever the data-growth arc pointed to ([P2](P2_corpus_multisnr.md)): the frozen-backbone
+ linear head is capacity-saturated, so only adding model capacity (unfreezing the backbone) can
lift all classes together.

## How it works (and why it's separate)

The upstream `yamnet/training/train_yamnet.py` crashed rebuilding the backbone — a **Keras 3 vs
Keras 2** mismatch, not a logic bug. This script sidesteps it without touching repo code:

1. `TF_USE_LEGACY_KERAS=1` → the official `yamnet.py` layer defs build under Keras 2.
2. Rebuild the YAMNet core (Reshape → 14 conv/sep-conv blocks → `embeddings` → Dense) and **load
   the pretrained weights from the local SavedModel** (`yamnet/export_out/tf2`) via the serving
   signature's `variables`, in order. **Verified:** `max|Δ prediction| = 1.8e-7` vs the SavedModel.
3. Attach a fresh 7-class head; **Phase 1** train head (backbone frozen); **Phase 2** unfreeze the
   top-N separable-conv blocks (BatchNorm kept frozen), lr 1e-5.
4. Inputs = the **corpus mel patches** re-derived from each sample's `.bin` (so we fine-tune on
   exactly the pooled, balanced, multi-SNR corpus). Eval on the test fold.

## Result (8+8 epochs, unfreeze top 4 blocks)

| class | frozen linear head | **backbone fine-tune** |
|---|---|---|
| Bear | 0.25 | **0.47** |
| drone_binary | 0.08 | **0.31** |
| drone_bebop | 0.29 | 0.37 |
| Frog | 0.49 | 0.55 |
| Elephant | 0.26 | 0.32 |
| Lion | 0.63 | 0.68 |
| background | 0.63 | 0.58 |
| overall / mixed | ~0.48 / ~0.59 | 0.51 / 0.51 |

- **The capacity hypothesis is confirmed:** unfreezing the backbone lifts the **rare classes
  substantially** (Bear 0.25→0.47, drone_binary 0.08→0.31) **without** the zero-sum degradation the
  linear head showed — Lion/Frog/Elephant also rose. This is the net gain data alone couldn't buy.
- **But it overfits at current scale:** Phase-2 **train accuracy 0.82 vs test 0.51**. This quick run
  used no validation split / early stopping. The headroom is real (train fits well); closing the
  generalization gap is the next job.

## Roadmap-A result — regularization + SpecAugment broke the ceiling

We then matured the fine-tune (validation split + EarlyStopping, dropout 0.5, AdamW weight-decay,
unfreeze-depth sweep) and finally added **SpecAugment** (random time/frequency masking of the mel
patches). SpecAugment was the decisive lever.

| | frozen linear head | backbone FT (no aug) | **FT + SpecAugment** |
|---|---|---|---|
| overall test | ~0.48 | 0.51 | **0.62** |
| mixed (deployment-realistic) | ~0.59 | 0.53 | **0.61** |
| train/test gap | — | 0.33 | **0.13** |
| Bear | 0.25 | 0.37 | **0.53** |
| Elephant | 0.26 | 0.38 | **0.54** |
| Frog | 0.49 | 0.47 | **0.68** |
| Lion | 0.63 | 0.70 | **0.90** |
| drone_bebop | 0.29 | 0.34 | **0.54** |
| drone_binary | 0.08 | 0.03 | 0.19 |
| background | 0.63 | 0.66 | 0.64 |

(unfreeze=6, 2 seeds, val/early-stopping; numbers are the better seed, both ~0.61–0.62.)

**Findings:**
- **The ~0.50 ceiling broke: overall 0.48 → 0.62 (+14 pts), gap 0.33 → 0.13.** Nearly every class
  jumped (Lion 0.90, Frog 0.68, Elephant/Bear/drone_bebop ~0.53).
- **The binding issue was cross-render generalization, not capacity or volume.** Weight-decay +
  dropout couldn't fix it (gap stayed 0.33); **SpecAugment** — which forces the model to ignore
  render-specific spectral detail — closed it. Diagnosis confirmed.
- **drone_binary is the lone laggard** (0.19, n=32 test — noisy). Needs more distinct drone_binary
  source variety; also our early-stopping val is leaky (random patches from train renders), so a
  render-level val would give a cleaner stop.

## What remains for this track

1. **drone_binary:** more distinct source clips / targeted renders; render-level (leakage-free) val.
2. **Cleaner, larger real-ambient test** + more seeds (635-sample holdout is noisy, ±0.02–0.06).
3. **Tune SpecAugment + unfreeze depth** further now that it clearly helps.
4. **Export → TFLite → deploy** the fine-tuned 7-class model (see roadmap).

## Run

```bash
python experiments/scripts/finetune_backbone.py --sweep 6 --seeds 0,1 --augment
```

## Run

```bash
python experiments/scripts/finetune_backbone.py --phase1 8 --phase2 8 --unfreeze 4
# weight transfer is verified at startup (aborts if variable order mismatches)
```

> Kept entirely under `experiments/`. When this track matures, the fix to fold back into
> `yamnet/training/train_yamnet.py` is simply pinning the `tensorflow/models` commit + running under
> `TF_USE_LEGACY_KERAS=1` (the calling convention is already correct in current upstream).
