# L0 — plan.md Results Summary (Level-0 warm-up)

The four `plan.md` legs after Data Prep, run on the 450-clip inventory. **One question:** is
YAMNet's embedding space good enough for our classes, or does it need backbone work?
**Answer: the embedding space is excellent; only the head needs adapting.**

## Where everything lives

| What | Path |
|---|---|
| Plots (PNG) | `experiments/outputs/figures/` |
| Per-clip top-5, metrics, embeddings | `experiments/outputs/*.csv`, `*.json`, `yamnet_features.npz` |
| Scripts | `experiments/scripts/{extract_yamnet,leg1_baseline,leg2_embeddings,leg3_linear_probe,leg4_mlp_head}.py` |
| Per-leg reports | `docs/L1`–`L4` |

## Findings at a glance

| Leg | Result | Plot |
|---|---|---|
| [L1 baseline](L1_yamnet_baseline.md) | Raw 521-class head useless (Elephant→Jackhammer, Frog→Eruption; 0–12% relevant in top-5) | `baseline_top1_hist.png` |
| [L2 embeddings](L2_embedding_viz.md) | kNN purity 0.80; Drone/Lion/Frog/Monkey cluster cleanly; Elephant overlaps background | `embeddings_2d.png` |
| [L3 linear probe](L3_linear_probe.md) | AUC 0.90 (Elephant) → 1.00 (Drone/Lion/Monkey), Frog 0.99 — **all PASS** | `roc_linear_probe.png` |
| [L4 MLP head](L4_mlp_head.md) | No improvement over linear (Δ ≤ 0); Elephant overfits | `mlp_loss_curves.png`, `roc_mlp_head.png` |

## The throughline

1. The **head** knows nothing about our classes out of the box (L1).
2. But the **embeddings** separate them almost perfectly (L2, L3) — the head is the only problem.
3. So a **linear probe is at ceiling**, and added non-linearity only overfits (L4).
4. ⇒ Effort belongs in **features/data realism**, not model capacity. **Elephant** is the one
   class that's genuinely entangled with background and needs targeted help.

## Caveat → hand-off to the bigger plan (`experiments.pdf`)

These are **clean library clips**, so this is an *upper bound*, not a deployment estimate. The
real program trains/evaluates on **post-ODAS** reconstructed spectra (beamformed, Griffin-Lim
artefacts, ambient ghost-track false positives). What L1–L4 establish for that program:
- YAMNet embeddings are the right representation — proceed with the head/fine-tune approach.
- Watch **Elephant** (low-freq rumble ≈ ambient) and **false positives from background** — exactly
  the focus of `experiments.pdf` Phase 1 (ambient ablation) and Phase 2 EXP-B4 (hard negatives).

See [`docs/README.md`](README.md) for the full step/leg index.
