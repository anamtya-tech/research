# Phase 1 · EXP-A1 — SST Preset Sweep (FP/min vs recall)

**Motivation:** EXP-A1 Track B ([`P1_exp_a1_trackB.md`](P1_exp_a1_trackB.md)) found ~20 FP/min even
with no ambient. Before attacking false positives with hard-negative training, we tested whether
**SST configuration alone** can fix it — and documented the full trade-off.

## Method (controlled)

Re-ran `odaslive` on the **same 300 s holdout render** four times, changing **only** the 7 SST
parameters per preset. Everything else identical (model, mic geometry, sim_mode, the render
itself). Each run analyzed through the same `expA1_analyze.py` pipeline (15° matching). Presets
from the simulator's `SST_PRESETS` (= the PDF's Phase-3 EXP-C3 presets):

| param | default (template) | Balanced | Low-FP | High-Recall |
|---|---|---|---|---|
| Pnew | 0.6 | 0.06 | 0.03 | 0.15 |
| N_prob | 3 | 6 | 8 | 3 |
| theta_prob | 0.8 | 0.65 | 0.75 | 0.60 |
| theta_new | 0.85 | 0.80 | 0.85 | 0.60 |
| Pfalse | 0.4 | 0.1 | 0.05 | 0.2 |
| gainMin | 0.25 | 0.40 | 0.45 | 0.30 |
| theta_inactive | 0.75 | 0.80 | 0.85 | 0.70 |

> Note: the repo's **template default** was effectively a high-recall config (`Pnew=0.6, N_prob=3`)
> — more FP-prone than even the named "High-Recall" preset on the new-track probability.

## Result

| preset | N_prob / Pnew | event recall | GT detected | FP in quiet | **FP/min** |
|---|---|---|---|---|---|
| default (template) | 3 / 0.6 | 0.967 | 58/60 | 51 | **18.7** |
| Balanced | 6 / 0.06 | 0.917 | 55/60 | 51 | **18.7** |
| Low-FP | 8 / 0.03 | 0.883 | 53/60 | 38 | **14.0** |
| High-Recall | 3 / 0.15 | 0.933 | 56/60 | 81 | **29.8** |

![SST preset FP/min vs recall trade-off](figures/sst_sweep_tradeoff.png)

(quiet time = 163 s for all; FP/min = FP-in-quiet ÷ quiet-minutes. **Precise FP definition** —
classified detection in a zero-GT-active window, ≈ spurious tracks/min — is in
[`P1_exp_a1_trackB.md` § What counts as a False Positive](P1_exp_a1_trackB.md#what-counts-as-a-false-positive-precise-definition).)

## Observations (documented carefully)

1. **The trade-off is monotonic with permissiveness.** High-Recall (loosest) = most FPs (29.8) and
   most spurious tracks; Low-FP (tightest) = fewest FPs (14.0) but lowest recall (0.88, misses
   7/60 events).
2. **SST tuning alone is NOT enough.** The best preset (Low-FP) still produces **14 FP/min — 7×
   over the ≤ 2 acceptance target**. No SST configuration gets close. This **empirically confirms
   the PDF's stated known issue**: *"GCC-PHAT hotspots → structural FP floor ~0.6/s irreducible by
   SST tuning alone; mitigation: hard-negative training + confidence filtering."*
3. **Balanced was strictly worse than the template default here** — identical FP/min (51 quiet-FPs)
   but lower recall (0.92 vs 0.97). So in this scene the FP floor is **not** set by the new-track
   probability; it's set by structural hotspot directions on which YAMNet still emits an event.
   Lowering `Pnew`/raising `N_prob` suppressed *real* tracks faster than it suppressed the
   structural ghosts.
4. **Recall cost is real but modest** across the usable range (0.88–0.97).

## Conclusion → next step

SST tuning buys a ~25 % FP reduction (18.7 → 14.0 via Low-FP) at a recall cost — useful, but it
**cannot** reach deployment-grade FP rates on its own. The binding constraint is the structural
ghost-track floor, which needs **EXP-B4 hard-negative training** (label ambient/ghost-track
detections as `background`) + confidence filtering (EXP-C1). That is now the clear next priority.

**Operating-point guidance (per PDF roles):**
- **Dataset collection:** use **High-Recall** (catch the most events; FPs get curated out).
- **Deployment simulation:** use **Low-FP** as the SST floor, then stack hard negatives +
  per-class confidence thresholds on top to drive FP/min down toward target.

## Caveats

- Single 300 s render, one scene family (no ambient). FP/min absolute values will shift with
  ambient (Phase-1 A2–A5) and scene density; the **relative** ordering and the "SST-insufficient"
  conclusion are the robust takeaways.
- FP/min here counts classified detections (those that passed YAMNet's no-class gate) landing in
  quiet periods — consistent across all four runs, so comparisons are valid.

## Artifacts

`experiments/outputs/{a1_hold,a1_bal,a1_lowfp,a1_hirec}_deploy_metrics.json`,
`figures/sst_sweep_tradeoff.png`. cfgs: `experiments/odas/local_socket_{balanced,lowfp,hirecall}.cfg`.
ODAS logs: `experiments/odas/logs_{hold,bal,lowfp,hirec}/`.
