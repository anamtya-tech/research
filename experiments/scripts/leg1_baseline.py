#!/usr/bin/env python3
"""plan.md leg 1 — YAMNet baseline scores. What does raw AudioSet YAMNet already think?

Reads experiments/outputs/baseline_topk.csv (+ class map) and reports, per target class:
  - the most common top-1 AudioSet prediction
  - whether top-1 is "sensible" (animal/relevant vs Music/Speech/etc.)
  - a relevant-AudioSet-class score distribution vs background
Writes figures/baseline_topk_<class>.png histograms and baseline_report.md fragment.
"""
import os, csv, json, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/Users/abhinav/research/experiments/outputs"
FIG = f"{OUT}/figures"; os.makedirs(FIG, exist_ok=True)

# AudioSet classes considered "relevant / sensible" per target (lowercase substring match)
RELEVANT = {
    "Elephant": ["elephant", "animal", "roar", "growl", "rumble", "wild animals", "bellow"],
    "Drone":    ["aircraft", "helicopter", "drone", "engine", "propeller", "motor", "fixed-wing",
                 "buzz", "mosquito", "aircraft engine"],
    "Lion":     ["roar", "animal", "growl", "lion", "cat", "wild animals", "snarl"],
    "Monkey":   ["monkey", "animal", "primate", "ape", "wild animals", "chatter", "squeak"],
    "Frog":     ["frog", "croak", "animal", "insect", "cricket", "wild animals"],
}
CLASSES = list(RELEVANT.keys())

def main():
    rows = list(csv.DictReader(open(f"{OUT}/baseline_topk.csv")))
    by = collections.defaultdict(list)
    for r in rows: by[r["label"]].append(r)

    report = ["## Leg 1 — YAMNet baseline scores\n"]
    report.append("| class | n | most common top-1 | top-1 sensible? | mean top-1 score | "
                  "% clips with a relevant class in top-5 |")
    report.append("|---|---|---|---|---|---|")

    summary = {}
    for c in CLASSES:
        g = by[c]
        top1s = collections.Counter(r["top1"] for r in g)
        common_top1, common_n = top1s.most_common(1)[0]
        rel = [w for w in RELEVANT[c]]
        def has_rel(r):
            t5 = [r[f"top{j}"].lower() for j in range(1, 6)]
            return any(any(w in t for w in rel) for t in t5)
        pct_rel = 100.0 * sum(has_rel(r) for r in g) / len(g)
        top1_sensible = any(w in common_top1.lower() for w in rel)
        mean_top1 = np.mean([float(r["top1_score"]) for r in g])
        report.append(f"| {c} | {len(g)} | {common_top1} ({common_n}) | "
                      f"{'yes' if top1_sensible else 'NO'} | {mean_top1:.3f} | {pct_rel:.0f}% |")
        summary[c] = dict(common_top1=common_top1, common_top1_n=common_n,
                          top1_sensible=bool(top1_sensible), mean_top1_score=round(float(mean_top1),3),
                          pct_relevant_in_top5=round(pct_rel,1),
                          top1_distribution=dict(top1s.most_common(5)))

    # histogram: for each class, "best relevant-class score per clip" vs same metric on background
    # use top1_score as a proxy for peak confidence; compare target vs background distributions
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(4*len(CLASSES), 3.2), sharey=True)
    bg_scores = [float(r["top1_score"]) for r in by["background"]]
    for ax, c in zip(axes, CLASSES):
        ts = [float(r["top1_score"]) for r in by[c]]
        ax.hist(bg_scores, bins=20, range=(0,1), alpha=0.5, label="background", color="grey")
        ax.hist(ts, bins=20, range=(0,1), alpha=0.6, label=c, color="tab:red")
        ax.set_title(c, fontsize=10); ax.set_xlabel("top-1 score")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("clips")
    fig.suptitle("YAMNet top-1 confidence: target vs background")
    fig.tight_layout(); fig.savefig(f"{FIG}/baseline_top1_hist.png", dpi=120); plt.close(fig)

    json.dump(summary, open(f"{OUT}/baseline_summary.json","w"), indent=2)
    print("\n".join(report))
    print(f"\nwrote {FIG}/baseline_top1_hist.png and baseline_summary.json")

if __name__ == "__main__":
    main()
