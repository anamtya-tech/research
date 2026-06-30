#!/usr/bin/env python3
"""plan.md leg 3 — Linear probe. Ceiling for a frozen YAMNet backbone + logistic regression.

One binary classifier per target class (vs background) on mean-pooled 1024-d embeddings.
Uses the existing source-level 80/20 split. Reports Acc/P/R/F1/AUC; saves ROC curves
(overlaid) — this is the canonical ROC plot every later experiment is compared against.
Pass criterion: AUC > 0.70.
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)

OUT = "/Users/abhinav/research/experiments/outputs"
FIG = f"{OUT}/figures"; os.makedirs(FIG, exist_ok=True)
CLASSES = ["Elephant","Drone","Lion","Monkey","Frog"]

def main():
    d = np.load(f"{OUT}/yamnet_features.npz", allow_pickle=True)
    X, y, split = d["emb"], d["label"], d["split"]
    bg = y == "background"

    results = {}
    fig, ax = plt.subplots(figsize=(7,7))
    for c in CLASSES:
        sel = (y == c) | bg
        Xc, yc, sp = X[sel], (y[sel] == c).astype(int), split[sel]
        tr, te = sp == "train", sp == "test"
        scaler = StandardScaler().fit(Xc[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(scaler.transform(Xc[tr]), yc[tr])
        prob = clf.predict_proba(scaler.transform(Xc[te]))[:,1]
        pred = (prob >= 0.5).astype(int)
        auc = roc_auc_score(yc[te], prob)
        results[c] = dict(
            n_train=int(tr.sum()), n_test=int(te.sum()), n_pos_test=int(yc[te].sum()),
            accuracy=round(accuracy_score(yc[te], pred),3),
            precision=round(precision_score(yc[te], pred, zero_division=0),3),
            recall=round(recall_score(yc[te], pred, zero_division=0),3),
            f1=round(f1_score(yc[te], pred, zero_division=0),3),
            auc=round(float(auc),3))
        fpr, tpr, _ = roc_curve(yc[te], prob)
        ax.plot(fpr, tpr, lw=2, label=f"{c} (AUC {auc:.3f})")
        # save roc points for later overlay
        results[c]["roc"] = dict(fpr=fpr.round(4).tolist(), tpr=tpr.round(4).tolist())

    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Linear probe (LogReg on YAMNet embeddings) — class vs background")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(f"{FIG}/roc_linear_probe.png", dpi=130); plt.close(fig)

    json.dump(results, open(f"{OUT}/leg3_linear_probe.json","w"), indent=2)
    print(f"{'class':10s} {'AUC':>6} {'Acc':>6} {'P':>6} {'R':>6} {'F1':>6}  pass(AUC>0.70)")
    for c in CLASSES:
        r = results[c]
        print(f"{c:10s} {r['auc']:6.3f} {r['accuracy']:6.3f} {r['precision']:6.3f} "
              f"{r['recall']:6.3f} {r['f1']:6.3f}  {'PASS' if r['auc']>0.70 else 'FAIL'}")
    print(f"\nwrote {FIG}/roc_linear_probe.png and leg3_linear_probe.json")

if __name__ == "__main__":
    main()
