#!/usr/bin/env python3
"""plan.md leg 4 — Small MLP head. Does non-linearity beat logistic regression?

Per target class: MLP(1024 -> 256 -> 128 -> 1 sigmoid), Adam lr=1e-3, binary cross-entropy,
50 epochs, early stopping on val loss (patience=10). Plots train/val loss; compares AUC to
leg 3. Pass criterion: AUC improves > 2% (0.02) over the linear probe.
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

OUT = "/Users/abhinav/research/experiments/outputs"
FIG = f"{OUT}/figures"; os.makedirs(FIG, exist_ok=True)
CLASSES = ["Elephant","Drone","Lion","Monkey","Frog"]
tf.random.set_seed(42); np.random.seed(42)

def build():
    m = tf.keras.Sequential([
        tf.keras.layers.Input((1024,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["AUC"])
    return m

def main():
    d = np.load(f"{OUT}/yamnet_features.npz", allow_pickle=True)
    X, y, split = d["emb"], d["label"], d["split"]
    lin = json.load(open(f"{OUT}/leg3_linear_probe.json"))
    bg = y == "background"

    results = {}
    fig_l, axes_l = plt.subplots(1, len(CLASSES), figsize=(4*len(CLASSES), 3.2))
    fig_r, ax_r = plt.subplots(figsize=(7,7))
    for ax_loss, c in zip(axes_l, CLASSES):
        sel = (y == c) | bg
        Xc, yc, sp = X[sel], (y[sel] == c).astype(int), split[sel]
        tr, te = sp == "train", sp == "test"
        scaler = StandardScaler().fit(Xc[tr])
        Xtr, Xte = scaler.transform(Xc[tr]), scaler.transform(Xc[te])
        ytr, yte = yc[tr], yc[te]
        Xt, Xv, yt, yv = train_test_split(Xtr, ytr, test_size=0.2, stratify=ytr, random_state=42)
        cw = {0: len(ytr)/(2*(ytr==0).sum()), 1: len(ytr)/(2*(ytr==1).sum())}
        m = build()
        hist = m.fit(Xt, yt, validation_data=(Xv, yv), epochs=50, batch_size=16,
                     class_weight=cw, verbose=0,
                     callbacks=[tf.keras.callbacks.EarlyStopping(
                         monitor="val_loss", patience=10, restore_best_weights=True)])
        prob = m.predict(Xte, verbose=0).ravel()
        auc = float(roc_auc_score(yte, prob))
        lin_auc = lin[c]["auc"]
        results[c] = dict(auc_mlp=round(auc,3), auc_linear=lin_auc,
                          delta=round(auc-lin_auc,3), epochs_run=len(hist.history["loss"]),
                          pass_gt_2pct=bool(auc-lin_auc > 0.02))
        ax_loss.plot(hist.history["loss"], label="train")
        ax_loss.plot(hist.history["val_loss"], label="val")
        ax_loss.set_title(f"{c}", fontsize=10); ax_loss.set_xlabel("epoch"); ax_loss.legend(fontsize=7)
        fpr, tpr, _ = roc_curve(yte, prob)
        ax_r.plot(fpr, tpr, lw=2, label=f"{c} MLP {auc:.3f} / lin {lin_auc:.3f}")

    axes_l[0].set_ylabel("BCE loss")
    fig_l.suptitle("MLP head — train vs val loss"); fig_l.tight_layout()
    fig_l.savefig(f"{FIG}/mlp_loss_curves.png", dpi=120); plt.close(fig_l)
    ax_r.plot([0,1],[0,1],"k--",lw=1); ax_r.set_xlabel("FPR"); ax_r.set_ylabel("TPR")
    ax_r.set_title("MLP head ROC (vs linear-probe AUC in legend)"); ax_r.legend(loc="lower right")
    fig_r.tight_layout(); fig_r.savefig(f"{FIG}/roc_mlp_head.png", dpi=130); plt.close(fig_r)

    json.dump(results, open(f"{OUT}/leg4_mlp_head.json","w"), indent=2)
    print(f"{'class':10s} {'MLP':>6} {'lin':>6} {'Δ':>7}  >2%?")
    for c in CLASSES:
        r = results[c]
        print(f"{c:10s} {r['auc_mlp']:6.3f} {r['auc_linear']:6.3f} {r['delta']:+7.3f}  "
              f"{'YES' if r['pass_gt_2pct'] else 'no'}")
    print(f"\nwrote {FIG}/mlp_loss_curves.png, roc_mlp_head.png, leg4_mlp_head.json")

if __name__ == "__main__":
    main()
