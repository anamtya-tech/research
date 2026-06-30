#!/usr/bin/env python3
"""EXP-A1 (Track A) — train model_a1_gt: frozen YAMNet backbone + trainable head on the GT
(no-ambient) clips, and evaluate clip-level accuracy on the internal test fold and (optionally)
a separate holdout render.

NOTE: this is Phase-1 of the experiments.pdf 2-phase recipe (frozen backbone). The upstream
train_yamnet.py reconstructs the backbone from tensorflow/models layer-defs for the Phase-2
top-unfreeze and currently breaks on a Keras-version/layer-def mismatch (see report). The
GT-vs-ODAS comparison that EXP-A1/B tests is unaffected: the experimental variable is the
training DATA, and both arms use this identical head-training procedure.

  python expA1_train_gt.py --dataset <gt_dataset_dir> [--holdout <gt_holdout_dir>]
"""
import os, sys, csv, json, argparse, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ROOT = "/Users/abhinav/research"
sys.path.insert(0, f"{ROOT}/experiments/scripts")
from extract_yamnet import waveform_to_patches, MODEL_DIR   # reuse the exact front-end
import soundfile as sf
tf.random.set_seed(42); np.random.seed(42)
OUT = f"{ROOT}/experiments/outputs"; FIG = f"{OUT}/figures"

def load_clips(dataset_dir):
    rows = list(csv.DictReader(open(f"{dataset_dir}/labels.csv")))
    audio = f"{dataset_dir}/audio"
    return rows, audio

def embed_dataset(dataset_dir, sig, in_key):
    rows, audio = load_clips(dataset_dir)
    X, y, folds = [], [], []
    for r in rows:
        w, sr = sf.read(f"{audio}/{r['filename']}")
        if w.ndim > 1: w = w.mean(1)
        patches = waveform_to_patches(tf.constant(w.astype(np.float32)))
        if patches.shape[0] == 0:
            patches = tf.zeros((1, 96, 64))
        emb = sig(**{in_key: patches})["embeddings"].numpy().mean(0)
        X.append(emb); y.append(r["label"]); folds.append(r.get("fold", "train"))
    return np.stack(X), np.array(y), np.array(folds)

def build_head(n_in, n_cls):
    m = tf.keras.Sequential([
        tf.keras.layers.Input((n_in,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(n_cls, activation="softmax")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--tag", default="model_a1_gt")
    args = ap.parse_args()

    model = tf.saved_model.load(MODEL_DIR)
    sig = model.signatures["serving_default"]
    in_key = list(sig.structured_input_signature[1])[0]

    X, y, folds = embed_dataset(args.dataset, sig, in_key)
    classes = sorted(set(y)); cidx = {c: i for i, c in enumerate(classes)}
    yi = np.array([cidx[c] for c in y])
    tr, va, te = folds == "train", folds == "val", folds == "test"
    if va.sum() == 0: va = te                      # tiny datasets: reuse test as val

    scaler = StandardScaler().fit(X[tr])
    Xs = scaler.transform(X)
    cw = collections.Counter(yi[tr])
    n = tr.sum(); class_weight = {i: n/(len(classes)*cw.get(i,1)) for i in range(len(classes))}

    head = build_head(X.shape[1], len(classes))
    head.fit(Xs[tr], yi[tr], validation_data=(Xs[va], yi[va]), epochs=100, batch_size=16,
             class_weight=class_weight, verbose=0,
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                                         restore_best_weights=True)])

    def evaluate(Xe, ye, name):
        pred = head.predict(scaler.transform(Xe), verbose=0).argmax(1)
        acc = accuracy_score(ye, pred)
        cm = confusion_matrix(ye, pred, labels=range(len(classes)))
        rep = classification_report(ye, pred, labels=range(len(classes)),
                                    target_names=classes, output_dict=True, zero_division=0)
        return acc, cm, rep

    te_acc, te_cm, te_rep = evaluate(X[te], yi[te], "test")
    results = {"classes": classes, "n_train": int(tr.sum()), "n_test": int(te.sum()),
               "test_clip_accuracy": round(float(te_acc), 3),
               "per_class_f1": {c: round(te_rep[c]["f1-score"], 3) for c in classes}}

    # confusion matrix figure (internal test)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(te_cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, te_cm[i, j], ha="center", va="center",
                    color="white" if te_cm[i, j] > te_cm.max()/2 else "black", fontsize=9)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{args.tag} — internal test (clip acc {te_acc:.2f})")
    fig.tight_layout(); fig.savefig(f"{FIG}/{args.tag}_confusion.png", dpi=130); plt.close(fig)

    if args.holdout:
        Xh, yh, _ = embed_dataset(args.holdout, sig, in_key)
        hk = np.array([c in cidx for c in yh])
        Xh, yh = Xh[hk], np.array([cidx[c] for c in yh[hk]])
        h_acc, h_cm, h_rep = evaluate(Xh, yh, "holdout")
        results["n_holdout"] = int(len(yh))
        results["holdout_clip_accuracy"] = round(float(h_acc), 3)
        results["holdout_per_class_f1"] = {c: round(h_rep[c]["f1-score"], 3) for c in classes}

    json.dump(results, open(f"{OUT}/{args.tag}_results.json", "w"), indent=2)
    head.save(f"{ROOT}/experiments/sim/checkpoints/{args.tag}_head.keras")
    print(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}/{args.tag}_results.json and {FIG}/{args.tag}_confusion.png")

if __name__ == "__main__":
    main()
