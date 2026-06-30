#!/usr/bin/env python3
"""EXP-A1 core test — GT-trained vs post-ODAS-trained YAMNet head, both evaluated on the SAME
post-ODAS holdout (the deployment distribution). Tests experiments.pdf's central hypothesis:
a model trained on post-ODAS data should transfer better to live deployment.

Embeddings come from the validated YAMNet core (export_out/tf2):
  - GT clips:        wav -> STFT mel patches -> embeddings
  - post-ODAS .bin:  96x257 spectra -> mel (257->64) -> embeddings   (deployment-faithful)

  python expA1_compare.py --gt-dataset <gt_dir> --odas-train <a1_matches.json> \
                          --odas-holdout <holdout_matches.json>
"""
import os, sys, json, argparse, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ROOT = "/Users/abhinav/research"
sys.path.insert(0, f"{ROOT}/experiments/scripts")
from extract_yamnet import waveform_to_patches, MODEL_DIR, _MEL, LOG_OFF
import soundfile as sf
tf.random.set_seed(0); np.random.seed(0)
OUT = f"{ROOT}/experiments/outputs"; FIG = f"{OUT}/figures"

_model = tf.saved_model.load(MODEL_DIR)
_sig = _model.signatures["serving_default"]
_inkey = list(_sig.structured_input_signature[1])[0]

def emb_from_patches(patches):
    if patches.shape[0] == 0: patches = tf.zeros((1, 96, 64))
    return _sig(**{_inkey: patches})["embeddings"].numpy().mean(0)

def embed_wav(path):
    w, sr = sf.read(path)
    if w.ndim > 1: w = w.mean(1)
    return emb_from_patches(waveform_to_patches(tf.constant(w.astype(np.float32))))

def embed_bin(path):
    a = np.fromfile(path, dtype=np.float32)
    if a.size % 257: return None
    spec = a.reshape(-1, 257)                      # (96, 257) linear magnitude
    mel = np.log(spec @ _MEL.numpy() + LOG_OFF)    # (96, 64) — same mel as training/firmware
    return emb_from_patches(tf.constant(mel[None].astype(np.float32)))

def load_gt(dataset_dir):
    import csv
    rows = list(csv.DictReader(open(f"{dataset_dir}/labels.csv")))
    X, y, g = [], [], []
    for r in rows:
        X.append(embed_wav(f"{dataset_dir}/audio/{r['filename']}"))
        y.append(r["label"]); g.append(r["filename"].split("/")[0])  # group by source label dir
    return np.array(X), np.array(y), np.array(g)

def load_odas(matches_json):
    rows = json.load(open(matches_json))
    X, y, g = [], [], []
    for r in rows:
        sf_ = r.get("spectra_file")
        if not sf_ or not os.path.exists(sf_): continue
        e = embed_bin(sf_)
        if e is None: continue
        X.append(e); y.append(r["label"]); g.append(f"{r['label']}_{r['src_start']}_{r['src_end']}")
    return np.array(X), np.array(y), np.array(g)

def split_by_group(g, frac=0.2, seed=0):
    rng = np.random.default_rng(seed); groups = np.array(sorted(set(g)))
    rng.shuffle(groups); n_te = max(1, int(round(len(groups)*frac)))
    te = set(groups[:n_te]); return np.array([gi not in te for gi in g]), np.array([gi in te for gi in g])

def build_head(n_in, n_cls):
    m = tf.keras.Sequential([tf.keras.layers.Input((n_in,)),
        tf.keras.layers.Dense(256, activation="relu"), tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation="relu"), tf.keras.layers.Dense(n_cls, activation="softmax")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def train_head(X, y, classes):
    cidx = {c: i for i, c in enumerate(classes)}; yi = np.array([cidx[c] for c in y])
    sc = StandardScaler().fit(X); Xs = sc.transform(X)
    cw = collections.Counter(yi); n = len(yi)
    cwd = {i: n/(len(classes)*cw.get(i, 1)) for i in range(len(classes))}
    h = build_head(X.shape[1], len(classes))
    h.fit(Xs, yi, epochs=80, batch_size=16, class_weight=cwd, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss", patience=12, restore_best_weights=True)])
    return h, sc

def evaluate(h, sc, X, y, classes):
    cidx = {c: i for i, c in enumerate(classes)}; yi = np.array([cidx[c] for c in y])
    pred = h.predict(sc.transform(X), verbose=0).argmax(1)
    rep = classification_report(yi, pred, labels=range(len(classes)), target_names=classes,
                                output_dict=True, zero_division=0)
    return accuracy_score(yi, pred), {c: round(rep[c]["f1-score"], 3) for c in classes}, \
           confusion_matrix(yi, pred, labels=range(len(classes)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dataset", required=True)
    ap.add_argument("--odas-train", required=True)
    ap.add_argument("--odas-holdout", required=True)
    args = ap.parse_args()

    print("embedding GT clips ..."); Xg, yg, gg = load_gt(args.gt_dataset)
    print("embedding ODAS-train .bins ..."); Xo, yo, go = load_odas(args.odas_train)
    print("embedding ODAS-holdout .bins ..."); Xh, yh, gh = load_odas(args.odas_holdout)
    classes = sorted(set(yg) | set(yo) | set(yh))
    print(f"classes: {classes}")
    print(f"  GT train clips: {len(yg)}  ODAS train: {len(yo)}  ODAS holdout: {len(yh)}")

    hg, scg = train_head(Xg, yg, classes)
    ho, sco = train_head(Xo, yo, classes)

    acc_gt, f1_gt, cm_gt = evaluate(hg, scg, Xh, yh, classes)
    acc_od, f1_od, cm_od = evaluate(ho, sco, Xh, yh, classes)

    res = dict(classes=classes, n_gt_train=len(yg), n_odas_train=len(yo), n_holdout=len(yh),
               holdout="post-ODAS",
               gt_model_holdout_acc=round(float(acc_gt), 3), gt_model_f1=f1_gt,
               odas_model_holdout_acc=round(float(acc_od), 3), odas_model_f1=f1_od)
    json.dump(res, open(f"{OUT}/expA1_compare.json", "w"), indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for a, cm, ttl, acc in [(ax[0], cm_gt, "GT-trained", acc_gt), (ax[1], cm_od, "post-ODAS-trained", acc_od)]:
        im = a.imshow(cm, cmap="Blues"); a.set_title(f"{ttl} → post-ODAS holdout (acc {acc:.2f})", fontsize=10)
        a.set_xticks(range(len(classes))); a.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        a.set_yticks(range(len(classes))); a.set_yticklabels(classes, fontsize=8)
        for i in range(len(classes)):
            for j in range(len(classes)):
                a.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                       color="white" if cm[i, j] > cm.max()/2 else "black")
        a.set_xlabel("pred"); a.set_ylabel("true")
    fig.suptitle("EXP-A1: GT-trained vs post-ODAS-trained, evaluated on post-ODAS holdout")
    fig.tight_layout(); fig.savefig(f"{FIG}/expA1_gt_vs_odas.png", dpi=130); plt.close(fig)

    print("\n=== EXP-A1 core comparison (both eval on post-ODAS holdout) ===")
    print(f"  GT-trained model    : holdout acc {acc_gt:.3f}  f1 {f1_gt}")
    print(f"  post-ODAS-trained   : holdout acc {acc_od:.3f}  f1 {f1_od}")
    print(f"\nwrote {OUT}/expA1_compare.json and {FIG}/expA1_gt_vs_odas.png")

if __name__ == "__main__":
    main()
