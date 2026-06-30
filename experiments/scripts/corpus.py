#!/usr/bin/env python3
"""Persistent, growing labeled training corpus for collective YAMNet-head training.

Every ODAS run contributes post-ODAS .bin spectra, embedded once and appended with provenance:
  - ambient / silence runs  -> every peak = `background`
  - clean / mixed runs       -> GT-matched peaks = animal label; quiet peaks = `background`

Store:  experiments/corpus/embeddings.npy   (N x 1024 float32)
        experiments/corpus/meta.csv         (sample_id,label,source,env,snr_db,run_tag,bin_path,fold)
De-dups by bin_path, so re-adding a run is idempotent.

  python corpus.py --build        # (re)populate from all known runs
  python corpus.py --train        # pooled head training + eval on test fold
"""
import os, sys, csv, glob, argparse
import numpy as np
ROOT="/Users/abhinav/research"; EXP=f"{ROOT}/experiments"; CORP=f"{EXP}/corpus"
sys.path.insert(0,f"{EXP}/scripts")
import expB4_hard_negatives as B4   # embed_bin, load_run
COLS=["sample_id","label","source","env","snr_db","run_tag","bin_path","fold"]

# ---- provenance of every run we've produced ----
# (analysis pkl OR logs dir, scene, source, env, snr, run_tag, fold)
GT_RUNS=[  # clean/mixed: GT->animal, quiet->background
    dict(pkl="a1_analysis.pkl",      scene="exp_a1_no_ambient", source="clean", env="none",   snr="high", tag="a1",      fold="train"),
    dict(pkl="a1_hold_analysis.pkl", scene="exp_a1_holdout",    source="clean", env="none",   snr="high", tag="a1_hold", fold="test"),
    dict(pkl="a5_analysis.pkl",      scene="exp_a5_real",       source="mixed", env="mar30",  snr="~5dB", tag="a5",      fold="train"),
    dict(pkl="a5_hold_analysis.pkl", scene="exp_a5_holdout",    source="mixed", env="mar30",  snr="~5dB", tag="a5_hold", fold="test"),
    dict(pkl="a5_snrhi_analysis.pkl",scene="exp_a5_snrhi",      source="mixed", env="mar30",  snr="~15dB",tag="a5_snrhi",fold="train"),
    dict(pkl="a5_snrlo_analysis.pkl",scene="exp_a5_snrlo",      source="mixed", env="mar30",  snr="~-5dB",tag="a5_snrlo",fold="train"),
    dict(pkl="g1_analysis.pkl",      scene="exp_g1_ecopark",    source="mixed", env="ecopark",snr="~5dB", tag="g1",      fold="train"),
    dict(pkl="g2_analysis.pkl",      scene="exp_g2_first",      source="mixed", env="first",  snr="~11dB",tag="g2",      fold="train"),
    dict(pkl="rare1_analysis.pkl",   scene="exp_rare1",         source="mixed", env="mar30",  snr="~5dB", tag="rare1",   fold="train"),
    dict(pkl="rare2_analysis.pkl",   scene="exp_rare2",         source="mixed", env="first",  snr="~15dB",tag="rare2",   fold="train"),
]
AMBIENT_RUNS=[  # all peaks -> background
    dict(logs="logs_ambient", source="ambient",  env="ecopark", snr="na", tag="amb_ecopark", fold="train"),
    dict(logs="logs_a1",      source="silence",  env="none",    snr="na", tag="sil_a1struct", fold="train", quiet_only=True),
    dict(logs="logs_amb_first",  source="ambient", env="first",  snr="na", tag="amb_first",  fold="train"),
    dict(logs="logs_amb_second", source="ambient", env="second", snr="na", tag="amb_second", fold="train"),
    dict(logs="logs_amb_eco10",  source="ambient", env="eco10",  snr="na", tag="amb_eco10",  fold="train"),
]

def is_null_sp(sp): return sp.sum()<=0 or sp.any(1).sum()<8

def load_corpus():
    if os.path.exists(f"{CORP}/embeddings.npy") and os.path.exists(f"{CORP}/meta.csv"):
        X=np.load(f"{CORP}/embeddings.npy")
        rows=list(csv.DictReader(open(f"{CORP}/meta.csv")))
        return X, rows
    return np.zeros((0,1024),np.float32), []

def save_corpus(X, rows):
    os.makedirs(CORP, exist_ok=True)
    np.save(f"{CORP}/embeddings.npy", X.astype(np.float32))
    with open(f"{CORP}/meta.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=COLS); w.writeheader(); w.writerows(rows)

def build():
    X, rows = np.zeros((0,1024),np.float32), []
    seen=set()
    def add(emb, label, src, env, snr, tag, binp, fold):
        nonlocal X
        if binp in seen: return False
        seen.add(binp); X=np.vstack([X, emb[None]])
        rows.append(dict(sample_id=str(len(rows)), label=label, source=src, env=env,
                         snr_db=snr, run_tag=tag, bin_path=binp, fold=fold)); return True

    # GT runs (clean/mixed)
    for r in GT_RUNS:
        pkl=f"{EXP}/outputs/{r['pkl']}"
        if not os.path.exists(pkl): print("skip (missing):", r['pkl']); continue
        pos,ghost,_,_=B4.load_run(pkl, r["scene"]); n0=len(rows)
        for sf,lbl,_ in pos:
            e=B4.embed_bin(sf)
            if e is not None: add(e,lbl,r["source"],r["env"],r["snr"],r["tag"],sf,r["fold"])
        for sf in ghost:
            e=B4.embed_bin(sf)
            if e is not None: add(e,"background",r["source"],r["env"],r["snr"],r["tag"],sf,r["fold"])
        print(f"{r['tag']:10s} ({r['source']}): +{len(rows)-n0} samples")
    # ambient/silence runs (all peaks -> background)
    for r in AMBIENT_RUNS:
        d=f"{EXP}/odas/{r['logs']}"; n0=len(rows)
        for sf in sorted(glob.glob(f"{d}/*.bin")):
            a=np.fromfile(sf,np.float32)
            if a.size%257: continue
            if is_null_sp(a.reshape(-1,257)): continue
            e=B4.embed_bin(sf)
            if e is not None: add(e,"background",r["source"],r["env"],r["snr"],r["tag"],sf,r["fold"])
        print(f"{r['tag']:10s} ({r['source']}): +{len(rows)-n0} samples")
    save_corpus(X, rows)
    # summary
    import collections
    by=collections.Counter((rw["fold"],rw["label"]) for rw in rows)
    print(f"\ncorpus: {len(rows)} samples -> {CORP}")
    for fold in ("train","test"):
        d={l:n for (f,l),n in by.items() if f==fold}
        print(f"  {fold}: {sum(d.values())}  {dict(sorted(d.items()))}")

def train(bg_cap=None):
    import tensorflow as tf, collections
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    X, rows = load_corpus()
    y=np.array([r["label"] for r in rows]); fold=np.array([r["fold"] for r in rows])
    src=np.array([r["source"] for r in rows])
    classes=sorted(set(y)); ci={c:i for i,c in enumerate(classes)}; yi=np.array([ci[c] for c in y])
    tr=fold=="train"; te=fold=="test"
    if bg_cap:   # downsample background in TRAIN to bg_cap (rebalance)
        rng=np.random.default_rng(0); tri=np.where(tr)[0]
        bg=[i for i in tri if y[i]=="background"]; rng.shuffle(bg)
        drop=set(bg[bg_cap:]); tr=np.array([i not in drop and tr[i] for i in range(len(y))])
        print(f"bg_cap={bg_cap}: train background {len([i for i in tri if y[i]=='background'])} -> {min(bg_cap,len(bg))}")
    sc=StandardScaler().fit(X[tr])
    cw=collections.Counter(yi[tr]); n=tr.sum()
    cwd={i:n/(len(classes)*cw.get(i,1)) for i in range(len(classes))}
    m=tf.keras.Sequential([tf.keras.layers.Input((1024,)),
        tf.keras.layers.Dense(256,activation="relu"),tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128,activation="relu"),tf.keras.layers.Dense(len(classes),activation="softmax")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    m.fit(sc.transform(X[tr]),yi[tr],epochs=80,batch_size=32,class_weight=cwd,verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss",patience=12,restore_best_weights=True)])
    pred=m.predict(sc.transform(X[te]),verbose=0).argmax(1)
    print(f"classes: {classes}")
    print(f"train={tr.sum()} test={te.sum()}  overall test acc={accuracy_score(yi[te],pred):.3f}")
    # per source-of-test breakdown
    for s in sorted(set(src[te])):
        mask=te & (src==s); idx=np.where(mask)[0]
        if len(idx):
            p=m.predict(sc.transform(X[mask]),verbose=0).argmax(1)
            print(f"  test source={s}: n={len(idx)} acc={accuracy_score(yi[mask],p):.3f}")
    os.makedirs(f"{ROOT}/experiments/sim/checkpoints",exist_ok=True)
    m.save(f"{ROOT}/experiments/sim/checkpoints/corpus_head.keras")
    print("saved corpus_head.keras")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--build",action="store_true"); ap.add_argument("--train",action="store_true")
    ap.add_argument("--bg-cap",type=int,default=None)
    a=ap.parse_args()
    if a.build: build()
    if a.train: train(bg_cap=a.bg_cap)
    if not (a.build or a.train): ap.print_help()
