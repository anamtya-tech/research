#!/usr/bin/env python3
"""EXP-B4 — hard negatives. Do post-ODAS ghost spectra, labeled `background`, let the model
reject structural false positives WITHOUT the recall collapse a vote gate causes?

Train two heads on post-ODAS YAMNet embeddings, evaluate both on the holdout:
  - 6-class  : GT-matched detections only (no background)         [baseline]
  - 7-class  : + quiet-period ghost .bins labeled `background`    [hard negatives]

Hard negatives come from the 600s render's quiet ghosts; holdout ghosts come from the separate
300s render → no leakage. FP/min counts holdout quiet-ghosts classified as a NON-background class
(i.e., a ghost that still fires an animal/drone alert). Recall = GT events with >=1 matched
detection classified as the correct animal.
"""
import os, sys, json, math, pickle, collections
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
tf.random.set_seed(0); np.random.seed(0)
ROOT="/Users/abhinav/research"; EXP=f"{ROOT}/experiments"
sys.path.insert(0, f"{EXP}/scripts")
from extract_yamnet import MODEL_DIR, _MEL, LOG_OFF
_m=tf.saved_model.load(MODEL_DIR); _sig=_m.signatures["serving_default"]
_ik=list(_sig.structured_input_signature[1])[0]
_MELn=_MEL.numpy()

def embed_bin(sf):
    a=np.fromfile(sf,np.float32)
    if a.size%257: return None
    mel=np.log(a.reshape(-1,257)@_MELn + LOG_OFF)
    return _sig(**{_ik: tf.constant(mel[None].astype(np.float32))})["embeddings"].numpy().mean(0)

def load_run(pkl, scene_f):
    scene=json.load(open(f"{EXP}/sim/scenes/{scene_f}.json"))
    iv=sorted((s["start_time"],s["end_time"]) for s in scene["directional_sources"])
    dur=scene["duration"]; cov=0;cs=ce=None
    for a,b in iv:
        if cs is None:cs,ce=a,b
        elif a<=ce:ce=max(ce,b)
        else:cov+=ce-cs;cs,ce=a,b
    if cs is not None:cov+=ce-cs
    quiet_min=max(1e-6,dur-cov)/60
    gtev={(s["label"],s["start_time"],s["end_time"]) for s in scene["directional_sources"]}
    def inq(t): return t is not None and not any(a<=t<=b for a,b in iv)
    def isnull(d): return (d.get("x")==0 and d.get("y")==0 and d.get("z")==0) or (d.get("activity") is not None and d.get("activity")<=0)
    r=pickle.load(open(pkl,"rb"))
    pos=[]; ghost=[]
    for m in r["matches"]:
        d=m.get("detection") or {}; sf=d.get("spectra_file")
        if not sf: continue
        sf=sf.replace("/exp/", f"{EXP}/")
        if not os.path.exists(sf): continue
        if m.get("match_type")=="ground_truth":
            pos.append((sf, m["label"], (m["label"],m["source"]["start_time"],m["source"]["end_time"])))
        elif inq(d.get("timestamp")) and not isnull(d):
            ghost.append(sf)
    return pos, ghost, gtev, quiet_min

def embed_many(sfs):
    out=[];
    for s in sfs:
        e=embed_bin(s)
        if e is not None: out.append(e)
    return np.array(out)

def build_head(n_in,n_cls):
    m=tf.keras.Sequential([tf.keras.layers.Input((n_in,)),
        tf.keras.layers.Dense(256,activation="relu"),tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128,activation="relu"),tf.keras.layers.Dense(n_cls,activation="softmax")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return m

def train(X,y,classes):
    ci={c:i for i,c in enumerate(classes)}; yi=np.array([ci[c] for c in y])
    sc=StandardScaler().fit(X); cw=collections.Counter(yi); n=len(yi)
    cwd={i:n/(len(classes)*cw.get(i,1)) for i in range(len(classes))}
    h=build_head(X.shape[1],len(classes))
    h.fit(sc.transform(X),yi,epochs=80,batch_size=16,class_weight=cwd,verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss",patience=12,restore_best_weights=True)])
    return h,sc,ci

def main():
    # ---- training data: 600s render ----
    tr_pos, tr_ghost, _, _ = load_run(f"{EXP}/outputs/a1_analysis.pkl","exp_a1_no_ambient")
    Xp=embed_many([p[0] for p in tr_pos]); yp=[p[1] for p in tr_pos]
    Xg=embed_many(tr_ghost);
    print(f"train: {len(yp)} positives ({dict(collections.Counter(yp))}), {len(Xg)} ghost hard-negatives")

    animal=sorted(set(yp))
    # 6-class baseline (positives only)
    h6,sc6,ci6 = train(Xp,yp,animal)
    # 7-class with hard negatives
    classes7=animal+["background"]
    X7=np.vstack([Xp,Xg]); y7=yp+["background"]*len(Xg)
    h7,sc7,ci7 = train(X7,y7,classes7)

    # ---- holdout eval: 300s render ----
    h_pos,h_ghost,gtev,qmin = load_run(f"{EXP}/outputs/a1_hold_analysis.pkl","exp_a1_holdout")
    Xhp=embed_many([p[0] for p in h_pos]); keys=[p[2] for p in h_pos]; ylab=[p[1] for p in h_pos]
    Xhg=embed_many(h_ghost)
    print(f"holdout: {len(Xhp)} positives, {len(Xhg)} ghosts (quiet, non-null)")

    def evalmodel(h,sc,ci,tag):
        inv={v:k for k,v in ci.items()}
        # positives -> recall (GT event has >=1 detection predicted as its correct animal)
        pp=h.predict(sc.transform(Xhp),verbose=0).argmax(1); pred=[inv[i] for i in pp]
        detected=set()
        for k,yl,pr in zip(keys,ylab,pred):
            if pr==yl: detected.add(k)
        recall=len(detected & gtev)/len(gtev)
        clipacc=np.mean([pr==yl for yl,pr in zip(ylab,pred)])
        # ghosts -> FP = predicted non-background
        if len(Xhg):
            gp=h.predict(sc.transform(Xhg),verbose=0).argmax(1); gpred=[inv[i] for i in gp]
            fp=sum(1 for p in gpred if p!="background")
            bg=sum(1 for p in gpred if p=="background")
        else: fp=bg=0
        fpmin=fp/qmin
        print(f"  [{tag}] recall={recall:.3f} clip_acc={clipacc:.3f} | ghosts: {fp} fired / {len(Xhg)} "
              f"({bg} -> background) | FP/min={fpmin:.1f}")
        return dict(model=tag,recall=round(recall,3),clip_acc=round(float(clipacc),3),
                    ghost_total=len(Xhg),ghost_fired=fp,ghost_to_background=bg,fp_per_min=round(fpmin,2))

    print("\n=== EXP-B4: effect of hard negatives (holdout, votes>=1, null-filtered) ===")
    r6=evalmodel(h6,sc6,ci6,"6-class (no hard-neg)")
    r7=evalmodel(h7,sc7,ci7,"7-class (+hard-neg)")
    res=dict(baseline=r6,hard_neg=r7,
             fp_reduction=f"{r6['fp_per_min']} -> {r7['fp_per_min']} /min",
             recall_change=f"{r6['recall']} -> {r7['recall']}")
    json.dump(res,open(f"{EXP}/outputs/expB4_results.json","w"),indent=2)

    fig,ax=plt.subplots(figsize=(7,5))
    ax.scatter([r6["fp_per_min"]],[r6["recall"]],s=120,c="tab:gray",label="6-class (no hard-neg)",zorder=3)
    ax.scatter([r7["fp_per_min"]],[r7["recall"]],s=120,c="tab:green",label="7-class (+hard-neg)",zorder=3)
    ax.annotate("baseline",(r6["fp_per_min"],r6["recall"]),xytext=(6,-12),textcoords="offset points",fontsize=9)
    ax.annotate("+hard negatives",(r7["fp_per_min"],r7["recall"]),xytext=(6,6),textcoords="offset points",fontsize=9)
    ax.axvline(2,ls="--",c="green",lw=1.2); ax.text(2.2,min(r6["recall"],r7["recall"]),"FP/min ≤ 2",color="green",fontsize=8)
    ax.set_xlabel("FP/min (lower better)"); ax.set_ylabel("event recall (higher better)")
    ax.set_title("EXP-B4: hard negatives — FP/min vs recall (holdout)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{EXP}/outputs/figures/expB4_hard_negatives.png",dpi=130)
    import shutil; shutil.copy(f"{EXP}/outputs/figures/expB4_hard_negatives.png",f"{ROOT}/docs/figures/")
    print(f"\nwrote expB4_results.json and figures/expB4_hard_negatives.png")

if __name__=="__main__":
    main()
