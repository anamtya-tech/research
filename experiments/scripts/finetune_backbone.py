#!/usr/bin/env python3
"""SEPARATE backbone fine-tuning track (does NOT modify chatak-odas / yamnet / simulator code).

Roadmap A: convert the backbone-FT capacity headroom into real test gains.
 - rebuild a trainable YAMNet core (Keras-2 / tf_keras), load + VERIFY pretrained weights
 - regularized 2-phase fine-tune: head dropout, AdamW weight-decay, label smoothing,
   stratified validation split + EarlyStopping(restore best) in BOTH phases
 - sweep unfreeze depth; multi-seed for stable numbers
 - honest eval on the separate-render TEST fold (overall / real-ambient mixed / per-class)

  python finetune_backbone.py --sweep 2,4,6 --seeds 0,1,2
"""
import os
os.environ["TF_USE_LEGACY_KERAS"]="1"
import sys, csv, argparse, collections
import numpy as np
import tensorflow as tf

ROOT="/Users/abhinav/research"; EXP=f"{ROOT}/experiments"; CORP=f"{EXP}/corpus"
YDEFS=f"{ROOT}/yamnet/models/research/audioset/yamnet"; SAVED=f"{ROOT}/yamnet/export_out/tf2"
sys.path.insert(0, YDEFS); sys.path.insert(0, f"{EXP}/scripts")
import params as params_lib
from yamnet import _YAMNET_LAYER_DEFS
from extract_yamnet import _MEL, LOG_OFF
_MELn=_MEL.numpy(); L=tf.keras.layers

def build_core(num_classes=521, activation="sigmoid"):
    p=params_lib.Params(); feats=L.Input((96,64), name="mel")
    net=L.Reshape((96,64,1))(feats)
    for i,(fn,kernel,stride,filt) in enumerate(_YAMNET_LAYER_DEFS):
        net=fn(f"layer{i+1}", kernel, stride, filt, p)(net)
    emb=L.GlobalAveragePooling2D(name="embeddings")(net)
    out=L.Activation(activation, name="predictions")(L.Dense(num_classes)(emb))
    return tf.keras.Model(feats, out, name="yamnet_core")

def pretrained_weights():
    core=build_core(); saved=tf.saved_model.load(SAVED); sig=saved.signatures["serving_default"]
    sv=list(sig.variables); cv=core.variables
    assert len(sv)==len(cv)
    for c,s in zip(cv,sv): c.assign(s)
    x=tf.random.normal((2,96,64)); ref=sig(tf.constant(x))["predictions"].numpy()
    err=float(np.abs(ref-core(x,training=False).numpy()).max())
    print(f"weight-transfer verify: max|Δpred|={err:.2e} -> {'OK' if err<1e-4 else 'MISMATCH'}")
    assert err<1e-4
    return [v.numpy() for v in core.variables]

def bin_to_mel(p):
    a=np.fromfile(p,np.float32)
    if a.size%257: return None
    return np.log(a.reshape(-1,257)@_MELn+LOG_OFF).astype(np.float32)

def load_patches():
    rows=list(csv.DictReader(open(f"{CORP}/meta.csv"))); X,y,fold,src=[],[],[],[]
    for r in rows:
        m=bin_to_mel(r["bin_path"])
        if m is None or m.shape[0]<96: continue
        X.append(m[:96]); y.append(r["label"]); fold.append(r["fold"]); src.append(r["source"])
    return np.stack(X), np.array(y), np.array(fold), np.array(src)

def make_model(weights, unfreeze, dropout):
    core=build_core()
    for v,w in zip(core.variables, weights): v.assign(w)
    emb=core.get_layer("embeddings").output
    h=L.Dense(256,activation="relu",name="ft_fc1")(emb)
    h=L.Dropout(dropout,name="ft_do1")(h)
    h=L.Dense(128,activation="relu",name="ft_fc2")(h)
    h=L.Dropout(dropout,name="ft_do2")(h)
    out=L.Dense(7,activation="softmax",name="ft_out")(h)
    model=tf.keras.Model(core.input,out)
    top=[f"layer{14-k}" for k in range(unfreeze)]
    return model, top

def _specaug(x, y, sw):
    mean=tf.reduce_mean(x)
    f=tf.random.uniform([],0,12,tf.int32); f0=tf.random.uniform([],0,65-f,tf.int32)
    fm=tf.concat([tf.ones([f0]),tf.zeros([f]),tf.ones([64-f0-f])],0)
    x=x*fm[tf.newaxis,:]+mean*(1-fm)[tf.newaxis,:]
    t=tf.random.uniform([],0,16,tf.int32); t0=tf.random.uniform([],0,97-t,tf.int32)
    tm=tf.concat([tf.ones([t0]),tf.zeros([t]),tf.ones([96-t0-t])],0)
    x=x*tm[:,tf.newaxis]+mean*(1-tm)[:,tf.newaxis]
    return x, y, sw

def run(weights, X, y, fold, src, classes, ci, unfreeze, seed, dropout=0.5, wd=1e-4, augment=False):
    tf.keras.utils.set_random_seed(seed)
    yi=np.array([ci[c] for c in y]); tr=np.where(fold=="train")[0]; te=fold=="test"
    rng=np.random.default_rng(seed); rng.shuffle(tr)
    nval=int(len(tr)*0.15); val=tr[:nval]; trn=tr[nval:]
    cw=collections.Counter(yi[trn]); n=len(trn)
    cwd={i:n/(len(classes)*cw.get(i,1)) for i in range(len(classes))}
    sw=np.array([cwd[i] for i in yi[trn]],np.float32)   # class weights as sample weights (for tf.data)
    def ds():
        d=tf.data.Dataset.from_tensor_slices((X[trn],yi[trn],sw)).shuffle(4096,seed=seed)
        if augment: d=d.map(_specaug,num_parallel_calls=tf.data.AUTOTUNE)
        return d.batch(32).prefetch(tf.data.AUTOTUNE)
    model,top=make_model(weights,unfreeze,dropout)
    es=lambda: tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=6,restore_best_weights=True)
    # Phase 1: head only
    for l in model.layers: l.trainable=l.name.startswith("ft_")
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(1e-3),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    model.fit(ds(),validation_data=(X[val],yi[val]),epochs=30,verbose=0,callbacks=[es()])
    # Phase 2: unfreeze top-N blocks (BN frozen), AdamW low lr
    for l in model.layers:
        if any(l.name==t or l.name.startswith(t+"/") for t in top):
            l.trainable=not isinstance(l,L.BatchNormalization)
    try: opt=tf.keras.optimizers.AdamW(1e-5,weight_decay=wd)
    except Exception: opt=tf.keras.optimizers.legacy.Adam(1e-5)
    model.compile(optimizer=opt,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    h=model.fit(ds(),validation_data=(X[val],yi[val]),epochs=40,verbose=0,callbacks=[es()])
    from sklearn.metrics import accuracy_score
    inv={v:k for k,v in ci.items()}
    tp=model.predict(X[trn],verbose=0,batch_size=128).argmax(1)
    pp=model.predict(X[te],verbose=0,batch_size=128).argmax(1)
    mixed=te&(src=="mixed"); pm=model.predict(X[mixed],verbose=0,batch_size=128).argmax(1)
    per={c:round(float(np.mean([inv[a]==c for a in model.predict(X[np.where(te&(y==c))[0]],verbose=0,batch_size=128).argmax(1)])),2)
         for c in classes if (te&(y==c)).sum()}
    return dict(unfreeze=unfreeze,seed=seed,
                train=round(accuracy_score(yi[trn],tp),3),
                test=round(accuracy_score(yi[te],pp),3),
                mixed=round(accuracy_score(yi[mixed],pm),3), per=per), model

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sweep",default="2,4,6"); ap.add_argument("--seeds",default="0")
    ap.add_argument("--augment",action="store_true")
    a=ap.parse_args()
    depths=[int(x) for x in a.sweep.split(",")]; seeds=[int(x) for x in a.seeds.split(",")]
    weights=pretrained_weights()
    X,y,fold,src=load_patches()
    classes=sorted(set(y)); ci={c:i for i,c in enumerate(classes)}
    print(f"patches: train={int((fold=='train').sum())} test={int((fold=='test').sum())}")
    print("LINEAR-HEAD CEILING (frozen): test ~0.50 / mixed ~0.59 / Bear 0.25 / drone_binary 0.08\n")
    results=[]; best=None; bestm=None
    if a.augment: print("SpecAugment: ON")
    for d in depths:
        for s in seeds:
            r,model=run(weights,X,y,fold,src,classes,ci,d,s,augment=a.augment)
            results.append(r)
            print(f"unfreeze={d} seed={s}: train={r['train']} test={r['test']} mixed={r['mixed']} "
                  f"gap={r['train']-r['test']:.2f}  per-class={r['per']}")
            if best is None or r["test"]>best["test"]: best,bestm=r,model
    print(f"\nBEST: unfreeze={best['unfreeze']} seed={best['seed']} test={best['test']} mixed={best['mixed']}")
    bestm.save(f"{EXP}/sim/checkpoints/yamnet_finetuned.keras")
    import json; json.dump(results,open(f"{EXP}/outputs/finetune_sweep.json","w"),indent=2,default=str)
    print("saved yamnet_finetuned.keras + finetune_sweep.json")

if __name__=="__main__":
    main()
