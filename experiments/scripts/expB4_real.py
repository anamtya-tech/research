#!/usr/bin/env python3
"""EXP-B4 at scale — hard negatives harvested from a REAL ambient capture run through ODAS
(ambient-only → every .bin is a realistic background spectrum). Compare to the no-hard-neg
baseline and the small-structural-ghost version, on the same holdout.

  python expB4_real.py --neg-dir experiments/odas/logs_ambient --cap 1500
"""
import os, sys, glob, json, argparse, collections
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT="/Users/abhinav/research"; EXP=f"{ROOT}/experiments"
sys.path.insert(0, f"{EXP}/scripts")
# reuse the validated embedding + train + holdout-eval from the first B4 script
import expB4_hard_negatives as B4
import numpy as np

def load_bg_dir(d, cap, seed=0):
    files=sorted(glob.glob(f"{d}/*.bin"))
    rng=np.random.default_rng(seed); rng.shuffle(files)
    X=[]; used=0; skipped=0
    for f in files:
        a=np.fromfile(f,np.float32)
        if a.size%257: continue
        sp=a.reshape(-1,257)
        if sp.sum() <= 0 or (sp.any(1).sum() < 8):   # skip empty/near-empty (null) patches
            skipped+=1; continue
        e=B4.embed_bin(f)
        if e is None: continue
        X.append(e); used+=1
        if used>=cap: break
    return np.array(X), used, skipped, len(files)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--neg-dir", required=True)
    ap.add_argument("--cap", type=int, default=1500)
    args=ap.parse_args()

    # positives (600s GT matches) + holdout (300s), via the existing loader
    tr_pos,_,_,_ = B4.load_run(f"{EXP}/outputs/a1_analysis.pkl","exp_a1_no_ambient")
    Xp=B4.embed_many([p[0] for p in tr_pos]); yp=[p[1] for p in tr_pos]
    Xneg,used,skipped,total = load_bg_dir(args.neg_dir, args.cap)
    print(f"positives: {len(yp)} ({dict(collections.Counter(yp))})")
    print(f"real-ambient hard negatives: {used} used / {total} .bin ({skipped} empty/null skipped)")

    animal=sorted(set(yp)); classes7=animal+["background"]
    h6,sc6,ci6 = B4.train(Xp,yp,animal)
    X7=np.vstack([Xp,Xneg]); y7=yp+["background"]*len(Xneg)
    h7,sc7,ci7 = B4.train(X7,y7,classes7)

    h_pos,h_ghost,gtev,qmin = B4.load_run(f"{EXP}/outputs/a1_hold_analysis.pkl","exp_a1_holdout")
    B4.Xhp=B4.embed_many([p[0] for p in h_pos]); B4.keys=[p[2] for p in h_pos]; B4.ylab=[p[1] for p in h_pos]
    B4.Xhg=B4.embed_many(h_ghost); B4.gtev=gtev; B4.qmin=qmin

    def evalm(h,sc,ci,tag):
        inv={v:k for k,v in ci.items()}
        pp=h.predict(sc.transform(B4.Xhp),verbose=0).argmax(1); pred=[inv[i] for i in pp]
        det={k for k,yl,pr in zip(B4.keys,B4.ylab,pred) if pr==yl}
        recall=len(det & gtev)/len(gtev); clip=float(np.mean([pr==yl for yl,pr in zip(B4.ylab,pred)]))
        gp=h.predict(sc.transform(B4.Xhg),verbose=0).argmax(1) if len(B4.Xhg) else []
        gpred=[inv[i] for i in gp]
        fp=sum(1 for p in gpred if p!="background"); bg=sum(1 for p in gpred if p=="background")
        print(f"  [{tag}] recall={recall:.3f} clip_acc={clip:.3f} | ghosts {fp} fired/{len(B4.Xhg)} ({bg}->bg) | FP/min={fp/qmin:.1f}")
        return dict(model=tag,recall=round(recall,3),clip_acc=round(clip,3),
                    ghost_total=len(B4.Xhg),ghost_fired=fp,fp_per_min=round(fp/qmin,2))

    print("\n=== EXP-B4 (REAL-ambient hard negatives) — holdout ===")
    r6=evalm(h6,sc6,ci6,"6-class (no hard-neg)")
    r7=evalm(h7,sc7,ci7,f"7-class (+{used} real-ambient neg)")
    json.dump(dict(baseline=r6,real_hardneg=r7,n_negatives=used),
              open(f"{EXP}/outputs/expB4_real_results.json","w"),indent=2)

    fig,ax=plt.subplots(figsize=(7,5))
    # include the small-ghost B4 point if available
    pts=[("6-class baseline",r6["fp_per_min"],r6["recall"],"tab:gray")]
    try:
        prev=json.load(open(f"{EXP}/outputs/expB4_results.json"))["hard_neg"]
        pts.append(("+80 structural neg",prev["fp_per_min"],prev["recall"],"tab:orange"))
    except Exception: pass
    pts.append((f"+{used} real-ambient neg",r7["fp_per_min"],r7["recall"],"tab:green"))
    for lab,x,y,c in pts:
        ax.scatter([x],[y],s=120,c=c,zorder=3,label=lab)
    ax.axvline(2,ls="--",c="green",lw=1.2); ax.text(2.2,min(p[2] for p in pts),"FP/min ≤ 2",color="green",fontsize=8)
    ax.set_xlabel("FP/min (lower better)"); ax.set_ylabel("event recall (higher better)")
    ax.set_title("EXP-B4: hard negatives at scale (real ambient) — holdout"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{EXP}/outputs/figures/expB4_real.png",dpi=130)
    import shutil; shutil.copy(f"{EXP}/outputs/figures/expB4_real.png",f"{ROOT}/docs/figures/")
    print(f"\nwrote expB4_real_results.json and figures/expB4_real.png")

if __name__=="__main__":
    main()
