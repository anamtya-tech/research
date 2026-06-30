#!/usr/bin/env python3
"""EXP-A5 + B4 + C1 — background-confidence threshold sweep on the real-ambient holdout.

The 7-class (+same-env hard negatives) head suppresses a detection (calls it `background`) only
when P(background) >= tau, else assigns the top animal class. Sweep tau to trace FP/min vs recall
and find the operating point. Positives are the A5 real-ambient post-ODAS clips (deployment
distribution); negatives are A5-train same-env ghosts.
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT="/Users/abhinav/research"; EXP=f"{ROOT}/experiments"
sys.path.insert(0,f"{EXP}/scripts")
import expB4_hard_negatives as B4

def main():
    pos,a5_ghost,_,_ = B4.load_run(f"{EXP}/outputs/a5_analysis.pkl","exp_a5_real")
    Xp=B4.embed_many([p[0] for p in pos]); yp=[p[1] for p in pos]
    Xg=B4.embed_many(a5_ghost)
    animal=sorted(set(yp)); classes=animal+["background"]
    h,sc,ci=B4.train(np.vstack([Xp,Xg]),yp+["background"]*len(Xg),classes)
    bgi=ci["background"]; inv={v:k for k,v in ci.items()}

    h_pos,h_ghost,gtev,qmin=B4.load_run(f"{EXP}/outputs/a5_hold_analysis.pkl","exp_a5_holdout")
    Xhp=B4.embed_many([p[0] for p in h_pos]); keys=[p[2] for p in h_pos]; ylab=[p[1] for p in h_pos]
    Xhg=B4.embed_many(h_ghost)
    Pp=h.predict(sc.transform(Xhp),verbose=0); Pg=h.predict(sc.transform(Xhg),verbose=0)

    def animal_argmax(prob):  # best non-background class
        p=prob.copy(); p[bgi]=-1; return inv[p.argmax()]

    rows=[]
    for tau in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.01]:
        # positives: suppressed if P(bg)>=tau, else top-animal; event detected if any correct & not suppressed
        det=set()
        for k,yl,pr in zip(keys,ylab,Pp):
            if pr[bgi]>=tau: continue                 # suppressed -> miss
            if animal_argmax(pr)==yl: det.add(k)
        recall=len(det&gtev)/max(1,len(gtev))
        # ghosts: FP if NOT suppressed
        fp=sum(1 for pr in Pg if pr[bgi]<tau)
        rows.append((tau,fp/qmin,recall))
    print(f"holdout positives={len(Xhp)} ghosts={len(Xhg)} quiet_min={qmin:.2f}  GT events={len(gtev)}")
    print(f"{'tau':>5} {'FP/min':>8} {'recall':>7}")
    for t,f,r in rows: print(f"{t:>5.2f} {f:>8.1f} {r:>7.3f}")
    # operating points
    le2=[r for r in rows if r[1]<=2]; le10=[r for r in rows if r[1]<=10]
    best2 = max(le2,key=lambda r:r[2]) if le2 else None
    best10= max(le10,key=lambda r:r[2]) if le10 else None
    print("\nbest recall @ FP/min<=2 :", best2)
    print("best recall @ FP/min<=10:", best10)
    json.dump(dict(curve=[dict(tau=t,fp_per_min=round(f,2),recall=round(r,3)) for t,f,r in rows],
                   best_le2=best2,best_le10=best10),
              open(f"{EXP}/outputs/expA5_bgthresh.json","w"),indent=2)

    fig,ax=plt.subplots(figsize=(7,5))
    fps=[r[1] for r in rows]; rcs=[r[2] for r in rows]
    ax.plot(fps,rcs,"-o",color="tab:purple",zorder=3)
    for t,f,r in rows:
        if t in (0.0,0.5,0.8,0.9,1.01): ax.annotate(f"τ={t:.2f}",(f,r),fontsize=7,xytext=(4,4),textcoords="offset points")
    ax.axvline(2,ls="--",c="green",lw=1.2); ax.text(2.4,min(rcs),"FP/min ≤ 2",color="green",fontsize=8)
    ax.set_xlabel("FP/min (lower better)"); ax.set_ylabel("event recall (detect+correct class)")
    ax.set_title("EXP-A5+B4+C1: background-confidence threshold (real-ambient holdout)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{EXP}/outputs/figures/expA5_bgthresh.png",dpi=130)
    import shutil; shutil.copy(f"{EXP}/outputs/figures/expA5_bgthresh.png",f"{ROOT}/docs/figures/")
    print(f"\nwrote expA5_bgthresh.json and figures/expA5_bgthresh.png")

if __name__=="__main__":
    main()
