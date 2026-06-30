#!/usr/bin/env python3
"""EXP-A5 + B4 on a DEPLOYMENT-REALISTIC testbed: scenes carry real-capture ambient, so the
holdout's false positives are real-ambient-driven (not silent-render structural artifacts).

Tests whether real-ambient hard negatives now transfer:
  - 6-class                       (no background)                          baseline
  - 7-class + A5-train ghosts     (Mar-30 ambient, SAME env as holdout)    matched
  - 7-class + Eco_Park ghosts     (different real environment)             cross-env

All evaluated on the A5 holdout (Mar-30 ambient). FP/min = holdout quiet-ghosts predicted
non-background; recall = GT events classified as the correct animal.
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT="/Users/abhinav/research"; EXP=f"{ROOT}/experiments"
sys.path.insert(0,f"{EXP}/scripts")
import expB4_hard_negatives as B4
from expB4_real import load_bg_dir

def main():
    # positives + same-env negatives from A5 train (Mar-30 ambient)
    pos,a5_ghost,_,_ = B4.load_run(f"{EXP}/outputs/a5_analysis.pkl","exp_a5_real")
    Xp=B4.embed_many([p[0] for p in pos]); yp=[p[1] for p in pos]
    Xg_same=B4.embed_many(a5_ghost)                                  # Mar-30 ambient ghosts
    Xg_cross,used,_,_=load_bg_dir(f"{EXP}/odas/logs_ambient",1500)   # Eco_Park ghosts
    print(f"positives={len(yp)}  A5-same-env negs={len(Xg_same)}  Eco_Park cross-env negs={used}")

    # holdout (A5, Mar-30 ambient, different segment)
    h_pos,h_ghost,gtev,qmin=B4.load_run(f"{EXP}/outputs/a5_hold_analysis.pkl","exp_a5_holdout")
    Xhp=B4.embed_many([p[0] for p in h_pos]); keys=[p[2] for p in h_pos]; ylab=[p[1] for p in h_pos]
    Xhg=B4.embed_many(h_ghost)
    print(f"holdout positives={len(Xhp)}  holdout real-ambient ghosts={len(Xhg)}  quiet_min={qmin:.2f}")

    animal=sorted(set(yp)); c7=animal+["background"]
    def evalm(h,sc,ci,tag):
        inv={v:k for k,v in ci.items()}
        pred=[inv[i] for i in h.predict(sc.transform(Xhp),verbose=0).argmax(1)]
        det={k for k,yl,pr in zip(keys,ylab,pred) if pr==yl}
        recall=len(det&gtev)/max(1,len(gtev)); clip=float(np.mean([pr==yl for yl,pr in zip(ylab,pred)]))
        gp=[inv[i] for i in h.predict(sc.transform(Xhg),verbose=0).argmax(1)] if len(Xhg) else []
        fp=sum(1 for p in gp if p!="background"); bg=sum(1 for p in gp if p=="background")
        print(f"  [{tag}] recall={recall:.3f} clip={clip:.3f} | ghosts {fp} fired/{len(Xhg)} ({bg}->bg) | FP/min={fp/qmin:.1f}")
        return dict(model=tag,recall=round(recall,3),clip_acc=round(clip,3),fp_per_min=round(fp/qmin,2),
                    ghost_total=len(Xhg),ghost_fired=fp)
    print("\n=== EXP-A5+B4 (real-ambient holdout) ===")
    res={}
    h,sc,ci=B4.train(Xp,yp,animal); res["baseline"]=evalm(h,sc,ci,"6-class baseline")
    h,sc,ci=B4.train(np.vstack([Xp,Xg_same]),yp+["background"]*len(Xg_same),c7); res["same_env"]=evalm(h,sc,ci,"+A5 same-env neg")
    h,sc,ci=B4.train(np.vstack([Xp,Xg_cross]),yp+["background"]*len(Xg_cross),c7); res["cross_env"]=evalm(h,sc,ci,"+Eco_Park cross-env neg")
    json.dump(res,open(f"{EXP}/outputs/expA5_b4_results.json","w"),indent=2)

    fig,ax=plt.subplots(figsize=(7,5))
    col={"baseline":"tab:gray","same_env":"tab:green","cross_env":"tab:blue"}
    for k,r in res.items():
        ax.scatter([r["fp_per_min"]],[r["recall"]],s=130,c=col[k],zorder=3,label=r["model"])
    ax.axvline(2,ls="--",c="green",lw=1.2); ax.text(2.2,min(r["recall"] for r in res.values()),"FP/min ≤ 2",color="green",fontsize=8)
    ax.set_xlabel("FP/min (lower better)"); ax.set_ylabel("event recall"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("EXP-A5+B4: real-ambient holdout — do real negatives now transfer?")
    fig.tight_layout(); fig.savefig(f"{EXP}/outputs/figures/expA5_b4.png",dpi=130)
    import shutil; shutil.copy(f"{EXP}/outputs/figures/expA5_b4.png",f"{ROOT}/docs/figures/")
    print(f"\nwrote expA5_b4_results.json and figures/expA5_b4.png")

if __name__=="__main__":
    main()
