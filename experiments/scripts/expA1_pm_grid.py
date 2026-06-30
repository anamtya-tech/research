#!/usr/bin/env python3
"""Build the probMin × min_event_votes grid (FP/min vs event recall) on the holdout.
probMin needs separate ODAS runs (analyzed to *_analysis.pkl); votes is applied offline.
The free null/zero-activity filter is applied throughout (honest 'deployed' counting).
"""
import pickle, json
EXP="/Users/abhinav/research/experiments"
scene=json.load(open(f"{EXP}/sim/scenes/exp_a1_holdout.json"))
iv=sorted((s["start_time"],s["end_time"]) for s in scene["directional_sources"])
dur=scene["duration"]
cov=0;cs=ce=None
for a,b in iv:
    if cs is None:cs,ce=a,b
    elif a<=ce:ce=max(ce,b)
    else:cov+=ce-cs;cs,ce=a,b
if cs is not None:cov+=ce-cs
quiet_min=max(1e-6,dur-cov)/60
gt_ev={(s["label"],s["start_time"],s["end_time"]) for s in scene["directional_sources"]}
def inq(t): return t is not None and not any(a<=t<=b for a,b in iv)
def is_null(d):
    return (d.get("x")==0 and d.get("y")==0 and d.get("z")==0) or (d.get("activity") is not None and d.get("activity")<=0.0)
def v(m): return (m.get("detection") or {}).get("event_votes") or 0

RUNS=[("0.5",f"{EXP}/outputs/a1_hold_analysis.pkl"),
      ("0.7",f"{EXP}/outputs/a1_pm07_analysis.pkl"),
      ("0.8",f"{EXP}/outputs/a1_pm08_analysis.pkl")]
print(f"holdout, quiet={quiet_min*60:.0f}s, GT events={len(gt_ev)}  (FP/min after null-activity filter)")
print(f"{'probMin':8} {'votes>=1 FP/min':>16} {'recall':>7} | {'votes>=2 FP/min':>16} {'recall':>7}")
for pm,pkl in RUNS:
    try: r=pickle.load(open(pkl,"rb"))
    except FileNotFoundError:
        print(f"{pm:8} (run not analyzed yet: {pkl})"); continue
    M=r["matches"]; gt=[m for m in M if m.get("match_type")=="ground_truth"]
    ng=[m for m in M if m.get("match_type")!="ground_truth"]
    out=[]
    for thr in (1,2):
        fp=[m for m in ng if inq((m.get('detection') or {}).get('timestamp'))
            and v(m)>=thr and not is_null(m.get('detection') or {})]
        det={(m['label'],m['source']['start_time'],m['source']['end_time']) for m in gt
             if v(m)>=thr and m.get('source')}
        out.append((len(fp)/quiet_min, len(det & gt_ev)/len(gt_ev)))
    print(f"{pm:8} {out[0][0]:>16.1f} {out[0][1]:>7.3f} | {out[1][0]:>16.1f} {out[1][1]:>7.3f}")
