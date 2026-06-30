#!/usr/bin/env python3
"""EXP-A1 Track B — analyze ODAS output: match detections to scene GT, compute deployment
metrics (FP/min, event P/R), and pickle the matched results for the post-ODAS dataset builder.

Drives the real simulator analyzer headlessly (streamlit stubbed). Rewrites the container-side
.bin paths (/exp/...) to host paths so the curator can read the spectra sidecars.

  python expA1_analyze.py --logs <classifier_log_dir> --scene <scene.json> --tag exp_a1
"""
import os, sys, json, glob, argparse, pickle
import unittest.mock as mock

ROOT = "/Users/abhinav/research"
EXP  = f"{ROOT}/experiments"
OUT  = f"{EXP}/outputs"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)        # host dir with sst_session_live + .bin
    ap.add_argument("--scene", required=True)        # scene json (GT)
    ap.add_argument("--tag", default="exp_a1")
    ap.add_argument("--warmup", type=float, default=10.0)
    ap.add_argument("--angle", type=float, default=15.0)
    args = ap.parse_args()

    sess = sorted(glob.glob(f"{args.logs}/sst_session_live.json_*.json"))
    sess = [s for s in sess if "fingerprint" not in s]
    if not sess:
        sys.exit(f"no session json in {args.logs}")
    sess = sess[-1]

    # Rewrite container paths (/exp/...) -> host paths so spectra_file resolves.
    txt = open(sess).read().replace("/exp/", f"{EXP}/")
    fixed = sess.replace(".json", "_hostpaths.json")
    open(fixed, "w").write(txt)

    sys.modules["streamlit"] = mock.MagicMock()
    sys.path.insert(0, f"{ROOT}/simulator")
    from analyzer import ResultAnalyzer

    ra = ResultAnalyzer(output_dir=f"{EXP}/sim", odas_logs_dir=args.logs)
    run_id = f"{args.tag}_run"
    run_data = dict(run_id=run_id, render_id=args.tag, scene_name=args.tag,
                    scene_file=args.scene, session_live_file=fixed,
                    warmup_seconds=args.warmup,
                    scene_metadata={"warmup_seconds": args.warmup})

    results = ra._analyze_run(run_data, angle_threshold=args.angle, save_unmatched=True)
    if results is None:
        sys.exit("analyzer returned None (see stubbed st.error above)")
    results = ra._apply_yamnet_classifications(results, label_strategy="Ground truth only")

    summ = results.get("summary", {})
    print("=== analysis summary ===")
    for k in ("total_detections", "matched", "unmatched", "match_rate",
              "avg_angular_error", "unique_sources"):
        if k in summ: print(f"  {k}: {summ[k]}")

    # ---- transparent decision metrics (computed here, not the buggy curator fn) ----
    scene = json.load(open(args.scene))
    duration = float(scene["duration"])
    srcs = scene["directional_sources"]
    # quiet seconds = duration minus union of GT active intervals
    iv = sorted((s["start_time"], s["end_time"]) for s in srcs)
    covered, cs, ce = 0.0, None, None
    for a, b in iv:
        if cs is None: cs, ce = a, b
        elif a <= ce: ce = max(ce, b)
        else: covered += ce - cs; cs, ce = a, b
    if cs is not None: covered += ce - cs
    quiet_s = max(1e-6, duration - covered)

    M = results["matches"];
    gt = [m for m in M if m.get("match_type") == "ground_truth"]
    nongt = [m for m in M if m.get("match_type") != "ground_truth"]

    def det_ts(m):
        d = m.get("detection", {});
        return d.get("timestamp") if isinstance(d, dict) else None
    def in_quiet(t):
        if t is None: return False
        return not any(a <= t <= b for a, b in iv)

    # event recall: GT source instances with >=1 matched detection
    gt_events = {(s["label"], s["start_time"], s["end_time"]) for s in srcs}
    detected = {(m["label"], m["source"]["start_time"], m["source"]["end_time"])
                for m in gt if m.get("source")}
    event_recall = len(detected & gt_events) / max(1, len(gt_events))

    # FP/min: non-GT (unmatched) detections occurring in quiet periods
    fp_quiet = [m for m in nongt if in_quiet(det_ts(m))]
    fp_per_min = len(fp_quiet) / (quiet_s / 60.0)

    # deployed-model class correctness (firmware event_class_name vs GT label)
    import collections
    conf = collections.Counter()
    correct = 0
    for m in gt:
        pred = (m.get("detection") or {}).get("event_class_name")
        conf[(m["label"], pred)] += 1
        if pred and pred.lower() == m["label"].lower(): correct += 1
    deployed_class_acc = correct / max(1, len(gt))

    metrics = dict(
        scene_duration_s=duration, quiet_s=round(quiet_s, 1),
        total_detections=len(M), matched_gt=len(gt), unmatched=len(nongt),
        gt_events_total=len(gt_events), gt_events_detected=len(detected & gt_events),
        event_recall=round(event_recall, 3),
        fp_in_quiet=len(fp_quiet), fp_per_min=round(fp_per_min, 3),
        deployed_base_class_accuracy=round(deployed_class_acc, 3),
        confusion={f"{g}->{p}": n for (g, p), n in sorted(conf.items())},
    )
    print("\n=== EXP-A1 deployment metrics (no-ambient baseline) ===")
    for k in ("scene_duration_s","quiet_s","gt_events_total","gt_events_detected",
              "event_recall","unmatched","fp_in_quiet","fp_per_min","deployed_base_class_accuracy"):
        print(f"  {k}: {metrics[k]}")
    print("  base-YAMNet confusion (GT->firmware pred):")
    for k, n in metrics["confusion"].items(): print(f"    {k}: {n}")

    # ---- per-match table for the post-ODAS dataset builder (host-fixed .bin paths) ----
    rows = []
    for m in gt:
        d = m.get("detection") or {}
        sf = d.get("spectra_file")
        if sf: sf = sf.replace("/exp/", f"{EXP}/")
        rows.append(dict(label=m["label"], spectra_file=sf,
                         spectral_count=d.get("spectral_count"),
                         angular_error=m.get("angular_error"),
                         src_start=m["source"]["start_time"], src_end=m["source"]["end_time"]))
    os.makedirs(OUT, exist_ok=True)
    json.dump(rows, open(f"{OUT}/{args.tag}_matches.json", "w"), indent=2, default=str)
    json.dump(metrics, open(f"{OUT}/{args.tag}_deploy_metrics.json", "w"), indent=2, default=str)
    pickle.dump(results, open(f"{OUT}/{args.tag}_analysis.pkl", "wb"))
    print(f"\nwrote {args.tag}_deploy_metrics.json, {args.tag}_matches.json ({len(rows)} gt matches)")

if __name__ == "__main__":
    main()
