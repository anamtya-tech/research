#!/usr/bin/env python3
"""EXP-A1 (Track A) — build a GT clip dataset from a render's metadata via the simulator's
GTDatasetBuilder (3 s windows, 1.5 s hop, source-level split). ODAS-free.

  python expA1_build_dataset.py --meta <render>.json --dataset gt_a1_no_ambient
"""
import os, sys, json, csv, argparse, collections
import unittest.mock as mock

ROOT = "/Users/abhinav/research"
WORK = f"{ROOT}/experiments/sim"
GT_OUT = f"{WORK}/gt_datasets"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--dataset", default="gt_a1_no_ambient")
    args = ap.parse_args()

    meta = json.load(open(args.meta))
    os.makedirs(GT_OUT, exist_ok=True)
    sys.modules["streamlit"] = mock.MagicMock()
    sys.path.insert(0, f"{ROOT}/simulator")
    from gt_dataset_builder import GTDatasetBuilder
    b = GTDatasetBuilder(os.path.dirname(args.meta), GT_OUT)
    b.build_from_render(meta, args.dataset, mic_channel="random",
                        include_background=True, window_s=3.0, min_rms=0.01)

    # GTDatasetBuilder appends its own 'gt_datasets/' subdir under output_dir
    ds = f"{GT_OUT}/gt_datasets/{args.dataset}"
    man = f"{ds}/manifest.csv"
    rows = list(csv.DictReader(open(man)))
    by_lbl = collections.Counter(r["label"] for r in rows)
    by_fold = collections.Counter(r.get("fold","?") for r in rows)

    # Adapter: train_yamnet's data_loader expects labels.csv (filename,label,fold).
    # The builder emits manifest.csv (wav_path,label,fold) — convert. Keep only rows
    # whose audio file actually exists under audio/.
    # data_loader resolves audio_dir / filename, so filename = path under audio/
    # (clips are saved in per-label subdirs, e.g. "Elephant/src00_clip0000.wav").
    kept = 0
    with open(f"{ds}/labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "label", "fold"]); w.writeheader()
        for r in rows:
            ap = r["wav_path"]
            fn = ap.split("/audio/", 1)[-1] if "/audio/" in ap else os.path.basename(ap)
            if os.path.exists(ap):
                w.writerow(dict(filename=fn, label=r["label"], fold=r.get("fold", "train")))
                kept += 1

    print(f"\ndataset {args.dataset}: {len(rows)} manifest rows, "
          f"{kept} with audio present -> {ds}")
    print("  by label:", dict(by_lbl))
    print("  by fold :", dict(by_fold))
    print(f"  wrote labels.csv ({kept} rows) for train_yamnet")

if __name__ == "__main__":
    main()
