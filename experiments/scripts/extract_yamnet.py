#!/usr/bin/env python3
"""Extract YAMNet scores + 1024-d embeddings for every clip in the data_prep inventory.

Uses the LOCAL SavedModel at yamnet/export_out/tf2 (full AudioSet YAMNet, 521 classes) —
no TF-Hub / network needed. Caches results so every plan.md leg reuses them:

  experiments/outputs/yamnet_features.npz   clip_id, label, split, emb(1024 mean-pooled), top1..top5
  experiments/outputs/baseline_topk.csv     per-clip top-5 AudioSet class names + scores
"""
import os, csv, json
import numpy as np
import soundfile as sf
import tensorflow as tf

ROOT = "/Users/abhinav/research"
MODEL_DIR = f"{ROOT}/yamnet/export_out/tf2"
CLASSMAP  = f"{MODEL_DIR}/assets/yamnet_class_map.csv"
INV       = f"{ROOT}/data_prep/inventory.csv"
OUT       = f"{ROOT}/experiments/outputs"

def load_class_map():
    names = []
    with open(CLASSMAP) as f:
        for r in csv.DictReader(f):
            names.append(r["display_name"])
    return names

def clip_path(row):
    lbl = row["label"]
    sub = "background" if lbl == "background" else f"positive/{lbl}"
    return f"{ROOT}/data_prep/clips/{sub}/{row['clip_id']}.wav"

# YAMNet front-end constants (must match training/data_loader.py and the ODAS C wrapper)
SR, WIN, HOP, FFT = 16000, 400, 160, 512
N_MEL, FMIN, FMAX, LOG_OFF = 64, 125.0, 7500.0, 0.001
PATCH, PATCH_HOP = 96, 48
_MEL = tf.signal.linear_to_mel_weight_matrix(N_MEL, FFT // 2 + 1, SR, FMIN, FMAX)  # (257,64)

def waveform_to_patches(w):
    """waveform float32 [-1,1] -> spectrogram patches (n_patches, 96, 64)."""
    stft = tf.signal.stft(w, frame_length=WIN, frame_step=HOP, fft_length=FFT, pad_end=True)
    mag = tf.abs(stft)                                  # (n_frames, 257)
    mel = tf.tensordot(mag, _MEL, 1)                    # (n_frames, 64)
    logmel = tf.math.log(mel + LOG_OFF)
    frames = tf.signal.frame(logmel, PATCH, PATCH_HOP, axis=0)  # (n_patches, 96, 64)
    return frames

def make_infer(model):
    """Return fn(waveform_1d_float32) -> (scores[patches,521], embeddings[patches,1024])."""
    sig = model.signatures["serving_default"]
    in_key = list(sig.structured_input_signature[1])[0]   # 'spectrogram_patches'
    def fn(w):
        patches = waveform_to_patches(tf.constant(w, tf.float32))
        if patches.shape[0] == 0:                          # clip shorter than one patch
            patches = tf.pad(patches if patches.shape[0] else tf.zeros((0,PATCH,N_MEL)),
                             [[0,1],[0,0],[0,0]])           # one zero patch fallback
        res = sig(**{in_key: patches})
        return res["predictions"].numpy(), res["embeddings"].numpy()
    return fn

def main():
    os.makedirs(OUT, exist_ok=True)
    names = load_class_map()
    print(f"class map: {len(names)} classes (e.g. {names[:3]} ...)")
    model = tf.saved_model.load(MODEL_DIR)
    infer = make_infer(model)

    rows = list(csv.DictReader(open(INV)))
    clip_ids, labels, splits, embs = [], [], [], []
    topk_rows = []
    for i, r in enumerate(rows):
        y, sr = sf.read(clip_path(r))
        if y.ndim > 1: y = y.mean(1)
        y = y.astype(np.float32)
        scores, emb = infer(y)                      # [frames,521], [frames,1024]
        mean_scores = scores.mean(0)
        mean_emb    = emb.mean(0)
        top5 = np.argsort(mean_scores)[::-1][:5]
        clip_ids.append(r["clip_id"]); labels.append(r["label"]); splits.append(r["split"])
        embs.append(mean_emb)
        topk_rows.append(dict(clip_id=r["clip_id"], label=r["label"], split=r["split"],
            **{f"top{j+1}": names[c] for j, c in enumerate(top5)},
            **{f"top{j+1}_score": round(float(mean_scores[c]), 4) for j, c in enumerate(top5)}))
        if (i + 1) % 50 == 0: print(f"  {i+1}/{len(rows)}")

    embs = np.stack(embs).astype(np.float32)
    np.savez(f"{OUT}/yamnet_features.npz",
             clip_id=np.array(clip_ids), label=np.array(labels),
             split=np.array(splits), emb=embs)
    cols = list(topk_rows[0].keys())
    with open(f"{OUT}/baseline_topk.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(topk_rows)
    print(f"\nsaved {embs.shape} embeddings + top-5 for {len(rows)} clips to {OUT}")

if __name__ == "__main__":
    main()
