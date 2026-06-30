#!/usr/bin/env python3
"""
Data Prep (plan.md, stage 1) — build an annotated clip inventory.

Outputs:
  data_prep/clips/positive/<Class>/<class>_NNN.wav   16k mono 1.0s positives (50/class)
  data_prep/clips/background/bg_NNN.wav              16k mono 1.0s negatives (~200)
  data_prep/inventory.csv                            one row per clip + SNR/RMS/quality
  data_prep/summary.json                             per-class stats + listen candidates

Conventions match the prior curator pipeline: sample_rate=16000, target_duration=1.0s.
All positives standardized to 16 kHz mono. Multiple windows may be drawn from one
source file; every window from a given source stays in the same train/test split
(prevents leakage). Lion/Monkey/Frog use only the wild_animals tree because the
jungle_animals copies are byte-identical duplicates.
"""
import os, glob, json, hashlib, math
import numpy as np
import soundfile as sf
import librosa

BASE = "/Users/abhinav/research/backup"
OUT  = "/Users/abhinav/research/data_prep"
SR        = 16000
CLIP_LEN  = 1.0                      # seconds
CLIP_N    = int(SR * CLIP_LEN)       # 16000 samples
N_POS     = 50                       # clips per positive class
N_BG      = 200                      # background clips
AMB_CH    = 2                        # mic-array channel index (0,5 empty; 2 is mid)
SEED      = 1337
rng = np.random.default_rng(SEED)

POS_SOURCES = {
    "Elephant": [f"{BASE}/audio_cache/elephant_samples_new/*.wav"],          # exclude *_aug below
    "Drone":    [f"{BASE}/audio_cache/yes_drone_binary/*.wav"],
    "Lion":     [f"{BASE}/audio_cache/wild_animals/Animal-Soundprepros/Aslan/*.wav"],
    "Monkey":   [f"{BASE}/audio_cache/wild_animals/Animal-Soundprepros/Monkey/*.wav"],
    "Frog":     [f"{BASE}/audio_cache/wild_animals/Animal-Soundprepros/Frog/*.wav"],
}
# Ambient captures (RIFF/WAV despite .raw ext): 6ch S16 16k. Spread negatives across these.
AMBIENT = [
    f"{BASE}/audio_cache/ambient_captures/capture_20260408_035756_Eco10.raw",
    f"{BASE}/audio_cache/ambient_captures/capture_20260330_145737.raw",
    f"{BASE}/audio_cache/ambient_captures/capture_20260330_155534.raw",
    f"{BASE}/audio_cache/ambient_captures/capture_20260317_050000_Eco_Park.raw",
]
WARMUP_S = 12.0   # skip render warmup / capture start

# ---------------------------------------------------------------- helpers
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def dedup(paths):
    seen, out = set(), []
    for p in sorted(paths):
        d = md5(p)
        if d not in seen:
            seen.add(d); out.append(p)
    return out

def load_mono16k(path):
    y, sr = sf.read(path, always_2d=True)
    y = y.mean(axis=1)                                  # mix to mono
    if sr != SR:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR)
    return y.astype(np.float32)

def frame_rms(y, frame=400, hop=160):
    if len(y) < frame:
        return np.array([np.sqrt(np.mean(y**2)) + 1e-12])
    n = 1 + (len(y) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    fr = y[idx]
    return np.sqrt(np.mean(fr**2, axis=1)) + 1e-12

def best_windows(y, k):
    """Return up to k non-overlapping CLIP_N windows, most energetic first.
    Pads short signals to CLIP_N. Returns list of (start_sample, clip_array, padded)."""
    if len(y) <= CLIP_N:
        pad = np.zeros(CLIP_N, np.float32)
        off = (CLIP_N - len(y)) // 2
        pad[off:off + len(y)] = y
        return [(0, pad, True)]
    energy = np.convolve(y.astype(np.float64) ** 2, np.ones(CLIP_N), mode="valid")  # len = N-CLIP_N+1
    taken, out = np.zeros(len(energy), bool), []
    for _ in range(k):
        e = np.where(taken, -1.0, energy)
        if e.max() < 0:
            break
        s = int(e.argmax())
        out.append((s, y[s:s + CLIP_N].copy(), False))
        lo, hi = max(0, s - CLIP_N + 1), min(len(taken), s + CLIP_N)
        taken[lo:hi] = True
    return out

def metrics(clip):
    """Per-clip levels + a within-clip dynamic-range proxy (dr_db).
    dr_db = P90/P10 of frame RMS: high for impulsive events, low for steady tones.
    True quality is assigned later vs the dataset ambient floor (snr_vs_ambient_db)."""
    rms = float(np.sqrt(np.mean(clip ** 2)) + 1e-12)
    peak = float(np.max(np.abs(clip)) + 1e-12)
    fr = frame_rms(clip)
    sig   = float(np.percentile(fr, 90))
    noise = float(np.percentile(fr, 10))
    noise = max(noise, sig * 0.02, 1e-6)                # clamp: kills padded-silence blowups
    dr_db = min(20.0 * math.log10(sig / noise), 60.0)   # cap at 60 dB
    active = float(np.mean(fr > noise * 3.162))         # frames > floor+10 dB
    rms_db, peak_db = 20*math.log10(rms), 20*math.log10(peak)
    clip_frac = float(np.mean(np.abs(clip) > 0.99))      # truly railed samples
    notes = []
    if clip_frac > 0.01: notes.append("clipping")        # >1% railed = real distortion
    if rms_db < -45:     notes.append("near-silent")
    return dict(rms_db=round(rms_db,2), peak_db=round(peak_db,2), dr_db=round(dr_db,2),
                active_frac=round(active,3), notes=";".join(notes))

def split_sources(srcs):
    """80/20 source-level split (so multi-window clips never straddle splits)."""
    s = list(srcs); rng.shuffle(s)
    n_test = max(1, round(len(s) * 0.2))
    test = set(s[:n_test])
    return {p: ("test" if p in test else "train") for p in s}

# ---------------------------------------------------------------- main
def main():
    rows = []

    # ---- positives ----
    for cls, patterns in POS_SOURCES.items():
        files = []
        for pat in patterns:
            files += glob.glob(pat)
        if cls == "Elephant":
            files = [f for f in files if "_aug" not in os.path.basename(f).lower()]
        files = dedup(files)
        split_map = split_sources(files)
        outdir = f"{OUT}/clips/positive/{cls}"
        os.makedirs(outdir, exist_ok=True)

        # round-robin windows across files until N_POS reached
        per_file = {f: best_windows(load_mono16k(f), k=8) for f in files}
        order = list(files); produced = 0; ptr = {f: 0 for f in files}
        while produced < N_POS and any(ptr[f] < len(per_file[f]) for f in files):
            for f in order:
                if produced >= N_POS: break
                if ptr[f] >= len(per_file[f]): continue
                s, clip, padded = per_file[f][ptr[f]]; ptr[f] += 1
                cid = f"{cls.lower()}_{produced:03d}"
                sf.write(f"{outdir}/{cid}.wav", clip, SR, subtype="PCM_16")
                m = metrics(clip)
                note = m["notes"] + (";padded" if padded else "")
                rows.append(dict(clip_id=cid, label=cls, split=split_map[f],
                    source_file=os.path.relpath(f, BASE), source_start_sec=round(s/SR,3),
                    duration_sec=CLIP_LEN, sr=SR, channel="mono",
                    rms_db=m["rms_db"], peak_db=m["peak_db"], dr_db=m["dr_db"],
                    active_frac=m["active_frac"], notes=note.strip(";")))
                produced += 1
        print(f"{cls:9s}: {produced} clips from {len(files)} unique sources "
              f"(train/test sources: {sum(v=='train' for v in split_map.values())}/"
              f"{sum(v=='test' for v in split_map.values())})")

    # ---- background / negatives ----
    os.makedirs(f"{OUT}/clips/background", exist_ok=True)
    avail = [f for f in AMBIENT if os.path.exists(f)]
    per = [N_BG // len(avail)] * len(avail)
    for i in range(N_BG - sum(per)): per[i] += 1
    bg_i = 0
    N_AMB_CH = 6                                     # mic-array .raw layout
    HDR = 44                                          # WAV header on the .raw files
    for f, count in zip(avail, per):
        # .raw are RIFF/WAV but libsndfile won't auto-detect via extension;
        # memmap past the 44-byte header, int16 LE, 6-ch interleaved.
        mm = np.memmap(f, dtype="<i2", mode="r", offset=HDR)
        total = len(mm) // N_AMB_CH
        ch = min(AMB_CH, N_AMB_CH - 1)
        start0 = int(WARMUP_S * SR)
        usable = total - start0 - SR        # leave 1s tail
        if usable <= CLIP_N: continue
        starts = (start0 + np.linspace(0, usable - CLIP_N, count)).astype(int)
        y = (mm.reshape(-1, N_AMB_CH)[:, ch].astype(np.float32)) / 32768.0
        for st in starts:
            clip = y[st:st + CLIP_N].copy()
            cid = f"bg_{bg_i:03d}"
            sf.write(f"{OUT}/clips/background/{cid}.wav", clip, SR, subtype="PCM_16")
            m = metrics(clip)
            split = "test" if (bg_i % 5 == 0) else "train"
            rows.append(dict(clip_id=cid, label="background", split=split,
                source_file=os.path.relpath(f, BASE), source_start_sec=round(st/SR,3),
                duration_sec=CLIP_LEN, sr=SR, channel=f"ch{ch}",
                rms_db=m["rms_db"], peak_db=m["peak_db"], dr_db=m["dr_db"],
                active_frac=m["active_frac"], notes=m["notes"]))
            bg_i += 1
    print(f"background: {bg_i} clips from {len(avail)} ambient captures (channel {AMB_CH})")

    # ---- level vs dataset ambient floor + quality (robust for steady sounds) ----
    bg_med_rms = float(np.median([r["rms_db"] for r in rows if r["label"] == "background"]))
    for r in rows:
        r["snr_vs_ambient_db"] = round(r["rms_db"] - bg_med_rms, 2)
        n = r["notes"]
        if "near-silent" in n:                 r["quality"] = 1
        elif r["snr_vs_ambient_db"] >= 25:     r["quality"] = 3
        elif r["snr_vs_ambient_db"] >= 12:     r["quality"] = 2
        else:                                  r["quality"] = 1
        if "clipping" in n and r["quality"] == 3: r["quality"] = 2   # loud but distorted
    print(f"\nambient floor (median bg RMS) = {bg_med_rms:.1f} dB; "
          f"snr_vs_ambient_db = clip_rms_db - ambient_floor")

    # ---- write inventory ----
    import csv
    cols = ["clip_id","label","split","source_file","source_start_sec","duration_sec","sr",
            "channel","rms_db","peak_db","dr_db","snr_vs_ambient_db","active_frac","quality","notes"]
    with open(f"{OUT}/inventory.csv","w",newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)

    # ---- summary + listen candidates ----
    summary = {"total_clips": len(rows), "by_label": {}}
    for lbl in sorted({r["label"] for r in rows}):
        g = [r for r in rows if r["label"] == lbl]
        snrs = [r["snr_vs_ambient_db"] for r in g]
        qd = {q: sum(r["quality"]==q for r in g) for q in (1,2,3)}
        gs = sorted(g, key=lambda r: r["snr_vs_ambient_db"])
        summary["by_label"][lbl] = dict(
            n=len(g), train=sum(r["split"]=="train" for r in g), test=sum(r["split"]=="test" for r in g),
            snr_vs_ambient_db_min=round(min(snrs),2), snr_vs_ambient_db_mean=round(float(np.mean(snrs)),2),
            snr_vs_ambient_db_max=round(max(snrs),2),
            dr_db_mean=round(float(np.mean([r["dr_db"] for r in g])),2),
            mean_active_frac=round(float(np.mean([r["active_frac"] for r in g])),3),
            quality_dist=qd,
            listen_best=[r["clip_id"] for r in gs[-3:]],
            listen_weakest=[r["clip_id"] for r in gs[:3]])
    with open(f"{OUT}/summary.json","w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
