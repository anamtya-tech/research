#!/usr/bin/env python3
"""Visualize the clip inventory (plan.md stage 1):
  figures/dist_snr_vs_ambient.png  - level-above-ambient distribution per class
  figures/scatter_dr_vs_snr.png    - impulsiveness vs level (class separability preview)
  figures/spectrograms_<class>.png - listen-candidate spectrograms (eyeball target presence)
"""
import os, json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
import librosa, librosa.display

OUT = "/Users/abhinav/research/data_prep"
FIG = f"{OUT}/figures"; os.makedirs(FIG, exist_ok=True)
rows = list(csv.DictReader(open(f"{OUT}/inventory.csv")))
summ = json.load(open(f"{OUT}/summary.json"))
classes = ["Elephant","Drone","Lion","Monkey","Frog","background"]
colors  = dict(zip(classes, plt.cm.tab10(np.linspace(0,1,len(classes)))))
def col(r,k): return float(r[k])

# 1) level-above-ambient distribution (strip + box) ------------------------
fig, ax = plt.subplots(figsize=(9,5))
for i,c in enumerate(classes):
    v = [col(r,"snr_vs_ambient_db") for r in rows if r["label"]==c]
    ax.boxplot(v, positions=[i], widths=0.5, patch_artist=True,
               boxprops=dict(facecolor=colors[c], alpha=0.4), showfliers=False)
    ax.scatter(np.full(len(v), i)+np.random.uniform(-0.12,0.12,len(v)), v,
               s=10, color=colors[c], alpha=0.6)
ax.axhline(0, ls="--", c="grey", lw=1); ax.set_xticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=20); ax.set_ylabel("dB above ambient floor")
ax.set_title("Clip level vs dataset ambient floor (separation from background)")
fig.tight_layout(); fig.savefig(f"{FIG}/dist_snr_vs_ambient.png", dpi=120); plt.close(fig)

# 2) impulsiveness vs level scatter ----------------------------------------
fig, ax = plt.subplots(figsize=(8,6))
for c in classes:
    g = [r for r in rows if r["label"]==c]
    ax.scatter([col(r,"snr_vs_ambient_db") for r in g], [col(r,"dr_db") for r in g],
               s=18, color=colors[c], alpha=0.7, label=c)
ax.set_xlabel("dB above ambient floor"); ax.set_ylabel("dynamic range dr_db (impulsiveness)")
ax.set_title("Feature preview: level vs impulsiveness"); ax.legend()
fig.tight_layout(); fig.savefig(f"{FIG}/scatter_dr_vs_snr.png", dpi=120); plt.close(fig)

# 3) spectrogram montages for listen candidates ----------------------------
def melspec(ax, path, title):
    y,_ = sf.read(path)
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=16000, n_mels=64), ref=np.max)
    librosa.display.specshow(S, sr=16000, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    ax.set_title(title, fontsize=8); ax.set_xlabel(""); ax.set_ylabel("")

def find(cid):
    r = next(r for r in rows if r["clip_id"]==cid)
    sub = f"positive/{r['label']}" if r["label"]!="background" else "background"
    return f"{OUT}/clips/{sub}/{cid}.wav", r

for c in classes:
    cand = summ["by_label"][c]["listen_best"] + summ["by_label"][c]["listen_weakest"]
    fig, axes = plt.subplots(2, 3, figsize=(11,5))
    for ax, cid in zip(axes.ravel(), cand):
        p, r = find(cid)
        melspec(ax, p, f"{cid}  Δ{r['snr_vs_ambient_db']}dB Q{r['quality']}")
    tag = "best (top) / weakest (bottom)"
    fig.suptitle(f"{c}: listen candidates — {tag}", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/spectrograms_{c}.png", dpi=110); plt.close(fig)

print("wrote figures to", FIG)
print(os.listdir(FIG))
