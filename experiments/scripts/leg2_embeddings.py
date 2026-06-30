#!/usr/bin/env python3
"""plan.md leg 2 — Embedding visualization. Do target sounds cluster in YAMNet space?

Reads experiments/outputs/yamnet_features.npz (1024-d mean-pooled embeddings).
Produces UMAP + t-SNE 2D scatter colored by class, and a quantitative separation metric
(k-NN label purity + silhouette) so the read isn't purely visual.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

OUT = "/Users/abhinav/research/experiments/outputs"
FIG = f"{OUT}/figures"; os.makedirs(FIG, exist_ok=True)
ORDER = ["Elephant","Drone","Lion","Monkey","Frog","background"]

def knn_purity(X, y, k=10):
    nn = NearestNeighbors(n_neighbors=k+1).fit(X)
    _, idx = nn.kneighbors(X)
    same = [(y[i[1:]] == y[c]).mean() for c, i in enumerate(idx)]
    return float(np.mean(same))

def main():
    d = np.load(f"{OUT}/yamnet_features.npz", allow_pickle=True)
    X, y = d["emb"], d["label"]
    Xs = StandardScaler().fit_transform(X)
    colors = dict(zip(ORDER, plt.cm.tab10(np.linspace(0,1,len(ORDER)))))

    # quantitative separation
    purity = knn_purity(Xs, y, k=10)
    sil = silhouette_score(Xs, y)
    print(f"k=10 NN label purity: {purity:.3f}   silhouette: {sil:.3f}")

    embeds = {}
    try:
        import umap
        embeds["UMAP"] = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(Xs)
    except Exception as e:
        print("UMAP unavailable:", e)
    embeds["t-SNE"] = TSNE(n_components=2, perplexity=30, random_state=42,
                           init="pca").fit_transform(Xs)

    fig, axes = plt.subplots(1, len(embeds), figsize=(7*len(embeds), 6), squeeze=False)
    for ax, (name, Z) in zip(axes[0], embeds.items()):
        for c in ORDER:
            m = y == c
            ax.scatter(Z[m,0], Z[m,1], s=16, alpha=0.7, color=colors[c],
                       label=c, edgecolors="none")
        ax.set_title(f"{name}  (kNN purity {purity:.2f}, silhouette {sil:.2f})")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0][0].legend(markerscale=1.5, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/embeddings_2d.png", dpi=130); plt.close(fig)

    import json
    json.dump(dict(knn_purity_k10=round(purity,3), silhouette=round(sil,3),
                   methods=list(embeds)), open(f"{OUT}/embedding_separation.json","w"), indent=2)
    print(f"wrote {FIG}/embeddings_2d.png")

if __name__ == "__main__":
    main()
