"""
Step 4: k-Means on PCA-Reduced Datasets (2, 5, 10 components)
- Run k-Means on each PCA-reduced dataset
- Compare clustering quality metrics with raw 28-D results
- Save cluster labels and visualizations
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import time
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Load data
X_scaled = np.load(os.path.join(DATA_DIR, "X_scaled.npy"))
y_labels = np.load(os.path.join(DATA_DIR, "y_labels.npy"))
X_pca_2 = np.load(os.path.join(DATA_DIR, "X_pca_2.npy"))
X_pca_5 = np.load(os.path.join(DATA_DIR, "X_pca_5.npy"))
X_pca_10 = np.load(os.path.join(DATA_DIR, "X_pca_10.npy"))

datasets = {
    "Raw (28-D)": X_scaled,
    "PCA 2-D": X_pca_2,
    "PCA 5-D": X_pca_5,
    "PCA 10-D": X_pca_10,
}

k_values = [2, 3, 4, 5, 6, 7, 8]
all_results = {}

for name, X in datasets.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {name} — shape {X.shape}")
    print(f"{'='*60}")
    results = []
    for k in k_values:
        start = time.time()
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(X)
        runtime = time.time() - start

        sil = silhouette_score(X, labels, sample_size=20000, random_state=42)
        db = davies_bouldin_score(X, labels)
        compactness = kmeans.inertia_ / len(X)

        centers = kmeans.cluster_centers_
        sep_sum = sum(np.linalg.norm(centers[i] - centers[j])
                      for i in range(len(centers)) for j in range(i+1, len(centers)))
        sep_count = len(centers) * (len(centers) - 1) / 2
        separation = sep_sum / sep_count if sep_count > 0 else 0

        results.append({
            "k": k, "silhouette": sil, "davies_bouldin": db,
            "inertia": kmeans.inertia_, "compactness": compactness,
            "separation": separation, "runtime": runtime,
            "n_iter": kmeans.n_iter_,
        })
        print(f"  k={k}: Sil={sil:.4f} DB={db:.4f} Time={runtime:.2f}s Iter={kmeans.n_iter_}")

    all_results[name] = results

# Save k=2 cluster labels for each PCA dataset
for n_comp, X in [(2, X_pca_2), (5, X_pca_5), (10, X_pca_10)]:
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    lbl = km.fit_predict(X)
    np.save(os.path.join(DATA_DIR, f"kmeans_pca{n_comp}_labels_k2.npy"), lbl)

# ============================================================
# Comparison Plots
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Silhouette
for name, results in all_results.items():
    axes[0, 0].plot([r["k"] for r in results], [r["silhouette"] for r in results], "o-", label=name)
axes[0, 0].set_xlabel("k")
axes[0, 0].set_ylabel("Silhouette Score")
axes[0, 0].set_title("Silhouette Score Comparison")
axes[0, 0].legend()
axes[0, 0].grid(True)

# Davies-Bouldin
for name, results in all_results.items():
    axes[0, 1].plot([r["k"] for r in results], [r["davies_bouldin"] for r in results], "o-", label=name)
axes[0, 1].set_xlabel("k")
axes[0, 1].set_ylabel("Davies-Bouldin Index")
axes[0, 1].set_title("Davies-Bouldin Index Comparison")
axes[0, 1].legend()
axes[0, 1].grid(True)

# Runtime
for name, results in all_results.items():
    axes[1, 0].plot([r["k"] for r in results], [r["runtime"] for r in results], "o-", label=name)
axes[1, 0].set_xlabel("k")
axes[1, 0].set_ylabel("Runtime (s)")
axes[1, 0].set_title("k-Means Runtime Comparison")
axes[1, 0].legend()
axes[1, 0].grid(True)

# Convergence iterations
for name, results in all_results.items():
    axes[1, 1].plot([r["k"] for r in results], [r["n_iter"] for r in results], "o-", label=name)
axes[1, 1].set_xlabel("k")
axes[1, 1].set_ylabel("Iterations to Converge")
axes[1, 1].set_title("k-Means Convergence Speed")
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "step4_kmeans_comparison.png"), dpi=150)
print(f"\nSaved comparison plot to output/step4_kmeans_comparison.png")

# ============================================================
# 2D PCA scatter plots colored by cluster (k=2)
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
sample_idx = np.random.RandomState(42).choice(len(X_pca_2), size=10000, replace=False)

from sklearn.decomposition import PCA as PCATransform

titles = ["Raw 28-D (projected)", "PCA 2-D", "PCA 5-D", "PCA 10-D"]
label_files = [
    os.path.join(DATA_DIR, "kmeans_raw_labels_k2.npy"),
    os.path.join(DATA_DIR, "kmeans_pca2_labels_k2.npy"),
    os.path.join(DATA_DIR, "kmeans_pca5_labels_k2.npy"),
    os.path.join(DATA_DIR, "kmeans_pca10_labels_k2.npy"),
]

for i, (title, lf) in enumerate(zip(titles, label_files)):
    lbl = np.load(lf)
    axes[i].scatter(X_pca_2[sample_idx, 0], X_pca_2[sample_idx, 1],
                    c=lbl[sample_idx], cmap="Set1", alpha=0.3, s=5)
    axes[i].set_xlabel("PC1")
    axes[i].set_ylabel("PC2")
    axes[i].set_title(f"k=2 Clusters — {title}")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "step4_cluster_scatter_comparison.png"), dpi=150)
print(f"Saved cluster scatter comparison to output/step4_cluster_scatter_comparison.png")

# ============================================================
# Summary Table
# ============================================================
print(f"\n{'='*90}")
print(f"Summary at k=2 (most natural for binary dataset)")
print(f"{'='*90}")
print(f"{'Dataset':<15} | {'Silhouette':>10} | {'Davies-Bouldin':>14} | {'Compactness':>11} | {'Separation':>10} | {'Time(s)':>8} | {'Iters':>5}")
print("-" * 90)
for name, results in all_results.items():
    r = results[0]  # k=2
    print(f"{name:<15} | {r['silhouette']:>10.4f} | {r['davies_bouldin']:>14.4f} | {r['compactness']:>11.4f} | {r['separation']:>10.4f} | {r['runtime']:>8.2f} | {r['n_iter']:>5}")
print("=" * 90)

print("\nStep 4 complete!")
