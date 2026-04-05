"""
Step 2: k-Means Clustering on Full 28-D Feature Space
- Run k-Means with various k values
- Use Elbow method and Silhouette analysis to find optimal k
- Record runtime, inertia, cluster sizes
- Save cluster labels for later comparison
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
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load preprocessed data
print("Loading preprocessed data...")
X_scaled = np.load(os.path.join(DATA_DIR, "X_scaled.npy"))
y_labels = np.load(os.path.join(DATA_DIR, "y_labels.npy"))
print(f"Data shape: {X_scaled.shape}")

# ============================================================
# 2a. Run k-Means for different k values
# ============================================================
k_values = [2, 3, 4, 5, 6, 7, 8]
results = []

for k in k_values:
    print(f"\nRunning k-Means with k={k}...")
    start = time.time()
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(X_scaled)
    runtime = time.time() - start

    inertia = kmeans.inertia_
    sil_score = silhouette_score(X_scaled, cluster_labels, sample_size=20000, random_state=42)
    db_score = davies_bouldin_score(X_scaled, cluster_labels)

    # Cluster sizes
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))

    # Compactness: average intra-cluster distance (inertia / n)
    compactness = inertia / len(X_scaled)

    # Separation: average distance between cluster centers
    centers = kmeans.cluster_centers_
    n_centers = len(centers)
    sep_sum = 0
    sep_count = 0
    for i in range(n_centers):
        for j in range(i + 1, n_centers):
            sep_sum += np.linalg.norm(centers[i] - centers[j])
            sep_count += 1
    separation = sep_sum / sep_count if sep_count > 0 else 0

    results.append({
        "k": k,
        "inertia": inertia,
        "silhouette": sil_score,
        "davies_bouldin": db_score,
        "compactness": compactness,
        "separation": separation,
        "runtime": runtime,
        "cluster_sizes": cluster_sizes,
        "n_iterations": kmeans.n_iter_,
    })

    print(f"  Runtime: {runtime:.2f}s | Iterations: {kmeans.n_iter_}")
    print(f"  Inertia: {inertia:.2f}")
    print(f"  Silhouette: {sil_score:.4f}")
    print(f"  Davies-Bouldin: {db_score:.4f}")
    print(f"  Compactness: {compactness:.4f} | Separation: {separation:.4f}")
    print(f"  Cluster sizes: {cluster_sizes}")

# Save k=2 labels (most natural for binary dataset)
kmeans_k2 = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_k2 = kmeans_k2.fit_predict(X_scaled)
np.save(os.path.join(DATA_DIR, "kmeans_raw_labels_k2.npy"), labels_k2)

# ============================================================
# 2b. Plots
# ============================================================

# Elbow plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ks = [r["k"] for r in results]

axes[0].plot(ks, [r["inertia"] for r in results], "bo-", linewidth=2)
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method — k-Means on Raw 28-D Data")
axes[0].grid(True)

axes[1].plot(ks, [r["silhouette"] for r in results], "rs-", linewidth=2)
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score vs. k")
axes[1].grid(True)

axes[2].plot(ks, [r["davies_bouldin"] for r in results], "g^-", linewidth=2)
axes[2].set_xlabel("Number of Clusters (k)")
axes[2].set_ylabel("Davies-Bouldin Index")
axes[2].set_title("Davies-Bouldin Index vs. k")
axes[2].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "step2_kmeans_raw_metrics.png"), dpi=150)
print(f"\nSaved metrics plot to output/step2_kmeans_raw_metrics.png")

# Runtime plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(ks, [r["runtime"] for r in results], color="steelblue")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Runtime (seconds)")
ax.set_title("k-Means Runtime on Raw 28-D Data")
ax.grid(axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "step2_kmeans_raw_runtime.png"), dpi=150)
print(f"Saved runtime plot to output/step2_kmeans_raw_runtime.png")

# Summary table
print("\n" + "=" * 80)
print(f"{'k':>3} | {'Inertia':>14} | {'Silhouette':>10} | {'Davies-Bouldin':>14} | {'Compact':>10} | {'Separation':>10} | {'Time(s)':>8}")
print("-" * 80)
for r in results:
    print(f"{r['k']:>3} | {r['inertia']:>14.2f} | {r['silhouette']:>10.4f} | {r['davies_bouldin']:>14.4f} | {r['compactness']:>10.4f} | {r['separation']:>10.4f} | {r['runtime']:>8.2f}")
print("=" * 80)

print("\nStep 2 complete!")
