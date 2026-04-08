import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid", context="talk")

#Load Data
X_2 = pd.read_csv("../outputs/X_pca_2.csv").values
labels = pd.read_csv("../labels/labels_PCA_2_k2.csv").values.flatten()

raw_results = pd.read_csv("../outputs/clustering_raw_results.csv")
pca_results = pd.read_csv("../outputs/clustering_pca_results.csv")

#1. PCA 2D Cluster Plot 
plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_2[:, 0], X_2[:, 1],
    c=labels,
    cmap='viridis',
    s=8,
    alpha=0.6
)

plt.title("k-Means Clusters (PCA 2D, k=2)", fontsize=16)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig("../figures/pca_2d_clusters_advanced.png")
plt.show()

#2. Elbow Plot (Raw vs PCA)
plt.figure(figsize=(10, 6))

sns.lineplot(data=raw_results, x="k", y="inertia", marker='o', label="Raw (28D)")

for dataset in ["PCA_2", "PCA_5", "PCA_10"]:
    subset = pca_results[pca_results["dataset"] == dataset]
    sns.lineplot(data=subset, x="k", y="inertia", marker='o', label=dataset)

plt.title("Elbow Method Comparison", fontsize=16)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.legend()
plt.tight_layout()
plt.savefig("../figures/elbow_comparison.png")
plt.show()

#3. Silhouette Score Comparison
plt.figure(figsize=(10, 6))

sns.lineplot(data=raw_results, x="k", y="silhouette", marker='o', label="Raw")

for dataset in ["PCA_2", "PCA_5", "PCA_10"]:
    subset = pca_results[pca_results["dataset"] == dataset]
    sns.lineplot(data=subset, x="k", y="silhouette", marker='o', label=dataset)

plt.title("Silhouette Score Comparison", fontsize=16)
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.legend()
plt.tight_layout()
plt.savefig("../figures/silhouette_comparison.png")
plt.show()

#4. Davies-Bouldin Index
plt.figure(figsize=(10, 6))

sns.lineplot(data=raw_results, x="k", y="davies_bouldin", marker='o', label="Raw")

for dataset in ["PCA_2", "PCA_5", "PCA_10"]:
    subset = pca_results[pca_results["dataset"] == dataset]
    sns.lineplot(data=subset, x="k", y="davies_bouldin", marker='o', label=dataset)

plt.title("Davies-Bouldin Index Comparison", fontsize=16)
plt.xlabel("k")
plt.ylabel("DB Index (Lower is Better)")
plt.legend()
plt.tight_layout()
plt.savefig("../figures/db_index_comparison.png")
plt.show()

#5. Runtime Comparison
plt.figure(figsize=(10, 6))

sns.lineplot(data=raw_results, x="k", y="runtime", marker='o', label="Raw")

for dataset in ["PCA_2", "PCA_5", "PCA_10"]:
    subset = pca_results[pca_results["dataset"] == dataset]
    sns.lineplot(data=subset, x="k", y="runtime", marker='o', label=dataset)

plt.title("Runtime Comparison", fontsize=16)
plt.xlabel("k")
plt.ylabel("Seconds")
plt.legend()
plt.tight_layout()
plt.savefig("../figures/runtime_comparison.png")
plt.show()

#6. Variance Explained Curve
from sklearn.decomposition import PCA

X_full = pd.read_csv("../outputs/X_scaled.csv").values

pca_full = PCA().fit(X_full)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(10, 6))

plt.plot(cum_var, marker='o')
plt.axhline(y=0.8, color='r', linestyle='--', label="80% Variance")
plt.axhline(y=0.9, color='g', linestyle='--', label="90% Variance")

plt.title("Cumulative Explained Variance", fontsize=16)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")

plt.legend()
plt.tight_layout()
plt.savefig("../figures/variance_explained.png")
plt.show()