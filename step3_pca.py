"""
Step 3: PCA Dimensionality Reduction
- Reduce to 2, 5, and 10 components
- Analyze explained variance
- Save reduced datasets for Step 4
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
# 3a. Full PCA (28 components) to analyze variance
# ============================================================
print("\nFitting full PCA (28 components)...")
start = time.time()
pca_full = PCA(n_components=28, random_state=42)
pca_full.fit(X_scaled)
full_pca_time = time.time() - start
print(f"Full PCA fit time: {full_pca_time:.2f}s")

explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

print("\n--- Explained Variance per Component ---")
for i, (ev, cv) in enumerate(zip(explained, cumulative)):
    bar = "#" * int(ev * 100)
    print(f"  PC{i+1:>2}: {ev:.4f} (cumulative: {cv:.4f}) {bar}")

# ============================================================
# 3b. Reduce to 2, 5, 10 components
# ============================================================
n_components_list = [2, 5, 10]

for n_comp in n_components_list:
    print(f"\n--- PCA with {n_comp} components ---")
    start = time.time()
    pca = PCA(n_components=n_comp, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)
    pca_time = time.time() - start

    var_preserved = np.sum(pca.explained_variance_ratio_)
    print(f"  Transform time: {pca_time:.2f}s")
    print(f"  Variance preserved: {var_preserved:.4f} ({var_preserved*100:.1f}%)")
    print(f"  Output shape: {X_reduced.shape}")

    # Save reduced data
    np.save(os.path.join(DATA_DIR, f"X_pca_{n_comp}.npy"), X_reduced)
    print(f"  Saved to data/X_pca_{n_comp}.npy")

# ============================================================
# 3c. Plots
# ============================================================

# 1. Cumulative explained variance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(1, 29), explained, color="steelblue", alpha=0.7, label="Individual")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance Ratio")
axes[0].set_title("Explained Variance per Component")
axes[0].grid(axis="y")

axes[1].plot(range(1, 29), cumulative, "ro-", linewidth=2)
axes[1].axhline(y=0.90, color="gray", linestyle="--", label="90% variance")
axes[1].axhline(y=0.95, color="gray", linestyle=":", label="95% variance")
for nc in n_components_list:
    axes[1].axvline(x=nc, color="green", linestyle="--", alpha=0.5)
    axes[1].annotate(f"n={nc}\n({cumulative[nc-1]*100:.1f}%)",
                     xy=(nc, cumulative[nc-1]), fontsize=9,
                     xytext=(nc+0.5, cumulative[nc-1]-0.05))
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Explained Variance")
axes[1].set_title("Cumulative Explained Variance")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "step3_pca_variance.png"), dpi=150)
print(f"\nSaved variance plot to output/step3_pca_variance.png")

# 2. 2D PCA scatter plot colored by true label
X_2d = np.load(os.path.join(DATA_DIR, "X_pca_2.npy"))
fig, ax = plt.subplots(figsize=(10, 7))
sample_idx = np.random.RandomState(42).choice(len(X_2d), size=10000, replace=False)
scatter = ax.scatter(X_2d[sample_idx, 0], X_2d[sample_idx, 1],
                     c=y_labels[sample_idx], cmap="coolwarm", alpha=0.3, s=5)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("2D PCA Projection (colored by true label, 10K sample)")
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Label (0=background, 1=signal)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "step3_pca_2d_scatter.png"), dpi=150)
print(f"Saved 2D scatter plot to output/step3_pca_2d_scatter.png")

print("\nStep 3 complete!")
