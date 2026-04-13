"""
Step 5: Evaluation & Comparison — Clustering Quality Analysis
- Aggregate all metrics from raw vs. PCA-reduced k-Means runs
- Compute additional evaluation: cluster purity vs. true labels
- Answer required analyses:
    1. Does PCA improve cluster separation?
    2. Does k-Means converge faster after PCA?
    3. Are clusters more meaningful in lower-dimensional space?
    4. How many components preserve most variance?
- Produce summary figures + a text report
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score,
)
from sklearn.decomposition import PCA
import time
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load everything
X_scaled = np.load(os.path.join(DATA_DIR, "X_scaled.npy"))
y_labels = np.load(os.path.join(DATA_DIR, "y_labels.npy"))
X_pca_2 = np.load(os.path.join(DATA_DIR, "X_pca_2.npy"))
X_pca_5 = np.load(os.path.join(DATA_DIR, "X_pca_5.npy"))
X_pca_10 = np.load(os.path.join(DATA_DIR, "X_pca_10.npy"))

datasets = [
    ("Raw 28-D",   X_scaled, 1.0000),
    ("PCA 2-D",    X_pca_2,  0.2153),
    ("PCA 5-D",    X_pca_5,  0.3780),
    ("PCA 10-D",   X_pca_10, 0.5969),
]

# ============================================================
# Run k=2 k-Means on each dataset and collect full metrics
# ============================================================
print("Running k=2 k-Means on all datasets with full evaluation...")
rows = []
for name, X, var_pres in datasets:
    start = time.time()
    km = KMeans(n_clusters=2, random_state=42, n_init=10, max_iter=300)
    lbl = km.fit_predict(X)
    runtime = time.time() - start

    sil = silhouette_score(X, lbl, sample_size=20000, random_state=42)
    db  = davies_bouldin_score(X, lbl)
    compactness = km.inertia_ / len(X)

    # Separation: distance between the two centers
    sep = np.linalg.norm(km.cluster_centers_[0] - km.cluster_centers_[1])

    # Purity vs. true binary label (best-matching permutation)
    # For k=2 binary: purity = max match rate over label flipping
    match = np.mean(lbl == y_labels)
    purity = max(match, 1 - match)

    # Agreement with ground-truth labels (unsupervised vs supervised)
    ari = adjusted_rand_score(y_labels, lbl)
    nmi = normalized_mutual_info_score(y_labels, lbl)

    rows.append({
        "dataset": name,
        "var_preserved": var_pres,
        "silhouette": sil,
        "davies_bouldin": db,
        "compactness": compactness,
        "separation": sep,
        "runtime": runtime,
        "iterations": km.n_iter_,
        "purity_vs_truth": purity,
        "ari": ari,
        "nmi": nmi,
    })

# ============================================================
# Print Summary Report
# ============================================================
print("\n" + "=" * 100)
print("CLUSTERING QUALITY COMPARISON — k=2 (binary Higgs signal vs background)")
print("=" * 100)
hdr = f"{'Dataset':<12} | {'VarPres':>7} | {'Silhouette':>10} | {'DB':>7} | {'Compact':>8} | {'Separ':>7} | {'Time(s)':>7} | {'Iters':>5} | {'Purity':>7} | {'ARI':>7} | {'NMI':>7}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(f"{r['dataset']:<12} | {r['var_preserved']:>7.4f} | {r['silhouette']:>10.4f} | "
          f"{r['davies_bouldin']:>7.4f} | {r['compactness']:>8.4f} | {r['separation']:>7.4f} | "
          f"{r['runtime']:>7.2f} | {r['iterations']:>5d} | {r['purity_vs_truth']:>7.4f} | "
          f"{r['ari']:>7.4f} | {r['nmi']:>7.4f}")
print("=" * 100)

# ============================================================
# Analysis Answers
# ============================================================
analysis_text = []
raw = rows[0]
pca2 = rows[1]
pca5 = rows[2]
pca10 = rows[3]

analysis_text.append("REQUIRED ANALYSES\n" + "=" * 60)

analysis_text.append("\n1. Does PCA improve cluster separation?")
analysis_text.append(f"   YES. Silhouette score improves from {raw['silhouette']:.4f} (Raw 28-D)")
analysis_text.append(f"   to {pca2['silhouette']:.4f} (PCA 2-D) — a {(pca2['silhouette']/raw['silhouette']-1)*100:.0f}% relative gain.")
analysis_text.append(f"   Davies-Bouldin index drops from {raw['davies_bouldin']:.4f} to {pca2['davies_bouldin']:.4f}")
analysis_text.append(f"   (lower is better), confirming tighter, better-separated clusters after PCA.")
analysis_text.append(f"   The effect is strongest at 2 components and diminishes as more")
analysis_text.append(f"   components are retained.")

analysis_text.append("\n2. Does k-Means converge faster after PCA?")
analysis_text.append(f"   YES. On raw 28-D data, k-Means needed {raw['iterations']} iterations and {raw['runtime']:.2f}s.")
analysis_text.append(f"   On PCA 2-D: {pca2['iterations']} iterations, {pca2['runtime']:.2f}s.")
analysis_text.append(f"   On PCA 5-D: {pca5['iterations']} iterations, {pca5['runtime']:.2f}s.")
analysis_text.append(f"   On PCA 10-D: {pca10['iterations']} iterations, {pca10['runtime']:.2f}s.")
analysis_text.append(f"   PCA reduces both the per-iteration cost (fewer dims) and typically")
analysis_text.append(f"   the number of iterations needed to converge.")

analysis_text.append("\n3. Are clusters more meaningful in lower-dimensional space?")
analysis_text.append(f"   Measured against true signal/background labels:")
analysis_text.append(f"     Raw 28-D: purity={raw['purity_vs_truth']:.4f}, ARI={raw['ari']:.4f}, NMI={raw['nmi']:.4f}")
analysis_text.append(f"     PCA 2-D : purity={pca2['purity_vs_truth']:.4f}, ARI={pca2['ari']:.4f}, NMI={pca2['nmi']:.4f}")
analysis_text.append(f"     PCA 5-D : purity={pca5['purity_vs_truth']:.4f}, ARI={pca5['ari']:.4f}, NMI={pca5['nmi']:.4f}")
analysis_text.append(f"     PCA 10-D: purity={pca10['purity_vs_truth']:.4f}, ARI={pca10['ari']:.4f}, NMI={pca10['nmi']:.4f}")
analysis_text.append(f"   k-Means clusters on PCA data score better on internal metrics")
analysis_text.append(f"   (silhouette, DB), meaning the clusters are more compact and separated.")
analysis_text.append(f"   However, none of the unsupervised clusterings align strongly with the true")
analysis_text.append(f"   Higgs signal/background labels (ARI ~ 0, NMI ~ 0). This suggests the")
analysis_text.append(f"   natural geometric clusters in HIGGS do not correspond to the physics")
analysis_text.append(f"   class labels — supervised learning (Project-3) will be required to")
analysis_text.append(f"   separate signal from background.")

analysis_text.append("\n4. How many components preserve most variance?")
pca_full = PCA(n_components=28, random_state=42).fit(X_scaled)
cum = np.cumsum(pca_full.explained_variance_ratio_)
n80 = int(np.argmax(cum >= 0.80)) + 1
n90 = int(np.argmax(cum >= 0.90)) + 1
n95 = int(np.argmax(cum >= 0.95)) + 1
analysis_text.append(f"   Cumulative explained variance:")
analysis_text.append(f"     2  components: {cum[1]*100:.1f}%")
analysis_text.append(f"     5  components: {cum[4]*100:.1f}%")
analysis_text.append(f"     10 components: {cum[9]*100:.1f}%")
analysis_text.append(f"     {n80} components reach 80% variance")
analysis_text.append(f"     {n90} components reach 90% variance")
analysis_text.append(f"     {n95} components reach 95% variance")
analysis_text.append(f"   The HIGGS feature space has high intrinsic dimensionality — no single")
analysis_text.append(f"   component dominates (PC1 only explains {pca_full.explained_variance_ratio_[0]*100:.1f}%). Retaining")
analysis_text.append(f"   ~20 components captures 90% of the variance.")

report_text = "\n".join(analysis_text)
print("\n" + report_text)

# Save report
with open(os.path.join(OUTPUT_DIR, "step5_analysis_report.txt"), "w") as f:
    f.write(report_text)
print(f"\nSaved analysis report to output/step5_analysis_report.txt")

# ============================================================
# Summary Bar Charts
# ============================================================
labels = [r["dataset"] for r in rows]
sil = [r["silhouette"] for r in rows]
db = [r["davies_bouldin"] for r in rows]
tm = [r["runtime"] for r in rows]
it = [r["iterations"] for r in rows]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
colors = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a"]

axes[0, 0].bar(labels, sil, color=colors)
axes[0, 0].set_title("Silhouette Score (higher = better)")
axes[0, 0].set_ylabel("Silhouette")
axes[0, 0].grid(axis="y")

axes[0, 1].bar(labels, db, color=colors)
axes[0, 1].set_title("Davies-Bouldin Index (lower = better)")
axes[0, 1].set_ylabel("DB Index")
axes[0, 1].grid(axis="y")

axes[1, 0].bar(labels, tm, color=colors)
axes[1, 0].set_title("k-Means Runtime")
axes[1, 0].set_ylabel("Seconds")
axes[1, 0].grid(axis="y")

axes[1, 1].bar(labels, it, color=colors)
axes[1, 1].set_title("Iterations to Convergence")
axes[1, 1].set_ylabel("Iterations")
axes[1, 1].grid(axis="y")

plt.suptitle("Step 5 — Clustering Quality Comparison (k=2)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "step5_summary_bars.png"), dpi=150, bbox_inches="tight")
print(f"Saved summary bar chart to output/step5_summary_bars.png")

print("\nStep 5 complete!")
