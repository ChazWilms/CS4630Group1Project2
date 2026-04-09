import pandas as pd
import numpy as np

raw = pd.read_csv("../outputs/clustering_raw_results.csv")
pca = pd.read_csv("../outputs/clustering_pca_results.csv")

combined = pd.concat([raw, pca])
combined.to_csv("../outputs/final_comparison.csv", index=False)

print("=" * 60)
print("FINAL ANALYSIS SUMMARY")
print("=" * 60)

# --- Q1: Does PCA improve cluster separation? ---
print("\n[1] Does PCA improve cluster separation?")
print("    (Higher silhouette = better; lower DB = better; higher separation = better)\n")

all_datasets = ["RAW", "PCA_2", "PCA_5", "PCA_10"]
for ds in all_datasets:
    subset = combined[combined["dataset"] == ds]
    best_sil = subset.loc[subset["silhouette"].idxmax()]
    print(f"  {ds:8s} | best silhouette={best_sil['silhouette']:.4f} at k={int(best_sil['k'])} "
          f"| DB={best_sil['davies_bouldin']:.4f} "
          f"| centroid separation={best_sil['mean_centroid_separation']:.4f}")

raw_best_sil = raw["silhouette"].max()
pca_best_sil = pca["silhouette"].max()
if pca_best_sil > raw_best_sil:
    print(f"\n  -> PCA IMPROVES cluster separation (PCA best={pca_best_sil:.4f} vs RAW={raw_best_sil:.4f})")
else:
    print(f"\n  -> PCA does NOT clearly improve cluster separation (PCA best={pca_best_sil:.4f} vs RAW={raw_best_sil:.4f})")

# --- Q2: Does k-Means converge faster after PCA? ---
print("\n[2] Does k-Means converge faster after PCA?")
print("    (Mean runtime in seconds across k=2..9)\n")

for ds in all_datasets:
    subset = combined[combined["dataset"] == ds]
    print(f"  {ds:8s} | mean runtime={subset['runtime'].mean():.2f}s "
          f"| mean iterations={subset['iterations'].mean():.1f}")

raw_rt = raw["runtime"].mean()
pca_rt = pca[pca["dataset"] == "PCA_10"]["runtime"].mean()
speedup = raw_rt / pca_rt if pca_rt > 0 else float("nan")
print(f"\n  -> PCA_10 is {speedup:.1f}x faster than RAW on average")

# --- Q3: Are clusters more meaningful in lower-dimensional space? ---
print("\n[3] Are clusters more meaningful in lower-dimensional space?")
print("    (Comparing silhouette and Davies-Bouldin at each k)\n")

header = f"  {'k':>3}  {'RAW sil':>9}  {'PCA_2 sil':>10}  {'PCA_5 sil':>10}  {'PCA_10 sil':>11}"
print(header)
print("  " + "-" * (len(header) - 2))

for k in sorted(combined["k"].unique()):
    row = f"  {int(k):>3}"
    for ds in all_datasets:
        val = combined[(combined["dataset"] == ds) & (combined["k"] == k)]["silhouette"]
        row += f"  {val.values[0]:>10.4f}" if len(val) > 0 else f"  {'N/A':>10}"
    print(row)

# --- Q4: How many components preserve most variance? ---
print("\n[4] Variance retained by PCA reduction:")
print("    (Refer to step3_pca.py console output for exact values)")
print("    PCA_2  -> 2 components")
print("    PCA_5  -> 5 components")
print("    PCA_10 -> 10 components")
print("    See figures/variance_explained.png for cumulative variance curve.")

print("\n" + "=" * 60)
print("Step 6 complete. Final comparison saved to outputs/final_comparison.csv")
print("=" * 60)
