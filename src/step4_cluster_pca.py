import pandas as pd
from clustering import run_kmeans

datasets = {
    "PCA_2": pd.read_csv("../outputs/X_pca_2.csv").values,
    "PCA_5": pd.read_csv("../outputs/X_pca_5.csv").values,
    "PCA_10": pd.read_csv("../outputs/X_pca_10.csv").values
}

print("\n--- Clustering PCA Data ---")

results = []

for name, data in datasets.items():
    for k in range(2, 10):
        res = run_kmeans(data, k)

        print(f"{name} k={k} | Sil={res['silhouette']:.4f}")

        # Save labels
        pd.Series(res["labels"]).to_csv(f"../labels/labels_{name}_k{k}.csv", index=False)

        results.append({
            "dataset": name,
            "k": k,
            "runtime": res["runtime"],
            "silhouette": res["silhouette"],
            "davies_bouldin": res["davies_bouldin"],
            "inertia": res["inertia"],
            "iterations": res["iterations"],
            "mean_centroid_separation": res["mean_centroid_separation"]
        })

pd.DataFrame(results).to_csv("../outputs/clustering_pca_results.csv", index=False)

print("Step 4 complete.")