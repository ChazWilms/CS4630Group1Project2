import pandas as pd
import matplotlib.pyplot as plt
from clustering import run_kmeans

X = pd.read_csv("../outputs/X_scaled.csv").values

print("\n--- Clustering Raw Data ---")

results = []
inertias = []

for k in range(2, 10):
    res = run_kmeans(X, k)

    print(f"k={k} | Sil={res['silhouette']:.4f} | DB={res['davies_bouldin']:.4f}")

    inertias.append(res["inertia"])

    # Save labels
    pd.Series(res["labels"]).to_csv(f"../outputs/labels_raw_k{k}.csv", index=False)

    results.append({
        "dataset": "RAW",
        "k": k,
        "runtime": res["runtime"],
        "silhouette": res["silhouette"],
        "davies_bouldin": res["davies_bouldin"],
        "inertia": res["inertia"],
        "iterations": res["iterations"]
    })

# Save results
pd.DataFrame(results).to_csv("../outputs/clustering_raw_results.csv", index=False)

# Elbow plot
plt.plot(range(2, 10), inertias, marker='o')
plt.title("Elbow Method (Raw Data)")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.savefig("../outputs/elbow_raw.png")
plt.close()

print("Step 2 complete.")