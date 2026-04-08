import pandas as pd

raw = pd.read_csv("../outputs/clustering_raw_results.csv")
pca = pd.read_csv("../outputs/clustering_pca_results.csv")

combined = pd.concat([raw, pca])
combined.to_csv("../outputs/final_comparison.csv", index=False)

print("Final comparison file created.")