"""
Step 1: Data Loading & Preprocessing
- Load the UCI HIGGS dataset (HIGGS.csv.gz)
- Explore basic statistics
- Subsample to 200K rows for scalability
- Standardize features
- Save preprocessed data for subsequent steps
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
import os

# ============================================================
# 1a. Load the full dataset
# ============================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "HIGGS.csv.gz")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

# Column names: first column is the label, remaining 28 are features
feature_names = [f"feature_{i}" for i in range(1, 29)]
col_names = ["label"] + feature_names

print("Loading HIGGS dataset (this may take a few minutes)...")
start = time.time()
df_full = pd.read_csv(DATA_PATH, header=None, names=col_names)
load_time = time.time() - start
print(f"Loaded in {load_time:.1f}s")

# ============================================================
# 1b. Basic exploration
# ============================================================
print(f"\n--- Dataset Shape ---")
print(f"Rows: {df_full.shape[0]:,}")
print(f"Columns: {df_full.shape[1]}")

print(f"\n--- Label Distribution ---")
label_counts = df_full["label"].value_counts().sort_index()
for lab, cnt in label_counts.items():
    print(f"  Label {int(lab)}: {cnt:,} ({cnt/len(df_full)*100:.1f}%)")

print(f"\n--- Feature Statistics (first 5 features) ---")
print(df_full[feature_names[:5]].describe().round(4).to_string())

print(f"\n--- Missing Values ---")
missing = df_full.isnull().sum().sum()
print(f"Total missing values: {missing}")

# ============================================================
# 1c. Subsample to 200K rows (stratified by label)
# ============================================================
SAMPLE_SIZE = 200_000
print(f"\n--- Subsampling to {SAMPLE_SIZE:,} rows (stratified) ---")

# Stratified sampling to preserve label ratio
np.random.seed(42)
# Stratified sample: preserve label proportions
idx_0 = df_full.index[df_full["label"] == 0.0]
idx_1 = df_full.index[df_full["label"] == 1.0]
n0 = int(SAMPLE_SIZE * len(idx_0) / len(df_full))
n1 = SAMPLE_SIZE - n0
sampled_idx = np.concatenate([
    np.random.choice(idx_0, size=n0, replace=False),
    np.random.choice(idx_1, size=n1, replace=False)
])
df_sample = df_full.loc[sampled_idx].reset_index(drop=True)
print(f"Sample size: {len(df_sample):,}")
print(f"Sample label distribution:")
sample_counts = df_sample["label"].value_counts().sort_index()
for lab, cnt in sample_counts.items():
    print(f"  Label {int(lab)}: {cnt:,} ({cnt/len(df_sample)*100:.1f}%)")

# Justification for subsampling
print(f"\n--- Subsampling Justification ---")
print(f"Full dataset has {df_full.shape[0]:,} rows.")
print(f"Silhouette score computation is O(n^2) and infeasible on 11M rows.")
print(f"k-Means on 11M rows is very slow for iterative experimentation.")
print(f"200K rows preserves the label distribution and is sufficient for")
print(f"meaningful clustering analysis while keeping runtime manageable.")

# ============================================================
# 1d. Separate features and labels, then standardize
# ============================================================
X = df_sample[feature_names].values
y = df_sample["label"].values

print(f"\n--- Standardizing Features ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Feature means after scaling (first 5): {X_scaled[:, :5].mean(axis=0).round(6)}")
print(f"Feature stds after scaling (first 5):  {X_scaled[:, :5].std(axis=0).round(6)}")

# ============================================================
# 1e. Save preprocessed data
# ============================================================
np.save(os.path.join(OUTPUT_DIR, "X_scaled.npy"), X_scaled)
np.save(os.path.join(OUTPUT_DIR, "y_labels.npy"), y)

# Also save the scaler parameters for reproducibility
scaler_params = pd.DataFrame({
    "feature": feature_names,
    "mean": scaler.mean_,
    "std": scaler.scale_
})
scaler_params.to_csv(os.path.join(OUTPUT_DIR, "scaler_params.csv"), index=False)

print(f"\nSaved preprocessed data to {OUTPUT_DIR}/")
print(f"  - X_scaled.npy: {X_scaled.shape}")
print(f"  - y_labels.npy: {y.shape}")
print(f"  - scaler_params.csv")
print(f"\nStep 1 complete!")
