# K-MEANS CLUSTERING
# CS 4/5630 - Project 2
# Jack Handley

from sklearn.cluster import MiniBatchKMeans
import numpy as np
import time


def run_kmeans(X: np.ndarray, label: str, k: int = 2, batch_size: int = 10_000, random_state: int = 42):

    print(f"  k-Means on [{label}]  shape={X.shape}...", end=" ")
    t0 = time.time()

    km = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        random_state=random_state,
        n_init=5,
        max_iter=300
    )

    cluster_labels = km.fit_predict(X)
    elapsed = time.time() - t0

    cluster_sizes = dict(zip(*np.unique(cluster_labels, return_counts=True)))
    print(f"done.  Time: {elapsed:.2f}s  |  Cluster sizes: {cluster_sizes}")

    return cluster_labels, km, elapsed


def run_all_kmeans(X_scaled: np.ndarray, pca_results: dict, k: int = 2,
                   batch_size: int = 10_000, random_state: int = 42):

    print("=" * 60)
    print("SECTION 5: K-MEANS CLUSTERING")
    print("=" * 60)
    print(f"  Algorithm : MiniBatchKMeans  (k={k})")
    print(f"  Batch size: {batch_size:,}")
    print()

    km_results = {}

    # Raw 28D data
    labels, km, elapsed = run_kmeans(
        X_scaled, label="Raw 28D", k=k,
        batch_size=batch_size, random_state=random_state
    )
    km_results["Raw 28D"] = (labels, km, elapsed)

    # PCA-reduced datasets
    for n_components, (X_pca, pca_model, explained) in pca_results.items():
        lbl = f"PCA {n_components}D"
        labels, km, elapsed = run_kmeans(
            X_pca, label=lbl, k=k,
            batch_size=batch_size, random_state=random_state
        )
        km_results[lbl] = (labels, km, elapsed)

    print()
    return km_results


# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    from Setup import *
    from _4630_PCA import run_all_pca, variance_analysis

    X_raw, y    = load_and_subsample(DATA_PATH, SAMPLE_SIZE, RANDOM_STATE)
    X_scaled, _ = preprocess(X_raw)

    variance_analysis(X_scaled)

    pca_results = run_all_pca(X_scaled, PCA_COMPONENTS)

    km_results = run_all_kmeans(
        X_scaled, pca_results,
        k=K, batch_size=MINI_BATCH_SIZE, random_state=RANDOM_STATE
    )
