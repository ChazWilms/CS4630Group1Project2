#PCA — DIMENSIONALITY REDUCTION
# CS 4/5630 - Project 2
# Jack Handley

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import time


def load_scaled_data(filename: str = "data_scaled.txt"):
    # Reads the standardized feature matrix saved by setup.py
    print(f"  Loading scaled data from {filename}...", end=" ")
    t0 = time.time()
    X_scaled = np.loadtxt(filename, delimiter="\t")
    print(f"done. Shape: {X_scaled.shape}  ({time.time() - t0:.2f}s)\n")
    return X_scaled


def variance_analysis(X_scaled: np.ndarray, save_path: str = "PcaFiles/pca_variance.png"):
    # Fits PCA with all 28 components and plots them based on how to preserve the most variance
    print("=" * 60)
    print("SECTION 4A: PCA VARIANCE ANALYSIS (all 28 components)")
    print("=" * 60)

    pca_full = PCA(n_components=28, random_state=42)
    pca_full.fit(X_scaled)

    individual_var = pca_full.explained_variance_ratio_ * 100  # convert to percentage
    cum_var = np.cumsum(individual_var)

    # Prints table
    print(f"  {'PC':<6} {'Individual %':<16} {'Cumulative %'}")
    print(f"  {'-'*40}")
    for i, (ind, cum) in enumerate(zip(individual_var, cum_var)):
        marker = ""
        for t in [80, 90, 95]:
            if abs(cum - t) < ind:
                marker = f"  <- {t}% threshold"
        print(f"  PC{i+1:<4} {ind:<16.2f} {cum:.2f}%{marker}")
    print()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(range(1, 29), individual_var, color='steelblue', edgecolor='black')
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("Individual Explained Variance per PC")
    axes[0].set_xticks(range(1, 29))

    axes[1].plot(range(1, 29), cum_var, marker='o', linewidth=2, color='steelblue')
    for threshold, color in [(80, 'orange'), (90, 'green'), (95, 'red')]:
        idx = int(np.searchsorted(cum_var, threshold))
        axes[1].axhline(threshold, linestyle='--', color=color, linewidth=1,
                        label=f"{threshold}% @ PC{idx+1}")
        axes[1].axvline(idx + 1, linestyle=':', color=color, linewidth=1)
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance (%)")
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].legend()
    axes[1].set_xticks(range(1, 29))

    plt.suptitle("PCA Variance Analysis — HIGGS Dataset (200k sample)", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Saved: {save_path}\n")


def run_pca(X_scaled: np.ndarray, n_components: int):
    # Reduces X_scaled to n_components using PCA and saves the result to a text file
    filename = f"PcaFiles/pca_{n_components}components.txt"
    print(f"  Running PCA -> {n_components} components...", end=" ")

    t0 = time.time()
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    elapsed = time.time() - t0

    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"done.  Variance explained: {explained:.1f}%  |  Time: {elapsed:.2f}s")

    # Save reduced data to text file
    np.savetxt(filename, X_pca, fmt="%.6f", delimiter="\t")
    print(f"  Saved: {filename}  (shape {X_pca.shape})")


def run_all_pca(X_scaled: np.ndarray, components_list: list):
    import os
    os.makedirs("PcaFiles", exist_ok=True)

    results = {}

    for n in components_list:
        print(f"  Running PCA -> {n} components...", end=" ")

        t0 = time.time()
        pca = PCA(n_components=n, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        elapsed = time.time() - t0

        explained = pca.explained_variance_ratio_.sum() * 100
        print(f"done.  Variance explained: {explained:.1f}%  |  Time: {elapsed:.2f}s")

        filename = f"PcaFiles/pca_{n}components.txt"
        np.savetxt(filename, X_pca, fmt="%.6f", delimiter="\t")
        print(f"  Saved: {filename}  (shape {X_pca.shape})")

        results[n] = (X_pca, pca, explained)

    print()
    return results



# ── Run ──────────────────────────────────────────────────────
if __name__ == "__main__":
    from Setup import PCA_COMPONENTS

    X_scaled = load_scaled_data("data_scaled.txt")
    variance_analysis(X_scaled)
    run_all_pca(X_scaled, PCA_COMPONENTS)