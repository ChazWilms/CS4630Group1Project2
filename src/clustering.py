from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
import time

def evaluate_subset(X, labels, sample_size=10000):
    idx = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
    return X[idx], labels[idx]

def run_kmeans(X, k):
    start = time.time()

    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(X)

    runtime = time.time() - start

    # Subsample for metrics
    X_sample, labels_sample = evaluate_subset(X, labels)

    sil = silhouette_score(X_sample, labels_sample)
    db = davies_bouldin_score(X_sample, labels_sample)

    return {
        "k": k,
        "runtime": runtime,
        "silhouette": sil,
        "davies_bouldin": db,
        "inertia": model.inertia_,
        "iterations": model.n_iter_,
        "labels": labels
    }