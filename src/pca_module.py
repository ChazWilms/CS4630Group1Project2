from sklearn.decomposition import PCA

def apply_pca(X, n_components):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)

    return X_reduced, pca.explained_variance_ratio_