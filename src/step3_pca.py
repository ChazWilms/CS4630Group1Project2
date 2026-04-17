import pandas as pd
import numpy as np
from pca_module import apply_pca

X = pd.read_csv("../outputs/X_scaled.csv").values

X_pca_2, var2 = apply_pca(X, 2)
X_pca_5, var5 = apply_pca(X, 5)
X_pca_10, var10 = apply_pca(X, 10)

# Save datasets
pd.DataFrame(X_pca_2).to_csv("../outputs/X_pca_2.csv", index=False)
pd.DataFrame(X_pca_5).to_csv("../outputs/X_pca_5.csv", index=False)
pd.DataFrame(X_pca_10).to_csv("../outputs/X_pca_10.csv", index=False)

# Variance info
print("Variance retained:")
print("2 components:", np.sum(var2))
print("5 components:", np.sum(var5))
print("10 components:", np.sum(var10))

print("Step 3 complete.")