import pandas as pd
from preprocessing import load_data, preprocess

# Load subset
df = load_data("../data/HIGGS.csv.gz", nrows=200000)

# Preprocess
X, y, scaler = preprocess(df)

# Save outputs
pd.DataFrame(X).to_csv("../outputs/X_scaled.csv", index=False)
pd.Series(y).to_csv("../outputs/y.csv", index=False)

print("Step 1 complete: Preprocessing done.")