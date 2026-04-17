import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath, nrows=None):
    cols = ['label'] + [f'feature_{i}' for i in range(28)]
    df = pd.read_csv(filepath, names=cols, nrows=nrows)
    return df

def preprocess(df):
    X = df.drop('label', axis=1)
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler