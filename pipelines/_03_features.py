from pipelines.utils import logger, RAW_DIR, PROCESSED_DIR
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def create_features():
    df = pd.read_parquet(f"{PROCESSED_DIR}/raw_merged.parquet")
    df = df.sort_values(["id", "date"])
    logger.info("🧠 Creating time-based features...")

    # Lags
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df.groupby("id")["sales"].shift(lag)

    # Rolling windows
    for w in [7, 14, 28]:
        grp = df.groupby("id")["sales"]
        df[f"rmean_{w}"] = grp.shift(1).rolling(w).mean()
        df[f"rstd_{w}"] = grp.shift(1).rolling(w).std()

    # Date features
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Cyclic encoding
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # Price normalization
    df['price_max'] = df.groupby(['store_id','item_id'])['sell_price'].transform('max')
    df['price_min'] = df.groupby(['store_id','item_id'])['sell_price'].transform('min')
    df['price_norm'] = (df['sell_price'] - df['price_min']) / (df['price_max'] - df['price_min'])

    # Label encoding
    for col in ['item_id','dept_id','cat_id','store_id','state_id']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    df = df.dropna(subset=['lag_28']).fillna(0)
    out = f"{PROCESSED_DIR}/features.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"✅ Feature file saved → {out}")
    return out

if __name__ == "__main__":
    create_features()
