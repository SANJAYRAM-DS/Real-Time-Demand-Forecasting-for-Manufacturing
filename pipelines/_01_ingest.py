from pipelines.utils import logger, RAW_DIR, PROCESSED_DIR
import pandas as pd

def ingest_data():
    logger.info("🚀 Ingesting raw CSV files...")
    calendar = pd.read_csv(f"{RAW_DIR}/calendar.csv")
    prices = pd.read_csv(f"{RAW_DIR}/sell_prices.csv")
    sales = pd.read_csv(f"{RAW_DIR}/sales_train_validation.csv")

    # melt sales
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    df = sales.melt(id_vars=id_cols, var_name="d", value_name="sales")

    # merge
    df = df.merge(calendar, on="d", how="left")
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["date"])

    out = f"{PROCESSED_DIR}/raw_merged.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"✅ Ingested & saved merged data → {out}")
    return out

if __name__ == "__main__":
    ingest_data()
