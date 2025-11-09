from pipelines.utils import logger, RAW_DIR, PROCESSED_DIR
import pandas as pd

def validate_data():
    logger.info("🔍 Validating merged data...")
    df = pd.read_parquet(f"{PROCESSED_DIR}/raw_merged.parquet")

    assert df['sales'].ge(0).all(), "❌ Negative sales found!"
    assert df['date'].is_monotonic_increasing or df['date'].is_monotonic_decreasing, "❌ Dates not sorted"
    logger.info(f"✅ Validation passed — {len(df):,} rows, {df['id'].nunique()} SKUs")

if __name__ == "__main__":
    validate_data()
