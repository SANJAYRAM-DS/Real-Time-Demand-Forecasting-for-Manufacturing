from pipelines.utils import logger, RAW_DIR, PROCESSED_DIR
import pandas as pd
import matplotlib.pyplot as plt

def evaluate():
    hist = pd.read_parquet(f"{PROCESSED_DIR}/features.parquet")[['id','date','sales']]
    pred = pd.read_parquet(f"{PROCESSED_DIR}/predictions.parquet")

    full = hist.merge(pred, on=['id','date'], how='left')
    full['abs_err'] = (full['sales'] - full['yhat']).abs()
    err_cat = full.groupby('id')['abs_err'].mean().mean()

    plt.figure(figsize=(10,5))
    agg = full.groupby('date')[['sales','yhat']].sum()
    plt.plot(agg.index, agg['sales'], label='Actual')
    plt.plot(agg.index, agg['yhat'], label='Predicted')
    plt.legend()
    plt.title("Daily Sales: Actual vs Predicted")
    plt.savefig(f"{REPORT_DIR}/agg_actual_vs_predicted.png", dpi=160)
    logger.info("✅ Evaluation complete & charts saved.")

if __name__ == "__main__":
    evaluate()
