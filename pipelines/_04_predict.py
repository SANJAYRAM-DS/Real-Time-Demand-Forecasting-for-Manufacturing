from pipelines.utils import logger, RAW_DIR, PROCESSED_DIR
import pandas as pd
import lightgbm as lgb
import joblib

def train_and_predict():
    df = pd.read_parquet(f"{PROCESSED_DIR}/features.parquet")
    cutoff = pd.Timestamp("2016-03-27")

    train = df[df.date <= cutoff]
    valid = df[df.date > cutoff]

    X_cols = [c for c in df.columns if c not in ['id','date','sales']]
    X_tr, y_tr = train[X_cols], train['sales']
    X_va, y_va = valid[X_cols], valid['sales']

    logger.info(f"🧩 Training LightGBM — {len(X_tr):,} rows")
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 127,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_data_in_leaf': 100,
        'verbosity': -1
    }

    model = lgb.LGBMRegressor(**params, n_estimators=2000)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='rmse',
              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])

    joblib.dump(model, f"{MODEL_DIR}/model.pkl")
    logger.info(f"✅ Model saved → {MODEL_DIR}/model.pkl")

    yhat = model.predict(X_va)
    metrics = {'MAE': mae(y_va, yhat), 'RMSE': rmse(y_va, yhat), 'MAPE': mape(y_va, yhat)}
    logger.info(f"📊 Validation Metrics → {metrics}")
    pd.DataFrame({'id': valid['id'], 'date': valid['date'], 'yhat': yhat}).to_parquet(f"{PROCESSED_DIR}/predictions.parquet", index=False)
    return metrics

if __name__ == "__main__":
    train_and_predict()
