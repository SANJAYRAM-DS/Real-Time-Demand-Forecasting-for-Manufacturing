import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# 📂 PROJECT STRUCTURE & PATHS (absolute + safe)
# =========================================================

# Dynamically locate the project root (folder containing 'pipelines')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Main directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "lightgbm")

# Ensure directories exist
for path in [DATA_DIR, RAW_DIR, PROCESSED_DIR, REPORT_DIR, MODEL_DIR]:
    os.makedirs(path, exist_ok=True)


# =========================================================
# 🧠 LOGGING SETUP
# =========================================================

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("forecasting_utils")

def log(msg: str, level: str = "info"):
    """
    Simple logger wrapper with timestamp and emoji-style labels.
    """
    prefix = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
    message = f"{prefix} {msg}"
    if level.lower() == "info":
        logger.info(message)
    elif level.lower() == "warning":
        logger.warning(message)
    elif level.lower() == "error":
        logger.error(message)
    else:
        print(message)


# =========================================================
# 📊 METRIC FUNCTIONS
# =========================================================

def mae(y, yhat):
    """Mean Absolute Error"""
    return float(np.mean(np.abs(y - yhat)))

def rmse(y, yhat):
    """Root Mean Squared Error"""
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def mape(y, yhat):
    """Mean Absolute Percentage Error (safe for zeros)"""
    denom = np.where(y == 0, 1, y)
    return float(np.mean(np.abs((y - yhat) / denom)) * 100)

def evaluate_forecast(y_true, y_pred):
    """
    Returns a dictionary of core metrics for model evaluation.
    """
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred)
    }


# =========================================================
# 💾 FILE UTILITIES
# =========================================================

def save_json(obj, path):
    """
    Save a Python object as JSON.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    log(f"✅ Saved JSON → {path}")

def load_json(path):
    """
    Load a JSON file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ JSON not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def save_dataframe(df, path, index=False):
    """
    Save DataFrame as CSV or Parquet based on file extension.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".csv"):
        df.to_csv(path, index=index)
    elif path.endswith(".parquet"):
        df.to_parquet(path, index=index)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")
    log(f"✅ Saved DataFrame → {path}")

def load_dataframe(path):
    """
    Load DataFrame from CSV or Parquet file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")


# =========================================================
# 🧪 DEBUGGING / SELF-TEST
# =========================================================

if __name__ == "__main__":
    log("🔧 Running utils self-test...")
    log(f"PROJECT_ROOT: {PROJECT_ROOT}")
    log(f"RAW_DIR: {RAW_DIR}")
    log(f"Contents of RAW_DIR: {os.listdir(RAW_DIR) if os.path.exists(RAW_DIR) else '❌ RAW_DIR missing!'}")

    # Example metric test
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    log(f"Metrics → {evaluate_forecast(y_true, y_pred)}")