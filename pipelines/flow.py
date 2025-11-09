# pipeline/flow.py
from prefect import flow, task
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pipelines import _01_ingest, _02_validate, _03_features, _04_predict, _05_evaluate

@flow(name="m5_forecasting_pipeline")
def full_pipeline():
    data = _01_ingest.ingest_data()
    _02_validate.validate_data()
    features = _03_features.create_features()
    _04_predict.train_and_predict()
    _05_evaluate.evaluate()

if __name__ == "__main__":
    full_pipeline()
