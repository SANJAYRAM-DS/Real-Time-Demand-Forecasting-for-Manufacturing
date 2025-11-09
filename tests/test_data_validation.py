import pandas as pd
import pytest
from pipeline import validate

def test_no_negative_sales():
    df = pd.read_parquet("data/processed/raw_merged.parquet")
    assert (df['sales'] >= 0).all()
