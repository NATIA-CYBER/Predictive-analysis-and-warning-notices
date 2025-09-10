#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

def load_raw_hr(csv_path):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Department": "sales"})
    return df

def load_silver_hr(parquet_path):
    return pd.read_parquet(parquet_path)

def load_gold_employee(parquet_path):
    return pd.read_parquet(parquet_path)

def load_gold_department(parquet_path):
    return pd.read_parquet(parquet_path)
