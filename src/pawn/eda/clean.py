#!/usr/bin/env python3

import pandas as pd
import numpy as np

def clean_basic(df):
    df = df.copy()
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def synthesize_weekly_ts(df, rows_per_week=500):
    n_weeks = len(df) // rows_per_week + 1
    week_starts = pd.date_range('2024-01-01', periods=n_weeks, freq='W-MON')
    
    weekly_ts = []
    for i, week_start in enumerate(week_starts):
        start_idx = i * rows_per_week
        end_idx = min((i + 1) * rows_per_week, len(df))
        weekly_ts.extend([week_start] * (end_idx - start_idx))
    
    df['weekly_ts'] = weekly_ts[:len(df)]
    return df

def validate_columns(df, required_cols):
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return True
