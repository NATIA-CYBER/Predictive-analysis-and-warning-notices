#!/usr/bin/env python3

from pathlib import Path
import argparse
import warnings

import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# --- Paths ---
REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
MODELS_DIR = REPO / "models"

def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("[TRAIN-IFOREST] Starting Isolation Forest training…")

    # 1) IO paths
    dept_gold_parquet = DATA_DIR / "gold" / "hr_dept_gold.parquet"
    model_path = MODELS_DIR / "iforest_dept.joblib"

    # 2) Load gold data
    if not dept_gold_parquet.exists():
        raise SystemExit(f"Missing department gold data: {dept_gold_parquet}. Run feature engineering first.")
    df_dept = pd.read_parquet(dept_gold_parquet)

    # 3) Select features for the model
    features = [c for c in df_dept.columns if c.endswith('_lag1') or c in [
        'leavers', 'attrition_rate', 'headcount', 'satisfaction_level_mean',
        'average_montly_hours_mean', 'time_spend_company_mean',
        'v1_overtime_sum', 'v2_overload_satisfaction_sum', 'v3_stagnation_sum',
        'v4_post_accident_overwork_sum', 'violation_density'
    ]]
    
    X = df_dept[features].fillna(0) # Fill NaNs from lag operation

    # 4) Train model
    model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    model.fit(X)

    # 5) Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[TRAIN-IFOREST] Saved trained model to {model_path}")

    print("[TRAIN-IFOREST] Done.")


if __name__ == "__main__":
    main()
