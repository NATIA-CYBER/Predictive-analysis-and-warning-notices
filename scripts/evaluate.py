#!/usr/bin/env python3

import json
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import confusion_matrix

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
MODELS_DIR = REPO / "models"
RESULTS_DIR = REPO / "results" / "experiments"

def main(args):
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("[EVAL] Starting evaluation...")
    
    # load data and models
    df_emp = pd.read_parquet(DATA_DIR / "gold" / "hr_emp_gold.parquet")
    df_dept = pd.read_parquet(DATA_DIR / "gold" / "hr_dept_gold.parquet")
    
    xgb_model = xgb.Booster()
    xgb_model.load_model(MODELS_DIR / "xgb.json")
    
    iforest_model = joblib.load(MODELS_DIR / "iforest_dept.joblib")
    
    # recreate sales column
    dept_cols = [col for col in df_emp.columns if col.startswith('dept_')]
    df_emp['sales'] = df_emp[dept_cols].idxmax(axis=1).str.replace('dept_', '')
    
    # XGBoost scores
    xgb_features = [col for col in df_emp.columns if col.startswith('dept_') or col in [
        'satisfaction_level', 'last_evaluation', 'number_project',
        'average_montly_hours', 'time_spend_company', 'Work_accident',
        'promotion_last_5years', 'salary_ord', 'satisfaction_x_eval',
        'hours_x_projects'
    ]]
    dmatrix = xgb.DMatrix(df_emp[xgb_features].fillna(0))
    df_emp['xgb_score'] = xgb_model.predict(dmatrix)
    
    # Isolation Forest scores
    iforest_features = [c for c in df_dept.columns if c.endswith('_lag1') or c in [
        'leavers', 'attrition_rate', 'headcount', 'satisfaction_level_mean',
        'average_montly_hours_mean', 'time_spend_company_mean',
        'v1_overtime_sum', 'v2_overload_satisfaction_sum', 'v3_stagnation_sum',
        'v4_post_accident_overwork_sum', 'violation_density'
    ]]
    
    X_iforest = df_dept[iforest_features].fillna(0)
    iforest_scores = iforest_model.decision_function(X_iforest)
    df_dept['iforest_score'] = 1 - (iforest_scores - iforest_scores.min()) / (iforest_scores.max() - iforest_scores.min())
    
    df_emp = df_emp.merge(df_dept[['sales', 'weekly_ts', 'iforest_score']], on=['sales', 'weekly_ts'], how='left')
    df_emp['iforest_score'] = df_emp['iforest_score'].fillna(0.5)
    
    # fuse scores
    df_emp['fused_score'] = 0.7 * df_emp['xgb_score'] + 0.3 * df_emp['iforest_score']
    
    # find optimal threshold
    thresholds = np.linspace(0, 1, 100)
    costs = []
    for t in thresholds:
        y_pred = (df_emp['fused_score'] > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(df_emp['left'], y_pred).ravel()
        cost = 10 * fn + 1 * fp
        costs.append(cost)
    
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    
    # save result
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({'optimal_threshold': optimal_threshold}, f, indent=4)
    
    print(f"[EVAL] Optimal cost-aware threshold: {optimal_threshold:.4f}")
    print(f"[EVAL] Saved threshold to {output_path}")
    print("[EVAL] Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RESULTS_DIR / "last_metrics.json"))
    args = parser.parse_args()
    main(args)
