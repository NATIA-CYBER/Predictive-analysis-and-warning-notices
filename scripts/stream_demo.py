#!/usr/bin/env python3

from pathlib import Path
import warnings
import json
import time
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
MODELS_DIR = REPO / "models"
RESULTS_DIR = REPO / "results" / "experiments"

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("[STREAM] Starting stream demo…")

    df_emp = pd.read_parquet(DATA_DIR / "gold" / "hr_emp_gold.parquet")
    df_dept = pd.read_parquet(DATA_DIR / "gold" / "hr_dept_gold.parquet")
    
    xgb_model = xgb.Booster()
    xgb_model.load_model(MODELS_DIR / "xgb.json")
    iforest_model = joblib.load(MODELS_DIR / "iforest_dept.joblib")
    
    with open(RESULTS_DIR / "last_metrics.json", 'r') as f:
        optimal_threshold = json.load(f)['optimal_threshold']

    dept_cols = [col for col in df_emp.columns if col.startswith('dept_')]
    df_emp['sales'] = df_emp[dept_cols].idxmax(axis=1).str.replace('dept_', '')

    xgb_features = [col for col in df_emp.columns if col.startswith('dept_') or col in [
        'satisfaction_level', 'last_evaluation', 'number_project',
        'average_montly_hours', 'time_spend_company', 'Work_accident',
        'promotion_last_5years', 'salary_ord', 'satisfaction_x_eval',
        'hours_x_projects'
    ]]
    dmatrix = xgb.DMatrix(df_emp[xgb_features].fillna(0))
    df_emp['xgb_score'] = xgb_model.predict(dmatrix)

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

    df_emp['fused_score'] = 0.7 * df_emp['xgb_score'] + 0.3 * df_emp['iforest_score']

    dept_week_scores = df_emp.groupby(['sales', 'weekly_ts']).agg({
        'fused_score': 'mean'
    }).reset_index()

    weeks = sorted(dept_week_scores['weekly_ts'].unique())
    
    print(f"[STREAM] Simulating {len(weeks)} weeks of data...")
    print(f"[STREAM] Threshold: {optimal_threshold:.4f}")
    print("[STREAM] Starting stream (press Ctrl+C to stop):")
    
    try:
        for week in weeks:
            week_data = dept_week_scores[dept_week_scores['weekly_ts'] == week]
            
            for _, row in week_data.iterrows():
                notice = {
                    "dept": row['sales'],
                    "week": row['weekly_ts'].strftime('%Y-%m-%d'),
                    "score": round(float(row['fused_score']), 4),
                    "notice": bool(row['fused_score'] > optimal_threshold)
                }
                
                print(json.dumps(notice))
                time.sleep(0.1)
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n[STREAM] Stream stopped by user")
    
    print("[STREAM] Done.")

if __name__ == "__main__":
    main()
