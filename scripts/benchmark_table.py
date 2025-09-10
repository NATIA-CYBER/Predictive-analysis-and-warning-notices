#!/usr/bin/env python3
"""
Creates the final benchmark table for the PAWN project.

- Loads gold data and trained models.
- Makes predictions with both models.
- Fuses the scores.
- Calculates precision, recall, and F1-score at different thresholds.
- Saves the benchmark table to a CSV file.

Run:
    python scripts/benchmark_table.py
"""

from pathlib import Path
import argparse
import warnings

import pandas as pd
import joblib
from sklearn.metrics import precision_recall_fscore_support
import json
import xgboost as xgb

# --- Paths ---
REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
MODELS_DIR = REPO / "models"
RESULTS_DIR = REPO / "results" / "experiments"

def main(args) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("[BENCH] Starting benchmark table generation…")

    # 1) IO paths
    emp_gold_parquet = DATA_DIR / "gold" / "hr_emp_gold.parquet"
    dept_gold_parquet = DATA_DIR / "gold" / "hr_dept_gold.parquet"
    xgb_model_path = MODELS_DIR / "xgb.json"
    logreg_model_path = MODELS_DIR / "logreg.joblib"
    iforest_model_path = MODELS_DIR / "iforest_dept.joblib"
    thresholds_path = RESULTS_DIR / "last_metrics.json"
    output_csv = Path(args.out)

    # 2) Load data and models
    df_emp = pd.read_parquet(emp_gold_parquet)
    df_dept = pd.read_parquet(dept_gold_parquet)
    
    xgb_model = xgb.Booster()
    xgb_model.load_model(xgb_model_path)
    
    logreg_bundle = joblib.load(logreg_model_path)
    logreg_model = logreg_bundle['model']
    logreg_scaler = logreg_bundle['scaler']
    
    iforest_model = joblib.load(iforest_model_path)

    with open(thresholds_path, 'r') as f:
        optimal_threshold = json.load(f)['optimal_threshold']

    # 3) Recreate 'sales' column in employee data
    dept_cols = [col for col in df_emp.columns if col.startswith('dept_')]
    df_emp['sales'] = df_emp[dept_cols].idxmax(axis=1).str.replace('dept_', '')

    # 4) Get scores for each model
    # --- XGBoost
    xgb_features = [col for col in df_emp.columns if col.startswith('dept_') or col in [
        'satisfaction_level', 'last_evaluation', 'number_project',
        'average_montly_hours', 'time_spend_company', 'Work_accident',
        'promotion_last_5years', 'salary_ord', 'satisfaction_x_eval',
        'hours_x_projects'
    ]]
    dmatrix = xgb.DMatrix(df_emp[xgb_features].fillna(0))
    df_emp['xgb_score'] = xgb_model.predict(dmatrix)

    # --- Logistic Regression
    logreg_features = [col for col in df_emp.columns if col.startswith('dept_') or col in [
        'satisfaction_level', 'last_evaluation', 'number_project',
        'average_montly_hours', 'time_spend_company', 'Work_accident',
        'promotion_last_5years', 'salary_ord'
    ]]
    X_logreg_scaled = logreg_scaler.transform(df_emp[logreg_features].fillna(0))
    df_emp['logreg_score'] = logreg_model.predict_proba(X_logreg_scaled)[:, 1]

    # --- Isolation Forest
    iforest_features = [c for c in df_dept.columns if c.endswith('_lag1') or c in [
        'leavers', 'attrition_rate', 'headcount', 'satisfaction_level_mean',
        'average_montly_hours_mean', 'time_spend_company_mean',
        'v1_overtime_sum', 'v2_overload_satisfaction_sum', 'v3_stagnation_sum',
        'v4_post_accident_overwork_sum', 'violation_density'
    ]]
    df_iforest = df_dept.set_index(['sales', 'weekly_ts'])
    iforest_scores = iforest_model.decision_function(df_iforest[iforest_features].fillna(0))
    df_iforest['iforest_score'] = 1 - (iforest_scores - iforest_scores.min()) / (iforest_scores.max() - iforest_scores.min())
    
    df_emp = df_emp.merge(df_iforest[['iforest_score']], on=['sales', 'weekly_ts'], how='left')
    df_emp['iforest_score'] = df_emp['iforest_score'].fillna(0.5)

    # --- Fused Score
    df_emp['fused_score'] = 0.7 * df_emp['xgb_score'] + 0.3 * df_emp['iforest_score']

    # 4) Calculate metrics for each model at the optimal threshold
    y_true = df_emp['left']
    models_to_bench = ['logreg_score', 'xgb_score', 'iforest_score', 'fused_score']
    results = []

    for model_name in models_to_bench:
        y_pred = (df_emp[model_name] > optimal_threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        results.append({
            'model': model_name.replace('_score', '').capitalize(),
            'threshold': optimal_threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

    benchmark_df = pd.DataFrame(results)

    # 5) Save benchmark table
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_csv(output_csv, index=False)
    print(f"[BENCH] Saved benchmark table to {output_csv}")
    print(benchmark_df.to_string())
    print("[BENCH] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RESULTS_DIR / "benchmark.csv"))
    args = parser.parse_args()
    main(args)
