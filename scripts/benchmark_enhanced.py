#!/usr/bin/env python3

from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import json

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
MODELS_DIR = REPO / "models"
RESULTS_DIR = REPO / "results" / "experiments"

def load_model_variant(variant_name, model_type):
    if variant_name == "baseline":
        if model_type == "xgb":
            model = xgb.Booster()
            model.load_model(MODELS_DIR / "baseline" / "xgb_baseline.json")
            return model
        elif model_type == "fusion":
            return np.load(MODELS_DIR / "baseline" / "fusion_naive_weights.npy")
    elif variant_name == "enhanced":
        if model_type == "xgb":
            model = xgb.Booster()
            model.load_model(MODELS_DIR / "enhanced" / "xgb_cost_sensitive.json")
            return model
        elif model_type == "xgb_calibrated":
            return joblib.load(MODELS_DIR / "enhanced" / "xgb_calibrated.joblib")
        elif model_type == "fusion":
            return joblib.load(MODELS_DIR / "enhanced" / "fusion_learned.joblib")

def calculate_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return 10 * fn + 1 * fp

def find_optimal_threshold(y_true, scores):
    thresholds = np.linspace(0, 1, 100)
    costs = []
    for t in thresholds:
        y_pred = (scores > t).astype(int)
        cost = calculate_cost(y_true, y_pred)
        costs.append(cost)
    
    optimal_idx = np.argmin(costs)
    return thresholds[optimal_idx], costs[optimal_idx]

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("[BENCHMARK-ENHANCED] Starting enhanced benchmark comparison...")
    
    df_emp = pd.read_parquet(DATA_DIR / "gold" / "hr_emp_gold.parquet")
    df_dept = pd.read_parquet(DATA_DIR / "gold" / "hr_dept_gold.parquet")
    
    logreg_data = joblib.load(MODELS_DIR / "baseline" / "logreg_baseline.joblib")
    if isinstance(logreg_data, dict):
        logreg_model = logreg_data['model']
        logreg_scaler = logreg_data['scaler']
    else:
        logreg_model = logreg_data
        logreg_scaler = None
    iforest_model = joblib.load(MODELS_DIR / "iforest_dept.joblib")
    
    # prepare features
    dept_cols = [col for col in df_emp.columns if col.startswith('dept_')]
    df_emp['sales'] = df_emp[dept_cols].idxmax(axis=1).str.replace('dept_', '')
    
    xgb_features = [col for col in df_emp.columns if col.startswith('dept_') or col in [
        'satisfaction_level', 'last_evaluation', 'number_project',
        'average_montly_hours', 'time_spend_company', 'Work_accident',
        'promotion_last_5years', 'salary_ord', 'satisfaction_x_eval',
        'hours_x_projects'
    ]]
    
    # isolation forest scores
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
    
    # logistic regression uses same features as XGB
    logreg_features = xgb_features
    
    # model variants to test
    variants = [
        ("XGB_Baseline", "baseline", "xgb"),
        ("XGB_CostSensitive", "enhanced", "xgb"),
        ("XGB_Calibrated", "enhanced", "xgb_calibrated"),
        ("LogReg_Baseline", "baseline", "logreg"),
        ("Fusion_Naive", "baseline", "fusion"),
        ("Fusion_Learned", "enhanced", "fusion")
    ]
    
    results = []
    
    for variant_name, model_category, model_type in variants:
        print(f"[BENCHMARK-ENHANCED] Evaluating {variant_name}...")
        
        if model_type == "logreg":
            # use original LogReg features (no interaction terms)
            original_features = [col for col in df_emp.columns if col in [
                'satisfaction_level', 'last_evaluation', 'number_project',
                'average_montly_hours', 'time_spend_company', 'Work_accident',
                'promotion_last_5years', 'salary_ord'
            ]]
            original_features.extend([col for col in df_emp.columns if col.startswith('dept_')])
            
            X_logreg = df_emp[original_features]
            if logreg_scaler:
                X_logreg = logreg_scaler.transform(X_logreg)
            scores = logreg_model.predict_proba(X_logreg)[:, 1]
        elif model_type in ["xgb", "xgb_calibrated"]:
            model = load_model_variant(model_category, model_type)
            if model_type == "xgb":
                dmatrix = xgb.DMatrix(df_emp[xgb_features].fillna(0))
                scores = model.predict(dmatrix)
            else:  # calibrated
                scores = model.predict_proba(df_emp[xgb_features].fillna(0))[:, 1]
        elif model_type == "fusion":
            if model_category == "baseline":
                weights = load_model_variant(model_category, model_type)
                # get baseline XGB scores
                xgb_baseline = load_model_variant("baseline", "xgb")
                dmatrix = xgb.DMatrix(df_emp[xgb_features].fillna(0))
                xgb_scores = xgb_baseline.predict(dmatrix)
                scores = weights[0] * xgb_scores + weights[1] * df_emp['iforest_score']
            else:  # learned fusion
                fusion_model = load_model_variant(model_category, model_type)
                scaler = joblib.load(MODELS_DIR / "enhanced" / "fusion_scaler.joblib")
                
                # get enhanced XGB scores
                xgb_enhanced = load_model_variant("enhanced", "xgb")
                dmatrix = xgb.DMatrix(df_emp[xgb_features].fillna(0))
                xgb_scores = xgb_enhanced.predict(dmatrix)
                
                meta_features = np.column_stack([
                    xgb_scores,
                    df_emp['iforest_score'],
                    np.zeros(len(df_emp))  # violation_density placeholder
                ])
                meta_features_scaled = scaler.transform(meta_features)
                scores = fusion_model.predict_proba(meta_features_scaled)[:, 1]
        
        # find optimal threshold and calculate metrics
        optimal_threshold, optimal_cost = find_optimal_threshold(df_emp['left'], scores)
        y_pred = (scores > optimal_threshold).astype(int)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            df_emp['left'], y_pred, average='binary', zero_division=0
        )
        
        # calculate cost reduction vs baseline
        if variant_name == "XGB_Baseline":
            baseline_cost = optimal_cost
            cost_reduction = 0.0
        else:
            cost_reduction = (baseline_cost - optimal_cost) / baseline_cost * 100
        
        results.append({
            'model_variant': variant_name,
            'threshold': round(optimal_threshold, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'total_cost': int(optimal_cost),
            'cost_reduction_pct': round(cost_reduction, 1)
        })
    
    # save enhanced benchmark
    df_results = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(RESULTS_DIR / "benchmark_enhanced.csv", index=False)
    
    print(f"[BENCHMARK-ENHANCED] Enhanced benchmark results:")
    print(df_results.to_string(index=False))
    print(f"[BENCHMARK-ENHANCED] Saved to {RESULTS_DIR / 'benchmark_enhanced.csv'}")
    print("[BENCHMARK-ENHANCED] Done.")


if __name__ == "__main__":
    main()
