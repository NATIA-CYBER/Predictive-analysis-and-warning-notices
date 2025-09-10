#!/usr/bin/env python3

import json
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
MODELS_DIR = REPO / "models"
RESULTS_DIR = REPO / "results" / "experiments"
FIGS_DIR = REPO / "figs"

def plot_cost_curve(y_true, score, out_png, fn_w=10.0, fp_w=1.0):
    """
    Sweep τ in [0,1], compute C = fn_w*FN + fp_w*FP; save a small figure and return best τ.
    """
    taus = np.linspace(0.0, 1.0, 101)
    costs = []
    fps = []
    fns = []
    y = y_true.astype(int)

    for t in taus:
        yhat = (score >= t).astype(int)
        fp = int(((yhat == 1) & (y == 0)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())
        fps.append(fp)
        fns.append(fn)
        costs.append(fn_w * fn + fp_w * fp)

    best_idx = int(np.argmin(costs))
    best_tau = float(taus[best_idx])

    # plot
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(taus, costs, label="C = 10·FN + 1·FP")
    plt.axvline(best_tau, linestyle="--")
    plt.scatter([best_tau], [costs[best_idx]])
    plt.xlabel("threshold (τ)")
    plt.ylabel("total cost")
    plt.title("Cost sweep")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    return best_tau

def train_fusion_model(df_emp):
    # prepare meta-features for fusion
    meta_features = ['xgb_score', 'iforest_score', 'violation_density']
    X_meta = df_emp[meta_features].fillna(0)
    y_meta = df_emp['left']
    
    # scale features
    scaler = MinMaxScaler()
    X_meta_scaled = scaler.fit_transform(X_meta)
    
    # train meta-model
    X_train, X_val, y_train, y_val = train_test_split(
        X_meta_scaled, y_meta, test_size=0.3, random_state=42, stratify=y_meta
    )
    
    fusion_model = LogisticRegression(random_state=42)
    fusion_model.fit(X_train, y_train)
    
    # get fusion scores
    fusion_scores = fusion_model.predict_proba(X_meta_scaled)[:, 1]
    
    return fusion_model, scaler, fusion_scores

def main(args):
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("[EVAL] Starting enhanced evaluation...")
    
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
    
    df_emp = df_emp.merge(df_dept[['sales', 'weekly_ts', 'iforest_score', 'violation_density']], 
                          on=['sales', 'weekly_ts'], how='left')
    df_emp['iforest_score'] = df_emp['iforest_score'].fillna(0.5)
    if 'violation_density' not in df_emp.columns:
        df_emp['violation_density'] = 0
    else:
        df_emp['violation_density'] = df_emp['violation_density'].fillna(0)
    
    # naive fusion (original)
    df_emp['fused_score'] = 0.7 * df_emp['xgb_score'] + 0.3 * df_emp['iforest_score']
    
    # learned fusion
    fusion_model, scaler, fusion_scores = train_fusion_model(df_emp)
    df_emp['fusion_lr_score'] = fusion_scores
    
    # save fusion model
    joblib.dump(fusion_model, MODELS_DIR / "fusion_lr.joblib")
    joblib.dump(scaler, MODELS_DIR / "fusion_scaler.joblib")
    
    # find optimal thresholds using cost curve plotting
    optimal_threshold_naive = plot_cost_curve(
        df_emp['left'].values, 
        df_emp['fused_score'].values, 
        FIGS_DIR / "cost_curve_naive.png", 
        fn_w=10.0, 
        fp_w=1.0
    )
    
    optimal_threshold_lr = plot_cost_curve(
        df_emp['left'].values, 
        df_emp['fusion_lr_score'].values, 
        FIGS_DIR / "cost_curve_learned.png", 
        fn_w=10.0, 
        fp_w=1.0
    )
    
    # calculate final costs
    y_pred_naive = (df_emp['fused_score'] > optimal_threshold_naive).astype(int)
    tn, fp, fn, tp = confusion_matrix(df_emp['left'], y_pred_naive).ravel()
    naive_cost = 10 * fn + 1 * fp
    
    y_pred_lr = (df_emp['fusion_lr_score'] > optimal_threshold_lr).astype(int)
    tn, fp, fn, tp = confusion_matrix(df_emp['left'], y_pred_lr).ravel()
    lr_cost = 10 * fn + 1 * fp
    
    # save results
    results = {
        'optimal_threshold': float(optimal_threshold_naive),
        'fusion_lr_threshold': float(optimal_threshold_lr),
        'naive_fusion_cost': int(naive_cost),
        'learned_fusion_cost': int(lr_cost),
        'cost_fn': "10*FN + 1*FP"
    }
    
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"[EVAL] Naive fusion threshold: {optimal_threshold_naive:.4f}")
    print(f"[EVAL] Learned fusion threshold: {optimal_threshold_lr:.4f}")
    print(f"[EVAL] Cost improvement: {naive_cost - lr_cost:.0f}")
    print(f"[EVAL] Saved results to {output_path}")
    print("[EVAL] Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RESULTS_DIR / "last_metrics.json"))
    args = parser.parse_args()
    main(args)
