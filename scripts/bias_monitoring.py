#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score
import json
import warnings

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
RESULTS_DIR = REPO / "results" / "experiments"

def calculate_demographic_parity(y_true, y_pred, groups):
    results = {}
    overall_positive_rate = y_pred.mean()
    
    for group in np.unique(groups):
        mask = groups == group
        group_positive_rate = y_pred[mask].mean()
        parity_diff = abs(group_positive_rate - overall_positive_rate)
        results[group] = {
            'positive_rate': float(group_positive_rate),
            'parity_difference': float(parity_diff),
            'sample_size': int(mask.sum())
        }
    
    return results

def calculate_equalized_odds(y_true, y_pred, groups):
    results = {}
    
    for group in np.unique(groups):
        mask = groups == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        
        if len(np.unique(y_true_group)) < 2:
            continue
            
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        results[group] = {
            'true_positive_rate': float(tpr),
            'false_positive_rate': float(fpr),
            'accuracy': float(accuracy_score(y_true_group, y_pred_group))
        }
    
    return results

def assess_model_fairness(df_emp, model_predictions, group_column='sales'):
    y_true = df_emp['left'].values
    y_pred = (model_predictions > 0.5).astype(int)
    groups = df_emp[group_column].values
    
    fairness_report = {
        'demographic_parity': calculate_demographic_parity(y_true, y_pred, groups),
        'equalized_odds': calculate_equalized_odds(y_true, y_pred, groups),
        'overall_metrics': {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'positive_rate': float(y_pred.mean()),
            'base_rate': float(y_true.mean())
        }
    }
    
    parity_diffs = [v['parity_difference'] for v in fairness_report['demographic_parity'].values()]
    fairness_report['max_parity_difference'] = float(max(parity_diffs))
    fairness_report['fairness_violation'] = fairness_report['max_parity_difference'] > 0.05
    
    return fairness_report

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("[BIAS-MONITOR] Starting fairness assessment...")
    
    # Load data
    df_emp = pd.read_parquet(DATA_DIR / "gold" / "hr_emp_gold.parquet")
    
    benchmark_path = RESULTS_DIR / "benchmark_enhanced.csv"
    if not benchmark_path.exists():
        raise SystemExit("Run benchmark first to generate predictions")
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    
    features = [col for col in df_emp.columns if col.startswith('dept_') or col in [
        'satisfaction_level', 'last_evaluation', 'number_project',
        'average_montly_hours', 'time_spend_company', 'Work_accident',
        'promotion_last_5years', 'salary_ord'
    ]]
    
    X = df_emp[features].fillna(0)
    y = df_emp['left']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)[:, 1]
    
    test_df = df_emp.iloc[X_test.index].copy()
    
    dept_fairness = assess_model_fairness(test_df, predictions, 'sales')
    
    salary_fairness = assess_model_fairness(test_df, predictions, 'salary')
    
    bias_report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'department_fairness': dept_fairness,
        'salary_fairness': salary_fairness,
        'recommendations': []
    }
    
    if dept_fairness['fairness_violation']:
        bias_report['recommendations'].append(
            f"ALERT: Department bias detected. Max parity difference: {dept_fairness['max_parity_difference']:.3f}"
        )
    
    if salary_fairness['fairness_violation']:
        bias_report['recommendations'].append(
            f"ALERT: Salary-based bias detected. Max parity difference: {salary_fairness['max_parity_difference']:.3f}"
        )
    
    if not bias_report['recommendations']:
        bias_report['recommendations'].append("No significant bias detected. Model meets fairness criteria.")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "bias_assessment.json", 'w') as f:
        json.dump(bias_report, f, indent=2)
    
    print(f"[BIAS-MONITOR] Department fairness violation: {dept_fairness['fairness_violation']}")
    print(f"[BIAS-MONITOR] Salary fairness violation: {salary_fairness['fairness_violation']}")
    print(f"[BIAS-MONITOR] Report saved to {RESULTS_DIR / 'bias_assessment.json'}")
    print("[BIAS-MONITOR] Done.")

if __name__ == "__main__":
    main()
