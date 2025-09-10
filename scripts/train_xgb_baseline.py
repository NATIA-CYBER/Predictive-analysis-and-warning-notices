#!/usr/bin/env python3

import warnings
from pathlib import Path
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
MODELS_DIR = REPO / "models" / "baseline"
FIGS_DIR = REPO / "figs"

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("[TRAIN-XGB-BASELINE] Starting baseline XGBoost training...")
    
    gold_parquet = DATA_DIR / "gold" / "hr_emp_gold.parquet"
    model_path = MODELS_DIR / "xgb_baseline.json"
    
    if not gold_parquet.exists():
        raise SystemExit(f"Missing gold data: {gold_parquet}. Run feature engineering first.")
    
    df = pd.read_parquet(gold_parquet)
    
    target = "left"
    features = [
        'satisfaction_level', 'last_evaluation', 'number_project',
        'average_montly_hours', 'time_spend_company', 'Work_accident',
        'promotion_last_5years', 'salary_ord', 'satisfaction_x_eval',
        'hours_x_projects'
    ]
    features.extend([col for col in df.columns if col.startswith('dept_')])
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # simple baseline model
    model = xgb.XGBClassifier(
        objective='binary:logistic', 
        eval_metric='aucpr', 
        use_label_encoder=False, 
        n_estimators=100,
        random_state=42
    )
    eval_set = [(X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    print(f"[TRAIN-XGB-BASELINE] Saved baseline model to {model_path}")
    print("[TRAIN-XGB-BASELINE] Done.")


if __name__ == "__main__":
    main()
