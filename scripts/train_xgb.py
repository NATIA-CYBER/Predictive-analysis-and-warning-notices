#!/usr/bin/env python3

import warnings
from pathlib import Path
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
MODELS_DIR = REPO / "models"
FIGS_DIR = REPO / "figs"

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("[TRAIN-XGB] Starting XGBoost training...")
    
    # paths
    gold_parquet = DATA_DIR / "gold" / "hr_emp_gold.parquet"
    model_path = MODELS_DIR / "xgb.json"
    convergence_plot_path = FIGS_DIR / "xgb_convergence.png"
    
    if not gold_parquet.exists():
        raise SystemExit(f"Missing gold data: {gold_parquet}. Run feature engineering first.")
    
    df = pd.read_parquet(gold_parquet)
    
    # features and target
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
    
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # train model
    model = xgb.XGBClassifier(
        objective='binary:logistic', 
        eval_metric='aucpr', 
        use_label_encoder=False, 
        n_estimators=100
    )
    eval_set = [(X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    
    # convergence plot
    results = model.evals_result()
    plt.figure(figsize=(10, 6))
    plt.plot(results['validation_0']['aucpr'], label='Validation PR-AUC')
    plt.xlabel('Boosting Iteration')
    plt.ylabel('PR-AUC')
    plt.title('XGBoost Convergence')
    plt.legend()
    convergence_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(convergence_plot_path)
    plt.close()
    print(f"[TRAIN-XGB] Saved convergence plot to {convergence_plot_path}")
    
    # save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    print(f"[TRAIN-XGB] Saved trained model to {model_path}")
    
    print("[TRAIN-XGB] Done.")


if __name__ == "__main__":
    main()
