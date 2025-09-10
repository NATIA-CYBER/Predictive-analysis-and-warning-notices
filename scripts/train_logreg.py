#!/usr/bin/env python3

from pathlib import Path
import warnings

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
import joblib

# --- Paths ---
REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
MODELS_DIR = REPO / "models"
FIGS_DIR = REPO / "figs"

def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("[TRAIN-LOGREG] Starting Logistic Regression training…")

    # 1) IO paths
    gold_parquet = DATA_DIR / "gold" / "hr_emp_gold.parquet"
    model_path = MODELS_DIR / "logreg.joblib"
    calibration_plot_path = FIGS_DIR / "logreg_calibration.png"

    # 2) Load gold data
    if not gold_parquet.exists():
        raise SystemExit(f"Missing gold data: {gold_parquet}. Run feature engineering first.")
    df = pd.read_parquet(gold_parquet)

    # 3) Define features and target
    target = "left"
    features = [
        'satisfaction_level', 'last_evaluation', 'number_project',
        'average_montly_hours', 'time_spend_company', 'Work_accident',
        'promotion_last_5years', 'salary_ord'
    ]
    # Add one-hot encoded department columns
    features.extend([col for col in df.columns if col.startswith('dept_')])

    X = df[features].fillna(0) # Simple imputation for now
    y = df[target]

    # 4) Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5) Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6) Train model
    model = LogisticRegression(random_state=42, solver='liblinear')
    model.fit(X_train_scaled, y_train)

    # 7) Plot calibration curve
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_pred_proba, n_bins=10)

    plt.figure(figsize=(8, 8))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label='Logistic Regression')
    plt.plot([0, 1], [0, 1], "k:", label='Perfectly calibrated')
    plt.xlabel('Mean predicted value')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Plot')
    plt.legend()
    calibration_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(calibration_plot_path)
    plt.close()
    print(f"[TRAIN-LOGREG] Saved calibration plot to {calibration_plot_path}")

    # 8) Save model and scaler
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, model_path)
    print(f"[TRAIN-LOGREG] Saved trained model to {model_path}")

    print("[TRAIN-LOGREG] Done.")


if __name__ == "__main__":
    main()
