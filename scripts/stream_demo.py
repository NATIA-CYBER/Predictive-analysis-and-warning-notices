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



    #!/usr/bin/env python3
from pathlib import Path
import warnings
import json
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from pawn.genai import summarize

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
MODELS_DIR = REPO / "models"
RESULTS_DIR = REPO / "results" / "experiments"


def minmax_scale(series: pd.Series) -> pd.Series:
    lo = float(series.min())
    hi = float(series.max())

    if hi - lo == 0:
        return pd.Series(np.full(len(series), 0.5), index=series.index)

    return (series - lo) / (hi - lo)


def infer_top_drivers(row: pd.Series) -> list[str]:
    drivers: list[str] = []

    if row.get("attrition_rate", 0) >= 0.25:
        drivers.append("attrition_rate")

    if row.get("average_montly_hours_mean", 0) >= 210:
        drivers.append("average_montly_hours_mean")

    if row.get("satisfaction_level_mean", 1) <= 0.45:
        drivers.append("satisfaction_level_mean")

    if row.get("time_spend_company_mean", 0) >= 4:
        drivers.append("time_spend_company_mean")

    if row.get("v1_overtime_sum", 0) > 0:
        drivers.append("v1_overtime_sum")

    if row.get("v2_overload_satisfaction_sum", 0) > 0:
        drivers.append("v2_overload_satisfaction_sum")

    if row.get("v3_stagnation_sum", 0) > 0:
        drivers.append("v3_stagnation_sum")

    if row.get("v4_post_accident_overwork_sum", 0) > 0:
        drivers.append("v4_post_accident_overwork_sum")

    if row.get("violation_density", 0) > 0:
        drivers.append("violation_density")

    if not drivers:
        drivers.append("fused_score_pattern")

    return drivers[:5]


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("[STREAM] Starting stream demo...")

    emp_path = DATA_DIR / "gold" / "hr_emp_gold.parquet"
    dept_path = DATA_DIR / "gold" / "hr_dept_gold.parquet"
    xgb_path = MODELS_DIR / "xgb.json"
    iforest_path = MODELS_DIR / "iforest_dept.joblib"
    metrics_path = RESULTS_DIR / "last_metrics.json"
    alerts_out_path = RESULTS_DIR / "genai_alerts.jsonl"

    if not emp_path.exists():
        raise SystemExit(f"Missing employee gold data: {emp_path}")
    if not dept_path.exists():
        raise SystemExit(f"Missing department gold data: {dept_path}")
    if not xgb_path.exists():
        raise SystemExit(f"Missing XGBoost model: {xgb_path}")
    if not iforest_path.exists():
        raise SystemExit(f"Missing Isolation Forest model: {iforest_path}")
    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics file: {metrics_path}")

    df_emp = pd.read_parquet(emp_path)
    df_dept = pd.read_parquet(dept_path)

    xgb_model = xgb.Booster()
    xgb_model.load_model(xgb_path)

    iforest_model = joblib.load(iforest_path)

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    optimal_threshold = metrics.get("optimal_threshold", metrics.get("tau"))
    if optimal_threshold is None:
        raise SystemExit("Could not find optimal threshold in last_metrics.json")

    dept_cols = [col for col in df_emp.columns if col.startswith("dept_")]
    if dept_cols:
        df_emp["sales"] = df_emp[dept_cols].idxmax(axis=1).str.replace("dept_", "", regex=False)

    xgb_features = [
        col
        for col in df_emp.columns
        if col.startswith("dept_")
        or col in [
            "satisfaction_level",
            "last_evaluation",
            "number_project",
            "average_montly_hours",
            "time_spend_company",
            "Work_accident",
            "promotion_last_5years",
            "salary_ord",
            "satisfaction_x_eval",
            "hours_x_projects",
        ]
    ]

    dmatrix = xgb.DMatrix(df_emp[xgb_features].fillna(0))
    df_emp["xgb_score"] = xgb_model.predict(dmatrix)

    iforest_features = [
        c
        for c in df_dept.columns
        if c.endswith("_lag1")
        or c in [
            "leavers",
            "attrition_rate",
            "headcount",
            "satisfaction_level_mean",
            "average_montly_hours_mean",
            "time_spend_company_mean",
            "v1_overtime_sum",
            "v2_overload_satisfaction_sum",
            "v3_stagnation_sum",
            "v4_post_accident_overwork_sum",
            "violation_density",
        ]
    ]

    X_iforest = df_dept[iforest_features].fillna(0)
    raw_iforest = pd.Series(iforest_model.decision_function(X_iforest), index=df_dept.index)
    df_dept["iforest_score"] = 1.0 - minmax_scale(raw_iforest)

    df_emp = df_emp.merge(
        df_dept[["sales", "weekly_ts", "iforest_score"]],
        on=["sales", "weekly_ts"],
        how="left",
    )
    df_emp["iforest_score"] = df_emp["iforest_score"].fillna(0.5)
    df_emp["fused_score"] = 0.7 * df_emp["xgb_score"] + 0.3 * df_emp["iforest_score"]

    dept_week_scores = (
        df_emp.groupby(["sales", "weekly_ts"], as_index=False)
        .agg({"fused_score": "mean"})
        .rename(columns={"fused_score": "risk_score"})
    )

    dept_context_cols = [
        col
        for col in [
            "sales",
            "weekly_ts",
            "headcount",
            "leavers",
            "attrition_rate",
            "satisfaction_level_mean",
            "average_montly_hours_mean",
            "time_spend_company_mean",
            "v1_overtime_sum",
            "v2_overload_satisfaction_sum",
            "v3_stagnation_sum",
            "v4_post_accident_overwork_sum",
            "violation_density",
        ]
        if col in df_dept.columns
    ]

    dept_week_scores = dept_week_scores.merge(
        df_dept[dept_context_cols],
        on=["sales", "weekly_ts"],
        how="left",
    )

    weeks = sorted(dept_week_scores["weekly_ts"].dropna().unique())

    print(f"[STREAM] Simulating {len(weeks)} weeks")
    print(f"[STREAM] Threshold: {optimal_threshold:.4f}")
    print(f"[STREAM] Writing alerts to: {alerts_out_path}")

    alerts_out_path.parent.mkdir(parents=True, exist_ok=True)
    if alerts_out_path.exists():
        alerts_out_path.unlink()

    try:
        for week in weeks:
            week_data = dept_week_scores[dept_week_scores["weekly_ts"] == week].copy()

            for _, row in week_data.iterrows():
                warning = bool(float(row["risk_score"]) > float(optimal_threshold))
                top_drivers = infer_top_drivers(row)

                alert = {
                    "dept": row["sales"],
                    "weekly_ts": pd.Timestamp(row["weekly_ts"]).strftime("%Y-%m-%d"),
                    "headcount": int(row["headcount"]) if pd.notna(row.get("headcount")) else None,
                    "leavers": int(row["leavers"]) if pd.notna(row.get("leavers")) else None,
                    "attrition_rate": float(row["attrition_rate"]) if pd.notna(row.get("attrition_rate")) else None,
                    "risk_score": round(float(row["risk_score"]), 4),
                    "threshold": round(float(optimal_threshold), 4),
                    "top_drivers": top_drivers,
                    "warning": warning,
                    "notes": "Simulated weekly monitoring on HR prototype data.",
                }

                summary = summarize(alert, use_llm=False)

                record = {
                    "alert": alert,
                    "summary": summary,
                }

                print(json.dumps(alert, ensure_ascii=False))
                if warning:
                    print(summary)
                    print("-" * 80)

                with open(alerts_out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                time.sleep(0.08)

            time.sleep(0.35)

    except KeyboardInterrupt:
        print("\n[STREAM] Stream stopped by user")

    print("[STREAM] Done.")


if __name__ == "__main__":
    main()
