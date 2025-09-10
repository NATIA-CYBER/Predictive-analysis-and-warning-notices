#!/usr/bin/env python3

from pathlib import Path
import warnings
import pandas as pd

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"

def build_employee_features(df):
    result = df.copy()
    salary_map = {"low": 1, "medium": 2, "high": 3}
    result["salary_ord"] = result["salary"].map(salary_map)
    result["satisfaction_x_eval"] = result["satisfaction_level"] * result["last_evaluation"]
    result["hours_x_projects"] = result["average_montly_hours"] * result["number_project"]
    result = pd.get_dummies(result, columns=["sales"], prefix="dept")
    return result

def create_dept_week_features(df):
    dept_df = df.copy()
    p95_hours = dept_df["average_montly_hours"].quantile(0.95)
    p75_projects = dept_df["number_project"].quantile(0.75)
    p25_satisfaction = dept_df["satisfaction_level"].quantile(0.25)
    p75_tenure = dept_df["time_spend_company"].quantile(0.75)
    median_hours = dept_df["average_montly_hours"].median()
    
    dept_df["v1_overtime"] = (dept_df["average_montly_hours"] > p95_hours).astype(int)
    dept_df["v2_overload_satisfaction"] = (
        (dept_df["number_project"] >= p75_projects) & 
        (dept_df["satisfaction_level"] < p25_satisfaction)
    ).astype(int)
    dept_df["v3_stagnation"] = (
        (dept_df["time_spend_company"] >= p75_tenure) & 
        (dept_df["promotion_last_5years"] == 0) & 
        (dept_df["average_montly_hours"] > median_hours)
    ).astype(int)
    dept_df["v4_post_accident_overwork"] = (
        (dept_df["Work_accident"] == 1) & 
        (dept_df["average_montly_hours"] > p95_hours)
    ).astype(int)
    
    agg_dict = {
        "left": ["sum", "mean"],
        "satisfaction_level": ["mean"],
        "average_montly_hours": ["mean"],
        "time_spend_company": ["mean"],
        "v1_overtime": ["sum"],
        "v2_overload_satisfaction": ["sum"],
        "v3_stagnation": ["sum"],
        "v4_post_accident_overwork": ["sum"],
        "EmployeeID": ["count"]
    }
    
    dept_week = dept_df.groupby(["sales", "weekly_ts"]).agg(agg_dict)
    dept_week.columns = ["_".join(col).strip() for col in dept_week.columns.values]
    dept_week = dept_week.rename(columns={
        "left_sum": "leavers",
        "left_mean": "attrition_rate",
        "EmployeeID_count": "headcount"
    })
    
    violation_cols = [c for c in dept_week.columns if c.startswith('v') and c.endswith('_sum')]
    dept_week["violation_density"] = dept_week[violation_cols].sum(axis=1) / dept_week["headcount"]
    
    for col in violation_cols + ["violation_density", "attrition_rate"]:
        dept_week[f"{col}_lag1"] = dept_week.groupby("sales")[col].shift(1)
    
    return dept_week.reset_index()

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("[FEAT] Starting feature build...")
    
    silver_parquet = data_dir / "silver" / "hr_silver.parquet"
    emp_gold_parquet = data_dir / "gold" / "hr_emp_gold.parquet"
    dept_gold_parquet = data_dir / "gold" / "hr_dept_gold.parquet"
    
    if not silver_parquet.exists():
        raise SystemExit(f"Missing silver data: {silver_parquet}. Run EDA first.")
    
    df = pd.read_parquet(silver_parquet)
    df['EmployeeID'] = range(len(df))
    
    df_dept_week = create_dept_week_features(df)
    dept_gold_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_dept_week.to_parquet(dept_gold_parquet, index=False)
    print(f"[FEAT] Saved department-week gold data → {dept_gold_parquet}")
    
    df_merged = pd.merge(df, df_dept_week, on=["sales", "weekly_ts"], how="left")
    df_emp_gold = build_employee_features(df_merged)
    emp_gold_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_emp_gold.to_parquet(emp_gold_parquet, index=False)
    print(f"[FEAT] Saved employee-level gold data → {emp_gold_parquet}")
    
    print("[FEAT] Done.")

if __name__ == "__main__":
    main()
