#!/usr/bin/env python3

from pathlib import Path
import warnings

import pandas as pd
import numpy as np

# --- Paths ---
REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
RESULTS_DIR = REPO / "results" / "experiments"

def calculate_dpi(df_dept: pd.DataFrame) -> pd.DataFrame:
    
    # Aggregate by department (across all weeks)
    dept_agg = df_dept.groupby('sales').agg({
        'attrition_rate': 'mean',
        'average_montly_hours_mean': 'mean', 
        'satisfaction_level_mean': 'mean',
        'headcount': 'mean',
        'violation_density': 'mean'
    }).reset_index()
    
    # Calculate DPI components (higher is better)
    # 1. Retention score (inverse of attrition rate)
    dept_agg['retention_score'] = 1 - dept_agg['attrition_rate']
    
    # 2. Sustainable workload (penalize excessive hours, reward satisfaction)
    max_hours = dept_agg['average_montly_hours_mean'].max()
    dept_agg['workload_score'] = (
        (1 - dept_agg['average_montly_hours_mean'] / max_hours) * 0.5 +
        dept_agg['satisfaction_level_mean'] * 0.5
    )
    
    # 3. Policy compliance (inverse of violation density)
    max_violations = dept_agg['violation_density'].max()
    if max_violations > 0:
        dept_agg['compliance_score'] = 1 - (dept_agg['violation_density'] / max_violations)
    else:
        dept_agg['compliance_score'] = 1.0
    
    # Calculate overall DPI (weighted average)
    dept_agg['dpi'] = (
        dept_agg['retention_score'] * 0.4 +
        dept_agg['workload_score'] * 0.4 +
        dept_agg['compliance_score'] * 0.2
    )
    
    # Rank departments
    dept_agg = dept_agg.sort_values('dpi', ascending=False).reset_index(drop=True)
    dept_agg['rank'] = range(1, len(dept_agg) + 1)
    
    return dept_agg[['rank', 'sales', 'dpi', 'retention_score', 'workload_score', 'compliance_score', 'headcount']]

def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("[DPI] Starting DPI leaderboard generation…")

    # 1) Load department gold data
    dept_gold_parquet = DATA_DIR / "gold" / "hr_dept_gold.parquet"
    if not dept_gold_parquet.exists():
        raise SystemExit(f"Missing department gold data: {dept_gold_parquet}")
    
    df_dept = pd.read_parquet(dept_gold_parquet)
    
    # 2) Calculate DPI
    dpi_leaderboard = calculate_dpi(df_dept)
    
    # 3) Save results
    output_csv = RESULTS_DIR / "dpi_leaderboard.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    dpi_leaderboard.to_csv(output_csv, index=False)
    
    print(f"[DPI] Saved DPI leaderboard to {output_csv}")
    print("\nTop 3 Departments:")
    print(dpi_leaderboard.head(3)[['rank', 'sales', 'dpi']].to_string(index=False))
    print("[DPI] Done.")

if __name__ == "__main__":
    main()
