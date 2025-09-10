from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np

REQ_RAW_COLS = {
    "satisfaction_level": float,
    "last_evaluation": float,
    "number_project": (int, np.integer),
    "average_montly_hours": (int, np.integer),
    "time_spend_company": (int, np.integer),
    "Work_accident": (int, np.integer),
    "promotion_last_5years": (int, np.integer),
    "salary": str,
    "sales": str,
    "left": (int, np.integer),
}

SALARY_VALUES = {"low", "medium", "high"}

def check_columns(df: pd.DataFrame) -> list[str]:
    missing = [c for c in REQ_RAW_COLS if c not in df.columns]
    return [f"missing: {c}" for c in missing]

def check_types(df: pd.DataFrame) -> list[str]:
    issues = []
    for col, typ in REQ_RAW_COLS.items():
        if col not in df.columns:
            continue
        s = df[col]
        if col in {"salary", "sales"}:
            if not pd.api.types.is_object_dtype(s):
                issues.append(f"{col}: bad dtype {s.dtype}")
        else:
            if not pd.api.types.is_numeric_dtype(s):
                issues.append(f"{col}: bad dtype {s.dtype}")
    return issues

def check_ranges(df: pd.DataFrame) -> list[str]:
    issues = []
    if "satisfaction_level" in df:
        bad = ~df["satisfaction_level"].between(0, 1)
        if bad.any(): issues.append(f"satisfaction_level: {int(bad.sum())} bad")
    if "last_evaluation" in df:
        bad = ~df["last_evaluation"].between(0, 1)
        if bad.any(): issues.append(f"last_evaluation: {int(bad.sum())} bad")
    if "number_project" in df:
        bad = ~df["number_project"].between(0, 20)
        if bad.any(): issues.append(f"number_project: {int(bad.sum())} bad")
    if "average_montly_hours" in df:
        bad = ~df["average_montly_hours"].between(0, 400)
        if bad.any(): issues.append(f"average_montly_hours: {int(bad.sum())} bad")
    if "time_spend_company" in df:
        bad = ~df["time_spend_company"].between(0, 50)
        if bad.any(): issues.append(f"time_spend_company: {int(bad.sum())} bad")
    if "Work_accident" in df:
        bad = ~df["Work_accident"].isin([0,1])
        if bad.any(): issues.append(f"Work_accident: {int(bad.sum())} bad")
    if "promotion_last_5years" in df:
        bad = ~df["promotion_last_5years"].isin([0,1])
        if bad.any(): issues.append(f"promotion_last_5years: {int(bad.sum())} bad")
    if "salary" in df:
        bad = ~df["salary"].isin(SALARY_VALUES)
        if bad.any(): issues.append(f"salary: {int(bad.sum())} bad")
    return issues

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_csv", type=Path, default=Path("data/raw/HR_comma_sep.csv"))
    args = ap.parse_args()

    if not args.raw_csv.exists():
        print(f"missing: {args.raw_csv}")
        sys.exit(2)

    df = pd.read_csv(args.raw_csv)
    errs = check_columns(df) + check_types(df) + check_ranges(df)

    if errs:
        for e in errs: print(f"ERROR: {e}")
        print(f"validation failed: {len(errs)} errors")
        sys.exit(1)
    else:
        print(f"OK: {args.raw_csv}")

if __name__ == "__main__":
    main()
