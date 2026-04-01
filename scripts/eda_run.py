#!/usr/bin/env python3
from pathlib import Path
import argparse
import warnings
import json

import yaml
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "data" / "profile.yml"
DATA_DIR = REPO / "data"
FIGS_DIR = REPO / "figs"
RESULTS_DIR = REPO / "results" / "eda"

REQUIRED_COLS = {
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
    "sales",
    "salary",
    "left",
}


def read_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def synthesize_weekly_ts(df: pd.DataFrame, rows_per_week: int = 50) -> pd.DataFrame:
    start = pd.Timestamp("2024-01-01")
    df = df.copy()
    df["weekly_ts"] = start + pd.to_timedelta((df.index // max(rows_per_week, 1)), unit="W")
    return df


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df.to_csv(path, index=True)
    else:
        df.to_parquet(path, index=False)


def plot_class_balance(df: pd.DataFrame, out_png: Path) -> None:
    counts = df["left"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar")
    plt.title("Employee attrition: 0 = stayed, 1 = left")
    plt.xlabel("left")
    plt.ylabel("count")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def plot_attrition_rate_by_dept(df: pd.DataFrame, out_png: Path) -> None:
    rate = df.groupby("sales", dropna=False)["left"].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    rate.plot(kind="bar")
    plt.title("Attrition rate by department")
    plt.xlabel("department (sales)")
    plt.ylabel("attrition rate")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def plot_attrition_rate_by_salary(df: pd.DataFrame, out_png: Path) -> None:
    rate = df.groupby("salary", dropna=False)["left"].mean()
    order = [x for x in ["low", "medium", "high"] if x in rate.index]
    rest = [x for x in rate.index if x not in order]
    rate = rate.reindex(order + rest)

    plt.figure(figsize=(6, 4))
    rate.plot(kind="bar")
    plt.title("Attrition rate by salary")
    plt.xlabel("salary")
    plt.ylabel("attrition rate")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def plot_corr_heatmap(df: pd.DataFrame, out_png: Path) -> None:
    cols = [
        "satisfaction_level",
        "last_evaluation",
        "number_project",
        "average_montly_hours",
        "time_spend_company",
        "left",
    ]
    corr = df[cols].corr(numeric_only=True)

    plt.figure(figsize=(7, 5))
    plt.imshow(corr.values)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Correlation heatmap")
    plt.colorbar()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "dtype": df.dtypes.astype(str),
        }
    )
    return out.sort_values(["missing_count", "missing_pct"], ascending=False)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric_cols = [
        "satisfaction_level",
        "last_evaluation",
        "number_project",
        "average_montly_hours",
        "time_spend_company",
    ]

    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((df[col] < lower) | (df[col] > upper)).sum())

        rows.append(
            {
                "feature": col,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": count,
                "outlier_pct": round(100 * count / len(df), 2),
            }
        )

    return pd.DataFrame(rows)


def grouped_numeric_summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    cols = [
        "satisfaction_level",
        "last_evaluation",
        "number_project",
        "average_montly_hours",
        "time_spend_company",
    ]
    return df.groupby(by)[cols].agg(["mean", "median", "std"]).round(3)


def clean_basic(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()

    report = {
        "rows_before": int(len(df)),
        "duplicates_removed": 0,
        "rows_missing_left_removed": 0,
        "blank_sales_to_na": 0,
        "blank_salary_to_na": 0,
    }

    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = int(before - len(df))

    numeric_cols = [
        "satisfaction_level",
        "last_evaluation",
        "number_project",
        "average_montly_hours",
        "time_spend_company",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "left" in df.columns:
        df["left"] = pd.to_numeric(df["left"], errors="coerce")
        missing_left = int(df["left"].isna().sum())
        df = df[df["left"].notna()].copy()
        report["rows_missing_left_removed"] = missing_left

    if "sales" in df.columns:
        report["blank_sales_to_na"] = int((df["sales"] == "").sum())
        df["sales"] = df["sales"].replace("", pd.NA)

    if "salary" in df.columns:
        report["blank_salary_to_na"] = int((df["salary"] == "").sum())
        df["salary"] = df["salary"].replace("", pd.NA)

    report["rows_after"] = int(len(df))
    return df, report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run EDA and produce cleaned silver data.")
    p.add_argument(
        "--raw-csv",
        type=Path,
        default=DATA_DIR / "raw" / "HR_comma_sep.csv",
        help="Path to Kaggle HR CSV (default: data/raw/HR_comma_sep.csv)",
    )
    p.add_argument(
        "--rows-per-week",
        type=int,
        default=50,
        help="How many rows to bucket into one synthetic week (default: 50).",
    )
    return p.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("[EDA] Starting...")

    cfg = read_config(CONFIG)
    if cfg.get("dataset") != "HR":
        raise SystemExit("Config requires dataset: HR. Edit configs/data/profile.yml accordingly.")

    args = parse_args()
    raw_csv = args.raw_csv

    silver_parquet = DATA_DIR / "silver" / "hr_silver.parquet"
    class_balance_png = FIGS_DIR / "class_balance.png"
    anomaly_rate_png = FIGS_DIR / "anomaly_rate.png"
    salary_rate_png = FIGS_DIR / "attrition_rate_by_salary.png"
    corr_png = FIGS_DIR / "correlation_heatmap.png"

    missing_csv = RESULTS_DIR / "missing_summary.csv"
    describe_csv = RESULTS_DIR / "numeric_describe.csv"
    outliers_csv = RESULTS_DIR / "outlier_summary.csv"
    by_left_csv = RESULTS_DIR / "grouped_by_left.csv"
    by_sales_csv = RESULTS_DIR / "grouped_by_sales.csv"
    by_salary_csv = RESULTS_DIR / "grouped_by_salary.csv"
    cleanup_json = RESULTS_DIR / "cleanup_report.json"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not raw_csv.exists():
        raise SystemExit(
            f"Missing dataset: {raw_csv}\n"
            "Download HR_comma_sep.csv from Kaggle and place it under data/raw/."
        )

    df = pd.read_csv(raw_csv)
    df = df.rename(columns={"Department": "sales"})
    print(f"[EDA] Loaded {len(df):,} rows from {raw_csv}")

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise SystemExit(f"Missing expected columns: {sorted(missing)}")

    raw_missing = missing_summary(df)
    raw_missing.to_csv(missing_csv, index=True)

    df, cleanup = clean_basic(df)
    df = synthesize_weekly_ts(df, rows_per_week=args.rows_per_week)

    numeric_describe = df[
        [
            "satisfaction_level",
            "last_evaluation",
            "number_project",
            "average_montly_hours",
            "time_spend_company",
            "left",
        ]
    ].describe().round(3).T
    numeric_describe.to_csv(describe_csv, index=True)

    outliers = outlier_summary(df)
    outliers.to_csv(outliers_csv, index=False)

    grouped_numeric_summary(df, "left").to_csv(by_left_csv)
    grouped_numeric_summary(df, "sales").to_csv(by_sales_csv)
    grouped_numeric_summary(df, "salary").to_csv(by_salary_csv)

    with open(cleanup_json, "w") as f:
        json.dump(cleanup, f, indent=2)

    plot_class_balance(df, class_balance_png)
    plot_attrition_rate_by_dept(df, anomaly_rate_png)
    plot_attrition_rate_by_salary(df, salary_rate_png)
    plot_corr_heatmap(df, corr_png)

    save_df(df, silver_parquet)

    rate = df["left"].mean()
    dept = df.groupby("sales")["left"].mean().sort_values(ascending=False).head(5)
    salary = df.groupby("salary")["left"].mean().sort_values(ascending=False)

    print(f"[EDA] Saved silver data -> {silver_parquet}")
    print(f"[EDA] Overall attrition rate: {rate:.3f}")

    print("[EDA] Top departments by attrition rate:")
    for k, v in dept.items():
        print(f" - {k}: {v:.3f}")

    print("[EDA] Attrition rate by salary:")
    for k, v in salary.items():
        print(f" - {k}: {v:.3f}")

    print(f"[EDA] Missing summary -> {missing_csv}")
    print(f"[EDA] Numeric describe -> {describe_csv}")
    print(f"[EDA] Outlier summary -> {outliers_csv}")
    print(f"[EDA] Cleanup report -> {cleanup_json}")
    print("[EDA] Done.")


if __name__ == "__main__":
    main()
