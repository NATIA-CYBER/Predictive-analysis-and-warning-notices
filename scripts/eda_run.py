#!/usr/bin/env python3

from pathlib import Path 
import argparse
import warnings
import yaml
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "data" / "profile.yml"
DATA_DIR = REPO / "data"
FIGS_DIR = REPO / "figs"

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
    rate = df.groupby("sales", dropna=False)["left"].mean().sort_values()
    plt.figure(figsize=(8, 5))
    rate.plot(kind="bar")
    plt.title("Attrition rate by department")
    plt.xlabel("department (sales)")
    plt.ylabel("attrition rate")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close()

def clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)

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

    df = df[df["left"].notna()]

    for c in ["sales", "salary"]:
        if c in df.columns:
            df[c] = df[c].replace("", pd.NA)

    return df

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run EDA and produce silver parquet + required figs.")
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
        help="How many rows to bucket into one synthetic 'week' (default: 50).",
    )
    return p.parse_args()

def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("[EDA] Starting…")

    cfg = read_config(CONFIG)
    if cfg.get("dataset") != "HR":
        raise SystemExit("Config requires dataset: HR. Edit configs/data/profile.yml accordingly.")

    args = parse_args()
    raw_csv = args.raw_csv
    silver_parquet = DATA_DIR / "silver" / "hr_silver.parquet"
    class_balance_png = FIGS_DIR / "class_balance.png"
    anomaly_rate_png = FIGS_DIR / "anomaly_rate.png"

    (DATA_DIR / "silver").mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

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

    df = clean_basic(df)
    df = synthesize_weekly_ts(df, rows_per_week=args.rows_per_week)

    plot_class_balance(df, class_balance_png)
    plot_attrition_rate_by_dept(df, anomaly_rate_png)

    silver_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(silver_parquet, index=False)
    print(f"[EDA] Saved silver data → {silver_parquet}")

    rate = df["left"].mean()
    dept = df.groupby("sales")["left"].mean().sort_values(ascending=False).head(3)
    print(f"[EDA] Overall attrition rate: {rate:.3f}")
    print("[EDA] Top-3 departments by attrition rate:")
    for k, v in dept.items():
        print(f"       - {k}: {v:.3f}")

    print("[EDA] Done.")

if __name__ == "__main__":
    main()
