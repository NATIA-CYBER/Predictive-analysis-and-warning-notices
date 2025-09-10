# PAWN — Predictive Analysis & Warning Notices

Early-warning HR tool for employee attrition and department-level risk spikes.

## Quickstart

```bash
conda env create -f environment.yml
conda activate pawn
make eda
make features
make train_xgb
make train_iforest
make eval
make bench
make dashboard
```

## What it does

PAWN predicts employee attrition using supervised learning (XGBoost + Logistic Regression) and detects department-week anomalies using unsupervised methods (Isolation Forest). It fuses these signals with policy violation flags to generate cost-optimized early warnings.

## Pipeline

1. **EDA**: Load Kaggle HR data, create synthetic weekly timestamps, generate class balance and anomaly rate plots
2. **Features**: Build employee-level features and department-week aggregations with violation flags (V1-V4)
3. **Training**: Train XGBoost, Logistic Regression, and Isolation Forest models
4. **Evaluation**: Find optimal threshold using 10:1 false negative penalty
5. **Benchmark**: Compare all models on precision, recall, F1-score
6. **Dashboard**: Streamlit interface showing results and DPI leaderboard

## Artifacts

- **Plots**: `figs/class_balance.png`, `figs/anomaly_rate.png`, `figs/xgb_convergence.png`
- **Data**: `data/silver/hr_silver.parquet`, `data/gold/hr_emp_gold.parquet`, `data/gold/hr_dept_gold.parquet`
- **Models**: `models/xgb.json`, `models/logreg.joblib`, `models/iforest_dept.joblib`
- **Results**: `results/experiments/benchmark.csv`, `results/experiments/last_metrics.json`, `results/experiments/dpi_leaderboard.csv`

## Performance

- **Fused Model**: 93.1% precision, 99.4% recall, 96.1% F1-score
- **Optimal Threshold**: 0.1919 (cost-aware for 10:1 FN penalty)
- **Top Department**: R&D (DPI: 0.495)
---

## Dataset

**Kaggle Human Resources dataset** (classic columns: `satisfaction_level`, `last_evaluation`, `number_project`, `average_montly_hours`, `time_spend_company`, `Work_accident`, `promotion_last_5years`, `sales` (department), `salary`, `left`).

The CSV has no timestamps; the code **synthesizes a weekly `weekly_ts`** from row order for windowing and the stream demo.

**Place the raw file here (manual):**
 
---

## Problem Statements

- **Who will quit?** Employee-level classifier for `left = 1`.
- **When (coarse)?** “Will leave within N months” via a tenure-based proxy (limitation documented).
- **Where are risks rising?** Dept-week spikes using policy-style violation flags + **IsolationForest**.
- **Which departments to reward?** Simple **Department Performance Index** (retention + sustainable workload + eval/promo balance) and a ranking.

---

## Quickstart

```bash
# 0) Prereq: put the Kaggle CSV in data/raw/
#    data/raw/HR_comma_sep.csv

# 1) Create the Conda environment
conda env create -f environment.yml

# 2) Activate it
conda activate pawn

# 3) Run the pipeline
make eda         # writes data/silver/hr_silver.parquet + figs/class_balance.png, figs/anomaly_rate.png
make features    # writes data/gold/hr_emp_gold.parquet and/or data/gold/hr_dept_gold.parquet (or hr_gold.parquet)
make train_xgb   # writes models/xgb.json + figs/xgb_convergence.png
make train_iforest  # writes models/iforest.joblib
make eval        # writes results/experiments/thresholds.json (cost-aware τ)
make bench       # writes results/experiments/benchmark.csv
make dashboard   # launches Streamlit  If you prefer not to activate the env in your shell, prefix with conda run -n pawn …. (reads Gold + Results)
```
