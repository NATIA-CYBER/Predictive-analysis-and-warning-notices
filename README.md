# PAWN — Predictive Analysis & Warning Notices

Early-warning HR tool for employee attrition and department-level risk spikes.

## Platform Setup & Running

### Prerequisites
1. Download `HR_comma_sep.csv` from Kaggle and place in `data/raw/`
2. Ensure conda/miniconda is installed

### Quick Start
```bash
# Setup environment
conda env create -f environment.yml
conda activate pawn

# Run full pipeline
make eda         # EDA analysis and plots
make features    # Feature engineering
make train_xgb   # Train XGBoost models
make train_iforest # Train anomaly detection
make eval        # Model evaluation
make bench       # Benchmark comparison
make dashboard   # Launch Streamlit dashboard

# Individual components
python scripts/stream_demo.py        # Real-time streaming demo
python scripts/bias_monitoring.py   # Fairness assessment
python scripts/security_audit.py    # Security evaluation
```

### Dashboard Access
After running `make dashboard`, access the platform at:
- **URL**: http://localhost:8501
- **Features**: DPI Leaderboard, Model Benchmarks, Analysis Plots, Data Explorer

## What it does

PAWN predicts employee attrition using supervised learning (XGBoost + Logistic Regression) and detects department-week anomalies using unsupervised methods (Isolation Forest). It fuses these signals with policy violation flags to generate cost-optimized early warnings.

## Pipeline

1. **EDA**: Load Kaggle HR data, create synthetic weekly timestamps, generate class balance and anomaly rate plots
2. **Features**: Build employee-level features and department-week aggregations with violation flags (V1-V4)
3. **Training**: Train XGBoost, Logistic Regression, and Isolation Forest models
4. **Evaluation**: Find optimal threshold using 10:1 false negative penalty
5. **Benchmark**: Compare all models on precision, recall, F1-score
6. **Dashboard**: Streamlit interface showing results and DPI leaderboard

## Model Architecture

### Baseline Models (Production-Ready)
- `models/baseline/xgb_baseline.json` - Simple XGBoost classifier
- `models/baseline/logreg_baseline.joblib` - Logistic regression with scaling
- `models/baseline/fusion_naive_weights.npy` - Fixed 70/30 ensemble weights

### Enhanced Models (Experimental)
- `models/enhanced/xgb_cost_sensitive.json` - Cost-aware XGBoost with monotonic constraints
- `models/enhanced/xgb_calibrated.joblib` - Probability calibration using isotonic regression
- `models/enhanced/fusion_learned.joblib` - Meta-model logistic regression fusion
- `models/enhanced/fusion_scaler.joblib` - Feature scaling for meta-model

### Shared Models
- `models/iforest_dept.joblib` - Department-level anomaly detection

## Artifacts

- **Plots**: `figs/class_balance.png`, `figs/anomaly_rate.png`, `figs/xgb_convergence.png`, `figs/xgb_calibration.png`, `figs/perm_importance.png`, `figs/pdp_hours_satisfaction.png`
- **Data**: `data/silver/hr_silver.parquet`, `data/gold/hr_emp_gold.parquet`, `data/gold/hr_dept_gold.parquet`
- **Results**: `results/experiments/benchmark.csv`, `results/experiments/benchmark_enhanced.csv`, `results/experiments/model_analysis_report.md`, `results/experiments/last_metrics.json`, `results/experiments/dpi_leaderboard.csv`

## Performance

### Best Model: Fusion_Naive (19.3% cost reduction)
- **Precision**: 93.1%, **Recall**: 99.4%, **F1-Score**: 96.1%
- **Optimal Threshold**: 0.1919 (cost-aware for 10:1 FN penalty)
- **Total Cost**: 267 (vs 331 baseline)

### Model Comparison Summary
| Model | Cost Reduction | F1-Score | Notes |
|-------|----------------|----------|--------|
| Fusion_Naive | **+19.3%** | 0.9614 | **Production Ready** |
| XGB_Baseline | 0.0% | 0.9825 | Reference baseline |
| XGB_CostSensitive | -344.1% | 0.8017 | Needs tuning |
| Fusion_Learned | -100.0% | 0.9421 | Experimental |

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

## Ethics & Security Framework

### Implemented Safeguards
- **Bias Monitoring**: Automated fairness assessment across departments and salary levels
- **Security Auditing**: Comprehensive cybersecurity evaluation and recommendations  
- **Privacy Protection**: GDPR/CCPA compliance framework with data minimization
- **Transparency**: Model explainability and decision audit trails

### Critical Findings
- **Bias Violations Detected**: Department and salary-based discrimination identified
- **Security Risk Level**: CRITICAL (45/100 score) - immediate attention required
- **Framework Status**: Complete ethical guidelines and implementation plan established

Run `python scripts/bias_monitoring.py` and `python scripts/security_audit.py` for detailed assessments.
