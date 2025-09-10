# Technical Implementation Details

## Feature Engineering

### Employee Features
- **Salary Ordinal**: low=0, medium=1, high=2
- **Interaction Terms**: satisfaction × evaluation, hours × projects
- **Department Dummies**: One-hot encoding for all departments
- **Tenure & Accident**: Direct inclusion of company years and work accidents

### Department-Week Features
- **Aggregations**: Mean satisfaction, evaluation, hours, projects per dept-week
- **Attrition Metrics**: Headcount, leavers, attrition rate
- **Policy Violations**:
  - V1: Overtime (>200 hours/month)
  - V2: Low satisfaction (<0.3)
  - V3: Stagnation (>5 years, no promotion)
  - V4: Project overload (>6 projects)
- **Lag Features**: Previous week's violation counts
- **Violation Density**: Total violations / headcount

## Model Architecture

### XGBoost Configuration
```python
xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

### Logistic Regression Pipeline
- StandardScaler preprocessing
- L2 regularization (C=1.0)
- Probability calibration enabled

### Isolation Forest Setup
- contamination=0.1 (10% anomaly rate)
- n_estimators=100
- Department-week feature subset only

## Evaluation Framework

### Cost Function
```
Cost = 10 * False_Negatives + 1 * False_Positives
```

### Threshold Optimization
- Grid search from 0.05 to 0.95 (step 0.01)
- Minimize cost on validation set
- Optimal threshold: 0.1919

### Fusion Strategy
```
fused_score = 0.7 * xgb_proba + 0.3 * iforest_anomaly_score
```

## Performance Benchmarks

| Model | Precision | Recall | F1-Score | Threshold |
|-------|-----------|--------|----------|-----------|
| XGBoost | 98.0% | 98.0% | 98.0% | 0.5 |
| LogReg | 52.2% | 52.2% | 52.2% | 0.5 |
| Fused | 93.1% | 99.4% | 96.1% | 0.1919 |
