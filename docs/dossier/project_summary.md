# PAWN Project Dossier

## Executive Summary

PAWN is an early-warning HR analytics tool that combines supervised employee attrition prediction with unsupervised department-level anomaly detection. The system generates cost-optimized alerts when fused risk scores exceed a threshold tuned for 10:1 false negative penalty.

## Technical Implementation

### Data Pipeline
- **Input**: Kaggle HR dataset (14,999 employees)
- **Preprocessing**: Synthetic weekly timestamps, basic cleanup
- **Features**: Employee-level (satisfaction, evaluation, projects, hours, tenure) + Department-week aggregations with policy violation flags

### Models
- **XGBoost**: Employee attrition classifier (98.0% F1-score)
- **Logistic Regression**: Interpretable baseline (52.2% F1-score)  
- **Isolation Forest**: Department-week anomaly detection
- **Fusion**: 0.7 * XGB + 0.3 * IForest scores

### Performance Metrics
- **Fused Model**: 93.1% precision, 99.4% recall, 96.1% F1-score
- **Optimal Threshold**: 0.1919 (minimizes 10*FN + 1*FP cost)
- **Department Rankings**: R&D leads DPI at 0.495

## Artifacts Generated
- EDA plots: class balance, department attrition rates
- Model convergence and calibration curves
- Benchmark comparison table
- Department Performance Index leaderboard
- Real-time streaming demo with JSON alerts

## Ethics & Limitations
- No PII exposure, pseudonymized employee IDs
- Focus on operational risks, not individual labeling
- Coarse temporal resolution (weekly synthetic timestamps)
- Policy thresholds derived from data, not organizational standards
