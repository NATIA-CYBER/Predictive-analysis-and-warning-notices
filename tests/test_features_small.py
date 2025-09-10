#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from scripts.features_build import build_employee_features, create_dept_week_features

def test_employee_features():
    df = pd.DataFrame({
        'satisfaction_level': [0.5, 0.8, 0.3],
        'last_evaluation': [0.7, 0.9, 0.4],
        'number_project': [3, 5, 2],
        'average_montly_hours': [150, 200, 120],
        'time_spend_company': [2, 4, 1],
        'Work_accident': [0, 1, 0],
        'promotion_last_5years': [0, 1, 0],
        'salary': ['low', 'high', 'medium'],
        'sales': ['IT', 'sales', 'hr'],
        'left': [0, 1, 0]
    })
    
    result = build_employee_features(df)
    
    assert 'salary_ord' in result.columns
    assert 'satisfaction_x_eval' in result.columns
    assert 'hours_x_projects' in result.columns
    assert 'dept_IT' in result.columns
    assert 'dept_sales' in result.columns
    assert 'dept_hr' in result.columns
    assert len(result) == 3

def test_dept_week_features():
    df = pd.DataFrame({
        'satisfaction_level': [0.5, 0.8, 0.3, 0.6],
        'last_evaluation': [0.7, 0.9, 0.4, 0.8],
        'number_project': [3, 5, 2, 4],
        'average_montly_hours': [150, 200, 120, 180],
        'time_spend_company': [2, 4, 1, 3],
        'Work_accident': [0, 1, 0, 0],
        'promotion_last_5years': [0, 1, 0, 0],
        'sales': ['IT', 'IT', 'sales', 'sales'],
        'weekly_ts': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-08', '2024-01-08']),
        'left': [0, 1, 0, 1],
        'EmployeeID': [1, 2, 3, 4]
    })
    
    result = create_dept_week_features(df)
    
    assert 'leavers' in result.columns
    assert 'attrition_rate' in result.columns
    assert 'headcount' in result.columns
    assert 'violation_density' in result.columns
    assert 'v1_overtime_sum' in result.columns
    assert len(result) == 2

if __name__ == "__main__":
    test_employee_features()
    test_dept_week_features()
    print("All feature tests passed!")
