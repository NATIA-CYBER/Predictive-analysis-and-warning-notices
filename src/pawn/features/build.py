#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def build_employee_features(df):
    df = df.copy()
    
    salary_map = {'low': 0, 'medium': 1, 'high': 2}
    df['salary_ord'] = df['salary'].map(salary_map)
    
    df['satisfaction_x_eval'] = df['satisfaction_level'] * df['last_evaluation']
    df['hours_x_projects'] = df['average_montly_hours'] * df['number_project']
    
    dept_dummies = pd.get_dummies(df['sales'], prefix='dept')
    df = pd.concat([df, dept_dummies], axis=1)
    
    return df

def create_dept_week_features(df):
    agg_dict = {
        'satisfaction_level': ['mean', 'std'],
        'last_evaluation': ['mean', 'std'],
        'number_project': ['mean', 'std'],
        'average_montly_hours': ['mean', 'std'],
        'time_spend_company': ['mean', 'std'],
        'Work_accident': 'sum',
        'promotion_last_5years': 'sum',
        'left': ['sum', 'count'],
        'EmployeeID': 'nunique'
    }
    
    dept_week = df.groupby(['sales', 'weekly_ts']).agg(agg_dict).reset_index()
    dept_week.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in dept_week.columns]
    
    dept_week['leavers'] = dept_week['left_sum']
    dept_week['headcount'] = dept_week['EmployeeID_nunique']
    dept_week['attrition_rate'] = dept_week['leavers'] / dept_week['headcount']
    
    dept_week['v1_overtime_sum'] = (dept_week['average_montly_hours_mean'] > 200).astype(int)
    dept_week['v2_low_satisfaction_sum'] = (dept_week['satisfaction_level_mean'] < 0.3).astype(int)
    dept_week['v3_stagnation_sum'] = ((dept_week['time_spend_company_mean'] > 5) & 
                                     (dept_week['promotion_last_5years_sum'] == 0)).astype(int)
    dept_week['v4_overload_sum'] = (dept_week['number_project_mean'] > 6).astype(int)
    
    violation_cols = ['v1_overtime_sum', 'v2_low_satisfaction_sum', 'v3_stagnation_sum', 'v4_overload_sum']
    dept_week['violation_density'] = dept_week[violation_cols].sum(axis=1) / dept_week['headcount']
    
    return dept_week
