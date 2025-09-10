#!/usr/bin/env python3

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def cost_function(y_true, y_pred, fn_weight=10, fp_weight=1):
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return fn_weight * fn + fp_weight * fp

def find_optimal_threshold(y_true, y_proba, fn_weight=10):
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_threshold = 0.5
    best_cost = float('inf')
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        cost = cost_function(y_true, y_pred, fn_weight)
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold
    
    return best_threshold

def calculate_metrics(y_true, y_pred):
    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
