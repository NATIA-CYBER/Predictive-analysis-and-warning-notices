#!/usr/bin/env python3

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def save_model(model, path):
    if hasattr(model, 'save_model'):
        model.save_model(str(path))
    else:
        joblib.dump(model, path)
