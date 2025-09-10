#!/usr/bin/env python3

from sklearn.ensemble import IsolationForest
import joblib

def train_isolation_forest(X_train, contamination=0.1):
    model = IsolationForest(
        contamination=contamination,
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train)
    return model

def save_model(model, path):
    joblib.dump(model, path)
