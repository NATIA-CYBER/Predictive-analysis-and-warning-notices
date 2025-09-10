import pytest

def test_core_imports():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import sklearn
    assert True

def test_ml_imports():
    import xgboost as xgb
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_fscore_support
    assert True

def test_streamlit_import():
    import streamlit as st
    assert True

def test_joblib_import():
    import joblib
    assert True

if __name__ == "__main__":
    test_core_imports()
    test_ml_imports()
    test_streamlit_import()
    test_joblib_import()
    print("All imports successful!")
