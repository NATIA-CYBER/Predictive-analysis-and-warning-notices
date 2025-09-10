from setuptools import setup, find_packages

setup(
    name="pawn",
    version="1.0.0",
    description="Predictive Analysis & Warning Notices - HR Attrition Prediction",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0", 
        "scikit-learn>=1.1.0",
        "xgboost>=1.6.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "streamlit>=1.25.0",
        "joblib>=1.1.0",
        "pyarrow>=8.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0"],
    },
)
