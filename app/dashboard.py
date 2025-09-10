#!/usr/bin/env python3

import streamlit as st
import pandas as pd
from pathlib import Path
import json

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
RESULTS_DIR = REPO / "results" / "experiments"
FIGS_DIR = REPO / "figs"
ICONS_DIR = Path(__file__).parent / "static" / "icons"

def display_header_with_icon(title, icon_name):
    col1, col2 = st.columns([1, 10])
    with col1:
        icon_path = ICONS_DIR / f"{icon_name}.png"
        if icon_path.exists():
            st.image(str(icon_path), width=32)
    with col2:
        st.header(title)

def main():
    st.set_page_config(page_title="PAWN Dashboard", layout="wide")
    
    st.title("PAWN - Predictive Analysis & Warning Notices")
    st.markdown("Early-warning HR tool for employee attrition and department-level risk spikes")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["DPI Leaderboard", "Benchmark", "Plots", "Data Explorer"])
    
    if page == "DPI Leaderboard":
        display_header_with_icon("Department Performance Index (DPI) Leaderboard", "trophy")
        dpi_path = RESULTS_DIR / "dpi_leaderboard.csv"
        if dpi_path.exists():
            df_dpi = pd.read_csv(dpi_path)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Departments", len(df_dpi))
            with col2:
                st.metric("Top Performer", df_dpi.iloc[0]['sales'])
            with col3:
                st.metric("Avg DPI Score", f"{df_dpi['dpi'].mean():.3f}")
            
            st.subheader("Rankings")
            st.dataframe(df_dpi.round(3), use_container_width=True)
            
            st.subheader("Top 5 Departments")
            top5 = df_dpi.head(5)
            st.bar_chart(top5.set_index('sales')['dpi'])
            
        else:
            st.warning("DPI leaderboard not found. Run `make bench` first.")
    
    elif page == "Benchmark":
        display_header_with_icon("Model Benchmark", "benchmark")
        
        threshold_path = RESULTS_DIR / "last_metrics.json"
        if threshold_path.exists():
            with open(threshold_path, 'r') as f:
                td = json.load(f)
            tau = td.get("tau", td.get("optimal_threshold"))
            if tau is not None:
                st.info(f"Optimal threshold (τ): {tau:.4f}")
        
        benchmark_path = RESULTS_DIR / "benchmark.csv"
        if benchmark_path.exists():
            df_bench = pd.read_csv(benchmark_path)
            if not df_bench.empty:
                fused = df_bench[df_bench['model'].str.lower().eq('fused')]
                if len(fused) == 1:
                    fr = fused.iloc[0]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Fused Precision", f"{fr.get('precision', float('nan')):.3f}")
                    with col2:
                        st.metric("Fused Recall", f"{fr.get('recall', float('nan')):.3f}")
                    with col3:
                        st.metric("Fused F1-Score", f"{fr.get('f1_score', float('nan')):.3f}")
                
                st.subheader("Model Comparison")
                st.dataframe(df_bench.round(3), use_container_width=True)
                
                if 'f1_score' in df_bench.columns:
                    st.subheader("F1-Score Comparison")
                    st.bar_chart(df_bench.set_index('model')['f1_score'])
            else:
                st.warning("Benchmark is empty. Run `make bench`.")
        else:
            st.warning("Benchmark table not found. Run `make bench` first.")
    
    elif page == "Plots":
        display_header_with_icon("Analysis Plots", "chart")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("XGBoost Convergence")
            xgb_plot_path = FIGS_DIR / "xgb_convergence.png"
            if xgb_plot_path.exists():
                st.image(str(xgb_plot_path))
            else:
                st.warning("XGBoost plot not found.")
        
        with col2:
            st.subheader("Logistic Regression Calibration")
            logreg_plot_path = FIGS_DIR / "logreg_calibration.png"
            if logreg_plot_path.exists():
                st.image(str(logreg_plot_path))
            else:
                st.warning("Logistic Regression plot not found.")
    
    elif page == "Data Explorer":
        display_header_with_icon("Data Explorer", "search")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Employee-Level Data")
            emp_gold_path = DATA_DIR / "gold" / "hr_emp_gold.parquet"
            single_gold = DATA_DIR / "gold" / "hr_gold.parquet"
            
            if emp_gold_path.exists():
                df_emp = pd.read_parquet(emp_gold_path)
                st.write(f"Shape: {df_emp.shape}")
                st.dataframe(df_emp.head(50), use_container_width=True)
            elif single_gold.exists():
                df_emp = pd.read_parquet(single_gold)
                st.write(f"Shape: {df_emp.shape} (fallback to hr_gold.parquet)")
                st.dataframe(df_emp.head(50), use_container_width=True)
            else:
                st.warning("Employee gold data not found.")
        
        with col2:
            st.subheader("Department-Week Data")
            dept_gold_path = DATA_DIR / "gold" / "hr_dept_gold.parquet"
            
            if dept_gold_path.exists():
                df_dept = pd.read_parquet(dept_gold_path)
                st.write(f"Shape: {df_dept.shape}")
                st.dataframe(df_dept.head(50), use_container_width=True)
            elif single_gold.exists():
                df_dept = pd.read_parquet(single_gold)
                st.write(f"Shape: {df_dept.shape} (fallback to hr_gold.parquet)")
                st.dataframe(df_dept.head(50), use_container_width=True)
            else:
                st.warning("Department gold data not found.")

if __name__ == "__main__":
    main()
