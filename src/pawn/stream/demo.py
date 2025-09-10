#!/usr/bin/env python3

import json
import pandas as pd
from datetime import datetime

def generate_alert(employee_id, dept, week, fused_score, threshold):
    alert = {
        "timestamp": datetime.now().isoformat(),
        "employee_id": employee_id,
        "department": dept,
        "week": week.strftime("%Y-%m-%d"),
        "fused_score": float(fused_score),
        "threshold": float(threshold),
        "risk_level": "HIGH" if fused_score > threshold else "MEDIUM",
        "action_required": fused_score > threshold
    }
    return alert

def stream_alerts(df, fused_scores, threshold):
    alerts = []
    for idx, row in df.iterrows():
        if fused_scores[idx] > threshold * 0.8:  # Alert for scores above 80% of threshold
            alert = generate_alert(
                row['EmployeeID'], 
                row['sales'], 
                row['weekly_ts'], 
                fused_scores[idx], 
                threshold
            )
            alerts.append(alert)
    return alerts
