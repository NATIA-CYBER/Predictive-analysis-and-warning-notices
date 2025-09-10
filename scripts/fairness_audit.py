#!/usr/bin/env python3

import argparse, json, logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, precision_score, recall_score

try:
    import xgboost as xgb
except Exception:
    xgb = None

REPO = Path(__file__).resolve().parents[1]

def infer_dept(df: pd.DataFrame) -> pd.Series:
    if "sales" in df.columns:
        return df["sales"].astype(str)
    dept_cols = [c for c in df.columns if c.startswith("dept_")]
    if not dept_cols:
        return pd.Series(["unknown"] * len(df), index=df.index)
    idx = df[dept_cols].to_numpy().argmax(axis=1)
    has_any = df[dept_cols].to_numpy().max(axis=1) > 0
    labels = np.array([c.replace("dept_", "") for c in dept_cols])
    out = np.where(has_any, labels[idx], "unknown")
    return pd.Series(out, index=df.index)

def cost_weighted_tau(y: np.ndarray, p: np.ndarray, fn_weight: float = 10.0) -> float:
    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]
    best_cost, best_tau = float("inf"), 0.5
    for t in np.unique(p_sorted):
        yhat = (p_sorted >= t).astype(int)
        fn = ((y_sorted == 1) & (yhat == 0)).sum()
        fp = ((y_sorted == 0) & (yhat == 1)).sum()
        cost = fn_weight * fn + fp
        if cost < best_cost:
            best_cost, best_tau = cost, float(t)
    return best_tau

def load_model(model_path: Path):
    if model_path.suffix == ".json":
        if not xgb:
            sys.exit("need xgboost for .json models")
        m = xgb.XGBClassifier()
        m.load_model(str(model_path))
        return m
    elif model_path.suffix in {".joblib", ".pkl"}:
        from joblib import load
        return load(model_path)
    else:
        sys.exit(f"unsupported: {model_path.suffix}")

def predict(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "predict"):
        pred = model.predict(X)
        if pred.ndim == 1 and set(np.unique(pred)) <= {0,1}:
            return pred.astype(float)
    sys.exit("model has no predict_proba")

def group_metrics(y: np.ndarray, yhat: np.ndarray, p: np.ndarray, groups: pd.Series) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for g in sorted(groups.unique()):
        mask = (groups == g).to_numpy()
        if mask.sum() == 0:
            continue
        yg, yhg, pg = y[mask], yhat[mask], p[mask]
        rec = recall_score(yg, yhg, zero_division=0)
        prec = precision_score(yg, yhg, zero_division=0)
        fnr = 1 - rec
        fpr = ( (yhg==1) & (yg==0) ).sum() / max( (yg==0).sum(), 1 )
        brier = brier_score_loss(yg, pg) if len(np.unique(yg)) > 1 else float("nan")
        out[str(g)] = {
            "n": int(mask.sum()),
            "prevalence": float(yg.mean()),
            "recall": float(rec),
            "fnr": float(fnr),
            "precision": float(prec),
            "fpr": float(fpr),
            "brier": float(brier),
        }
    return out

def overall_metrics(y, yhat, p) -> Dict:
    rec = recall_score(y, yhat, zero_division=0)
    prec = precision_score(y, yhat, zero_division=0)
    fnr = 1 - rec
    fpr = ( (yhat==1) & (y==0) ).sum() / max( (y==0).sum(), 1 )
    brier = brier_score_loss(y, p) if len(np.unique(y)) > 1 else float("nan")
    return dict(recall=float(rec), fnr=float(fnr), precision=float(prec), fpr=float(fpr), brier=float(brier))

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gold", type=Path)
    src.add_argument("--predictions", type=Path)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--tau", type=float)
    ap.add_argument("--fn_weight", type=float, default=10.0)
    ap.add_argument("--max_fnr_gap", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "fairness_audit.json")
    args = ap.parse_args()

    if args.predictions:
        dfp = pd.read_csv(args.predictions)
        if "y_true" not in dfp or "y_prob" not in dfp:
            sys.exit("predictions CSV must have y_true,y_prob")
        y = dfp["y_true"].astype(int).to_numpy()
        p = dfp["y_prob"].astype(float).to_numpy()
        dept = dfp["dept"] if "dept" in dfp else pd.Series(["unknown"] * len(dfp))
    else:
        if not args.model:
            sys.exit("--gold requires --model")
        df = pd.read_parquet(args.gold)
        features = [
            "satisfaction_level","last_evaluation","number_project",
            "average_montly_hours","time_spend_company","Work_accident",
            "promotion_last_5years","salary_ord","satisfaction_x_eval","hours_x_projects",
        ] + [c for c in df.columns if c.startswith("dept_")]
        X = df[features]
        y = df["left"].astype(int).to_numpy()
        model = load_model(args.model)
        p = predict(model, X)
        dept = infer_dept(df)

    tau = args.tau or cost_weighted_tau(y, p, fn_weight=args.fn_weight)
    yhat = (p >= tau).astype(int)

    per_group = group_metrics(y, yhat, p, dept)
    ov = overall_metrics(y, yhat, p)
    fnrs = [d["fnr"] for d in per_group.values() if np.isfinite(d["fnr"])]
    fnr_gap = float(max(fnrs) - min(fnrs)) if fnrs else float("nan")

    report = {
        "tau": float(tau),
        "fn_weight": float(args.fn_weight),
        "overall": ov,
        "per_group": per_group,
        "gaps": {"fnr_gap": fnr_gap},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")

    if np.isfinite(fnr_gap) and fnr_gap > args.max_fnr_gap:
        print(f"FAIL: FNR gap {fnr_gap:.3f} > {args.max_fnr_gap:.3f}")
        sys.exit(1)

if __name__ == "__main__":
    main()
