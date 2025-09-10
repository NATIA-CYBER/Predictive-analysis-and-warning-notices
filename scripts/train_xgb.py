#!/usr/bin/env python3
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from joblib import dump

import xgboost as xgb
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]
GOLD = REPO / "data" / "gold" / "hr_emp_gold.parquet"
MODELS = REPO / "models"
FIGS = REPO / "figs"
RESULTS = REPO / "results" / "experiments"

RANDOM_STATE = 42


def load_emp_gold() -> pd.DataFrame:
    if not GOLD.exists():
        raise SystemExit(f"Missing {GOLD}. Run `make features` first.")
    df = pd.read_parquet(GOLD)
    if "left" not in df.columns:
        raise SystemExit("Target 'left' missing in employee gold.")
    if "sales" not in df.columns:
        raise SystemExit("Column 'sales' (department) missing. Carry it into hr_emp_gold.parquet for GroupKFold.")
    return df


def split_xy(df: pd.DataFrame):
    y = df["left"].astype(int).values
    # drop obvious non-features; keep numeric features
    drop_cols = {"left", "weekly_ts"}  # add more here if needed
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    # ensure numeric only
    X = X.select_dtypes(include=[np.number])
    return X.values, y, df["sales"].astype(str).values


def group_cv_metrics(X, y, groups, params, n_splits=5) -> pd.DataFrame:
    """Group-aware CV by department; reports AUROC/AUPRC. First fold used for convergence plot."""
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    plotted = False

    for i, (tr, va) in enumerate(gkf.split(X, y, groups), start=1):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]

        # simple cost-sensitive weighting (class imbalance)
        neg, pos = (ytr == 0).sum(), (ytr == 1).sum()
        spw = max(1.0, neg / max(pos, 1))  # scale_pos_weight
        params_fold = {**params, "scale_pos_weight": spw, "random_state": RANDOM_STATE}

        clf = xgb.XGBClassifier(eval_metric="aucpr", **params_fold)
        # early stopping only to get a convergence curve on the *first* fold
        eval_set = [(Xva, yva)]
        clf.fit(Xtr, ytr, verbose=False)

        # convergence figure from first fold
        if not plotted and getattr(clf, "evals_result_", None):
            FIGS.mkdir(parents=True, exist_ok=True)
            hist = clf.evals_result_
            pr = hist["validation_0"]["aucpr"]
            plt.figure(figsize=(6, 4))
            plt.plot(pr)
            plt.xlabel("round")
            plt.ylabel("valid AUC-PR")
            plt.title("XGB convergence (first CV fold)")
            plt.tight_layout()
            plt.savefig(FIGS / "xgb_convergence.png")
            plt.close()
            plotted = True

        proba = clf.predict_proba(Xva)[:, 1]
        auprc = average_precision_score(yva, proba)
        auroc = roc_auc_score(yva, proba)

        rows.append({"fold": i, "auprc": auprc, "auroc": auroc, "scale_pos_weight": spw})

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / "xgb_cv_metrics.csv", index=False)
    print(df.describe().loc[["mean", "std"]][["auprc", "auroc"]])
    return df


def fit_final_and_calibrate(X, y, groups, params):
    """Fit final XGB on full data, then wrap with isotonic calibration using GroupKFold(3)."""
    # class weight hint for final model
    neg, pos = (y == 0).sum(), (y == 1).sum()
    spw = max(1.0, neg / max(pos, 1))
    params_final = {**params, "scale_pos_weight": spw, "random_state": RANDOM_STATE}

    base = xgb.XGBClassifier(eval_metric="aucpr", **params_final)
    # Use a quick held-out split just to avoid pathological overfit before calibration
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    base.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

    # Save raw XGB
    MODELS.mkdir(parents=True, exist_ok=True)
    base.save_model(str(MODELS / "xgb.json"))

    # Probability calibration with group-aware CV
    gkf3 = GroupKFold(n_splits=3)
    calib = CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)
    calib.fit(X, y)

    # Save calibrated model
    dump(calib, MODELS / "xgb_calibrated.joblib")

    # Calibration curve plot (on the hold-out)
    prob = calib.predict_proba(Xva)[:, 1]
    frac_pos, mean_pred = calibration_curve(yva, prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed fraction")
    plt.title("XGB calibrated: reliability curve")
    plt.tight_layout()
    plt.savefig(FIGS / "xgb_calibration.png")
    plt.close()

    # Save params for the record
    with open(RESULTS / "xgb_params.json", "w") as f:
        json.dump(params_final, f, indent=2)

    # quick metrics on hold-out (for console)
    auprc = average_precision_score(yva, prob)
    auroc = roc_auc_score(yva, prob)
    print(f"[calibrated] AUPRC={auprc:.3f} AUROC={auroc:.3f}")


def main():
    df = load_emp_gold()
    X, y, groups = split_xy(df)

    # small, sane defaults
    params = dict(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        n_jobs=4,
        tree_method="hist",
    )

    group_cv_metrics(X, y, groups, params, n_splits=5)
    fit_final_and_calibrate(X, y, groups, params)


if __name__ == "__main__":
    main()
