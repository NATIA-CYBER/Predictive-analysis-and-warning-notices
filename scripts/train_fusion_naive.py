#!/usr/bin/env python3

import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO / "models" / "baseline"

def main():
    print("[TRAIN-FUSION-NAIVE] Creating naive fusion model...")
    
    # simple weighted average fusion (0.7 XGB + 0.3 IForest)
    naive_weights = np.array([0.7, 0.3])
    
    # save as simple array for consistency
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MODELS_DIR / "fusion_naive_weights.npy", naive_weights)
    
    print(f"[TRAIN-FUSION-NAIVE] Saved naive fusion weights: {naive_weights}")
    print("[TRAIN-FUSION-NAIVE] Done.")


if __name__ == "__main__":
    main()
