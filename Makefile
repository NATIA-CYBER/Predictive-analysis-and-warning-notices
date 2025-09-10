.PHONY: setup install eda features train_xgb train_logreg train_iforest train eval bench dpi dashboard stream test

# create/update env (optional)
setup:
	conda env update --file environment.yml --prune || true

# install package in development mode
install:
	conda run -n pawn pip install -e .

# run tests
test:
	conda run -n pawn python -m pytest -q

eda:
	conda run -n pawn python scripts/eda_run.py

features:
	conda run -n pawn python scripts/features_build.py

train_xgb:
	conda run -n pawn python scripts/train_xgb.py

train_iforest:
	conda run -n pawn python scripts/train_iforest_dept.py

train_logreg:
	conda run -n pawn python scripts/train_logreg.py

# convenience: trains all models
train: train_xgb train_logreg train_iforest

eval:
	conda run -n pawn python scripts/evaluate.py --out results/experiments/last_metrics.json

bench:
	conda run -n pawn python scripts/benchmark_table.py --out results/experiments/benchmark.csv

dpi:
	conda run -n pawn python scripts/dpi_leaderboard.py

dashboard:
	conda run -n pawn streamlit run app/dashboard.py

stream:
	conda run -n pawn python scripts/stream_demo.py
