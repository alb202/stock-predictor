# stock_predictor

A config-driven PyTorch framework that predicts the forward-looking
targets in the `ma_future` table from a multi-table market database
(price, technical indicators, regression trends, sector/index metadata,
and sector/market/volatility ETF context).

See **`docs/ARCHITECTURE.md`** for a full data-flow + model diagram, and
**`docs/REFERENCE.md`** for per-file/per-function API documentation.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate a small synthetic dataset so you can run the whole pipeline
#    without a real database (swap this for your real one -- see below).
python scripts/make_demo_data.py --out-dir data/demo_csv --n-symbols 40

# 2. Point config/config.yaml's data_source at it:
#      data_source:
#        type: csv_dir
#        path: data/demo_csv

# 3. Train (walk-forward CV + a final production fit + held-out test scoring)
python scripts/train.py --config config/config.yaml
#    -> writes artifacts/train_<timestamp>/{config.yaml, metrics.json,
#       model.pt, preprocessor.json, plots/}

# 4. Predict an explicit date range with the trained model
python scripts/predict.py \
    --config config/config.yaml \
    --run-dir artifacts/train_<timestamp> \
    --start-date 2023-06-01 --end-date 2023-06-30
#    -> writes predictions/predict_<timestamp>/{config.yaml,
#       predictions.csv, summary.json, plots/}
```

Plots can also be regenerated on their own, without retraining/repredicting:

```bash
python scripts/plot_training.py artifacts/train_<timestamp>/metrics.json
python scripts/plot_predictions.py predictions/predict_<timestamp>/predictions.csv
```

## Pointing this at your real database

Edit `config/config.yaml`'s `data_source` section:

```yaml
data_source:
  type: duckdb        # or: sqlite | parquet_dir | csv_dir
  path: data/market.duckdb
  symbols: []          # optional universe filter
  start_date: null      # optional history filter
  end_date: null
  max_rows: null         # optional cap, useful for a quick smoke test
```

`type: duckdb` and `type: sqlite` expect every table (`price`,
`equity_info`, `etf_info`, `indicies`, `regression`,
`regression_indicators`, `regression_indicators_ma`, `ta__change`,
`ta__ma_ratio`, `talib`, `ma_future`) to exist in that one file.
`parquet_dir`/`csv_dir` expect one file per table, named
`<table_name>.parquet` / `<table_name>.csv`, in that directory --
exactly what `scripts/make_demo_data.py` writes.

### If your column names differ

Every table's column roles are declared under `tables:` in
`config/config.yaml` -- `id_columns` (join keys), `categorical_columns`
(embedded), `numeric_columns` (standardized features), and, for
`ma_future` only, `target_columns`. **Nothing else needs to change**: add
or remove a column name in the config and the loader, preprocessor,
dataset and model all pick it up automatically (`numeric_columns: []`
even auto-detects every numeric, non-id column if you'd rather not list
them by hand -- see `stock_predictor/features/schema.py`).

Sector/market/volatility ETF context is configured separately, under
`etf_mappings` -- update the symbol on the right if a fund is ever
renamed:

```yaml
etf_mappings:
  sector_etfs:
    technology: "VGT"
    energy: "VDE"
    # ...
  market_etfs:
    sp500: "VOO"
    nasdaq: "QQQ"
  technical_etfs:
    vix_midterm_futures: "VIXM"
    vix_shortterm_futures: "VIXY"
```

## Project layout

```
config/config.yaml             single source of truth for every column list,
                                 ETF mapping, and hyperparameter (see above)

stock_predictor/
  config.py                     typed dataclasses + YAML loader
  data/
    source.py                    DuckDB-backed reader (duckdb/sqlite/parquet/csv)
    loader.py                    joins every table into one modeling frame
    synthetic.py                  synthetic dataset generator for demos/tests
  features/
    schema.py                     config-driven / auto-detected column resolution
    preprocessing.py               leakage-safe scaling + categorical vocabularies
    dataset.py                     torch Dataset, grouped by model branch
  splits/
    time_series_split.py           walk-forward, expanding-window, embargoed CV
  model/
    embeddings.py                  categorical + index-membership embeddings
    branches.py                    per-table MLP encoders + gated attention fusion
    architecture.py                 MultiTaskFusionNet (the full model)
    losses.py                       masked Gaussian NLL / MSE
  training/
    metrics.py                      MAE/RMSE/R2/Pearson/directional accuracy
    trainer.py                      walk-forward orchestration + JSON metrics
  inference/
    predict.py                      predict an explicit date range
  plotting/
    training_plots.py                loss curves, per-target metric curves, CV summary
    prediction_plots.py               confidence, best-symbols, detail grid, calibration
  utils/
    io.py                            timestamped run folders, JSON, config copying
    logging_utils.py                  logging setup

scripts/                        CLI entrypoints (train / predict / plot / demo data)
docs/
  ARCHITECTURE.md               data flow + model diagrams (start here)
  REFERENCE.md                    auto-generated per-file/function API reference
tests/
  test_loader_logic.py           join-logic smoke test (no duckdb/torch required)
```

## Design notes

- **Leakage-safe by construction.** `ma_future` targets are forward-
  looking, so a plain chronological split can leak: a training row's
  label can depend on prices that fall inside the next validation block.
  `splits/time_series_split.py` removes an `embargo_days` gap before every
  validation/test block, and `features/preprocessing.py` fits scalers and
  vocabularies on each split's training rows only.
- **Multi-branch tabular architecture, not a raw sequence model.** The
  source tables already encode history as engineered features as of each
  date (`ta__change` is literally a set of fixed-horizon lagged returns;
  `regression` already summarizes a whole trend window as slope/R/n).
  Each table gets its own small MLP encoder instead, combined with gated
  attention. See `docs/ARCHITECTURE.md` for the full rationale.
- **Uncertainty-aware.** Every target head predicts `(mu, log_var)` via a
  masked Gaussian NLL loss, which is what drives the `{target}__sigma`
  and `{target}__confidence` columns used throughout `plotting/`.
- **Everything is config-driven.** Adding, removing, or renaming a feature
  column never requires touching Python code -- see "If your column names
  differ" above.

## Scaling to a larger universe / more history

The loader materializes one pandas DataFrame per run (DuckDB does the
filtering/joining, but the result is pulled fully into memory). For a
very large universe or multi-decade history, narrow `data_source.symbols`
/ `start_date` / `end_date` / `max_rows`, or run on a machine with enough
RAM -- the join logic itself doesn't need to change.
