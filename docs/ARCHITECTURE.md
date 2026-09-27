# Architecture

This document explains how every input table and config parameter flows
through the pipeline, and how the PyTorch model is put together. See
`README.md` for how to actually run things and `docs/REFERENCE.md` for
per-file/per-function documentation.

## 1. Data flow: from tables to model input

Every table listed here is configured in `config/config.yaml` under
`tables:` -- the column lists shown are the *default* config, but every one
of them can be edited without touching code (see `features/schema.py`).

```mermaid
flowchart TB
    subgraph SOURCE["data_source (config.yaml)"]
        DS["DuckDB / SQLite / parquet_dir / csv_dir\n(stock_predictor/data/source.py)"]
    end

    DS --> PRICE[price\nsymbol, date, close, volume]
    DS --> EQ[equity_info\nsector_key, industry_key]
    DS --> ETFI[etf_info\ncategory_key]
    DS --> IDX[indicies\nsymbol -> index_name *]
    DS --> REG[regression\nsymbol, date, start_date, r_value, slope, k, n *]
    DS --> RI[regression_indicators\n9 CHANNEL_* columns]
    DS --> RIMA[regression_indicators_ma\nsmoothed CHANNEL_* columns]
    DS --> TACH[ta__change\nCHANGE_HIGH/LOW/CLOSE_*]
    DS --> TAMA[ta__ma_ratio\nSMA_RATIO_*, EMA_RATIO_*]
    DS --> TALIB[talib\nRSI, MACD, ADX, CDL*, ...]
    DS --> FUT[ma_future\ntarget columns]

    PRICE --> BASE(("base frame\n1 row per symbol/date"))
    EQ --> INSTR["instrument_info join\n(loader._attach_instrument_info)"]
    ETFI --> INSTR
    INSTR --> BASE
    IDX --> BASE

    REG -->|"pick strongest trend per\n(symbol, date): score =\n|r_value|^p * log1p(n)^q"| BESTREG(("best regression event\n+ chosen start_date"))
    BESTREG --> BASE
    RI -->|"join on (symbol, date, start_date)"| BASE
    RIMA -->|"join on (symbol, date, start_date)\ncolumns suffixed __MA"| BASE

    TACH -->|"join on (symbol, date)"| BASE
    TAMA -->|"join on (symbol, date)"| BASE
    TALIB -->|"join on (symbol, date)"| BASE

    BASE -->|"sector_key -> sector ETF\n(etf_mappings.sector_etfs,\nsubstring match)"| CTXSEC["context_sector branch\n(ta__change/ta__ma_ratio\nfor the mapped sector ETF,\nsame date)"]
    BASE -->|"fixed symbols\n(etf_mappings.market_etfs:\nVOO, QQQ)"| CTXMKT["context_market branch"]
    BASE -->|"fixed symbols\n(etf_mappings.technical_etfs:\nVIXM, VIXY)"| CTXTECH["context_technical branch"]

    CTXSEC --> BASE
    CTXMKT --> BASE
    CTXTECH --> BASE

    FUT -->|"join on (symbol, date);\nauto-detect numeric cols\nif target_columns empty"| BASE

    BASE --> LOADED[["LoadedDataset\n(data/loader.py)\nframe + numeric_branches +\ncategorical_columns + target_columns"]]
```

`*` = the loader reduces a multi-row-per-symbol table down to one row per
`(symbol, date)` before it reaches the model: `indicies` is grouped into a
list of memberships per symbol, and `regression` is reduced to its single
strongest trend per `(symbol, date)`.

### Preprocessing (leakage-safe)

`features/preprocessing.py`'s `Preprocessor` is **fit fresh on each walk-
forward split's training rows only** (never on validation/test/future
rows): numeric columns are standardized (mean/std from that split's
training data, clipped to `preprocessing.clip_std`), categorical columns
(`sector_key`, `industry_key`, `security_type`, index memberships) are
mapped to integer ids via a vocabulary with a `min_category_freq` cutoff,
and `ma_future` targets are standardized the same way, with a separate
mask marking which labels are actually observed.

## 2. Time-series splitting

```mermaid
flowchart LR
    A["[ train_1 ]"] -->|embargo| B["[ val_1 ]"]
    C["[   train_2   ]"] -->|embargo| D["[ val_2 ]"]
    E["[      train_3      ]"] -->|embargo| F["[ val_3 ]"]
    G["[  final train (all pre-test data)  ]"] -->|embargo| H["[ final val ]"]
    G -.->|"production model,\nscored once"| I["[ TEST ]"]
```

Each split's training window **expands** (rather than a fixed rolling
window) so the model sees progressively more market history across splits
-- this is what the project spec calls training on "multiple splits of
previous data to reduce the effect of broader market trends". An
`embargo_days` gap is removed immediately before every validation/test
block so that no training row's forward-looking `ma_future` label overlaps
a later block's feature dates (`splits/time_series_split.py`).

The `n_splits` CV splits measure how consistently the architecture/
hyperparameters generalize across different historical regimes. The
**final split** (`WalkForwardSplitter.final_split`) then fits the actual
production model on *all* pre-test history, which is the model scored on
the held-out test block and used for `scripts/predict.py`.

## 3. Model architecture

```mermaid
flowchart TB
    subgraph CAT["categorical inputs"]
        SEC[sector_key]
        IND[industry_key]
        SECTYPE[security_type]
        IDXM["index memberships\n(multi-valued)"]
    end
    SEC --> CATEMB["CategoricalEmbeddingGroup\n(model/embeddings.py)"]
    IND --> CATEMB
    SECTYPE --> CATEMB
    IDXM --> IDXEMB["IndexMembershipEmbedding\n(mean-pooled)"]

    subgraph NUM["numeric branches (one MLP encoder each)"]
        B1[price]
        B2[regression]
        B3[regression_indicators]
        B4[regression_indicators_ma]
        B5[ta__change]
        B6[ta__ma_ratio]
        B7[talib]
        B8[context_sector]
        B9[context_market]
        B10[context_technical]
    end
    B1 & B2 & B3 & B4 & B5 & B6 & B7 & B8 & B9 & B10 --> BE["BranchEncoder x10\n(model/branches.py)\nLinear -> LayerNorm -> GELU -> Dropout"]

    BE --> GATE["GatedAttentionFusion\nper-sample, per-branch\nsoftmax importance weights"]
    GATE --> CONCAT(("concat"))
    CATEMB --> CONCAT
    IDXEMB --> CONCAT
    CONCAT --> TRUNK["shared trunk MLP\n(fusion_hidden_dims)"]

    TRUNK --> H1["head: target 1\n(mu, log_var)"]
    TRUNK --> H2["head: target 2\n(mu, log_var)"]
    TRUNK --> H3["head: target N\n(mu, log_var)"]

    H1 --> L1["masked Gaussian NLL\n(model/losses.py)"]
    H2 --> L2[masked Gaussian NLL]
    H3 --> L3[masked Gaussian NLL]
```

**Why a multi-branch tabular network instead of a raw sequence model
(RNN/Transformer over price history)?** The source tables already encode
history as engineered features as of each date -- `ta__change` is
literally a set of fixed-horizon lagged returns (1, 2, 3, ..., 200 days
back), and `regression`/`regression_indicators` already summarize a whole
trend window as slope/R/n/channel-position. Re-deriving that signal from
raw price sequences would add real architectural complexity (variable-
length windows, padding, positional encoding) for little expected benefit
given the features already available. Each table instead gets its own
small encoder ("branch"), branches are combined with a gated-attention
layer (so the model can learn to down-weight a branch that's missing or
uninformative for a given row -- e.g. a symbol with no qualifying
regression trend that day), and the fused representation feeds a shared
trunk and one small head per `ma_future` target.

**Why per-target `(mu, log_var)` heads?** Each target is optimized with a
masked Gaussian negative-log-likelihood loss (`model/losses.py`), which
lets the model learn *how confident* to be about each prediction as well
as the prediction itself -- `exp(0.5 * log_var)` becomes the predicted
sigma used throughout `plotting/prediction_plots.py` and
`inference/predict.py`'s `{target}__confidence` column.

## 4. Config -> code map

| Config section | Consumed by |
| --- | --- |
| `data_source` | `data/source.py`, `data/loader.py` |
| `tables.*` | `features/schema.py` (column resolution), `data/loader.py` (joins) |
| `etf_mappings` | `data/loader.py::_attach_etf_context` |
| `regression_selection` | `data/loader.py::_attach_regression` |
| `preprocessing` | `features/preprocessing.py` |
| `splitting` | `splits/time_series_split.py` |
| `model` | `model/architecture.py` |
| `training` | `training/trainer.py` |
| `project` | `utils/io.py` (run folder locations) |

## 5. Prediction and plotting flow

```mermaid
flowchart LR
    RUN["scripts/train.py"] -->|writes| ARTIFACTS["artifacts/train_&lt;timestamp&gt;/\nconfig.yaml, metrics.json,\nmodel.pt, preprocessor.json, plots/"]
    ARTIFACTS --> PLOTTRAIN["plotting/training_plots.py\nloss curves, per-target metric\ncurves, CV summary, test table"]

    ARTIFACTS -->|"model.pt + preprocessor.json"| PRED["scripts/predict.py\n(explicit start/end date)"]
    PRED -->|writes| PREDOUT["predictions/predict_&lt;timestamp&gt;/\nconfig.yaml, predictions.csv,\nsummary.json, plots/"]
    PREDOUT --> PLOTPRED["plotting/prediction_plots.py\nconfidence-over-time, best-symbols,\nprediction-detail grid, calibration"]
```
