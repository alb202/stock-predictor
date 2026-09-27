# API Reference

Auto-generated from source docstrings by `scripts/generate_reference_docs.py` -- do not hand-edit; re-run the script instead. See `docs/ARCHITECTURE.md` for the narrative version with diagrams.

## `stock_predictor/__init__.py`

stock_predictor: config-driven PyTorch framework for predicting the
forward-looking targets in the ``ma_future`` table from a multi-table
market database (price, technical indicators, regression trends, sector
metadata and ETF context).

See README.md for a quickstart and docs/ARCHITECTURE.md for a full data
flow + model diagram.

## `stock_predictor/config.py`

Typed configuration objects for the whole project.

Every other module receives a :class:`Config` instance instead of reading
``config.yaml`` itself. This keeps column lists, hyperparameters and paths
in one place (the YAML file) while still giving the rest of the codebase
type-checked, autocomplete-friendly access to every setting.

Call :func:`load_config` once (e.g. in a CLI script) and pass the result
around.

### `class ProjectConfig`

### `class DataSourceConfig`

### `class TableConfig`

Column roles for a single database table.

``id_columns`` are join keys and are never treated as features.
``categorical_columns`` are embedded; ``numeric_columns`` are
standardized. An empty ``numeric_columns`` list means "auto-detect every
non-id numeric column at load time" (see ``features/schema.py``).

### `class EtfMappingsConfig`

#### `all_context_symbols_config(self) -> dict[str, str]`

Flatten every mapped ETF into a single ``{context_key: symbol}`` dict.

### `class RegressionSelectionConfig`

### `class PreprocessingConfig`

### `class SplittingConfig`

### `class ModelConfig`

### `class TrainingConfig`

### `class Config`

#### `table(self, name: str) -> TableConfig`

Look up a table's column configuration by its config key.

#### `_dataclass_from_dict(cls: type, data: dict[str, Any]) -> Any`

Build a dataclass instance from a dict, ignoring unknown keys.

Keeping this permissive (rather than failing on extra YAML keys) makes
it easy to add new config fields and roll them into the dataclasses
later without breaking existing config files.

#### `load_config(path: str | Path) -> Config`

Load and validate ``config.yaml`` into a :class:`Config` object.

#### `config_to_dict(cfg: Config) -> dict[str, Any]`

Return the config's original YAML dict (for writing run metadata).

## `stock_predictor/data/__init__.py`

data subpackage of stock_predictor.

## `stock_predictor/data/loader.py`

Builds the single wide, per-(symbol, date) modeling frame.

The heavy lifting lives here so that every downstream module (preprocessing,
dataset construction, the model) can stay simple and just ask "what are the
numeric columns for branch X" instead of knowing anything about joins.

High-level flow (see docs/ARCHITECTURE.md for a diagram):

1.  Start from ``price`` (one row per symbol/date).
2.  Attach ``equity_info`` / ``etf_info`` categorical fields per symbol.
3.  Attach ``indicies`` membership (kept as a list-per-symbol side table).
4.  Pick the single strongest ``regression`` event per (symbol, date) and
    attach its own numeric columns, plus the matching
    ``regression_indicators`` and ``regression_indicators_ma`` rows.
5.  Attach ``ta__change`` / ``ta__ma_ratio`` / ``talib`` for the row's own
    symbol.
6.  Attach sector / broad-market / volatility ETF context: for each row,
    join the *same date's* ``ta__change`` / ``ta__ma_ratio`` rows for the
    mapped sector ETF, plus every market_etfs / technical_etfs symbol.
7.  Optionally attach ``ma_future`` targets.

The result is one :class:`LoadedDataset`: a wide pandas DataFrame plus a
dict describing which columns belong to which "branch" (the model builds
one small encoder per branch).

### `class SupportsReadTable`

The subset of :class:`DataSource` that :class:`DataLoader` relies on.

Declared as a Protocol (rather than importing ``DataSource`` directly)
so ``DataLoader`` can be unit-tested with a lightweight fake table
lookup that doesn't require the ``duckdb`` package to be installed.

#### `read_table(self, table_name: str, columns: list[str] | None = None, symbols: list[str] | None = None, date_column: str | None = 'date', start_date: str | None = None, end_date: str | None = None, max_rows: int | None = None) -> pd.DataFrame`

### `class LoadedDataset`

Everything the rest of the pipeline needs about the joined data.

#### `all_numeric_columns(self) -> list[str]`

### `class DataLoader`

Loads and joins every configured table for a given symbol/date filter.

#### `__init__(self, config: Config, source: 'SupportsReadTable | None' = None) -> None`

#### `load(self, include_targets: bool = True) -> LoadedDataset`

#### `_read(self, table_key: str, symbols: list[str] | None = None, start_date: str | None = None, end_date: str | None = None, max_rows: int | None = None, date_column: str | None = 'date') -> pd.DataFrame`

#### `_attach_instrument_info(self, base: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]`

#### `_attach_index_membership(self, base: pd.DataFrame) -> tuple[pd.DataFrame, str | None]`

#### `_attach_regression(self, base: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]`

#### `_attach_regression_indicators(self, base: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]`

#### `_attach_regression_indicators_ma(self, base: pd.DataFrame, fallback_cols: list[str]) -> tuple[pd.DataFrame, list[str]]`

#### `_attach_symbol_table(self, base: pd.DataFrame, table_key: str) -> tuple[pd.DataFrame, list[str]]`

#### `_attach_etf_context(self, base: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]`

#### `_attach_targets(self, base: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]`

#### `_match_sector_etf(sector_key: str, sector_etfs: dict[str, str]) -> str | None`

Map an equity's sector_key to a sector ETF symbol via substring match.

Mirrors the project requirement: "the ETF 'VDE' ... matches any stock
with 'energy' in the sector_key column". Comparison is done on
lower-cased, hyphen/space-insensitive strings.

#### `_attach_context_frames(base: pd.DataFrame, feature_frames: dict[str, pd.DataFrame], symbol_selector: pd.Series | str | None, prefix: str) -> tuple[pd.DataFrame, list[str]]`

Left-join per-date context features for a fixed or per-row ETF symbol.

``symbol_selector`` is either a constant ETF symbol (market/technical
context, same for every row) or a per-row ``pd.Series`` of symbols
(sector context, which depends on each row's sector).

## `stock_predictor/data/source.py`

Unified access to the market database.

All four supported storage layouts (a DuckDB file, a SQLite file, a
directory of Parquet files, or a directory of CSV files) are queried
through DuckDB, so every other module only ever has to write one dialect
of SQL and call :meth:`DataSource.query`.

### `class DataSource`

Thin wrapper around a DuckDB connection that can see every table.

Parameters
----------
config:
    The ``data_source`` section of the project config.

#### `__init__(self, config: DataSourceConfig) -> None`

#### `_attach(self) -> None`

#### `list_tables(self) -> list[str]`

#### `list_columns(self, table_name: str) -> list[str]`

#### `query(self, sql: str) -> pd.DataFrame`

Run arbitrary SQL against the attached tables and return a DataFrame.

#### `read_table(self, table_name: str, columns: list[str] | None = None, symbols: list[str] | None = None, date_column: str | None = 'date', start_date: str | None = None, end_date: str | None = None, max_rows: int | None = None) -> pd.DataFrame`

Read one table with optional column pruning and universe/date filters.

#### `close(self) -> None`

#### `__enter__(self) -> 'DataSource'`

#### `__exit__(self, *exc_info) -> None`

## `stock_predictor/data/synthetic.py`

Generates a small synthetic dataset that matches the project's expected
schema, so the whole pipeline (load -> preprocess -> train -> predict ->
plot) can be smoke-tested without a real market database.

This is *not* meant to produce realistic prices -- it only has to exercise
every join and every branch of the model. Run ``scripts/make_demo_data.py``
to write the tables out as CSV or Parquet files that ``DataSource`` can read.

#### `_business_days(start: str, end: str) -> pd.DatetimeIndex`

#### `generate_synthetic_tables(n_symbols: int = 40, start_date: str = '2018-01-01', end_date: str = '2023-12-31', seed: int = 0) -> dict[str, pd.DataFrame]`

Return a dict of table_name -> DataFrame covering every configured table.

#### `write_tables(tables: dict[str, pd.DataFrame], out_dir: str, fmt: str = 'csv') -> None`

Write every table to ``out_dir`` as CSV or Parquet for DataSource to read.

## `stock_predictor/features/__init__.py`

features subpackage of stock_predictor.

## `stock_predictor/features/dataset.py`

A :class:`torch.utils.data.Dataset` over the preprocessed modeling frame.

Each sample is a dict of tensors: one tensor per numeric branch (price,
regression, regression_indicators, ..., context_sector, ...), one tensor
per categorical column, an index-membership tensor, and the target/mask
tensors. Keeping branches separate (rather than concatenating everything
into one flat feature vector before the Dataset) is what lets the model
apply a distinct small encoder per branch -- see
``model/architecture.py``.

### `class SampleBatch`

A collated batch, as produced by :func:`collate_samples`.

#### `to(self, device: torch.device) -> 'SampleBatch'`

### `class StockDataset`

Wraps a preprocessed slice of :class:`LoadedDataset` for a DataLoader.

Transformation happens once, eagerly, in ``__init__`` (the tables here
are feature-engineered and modest in width, so this comfortably fits in
memory for the row counts this project targets -- see README for notes
on scaling to a much larger universe).

#### `__init__(self, dataset: LoadedDataset, preprocessor: Preprocessor, row_idx: np.ndarray) -> None`

#### `__len__(self) -> int`

#### `__getitem__(self, i: int) -> dict`

#### `collate_samples(items: list[dict]) -> SampleBatch`

## `stock_predictor/features/preprocessing.py`

Fits numeric scalers and categorical vocabularies on the training split
only, then applies them consistently to validation/test/inference data.

Fitting only on the training rows of each walk-forward split (never on
val/test/future rows) is what keeps the whole pipeline leakage-free -- see
``training/trainer.py`` for where a fresh :class:`Preprocessor` is fit per
split.

### `class ColumnStats`

### `class Vocabulary`

#### `build(cls, values: pd.Series, min_freq: int) -> 'Vocabulary'`

#### `encode(self, values: pd.Series) -> np.ndarray`

#### `size(self) -> int`

### `class Preprocessor`

Holds every fitted statistic needed to transform raw feature frames.

#### `fit(cls, dataset: LoadedDataset, train_idx: np.ndarray, config: PreprocessingConfig) -> 'Preprocessor'`

#### `transform_numeric(self, frame: pd.DataFrame) -> dict[str, np.ndarray]`

Standardize, clip and impute every numeric branch. Returns one
``(n_rows, n_branch_cols)`` float32 array per branch name.

#### `transform_categorical(self, frame: pd.DataFrame) -> dict[str, np.ndarray]`

#### `transform_index_membership(self, frame: pd.DataFrame, max_len: int = 6) -> np.ndarray`

Encode each row's list of index memberships as a fixed-width,
zero-padded array of vocabulary ids (mean-pooled by the model).

#### `transform_targets(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]`

Return (standardized_targets, observed_mask) as float32 arrays,
both shaped ``(n_rows, n_targets)``. Missing labels are masked out
of the loss rather than imputed.

#### `inverse_transform_targets(self, standardized: np.ndarray) -> np.ndarray`

Convert standardized target predictions back to the original scale.

#### `inverse_transform_target_std(self, standardized_sigma: np.ndarray) -> np.ndarray`

Convert a predicted standardized-space sigma to the original scale
(scaling only -- no mean shift is applied to a standard deviation).

#### `categorical_cardinalities(self) -> dict[str, int]`

#### `index_vocab_size(self) -> int`

#### `to_dict(self) -> dict[str, Any]`

#### `save(self, path: str | Path) -> None`

#### `load(cls, path: str | Path, config: PreprocessingConfig) -> 'Preprocessor'`

## `stock_predictor/features/schema.py`

Resolves which columns of a loaded table are ids / categorical / numeric.

This is what makes the "add or remove a column via the config file" project
requirement actually work: every branch of the model asks this module for
its column list rather than hard-coding names, and if the config leaves a
table's ``numeric_columns`` empty, :func:`resolve_numeric_columns` falls
back to auto-detecting every numeric, non-id column instead of failing.

#### `resolve_numeric_columns(df: pd.DataFrame, table_cfg: TableConfig, fallback_columns: list[str] | None = None) -> list[str]`

Return the numeric feature columns to use for ``table_cfg``.

Resolution order: explicit ``numeric_columns`` from config > explicit
``fallback_columns`` (used by regression_indicators_ma to default to the
same names as regression_indicators) > auto-detect every numeric dtype
column not already claimed as an id/categorical/target column.

#### `resolve_categorical_columns(df: pd.DataFrame, table_cfg: TableConfig) -> list[str]`

#### `resolve_target_columns(df: pd.DataFrame, table_cfg: TableConfig) -> list[str]`

Return the ma_future target columns to use.

Falls back to "every non-id numeric column" when ``target_columns`` is
left empty in the config, minus anything listed in ``exclude_columns``.

## `stock_predictor/inference/__init__.py`

inference subpackage of stock_predictor.

## `stock_predictor/inference/predict.py`

Runs a trained model over an explicit ``[start_date, end_date]`` window
(project requirement 4) and returns one row per (symbol, date) with each
target's predicted value, predicted uncertainty (sigma) and a derived
confidence score, plus the realized value/error when it's already known
(useful for backtesting a date range that has since resolved).

Saving the result into a timestamped ``predictions/`` folder together with
a copy of the config (project requirement 6) is handled by
``scripts/predict.py``, which calls :func:`predict_date_range`.

#### `_load_for_date_range(config: Config, start_date: str | None, end_date: str | None, symbols: list[str] | None, source = None) -> LoadedDataset`

Loads a :class:`LoadedDataset` restricted to a date range, without
touching the config file on disk (the override is applied to an
in-memory copy of ``config.data_source``).

#### `predict_date_range(config: Config, model: MultiTaskFusionNet, preprocessor: Preprocessor, start_date: str | None = None, end_date: str | None = None, symbols: list[str] | None = None, source = None) -> pd.DataFrame`

Predict every ``ma_future`` target for every (symbol, date) row in
``[start_date, end_date]``.

Uses the *training-time* ``preprocessor`` to standardize features and
encode categoricals (never refit here), so predictions are on the same
scale the model was trained on. Returns a wide DataFrame: one row per
(symbol, date), with ``{target}__pred``, ``{target}__sigma`` and
``{target}__confidence`` columns per target, plus ``{target}__actual``
/ ``{target}__abs_error`` wherever the realized value is already known.

## `stock_predictor/model/__init__.py`

model subpackage of stock_predictor.

## `stock_predictor/model/architecture.py`

The main torch model: :class:`MultiTaskFusionNet`.

Architecture, top to bottom (see docs/ARCHITECTURE.md for a diagram):

1.  **Categorical embeddings** -- ``sector_key``, ``industry_key``,
    ``security_type`` each get an ``nn.Embedding``; index memberships get a
    mean-pooled embedding (``model/embeddings.py``).
2.  **Branch encoders** -- every numeric table (price, regression,
    regression_indicators, regression_indicators_ma, ta__change,
    ta__ma_ratio, talib, context_sector, context_market,
    context_technical) gets its own small MLP encoder
    (``model/branches.py``). This is a *tabular* multi-branch design rather
    than a raw sequence model, because the source tables already encode
    history as engineered features (e.g. ``ta__change`` is literally a set
    of fixed-horizon lagged returns, and ``regression`` already summarizes
    a whole trend window as slope/R/n) -- re-deriving that from raw price
    sequences would add real complexity for little benefit.
3.  **Fusion** -- branch embeddings are combined with a gated-attention
    layer (each branch gets a learned, per-sample importance weight, so the
    model can down-weight branches that are missing or uninformative for a
    given row) and concatenated with the categorical embeddings, then fed
    through a shared trunk MLP.
4.  **Per-target heads** -- one small MLP head per ``ma_future`` column.
    Each head predicts a Gaussian ``(mu, log_var)`` when
    ``predict_uncertainty`` is enabled, which is what powers the
    "confidence" numbers used throughout ``plotting/`` and
    ``inference/predict.py``; otherwise it predicts a plain point estimate.

### `class PredictionHead`

One MLP head per target, predicting either (mu, log_var) or just mu.

#### `__init__(self, input_dim: int, hidden_dim: int, dropout: float, predict_uncertainty: bool) -> None`

#### `forward(self, x: torch.Tensor) -> torch.Tensor`

### `class MultiTaskFusionNet`

Multi-branch tabular network with per-target Gaussian heads.

Parameters
----------
numeric_branch_dims:
    ``{branch_name: n_columns}`` for every numeric branch, in the exact
    order the dataset will hand over tensors (see
    ``features/dataset.py``).
categorical_cardinalities:
    ``{column_name: vocab_size}`` for every embedded categorical column.
index_vocab_size:
    Vocabulary size for the index-membership embedding.
target_columns:
    Names of the ``ma_future`` columns being predicted, in order.

#### `__init__(self, numeric_branch_dims: dict[str, int], categorical_cardinalities: dict[str, int], index_vocab_size: int, target_columns: list[str], config: ModelConfig) -> None`

#### `encode(self, numeric: dict[str, torch.Tensor], categorical: dict[str, torch.Tensor], index_membership: torch.Tensor) -> torch.Tensor`

Return the shared trunk representation, before the target heads.

#### `forward(self, numeric: dict[str, torch.Tensor], categorical: dict[str, torch.Tensor], index_membership: torch.Tensor) -> dict[str, torch.Tensor]`

Returns ``{target_name: (batch, 2 or 1)}`` raw head outputs.

#### `predict_stacked(self, numeric: dict[str, torch.Tensor], categorical: dict[str, torch.Tensor], index_membership: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]`

Returns ``(mu, log_var)`` stacked over targets, each ``(batch, n_targets)``.

``log_var`` is ``None`` when the model was configured without
uncertainty heads.

## `stock_predictor/model/branches.py`

Per-branch numeric encoders and the gated-attention fusion layer that
combines every branch's embedding into one vector per (symbol, date) row.

Each configured table (price, regression, regression_indicators, ta__change,
context_sector, ...) gets its own small MLP encoder (a "branch"). This
keeps the model modular: adding or removing a table only changes which
branches exist, never the branches' internal code (see
``model/architecture.py`` for where branches are constructed from the
config-resolved column lists).

### `class BranchEncoder`

Standard MLP with LayerNorm + GELU + Dropout, one per numeric branch.

#### `__init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None`

#### `forward(self, x: torch.Tensor) -> torch.Tensor`

### `class GatedAttentionFusion`

Learns a per-branch, per-sample importance weight before concatenation.

Each branch embedding is scored by a small shared gate network; scores
are softmax-normalized across branches and used to scale each branch's
embedding before everything is concatenated into the fusion trunk's
input. This lets the model down-weight branches that are uninformative
or frequently missing (e.g. a symbol with no qualifying regression
trend) on a per-sample basis, instead of relying on a single fixed
linear combination for every row.

#### `__init__(self, branch_dim: int, n_branches: int) -> None`

#### `forward(self, branch_embeddings: list[torch.Tensor]) -> torch.Tensor`

## `stock_predictor/model/embeddings.py`

Embedding layers for categorical instrument metadata.

Two kinds of categorical information are embedded:

* Single-valued columns (``sector_key``, ``industry_key``, ``security_type``)
  each get their own :class:`torch.nn.Embedding` table.
* Index membership (a stock can belong to several indices at once) is
  embedded per index and mean-pooled over the (padded) memberships of each
  row, so a symbol in two indices gets the average of both embeddings.

### `class CategoricalEmbeddingGroup`

One embedding table per categorical column, concatenated on output.

#### `__init__(self, cardinalities: dict[str, int], embedding_dim: int) -> None`

#### `forward(self, categorical: dict[str, torch.Tensor]) -> torch.Tensor`

### `class IndexMembershipEmbedding`

Mean-pooled embedding over a padded list of index memberships.

Padding uses index 0 (reserved for ``<UNK>``/padding in the vocabulary
built by ``features/preprocessing.py``); padded slots are excluded from
the mean whenever at least one real membership is present.

#### `__init__(self, vocab_size: int, embedding_dim: int) -> None`

#### `forward(self, index_ids: torch.Tensor) -> torch.Tensor`

## `stock_predictor/model/losses.py`

Loss functions for multi-target regression with (a) missing labels
(some symbols simply don't have every ``ma_future`` column populated for
every date) and (b) optional heteroscedastic uncertainty output.

#### `masked_gaussian_nll(mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor`

Gaussian negative log-likelihood, averaged over observed entries only.

``mu``, ``log_var``, ``target``, ``mask``: all ``(batch, n_targets)``.
``log_var`` is clamped to keep training numerically stable early on,
when the head's weights are still close to their random init.

#### `masked_mse(mu: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor`

#### `per_target_masked_mse(mu: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor`

Per-target MSE, returned as a ``(n_targets,)`` tensor (for logging).

## `stock_predictor/plotting/__init__.py`

plotting subpackage of stock_predictor.

## `stock_predictor/plotting/prediction_plots.py`

Plots built from a prediction run's wide predictions frame (the output of
``inference/predict.py::predict_date_range``, as saved by
``scripts/predict.py``).

Per project requirements: plot each target's prediction confidence for
every symbol/date (requirement: "plot the confidence of each prediction"),
show every prediction parameter rather than only confidence (pred, sigma,
confidence, and actual/error when known), and make it easy to spot the
symbols with the best predicted output.

#### `_safe(name: str) -> str`

#### `plot_confidence_over_time(df: pd.DataFrame, target: str, out_dir: Path) -> Path`

Every (symbol, date) prediction for one target, plotted as date (x)
vs. predicted value (y), colored by confidence -- shows at a glance
where in time the model is more/less sure of itself.

#### `plot_best_symbols(df: pd.DataFrame, target: str, out_dir: Path, as_of_date: str | None = None, top_n: int = 20) -> Path`

The single "where should I look first" plot: for one date (the most
recent date in the predictions by default), ranks symbols by predicted
value and shows predicted value, sigma (as an error bar) and confidence
(as bar color) together -- every prediction parameter in one place, not
just confidence, per project requirement.

#### `plot_prediction_detail_grid(df: pd.DataFrame, target: str, out_dir: Path, as_of_date: str | None = None, top_n: int = 20) -> Path`

A small-multiples view of every prediction parameter (pred, sigma,
confidence, and actual-vs-pred error when available) for the same
ranked symbol list as :func:`plot_best_symbols`, so no parameter is
hidden behind just the headline confidence number.

#### `plot_calibration(df: pd.DataFrame, target: str, out_dir: Path) -> Path | None`

Predicted vs. actual scatter (only for rows with a realized value),
colored by confidence -- a well-calibrated model should cluster near
the y = x line, especially for its highest-confidence predictions.

#### `generate_prediction_plots(df: pd.DataFrame, target_columns: list[str], out_dir: str | Path, top_n: int = 20) -> list[Path]`

Runs every plotting function above for every target and returns the
list of written PNG paths.

## `stock_predictor/plotting/training_plots.py`

Plots built from a training run's ``metrics.json`` (see
``training/trainer.py::training_output_to_dict`` for the schema, and
``scripts/plot_training.py`` for the CLI entrypoint).

Every function here takes the already-loaded run dict and an output
directory, and returns the list of PNG paths it wrote, so
``scripts/plot_training.py`` can print a summary and (optionally) publish
them.

#### `_all_splits(run: dict) -> list[dict]`

CV splits plus the final production split, in chronological order.

#### `plot_loss_curves(run: dict, out_dir: Path) -> list[Path]`

One figure per split: train loss vs. validation loss over epochs.

#### `plot_target_metric_curves(run: dict, out_dir: Path) -> list[Path]`

For each target variable, one figure with MAE/RMSE/R2/directional
accuracy over epochs for the production ("final") split, with each CV
split overlaid faintly for comparison.

#### `plot_split_timeline(run: dict, out_dir: Path) -> Path`

A Gantt-style overview of every split's train/val date ranges, plus
the final held-out test block, so the walk-forward layout is easy to
sanity-check at a glance.

#### `plot_test_summary(run: dict, out_dir: Path) -> Path`

Bar chart of held-out test-set metrics, one group of bars per target.

#### `plot_all_training_results(run: dict, out_dir: str | Path) -> list[Path]`

Runs every plotting function above and returns all written paths.

#### `_safe(name: str) -> str`

## `stock_predictor/splits/__init__.py`

splits subpackage of stock_predictor.

## `stock_predictor/splits/time_series_split.py`

Walk-forward, expanding-window time series splitting with an embargo gap.

Because ``ma_future`` targets are computed from prices *after* each row's
date, a plain chronological split can leak information: a training row
dated just before the validation cutoff can have a label that depends on
prices that fall inside the validation window. :class:`WalkForwardSplitter`
removes an ``embargo_days`` buffer immediately before each validation/test
block to prevent that (the same idea as the "purge/embargo" technique used
in "Advances in Financial Machine Learning", Lopez de Prado).

Each split's training window *expands* to include all prior data (rather
than a fixed-size rolling window), which is what the project spec means by
"multiple splits of previous data" -- the model sees progressively more
market history/regimes across splits rather than a single arbitrary window,
which reduces the chance of overfitting to one regime.

### `class Split`

One walk-forward split's index arrays into the source DataFrame.

### `class WalkForwardSplitter`

Builds expanding-window train/val splits, plus one final held-out test block.

Layout on the date axis (oldest -> newest), for ``n_splits = 3``::

    [ train_1 ][embargo][ val_1 ]
    [   train_2        ][embargo][ val_2 ]
    [        train_3            ][embargo][ val_3 ]
    [             final train (all splits)][embargo][   TEST   ]

Splits 1..n_splits are used for walk-forward model selection / early
stopping (see ``training/trainer.py``). The test block is only ever
scored once, after training is finished.

#### `__init__(self, config: SplittingConfig) -> None`

#### `_pre_test_and_test_dates(self, dates: pd.Series) -> tuple[np.ndarray, np.ndarray]`

#### `test_idx(self, dates: pd.Series) -> np.ndarray`

Row indices belonging to the final held-out test block.

#### `pre_test_idx(self, dates: pd.Series) -> np.ndarray`

Row indices for every date usable for training/validation
(i.e. everything before the test block's embargo gap).

#### `final_split(self, dates: pd.Series) -> Split`

One more expanding-window split covering *all* pre-test history:
train on everything except the most recent ``val_days`` (minus the
embargo gap), validate on that final block. Used to pick an epoch
count / apply early stopping for the production model that is then
used for held-out test scoring and for real predictions.

#### `split(self, dates: pd.Series) -> tuple[list[Split], np.ndarray, np.ndarray]`

Compute walk-forward CV splits for a DataFrame's ``date`` column.

Parameters
----------
dates:
    The ``date`` column of the frame being split (not necessarily
    sorted or unique -- multiple symbols share each date).

Returns
-------
(splits, test_idx, held_out_idx)
    ``splits`` is the list of walk-forward train/val splits.
    ``test_idx`` indexes the final held-out test block.
    ``held_out_idx`` indexes the embargoed rows dropped between
    train/val/test blocks (kept only for reporting).

## `stock_predictor/training/__init__.py`

training subpackage of stock_predictor.

## `stock_predictor/training/metrics.py`

Per-target regression metrics, computed in the original (unstandardized)
units of each ``ma_future`` column so they're directly interpretable.

#### `regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict[str, float]`

Metrics for one target column: MAE, RMSE, R^2, Pearson correlation,
and directional accuracy (how often the predicted sign matches the
actual sign -- a common sanity check for return-like targets).

All inputs are 1-D arrays of equal length; ``mask`` is 1.0 where the
label is observed and 0.0 where it's missing.

#### `metrics_for_all_targets(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, target_columns: list[str]) -> dict[str, dict[str, float]]`

``regression_metrics`` applied independently to each target column.

## `stock_predictor/training/trainer.py`

Orchestrates walk-forward training.

:meth:`Trainer.run` does three things, in order:

1.  For each walk-forward split from :class:`WalkForwardSplitter`, fits a
    fresh :class:`Preprocessor` on that split's training rows only, trains
    a fresh model with early stopping on that split's validation rows, and
    records per-epoch metrics. This measures how well the whole pipeline
    (feature scaling + architecture + hyperparameters) generalizes across
    different historical market regimes.
2.  Fits one more "final" split covering *all* pre-test history (see
    ``WalkForwardSplitter.final_split``) to produce the production model:
    the one actually used for test-set scoring and for
    ``inference/predict.py``.
3.  Scores the production model once on the held-out test block.

Every setting and every epoch's metrics are collected into a
JSON-serializable dict by :func:`training_output_to_dict`, written out by
``scripts/train.py``.

#### `resolve_device(device_setting: str) -> torch.device`

### `class EpochMetrics`

### `class SplitResult`

### `class RunResult`

#### `build_model(preprocessor: Preprocessor, model_config) -> MultiTaskFusionNet`

Constructs a model whose branch/embedding sizes match a fitted
preprocessor. Used both by :class:`Trainer` and by ``scripts/predict.py``
to rebuild the architecture before loading saved weights.

### `class Trainer`

Runs walk-forward CV, a final production fit, and test scoring.

#### `__init__(self, config: Config, dataset: LoadedDataset) -> None`

#### `run(self) -> RunResult`

#### `_build_model(self, preprocessor: Preprocessor) -> MultiTaskFusionNet`

#### `_make_loader(self, preprocessor: Preprocessor, row_idx: np.ndarray, shuffle: bool) -> TorchDataLoader`

#### `_compute_loss(self, mu: torch.Tensor, log_var: torch.Tensor | None, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor`

#### `_run_split(self, split: Split, max_epochs: int, return_model: bool = False) -> Any`

#### `_train_one_epoch(self, model: MultiTaskFusionNet, loader: TorchDataLoader, optimizer: torch.optim.Optimizer) -> float`

#### `_evaluate_loader(self, model: MultiTaskFusionNet, loader: TorchDataLoader, preprocessor: Preprocessor) -> tuple[float, dict[str, dict[str, float]], dict[str, np.ndarray]]`

#### `_evaluate_on_index(self, model: MultiTaskFusionNet, preprocessor: Preprocessor, row_idx: np.ndarray, tag: str) -> tuple[dict[str, dict[str, float]], int]`

#### `training_output_to_dict(run: RunResult, config: Config) -> dict[str, Any]`

Everything about a training run that should end up in the run's
``metrics.json`` (per project requirement: settings like epoch count and
split sizes, plus every metric, for every training run).

## `stock_predictor/utils/__init__.py`

utils subpackage of stock_predictor.

## `stock_predictor/utils/io.py`

Small filesystem helpers shared by the training and prediction scripts.

Centralizing "make a timestamped run folder and drop a copy of the config
in it" here means both ``scripts/train.py`` and ``scripts/predict.py`` (per
project requirement 6) get the same, consistent layout.

#### `make_timestamped_dir(base_dir: str | Path, prefix: str = 'run') -> Path`

Create and return ``base_dir/<prefix>_<YYYYmmdd_HHMMSS>/``.

A numeric suffix is appended if the exact timestamp already exists
(e.g. two runs started within the same second).

#### `save_json(data: dict[str, Any], path: str | Path) -> None`

#### `_json_default(obj: Any) -> Any`

#### `copy_config(config_path: str | Path, dest_dir: str | Path) -> Path`

Copy the exact config file used for a run into its run folder, per
project requirement 6 (trackability of which settings produced which
predictions).

## `stock_predictor/utils/logging_utils.py`

Consistent logging setup for CLI scripts.

#### `setup_logging(level: int = logging.INFO) -> None`

## `scripts/generate_reference_docs.py`

Generates ``docs/REFERENCE.md``: a per-file, per-function/class API
reference, extracted directly from source docstrings and signatures via
the ``ast`` module (so it can't drift out of sync with argument names).

Run whenever the code changes and you want the reference doc refreshed::

    python scripts/generate_reference_docs.py

#### `_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str`

#### `_doc_summary(node: ast.AST) -> str`

#### `_document_function(node: ast.FunctionDef | ast.AsyncFunctionDef, indent: str = '') -> list[str]`

#### `_document_class(node: ast.ClassDef) -> list[str]`

#### `_document_module(path: Path) -> list[str]`

#### `main() -> None`

## `scripts/make_demo_data.py`

Writes a synthetic demo dataset to disk so the rest of the pipeline can
be run end-to-end without a real market database.

Usage
-----
    python scripts/make_demo_data.py --out-dir data/demo_csv --format csv

Then point ``config/config.yaml``'s ``data_source`` at it, e.g.::

    data_source:
      type: csv_dir
      path: data/demo_csv

#### `main() -> None`

## `scripts/plot_predictions.py`

Regenerates prediction plots from an existing run's ``predictions.csv``
without re-running the model. Usage::

    python scripts/plot_predictions.py predictions/predict_20240101_120000/predictions.csv

#### `_infer_target_columns(df: pd.DataFrame) -> list[str]`

#### `main() -> None`

## `scripts/plot_training.py`

Regenerates training plots from an existing run's ``metrics.json``
without retraining. Usage::

    python scripts/plot_training.py artifacts/train_20240101_120000/metrics.json

#### `main() -> None`

## `scripts/predict.py`

Runs a trained model over an explicit date range and writes a timestamped
run folder under ``predictions_dir`` (project requirement 6) containing:

    config.yaml         -- exact copy of the config used for this run
    predictions.csv       -- one row per (symbol, date), every target's
                              pred / sigma / confidence / actual / abs_error
    summary.json          -- row counts, date range, per-target summary stats
    plots/                -- confidence + detail plots (prediction_plots.py)

Usage
-----
    python scripts/predict.py \
        --config config/config.yaml \
        --run-dir artifacts/train_20240101_120000 \
        --start-date 2024-01-01 --end-date 2024-01-31

#### `main() -> None`

## `scripts/train.py`

Trains the model per ``config.yaml``'s walk-forward + hyperparameter
settings, and writes one timestamped run folder under
``artifacts_dir`` containing:

    config.yaml       -- exact copy of the config used for this run
    metrics.json       -- every setting + every epoch's metrics (all splits)
    model.pt            -- production model weights (state_dict)
    preprocessor.json  -- fitted scalers/vocabularies for the production model
    plots/              -- training curve plots (see plotting/training_plots.py)

Usage
-----
    python scripts/train.py --config config/config.yaml

#### `main() -> None`

## `tests/test_loader_logic.py`

Exercises DataLoader's join logic against the synthetic dataset.

This test fakes out :class:`stock_predictor.data.source.DataSource` with a
plain in-memory lookup so it can run in environments without the ``duckdb``
package installed -- it is only the SQL plumbing in ``DataSource`` that
needs duckdb; the join/branch logic in ``DataLoader`` is pure pandas and is
what this test actually cares about.

### `class FakeDataSource`

Mimics the subset of DataSource.read_table used by DataLoader.

#### `__init__(self, tables: dict[str, pd.DataFrame]) -> None`

#### `read_table(self, table_name: str, columns = None, symbols = None, date_column: str | None = 'date', start_date = None, end_date = None, max_rows = None) -> pd.DataFrame`

#### `main() -> None`
