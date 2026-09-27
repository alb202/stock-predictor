"""Typed configuration objects for the whole project.

Every other module receives a :class:`Config` instance instead of reading
``config.yaml`` itself. This keeps column lists, hyperparameters and paths
in one place (the YAML file) while still giving the rest of the codebase
type-checked, autocomplete-friendly access to every setting.

Call :func:`load_config` once (e.g. in a CLI script) and pass the result
around.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
# Sub-sections (mirror the top-level keys of config.yaml)
# --------------------------------------------------------------------------- #


@dataclass
class ProjectConfig:
    name: str = "stock_predictor"
    random_seed: int = 1337
    predictions_dir: str = "predictions"
    artifacts_dir: str = "artifacts"


@dataclass
class DataSourceConfig:
    type: str = "duckdb"
    path: str = "data/market.duckdb"
    symbols: list[str] = field(default_factory=list)
    start_date: str | None = None
    end_date: str | None = None
    max_rows: int | None = None


@dataclass
class TableConfig:
    """Column roles for a single database table.

    ``id_columns`` are join keys and are never treated as features.
    ``categorical_columns`` are embedded; ``numeric_columns`` are
    standardized. An empty ``numeric_columns`` list means "auto-detect every
    non-id numeric column at load time" (see ``features/schema.py``).
    """

    table_name: str
    id_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    # ma_future only:
    target_columns: list[str] = field(default_factory=list)
    exclude_columns: list[str] = field(default_factory=list)


@dataclass
class EtfMappingsConfig:
    sector_etfs: dict[str, str] = field(default_factory=dict)
    market_etfs: dict[str, str] = field(default_factory=dict)
    technical_etfs: dict[str, str] = field(default_factory=dict)
    context_feature_tables: list[str] = field(default_factory=lambda: ["ta__change"])

    def all_context_symbols_config(self) -> dict[str, str]:
        """Flatten every mapped ETF into a single ``{context_key: symbol}`` dict."""
        merged: dict[str, str] = {}
        merged.update({f"sector::{k}": v for k, v in self.sector_etfs.items()})
        merged.update({f"market::{k}": v for k, v in self.market_etfs.items()})
        merged.update({f"technical::{k}": v for k, v in self.technical_etfs.items()})
        return merged


@dataclass
class RegressionSelectionConfig:
    r_power: float = 1.0
    n_power: float = 1.0
    min_n: int = 10


@dataclass
class PreprocessingConfig:
    numeric_scaler: str = "standard"
    clip_std: float = 8.0
    fill_value_after_scaling: float = 0.0
    min_category_freq: int = 5


@dataclass
class SplittingConfig:
    n_splits: int = 5
    val_days: int = 63
    test_days: int = 63
    embargo_days: int = 21
    min_train_days: int = 252


@dataclass
class ModelConfig:
    categorical_embedding_dim: int = 8
    branch_hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    fusion_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    fusion_attention: bool = True
    head_hidden_dim: int = 32
    dropout: float = 0.2
    predict_uncertainty: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 1024
    max_epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 8
    lr_scheduler_patience: int = 4
    lr_scheduler_factor: float = 0.5
    grad_clip_norm: float = 1.0
    num_workers: int = 2
    loss: str = "gaussian_nll"
    device: str = "auto"


@dataclass
class Config:
    project: ProjectConfig
    data_source: DataSourceConfig
    tables: dict[str, TableConfig]
    etf_mappings: EtfMappingsConfig
    regression_selection: RegressionSelectionConfig
    preprocessing: PreprocessingConfig
    splitting: SplittingConfig
    model: ModelConfig
    training: TrainingConfig

    # Kept so a run's exact settings (including anything not yet promoted to
    # a dataclass field) can always be written out verbatim to run metadata.
    raw: dict[str, Any] = field(default_factory=dict)

    def table(self, name: str) -> TableConfig:
        """Look up a table's column configuration by its config key."""
        try:
            return self.tables[name]
        except KeyError as exc:
            raise KeyError(
                f"No table config named '{name}'. Known tables: "
                f"{sorted(self.tables)}"
            ) from exc


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _dataclass_from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Build a dataclass instance from a dict, ignoring unknown keys.

    Keeping this permissive (rather than failing on extra YAML keys) makes
    it easy to add new config fields and roll them into the dataclasses
    later without breaking existing config files.
    """
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in data.items() if k in field_names}
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load and validate ``config.yaml`` into a :class:`Config` object."""
    path = Path(path)
    with path.open("r") as fh:
        raw = yaml.safe_load(fh)
    raw = raw or {}

    tables_raw = raw.get("tables", {})
    tables = {
        key: _dataclass_from_dict(TableConfig, {"table_name": key, **val})
        for key, val in tables_raw.items()
    }

    cfg = Config(
        project=_dataclass_from_dict(ProjectConfig, raw.get("project", {})),
        data_source=_dataclass_from_dict(DataSourceConfig, raw.get("data_source", {})),
        tables=tables,
        etf_mappings=_dataclass_from_dict(EtfMappingsConfig, raw.get("etf_mappings", {})),
        regression_selection=_dataclass_from_dict(
            RegressionSelectionConfig, raw.get("regression_selection", {})
        ),
        preprocessing=_dataclass_from_dict(PreprocessingConfig, raw.get("preprocessing", {})),
        splitting=_dataclass_from_dict(SplittingConfig, raw.get("splitting", {})),
        model=_dataclass_from_dict(ModelConfig, raw.get("model", {})),
        training=_dataclass_from_dict(TrainingConfig, raw.get("training", {})),
        raw=copy.deepcopy(raw),
    )
    return cfg


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Return the config's original YAML dict (for writing run metadata)."""
    return copy.deepcopy(cfg.raw)
