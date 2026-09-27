"""Walk-forward, expanding-window time series splitting with an embargo gap.

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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_predictor.config import SplittingConfig


@dataclass(frozen=True)
class Split:
    """One walk-forward split's index arrays into the source DataFrame."""

    split_index: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    train_date_range: tuple[str, str]
    val_date_range: tuple[str, str]


class WalkForwardSplitter:
    """Builds expanding-window train/val splits, plus one final held-out test block.

    Layout on the date axis (oldest -> newest), for ``n_splits = 3``::

        [ train_1 ][embargo][ val_1 ]
        [   train_2        ][embargo][ val_2 ]
        [        train_3            ][embargo][ val_3 ]
        [             final train (all splits)][embargo][   TEST   ]

    Splits 1..n_splits are used for walk-forward model selection / early
    stopping (see ``training/trainer.py``). The test block is only ever
    scored once, after training is finished.
    """

    def __init__(self, config: SplittingConfig) -> None:
        self.config = config

    # -- shared date-boundary computation -------------------------------- #

    def _pre_test_and_test_dates(self, dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        unique_dates = np.sort(dates.unique())
        n_unique = len(unique_dates)

        test_start_pos = n_unique - cfg.test_days
        if test_start_pos <= 0:
            raise ValueError(
                f"Not enough unique dates ({n_unique}) for test_days={cfg.test_days}."
            )
        test_dates = unique_dates[test_start_pos:]
        pre_test_embargo_start = max(0, test_start_pos - cfg.embargo_days)
        pre_test_dates = unique_dates[:pre_test_embargo_start]

        if len(pre_test_dates) < cfg.min_train_days + cfg.val_days + cfg.embargo_days:
            raise ValueError(
                "Not enough history before the test block to build even one "
                f"walk-forward split: have {len(pre_test_dates)} usable dates, need "
                f"at least {cfg.min_train_days + cfg.val_days + cfg.embargo_days}."
            )
        return pre_test_dates, test_dates

    def test_idx(self, dates: pd.Series) -> np.ndarray:
        """Row indices belonging to the final held-out test block."""
        _, test_dates = self._pre_test_and_test_dates(dates)
        return np.nonzero(dates.isin(test_dates).to_numpy())[0]

    def pre_test_idx(self, dates: pd.Series) -> np.ndarray:
        """Row indices for every date usable for training/validation
        (i.e. everything before the test block's embargo gap)."""
        pre_test_dates, _ = self._pre_test_and_test_dates(dates)
        return np.nonzero(dates.isin(pre_test_dates).to_numpy())[0]

    def final_split(self, dates: pd.Series) -> Split:
        """One more expanding-window split covering *all* pre-test history:
        train on everything except the most recent ``val_days`` (minus the
        embargo gap), validate on that final block. Used to pick an epoch
        count / apply early stopping for the production model that is then
        used for held-out test scoring and for real predictions.
        """
        cfg = self.config
        pre_test_dates, _ = self._pre_test_and_test_dates(dates)
        n_pre_test = len(pre_test_dates)
        val_start_pos = n_pre_test - cfg.val_days
        train_end_pos = val_start_pos - cfg.embargo_days
        if train_end_pos < cfg.min_train_days or val_start_pos <= 0:
            raise ValueError("Not enough pre-test history to build the final split.")

        train_dates = pre_test_dates[:train_end_pos]
        val_dates = pre_test_dates[val_start_pos:n_pre_test]
        train_mask = dates.isin(train_dates).to_numpy()
        val_mask = dates.isin(val_dates).to_numpy()
        return Split(
            split_index=-1,
            train_idx=np.nonzero(train_mask)[0],
            val_idx=np.nonzero(val_mask)[0],
            train_date_range=(str(train_dates[0]), str(train_dates[-1])),
            val_date_range=(str(val_dates[0]), str(val_dates[-1])),
        )

    def split(self, dates: pd.Series) -> tuple[list[Split], np.ndarray, np.ndarray]:
        """Compute walk-forward CV splits for a DataFrame's ``date`` column.

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
        """
        cfg = self.config
        pre_test_dates, test_dates = self._pre_test_and_test_dates(dates)
        unique_dates = np.sort(dates.unique())

        splits: list[Split] = []
        n_pre_test = len(pre_test_dates)
        # Evenly space `n_splits` validation blocks across the usable
        # (pre-test) history, each preceded by an embargo gap. `base_offset`
        # is where the very first validation block can start once both the
        # minimum training window *and* its embargo gap are reserved, so
        # every split (including split 0) keeps a full min_train_days of
        # training data even after the embargo is subtracted back out.
        base_offset = cfg.min_train_days + cfg.embargo_days
        last_i = max(cfg.n_splits - 1, 1)
        span_for_steps = n_pre_test - base_offset - cfg.val_days
        step = max(1, span_for_steps // last_i) if cfg.n_splits > 1 else 1

        for i in range(cfg.n_splits):
            val_end_pos = min(base_offset + cfg.val_days + i * step, n_pre_test)
            val_start_pos = val_end_pos - cfg.val_days
            train_end_pos = val_start_pos - cfg.embargo_days
            if train_end_pos < cfg.min_train_days or val_start_pos <= 0:
                continue

            train_dates = pre_test_dates[:train_end_pos]
            val_dates = pre_test_dates[val_start_pos:val_end_pos]
            if len(val_dates) == 0:
                continue

            train_mask = dates.isin(train_dates).to_numpy()
            val_mask = dates.isin(val_dates).to_numpy()
            splits.append(
                Split(
                    split_index=len(splits),
                    train_idx=np.nonzero(train_mask)[0],
                    val_idx=np.nonzero(val_mask)[0],
                    train_date_range=(str(train_dates[0]), str(train_dates[-1])),
                    val_date_range=(str(val_dates[0]), str(val_dates[-1])),
                )
            )

        test_mask = dates.isin(test_dates).to_numpy()
        test_idx = np.nonzero(test_mask)[0]

        embargoed_dates = np.setdiff1d(unique_dates, np.concatenate([pre_test_dates, test_dates]))
        held_out_mask = dates.isin(embargoed_dates).to_numpy()
        held_out_idx = np.nonzero(held_out_mask)[0]

        if not splits:
            raise ValueError(
                "No valid walk-forward splits could be built with the given "
                "splitting config -- try lowering n_splits, val_days or min_train_days."
            )

        return splits, test_idx, held_out_idx
