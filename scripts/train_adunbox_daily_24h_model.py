from __future__ import annotations

"""Compatibility layer for the production 24h scorer.

The runtime scorer historically imports ``train_adunbox_daily_24h_model`` for
feature engineering helpers. In the cleaned production package, the final model
is produced by ``train_adunbox_daily_24h_full_db_production``. This module keeps
the scorer API stable while routing all calls to the production implementation.
"""

import os
from pathlib import Path

import pandas as pd

import train_adunbox_daily_24h_full_db_production as production


ENTITY_COLS = production.base.ENTITY_COLS
RAW_TARGETS = production.base.RAW_TARGETS
DEFAULT_DAILY_INPUT_PATH = Path(
    os.getenv("ADUNBOX_DAILY_INPUT", r"H:\adunbox_daily_breakdown_kpis.csv")
)
MODEL_DIR = production.MODEL_DIR


def load_daily(
    force_rebuild_from_hourly: bool = False,
    daily_input_path: Path | str = DEFAULT_DAILY_INPUT_PATH,
) -> tuple[pd.DataFrame, str]:
    if force_rebuild_from_hourly:
        raise ValueError("24h production scoring expects daily input, not hourly rebuild.")
    path = Path(daily_input_path)
    daily = production.load_single_daily_flexible(path)
    return daily, str(path)


def add_features_and_targets(daily: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    return production.build_features_production_safe(daily)


def derive_kpis(preds: pd.DataFrame) -> pd.DataFrame:
    return production.base.derive_kpis(preds)
