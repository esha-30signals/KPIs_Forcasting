from __future__ import annotations

"""Hierarchical benchmark fallbacks for low-history forecasts.

These helpers are intentionally model-agnostic and can be used by both the 24h
daily scorer and the 6h hourly scorer. The rule is:

1. enough entity history -> keep model prediction
2. else adset same-window benchmark
3. else campaign same-window benchmark
4. else account same-window benchmark
5. else insufficient_history / monitoring
"""

import numpy as np
import pandas as pd


def peer_benchmark(
    frame: pd.DataFrame,
    value: pd.Series,
    group_cols: list[str],
    entity_col: str = "ad_id",
    min_peer_entities: int = 2,
) -> tuple[pd.Series, pd.Series]:
    if not all(col in frame.columns for col in group_cols):
        empty = pd.Series(0.0, index=frame.index, dtype="float64")
        ok = pd.Series(False, index=frame.index)
        return empty, ok

    key = frame[group_cols].astype(str).agg("||".join, axis=1)
    active_value = pd.to_numeric(value, errors="coerce").fillna(0.0).where(value > 0.0, np.nan)
    peer_sum = active_value.groupby(key).transform("sum").fillna(0.0)
    peer_count = active_value.groupby(key).transform("count").fillna(0.0)
    self_active = active_value.notna().astype("float64")
    peer_sum_ex_self = (peer_sum - active_value.fillna(0.0)).clip(lower=0.0)
    peer_count_ex_self = (peer_count - self_active).clip(lower=0.0)
    benchmark = pd.Series(
        np.divide(
            peer_sum_ex_self,
            peer_count_ex_self,
            out=np.zeros(len(frame), dtype="float64"),
            where=peer_count_ex_self.to_numpy() >= min_peer_entities,
        ),
        index=frame.index,
    )
    return benchmark.fillna(0.0), peer_count_ex_self >= min_peer_entities


def choose_hierarchical_forecast(
    frame: pd.DataFrame,
    model_prediction: pd.Series,
    same_window_history_value: pd.Series,
    needs_fallback: pd.Series,
    min_peer_entities: int = 2,
) -> tuple[pd.Series, pd.Series]:
    adset_benchmark, adset_ok = peer_benchmark(
        frame,
        same_window_history_value,
        ["account_id", "campaign_id", "adset_id"],
        min_peer_entities=min_peer_entities,
    )
    campaign_benchmark, campaign_ok = peer_benchmark(
        frame,
        same_window_history_value,
        ["account_id", "campaign_id"],
        min_peer_entities=min_peer_entities,
    )
    account_benchmark, account_ok = peer_benchmark(
        frame,
        same_window_history_value,
        ["account_id"],
        min_peer_entities=min_peer_entities,
    )

    fallback = np.select(
        [adset_ok, campaign_ok, account_ok],
        [adset_benchmark, campaign_benchmark, account_benchmark],
        default=0.0,
    )
    source = np.select(
        [adset_ok, campaign_ok, account_ok],
        ["adset_same_window_benchmark", "campaign_same_window_benchmark", "account_same_window_benchmark"],
        default="insufficient_history",
    )
    forecast = np.where(needs_fallback, fallback, pd.to_numeric(model_prediction, errors="coerce").fillna(0.0))
    return pd.Series(forecast, index=frame.index).fillna(0.0), pd.Series(source, index=frame.index)
