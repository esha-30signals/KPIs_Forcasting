from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn._loss as sklearn_loss

import train_adunbox_daily_24h_model as daily24

sys.modules.setdefault("_loss", sklearn_loss)


BASE_DIR = Path(os.getenv("ADUNBOX_PROJECT_DIR", Path(__file__).resolve().parents[1]))
OUTPUT_PATH = BASE_DIR / "adunbox_daily_24h_latest_forecasts.csv"
SUMMARY_PATH = BASE_DIR / "adunbox_daily_24h_latest_forecasts__summary.txt"
MIN_24H_KPI_IMPRESSIONS = 100.0
MIN_24H_KPI_CLICKS = 5.0
MIN_24H_RECENT_SPEND_MEAN = 1.0
MIN_24H_HISTORY_DAYS = 14
MIN_STABLE_HISTORY_DAYS = 5
MIN_REVENUE_HISTORY_DAYS = 3
MIN_CONVERSION_HISTORY_DAYS = 3
SPIKE_RATIO_HIGH = 3.0
LATEST_FEATURE_HISTORY_DAYS = int(os.getenv("ADUNBOX_24H_SCORE_HISTORY_DAYS", "16"))
FAST_SCORE_CSV = os.getenv("ADUNBOX_24H_FAST_SCORE", "true").strip().lower() in {"1", "true", "yes", "y"}
FAST_SCORE_CHUNKSIZE = int(os.getenv("ADUNBOX_24H_SCORE_CHUNKSIZE", "200000"))
SCORE_ACCOUNT_IDS = {
    item.strip()
    for item in os.getenv("ADUNBOX_SCORE_ACCOUNT_IDS", "").split(",")
    if item.strip()
}
SCORE_MAX_ADS = int(os.getenv("ADUNBOX_SCORE_MAX_ADS", "0") or "0")
MIN_FALLBACK_PEER_ADS = int(os.getenv("ADUNBOX_MIN_FALLBACK_PEER_ADS", "2"))
MIN_MODEL_HISTORY_DAYS_FOR_POINT = int(os.getenv("ADUNBOX_MIN_MODEL_HISTORY_DAYS_FOR_POINT", "7"))
FEATURE_CACHE_PATH = Path(os.getenv("ADUNBOX_24H_FEATURE_CACHE", BASE_DIR / "outputs" / "adunbox_24h_latest_feature_cache.joblib"))
REUSE_FEATURE_CACHE = os.getenv("ADUNBOX_REUSE_FEATURE_CACHE", "false").strip().lower() in {"1", "true", "yes", "y"}
WRITE_FEATURE_CACHE = os.getenv("ADUNBOX_WRITE_FEATURE_CACHE", "true").strip().lower() in {"1", "true", "yes", "y"}


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(frame.get(col, pd.Series(default, index=frame.index)), errors="coerce").fillna(default)


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce").fillna(0.0)
    den = pd.to_numeric(den, errors="coerce").fillna(0.0)
    return pd.Series(np.divide(num, den, out=np.zeros(len(num), dtype="float64"), where=den.to_numpy() > 0), index=num.index)


def _normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).replace({"nan": "", "None": "", "<NA>": ""})


def _daily_usecols(path: Path) -> tuple[list[str], dict[str, str]]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    conversion_col = "conversions" if "conversions" in header else ("tracker_conversions" if "tracker_conversions" in header else "tracker_conversion")
    revenue_col = "conversions_value" if "conversions_value" in header else "tracker_revenue"
    usecols = ["entity_type", "date", "timezone", *daily24.ENTITY_COLS, "spend", "impressions", "inline_link_clicks", conversion_col, revenue_col]
    rename_map = {}
    if conversion_col != "tracker_conversions":
        rename_map[conversion_col] = "tracker_conversions"
    if revenue_col != "tracker_revenue":
        rename_map[revenue_col] = "tracker_revenue"
    return [col for col in usecols if col in header], rename_map


def _prepare_daily_chunk(chunk: pd.DataFrame, rename_map: dict[str, str]) -> pd.DataFrame:
    if "entity_type" in chunk.columns:
        chunk = chunk[chunk["entity_type"].astype(str).str.lower().eq("ad")].copy()
    else:
        chunk = chunk.copy()
    if chunk.empty:
        return chunk
    if rename_map:
        chunk = chunk.rename(columns=rename_map)
    if "timezone" not in chunk.columns:
        chunk["timezone"] = ""
    for col in daily24.ENTITY_COLS:
        chunk[col] = _normalize_id(chunk[col])
    if SCORE_ACCOUNT_IDS:
        chunk = chunk[chunk["account_id"].isin(SCORE_ACCOUNT_IDS)].copy()
        if chunk.empty:
            return chunk
    chunk["timezone"] = chunk["timezone"].fillna("").astype(str)
    chunk["local_date"] = pd.to_datetime(chunk["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    chunk = chunk[chunk["local_date"].notna()].copy()
    for col in daily24.RAW_TARGETS:
        chunk[col] = pd.to_numeric(chunk.get(col, 0.0), errors="coerce").fillna(0.0).astype("float32")
    return chunk[["local_date", "timezone", *daily24.ENTITY_COLS, *daily24.RAW_TARGETS]]


def _load_latest_context_daily(daily_input: Path) -> tuple[pd.DataFrame, str, pd.DataFrame | None]:
    """Low-memory CSV loader for latest production scoring.

    Pass 1 builds compact full-history per-ad summaries and finds the latest
    date. Pass 2 keeps only the recent daily rows needed for lag/rolling
    features. This trades a second sequential disk read for much lower RAM.
    """
    usecols, rename_map = _daily_usecols(daily_input)
    summary_parts: list[pd.DataFrame] = []
    max_date: pd.Timestamp | None = None

    for chunk in pd.read_csv(daily_input, usecols=lambda c: c in usecols, chunksize=FAST_SCORE_CHUNKSIZE, low_memory=False):
        chunk = _prepare_daily_chunk(chunk, rename_map)
        if chunk.empty:
            continue
        chunk_max = chunk["local_date"].max()
        max_date = chunk_max if max_date is None else max(max_date, chunk_max)
        summary_parts.append(
            chunk.groupby("ad_id", as_index=False).agg(
                full_days_active=("local_date", "nunique"),
                full_cum_spend=("spend", "sum"),
                full_cum_clicks=("inline_link_clicks", "sum"),
                full_cum_conversions=("tracker_conversions", "sum"),
                full_cum_revenue=("tracker_revenue", "sum"),
                full_zero_spend_to_date=("spend", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) == 0.0).sum())),
            )
        )

    if max_date is None or not summary_parts:
        raise RuntimeError(f"No ad-level daily rows found in {daily_input}")

    summary = (
        pd.concat(summary_parts, ignore_index=True)
        .groupby("ad_id", as_index=False)
        .sum(numeric_only=True)
    )
    allowed_ads: set[str] | None = None
    if SCORE_MAX_ADS > 0 and len(summary) > SCORE_MAX_ADS:
        allowed_ads = set(
            summary.sort_values("full_cum_spend", ascending=False)
            .head(SCORE_MAX_ADS)["ad_id"]
            .astype(str)
        )
        summary = summary[summary["ad_id"].astype(str).isin(allowed_ads)].copy()
    min_date = max_date - pd.Timedelta(days=LATEST_FEATURE_HISTORY_DAYS + 2)

    recent_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(daily_input, usecols=lambda c: c in usecols, chunksize=FAST_SCORE_CHUNKSIZE, low_memory=False):
        chunk = _prepare_daily_chunk(chunk, rename_map)
        if chunk.empty:
            continue
        if allowed_ads is not None:
            chunk = chunk[chunk["ad_id"].astype(str).isin(allowed_ads)].copy()
            if chunk.empty:
                continue
        chunk = chunk[chunk["local_date"] >= min_date].copy()
        if chunk.empty:
            continue
        recent_parts.append(
            chunk.groupby(["local_date", "timezone", *daily24.ENTITY_COLS], as_index=False)[daily24.RAW_TARGETS].sum()
        )

    if not recent_parts:
        raise RuntimeError(f"No recent ad-level rows found in {daily_input} from {min_date.date()}")
    recent = (
        pd.concat(recent_parts, ignore_index=True)
        .groupby(["local_date", "timezone", *daily24.ENTITY_COLS], as_index=False)[daily24.RAW_TARGETS]
        .sum()
        .sort_values(["ad_id", "local_date"])
        .reset_index(drop=True)
    )
    source = f"{daily_input} | fast_latest_context min_date={min_date.date()} max_date={max_date.date()} chunksize={FAST_SCORE_CHUNKSIZE}"
    return recent, source, summary


def _add_history_segments(scored: pd.DataFrame) -> pd.DataFrame:
    """Classify each forecast by recent data quality before it is shown downstream."""
    days_active = _num(scored, "days_active")
    spend_sum_7d = _num(scored, "spend_roll_sum_7d", _num(scored, "spend_roll_mean_7d") * 7.0)
    impressions_sum_7d = _num(scored, "impressions_roll_sum_7d", _num(scored, "impressions_roll_mean_7d") * 7.0)
    clicks_sum_7d = _num(scored, "inline_link_clicks_roll_sum_7d", _num(scored, "inline_link_clicks_roll_mean_7d") * 7.0)
    conversions_sum_7d = _num(scored, "tracker_conversions_roll_sum_7d", _num(scored, "tracker_conversions_roll_mean_7d") * 7.0)
    revenue_sum_7d = _num(scored, "tracker_revenue_roll_sum_7d", _num(scored, "tracker_revenue_roll_mean_7d") * 7.0)
    revenue_mean_7d = _num(scored, "tracker_revenue_roll_mean_7d")
    revenue_lag_1d = _num(scored, "tracker_revenue_lag_1d")
    conversions_nonzero_days = _num(scored, "tracker_conversions_nonzero_count_7d")
    revenue_nonzero_days = _num(scored, "tracker_revenue_nonzero_count_7d")
    spend_zero_days = _num(scored, "spend_zero_count_7d")
    revenue_zero_days = _num(scored, "tracker_revenue_zero_count_7d")

    scored["history_spend_sum_7d"] = spend_sum_7d
    scored["history_impressions_sum_7d"] = impressions_sum_7d
    scored["history_clicks_sum_7d"] = clicks_sum_7d
    scored["history_conversions_sum_7d"] = conversions_sum_7d
    scored["history_revenue_sum_7d"] = revenue_sum_7d
    scored["history_revenue_nonzero_days_7d"] = revenue_nonzero_days
    scored["history_conversion_nonzero_days_7d"] = conversions_nonzero_days
    scored["history_zero_spend_days_7d"] = spend_zero_days
    scored["history_zero_revenue_days_7d"] = revenue_zero_days
    scored["history_revenue_spike_ratio_7d"] = _safe_ratio(revenue_lag_1d, revenue_mean_7d.replace(0, np.nan)).clip(0, 100)

    inactive = (spend_sum_7d <= 0) & (impressions_sum_7d <= 0) & (clicks_sum_7d <= 0) & (revenue_sum_7d <= 0)
    new_or_short = days_active < 7
    low_volume = (spend_sum_7d < 7.0) | (impressions_sum_7d < 300.0) | (clicks_sum_7d < 5.0)
    mostly_zero = (spend_zero_days >= 5) | ((revenue_zero_days >= 5) & (revenue_sum_7d > 0))
    spiky = (
        (scored["history_revenue_spike_ratio_7d"] >= SPIKE_RATIO_HIGH)
        | ((revenue_nonzero_days <= 2) & (revenue_sum_7d > 0))
        | ((conversions_nonzero_days <= 2) & (conversions_sum_7d > 0))
    )
    stable = (
        (days_active >= MIN_24H_HISTORY_DAYS)
        & (spend_sum_7d > 0)
        & (revenue_nonzero_days >= MIN_REVENUE_HISTORY_DAYS)
        & (conversions_nonzero_days >= MIN_CONVERSION_HISTORY_DAYS)
        & (spend_zero_days <= 2)
    )

    scored["history_segment"] = np.select(
        [inactive, new_or_short, low_volume, mostly_zero, spiky, stable],
        ["inactive", "new_ad", "low_volume", "mostly_zero_history", "spiky_history", "stable_history"],
        default="mixed_history",
    )
    scored["spike_risk"] = np.select(
        [inactive | new_or_short | low_volume, mostly_zero | spiky, stable],
        ["LOW_DATA", "HIGH", "LOW"],
        default="MEDIUM",
    )
    scored["forecast_data_quality"] = np.select(
        [inactive, new_or_short | low_volume, mostly_zero | spiky, stable],
        ["INACTIVE", "LOW", "RISKY", "HIGH"],
        default="MEDIUM",
    )
    return scored


def add_24h_confidence_and_ranges(out: pd.DataFrame) -> pd.DataFrame:
    scored = out.copy()
    scored = _add_history_segments(scored)
    spend_mean = pd.to_numeric(scored.get("spend_roll_mean_7d", 0.0), errors="coerce").fillna(0.0)
    impressions_mean = pd.to_numeric(scored.get("impressions_roll_mean_7d", 0.0), errors="coerce").fillna(0.0)
    clicks_mean = pd.to_numeric(scored.get("inline_link_clicks_roll_mean_7d", 0.0), errors="coerce").fillna(0.0)
    pred_impressions = pd.to_numeric(scored.get("pred_24h_impressions", 0.0), errors="coerce").fillna(0.0)
    pred_clicks = pd.to_numeric(scored.get("pred_24h_inline_link_clicks", 0.0), errors="coerce").fillna(0.0)
    days_active = pd.to_numeric(scored.get("days_active", 0), errors="coerce").fillna(0)

    low_history = days_active < MIN_24H_HISTORY_DAYS
    low_recent_volume = (
        (spend_mean < MIN_24H_RECENT_SPEND_MEAN)
        | (impressions_mean < MIN_24H_KPI_IMPRESSIONS)
        | (clicks_mean < 1.0)
    )
    low_kpi_volume = (pred_impressions < MIN_24H_KPI_IMPRESSIONS) | (pred_clicks < MIN_24H_KPI_CLICKS)

    segment_low = scored["history_segment"].isin(["inactive", "new_ad", "low_volume", "mostly_zero_history"])
    segment_medium = scored["history_segment"].isin(["spiky_history", "mixed_history"]) | low_kpi_volume

    scored["forecast_confidence"] = np.select(
        [segment_low | low_history | low_recent_volume, segment_medium],
        ["LOW", "MEDIUM"],
        default="HIGH",
    )
    scored["kpi_reliability_flag"] = np.select(
        [
            scored["history_segment"].eq("inactive"),
            scored["history_segment"].eq("mostly_zero_history"),
            scored["history_segment"].eq("spiky_history"),
            low_history,
            spend_mean < MIN_24H_RECENT_SPEND_MEAN,
            pred_impressions < MIN_24H_KPI_IMPRESSIONS,
            pred_clicks < MIN_24H_KPI_CLICKS,
        ],
        ["INACTIVE_HISTORY", "MOSTLY_ZERO_HISTORY", "SPIKY_HISTORY", "LOW_RECENT_HISTORY", "LOW_RECENT_SPEND", "LOW_IMPRESSIONS", "LOW_CLICKS"],
        default="OK",
    )
    scored["forecast_use_case"] = np.where(
        scored["forecast_confidence"].eq("LOW"),
        "benchmark_range",
        np.where(scored["forecast_confidence"].eq("MEDIUM"), "point_with_caution", "point_forecast"),
    )

    enough_model_history = (
        days_active >= MIN_MODEL_HISTORY_DAYS_FOR_POINT
    ) & ~scored["history_segment"].isin(["inactive", "new_ad", "low_volume"])
    needs_peer_fallback = ~enough_model_history
    scored["benchmark_source"] = np.where(enough_model_history, "model_point", "insufficient_history")
    width = np.select(
        [
            scored["history_segment"].eq("inactive"),
            scored["history_segment"].isin(["new_ad", "low_volume", "mostly_zero_history"]),
            scored["history_segment"].eq("spiky_history"),
            scored["forecast_confidence"].eq("MEDIUM"),
        ],
        [0.0, 0.9, 1.25, 0.55],
        default=0.3,
    )
    for metric in daily24.RAW_TARGETS:
        pred_col = f"pred_24h_{metric}"
        if pred_col not in scored.columns:
            continue
        pred = pd.to_numeric(scored[pred_col], errors="coerce").fillna(0.0)
        recent_col = f"{metric}_roll_mean_7d"
        recent = pd.to_numeric(scored.get(recent_col, pred), errors="coerce").fillna(0.0)
        adset_benchmark, adset_ok = _peer_benchmark(scored, recent, ["account_id", "campaign_id", "adset_id"])
        campaign_benchmark, campaign_ok = _peer_benchmark(scored, recent, ["account_id", "campaign_id"])
        account_benchmark, account_ok = _peer_benchmark(scored, recent, ["account_id"])

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
        recommended = np.where(needs_peer_fallback, fallback, pred)
        recommended = np.where(scored["history_segment"].eq("inactive"), 0.0, recommended)
        recommended = pd.Series(recommended, index=scored.index).fillna(0.0)
        source_series = pd.Series(source, index=scored.index)
        update_source = needs_peer_fallback & (source_series.ne("insufficient_history") | scored["benchmark_source"].eq("insufficient_history"))
        scored.loc[update_source, "benchmark_source"] = source_series.loc[update_source]
        scored[f"recommended_24h_{metric}"] = recommended
        scored[f"{pred_col}_p50"] = recommended
        scored[f"{pred_col}_p10"] = np.maximum(0.0, recommended * (1.0 - width))
        scored[f"{pred_col}_p90"] = recommended * (1.0 + width)

    spend = pd.to_numeric(scored.get("recommended_24h_spend", scored.get("pred_24h_spend", 0.0)), errors="coerce").fillna(0.0)
    impressions = pd.to_numeric(scored.get("recommended_24h_impressions", scored.get("pred_24h_impressions", 0.0)), errors="coerce").fillna(0.0)
    clicks = pd.to_numeric(scored.get("recommended_24h_inline_link_clicks", scored.get("pred_24h_inline_link_clicks", 0.0)), errors="coerce").fillna(0.0)
    conversions = pd.to_numeric(scored.get("recommended_24h_tracker_conversions", scored.get("pred_24h_tracker_conversions", 0.0)), errors="coerce").fillna(0.0)
    revenue = pd.to_numeric(scored.get("recommended_24h_tracker_revenue", scored.get("pred_24h_tracker_revenue", 0.0)), errors="coerce").fillna(0.0)
    scored["recommended_24h_roas"] = np.where(spend > 0.0, revenue / spend, 0.0)
    scored["recommended_24h_profit"] = revenue - spend
    scored["recommended_24h_ctr"] = np.where(impressions > 0.0, clicks / impressions * 100.0, 0.0)
    scored["recommended_24h_cvr"] = np.where(clicks > 0.0, conversions / clicks * 100.0, 0.0)
    scored["recommended_24h_cpc"] = np.where(clicks > 0.0, spend / clicks, 0.0)
    scored["recommended_24h_cpm"] = np.where(impressions > 0.0, spend / impressions * 1000.0, 0.0)

    scored["recommended_24h_roas_p10"] = np.where(
        scored["recommended_24h_spend"] > 0.0,
        scored["pred_24h_tracker_revenue_p10"] / scored["recommended_24h_spend"],
        0.0,
    )
    scored["recommended_24h_roas_p50"] = scored["recommended_24h_roas"]
    scored["recommended_24h_roas_p90"] = np.where(
        scored["recommended_24h_spend"] > 0.0,
        scored["pred_24h_tracker_revenue_p90"] / scored["recommended_24h_spend"],
        0.0,
    )

    scored["forecast_note"] = np.select(
        [
            scored["history_segment"].eq("inactive"),
            scored["history_segment"].isin(["mostly_zero_history", "spiky_history"]),
            scored["kpi_reliability_flag"].ne("OK"),
            scored["forecast_confidence"].eq("MEDIUM"),
        ],
        [
            "No recent activity; forecast forced to zero.",
            "History is zero-heavy or spiky; use p10-p90 range, not point forecast only.",
            "KPI ratios are low-confidence; use raw metric range.",
            "Use point forecast with caution.",
        ],
        default="Forecast is stable enough for point use.",
    )
    return scored


def _latest_scoring_frame(daily: pd.DataFrame, full_summary: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a compact latest-only frame for production scoring.

    Training uses full history, but live scoring only needs enough recent rows
    to compute D-1/D-2/D-7/D-14 and rolling features for the next-day anchor.
    This avoids densifying every old day for every ad on low-RAM machines.
    """
    daily = daily.sort_values(["ad_id", "local_date"]).reset_index(drop=True)
    grouped = daily.groupby("ad_id", sort=False)
    latest_actual = grouped.tail(1).reset_index(drop=True)
    if full_summary is None:
        summary = grouped.agg(
            full_days_active=("local_date", "nunique"),
            full_cum_spend=("spend", "sum"),
            full_cum_clicks=("inline_link_clicks", "sum"),
            full_cum_conversions=("tracker_conversions", "sum"),
            full_cum_revenue=("tracker_revenue", "sum"),
            full_zero_spend_to_date=("spend", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0.0) == 0.0).sum())),
        ).reset_index()
    else:
        summary = full_summary

    recent = grouped.tail(LATEST_FEATURE_HISTORY_DAYS).reset_index(drop=True)
    future_rows = latest_actual[["local_date", "timezone", *daily24.ENTITY_COLS, *daily24.RAW_TARGETS]].copy()
    future_rows["history_cutoff_local_date"] = future_rows["local_date"]
    future_rows["local_date"] = future_rows["local_date"] + pd.Timedelta(days=1)
    for metric in daily24.RAW_TARGETS:
        future_rows[metric] = 0.0

    scoring_daily = pd.concat(
        [recent, future_rows[["local_date", "timezone", *daily24.ENTITY_COLS, *daily24.RAW_TARGETS]]],
        ignore_index=True,
    )
    future_rows = future_rows.merge(summary, on="ad_id", how="left")
    return scoring_daily, future_rows


def _ensure_model_features(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing = [col for col in feature_cols if col not in frame.columns]
    for col in missing:
        frame[col] = 0.0
    return frame


def _empty_forecast(source: str, reason: str) -> pd.DataFrame:
    cols = [
        "forecast_anchor_local_date",
        "history_cutoff_local_date",
        "timezone",
        *daily24.ENTITY_COLS,
        "days_active",
        *[f"pred_24h_{raw}" for raw in daily24.RAW_TARGETS],
        "pred_24h_roas",
        "pred_24h_profit",
        "pred_24h_ctr",
        "pred_24h_cvr",
        "pred_24h_cpc",
        "pred_24h_cpm",
        "forecast_confidence",
        "kpi_reliability_flag",
        "forecast_use_case",
        "benchmark_source",
        "forecast_note",
        "model_source",
        "daily_source",
    ]
    out = pd.DataFrame(columns=cols)
    out.attrs["empty_reason"] = reason
    out.attrs["daily_source"] = source
    return out


def _peer_benchmark(
    scored: pd.DataFrame,
    recent: pd.Series,
    group_cols: list[str],
    min_peer_ads: int = MIN_FALLBACK_PEER_ADS,
) -> tuple[pd.Series, pd.Series]:
    if not all(col in scored.columns for col in group_cols):
        empty = pd.Series(0.0, index=scored.index, dtype="float64")
        ok = pd.Series(False, index=scored.index)
        return empty, ok

    keys = [scored[col].astype(str) for col in group_cols]
    key_frame = pd.concat(keys, axis=1)
    key_frame.columns = group_cols
    key = key_frame.agg("||".join, axis=1)

    active_recent = recent.where(recent > 0.0, np.nan)
    peer_sum = active_recent.groupby(key).transform("sum").fillna(0.0)
    peer_count = active_recent.groupby(key).transform("count").fillna(0.0)
    self_active = active_recent.notna().astype("float64")
    peer_sum_ex_self = (peer_sum - active_recent.fillna(0.0)).clip(lower=0.0)
    peer_count_ex_self = (peer_count - self_active).clip(lower=0.0)
    bench = pd.Series(
        np.divide(
            peer_sum_ex_self,
            peer_count_ex_self,
            out=np.zeros(len(scored), dtype="float64"),
            where=peer_count_ex_self.to_numpy() >= min_peer_ads,
        ),
        index=scored.index,
    )
    ok = peer_count_ex_self >= min_peer_ads
    return bench.fillna(0.0), ok


def predict_latest(daily_input: Path) -> pd.DataFrame:
    metadata = joblib.load(daily24.MODEL_DIR / "metadata.joblib")
    feature_cols = metadata["feature_cols"]

    if REUSE_FEATURE_CACHE and FEATURE_CACHE_PATH.exists():
        payload = joblib.load(FEATURE_CACHE_PATH)
        latest = payload["latest"]
        source = f"feature_cache:{FEATURE_CACHE_PATH}"
    else:
        if FAST_SCORE_CSV:
            daily, source, full_summary = _load_latest_context_daily(daily_input)
        else:
            daily, source = daily24.load_daily(force_rebuild_from_hourly=False, daily_input_path=daily_input)
            full_summary = None
        scoring_daily, future_rows = _latest_scoring_frame(daily, full_summary)
        del daily
        dataset, built_feature_cols = daily24.add_features_and_targets(scoring_daily)
        del scoring_daily
        feature_cols = metadata.get("feature_cols", built_feature_cols)

        future_keys = future_rows[["ad_id", "local_date", "history_cutoff_local_date"]].copy()
        latest = dataset.merge(future_keys, on=["ad_id", "local_date"], how="inner")
        latest = latest.sort_values(["ad_id", "local_date"]).drop_duplicates(["ad_id"], keep="last").reset_index(drop=True)
        latest = latest.merge(
            future_rows[
                [
                    "ad_id",
                    "full_days_active",
                    "full_cum_spend",
                    "full_cum_clicks",
                    "full_cum_conversions",
                    "full_cum_revenue",
                    "full_zero_spend_to_date",
                ]
            ],
            on="ad_id",
            how="left",
        )
        latest["days_active"] = latest["full_days_active"].fillna(latest["days_active"]).astype("float32")
        latest["cum_spend"] = latest["full_cum_spend"].fillna(latest.get("cum_spend", 0.0)).astype("float32")
        latest["cum_clicks"] = latest["full_cum_clicks"].fillna(latest.get("cum_clicks", 0.0)).astype("float32")
        latest["cum_conversions"] = latest["full_cum_conversions"].fillna(latest.get("cum_conversions", 0.0)).astype("float32")
        latest["cum_revenue"] = latest["full_cum_revenue"].fillna(latest.get("cum_revenue", 0.0)).astype("float32")
        latest["zero_spend_to_date"] = latest["full_zero_spend_to_date"].fillna(latest.get("zero_spend_to_date", 0.0)).astype("float32")
        latest = _ensure_model_features(latest, feature_cols)
        if WRITE_FEATURE_CACHE:
            FEATURE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"latest": latest, "feature_cols": feature_cols, "source": source}, FEATURE_CACHE_PATH, compress=3)

    if latest.empty:
        return _empty_forecast(
            source,
            "No eligible 24h ad-level scoring rows were built from the current DB slice. Increase ADUNBOX_24H_DB_LOOKBACK_DAYS/ROW_LIMIT or ensure ad-level rows exist.",
        )

    X = latest[feature_cols].astype("float32")

    roll_cols = [col for col in latest.columns if col.endswith("_roll_mean_7d")]
    out = latest[["local_date", "history_cutoff_local_date", "timezone", *daily24.ENTITY_COLS, "days_active", *roll_cols]].copy()
    out = out.rename(columns={"local_date": "forecast_anchor_local_date"})
    for raw_target in daily24.RAW_TARGETS:
        target = f"target_24h_{raw_target}"
        model = joblib.load(daily24.MODEL_DIR / f"{target}.joblib")
        pred = np.expm1(model.predict(X))
        out[f"pred_24h_{raw_target}"] = np.maximum(0.0, pred).astype("float32")

    out = daily24.derive_kpis(out)
    out = add_24h_confidence_and_ranges(out)
    out["model_source"] = "daily_24h_histgb"
    out["daily_source"] = source
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score latest 24h forecasts from the daily 24h model.")
    parser.add_argument("--daily-input", type=Path, default=daily24.DEFAULT_DAILY_INPUT_PATH)
    args = parser.parse_args()

    forecasts = predict_latest(args.daily_input)
    forecasts.to_csv(OUTPUT_PATH, index=False)
    confidence_counts = forecasts["forecast_confidence"].value_counts(dropna=False).to_dict() if "forecast_confidence" in forecasts else {}
    reliability_counts = forecasts["kpi_reliability_flag"].value_counts(dropna=False).to_dict() if "kpi_reliability_flag" in forecasts else {}
    anchor_min = forecasts["forecast_anchor_local_date"].min() if "forecast_anchor_local_date" in forecasts and not forecasts.empty else "n/a"
    anchor_max = forecasts["forecast_anchor_local_date"].max() if "forecast_anchor_local_date" in forecasts and not forecasts.empty else "n/a"
    lines = [
        "Adunbox Daily 24h Latest Forecasts",
        "",
        f"Rows scored: {len(forecasts):,}",
        f"Output: {OUTPUT_PATH}",
        f"Anchor date min: {anchor_min}",
        f"Anchor date max: {anchor_max}",
        f"Forecast confidence counts: {confidence_counts}",
        f"KPI reliability counts: {reliability_counts}",
        "",
        "Prediction columns:",
        "- pred_24h_spend",
        "- pred_24h_impressions",
        "- pred_24h_inline_link_clicks",
        "- pred_24h_tracker_conversions",
        "- pred_24h_tracker_revenue",
        "- pred_24h_*_p10 / p50 / p90 range columns",
        "- recommended_24h_* raw and KPI columns",
        "- pred_24h_roas",
        "- pred_24h_profit",
        "- forecast_confidence",
        "- kpi_reliability_flag",
        "- forecast_use_case",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
