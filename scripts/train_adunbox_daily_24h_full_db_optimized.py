from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(r"G:\ml_model_historical_data")
DEFAULT_DAILY_INPUT = Path(r"C:\Users\eshaa\Downloads\adunbox_daily_breakdown_kpis.csv")
OUTPUT_DIR = BASE_DIR / "github_release" / "outputs"
MODEL_DIR = BASE_DIR / "github_release" / "models" / "adunbox_daily_24h_histgb_full_db_optimized"
METRICS_CSV = OUTPUT_DIR / "adunbox_daily_24h_full_db_optimized__metrics.csv"
BACKTEST_CSV = OUTPUT_DIR / "adunbox_daily_24h_full_db_optimized__backtest.csv"
SUMMARY_TXT = OUTPUT_DIR / "adunbox_daily_24h_full_db_optimized__summary.txt"
DASHBOARD_HTML = OUTPUT_DIR / "adunbox_daily_24h_full_db_optimized__dashboard.html"
CALIBRATION_JSON = OUTPUT_DIR / "adunbox_daily_24h_full_db_optimized__calibration.json"

ENTITY_COLS = ["account_id", "campaign_id", "adset_id", "ad_id"]
RAW_TARGETS = ["spend", "impressions", "inline_link_clicks", "tracker_conversions", "tracker_revenue"]
DISPLAY_TARGETS = {
    "spend": "spend",
    "impressions": "impressions",
    "inline_link_clicks": "clicks",
    "tracker_conversions": "conversions",
    "tracker_revenue": "revenue",
}
TRAIN_END = pd.Timestamp("2026-04-30")
VALID_END = pd.Timestamp("2026-05-07")
MIN_AD_DAYS = 8


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).replace({"nan": ""})


def safe_div(numer: pd.Series | np.ndarray, denom: pd.Series | np.ndarray, multiplier: float = 1.0):
    numer_s = pd.Series(numer, copy=False)
    denom_s = pd.Series(denom, copy=False)
    out = pd.Series(np.zeros(len(numer_s), dtype=np.float32), index=numer_s.index)
    mask = denom_s != 0
    out.loc[mask] = (numer_s.loc[mask] / denom_s.loc[mask]) * multiplier
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float32")


def wmape(actual: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> float:
    actual_arr = np.asarray(actual, dtype=np.float64)
    pred_arr = np.asarray(pred, dtype=np.float64)
    denom = np.abs(actual_arr).sum()
    return float(np.abs(actual_arr - pred_arr).sum() / denom) if denom else 0.0


def bias(actual: pd.Series | np.ndarray, pred: pd.Series | np.ndarray) -> float:
    actual_arr = np.asarray(actual, dtype=np.float64)
    pred_arr = np.asarray(pred, dtype=np.float64)
    denom = actual_arr.sum()
    return float((pred_arr.sum() - actual_arr.sum()) / denom) if denom else 0.0


def load_ad_daily(path: Path, sample_ads: int | None = None) -> tuple[pd.DataFrame, str]:
    usecols = ["entity_type", "date", "timezone", *ENTITY_COLS, *RAW_TARGETS]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=50_000, low_memory=False):
        chunk = chunk[chunk["entity_type"].astype(str).str.lower().eq("ad")].copy()
        if chunk.empty:
            continue
        for col in ENTITY_COLS:
            chunk[col] = normalize_id(chunk[col])
        chunk["timezone"] = chunk["timezone"].fillna("").astype(str)
        chunk["local_date"] = pd.to_datetime(chunk["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
        chunk = chunk[chunk["local_date"].notna()].copy()
        for col in RAW_TARGETS:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0.0).astype("float32")
        parts.append(chunk[["local_date", "timezone", *ENTITY_COLS, *RAW_TARGETS]])

    daily = pd.concat(parts, ignore_index=True)
    daily = (
        daily.groupby(["local_date", "timezone", *ENTITY_COLS], as_index=False)[RAW_TARGETS]
        .sum()
        .sort_values(["ad_id", "local_date"])
        .reset_index(drop=True)
    )
    eligible = daily.groupby("ad_id")["local_date"].nunique()
    eligible_ads = eligible[eligible >= MIN_AD_DAYS].index.astype(str)
    if sample_ads:
        eligible_ads = eligible.loc[eligible_ads].sort_values(ascending=False).head(sample_ads).index.astype(str)
    daily = daily[daily["ad_id"].isin(eligible_ads)].copy()
    return daily, str(path)


def add_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["kpi_ctr"] = safe_div(out["inline_link_clicks"], out["impressions"], 100.0)
    out["kpi_cpc"] = safe_div(out["spend"], out["inline_link_clicks"])
    out["kpi_cpm"] = safe_div(out["spend"], out["impressions"], 1000.0)
    out["kpi_cvr"] = safe_div(out["tracker_conversions"], out["inline_link_clicks"], 100.0)
    out["kpi_roas"] = safe_div(out["tracker_revenue"], out["spend"])
    out["kpi_profit"] = (out["tracker_revenue"] - out["spend"]).astype("float32")
    return out


def densify_by_ad(daily: pd.DataFrame) -> pd.DataFrame:
    dense_parts: list[pd.DataFrame] = []
    for ad_id, grp in daily.groupby("ad_id", sort=False):
        grp = grp.sort_values("local_date").drop_duplicates("local_date", keep="last")
        date_index = pd.date_range(grp["local_date"].min(), grp["local_date"].max(), freq="1D")
        dense = pd.DataFrame({"local_date": date_index})
        dense = dense.merge(grp, on="local_date", how="left")
        dense["ad_id"] = str(ad_id)
        for col in ["timezone", "account_id", "campaign_id", "adset_id"]:
            dense[col] = dense[col].ffill().bfill().fillna("")
        dense[RAW_TARGETS] = dense[RAW_TARGETS].fillna(0.0)
        dense_parts.append(dense)
    return pd.concat(dense_parts, ignore_index=True)


def hierarchy_rolling(dense: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    daily = dense.groupby([*keys, "local_date"], as_index=False)[RAW_TARGETS].sum().sort_values([*keys, "local_date"])
    grouped = daily.groupby(keys, sort=False)
    feature_data: dict[str, pd.Series] = {}
    for col in RAW_TARGETS:
        shifted = grouped[col].shift(1)
        roll = shifted.groupby([daily[k] for k in keys], sort=False)
        feature_data[f"{prefix}_{col}_roll_mean_7d"] = roll.rolling(7, min_periods=1).mean().reset_index(level=list(range(len(keys))), drop=True).fillna(0.0).astype("float32")
        feature_data[f"{prefix}_{col}_roll_sum_7d"] = roll.rolling(7, min_periods=1).sum().reset_index(level=list(range(len(keys))), drop=True).fillna(0.0).astype("float32")
    features = pd.concat([daily[[*keys, "local_date"]], pd.DataFrame(feature_data, index=daily.index)], axis=1)
    return features


def build_features(daily: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = add_kpis(densify_by_ad(daily))
    out = out.sort_values(["ad_id", "local_date"]).reset_index(drop=True)
    grouped = out.groupby("ad_id", sort=False)

    day_of_week = out["local_date"].dt.dayofweek.astype("int16")
    feature_data: dict[str, pd.Series | np.ndarray] = {
        "day_of_week": day_of_week,
        "dow_sin": np.sin(2.0 * np.pi * day_of_week / 7.0).astype("float32"),
        "dow_cos": np.cos(2.0 * np.pi * day_of_week / 7.0).astype("float32"),
        "days_active": (grouped.cumcount() + 1).astype("int32"),
        "cum_spend": grouped["spend"].cumsum().astype("float32"),
        "cum_clicks": grouped["inline_link_clicks"].cumsum().astype("float32"),
        "cum_conversions": grouped["tracker_conversions"].cumsum().astype("float32"),
        "cum_revenue": grouped["tracker_revenue"].cumsum().astype("float32"),
        "zero_spend_to_date": grouped["spend"].transform(lambda s: s.eq(0).cumsum()).astype("float32"),
    }
    feature_cols = list(feature_data.keys())
    base_cols = [*RAW_TARGETS, "kpi_ctr", "kpi_cpc", "kpi_cpm", "kpi_cvr", "kpi_roas", "kpi_profit"]
    raw_set = set(RAW_TARGETS)

    for col in base_cols:
        for lag in [1, 2, 3, 7, 14]:
            name = f"{col}_lag_{lag}d"
            feature_data[name] = grouped[col].shift(lag).fillna(0.0).astype("float32")
            feature_cols.append(name)
        shifted = grouped[col].shift(1)
        shifted_grouped = shifted.groupby(out["ad_id"], sort=False)
        for window in [3, 7, 14]:
            mean_name = f"{col}_roll_mean_{window}d"
            std_name = f"{col}_roll_std_{window}d"
            feature_data[mean_name] = shifted_grouped.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0.0).astype("float32")
            feature_data[std_name] = shifted_grouped.rolling(window, min_periods=2).std().reset_index(level=0, drop=True).fillna(0.0).astype("float32")
            feature_cols.extend([mean_name, std_name])
            if col in raw_set:
                sum_name = f"{col}_roll_sum_{window}d"
                zero_name = f"{col}_zero_count_{window}d"
                feature_data[sum_name] = shifted_grouped.rolling(window, min_periods=1).sum().reset_index(level=0, drop=True).fillna(0.0).astype("float32")
                feature_data[zero_name] = shifted_grouped.rolling(window, min_periods=1).apply(lambda x: float(np.sum(x == 0)), raw=True).reset_index(level=0, drop=True).fillna(0.0).astype("float32")
                feature_cols.extend([sum_name, zero_name])

    feature_frame = pd.DataFrame(feature_data, index=out.index)
    out = pd.concat([out, feature_frame], axis=1)

    # Explicit stability/spike context helps the regressor distinguish normal
    # repeatable ads from sparse or spiky ads where raw metric jumps are common.
    for col in RAW_TARGETS:
        mean_3 = out.get(f"{col}_roll_mean_3d", 0.0)
        mean_7 = out.get(f"{col}_roll_mean_7d", 0.0)
        std_7 = out.get(f"{col}_roll_std_7d", 0.0)
        lag_1 = out.get(f"{col}_lag_1d", 0.0)
        out[f"{col}_trend_3d_vs_7d"] = safe_div(mean_3, pd.Series(mean_7).replace(0, np.nan)).fillna(0.0).astype("float32")
        out[f"{col}_cv_7d"] = safe_div(std_7, pd.Series(mean_7).replace(0, np.nan)).fillna(0.0).astype("float32")
        out[f"{col}_lag1_vs_mean_7d"] = safe_div(lag_1, pd.Series(mean_7).replace(0, np.nan)).fillna(0.0).astype("float32")
        feature_cols.extend([f"{col}_trend_3d_vs_7d", f"{col}_cv_7d", f"{col}_lag1_vs_mean_7d"])

    for keys, prefix in [(["account_id"], "account"), (["account_id", "campaign_id"], "campaign")]:
        hier = hierarchy_rolling(out, keys, prefix)
        out = out.merge(hier, on=[*keys, "local_date"], how="left")
        new_cols = [c for c in hier.columns if c not in {*keys, "local_date"}]
        feature_cols.extend(new_cols)

    for target in RAW_TARGETS:
        out[f"target_24h_{target}"] = out[target].astype("float32")
    out["target_24h_roas"] = out["kpi_roas"].astype("float32")
    out["target_24h_profit"] = out["kpi_profit"].astype("float32")
    out["target_24h_ctr"] = out["kpi_ctr"].astype("float32")
    out["target_24h_cvr"] = out["kpi_cvr"].astype("float32")
    out["target_24h_cpc"] = out["kpi_cpc"].astype("float32")
    out["target_24h_cpm"] = out["kpi_cpm"].astype("float32")
    out = out[out["days_active"] >= MIN_AD_DAYS].replace([np.inf, -np.inf], np.nan).fillna(0.0).copy()
    feature_cols = list(dict.fromkeys(feature_cols))
    return out, feature_cols


def split_dataset(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = dataset[dataset["local_date"] <= TRAIN_END].copy()
    valid = dataset[(dataset["local_date"] > TRAIN_END) & (dataset["local_date"] <= VALID_END)].copy()
    test = dataset[dataset["local_date"] > VALID_END].copy()
    if len(train) and len(valid) and len(test):
        return train, valid, test
    dates = np.array(sorted(dataset["local_date"].unique()))
    train_end = int(len(dates) * 0.70)
    valid_end = int(len(dates) * 0.85)
    return (
        dataset[dataset["local_date"] <= pd.Timestamp(dates[train_end - 1])].copy(),
        dataset[(dataset["local_date"] > pd.Timestamp(dates[train_end - 1])) & (dataset["local_date"] <= pd.Timestamp(dates[valid_end - 1]))].copy(),
        dataset[dataset["local_date"] > pd.Timestamp(dates[valid_end - 1])].copy(),
    )


def build_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=120,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        max_bins=128,
        l2_regularization=0.05,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=12,
        random_state=42,
    )


def add_error_control_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Features used only for post-model calibration/capping.

    The base model still learns from the normal feature matrix. This layer reduces
    dashboard-visible over/under-shoots by correcting predictions with validation
    actual-vs-predicted behavior and recent 7-day scale.
    """
    out = frame.copy()
    for target in RAW_TARGETS:
        out[f"{target}_recent_mean_7d"] = pd.to_numeric(
            out.get(f"{target}_roll_mean_7d", 0.0), errors="coerce"
        ).fillna(0.0).astype("float32")
        out[f"{target}_recent_sum_7d"] = pd.to_numeric(
            out.get(f"{target}_roll_sum_7d", 0.0), errors="coerce"
        ).fillna(0.0).astype("float32")
        out[f"{target}_recent_lag_1d"] = pd.to_numeric(
            out.get(f"{target}_lag_1d", 0.0), errors="coerce"
        ).fillna(0.0).astype("float32")
    return out


def _bucketize_scale(scale: pd.Series) -> pd.Series:
    scale = pd.to_numeric(scale, errors="coerce").fillna(0.0)
    return pd.cut(
        scale,
        bins=[-0.001, 0.0, 1.0, 10.0, 50.0, np.inf],
        labels=["zero", "tiny", "small", "medium", "large"],
    ).astype(str)


def classify_prediction_segment(frame: pd.DataFrame) -> pd.Series:
    days_active = pd.to_numeric(frame.get("days_active", 0), errors="coerce").fillna(0)
    spend_7d = pd.to_numeric(frame.get("spend_recent_sum_7d", frame.get("spend_roll_sum_7d", 0.0)), errors="coerce").fillna(0.0)
    impressions_7d = pd.to_numeric(frame.get("impressions_recent_sum_7d", frame.get("impressions_roll_sum_7d", 0.0)), errors="coerce").fillna(0.0)
    clicks_7d = pd.to_numeric(frame.get("inline_link_clicks_recent_sum_7d", frame.get("inline_link_clicks_roll_sum_7d", 0.0)), errors="coerce").fillna(0.0)
    revenue_cv = pd.to_numeric(frame.get("tracker_revenue_cv_7d", 0.0), errors="coerce").fillna(0.0)
    conv_cv = pd.to_numeric(frame.get("tracker_conversions_cv_7d", 0.0), errors="coerce").fillna(0.0)
    revenue_spike = pd.to_numeric(frame.get("tracker_revenue_lag1_vs_mean_7d", 0.0), errors="coerce").fillna(0.0)
    conv_spike = pd.to_numeric(frame.get("tracker_conversions_lag1_vs_mean_7d", 0.0), errors="coerce").fillna(0.0)

    segment = pd.Series("stable", index=frame.index, dtype="object")
    segment.loc[(days_active < 14)] = "new"
    segment.loc[(days_active >= 14) & ((spend_7d < 5) | (impressions_7d < 100) | (clicks_7d < 3))] = "low_volume"
    spiky_mask = (
        (days_active >= 14)
        & ~segment.eq("low_volume")
        & (
            (revenue_cv > 1.75)
            | (conv_cv > 1.75)
            | ((revenue_spike > 3.0) & (spend_7d >= 10))
            | ((conv_spike > 3.0) & (clicks_7d >= 5))
        )
    )
    segment.loc[spiky_mask] = "spiky"
    return segment


def fit_target_calibration(valid: pd.DataFrame, target: str, pred: np.ndarray) -> dict[str, object]:
    actual = pd.to_numeric(valid[f"target_24h_{target}"], errors="coerce").fillna(0.0).astype("float64")
    pred_s = pd.Series(pred, index=valid.index, dtype="float64").clip(lower=0.0)
    scale = pd.to_numeric(valid.get(f"{target}_recent_sum_7d", 0.0), errors="coerce").fillna(0.0)
    buckets = _bucketize_scale(scale)
    segments = classify_prediction_segment(valid)

    global_factor = float(actual.sum() / pred_s.sum()) if pred_s.sum() > 0 else 1.0
    global_factor = float(np.clip(global_factor, 0.20, 5.00))
    bucket_factors: dict[str, float] = {}
    for bucket in ["zero", "tiny", "small", "medium", "large"]:
        mask = buckets.eq(bucket)
        if mask.sum() < 25 or pred_s.loc[mask].sum() <= 0:
            bucket_factors[bucket] = global_factor
            continue
        factor = float(actual.loc[mask].sum() / pred_s.loc[mask].sum())
        bucket_factors[bucket] = float(np.clip(factor, 0.20, 5.00))

    segment_factors: dict[str, float] = {}
    for segment in ["stable", "spiky", "new", "low_volume"]:
        mask = segments.eq(segment)
        if mask.sum() < 25 or pred_s.loc[mask].sum() <= 0:
            segment_factors[segment] = global_factor
            continue
        factor = float(actual.loc[mask].sum() / pred_s.loc[mask].sum())
        segment_factors[segment] = float(np.clip(factor, 0.15, 6.00))

    # Production correction: learn systematic under/over prediction by entity
    # from validation only. At scoring time we prefer ad-level correction, then
    # campaign, then account, and fall back to the segment/bucket correction.
    entity_factors: dict[str, dict[str, float]] = {"account": {}, "campaign": {}, "ad": {}}
    valid_keys = valid[["account_id", "campaign_id", "ad_id"]].astype(str).copy()
    tmp = valid_keys.assign(actual=actual.to_numpy(), pred=pred_s.to_numpy())
    for level, keys, min_rows, lo, hi in [
        ("account", ["account_id"], 10, 0.35, 3.00),
        ("campaign", ["account_id", "campaign_id"], 8, 0.30, 3.50),
        ("ad", ["ad_id"], 4, 0.25, 4.00),
    ]:
        grouped = tmp.groupby(keys, dropna=False).agg(rows=("actual", "size"), actual=("actual", "sum"), pred=("pred", "sum"))
        grouped = grouped[(grouped["rows"] >= min_rows) & (grouped["pred"] > 0)]
        factors = (grouped["actual"] / grouped["pred"]).clip(lo, hi)
        entity_factors[level] = {"|".join(map(str, idx if isinstance(idx, tuple) else (idx,))): float(val) for idx, val in factors.items()}

    ratio = actual.divide(pred_s.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    global_range = {
        "p10": float(np.clip(ratio.quantile(0.10), 0.05, 5.00)) if len(ratio) else 0.50,
        "p90": float(np.clip(ratio.quantile(0.90), 0.20, 8.00)) if len(ratio) else 2.00,
    }
    range_ratios: dict[str, dict[str, float]] = {}
    for segment in ["stable", "spiky", "new", "low_volume"]:
        mask = segments.eq(segment) & pred_s.gt(0)
        seg_ratio = actual.loc[mask].divide(pred_s.loc[mask]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(seg_ratio) < 25:
            range_ratios[segment] = global_range
        else:
            range_ratios[segment] = {
                "p10": float(np.clip(seg_ratio.quantile(0.10), 0.05, 5.00)),
                "p90": float(np.clip(seg_ratio.quantile(0.90), 0.20, 8.00)),
            }

    recent_mean = pd.to_numeric(valid.get(f"{target}_recent_mean_7d", 0.0), errors="coerce").fillna(0.0)
    nonzero_recent = recent_mean[recent_mean > 0]
    max_multiplier = 5.0
    if target in {"tracker_revenue", "tracker_conversions"}:
        max_multiplier = 3.0
    elif target in {"spend", "impressions", "inline_link_clicks"}:
        max_multiplier = 4.0
    fallback_cap = float(nonzero_recent.quantile(0.95) * max_multiplier) if len(nonzero_recent) else float(actual.quantile(0.95))
    fallback_cap = max(fallback_cap, float(actual.quantile(0.90)), 1.0)
    return {
        "global_factor": global_factor,
        "bucket_factors": bucket_factors,
        "segment_factors": segment_factors,
        "entity_factors": entity_factors,
        "range_ratios": range_ratios,
        "max_multiplier": max_multiplier,
        "fallback_cap": fallback_cap,
    }


def apply_target_calibration(frame: pd.DataFrame, target: str, pred: np.ndarray, spec: dict[str, object]) -> np.ndarray:
    pred_s = pd.Series(pred, index=frame.index, dtype="float64").clip(lower=0.0)
    scale = pd.to_numeric(frame.get(f"{target}_recent_sum_7d", 0.0), errors="coerce").fillna(0.0)
    buckets = _bucketize_scale(scale)
    segments = classify_prediction_segment(frame)
    bucket_factors = buckets.map(spec["bucket_factors"]).astype("float64").fillna(float(spec["global_factor"]))
    segment_factors = segments.map(spec["segment_factors"]).astype("float64").fillna(float(spec["global_factor"]))
    # Blend scale-specific and segment-specific correction. This avoids one bad
    # segment dominating every row while still treating spiky/new/low-volume ads
    # differently from stable ads.
    factors = np.sqrt(bucket_factors * segment_factors)
    entity_factor = pd.Series(1.0, index=frame.index, dtype="float64")
    entity_specs = spec.get("entity_factors", {})
    if isinstance(entity_specs, dict):
        account_map = entity_specs.get("account", {})
        campaign_map = entity_specs.get("campaign", {})
        ad_map = entity_specs.get("ad", {})
        account_keys = frame["account_id"].astype(str)
        campaign_keys = frame["account_id"].astype(str) + "|" + frame["campaign_id"].astype(str)
        ad_keys = frame["ad_id"].astype(str)
        account_factor = account_keys.map(account_map).astype("float64")
        campaign_factor = campaign_keys.map(campaign_map).astype("float64")
        ad_factor = ad_keys.map(ad_map).astype("float64")
        entity_factor = account_factor.fillna(entity_factor)
        entity_factor = campaign_factor.fillna(entity_factor)
        entity_factor = ad_factor.fillna(entity_factor)
    factors = np.clip(factors * np.sqrt(entity_factor), 0.10, 8.00)
    calibrated = pred_s * factors

    recent_mean = pd.to_numeric(frame.get(f"{target}_recent_mean_7d", 0.0), errors="coerce").fillna(0.0)
    lag_1d = pd.to_numeric(frame.get(f"{target}_recent_lag_1d", 0.0), errors="coerce").fillna(0.0)
    dynamic_cap = np.maximum(recent_mean * float(spec["max_multiplier"]), lag_1d * float(spec["max_multiplier"]))
    dynamic_cap = np.maximum(dynamic_cap, float(spec["fallback_cap"]))
    # Sparse targets should not create big non-zero forecasts when the ad had no recent signal.
    if target in {"tracker_revenue", "tracker_conversions"}:
        no_recent_signal = scale <= 0
        calibrated.loc[no_recent_signal] = np.minimum(calibrated.loc[no_recent_signal], float(spec["fallback_cap"]) * 0.15)
        spiky_rows = segments.eq("spiky")
        calibrated.loc[spiky_rows] = np.minimum(
            calibrated.loc[spiky_rows],
            np.maximum(recent_mean.loc[spiky_rows] * 4.0, lag_1d.loc[spiky_rows] * 2.5),
        )
        low_volume_rows = segments.isin(["new", "low_volume"])
        calibrated.loc[low_volume_rows] = np.minimum(
            calibrated.loc[low_volume_rows],
            np.maximum(recent_mean.loc[low_volume_rows] * 2.0, float(spec["fallback_cap"]) * 0.25),
        )
    calibrated = np.minimum(calibrated, dynamic_cap)
    return np.maximum(0.0, calibrated.to_numpy(dtype=np.float32))


def prediction_ranges(frame: pd.DataFrame, target: str, p50: np.ndarray, spec: dict[str, object]) -> pd.DataFrame:
    p50_s = pd.Series(p50, index=frame.index, dtype="float64").clip(lower=0.0)
    segments = classify_prediction_segment(frame)
    ratios = spec.get("range_ratios", {})
    p10_ratio = segments.map({k: v.get("p10", 0.50) for k, v in ratios.items()}).astype("float64").fillna(0.50)
    p90_ratio = segments.map({k: v.get("p90", 2.00) for k, v in ratios.items()}).astype("float64").fillna(2.00)
    p10 = np.minimum(p50_s, p50_s * p10_ratio).clip(lower=0.0)
    p90 = np.maximum(p50_s, p50_s * p90_ratio).clip(lower=0.0)
    return pd.DataFrame(
        {
            f"pred_p10_24h_{target}": p10.astype("float32"),
            f"pred_p50_24h_{target}": p50_s.astype("float32"),
            f"pred_p90_24h_{target}": p90.astype("float32"),
        },
        index=frame.index,
    )


def derive_kpis(preds: pd.DataFrame) -> pd.DataFrame:
    out = preds.copy()
    out["pred_24h_roas"] = safe_div(out["pred_24h_tracker_revenue"], out["pred_24h_spend"])
    out["pred_24h_profit"] = (out["pred_24h_tracker_revenue"] - out["pred_24h_spend"]).astype("float32")
    out["pred_24h_ctr"] = safe_div(out["pred_24h_inline_link_clicks"], out["pred_24h_impressions"], 100.0)
    out["pred_24h_cvr"] = safe_div(out["pred_24h_tracker_conversions"], out["pred_24h_inline_link_clicks"], 100.0)
    out["pred_24h_cpc"] = safe_div(out["pred_24h_spend"], out["pred_24h_inline_link_clicks"])
    out["pred_24h_cpm"] = safe_div(out["pred_24h_spend"], out["pred_24h_impressions"], 1000.0)
    return out


def eval_rows(target: str, split: str, actual: pd.Series, pred: np.ndarray) -> dict[str, object]:
    return {
        "target": target,
        "split": split,
        "rows": int(len(actual)),
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "r2": float(r2_score(actual, pred)) if len(actual) > 1 else 0.0,
        "wmape": wmape(actual, pred),
        "bias": bias(actual, pred),
        "underprediction_rate": float((pred < actual.to_numpy(dtype=np.float32)).mean()),
    }


def train_and_score(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, object]] = []
    pred_frames = {
        "valid": pd.DataFrame(index=valid.index),
        "test": pd.DataFrame(index=test.index),
    }
    calibration_specs: dict[str, dict[str, object]] = {}
    train = add_error_control_features(train)
    valid = add_error_control_features(valid)
    test = add_error_control_features(test)
    X_train = train[feature_cols].astype("float32")
    X_valid = valid[feature_cols].astype("float32")
    X_test = test[feature_cols].astype("float32")
    for raw_target in RAW_TARGETS:
        target = f"target_24h_{raw_target}"
        y_train = train[target].astype("float32")
        y_valid = valid[target].astype("float32")
        y_test = test[target].astype("float32")
        model = build_model()
        model.fit(X_train, np.log1p(np.maximum(0.0, y_train)))
        joblib.dump(model, MODEL_DIR / f"{target}.joblib")
        pred_valid = np.maximum(0.0, np.expm1(model.predict(X_valid))).astype("float32")
        pred_test = np.maximum(0.0, np.expm1(model.predict(X_test))).astype("float32")
        calibration_specs[raw_target] = fit_target_calibration(valid, raw_target, pred_valid)
        pred_valid_cal = apply_target_calibration(valid, raw_target, pred_valid, calibration_specs[raw_target])
        pred_test_cal = apply_target_calibration(test, raw_target, pred_test, calibration_specs[raw_target])
        pred_frames["valid"][f"pred_24h_{raw_target}"] = pred_valid
        pred_frames["test"][f"pred_24h_{raw_target}"] = pred_test
        pred_frames["valid"][f"pred_calibrated_24h_{raw_target}"] = pred_valid_cal
        pred_frames["test"][f"pred_calibrated_24h_{raw_target}"] = pred_test_cal
        for col, values in prediction_ranges(valid, raw_target, pred_valid_cal, calibration_specs[raw_target]).items():
            pred_frames["valid"][col] = values
        for col, values in prediction_ranges(test, raw_target, pred_test_cal, calibration_specs[raw_target]).items():
            pred_frames["test"][col] = values
        metrics.append(eval_rows(target, "valid", y_valid, pred_valid))
        metrics.append(eval_rows(target, "test", y_test, pred_test))
        metrics.append(eval_rows(f"{target}_calibrated", "valid", y_valid, pred_valid_cal))
        metrics.append(eval_rows(f"{target}_calibrated", "test", y_test, pred_test_cal))
    CALIBRATION_JSON.write_text(json.dumps(calibration_specs, indent=2), encoding="utf-8")

    metrics_df = pd.DataFrame(metrics)
    backtest_parts = []
    for split, frame, preds in [("valid", valid, pred_frames["valid"]), ("test", test, pred_frames["test"])]:
        pred_base = preds[[f"pred_24h_{target}" for target in RAW_TARGETS]].copy()
        pred_cal = preds[[f"pred_calibrated_24h_{target}" for target in RAW_TARGETS]].rename(
            columns={f"pred_calibrated_24h_{target}": f"pred_24h_{target}" for target in RAW_TARGETS}
        )
        pred_kpis = derive_kpis(pred_base)
        pred_kpis_cal = derive_kpis(pred_cal).rename(columns={col: col.replace("pred_24h_", "pred_calibrated_24h_") for col in derive_kpis(pred_cal).columns})
        actual_kpis = frame[["local_date", "timezone", *ENTITY_COLS, "days_active", *RAW_TARGETS, "kpi_roas", "kpi_profit", "kpi_ctr", "kpi_cvr", "kpi_cpc", "kpi_cpm"]].copy()
        out = actual_kpis.rename(columns={col: f"actual_24h_{col}" for col in RAW_TARGETS})
        out["prediction_segment"] = classify_prediction_segment(frame).to_numpy()
        for col in pred_kpis.columns:
            out[col] = pred_kpis[col].to_numpy()
        for col in pred_kpis_cal.columns:
            out[col] = pred_kpis_cal[col].to_numpy()
        for raw_target in RAW_TARGETS:
            for bound in ["p10", "p50", "p90"]:
                col = f"pred_{bound}_24h_{raw_target}"
                if col in preds:
                    out[col] = preds[col].to_numpy()
        out["split"] = split
        for raw_target in RAW_TARGETS:
            out[f"gap_24h_{raw_target}"] = out[f"pred_24h_{raw_target}"] - out[f"actual_24h_{raw_target}"]
            out[f"gap_calibrated_24h_{raw_target}"] = out[f"pred_calibrated_24h_{raw_target}"] - out[f"actual_24h_{raw_target}"]
        out["kpi_reliability_flag"] = np.select(
            [
                out["actual_24h_impressions"] < 100,
                out["actual_24h_inline_link_clicks"] < 5,
                out["actual_24h_spend"] < 1,
            ],
            ["LOW_IMPRESSIONS", "LOW_CLICKS", "LOW_SPEND"],
            default="OK",
        )
        backtest_parts.append(out)

        for target, pred_col in [
            ("target_24h_roas", "pred_24h_roas"),
            ("target_24h_profit", "pred_24h_profit"),
            ("target_24h_ctr", "pred_24h_ctr"),
            ("target_24h_cvr", "pred_24h_cvr"),
            ("target_24h_cpc", "pred_24h_cpc"),
            ("target_24h_cpm", "pred_24h_cpm"),
        ]:
            cal_col = pred_col.replace("pred_24h_", "pred_calibrated_24h_")
            metrics_df = pd.concat(
                [
                    metrics_df,
                    pd.DataFrame([eval_rows(target, split, frame[target].astype("float32"), pred_kpis[pred_col].to_numpy())]),
                    pd.DataFrame([eval_rows(f"{target}_calibrated", split, frame[target].astype("float32"), pred_kpis_cal[cal_col].to_numpy())]),
                ],
                ignore_index=True,
            )

    backtest = pd.concat(backtest_parts, ignore_index=True)
    return metrics_df, backtest


def write_dashboard(metrics: pd.DataFrame, backtest: pd.DataFrame, source: str, feature_count: int) -> None:
    test_metrics = metrics[metrics["split"].eq("test")].copy()
    cards = {
        "rows": len(backtest),
        "accounts": backtest["account_id"].nunique(),
        "ads": backtest["ad_id"].nunique(),
        "features": feature_count,
        "revenue_wmape": float(test_metrics.loc[test_metrics["target"].eq("target_24h_tracker_revenue_calibrated"), "wmape"].iloc[0]),
        "spend_wmape": float(test_metrics.loc[test_metrics["target"].eq("target_24h_spend_calibrated"), "wmape"].iloc[0]),
    }
    account_summary = (
        backtest[backtest["split"].eq("test")]
        .groupby("account_id", as_index=False)
        .agg(
            rows=("ad_id", "size"),
            ads=("ad_id", "nunique"),
            actual_revenue=("actual_24h_tracker_revenue", "sum"),
            pred_revenue=("pred_24h_tracker_revenue", "sum"),
            pred_calibrated_revenue=("pred_calibrated_24h_tracker_revenue", "sum"),
            actual_spend=("actual_24h_spend", "sum"),
            pred_spend=("pred_24h_spend", "sum"),
            pred_calibrated_spend=("pred_calibrated_24h_spend", "sum"),
            kpi_unstable_rate=("kpi_reliability_flag", lambda s: float((s != "OK").mean())),
        )
    )
    account_summary["base_revenue_gap_pct"] = safe_div((account_summary["pred_revenue"] - account_summary["actual_revenue"]).abs(), account_summary["actual_revenue"])
    account_summary["calibrated_revenue_gap_pct"] = safe_div((account_summary["pred_calibrated_revenue"] - account_summary["actual_revenue"]).abs(), account_summary["actual_revenue"])
    account_summary["base_spend_gap_pct"] = safe_div((account_summary["pred_spend"] - account_summary["actual_spend"]).abs(), account_summary["actual_spend"])
    account_summary["calibrated_spend_gap_pct"] = safe_div((account_summary["pred_calibrated_spend"] - account_summary["actual_spend"]).abs(), account_summary["actual_spend"])
    account_summary = account_summary.sort_values("calibrated_revenue_gap_pct", ascending=False).head(30)

    def table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        body = []
        for rec in df.itertuples(index=False):
            tds = []
            for val in rec:
                tds.append(f"<td>{val:.4f}</td>" if isinstance(val, float) else f"<td>{val}</td>")
            body.append("<tr>" + "".join(tds) + "</tr>")
        return "<table><thead><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Adunbox Full DB 24h Improved Model</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;background:#f7f8fb;margin:0;color:#111827}}.wrap{{max-width:1500px;margin:auto;padding:24px}}.grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}}.card{{background:#fff;border:1px solid #d9dee8;border-radius:8px;padding:12px}}.k{{font-size:11px;color:#667085;text-transform:uppercase}}.v{{font-size:20px;font-weight:800}}table{{width:100%;border-collapse:collapse;background:#fff;font-size:12px}}th,td{{border:1px solid #e5e7eb;padding:7px;text-align:left;white-space:nowrap}}th{{background:#eef2ff}}.note{{background:#ecfeff;border:1px solid #67e8f9;border-radius:8px;padding:12px;margin:14px 0}}</style></head>
<body><div class="wrap"><h1>Adunbox Full-Database 24h Optimized Model</h1>
<div class="note">Source: {source}. This challenger model predicts raw metrics, applies validation-fitted calibration/error controls, then derives KPIs.</div>
<div class="grid">
<div class="card"><div class="k">Backtest Rows</div><div class="v">{cards['rows']:,}</div></div>
<div class="card"><div class="k">Accounts</div><div class="v">{cards['accounts']:,}</div></div>
<div class="card"><div class="k">Ads</div><div class="v">{cards['ads']:,}</div></div>
<div class="card"><div class="k">Features</div><div class="v">{cards['features']:,}</div></div>
<div class="card"><div class="k">Revenue WMAPE</div><div class="v">{cards['revenue_wmape']:.1%}</div></div>
<div class="card"><div class="k">Spend WMAPE</div><div class="v">{cards['spend_wmape']:.1%}</div></div>
</div>
<h2>Test Metrics</h2>{table(test_metrics[['target','rows','mae','rmse','r2','wmape','bias','underprediction_rate']])}
<h2>Worst Account-Level Test Examples After Calibration</h2>{table(account_summary)}
</div></body></html>"""
    DASHBOARD_HTML.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train improved full-database 24h raw-metric models and write backtest outputs.")
    parser.add_argument("--daily-input", type=Path, default=DEFAULT_DAILY_INPUT)
    parser.add_argument("--sample-ads", type=int, default=0, help="Optional smoke-test cap on eligible ads.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily, source = load_ad_daily(args.daily_input, sample_ads=args.sample_ads or None)
    dataset, feature_cols = build_features(daily)
    train, valid, test = split_dataset(dataset)
    if train.empty or valid.empty or test.empty:
        raise RuntimeError(f"Bad split sizes: train={len(train):,}, valid={len(valid):,}, test={len(test):,}")
    metrics, backtest = train_and_score(train, valid, test, feature_cols)
    metrics.to_csv(METRICS_CSV, index=False)
    backtest.to_csv(BACKTEST_CSV, index=False)
    joblib.dump(
        {
            "feature_cols": feature_cols,
            "raw_targets": RAW_TARGETS,
            "source": source,
            "min_ad_days": MIN_AD_DAYS,
            "train_end": str(TRAIN_END.date()),
            "valid_end": str(VALID_END.date()),
            "training_basis": "full_db_optimized_raw_metric_histgb_with_validation_calibration",
            "calibration_json": str(CALIBRATION_JSON),
        },
        MODEL_DIR / "metadata.joblib",
    )
    write_dashboard(metrics, backtest, source, len(feature_cols))

    test_metrics = metrics[metrics["split"].eq("test")]
    lines = [
        "Adunbox Full-Database Optimized 24h Model",
        "",
        f"Source: {source}",
        f"Daily rows after eligibility filter: {len(daily):,}",
        f"Feature rows: {len(dataset):,}",
        f"Train rows: {len(train):,}",
        f"Valid rows: {len(valid):,}",
        f"Test rows: {len(test):,}",
        f"Features: {len(feature_cols):,}",
        f"Model dir: {MODEL_DIR}",
        f"Metrics: {METRICS_CSV}",
        f"Backtest: {BACKTEST_CSV}",
        f"Dashboard: {DASHBOARD_HTML}",
        f"Calibration: {CALIBRATION_JSON}",
        "",
        "Test WMAPE / Bias:",
    ]
    for rec in test_metrics.itertuples(index=False):
        lines.append(f"- {rec.target}: wmape={float(rec.wmape):.4f}, bias={float(rec.bias):.4f}, r2={float(rec.r2):.4f}")
    SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
