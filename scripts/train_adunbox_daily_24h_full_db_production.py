from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import train_adunbox_daily_24h_full_db_optimized as base

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - optional dependency fallback
    LGBMRegressor = None


BASE_DIR = Path(r"G:\ml_model_historical_data")
ORIGINAL_DAILY = base.DEFAULT_DAILY_INPUT
RECENT_DAILY = Path(r"H:\adunbox_daily_breakdown_kpis.csv")
OUTPUT_DIR = BASE_DIR / "github_release" / "outputs"
MODEL_DIR = BASE_DIR / "github_release" / "models" / "adunbox_daily_24h_histgb_full_db_production"
FULL_READY_FLAG = MODEL_DIR / "production_full_ready.flag"
METRICS_CSV = OUTPUT_DIR / "adunbox_daily_24h_full_db_production__metrics.csv"
BACKTEST_CSV = OUTPUT_DIR / "adunbox_daily_24h_full_db_production__backtest.csv"
SUMMARY_TXT = OUTPUT_DIR / "adunbox_daily_24h_full_db_production__summary.txt"
CALIBRATION_JSON = OUTPUT_DIR / "adunbox_daily_24h_full_db_production__calibration.json"

TRAIN_END = pd.Timestamp("2026-05-20")
VALID_END = pd.Timestamp("2026-05-28")
RECENCY_WEIGHT_START = pd.Timestamp("2026-05-13")
RECENCY_WEIGHT_MULTIPLIER = 2.5
MODEL_MAX_ITER = 120
MODEL_BACKEND = "lightgbm" if LGBMRegressor is not None else "histgb"
REVENUE_FOCUS_MODE = True
REVENUE_CLIP_Q = 0.995
FEATURE_UPGRADE_NAME = "deduped_lags_1_2_7_14_roll_3_7_14_revenue_quality_segment_weighted"


def load_single_daily_flexible(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    conversion_col = "conversions" if "conversions" in header else ("tracker_conversions" if "tracker_conversions" in header else "tracker_conversion")
    revenue_col = "conversions_value" if "conversions_value" in header else "tracker_revenue"
    usecols = ["entity_type", "date", "timezone", *base.ENTITY_COLS, "spend", "impressions", "inline_link_clicks", conversion_col, revenue_col]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=50_000, low_memory=False):
        chunk = chunk[chunk["entity_type"].astype(str).str.lower().eq("ad")].copy()
        if chunk.empty:
            continue
        rename_map = {}
        if conversion_col != "tracker_conversions":
            rename_map[conversion_col] = "tracker_conversions"
        if revenue_col != "tracker_revenue":
            rename_map[revenue_col] = "tracker_revenue"
        if rename_map:
            chunk = chunk.rename(columns=rename_map)
        for col in base.ENTITY_COLS:
            chunk[col] = base.normalize_id(chunk[col])
        chunk["timezone"] = chunk["timezone"].fillna("").astype(str)
        chunk["local_date"] = pd.to_datetime(chunk["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
        chunk = chunk[chunk["local_date"].notna()].copy()
        for col in base.RAW_TARGETS:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0.0).astype("float32")
        chunk = (
            chunk[["local_date", "timezone", *base.ENTITY_COLS, *base.RAW_TARGETS]]
            .groupby(["local_date", "timezone", *base.ENTITY_COLS], as_index=False)[base.RAW_TARGETS]
            .sum()
        )
        parts.append(chunk)
    if not parts:
        raise RuntimeError(f"No ad-level rows found in {path}")
    daily = pd.concat(parts, ignore_index=True)
    return (
        daily.groupby(["local_date", "timezone", *base.ENTITY_COLS], as_index=False)[base.RAW_TARGETS]
        .sum()
        .sort_values(["ad_id", "local_date"])
        .reset_index(drop=True)
    )


def load_multi_source_daily(paths: list[Path], sample_ads: int | None = None) -> tuple[pd.DataFrame, str]:
    frames = []
    sources = []
    dedupe_keys = ["local_date", "timezone", *base.ENTITY_COLS]
    for source_priority, path in enumerate(paths):
        if not path.exists():
            continue
        daily = load_single_daily_flexible(path)
        daily["__source_priority"] = source_priority
        frames.append(daily)
        sources.append(str(path))
    if not frames:
        raise RuntimeError("No daily input files found.")
    combined = pd.concat(frames, ignore_index=True)
    before_dedupe = len(combined)
    combined = (
        combined.sort_values([*dedupe_keys, "__source_priority"])
        .drop_duplicates(dedupe_keys, keep="last")
        .drop(columns="__source_priority")
        .sort_values(["ad_id", "local_date"])
        .reset_index(drop=True)
    )
    deduped_rows = before_dedupe - len(combined)
    eligible = combined.groupby("ad_id")["local_date"].nunique()
    eligible_ads = eligible[eligible >= base.MIN_AD_DAYS].index.astype(str)
    if sample_ads:
        eligible_ads = eligible.loc[eligible_ads].sort_values(ascending=False).head(sample_ads).index.astype(str)
    combined = combined[combined["ad_id"].isin(eligible_ads)].copy()
    return combined, " + ".join(sources) + f" | cross_source_deduped_rows={deduped_rows:,} keep=later_source"


def split_dataset(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = dataset[dataset["local_date"] <= TRAIN_END].copy()
    valid = dataset[(dataset["local_date"] > TRAIN_END) & (dataset["local_date"] <= VALID_END)].copy()
    test = dataset[dataset["local_date"] > VALID_END].copy()
    return train, valid, test


def build_features_production_safe(daily: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Memory-safe feature builder for full original + recent retraining.

    The optimized research builder is intentionally very wide and merge-heavy.
    On the combined May/June dataset it can exceed local RAM during hierarchy
    feature merges. This production builder keeps the high-signal features:
    D-1/D-2/D-3/D-7 lags, 3d/7d rolling context, spike/stability ratios, and
    target columns. It avoids the account/campaign merge that caused the memory
    allocation failure.
    """
    out = base.add_kpis(base.densify_by_ad(daily))
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
    # Use raw metrics as the primary model inputs. KPIs are still targets for
    # evaluation after prediction, but feeding every KPI lag/rolling variant
    # made the full deduped dataset too wide for local RAM.
    base_cols = [*base.RAW_TARGETS]
    raw_set = set(base.RAW_TARGETS)
    # Keep the strongest daily signals while avoiding a too-wide full-dataset
    # feature matrix. The removed 4/5/6/21-day variants were highly redundant
    # with 3/7/14-day rolling features and caused local RAM pressure after
    # cross-source deduping increased the usable row count.
    lag_days = [1, 2, 7]
    rolling_windows = [3, 7]
    long_context_cols = {"spend", "tracker_conversions", "tracker_revenue"}
    long_rolling_windows = [14]

    for col in base_cols:
        for lag in lag_days:
            name = f"{col}_lag_{lag}d"
            feature_data[name] = grouped[col].shift(lag).fillna(0.0).astype("float32")
            feature_cols.append(name)
        shifted = grouped[col].shift(1)
        shifted_grouped = shifted.groupby(out["ad_id"], sort=False)
        for window in rolling_windows:
            mean_name = f"{col}_roll_mean_{window}d"
            feature_data[mean_name] = (
                shifted_grouped.rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
                .fillna(0.0)
                .astype("float32")
            )
            feature_cols.append(mean_name)
            if col in raw_set:
                sum_name = f"{col}_roll_sum_{window}d"
                feature_data[sum_name] = (
                    shifted_grouped.rolling(window, min_periods=1)
                    .sum()
                    .reset_index(level=0, drop=True)
                    .fillna(0.0)
                    .astype("float32")
                )
                feature_cols.append(sum_name)
        if col in long_context_cols:
            for window in long_rolling_windows:
                mean_name = f"{col}_roll_mean_{window}d"
                sum_name = f"{col}_roll_sum_{window}d"
                feature_data[mean_name] = (
                    shifted_grouped.rolling(window, min_periods=2)
                    .mean()
                    .reset_index(level=0, drop=True)
                    .fillna(0.0)
                    .astype("float32")
                )
                feature_data[sum_name] = (
                    shifted_grouped.rolling(window, min_periods=2)
                    .sum()
                    .reset_index(level=0, drop=True)
                    .fillna(0.0)
                    .astype("float32")
                )
                feature_cols.extend([mean_name, sum_name])

        if col in raw_set:
            same_week_name = f"{col}_same_weekday_last_week"
            feature_data[same_week_name] = grouped[col].shift(7).fillna(0.0).astype("float32")
            feature_cols.append(same_week_name)
            if col in {"spend", "tracker_conversions", "tracker_revenue"}:
                lag_14_name = f"{col}_lag_14d"
                same_week_2_name = f"{col}_same_weekday_2w"
                feature_data[lag_14_name] = grouped[col].shift(14).fillna(0.0).astype("float32")
                feature_data[same_week_2_name] = grouped[col].shift(14).fillna(0.0).astype("float32")
                feature_cols.extend([lag_14_name, same_week_2_name])

    shifted_spend = grouped["spend"].shift(1).fillna(0.0)
    shifted_clicks = grouped["inline_link_clicks"].shift(1).fillna(0.0)
    shifted_conversions = grouped["tracker_conversions"].shift(1).fillna(0.0)
    shifted_revenue = grouped["tracker_revenue"].shift(1).fillna(0.0)
    quality_signals = {
        "tracker_revenue_nonzero_count_7d": shifted_revenue.gt(0).astype("float32"),
        "tracker_conversions_nonzero_count_7d": shifted_conversions.gt(0).astype("float32"),
        "tracker_revenue_nonzero_count_14d": shifted_revenue.gt(0).astype("float32"),
        "tracker_conversions_nonzero_count_14d": shifted_conversions.gt(0).astype("float32"),
        "spend_positive_revenue_zero_count_7d": (shifted_spend.gt(0) & shifted_revenue.eq(0)).astype("float32"),
        "clicks_positive_conversions_zero_count_7d": (shifted_clicks.gt(0) & shifted_conversions.eq(0)).astype("float32"),
        "spend_positive_revenue_zero_count_14d": (shifted_spend.gt(0) & shifted_revenue.eq(0)).astype("float32"),
        "clicks_positive_conversions_zero_count_14d": (shifted_clicks.gt(0) & shifted_conversions.eq(0)).astype("float32"),
    }
    for name, series in quality_signals.items():
        window = 14 if name.endswith("_14d") else 7
        feature_data[name] = (
            series.groupby(out["ad_id"], sort=False)
            .transform(lambda s, w=window: s.rolling(w, min_periods=1).sum())
            .fillna(0.0)
            .astype("float32")
        )
        feature_cols.append(name)

    feature_frame = pd.DataFrame(feature_data, index=out.index)
    out = pd.concat([out, feature_frame], axis=1)

    # Account/campaign hierarchy blending is applied in the serving layer after
    # prediction. Keeping it out of the training matrix avoids a full-data RAM
    # blow-up while still protecting weak-history production forecasts.

    for col in base.RAW_TARGETS:
        mean_3 = out.get(f"{col}_roll_mean_3d", 0.0)
        mean_7 = out.get(f"{col}_roll_mean_7d", 0.0)
        out[f"{col}_trend_3d_vs_7d"] = base.safe_div(mean_3, pd.Series(mean_7).replace(0, np.nan)).fillna(0.0).astype("float32")
        feature_cols.append(f"{col}_trend_3d_vs_7d")

    revenue_mean_7 = out.get("tracker_revenue_roll_mean_7d", pd.Series(0.0, index=out.index))
    revenue_sum_7 = out.get("tracker_revenue_roll_sum_7d", pd.Series(0.0, index=out.index))
    revenue_sum_14 = out.get("tracker_revenue_roll_sum_14d", pd.Series(0.0, index=out.index))
    spend_sum_7 = out.get("spend_roll_sum_7d", pd.Series(0.0, index=out.index))
    spend_sum_14 = out.get("spend_roll_sum_14d", pd.Series(0.0, index=out.index))
    conversion_sum_7 = out.get("tracker_conversions_roll_sum_7d", pd.Series(0.0, index=out.index))
    conversion_sum_14 = out.get("tracker_conversions_roll_sum_14d", pd.Series(0.0, index=out.index))
    out["revenue_density_7d"] = base.safe_div(revenue_sum_7, pd.Series(spend_sum_7).replace(0, np.nan)).fillna(0.0).clip(0, 50).astype("float32")
    out["revenue_density_14d"] = base.safe_div(revenue_sum_14, pd.Series(spend_sum_14).replace(0, np.nan)).fillna(0.0).clip(0, 50).astype("float32")
    out["revenue_per_conversion_7d"] = base.safe_div(revenue_sum_7, pd.Series(conversion_sum_7).replace(0, np.nan)).fillna(0.0).clip(0, 5000).astype("float32")
    out["revenue_per_conversion_14d"] = base.safe_div(revenue_sum_14, pd.Series(conversion_sum_14).replace(0, np.nan)).fillna(0.0).clip(0, 5000).astype("float32")
    out["revenue_7d_vs_14d"] = base.safe_div(revenue_sum_7, pd.Series(revenue_sum_14).replace(0, np.nan)).fillna(0.0).clip(0, 10).astype("float32")
    out["spend_7d_vs_14d"] = base.safe_div(spend_sum_7, pd.Series(spend_sum_14).replace(0, np.nan)).fillna(0.0).clip(0, 10).astype("float32")
    out["is_revenue_stable_7d"] = ((out["tracker_revenue_nonzero_count_7d"] >= 3) & (out["revenue_density_7d"] > 0)).astype("float32")
    out["is_revenue_stable_14d"] = ((out["tracker_revenue_nonzero_count_14d"] >= 5) & (out["revenue_density_14d"] > 0)).astype("float32")
    out["is_revenue_spiky_7d"] = ((out["tracker_revenue_nonzero_count_7d"] <= 2) & (out["revenue_density_7d"] > 0)).astype("float32")
    out["is_zero_heavy_revenue_14d"] = ((out["tracker_revenue_nonzero_count_14d"] <= 2) & (spend_sum_14 > 0)).astype("float32")
    feature_cols.extend(
        [
            "revenue_density_7d",
            "revenue_density_14d",
            "revenue_per_conversion_7d",
            "revenue_per_conversion_14d",
            "revenue_7d_vs_14d",
            "spend_7d_vs_14d",
            "is_revenue_stable_7d",
            "is_revenue_stable_14d",
            "is_revenue_spiky_7d",
            "is_zero_heavy_revenue_14d",
        ]
    )

    for target in base.RAW_TARGETS:
        out[f"target_24h_{target}"] = out[target].astype("float32")
    out["target_24h_roas"] = out["kpi_roas"].astype("float32")
    out["target_24h_profit"] = out["kpi_profit"].astype("float32")
    out["target_24h_ctr"] = out["kpi_ctr"].astype("float32")
    out["target_24h_cvr"] = out["kpi_cvr"].astype("float32")
    out["target_24h_cpc"] = out["kpi_cpc"].astype("float32")
    out["target_24h_cpm"] = out["kpi_cpm"].astype("float32")
    out = out[out["days_active"] >= base.MIN_AD_DAYS].replace([np.inf, -np.inf], np.nan).fillna(0.0).copy()
    feature_cols = list(dict.fromkeys(feature_cols))
    return out, feature_cols


def sample_weight(frame: pd.DataFrame, raw_target: str) -> np.ndarray:
    weights = np.ones(len(frame), dtype=np.float32)
    weights[pd.to_datetime(frame["local_date"]) >= RECENCY_WEIGHT_START] = RECENCY_WEIGHT_MULTIPLIER

    revenue_signal = frame.get("tracker_revenue_nonzero_count_7d", pd.Series(0.0, index=frame.index))
    revenue_signal_14 = frame.get("tracker_revenue_nonzero_count_14d", pd.Series(0.0, index=frame.index))
    conversion_signal = frame.get("tracker_conversions_nonzero_count_7d", pd.Series(0.0, index=frame.index))
    conversion_signal_14 = frame.get("tracker_conversions_nonzero_count_14d", pd.Series(0.0, index=frame.index))
    spend_zero_risk = frame.get("spend_positive_revenue_zero_count_7d", pd.Series(0.0, index=frame.index))
    spend_zero_risk_14 = frame.get("spend_positive_revenue_zero_count_14d", pd.Series(0.0, index=frame.index))
    click_zero_risk = frame.get("clicks_positive_conversions_zero_count_7d", pd.Series(0.0, index=frame.index))
    click_zero_risk_14 = frame.get("clicks_positive_conversions_zero_count_14d", pd.Series(0.0, index=frame.index))

    consistent_signal = (revenue_signal >= 2) | (conversion_signal >= 2)
    strong_signal = (revenue_signal >= 4) | (conversion_signal >= 4)
    strong_signal_14 = (revenue_signal_14 >= 6) | (conversion_signal_14 >= 6)
    suspicious_tracking_zero = (
        ((spend_zero_risk >= 5) & (revenue_signal == 0))
        | ((click_zero_risk >= 5) & (conversion_signal == 0))
        | ((spend_zero_risk_14 >= 10) & (revenue_signal_14 <= 1))
        | ((click_zero_risk_14 >= 10) & (conversion_signal_14 <= 1))
    )

    if raw_target in {"tracker_revenue", "tracker_conversions"}:
        weights[consistent_signal.to_numpy()] *= 1.60
        weights[strong_signal.to_numpy()] *= 1.30
        weights[strong_signal_14.to_numpy()] *= 1.20
        weights[suspicious_tracking_zero.to_numpy()] *= 0.30
        target_values = pd.to_numeric(frame.get(f"target_24h_{raw_target}", 0.0), errors="coerce").fillna(0.0)
        target_positive = target_values > 0
        weights[target_positive.to_numpy()] *= 1.35
        if target_positive.any():
            high_cut = float(target_values[target_positive].quantile(0.75))
            weights[target_values.ge(high_cut).to_numpy()] *= 1.15
        if REVENUE_FOCUS_MODE and raw_target == "tracker_revenue":
            stable = frame.get("is_revenue_stable_7d", pd.Series(0.0, index=frame.index)).fillna(0.0).ge(1)
            spiky = frame.get("is_revenue_spiky_7d", pd.Series(0.0, index=frame.index)).fillna(0.0).ge(1)
            campaign_signal = frame.get("campaign_tracker_revenue_nonzero_count_7d", pd.Series(0.0, index=frame.index)).fillna(0.0).ge(2)
            campaign_signal_14 = frame.get("campaign_tracker_revenue_nonzero_count_14d", pd.Series(0.0, index=frame.index)).fillna(0.0).ge(5)
            account_signal = frame.get("account_tracker_revenue_nonzero_count_7d", pd.Series(0.0, index=frame.index)).fillna(0.0).ge(2)
            account_signal_14 = frame.get("account_tracker_revenue_nonzero_count_14d", pd.Series(0.0, index=frame.index)).fillna(0.0).ge(5)
            weights[stable.to_numpy()] *= 1.60
            weights[(campaign_signal | account_signal).to_numpy()] *= 1.20
            weights[(campaign_signal_14 | account_signal_14).to_numpy()] *= 1.15
            weights[spiky.to_numpy()] *= 0.75
    elif raw_target == "inline_link_clicks":
        weights[(conversion_signal >= 1).to_numpy()] *= 1.15
        weights[strong_signal.to_numpy()] *= 1.05
    else:
        weights[consistent_signal.to_numpy()] *= 1.10
        weights[strong_signal.to_numpy()] *= 1.05
    return np.clip(weights, 0.20, 6.0).astype("float32")


def build_lightgbm_model(raw_target: str):
    if LGBMRegressor is None:
        return None
    is_revenue_like = raw_target in {"tracker_revenue", "tracker_conversions"}
    return LGBMRegressor(
        objective="regression",
        n_estimators=max(MODEL_MAX_ITER, 360 if is_revenue_like else 280),
        learning_rate=0.035 if is_revenue_like else 0.045,
        num_leaves=63 if is_revenue_like else 47,
        min_child_samples=45 if is_revenue_like else 75,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.25 if is_revenue_like else 0.15,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
    )


def build_production_model():
    return base.HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=MODEL_MAX_ITER,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        max_bins=128,
        l2_regularization=0.05,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=12,
        random_state=42,
    )


def build_target_model(raw_target: str):
    if MODEL_BACKEND == "lightgbm" and LGBMRegressor is not None:
        return build_lightgbm_model(raw_target)
    if REVENUE_FOCUS_MODE and raw_target == "tracker_revenue":
        return base.HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.045,
            max_iter=max(MODEL_MAX_ITER, 220),
            max_leaf_nodes=63,
            min_samples_leaf=45,
            max_bins=160,
            l2_regularization=0.08,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=16,
            random_state=42,
        )
    return build_production_model()


def training_target_values(y: pd.Series, raw_target: str) -> pd.Series:
    if REVENUE_FOCUS_MODE and raw_target == "tracker_revenue" and y.gt(0).any():
        cap = float(y[y > 0].quantile(REVENUE_CLIP_Q))
        return y.clip(lower=0.0, upper=cap).astype("float32")
    return y.astype("float32")


def ensure_calibration_compatibility(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for target in base.RAW_TARGETS:
        roll_sum = f"{target}_roll_sum_7d"
        recent_sum = f"{target}_recent_sum_7d"
        if recent_sum not in out.columns:
            out[recent_sum] = pd.to_numeric(out.get(roll_sum, 0.0), errors="coerce").fillna(0.0).astype("float32")

        cv = f"{target}_cv_7d"
        if cv not in out.columns:
            out[cv] = 0.0

        lag_vs = f"{target}_lag1_vs_mean_7d"
        if lag_vs not in out.columns:
            lag_1 = pd.to_numeric(out.get(f"{target}_lag_1d", 0.0), errors="coerce").fillna(0.0)
            mean_7 = pd.to_numeric(out.get(f"{target}_roll_mean_7d", 0.0), errors="coerce").fillna(0.0)
            out[lag_vs] = base.safe_div(lag_1, mean_7.replace(0, np.nan)).fillna(0.0).astype("float32")
    return out


def train_and_score(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    old_model_dir = base.MODEL_DIR
    old_calibration = base.CALIBRATION_JSON
    base.MODEL_DIR = MODEL_DIR
    base.CALIBRATION_JSON = CALIBRATION_JSON
    try:
        metrics: list[dict[str, object]] = []
        pred_frames = {"valid": pd.DataFrame(index=valid.index), "test": pd.DataFrame(index=test.index)}
        calibration_specs: dict[str, dict[str, object]] = {}
        train = base.add_error_control_features(train)
        valid = base.add_error_control_features(valid)
        test = base.add_error_control_features(test)
        train = ensure_calibration_compatibility(train)
        valid = ensure_calibration_compatibility(valid)
        test = ensure_calibration_compatibility(test)
        x_train = train[feature_cols].astype("float32")
        x_valid = valid[feature_cols].astype("float32")
        x_test = test[feature_cols].astype("float32")
        for raw_target in base.RAW_TARGETS:
            target = f"target_24h_{raw_target}"
            y_train_actual = train[target].astype("float32")
            y_train = training_target_values(y_train_actual, raw_target)
            y_valid = valid[target].astype("float32")
            y_test = test[target].astype("float32")
            model = build_target_model(raw_target)
            weights = sample_weight(train, raw_target)
            model.fit(x_train, np.log1p(np.maximum(0.0, y_train)), sample_weight=weights)
            joblib.dump(model, MODEL_DIR / f"{target}.joblib")
            pred_valid = np.maximum(0.0, np.expm1(model.predict(x_valid))).astype("float32")
            pred_test = np.maximum(0.0, np.expm1(model.predict(x_test))).astype("float32")
            calibration_specs[raw_target] = base.fit_target_calibration(valid, raw_target, pred_valid)
            pred_valid_cal = base.apply_target_calibration(valid, raw_target, pred_valid, calibration_specs[raw_target])
            pred_test_cal = base.apply_target_calibration(test, raw_target, pred_test, calibration_specs[raw_target])
            pred_frames["valid"][f"pred_24h_{raw_target}"] = pred_valid
            pred_frames["test"][f"pred_24h_{raw_target}"] = pred_test
            pred_frames["valid"][f"pred_calibrated_24h_{raw_target}"] = pred_valid_cal
            pred_frames["test"][f"pred_calibrated_24h_{raw_target}"] = pred_test_cal
            for col, values in base.prediction_ranges(valid, raw_target, pred_valid_cal, calibration_specs[raw_target]).items():
                pred_frames["valid"][col] = values
            for col, values in base.prediction_ranges(test, raw_target, pred_test_cal, calibration_specs[raw_target]).items():
                pred_frames["test"][col] = values
            metrics.append(base.eval_rows(target, "valid", y_valid, pred_valid))
            metrics.append(base.eval_rows(target, "test", y_test, pred_test))
            metrics.append(base.eval_rows(f"{target}_calibrated", "valid", y_valid, pred_valid_cal))
            metrics.append(base.eval_rows(f"{target}_calibrated", "test", y_test, pred_test_cal))
        CALIBRATION_JSON.write_text(json.dumps(calibration_specs, indent=2), encoding="utf-8")

        metrics_df = pd.DataFrame(metrics)
        backtest_parts = []
        for split, frame, preds in [("valid", valid, pred_frames["valid"]), ("test", test, pred_frames["test"])]:
            pred_base = preds[[f"pred_24h_{target}" for target in base.RAW_TARGETS]].copy()
            pred_cal = preds[[f"pred_calibrated_24h_{target}" for target in base.RAW_TARGETS]].rename(
                columns={f"pred_calibrated_24h_{target}": f"pred_24h_{target}" for target in base.RAW_TARGETS}
            )
            pred_kpis = base.derive_kpis(pred_base)
            pred_kpis_cal = base.derive_kpis(pred_cal).rename(
                columns={col: col.replace("pred_24h_", "pred_calibrated_24h_") for col in base.derive_kpis(pred_cal).columns}
            )
            actual_kpis = frame[
                ["local_date", "timezone", *base.ENTITY_COLS, "days_active", *base.RAW_TARGETS, "kpi_roas", "kpi_profit", "kpi_ctr", "kpi_cvr", "kpi_cpc", "kpi_cpm"]
            ].copy()
            out = actual_kpis.rename(columns={col: f"actual_24h_{col}" for col in base.RAW_TARGETS})
            out["prediction_segment"] = base.classify_prediction_segment(frame).to_numpy()
            for col in pred_kpis.columns:
                out[col] = pred_kpis[col].to_numpy()
            for col in pred_kpis_cal.columns:
                out[col] = pred_kpis_cal[col].to_numpy()
            for raw_target in base.RAW_TARGETS:
                for bound in ["p10", "p50", "p90"]:
                    col = f"pred_{bound}_24h_{raw_target}"
                    if col in preds:
                        out[col] = preds[col].to_numpy()
            out["split"] = split
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
                        pd.DataFrame([base.eval_rows(target, split, frame[target].astype("float32"), pred_kpis[pred_col].to_numpy())]),
                        pd.DataFrame([base.eval_rows(f"{target}_calibrated", split, frame[target].astype("float32"), pred_kpis_cal[cal_col].to_numpy())]),
                    ],
                    ignore_index=True,
                )
        return metrics_df, pd.concat(backtest_parts, ignore_index=True)
    finally:
        base.MODEL_DIR = old_model_dir
        base.CALIBRATION_JSON = old_calibration


def main() -> None:
    global TRAIN_END, VALID_END, RECENCY_WEIGHT_START, RECENCY_WEIGHT_MULTIPLIER, MODEL_MAX_ITER, MODEL_BACKEND, REVENUE_FOCUS_MODE, REVENUE_CLIP_Q

    parser = argparse.ArgumentParser(description="Production retrain with original + recent daily data and recency weighting.")
    parser.add_argument("--original-daily", type=Path, default=ORIGINAL_DAILY)
    parser.add_argument("--recent-daily", type=Path, default=RECENT_DAILY)
    parser.add_argument("--extra-daily", type=Path, nargs="*", default=[], help="Optional additional daily CSVs to merge into training, e.g. Vybres.")
    parser.add_argument("--train-end", type=str, default=str(TRAIN_END.date()), help="Last date included in train split, e.g. 2026-05-25.")
    parser.add_argument("--valid-end", type=str, default=str(VALID_END.date()), help="Last date included in validation split. Dates after this become test.")
    parser.add_argument("--recency-weight-start", type=str, default=str(RECENCY_WEIGHT_START.date()))
    parser.add_argument("--recency-weight-multiplier", type=float, default=RECENCY_WEIGHT_MULTIPLIER)
    parser.add_argument("--max-iter", type=int, default=MODEL_MAX_ITER, help="Boosting iterations. Use 120 for production HistGB, higher for LightGBM review runs.")
    parser.add_argument("--model-backend", choices=["histgb", "lightgbm"], default=MODEL_BACKEND, help="Use LightGBM if installed, otherwise HistGB fallback.")
    parser.add_argument("--revenue-focus", action="store_true", help="Use revenue-focused weights/model and clipped revenue training target.")
    parser.add_argument("--revenue-clip-q", type=float, default=REVENUE_CLIP_Q, help="Positive revenue quantile cap for revenue-focused training.")
    parser.add_argument("--sample-ads", type=int, default=0)
    args = parser.parse_args()

    TRAIN_END = pd.Timestamp(args.train_end)
    VALID_END = pd.Timestamp(args.valid_end)
    RECENCY_WEIGHT_START = pd.Timestamp(args.recency_weight_start)
    RECENCY_WEIGHT_MULTIPLIER = float(args.recency_weight_multiplier)
    MODEL_MAX_ITER = int(args.max_iter)
    MODEL_BACKEND = args.model_backend
    if MODEL_BACKEND == "lightgbm" and LGBMRegressor is None:
        print("LightGBM is not installed in this environment; falling back to HistGB.")
        MODEL_BACKEND = "histgb"
    REVENUE_FOCUS_MODE = bool(args.revenue_focus)
    REVENUE_CLIP_Q = float(args.revenue_clip_q)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_paths = [args.original_daily, args.recent_daily, *args.extra_daily]
    daily, source = load_multi_source_daily(input_paths, sample_ads=args.sample_ads or None)
    dataset, feature_cols = build_features_production_safe(daily)
    train, valid, test = split_dataset(dataset)
    if train.empty or valid.empty or test.empty:
        raise RuntimeError(f"Bad split sizes: train={len(train):,}, valid={len(valid):,}, test={len(test):,}")
    metrics, backtest = train_and_score(train, valid, test, feature_cols)
    metrics.to_csv(METRICS_CSV, index=False)
    backtest.to_csv(BACKTEST_CSV, index=False)
    joblib.dump(
        {
            "feature_cols": feature_cols,
            "raw_targets": base.RAW_TARGETS,
            "source": source,
            "min_ad_days": base.MIN_AD_DAYS,
            "train_end": str(TRAIN_END.date()),
            "valid_end": str(VALID_END.date()),
            "recency_weight_start": str(RECENCY_WEIGHT_START.date()),
            "recency_weight_multiplier": RECENCY_WEIGHT_MULTIPLIER,
            "model_backend": MODEL_BACKEND,
            "lightgbm_available": LGBMRegressor is not None,
            "revenue_focus": REVENUE_FOCUS_MODE,
            "revenue_clip_q": REVENUE_CLIP_Q,
            "feature_upgrade": FEATURE_UPGRADE_NAME,
            "sample_ads": int(args.sample_ads or 0),
            "training_basis": "production_original_plus_recent_recency_weighted_segment_calibrated_memory_safe_features",
            "calibration_json": str(CALIBRATION_JSON),
        },
        MODEL_DIR / "metadata.joblib",
    )
    if args.sample_ads:
        if FULL_READY_FLAG.exists():
            FULL_READY_FLAG.unlink()
    else:
        FULL_READY_FLAG.write_text("full production retrain completed\n", encoding="utf-8")
    test_metrics = metrics[metrics["split"].eq("test")]
    lines = [
        "Adunbox 24h Production Retrained Model",
        "",
        f"Source: {source}",
        f"Daily rows after eligibility filter: {len(daily):,}",
        f"Feature rows: {len(dataset):,}",
        f"Train rows: {len(train):,}",
        f"Valid rows: {len(valid):,}",
        f"Test rows: {len(test):,}",
        f"Date range: {dataset['local_date'].min()} -> {dataset['local_date'].max()}",
        f"Recency weight: {RECENCY_WEIGHT_MULTIPLIER}x from {RECENCY_WEIGHT_START.date()}",
        f"Model backend: {MODEL_BACKEND} (LightGBM available={LGBMRegressor is not None})",
        f"Boosting iterations: {MODEL_MAX_ITER}",
        f"Revenue focus: {REVENUE_FOCUS_MODE} clip_q={REVENUE_CLIP_Q}",
        f"Feature upgrade: {FEATURE_UPGRADE_NAME}",
        f"Features: {len(feature_cols):,}",
        f"Model dir: {MODEL_DIR}",
        f"Metrics: {METRICS_CSV}",
        f"Backtest: {BACKTEST_CSV}",
        f"Calibration: {CALIBRATION_JSON}",
        "",
        "Test WMAPE / Bias / R2:",
    ]
    for rec in test_metrics.itertuples(index=False):
        lines.append(f"- {rec.target}: wmape={float(rec.wmape):.4f}, bias={float(rec.bias):.4f}, r2={float(rec.r2):.4f}")
    SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
