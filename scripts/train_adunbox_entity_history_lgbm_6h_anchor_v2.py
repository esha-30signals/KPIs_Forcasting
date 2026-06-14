from __future__ import annotations

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
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(os.getenv("ADUNBOX_PROJECT_DIR", Path(__file__).resolve().parents[1]))
HOURLY_INPUT_PATH = Path(os.getenv("ADUNBOX_HOURLY_INPUT", BASE_DIR / "data" / "traffic_reports.csv"))
MODEL_DIR = BASE_DIR / "models" / "adunbox_entity_history_lgbm_6h_anchor_v2"
METRICS_PATH = BASE_DIR / "docs" / "adunbox_entity_history_lgbm_6h_anchor_v2__metrics.csv"
ANCHOR_METRICS_PATH = BASE_DIR / "docs" / "adunbox_entity_history_lgbm_6h_anchor_v2__anchor_metrics.csv"
SUMMARY_PATH = BASE_DIR / "docs" / "adunbox_entity_history_lgbm_6h_anchor_v2__summary.txt"

ENTITY_COLS = ["account_id", "campaign_id", "adset_id", "ad_id"]
RAW_FEATURES = ["spend", "impressions", "clicks", "inline_link_clicks", "tracker_conversions", "tracker_revenue"]
METRIC_COLS = ["spend", "impressions", "inline_link_clicks", "tracker_conversions", "tracker_revenue"]
TARGET_COLS = [
    "target_spend",
    "target_impressions",
    "target_inline_link_clicks",
    "target_tracker_conversions",
    "target_tracker_revenue",
]
WINDOWS = [1, 3, 6, 12, 24, 48, 72, 168]
SAME_WINDOW_LAGS = [24, 48, 72, 96, 120, 144, 168]
VALID_ANCHOR_HOURS = {0, 6, 12, 18}
TARGET_HOURS = 6
FINAL_HOURLY_GROUPBY = os.getenv("ADUNBOX_FINAL_HOURLY_GROUPBY", "0").strip() == "1"
MIN_OBSERVED_HISTORY_HOURS = int(os.getenv("ADUNBOX_MIN_OBSERVED_HISTORY_HOURS", "24"))
MIN_RECENT_24H_SPEND = float(os.getenv("ADUNBOX_MIN_RECENT_24H_SPEND", "1.0"))
MIN_RECENT_24H_IMPRESSIONS = float(os.getenv("ADUNBOX_MIN_RECENT_24H_IMPRESSIONS", "50.0"))
MIN_RECENT_24H_CLICKS = float(os.getenv("ADUNBOX_MIN_RECENT_24H_CLICKS", "1.0"))
TRAIN_END = pd.Timestamp(os.getenv("ADUNBOX_6H_TRAIN_END", "2026-04-12 18:00:00"))
VALID_END = pd.Timestamp(os.getenv("ADUNBOX_6H_VALID_END", "2026-04-27 06:00:00"))


def normalize_id(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    return text.str.replace(r"\.0$", "", regex=True)


def safe_div_scalar(numer: float, denom: float) -> float:
    return float(numer / denom) if float(denom) != 0.0 else 0.0


def to_local_ts(utc_ts: pd.Series, timezone: pd.Series) -> pd.Series:
    local_ts = pd.Series(pd.NaT, index=utc_ts.index, dtype="datetime64[ns]")
    tz_values = timezone.fillna("").astype(str)
    for tz_name, idx in tz_values.groupby(tz_values).groups.items():
        utc_slice = utc_ts.loc[idx]
        try:
            local_slice = utc_slice.dt.tz_convert(str(tz_name).strip()) if str(tz_name).strip() else utc_slice
        except Exception:
            local_slice = utc_slice
        local_ts.loc[idx] = local_slice.dt.tz_localize(None).dt.floor("h").to_numpy()
    return local_ts


def load_hourly_aggregated() -> pd.DataFrame:
    usecols = ["date", *ENTITY_COLS, "timezone", *RAW_FEATURES]
    parts: list[pd.DataFrame] = []
    for chunk_idx, chunk in enumerate(
        pd.read_csv(HOURLY_INPUT_PATH, usecols=usecols, chunksize=250_000, low_memory=False),
        start=1,
    ):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce", utc=True)
        chunk = chunk[chunk["date"].notna()].copy()
        for col in ENTITY_COLS:
            chunk[col] = normalize_id(chunk[col])
        chunk["timezone"] = chunk["timezone"].fillna("").astype(str)
        for col in RAW_FEATURES:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0.0).astype("float32")
        chunk["local_ts"] = to_local_ts(chunk["date"], chunk["timezone"])
        part = chunk.groupby(["local_ts", "timezone", *ENTITY_COLS], as_index=False)[RAW_FEATURES].sum()
        parts.append(part)
        if chunk_idx % 10 == 0:
            print(f"loaded chunks={chunk_idx:,}; partial_groups={sum(len(p) for p in parts):,}")

    hourly = pd.concat(parts, ignore_index=True)
    if FINAL_HOURLY_GROUPBY:
        hourly = hourly.groupby(["local_ts", "timezone", *ENTITY_COLS], as_index=False)[RAW_FEATURES].sum()
    hourly = hourly.sort_values(["ad_id", "local_ts"]).reset_index(drop=True)
    return hourly


def window_sum(cumulative: np.ndarray, end_pos: int, width: int) -> np.ndarray:
    start = max(0, end_pos - width)
    return cumulative[end_pos] - cumulative[start]


def future_sum(cumulative: np.ndarray, start_pos: int, width: int) -> np.ndarray:
    return cumulative[start_pos + width] - cumulative[start_pos]


def lagged_target_window(cumulative: np.ndarray, future_start: int, width: int, lag: int) -> np.ndarray:
    lag_start = future_start - lag
    lag_end = lag_start + width
    if lag_start < 0 or lag_end < 0:
        return np.zeros(cumulative.shape[1], dtype=np.float32)
    return cumulative[lag_end] - cumulative[lag_start]


def add_ratio_features(features: dict[str, float], prefix: str, values: np.ndarray) -> None:
    spend, impressions, clicks, conversions, revenue = [float(x) for x in values]
    features[f"{prefix}_ctr"] = safe_div_scalar(clicks, impressions) * 100.0
    features[f"{prefix}_cpc"] = safe_div_scalar(spend, clicks)
    features[f"{prefix}_cpm"] = safe_div_scalar(spend, impressions) * 1000.0
    features[f"{prefix}_cvr"] = safe_div_scalar(conversions, clicks) * 100.0
    features[f"{prefix}_roas"] = safe_div_scalar(revenue, spend)
    features[f"{prefix}_profit"] = revenue - spend


def add_same_window_stats(row: dict[str, object], lag_values: np.ndarray) -> None:
    # lag_values shape: days x metrics, representing prior target-window sums.
    for metric_idx, metric in enumerate(METRIC_COLS):
        values = lag_values[:, metric_idx].astype(np.float32)
        row[f"{metric}_same_6h_7d_mean"] = float(np.mean(values))
        row[f"{metric}_same_6h_7d_median"] = float(np.median(values))
        row[f"{metric}_same_6h_7d_std"] = float(np.std(values))
        row[f"{metric}_same_6h_7d_min"] = float(np.min(values))
        row[f"{metric}_same_6h_7d_max"] = float(np.max(values))
        row[f"{metric}_same_6h_7d_p75"] = float(np.percentile(values, 75))
        row[f"{metric}_same_6h_7d_p90"] = float(np.percentile(values, 90))
        row[f"{metric}_same_6h_7d_last_over_mean"] = safe_div_scalar(float(values[0]), float(np.mean(values)))
    add_ratio_features(row, "kpi_same_6h_7d_mean", np.mean(lag_values, axis=0))
    add_ratio_features(row, "kpi_same_6h_7d_median", np.median(lag_values, axis=0))
    add_ratio_features(row, "kpi_same_6h_7d_p90", np.percentile(lag_values, 90, axis=0))


def build_samples(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    targets: list[np.ndarray] = []
    skipped_short = 0
    skipped_inactive = 0
    dense_groups = 0

    for ad_idx, (_, group) in enumerate(hourly.groupby("ad_id", sort=False), start=1):
        group = group.sort_values("local_ts").drop_duplicates("local_ts", keep="last")
        if len(group) < MIN_OBSERVED_HISTORY_HOURS:
            skipped_short += 1
            continue
        min_ts = pd.Timestamp(group["local_ts"].min())
        max_ts = pd.Timestamp(group["local_ts"].max())
        span_hours = int((max_ts - min_ts).total_seconds() // 3600) + 1
        if span_hours < 168 + TARGET_HOURS:
            skipped_short += 1
            continue
        totals = group[METRIC_COLS].sum(numeric_only=True)
        if (
            float(totals.get("spend", 0.0)) < MIN_RECENT_24H_SPEND
            or float(totals.get("impressions", 0.0)) < MIN_RECENT_24H_IMPRESSIONS
            or float(totals.get("inline_link_clicks", 0.0)) < MIN_RECENT_24H_CLICKS
        ):
            skipped_inactive += 1
            continue

        idx = pd.date_range(group["local_ts"].min(), group["local_ts"].max(), freq="1h")
        dense = group.set_index("local_ts").reindex(idx)
        dense_groups += 1
        observed = dense["ad_id"].notna().to_numpy(dtype=np.int8)
        for col in RAW_FEATURES:
            dense[col] = dense[col].fillna(0.0)
        for col in ["timezone", *ENTITY_COLS]:
            dense[col] = dense[col].ffill().bfill()
        dense["local_ts"] = idx

        metric_arr = dense[METRIC_COLS].to_numpy(dtype=np.float32)
        cumulative = np.vstack([np.zeros((1, len(METRIC_COLS)), dtype=np.float32), np.cumsum(metric_arr, axis=0)])
        obs_cum = np.concatenate([[0], np.cumsum(observed)])

        for pos in range(167, len(dense) - TARGET_HOURS):
            anchor_ts = pd.Timestamp(dense.iloc[pos]["local_ts"])
            if anchor_ts.hour not in VALID_ANCHOR_HOURS:
                continue
            observed_history = int(obs_cum[pos + 1] - obs_cum[max(0, pos + 1 - 168)])
            if observed_history < MIN_OBSERVED_HISTORY_HOURS:
                continue
            recent_24 = window_sum(cumulative, pos + 1, 24)
            if (
                float(recent_24[0]) < MIN_RECENT_24H_SPEND
                or float(recent_24[1]) < MIN_RECENT_24H_IMPRESSIONS
                or float(recent_24[2]) < MIN_RECENT_24H_CLICKS
            ):
                continue
            future_start = pos + 1
            target = future_sum(cumulative, future_start, TARGET_HOURS).astype(np.float32)
            if float(target.sum()) <= 0:
                continue

            row: dict[str, object] = {
                "anchor_ts": anchor_ts,
                "anchor_hour": anchor_ts.hour,
                "anchor_hour_0": int(anchor_ts.hour == 0),
                "anchor_hour_6": int(anchor_ts.hour == 6),
                "anchor_hour_12": int(anchor_ts.hour == 12),
                "anchor_hour_18": int(anchor_ts.hour == 18),
                "dow": anchor_ts.dayofweek,
                "is_weekend": int(anchor_ts.dayofweek >= 5),
                "hour_sin": float(np.sin(2 * np.pi * anchor_ts.hour / 24.0)),
                "hour_cos": float(np.cos(2 * np.pi * anchor_ts.hour / 24.0)),
                "dow_sin": float(np.sin(2 * np.pi * anchor_ts.dayofweek / 7.0)),
                "dow_cos": float(np.cos(2 * np.pi * anchor_ts.dayofweek / 7.0)),
                "observed_history_hours": observed_history,
                "observed_history_ratio": observed_history / 168.0,
                "observed_recent_24h": int(obs_cum[pos + 1] - obs_cum[max(0, pos + 1 - 24)]),
                "hours_active": float(pos + 1),
            }
            for col in ENTITY_COLS:
                row[col] = dense.iloc[pos][col]

            for window in WINDOWS:
                values = window_sum(cumulative, pos + 1, window)
                for metric_idx, metric in enumerate(METRIC_COLS):
                    row[f"{metric}_sum_{window}h"] = float(values[metric_idx])
                    row[f"{metric}_avg_{window}h"] = float(values[metric_idx]) / float(window)
                add_ratio_features(row, f"kpi_{window}h", values)

            lag_values = []
            for lag in SAME_WINDOW_LAGS:
                values = lagged_target_window(cumulative, future_start, TARGET_HOURS, lag)
                lag_values.append(values)
                days = lag // 24
                for metric_idx, metric in enumerate(METRIC_COLS):
                    row[f"{metric}_same_target_6h_d{days}"] = float(values[metric_idx])
                add_ratio_features(row, f"kpi_same_target_6h_d{days}", values)
            add_same_window_stats(row, np.vstack(lag_values))

            cum_values = cumulative[pos + 1]
            for metric_idx, metric in enumerate(METRIC_COLS):
                row[f"{metric}_cum"] = float(cum_values[metric_idx])
            add_ratio_features(row, "kpi_cum", cum_values)

            rows.append(row)
            targets.append(target)

        if ad_idx % 5000 == 0:
            print(
                f"processed_ads={ad_idx:,}; dense_groups={dense_groups:,}; samples={len(rows):,}; "
                f"skipped_short={skipped_short:,}; skipped_inactive={skipped_inactive:,}",
                flush=True,
            )

    target_df = pd.DataFrame(targets, columns=TARGET_COLS)
    return pd.DataFrame(rows), target_df


def split_by_time(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = pd.to_datetime(df["anchor_ts"])
    train_mask = (times <= TRAIN_END).to_numpy()
    valid_mask = ((times > TRAIN_END) & (times <= VALID_END)).to_numpy()
    test_mask = (times > VALID_END).to_numpy()
    if train_mask.any() and valid_mask.any() and test_mask.any():
        return train_mask, valid_mask, test_mask
    unique_times = np.array(sorted(times.unique()))
    train_end = int(len(unique_times) * 0.70)
    valid_end = int(len(unique_times) * 0.85)
    train_cut = pd.Timestamp(unique_times[max(0, train_end - 1)])
    valid_cut = pd.Timestamp(unique_times[max(train_end, valid_end - 1)])
    return (times <= train_cut).to_numpy(), ((times > train_cut) & (times <= valid_cut)).to_numpy(), (times > valid_cut).to_numpy()


def build_model(target: str) -> LGBMRegressor:
    params_by_target = {
        "target_spend": dict(n_estimators=900, learning_rate=0.035, num_leaves=47, min_child_samples=45, reg_alpha=0.03, reg_lambda=0.20),
        "target_impressions": dict(n_estimators=900, learning_rate=0.035, num_leaves=47, min_child_samples=45, reg_alpha=0.03, reg_lambda=0.20),
        "target_inline_link_clicks": dict(n_estimators=950, learning_rate=0.035, num_leaves=55, min_child_samples=40, reg_alpha=0.02, reg_lambda=0.18),
        "target_tracker_conversions": dict(n_estimators=900, learning_rate=0.030, num_leaves=31, min_child_samples=80, reg_alpha=0.12, reg_lambda=0.55),
        "target_tracker_revenue": dict(n_estimators=950, learning_rate=0.028, num_leaves=31, min_child_samples=90, reg_alpha=0.16, reg_lambda=0.70),
    }
    params = params_by_target[target]
    return LGBMRegressor(
        objective="regression_l1",
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.82,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
        **params,
    )


def evaluate(target: str, split: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    denom = float(np.sum(np.abs(y_true)))
    wmape = float(np.sum(np.abs(y_pred - y_true)) / denom) if denom else 0.0
    bias = float(np.sum(y_pred - y_true) / denom) if denom else 0.0
    return {
        "target": target,
        "split": split,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "wmape": wmape,
        "bias": bias,
    }


def train() -> None:
    hourly = load_hourly_aggregated()
    features, targets = build_samples(hourly)
    if features.empty:
        raise RuntimeError("No 6h anchor-v2 samples were built from the hourly dataset.")

    train_mask, valid_mask, test_mask = split_by_time(features)
    drop_cols = ["anchor_ts", *ENTITY_COLS]
    feature_cols = [col for col in features.columns if col not in drop_cols]
    x = features[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float32")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    anchor_rows: list[dict[str, object]] = []

    for target in TARGET_COLS:
        print(f"training target={target}; features={len(feature_cols):,}; rows={len(x):,}")
        model = build_model(target)
        y_train = np.log1p(np.maximum(targets.loc[train_mask, target].to_numpy(dtype=np.float32), 0.0))
        y_valid = np.log1p(np.maximum(targets.loc[valid_mask, target].to_numpy(dtype=np.float32), 0.0))
        y_test = targets.loc[test_mask, target].to_numpy(dtype=np.float32)
        model.fit(
            x.loc[train_mask],
            y_train,
            eval_set=[(x.loc[valid_mask], y_valid)],
            eval_metric="l1",
            callbacks=[early_stopping(70, verbose=False), log_evaluation(0)],
            feature_name=feature_cols,
        )
        pred_valid = np.maximum(0.0, np.expm1(model.predict(x.loc[valid_mask])))
        pred_test = np.maximum(0.0, np.expm1(model.predict(x.loc[test_mask])))
        rows.append(evaluate(target, "valid", targets.loc[valid_mask, target].to_numpy(dtype=np.float32), pred_valid))
        rows.append(evaluate(target, "test", y_test, pred_test))

        test_hours = features.loc[test_mask, "anchor_hour"].to_numpy()
        for anchor_hour in sorted(VALID_ANCHOR_HOURS):
            hour_mask = test_hours == anchor_hour
            if int(hour_mask.sum()) < 2:
                continue
            hour_result = evaluate(target, "test", y_test[hour_mask], pred_test[hour_mask])
            hour_result["anchor_hour"] = anchor_hour
            anchor_rows.append(hour_result)
        joblib.dump(model, MODEL_DIR / f"{target}.joblib")

    metrics = pd.DataFrame(rows)
    anchor_metrics = pd.DataFrame(anchor_rows)
    metrics.to_csv(METRICS_PATH, index=False)
    anchor_metrics.to_csv(ANCHOR_METRICS_PATH, index=False)
    joblib.dump(
        {
            "feature_cols": feature_cols,
            "target_cols": TARGET_COLS,
            "training_basis": "anchor_window_lag_stats_lightgbm_6h_v2",
            "hourly_input": str(HOURLY_INPUT_PATH),
            "same_window_lags": SAME_WINDOW_LAGS,
            "windows": WINDOWS,
            "valid_anchor_hours": sorted(VALID_ANCHOR_HOURS),
        },
        MODEL_DIR / "metadata.joblib",
    )
    test_view = metrics[metrics["split"] == "test"]
    lines = [
        "Adunbox Entity History LightGBM 6H Anchor V2 Summary",
        "",
        f"Hourly input: {HOURLY_INPUT_PATH}",
        f"Hourly rows after aggregation: {len(hourly):,}",
        f"Anchor count: {len(features):,}",
        f"Feature count: {len(feature_cols):,}",
        f"Train rows: {int(train_mask.sum()):,}",
        f"Validation rows: {int(valid_mask.sum()):,}",
        f"Test rows: {int(test_mask.sum()):,}",
        "",
        "Feature upgrades:",
        "- D-1 through D-7 same target-window raw metric lags",
        "- 7-day same-window mean/median/std/min/max/p75/p90/last-over-mean",
        "- anchor-hour one-hot and observed-history reliability features",
        "- per-target LightGBM hyperparameters",
        "",
        "Test R2:",
    ]
    for _, row in test_view.iterrows():
        lines.append(f"- {row['target']}: {row['r2']:.6f}")
    lines.append("")
    lines.append("Test WMAPE:")
    for _, row in test_view.iterrows():
        lines.append(f"- {row['target']}: {row['wmape']:.6f}")
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(SUMMARY_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    train()
