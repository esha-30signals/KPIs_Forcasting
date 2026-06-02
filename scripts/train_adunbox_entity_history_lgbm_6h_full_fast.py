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
MODEL_DIR = BASE_DIR / "models" / "adunbox_entity_history_lgbm_6h_full_fast"
METRICS_PATH = BASE_DIR / "docs" / "adunbox_entity_history_lgbm_6h_full_fast__metrics.csv"
SUMMARY_PATH = BASE_DIR / "docs" / "adunbox_entity_history_lgbm_6h_full_fast__summary.txt"

ENTITY_COLS = ["account_id", "campaign_id", "adset_id", "ad_id"]
RAW_FEATURES = ["spend", "impressions", "clicks", "inline_link_clicks", "tracker_conversions", "tracker_revenue"]
TARGET_COLS = [
    "target_spend",
    "target_impressions",
    "target_inline_link_clicks",
    "target_tracker_conversions",
    "target_tracker_revenue",
]
METRIC_COLS = ["spend", "impressions", "inline_link_clicks", "tracker_conversions", "tracker_revenue"]
WINDOWS = [1, 3, 6, 12, 24, 48, 72, 168]
SAME_WINDOW_LAGS = [24, 48, 168]
TARGET_HOURS = 6
MIN_OBSERVED_HISTORY_HOURS = 24
MIN_RECENT_24H_SPEND = 1.0
MIN_RECENT_24H_IMPRESSIONS = 50.0
MIN_RECENT_24H_CLICKS = 1.0
TRAIN_END = pd.Timestamp("2026-04-12 18:00:00")
VALID_END = pd.Timestamp("2026-04-27 06:00:00")


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
    for chunk in pd.read_csv(HOURLY_INPUT_PATH, usecols=usecols, chunksize=250_000, low_memory=False):
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

    hourly = pd.concat(parts, ignore_index=True)
    hourly = (
        hourly.groupby(["local_ts", "timezone", *ENTITY_COLS], as_index=False)[RAW_FEATURES]
        .sum()
        .sort_values(["ad_id", "local_ts"])
        .reset_index(drop=True)
    )
    return hourly


def window_sum(cumulative: np.ndarray, end_pos: int, width: int) -> np.ndarray:
    start = max(0, end_pos - width)
    return cumulative[end_pos] - cumulative[start]


def future_sum(cumulative: np.ndarray, start_pos: int, width: int) -> np.ndarray:
    return cumulative[start_pos + width] - cumulative[start_pos]


def same_window_sum(cumulative: np.ndarray, end_pos: int, width: int, lag: int) -> np.ndarray:
    lag_end = end_pos - lag
    if lag_end <= 0:
        return np.zeros(cumulative.shape[1], dtype=np.float32)
    lag_start = max(0, lag_end - width)
    return cumulative[lag_end] - cumulative[lag_start]


def add_ratio_features(features: dict[str, float], prefix: str, values: np.ndarray) -> None:
    spend, impressions, clicks, conversions, revenue = [float(x) for x in values]
    features[f"{prefix}_ctr"] = safe_div_scalar(clicks, impressions) * 100.0
    features[f"{prefix}_cpc"] = safe_div_scalar(spend, clicks)
    features[f"{prefix}_cpm"] = safe_div_scalar(spend, impressions) * 1000.0
    features[f"{prefix}_cvr"] = safe_div_scalar(conversions, clicks) * 100.0
    features[f"{prefix}_roas"] = safe_div_scalar(revenue, spend)
    features[f"{prefix}_profit"] = revenue - spend


def build_samples(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    targets: list[np.ndarray] = []

    for ad_id, group in hourly.groupby("ad_id", sort=False):
        group = group.sort_values("local_ts").drop_duplicates("local_ts", keep="last")
        idx = pd.date_range(group["local_ts"].min(), group["local_ts"].max(), freq="1h")
        dense = group.set_index("local_ts").reindex(idx)
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
            if anchor_ts.hour not in {0, 6, 12, 18}:
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
            target = future_sum(cumulative, pos + 1, TARGET_HOURS).astype(np.float32)
            if float(target.sum()) <= 0:
                continue

            row = {
                "anchor_ts": anchor_ts,
                "hour": anchor_ts.hour,
                "dow": anchor_ts.dayofweek,
                "hour_sin": float(np.sin(2 * np.pi * anchor_ts.hour / 24.0)),
                "hour_cos": float(np.cos(2 * np.pi * anchor_ts.hour / 24.0)),
                "dow_sin": float(np.sin(2 * np.pi * anchor_ts.dayofweek / 7.0)),
                "dow_cos": float(np.cos(2 * np.pi * anchor_ts.dayofweek / 7.0)),
                "observed_history_hours": observed_history,
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

            for lag in SAME_WINDOW_LAGS:
                values = same_window_sum(cumulative, pos + 1, TARGET_HOURS, lag)
                for metric_idx, metric in enumerate(METRIC_COLS):
                    row[f"{metric}_same_6h_lag_{lag}h"] = float(values[metric_idx])
                add_ratio_features(row, f"kpi_same_6h_lag_{lag}h", values)

            cum_values = cumulative[pos + 1]
            for metric_idx, metric in enumerate(METRIC_COLS):
                row[f"{metric}_cum"] = float(cum_values[metric_idx])
            add_ratio_features(row, "kpi_cum", cum_values)

            rows.append(row)
            targets.append(target)

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


def build_model() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=700,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=60,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.3,
        random_state=42,
        n_jobs=1,
        verbosity=-1,
    )


def evaluate(target: str, split: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    return {
        "target": target,
        "split": split,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train() -> None:
    hourly = load_hourly_aggregated()
    features, targets = build_samples(hourly)
    if features.empty:
        raise RuntimeError("No 6h samples were built from the full hourly dataset.")

    train_mask, valid_mask, test_mask = split_by_time(features)
    drop_cols = ["anchor_ts", *ENTITY_COLS]
    feature_cols = [col for col in features.columns if col not in drop_cols]
    x = features[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float32")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for target in TARGET_COLS:
        model = build_model()
        y_train = np.log1p(np.maximum(targets.loc[train_mask, target].to_numpy(dtype=np.float32), 0.0))
        y_valid = np.log1p(np.maximum(targets.loc[valid_mask, target].to_numpy(dtype=np.float32), 0.0))
        y_test = targets.loc[test_mask, target].to_numpy(dtype=np.float32)
        model.fit(
            x.loc[train_mask],
            y_train,
            eval_set=[(x.loc[valid_mask], y_valid)],
            eval_metric="l1",
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
            feature_name=feature_cols,
        )
        pred_valid = np.maximum(0.0, np.expm1(model.predict(x.loc[valid_mask])))
        pred_test = np.maximum(0.0, np.expm1(model.predict(x.loc[test_mask])))
        rows.append(evaluate(target, "valid", targets.loc[valid_mask, target], pred_valid))
        rows.append(evaluate(target, "test", y_test, pred_test))
        joblib.dump(model, MODEL_DIR / f"{target}.joblib")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(METRICS_PATH, index=False)
    joblib.dump(
        {
            "feature_cols": feature_cols,
            "target_cols": TARGET_COLS,
            "training_basis": "full_hourly_summary_lag_lightgbm_6h",
            "hourly_input": str(HOURLY_INPUT_PATH),
        },
        MODEL_DIR / "metadata.joblib",
    )
    test_view = metrics[metrics["split"] == "test"]
    lines = [
        "Adunbox Entity History LightGBM 6H Full Fast Summary",
        "",
        f"Hourly input: {HOURLY_INPUT_PATH}",
        f"Hourly rows after aggregation: {len(hourly):,}",
        f"Anchor count: {len(features):,}",
        f"Feature count: {len(feature_cols):,}",
        f"Train rows: {int(train_mask.sum()):,}",
        f"Validation rows: {int(valid_mask.sum()):,}",
        f"Test rows: {int(test_mask.sum()):,}",
        "",
        "Test R2:",
    ]
    for _, row in test_view.iterrows():
        lines.append(f"- {row['target']}: {row['r2']:.6f}")
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(SUMMARY_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    train()
