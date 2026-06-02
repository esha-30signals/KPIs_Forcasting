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
MODEL_DIR = BASE_DIR / "models" / "adunbox_entity_history_lgbm_168h_padded_6h"
METRICS_PATH = BASE_DIR / "docs" / "adunbox_entity_history_lgbm_168h_padded_6h__metrics.csv"
SUMMARY_PATH = BASE_DIR / "docs" / "adunbox_entity_history_lgbm_168h_padded_6h__summary.txt"

SEQ_HOURS = 168
TARGET_HOURS = 6
MIN_OBSERVED_HISTORY_HOURS = 24
VALID_ANCHOR_HOURS = {0, 6, 12, 18}
MIN_RECENT_24H_SPEND = 1.0
MIN_RECENT_24H_IMPRESSIONS = 50.0
MIN_RECENT_24H_CLICKS = 1.0
TRAIN_END = pd.Timestamp("2026-04-12 18:00:00")
VALID_END = pd.Timestamp("2026-04-27 06:00:00")

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
SEQ_FEATURES = [
    "spend",
    "impressions",
    "clicks",
    "inline_link_clicks",
    "tracker_conversions",
    "tracker_revenue",
    "kpi_ctr",
    "kpi_cpc",
    "kpi_cpm",
    "kpi_cvr",
    "kpi_cost_per_result",
    "kpi_roas",
    "kpi_profit",
    "kpi_roi",
    "hour_of_day_sin",
    "hour_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]
RAW_SEQUENCE_FEATURES = [
    "spend",
    "impressions",
    "clicks",
    "inline_link_clicks",
    "tracker_conversions",
    "tracker_revenue",
    "hour_of_day_sin",
    "hour_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]
SEQUENCE_MODE = os.getenv("ADUNBOX_LGBM_SEQUENCE_MODE", "full").strip().lower()
ACTIVE_SEQ_FEATURES = RAW_SEQUENCE_FEATURES if SEQUENCE_MODE == "raw" else SEQ_FEATURES
STATIC_FEATURES_6H = [
    "recent_1h_spend",
    "recent_1h_revenue",
    "recent_1h_roas",
    "recent_3h_spend_avg",
    "recent_3h_revenue_avg",
    "recent_3h_roas",
    "recent_6h_spend",
    "recent_6h_revenue",
    "recent_6h_conversions",
    "recent_24h_spend",
    "recent_24h_revenue",
    "recent_24h_conversions",
    "same_window_1d_spend",
    "same_window_1d_revenue",
    "same_window_1d_conversions",
    "same_window_2d_spend",
    "same_window_2d_revenue",
    "same_window_2d_conversions",
    "same_window_7d_spend",
    "same_window_7d_revenue",
    "same_window_7d_conversions",
    "cum_spend",
    "cum_revenue",
    "hours_active",
]


def normalize_id(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    return text.str.replace(r"\.0$", "", regex=True)


def safe_div(numer, denom, multiplier: float = 1.0):
    numer_s = pd.Series(numer, copy=False)
    denom_s = pd.Series(denom, copy=False)
    out = pd.Series(np.zeros(len(numer_s), dtype=np.float32), index=numer_s.index)
    mask = denom_s != 0
    out.loc[mask] = (numer_s.loc[mask] / denom_s.loc[mask]) * multiplier
    return out.astype("float32")


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


def add_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["kpi_ctr"] = safe_div(out["inline_link_clicks"], out["impressions"], 100.0)
    out["kpi_cpc"] = safe_div(out["spend"], out["inline_link_clicks"])
    out["kpi_cpm"] = safe_div(out["spend"], out["impressions"], 1000.0)
    out["kpi_cvr"] = safe_div(out["tracker_conversions"], out["inline_link_clicks"], 100.0)
    out["kpi_cost_per_result"] = safe_div(out["spend"], out["tracker_conversions"])
    out["kpi_roas"] = safe_div(out["tracker_revenue"], out["spend"])
    out["kpi_profit"] = out["tracker_revenue"] - out["spend"]
    out["kpi_roi"] = safe_div(out["kpi_profit"], out["spend"])
    return out


def load_hourly() -> pd.DataFrame:
    usecols = ["date", *ENTITY_COLS, "timezone", *RAW_FEATURES]
    df = pd.read_csv(HOURLY_INPUT_PATH, usecols=usecols, low_memory=False)
    for col in ENTITY_COLS:
        df[col] = normalize_id(df[col])
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df[df["date"].notna()].copy()
    df["timezone"] = df["timezone"].fillna("").astype(str)
    for col in RAW_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32")
    df["local_ts"] = to_local_ts(df["date"], df["timezone"])
    df = (
        df.groupby(["local_ts", "timezone", *ENTITY_COLS], as_index=False)[RAW_FEATURES]
        .sum()
        .sort_values(["ad_id", "local_ts"])
        .reset_index(drop=True)
    )
    df["date"] = df["local_ts"]
    df = add_kpis(df)
    df["hour_of_day_sin"] = np.sin(2 * np.pi * df["local_ts"].dt.hour / 24.0).astype("float32")
    df["hour_of_day_cos"] = np.cos(2 * np.pi * df["local_ts"].dt.hour / 24.0).astype("float32")
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["local_ts"].dt.dayofweek / 7.0).astype("float32")
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["local_ts"].dt.dayofweek / 7.0).astype("float32")
    print(f"Loaded hourly rows after aggregation: {len(df):,}")
    return df


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


def dense_ad_group(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("local_ts").drop_duplicates("local_ts", keep="last")
    idx = pd.date_range(group["local_ts"].min(), group["local_ts"].max(), freq="1h")
    dense = group.set_index("local_ts").reindex(idx)
    for col in [*SEQ_FEATURES, *METRIC_COLS]:
        if col in dense.columns:
            dense[col] = dense[col].fillna(0.0)
    dense["hour_of_day_sin"] = np.sin(2 * np.pi * idx.hour / 24.0).astype("float32")
    dense["hour_of_day_cos"] = np.cos(2 * np.pi * idx.hour / 24.0).astype("float32")
    dense["day_of_week_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7.0).astype("float32")
    dense["day_of_week_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7.0).astype("float32")
    dense["observed_hour"] = dense["date"].notna().astype("int8")
    dense["date"] = dense["date"].ffill().bfill()
    dense["local_ts"] = idx
    return dense.reset_index(drop=True)


def build_static(cumulative: np.ndarray, pos: int) -> np.ndarray:
    recent_1 = window_sum(cumulative, pos + 1, 1)
    recent_3 = window_sum(cumulative, pos + 1, 3) / 3.0
    recent_6 = window_sum(cumulative, pos + 1, 6)
    recent_24 = window_sum(cumulative, pos + 1, 24)
    same_1d = same_window_sum(cumulative, pos + 1, TARGET_HOURS, 24)
    same_2d = same_window_sum(cumulative, pos + 1, TARGET_HOURS, 48)
    same_7d = same_window_sum(cumulative, pos + 1, TARGET_HOURS, 168)
    cum_total = cumulative[pos + 1]
    return np.array(
        [
            recent_1[0], recent_1[4], safe_div_scalar(recent_1[4], recent_1[0]),
            recent_3[0], recent_3[4], safe_div_scalar(recent_3[4], recent_3[0]),
            recent_6[0], recent_6[4], recent_6[3],
            recent_24[0], recent_24[4], recent_24[3],
            same_1d[0], same_1d[4], same_1d[3],
            same_2d[0], same_2d[4], same_2d[3],
            same_7d[0], same_7d[4], same_7d[3],
            cum_total[0], cum_total[4], float(pos + 1),
        ],
        dtype=np.float32,
    )


def passes_recent_activity(cumulative: np.ndarray, pos: int) -> bool:
    recent_24 = window_sum(cumulative, pos + 1, 24)
    return (
        float(recent_24[0]) >= MIN_RECENT_24H_SPEND
        and float(recent_24[1]) >= MIN_RECENT_24H_IMPRESSIONS
        and float(recent_24[2]) >= MIN_RECENT_24H_CLICKS
    )


def build_samples(hourly: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seqs: list[np.ndarray] = []
    statics: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    anchor_dates: list[pd.Timestamp] = []
    for group_num, (_, group_idx) in enumerate(hourly.groupby("ad_id", sort=False).groups.items(), start=1):
        if group_num % 500 == 0:
            print(f"Built samples through ad group {group_num:,}; samples so far: {len(seqs):,}")
        dense = dense_ad_group(hourly.loc[group_idx])
        if len(dense) < SEQ_HOURS + TARGET_HOURS:
            continue
        metric_arr = dense[METRIC_COLS].to_numpy(dtype=np.float32)
        observed = dense["observed_hour"].to_numpy(dtype=np.int8)
        cumulative = np.vstack([np.zeros((1, len(METRIC_COLS)), dtype=np.float32), np.cumsum(metric_arr, axis=0)])
        obs_cum = np.concatenate([[0], np.cumsum(observed)])
        for pos in range(SEQ_HOURS - 1, len(dense) - TARGET_HOURS):
            anchor_ts = pd.Timestamp(dense.loc[pos, "local_ts"])
            if anchor_ts.hour not in VALID_ANCHOR_HOURS:
                continue
            observed_history = int(obs_cum[pos + 1] - obs_cum[pos + 1 - SEQ_HOURS])
            if observed_history < MIN_OBSERVED_HISTORY_HOURS:
                continue
            if not passes_recent_activity(cumulative, pos):
                continue
            target = future_sum(cumulative, pos + 1, TARGET_HOURS).astype(np.float32)
            if float(target.sum()) <= 0:
                continue
            seq_sample = dense.loc[pos - SEQ_HOURS + 1:pos, ACTIVE_SEQ_FEATURES].to_numpy(dtype=np.float32)
            if np.isnan(seq_sample).any():
                continue
            seqs.append(seq_sample)
            statics.append(build_static(cumulative, pos))
            targets.append(target)
            anchor_dates.append(anchor_ts)
    return (
        np.asarray(seqs, dtype=np.float32),
        np.asarray(statics, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(anchor_dates),
    )


def split_by_anchor_time(anchor_dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_times = pd.to_datetime(anchor_dates)
    train_mask = all_times <= TRAIN_END
    valid_mask = (all_times > TRAIN_END) & (all_times <= VALID_END)
    test_mask = all_times > VALID_END
    if train_mask.any() and valid_mask.any() and test_mask.any():
        return train_mask, valid_mask, test_mask
    unique_times = np.array(sorted(pd.Series(all_times).unique()))
    train_end = int(len(unique_times) * 0.70)
    valid_end = int(len(unique_times) * 0.85)
    train_cut = pd.Timestamp(unique_times[max(0, train_end - 1)])
    valid_cut = pd.Timestamp(unique_times[max(train_end, valid_end - 1)])
    return all_times <= train_cut, (all_times > train_cut) & (all_times <= valid_cut), all_times > valid_cut


def build_feature_matrix(seq: np.ndarray, static: np.ndarray) -> tuple[np.ndarray, list[str]]:
    flat_seq = seq.reshape(seq.shape[0], -1)
    seq_names = [
        f"hist_t_minus_{SEQ_HOURS - hour}_{feature}"
        for hour in range(SEQ_HOURS)
        for feature in ACTIVE_SEQ_FEATURES
    ]
    static_names = list(STATIC_FEATURES_6H)
    x = np.concatenate([static, flat_seq], axis=1).astype(np.float32)
    return x, [*static_names, *seq_names]


def build_model() -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression_l1",
        n_estimators=int(os.getenv("ADUNBOX_LGBM_ESTIMATORS", "450")),
        learning_rate=0.035,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.05,
        reg_lambda=0.25,
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
    hourly = load_hourly()
    seq, static, target, anchor_dates = build_samples(hourly)
    if len(seq) == 0:
        raise RuntimeError("No padded 168h -> 6h samples were built.")
    print(f"Built sequential samples: {len(seq):,}; sequence mode: {SEQUENCE_MODE}; sequence features: {len(ACTIVE_SEQ_FEATURES):,}")

    x, feature_names = build_feature_matrix(seq, static)
    train_mask, valid_mask, test_mask = split_by_anchor_time(anchor_dates)
    x_train, x_valid, x_test = x[train_mask], x[valid_mask], x[test_mask]
    y_train, y_valid, y_test = target[train_mask], target[valid_mask], target[test_mask]

    if min(len(x_train), len(x_valid), len(x_test)) == 0:
        raise RuntimeError(
            f"Bad split sizes: train={len(x_train):,}, valid={len(x_valid):,}, test={len(x_test):,}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for idx, target_name in enumerate(TARGET_COLS):
        model = build_model()
        yt_train = np.log1p(np.maximum(y_train[:, idx], 0.0))
        yt_valid = np.log1p(np.maximum(y_valid[:, idx], 0.0))
        yt_test = y_test[:, idx]
        model.fit(
            x_train,
            yt_train,
            eval_set=[(x_valid, yt_valid)],
            eval_metric="l1",
            callbacks=[early_stopping(60, verbose=False), log_evaluation(0)],
            feature_name=feature_names,
        )
        pred_valid = np.expm1(model.predict(x_valid))
        pred_test = np.expm1(model.predict(x_test))
        pred_valid = np.maximum(pred_valid, 0.0)
        pred_test = np.maximum(pred_test, 0.0)

        rows.append(evaluate(target_name, "valid", y_valid[:, idx], pred_valid))
        rows.append(evaluate(target_name, "test", yt_test, pred_test))
        joblib.dump(model, MODEL_DIR / f"{target_name}.joblib")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(METRICS_PATH, index=False)
    joblib.dump(
        {
            "feature_names": feature_names,
            "seq_hours": SEQ_HOURS,
            "target_hours": TARGET_HOURS,
            "sequence_feature_names": ACTIVE_SEQ_FEATURES,
            "static_feature_names": STATIC_FEATURES_6H,
            "target_cols": list(TARGET_COLS),
            "training_basis": "lightgbm_flattened_168h_padded_6h",
            "sequence_mode": SEQUENCE_MODE,
            "hourly_input": str(HOURLY_INPUT_PATH),
        },
        MODEL_DIR / "metadata.joblib",
    )

    test_view = metrics[metrics["split"] == "test"]
    lines = [
        "Adunbox Entity History LightGBM 168h Padded 6H Summary",
        "",
        f"Hourly input: {HOURLY_INPUT_PATH}",
        f"Anchor count: {len(anchor_dates):,}",
        f"Feature count: {x.shape[1]:,}",
        f"Sequence mode: {SEQUENCE_MODE}",
        f"Train rows: {len(x_train):,}",
        f"Validation rows: {len(x_valid):,}",
        f"Test rows: {len(x_test):,}",
        "",
        "Test R2:",
    ]
    for _, row in test_view.iterrows():
        lines.append(f"- {row['target']}: {row['r2']:.6f}")
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(SUMMARY_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    train()
