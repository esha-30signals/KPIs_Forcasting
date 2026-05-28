from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


BASE_DIR = Path(os.getenv("ADUNBOX_PROJECT_DIR", Path(__file__).resolve().parents[1]))
HOURLY_INPUT_PATH = Path(os.getenv("ADUNBOX_HOURLY_INPUT", BASE_DIR / "data" / "traffic_reports.csv"))

SEQ_HOURS = 168
MODEL_DIR_6H = BASE_DIR / "models" / "adunbox_local_midnight_gru_fullrange_168h_6h"
MODEL_DIR_24H = BASE_DIR / "models" / "adunbox_local_midnight_gru_fullrange_168h_24h"
SUMMARY_6H_PATH = BASE_DIR / "adunbox_local_midnight_gru_fullrange_168h_6h__summary.txt"
SUMMARY_24H_PATH = BASE_DIR / "adunbox_local_midnight_gru_fullrange_168h_24h__summary.txt"
METRICS_6H_PATH = BASE_DIR / "adunbox_local_midnight_gru_fullrange_168h_6h__metrics.csv"
METRICS_24H_PATH = BASE_DIR / "adunbox_local_midnight_gru_fullrange_168h_24h__metrics.csv"

RAW_FEATURES = [
    "spend",
    "impressions",
    "clicks",
    "inline_link_clicks",
    "tracker_conversions",
    "tracker_revenue",
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

STATIC_FEATURES = [
    "recent_6h_spend",
    "recent_6h_revenue",
    "recent_6h_conversions",
    "recent_24h_spend",
    "recent_24h_revenue",
    "recent_24h_conversions",
    "recent_72h_spend",
    "recent_72h_revenue",
    "recent_72h_conversions",
    "recent_168h_spend",
    "recent_168h_revenue",
    "recent_168h_conversions",
]

TARGET_COLS_6H = [
    "target_next_6h_spend",
    "target_next_6h_impressions",
    "target_next_6h_inline_link_clicks",
    "target_next_6h_tracker_conversions",
    "target_next_6h_tracker_revenue",
]

TARGET_COLS_24H = [
    "target_next_24h_spend",
    "target_next_24h_impressions",
    "target_next_24h_inline_link_clicks",
    "target_next_24h_tracker_conversions",
    "target_next_24h_tracker_revenue",
]


def safe_div(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray, multiplier: float = 1.0) -> pd.Series | np.ndarray:
    if isinstance(a, pd.Series) or isinstance(b, pd.Series):
        a_series = pd.Series(a, copy=False)
        b_series = pd.Series(b, copy=False)
        b_safe = b_series.where(b_series != 0, 1.0)
        out = (a_series / b_safe) * multiplier
        return out.where(b_series != 0, 0.0)

    numer = np.asarray(a, dtype=np.float32)
    denom = np.asarray(b, dtype=np.float32)
    out = np.zeros_like(numer, dtype=np.float32)
    mask = denom != 0
    out[mask] = (numer[mask] / denom[mask]) * multiplier
    return out


def normalize_id(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    return text.str.replace(r"\.0$", "", regex=True)


def to_local_ts(date_series: pd.Series, tz_series: pd.Series) -> pd.Series:
    utc_ts = pd.to_datetime(date_series, errors="coerce", utc=True)
    local_parts: list[pd.Series] = []
    tz_values = tz_series.fillna("").astype(str)
    for tz_name, idx in tz_values.groupby(tz_values).groups.items():
        utc_slice = utc_ts.loc[idx]
        if not str(tz_name).strip():
            local_slice = utc_slice.dt.tz_localize(None)
        else:
            try:
                local_slice = utc_slice.dt.tz_convert(str(tz_name).strip()).dt.tz_localize(None)
            except Exception:
                local_slice = utc_slice.dt.tz_localize(None)
        local_parts.append(pd.Series(local_slice, index=idx))
    return pd.concat(local_parts).sort_index()


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
    usecols = ["date", "account_id", "campaign_id", "adset_id", "ad_id", "timezone", *RAW_FEATURES]
    df = pd.read_csv(HOURLY_INPUT_PATH, usecols=usecols, low_memory=False)
    for col in ["account_id", "campaign_id", "adset_id", "ad_id"]:
        df[col] = normalize_id(df[col])
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df[df["date"].notna()].copy()
    df["timezone"] = df["timezone"].fillna("").astype(str)
    for col in RAW_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32")
    df = add_kpis(df)
    df["local_ts"] = to_local_ts(df["date"], df["timezone"])
    df["hour_of_day"] = df["local_ts"].dt.hour.astype("int16")
    df["day_of_week"] = df["local_ts"].dt.dayofweek.astype("int16")
    df["hour_of_day_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24.0).astype("float32")
    df["hour_of_day_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24.0).astype("float32")
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0).astype("float32")
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0).astype("float32")
    return df.sort_values(["ad_id", "local_ts"]).reset_index(drop=True)


def expected_hour_range(end_ts: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(end=end_ts, periods=periods, freq="1h")


def build_samples(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seq_samples: list[np.ndarray] = []
    static_samples: list[np.ndarray] = []
    target_6h: list[np.ndarray] = []
    target_24h: list[np.ndarray] = []
    anchor_dates: list[pd.Timestamp] = []

    metric_cols = ["spend", "impressions", "inline_link_clicks", "tracker_conversions", "tracker_revenue"]

    for _, group_idx in df.groupby("ad_id", sort=False).groups.items():
        group = df.loc[group_idx].sort_values("local_ts").reset_index(drop=True)
        if len(group) < SEQ_HOURS + 24:
            continue

        group["is_midnight"] = (group["local_ts"].dt.hour == 0) & (group["local_ts"].dt.minute == 0)
        midnight_positions = np.flatnonzero(group["is_midnight"].to_numpy())
        if len(midnight_positions) == 0:
            continue

        local_ts_arr = group["local_ts"].to_numpy(dtype="datetime64[ns]")
        metric_arr = group[metric_cols].to_numpy(dtype=np.float32, copy=False)
        cumulative = np.vstack([np.zeros((1, metric_arr.shape[1]), dtype=np.float32), np.cumsum(metric_arr, axis=0)])

        for pos in midnight_positions:
            history_end = pos - 1
            history_start = history_end - SEQ_HOURS + 1
            if history_start < 0:
                continue

            history_times = group.loc[history_start:history_end, "local_ts"]
            expected_hist = expected_hour_range(history_times.iloc[-1], SEQ_HOURS)
            if not history_times.reset_index(drop=True).equals(pd.Series(expected_hist)):
                continue

            future_start = pos
            future_6_end = future_start + 6
            future_24_end = future_start + 24
            if future_24_end > len(group):
                continue

            future_times_6 = group.loc[future_start:future_6_end - 1, "local_ts"]
            expected_6 = pd.date_range(start=group.loc[future_start, "local_ts"], periods=6, freq="1h")
            if not future_times_6.reset_index(drop=True).equals(pd.Series(expected_6)):
                continue

            future_times_24 = group.loc[future_start:future_24_end - 1, "local_ts"]
            expected_24 = pd.date_range(start=group.loc[future_start, "local_ts"], periods=24, freq="1h")
            if not future_times_24.reset_index(drop=True).equals(pd.Series(expected_24)):
                continue

            seq_samples.append(group.loc[history_start:history_end, SEQ_FEATURES].to_numpy(dtype=np.float32))

            recent_6 = cumulative[future_start] - cumulative[max(0, future_start - 6)]
            recent_24 = cumulative[future_start] - cumulative[max(0, future_start - 24)]
            recent_72 = cumulative[future_start] - cumulative[max(0, future_start - 72)]
            recent_168 = cumulative[future_start] - cumulative[max(0, future_start - 168)]
            static_samples.append(
                np.array(
                    [
                        recent_6[0], recent_6[4], recent_6[3],
                        recent_24[0], recent_24[4], recent_24[3],
                        recent_72[0], recent_72[4], recent_72[3],
                        recent_168[0], recent_168[4], recent_168[3],
                    ],
                    dtype=np.float32,
                )
            )

            target_6h.append((cumulative[future_6_end] - cumulative[future_start]).astype(np.float32))
            target_24h.append((cumulative[future_24_end] - cumulative[future_start]).astype(np.float32))
            anchor_dates.append(pd.Timestamp(group.loc[future_start, "date"]))

    return (
        np.asarray(seq_samples, dtype=np.float32),
        np.asarray(static_samples, dtype=np.float32),
        np.asarray(target_6h, dtype=np.float32),
        np.asarray(target_24h, dtype=np.float32),
        np.asarray(anchor_dates),
    )


def split_by_time(anchor_dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_dates = np.array(sorted(pd.to_datetime(anchor_dates).unique()))
    train_end = int(len(unique_dates) * 0.70)
    valid_end = int(len(unique_dates) * 0.85)
    train_cutoff = pd.Timestamp(unique_dates[train_end - 1])
    valid_cutoff = pd.Timestamp(unique_dates[valid_end - 1])
    all_dates = pd.to_datetime(anchor_dates)
    train_mask = all_dates <= train_cutoff
    valid_mask = (all_dates > train_cutoff) & (all_dates <= valid_cutoff)
    test_mask = all_dates > valid_cutoff
    return train_mask, valid_mask, test_mask


def scale_data(
    seq_train: np.ndarray,
    seq_valid: np.ndarray,
    seq_test: np.ndarray,
    static_train: np.ndarray,
    static_valid: np.ndarray,
    static_test: np.ndarray,
    target_train: np.ndarray,
    target_valid: np.ndarray,
    target_test: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, StandardScaler]]:
    seq_scaler = StandardScaler()
    seq_train_2d = seq_train.reshape(-1, seq_train.shape[-1])
    seq_scaler.fit(seq_train_2d)

    def transform_seq(x: np.ndarray) -> np.ndarray:
        return seq_scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)

    static_scaler = StandardScaler()
    static_scaler.fit(static_train)

    target_scaler = StandardScaler()
    target_scaler.fit(target_train)

    return (
        (transform_seq(seq_train), transform_seq(seq_valid), transform_seq(seq_test)),
        (
            static_scaler.transform(static_train).astype(np.float32),
            static_scaler.transform(static_valid).astype(np.float32),
            static_scaler.transform(static_test).astype(np.float32),
        ),
        (
            target_scaler.transform(target_train).astype(np.float32),
            target_scaler.transform(target_valid).astype(np.float32),
            target_scaler.transform(target_test).astype(np.float32),
        ),
        {
            "seq_scaler": seq_scaler,
            "static_scaler": static_scaler,
            "target_scaler": target_scaler,
        },
    )


def build_sequence_model(seq_shape: tuple[int, int], static_dim: int, output_dim: int) -> keras.Model:
    seq_input = keras.Input(shape=seq_shape, name="sequence_input")
    x = keras.layers.GRU(48, return_sequences=True)(seq_input)
    x = keras.layers.GRU(32)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(0.15)(x)

    static_input = keras.Input(shape=(static_dim,), name="static_input")
    s = keras.layers.Dense(24, activation="relu")(static_input)

    merged = keras.layers.Concatenate()([x, s])
    merged = keras.layers.Dense(48, activation="relu")(merged)
    merged = keras.layers.Dropout(0.10)(merged)
    output = keras.layers.Dense(output_dim, name="raw_targets")(merged)

    model = keras.Model(inputs=[seq_input, static_input], outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_targets(y_true: np.ndarray, y_pred: np.ndarray, target_cols: list[str], split: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for idx, target in enumerate(target_cols):
        rows.append(
            {
                "target": target,
                "split": split,
                "mae": float(mean_absolute_error(y_true[:, idx], y_pred[:, idx])),
                "rmse": rmse(y_true[:, idx], y_pred[:, idx]),
                "r2": float(r2_score(y_true[:, idx], y_pred[:, idx])),
            }
        )
    return pd.DataFrame(rows)


def train_horizon(
    seq: np.ndarray,
    static: np.ndarray,
    target: np.ndarray,
    anchor_dates: np.ndarray,
    target_cols: list[str],
    model_dir: Path,
    summary_path: Path,
    metrics_path: Path,
    horizon_label: str,
) -> None:
    train_mask, valid_mask, test_mask = split_by_time(anchor_dates)
    seq_train, seq_valid, seq_test = seq[train_mask], seq[valid_mask], seq[test_mask]
    static_train, static_valid, static_test = static[train_mask], static[valid_mask], static[test_mask]
    target_train, target_valid, target_test = target[train_mask], target[valid_mask], target[test_mask]

    (seq_scaled, static_scaled, target_scaled, scalers) = scale_data(
        seq_train, seq_valid, seq_test,
        static_train, static_valid, static_test,
        target_train, target_valid, target_test,
    )
    seq_train_s, seq_valid_s, seq_test_s = seq_scaled
    static_train_s, static_valid_s, static_test_s = static_scaled
    target_train_s, target_valid_s, target_test_s = target_scaled

    model_dir.mkdir(parents=True, exist_ok=True)
    model = build_sequence_model((seq.shape[1], seq.shape[2]), static.shape[1], target.shape[1])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ]
    history = model.fit(
        [seq_train_s, static_train_s],
        target_train_s,
        validation_data=([seq_valid_s, static_valid_s], target_valid_s),
        epochs=40,
        batch_size=64,
        verbose=0,
        callbacks=callbacks,
    )

    pred_valid = scalers["target_scaler"].inverse_transform(model.predict([seq_valid_s, static_valid_s], verbose=0))
    pred_test = scalers["target_scaler"].inverse_transform(model.predict([seq_test_s, static_test_s], verbose=0))

    valid_metrics = evaluate_targets(target_valid, pred_valid, target_cols, "valid")
    test_metrics = evaluate_targets(target_test, pred_test, target_cols, "test")
    metrics = pd.concat([valid_metrics, test_metrics], ignore_index=True)
    metrics.to_csv(metrics_path, index=False)

    model.save(model_dir / "sequence_model.keras")
    joblib.dump(scalers, model_dir / "scalers.joblib")

    test_view = metrics[metrics["split"] == "test"].copy()
    lines = [
        f"Adunbox Local Midnight Sequence {horizon_label.upper()} Summary",
        "",
        f"Hourly input: {HOURLY_INPUT_PATH}",
        f"Anchor UTC min: {pd.Timestamp(pd.to_datetime(anchor_dates).min())}",
        f"Anchor UTC max: {pd.Timestamp(pd.to_datetime(anchor_dates).max())}",
        f"Anchor count: {len(anchor_dates):,}",
        f"Sequence hours: {SEQ_HOURS}",
        f"Sequence feature count: {seq.shape[2]}",
        f"Static feature count: {static.shape[1]}",
        f"Train rows: {len(target_train):,}",
        f"Validation rows: {len(target_valid):,}",
        f"Test rows: {len(target_test):,}",
        f"Epochs used: {len(history.history.get('loss', []))}",
        "",
        "Time handling:",
        "- account-local midnight anchors only",
        f"- sequence uses the prior {SEQ_HOURS} local hourly rows before midnight",
        f"- {horizon_label} targets sum future local hourly rows after midnight",
        "",
        "Raw target test R2:",
    ]
    for _, row in test_view.iterrows():
        lines.append(f"- {row['target']}: {row['r2']:.6f}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    hourly = load_hourly()
    seq, static, target_6h, target_24h, anchor_dates = build_samples(hourly)
    if len(anchor_dates) == 0:
        raise RuntimeError("No valid midnight sequence samples were built.")

    train_horizon(
        seq,
        static,
        target_6h,
        anchor_dates,
        TARGET_COLS_6H,
        MODEL_DIR_6H,
        SUMMARY_6H_PATH,
        METRICS_6H_PATH,
        "6h",
    )
    train_horizon(
        seq,
        static,
        target_24h,
        anchor_dates,
        TARGET_COLS_24H,
        MODEL_DIR_24H,
        SUMMARY_24H_PATH,
        METRICS_24H_PATH,
        "24h",
    )

    print(MODEL_DIR_6H)
    print(MODEL_DIR_24H)


if __name__ == "__main__":
    main()
