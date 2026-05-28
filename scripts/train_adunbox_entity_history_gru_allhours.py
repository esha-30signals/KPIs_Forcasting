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

import train_adunbox_local_midnight_sequence_models as base_seq


BASE_DIR = Path(os.getenv("ADUNBOX_PROJECT_DIR", Path(__file__).resolve().parents[1]))
MODEL_DIR_6H = BASE_DIR / "models" / "adunbox_entity_history_gru_allhours_168h_6h"
MODEL_DIR_24H = BASE_DIR / "models" / "adunbox_entity_history_gru_allhours_168h_24h"
SUMMARY_6H_PATH = BASE_DIR / "adunbox_entity_history_gru_allhours_168h_6h__summary.txt"
SUMMARY_24H_PATH = BASE_DIR / "adunbox_entity_history_gru_allhours_168h_24h__summary.txt"
METRICS_6H_PATH = BASE_DIR / "adunbox_entity_history_gru_allhours_168h_6h__metrics.csv"
METRICS_24H_PATH = BASE_DIR / "adunbox_entity_history_gru_allhours_168h_24h__metrics.csv"

SEQ_HOURS_6H = 168
SEQ_HOURS_24H = 168
TARGET_HOURS_6H = 6
TARGET_HOURS_24H = 24
MIN_HISTORY_HOURS = 24

TARGET_COLS = [
    "target_spend",
    "target_impressions",
    "target_inline_link_clicks",
    "target_tracker_conversions",
    "target_tracker_revenue",
]

VALID_6H_ANCHOR_HOURS = {0, 6, 12, 18}
MIN_RECENT_24H_SPEND_6H = 1.0
MIN_RECENT_24H_IMPRESSIONS_6H = 50.0
MIN_RECENT_24H_CLICKS_6H = 1.0

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

STATIC_FEATURES_24H = [
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

METRIC_COLS = ["spend", "impressions", "inline_link_clicks", "tracker_conversions", "tracker_revenue"]


def window_sum(cumulative: np.ndarray, end_pos_exclusive: int, hours: int) -> np.ndarray:
    start = max(0, end_pos_exclusive - hours)
    return cumulative[end_pos_exclusive] - cumulative[start]


def future_sum(cumulative: np.ndarray, start_pos: int, hours: int) -> np.ndarray:
    return cumulative[start_pos + hours] - cumulative[start_pos]


def same_window_sum(cumulative: np.ndarray, start_pos: int, hours: int, offset_hours: int) -> np.ndarray:
    begin = start_pos - offset_hours
    end = begin + hours
    if begin < 0 or end < 0:
        return np.zeros(cumulative.shape[1], dtype=np.float32)
    return cumulative[end] - cumulative[begin]


def build_samples_for_horizon(
    df: pd.DataFrame,
    seq_hours: int,
    target_hours: int,
    horizon_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seq_samples: list[np.ndarray] = []
    static_samples: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    anchor_dates: list[pd.Timestamp] = []

    for _, group_idx in df.groupby("ad_id", sort=False).groups.items():
        group = df.loc[group_idx].sort_values("local_ts").reset_index(drop=True)
        if len(group) < seq_hours + target_hours + 1:
            continue

        metric_arr = group[METRIC_COLS].to_numpy(dtype=np.float32, copy=False)
        cumulative = np.vstack([np.zeros((1, metric_arr.shape[1]), dtype=np.float32), np.cumsum(metric_arr, axis=0)])

        for pos in range(seq_hours - 1, len(group) - target_hours):
            if horizon_label == "6h":
                anchor_hour = int(group.loc[pos, "local_ts"].hour)
                if anchor_hour not in VALID_6H_ANCHOR_HOURS:
                    continue
            history_start = pos - seq_hours + 1
            history_end = pos
            history_times = group.loc[history_start:history_end, "local_ts"]
            expected_hist = base_seq.expected_hour_range(history_times.iloc[-1], seq_hours)
            if not history_times.reset_index(drop=True).equals(pd.Series(expected_hist)):
                continue

            future_start = pos + 1
            future_end = future_start + target_hours
            future_times = group.loc[future_start:future_end - 1, "local_ts"]
            expected_future = pd.date_range(start=group.loc[future_start, "local_ts"], periods=target_hours, freq="1h")
            if not future_times.reset_index(drop=True).equals(pd.Series(expected_future)):
                continue
            seq_sample = group.loc[history_start:history_end, base_seq.SEQ_FEATURES].to_numpy(dtype=np.float32)

            if horizon_label == "6h":
                recent_1 = window_sum(cumulative, pos + 1, 1)
                recent_3 = window_sum(cumulative, pos + 1, 3) / 3.0
                recent_6 = window_sum(cumulative, pos + 1, 6)
                recent_24 = window_sum(cumulative, pos + 1, 24)
                if (
                    float(recent_24[0]) < MIN_RECENT_24H_SPEND_6H
                    or float(recent_24[1]) < MIN_RECENT_24H_IMPRESSIONS_6H
                    or float(recent_24[2]) < MIN_RECENT_24H_CLICKS_6H
                ):
                    continue
                same_1d = same_window_sum(cumulative, future_start, target_hours, 24)
                same_2d = same_window_sum(cumulative, future_start, target_hours, 48)
                same_7d = same_window_sum(cumulative, future_start, target_hours, 168)
                cum_total = cumulative[pos + 1]
                last_hour_roas = float(base_seq.safe_div(recent_1[4], recent_1[0]))
                last_3h_roas = float(base_seq.safe_div(recent_3[4], recent_3[0]))
                static_samples.append(
                    np.array(
                        [
                            recent_1[0], recent_1[4], last_hour_roas,
                            recent_3[0], recent_3[4], last_3h_roas,
                            recent_6[0], recent_6[4], recent_6[3],
                            recent_24[0], recent_24[4], recent_24[3],
                            same_1d[0], same_1d[4], same_1d[3],
                            same_2d[0], same_2d[4], same_2d[3],
                            same_7d[0], same_7d[4], same_7d[3],
                            cum_total[0], cum_total[4], float(pos + 1),
                        ],
                        dtype=np.float32,
                    )
                )
                seq_samples.append(seq_sample)
            else:
                recent_6 = window_sum(cumulative, pos + 1, 6)
                recent_24 = window_sum(cumulative, pos + 1, 24)
                recent_72 = window_sum(cumulative, pos + 1, 72)
                recent_168 = window_sum(cumulative, pos + 1, 168)
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
                seq_samples.append(seq_sample)

            targets.append(future_sum(cumulative, future_start, target_hours).astype(np.float32))
            anchor_dates.append(pd.Timestamp(group.loc[pos, "date"]))

    return (
        np.asarray(seq_samples, dtype=np.float32),
        np.asarray(static_samples, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(anchor_dates),
    )


def scale_data_log_targets(
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
    seq_scaler.fit(seq_train.reshape(-1, seq_train.shape[-1]))

    def transform_seq(x: np.ndarray) -> np.ndarray:
        return seq_scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)

    static_scaler = StandardScaler()
    static_scaler.fit(static_train)

    target_scaler = StandardScaler()
    target_train_log = np.log1p(np.maximum(target_train, 0.0)).astype(np.float32)
    target_valid_log = np.log1p(np.maximum(target_valid, 0.0)).astype(np.float32)
    target_test_log = np.log1p(np.maximum(target_test, 0.0)).astype(np.float32)
    target_scaler.fit(target_train_log)

    return (
        (transform_seq(seq_train), transform_seq(seq_valid), transform_seq(seq_test)),
        (
            static_scaler.transform(static_train).astype(np.float32),
            static_scaler.transform(static_valid).astype(np.float32),
            static_scaler.transform(static_test).astype(np.float32),
        ),
        (
            target_scaler.transform(target_train_log).astype(np.float32),
            target_scaler.transform(target_valid_log).astype(np.float32),
            target_scaler.transform(target_test_log).astype(np.float32),
        ),
        {
            "seq_scaler": seq_scaler,
            "static_scaler": static_scaler,
            "target_scaler": target_scaler,
        },
    )


def inverse_transform_targets_log(target_scaled: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    target_log = scaler.inverse_transform(target_scaled)
    target_log = np.clip(target_log, 0.0, 20.0)
    return np.expm1(target_log).astype(np.float32)


def build_gru_model(seq_shape: tuple[int, int], static_dim: int, output_dim: int, horizon_label: str) -> keras.Model:
    seq_input = keras.Input(shape=seq_shape, name="sequence_input")
    static_input = keras.Input(shape=(static_dim,), name="static_input")

    if horizon_label == "6h":
        x = keras.layers.GRU(32, return_sequences=True)(seq_input)
        x = keras.layers.Dropout(0.30)(x)
        x = keras.layers.GRU(16)(x)
        x = keras.layers.Dropout(0.20)(x)
        x = keras.layers.Dense(16, activation="relu")(x)
        s = keras.layers.Dense(8, activation="relu")(static_input)
        learning_rate = 3e-4
        delta = 0.5
    else:
        x = keras.layers.GRU(64, return_sequences=True)(seq_input)
        x = keras.layers.Dropout(0.30)(x)
        x = keras.layers.GRU(32)(x)
        x = keras.layers.Dropout(0.20)(x)
        x = keras.layers.Dense(16, activation="relu")(x)
        s = keras.layers.Dense(8, activation="relu")(static_input)
        learning_rate = 1e-3
        delta = 1.0

    merged = keras.layers.Concatenate()([x, s])
    output = keras.layers.Dense(output_dim, name="raw_targets")(merged)
    model = keras.Model(inputs=[seq_input, static_input], outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=keras.losses.Huber(delta=delta),
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
    seq_hours: int,
    target_hours: int,
) -> None:
    train_mask, valid_mask, test_mask = base_seq.split_by_time(anchor_dates)
    seq_train, seq_valid, seq_test = seq[train_mask], seq[valid_mask], seq[test_mask]
    static_train, static_valid, static_test = static[train_mask], static[valid_mask], static[test_mask]
    target_train, target_valid, target_test = target[train_mask], target[valid_mask], target[test_mask]

    (seq_scaled, static_scaled, target_scaled, scalers) = scale_data_log_targets(
        seq_train, seq_valid, seq_test,
        static_train, static_valid, static_test,
        target_train, target_valid, target_test,
    )
    seq_train_s, seq_valid_s, seq_test_s = seq_scaled
    static_train_s, static_valid_s, static_test_s = static_scaled
    target_train_s, target_valid_s, _ = target_scaled

    model_dir.mkdir(parents=True, exist_ok=True)
    model = build_gru_model((seq.shape[1], seq.shape[2]), static.shape[1], target.shape[1], horizon_label)
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, min_lr=1e-5, verbose=0),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    ]
    history = model.fit(
        [seq_train_s, static_train_s],
        target_train_s,
        validation_data=([seq_valid_s, static_valid_s], target_valid_s),
        epochs=200,
        batch_size=128 if horizon_label == "6h" else 64,
        verbose=0,
        callbacks=callbacks,
    )

    pred_valid = inverse_transform_targets_log(model.predict([seq_valid_s, static_valid_s], verbose=0), scalers["target_scaler"])
    pred_test = inverse_transform_targets_log(model.predict([seq_test_s, static_test_s], verbose=0), scalers["target_scaler"])

    valid_metrics = evaluate_targets(target_valid, pred_valid, target_cols, "valid")
    test_metrics = evaluate_targets(target_test, pred_test, target_cols, "test")
    metrics = pd.concat([valid_metrics, test_metrics], ignore_index=True)
    metrics.to_csv(metrics_path, index=False)

    model.save(model_dir / "sequence_model.keras")
    joblib.dump(
        {
            **scalers,
            "seq_hours": seq_hours,
            "target_hours": target_hours,
            "static_feature_names": STATIC_FEATURES_6H if horizon_label == "6h" else STATIC_FEATURES_24H,
            "sequence_feature_names": base_seq.SEQ_FEATURES,
            "training_basis": "entity_history_allhours",
        },
        model_dir / "scalers.joblib",
    )

    test_view = metrics[metrics["split"] == "test"].copy()
    lines = [
        f"Adunbox Entity History GRU {horizon_label.upper()} Summary",
        "",
        f"Hourly input: {base_seq.HOURLY_INPUT_PATH}",
        f"Anchor UTC min: {pd.Timestamp(pd.to_datetime(anchor_dates).min())}",
        f"Anchor UTC max: {pd.Timestamp(pd.to_datetime(anchor_dates).max())}",
        f"Anchor count: {len(anchor_dates):,}",
        f"Sequence hours: {seq_hours}",
        f"Sequence feature count: {seq.shape[2]}",
        f"Static feature count: {static.shape[1]}",
        f"Train rows: {len(target_train):,}",
        f"Validation rows: {len(target_valid):,}",
        f"Test rows: {len(target_test):,}",
        f"Epochs used: {len(history.history.get('loss', []))}",
        "",
        "Time handling:",
        "- all valid hourly anchors per ad",
        "- sequence uses the latest contiguous local hourly history ending at the current row",
        f"- {horizon_label} targets sum the future local hourly rows after the current row",
        "- 6h stabilization: anchor hours limited to 00/06/12/18 and low-signal recent windows filtered",
        "",
        "Raw target test R2:",
    ]
    for _, row in test_view.iterrows():
        lines.append(f"- {row['target']}: {row['r2']:.6f}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    hourly = base_seq.load_hourly()
    seq_6h, static_6h, target_6h, anchor_dates_6h = build_samples_for_horizon(hourly, SEQ_HOURS_6H, TARGET_HOURS_6H, "6h")
    seq_24h, static_24h, target_24h, anchor_dates_24h = build_samples_for_horizon(hourly, SEQ_HOURS_24H, TARGET_HOURS_24H, "24h")

    if len(seq_6h) == 0 or len(seq_24h) == 0:
        raise RuntimeError("No valid all-hour entity-history samples were built.")

    train_horizon(
        seq_6h,
        static_6h,
        target_6h,
        anchor_dates_6h,
        TARGET_COLS,
        MODEL_DIR_6H,
        SUMMARY_6H_PATH,
        METRICS_6H_PATH,
        "6h",
        SEQ_HOURS_6H,
        TARGET_HOURS_6H,
    )
    train_horizon(
        seq_24h,
        static_24h,
        target_24h,
        anchor_dates_24h,
        TARGET_COLS,
        MODEL_DIR_24H,
        SUMMARY_24H_PATH,
        METRICS_24H_PATH,
        "24h",
        SEQ_HOURS_24H,
        TARGET_HOURS_24H,
    )


if __name__ == "__main__":
    main()
