from __future__ import annotations

import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

import train_adunbox_entity_history_gru_allhours as base_gru
import train_adunbox_local_midnight_sequence_models as base_seq


BASE_DIR = Path(os.getenv("ADUNBOX_PROJECT_DIR", Path(__file__).resolve().parents[1]))
MODEL_DIR = BASE_DIR / "models" / "adunbox_entity_history_gru_168h_padded_6h"
METRICS_PATH = BASE_DIR / "adunbox_entity_history_gru_168h_padded_6h__metrics.csv"
SUMMARY_PATH = BASE_DIR / "adunbox_entity_history_gru_168h_padded_6h__summary.txt"

SEQ_HOURS = 168
TARGET_HOURS = 6
MIN_OBSERVED_HISTORY_HOURS = 24
VALID_ANCHOR_HOURS = {0, 6, 12, 18}
MAX_SAMPLES = 30_000
TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TRAIN_EPOCHS = 80
TRAIN_BATCH_SIZE = 256


def dense_ad_group(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("local_ts").drop_duplicates("local_ts", keep="last")
    idx = pd.date_range(group["local_ts"].min(), group["local_ts"].max(), freq="1h")
    dense = group.set_index("local_ts").reindex(idx)

    # Padded rows must be numerically complete for every sequence feature.
    for col in base_seq.SEQ_FEATURES:
        if col in dense.columns:
            dense[col] = dense[col].fillna(0.0)

    for col in base_gru.METRIC_COLS:
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
    recent_1 = base_gru.window_sum(cumulative, pos + 1, 1)
    recent_3 = base_gru.window_sum(cumulative, pos + 1, 3) / 3.0
    recent_6 = base_gru.window_sum(cumulative, pos + 1, 6)
    recent_24 = base_gru.window_sum(cumulative, pos + 1, 24)
    same_1d = base_gru.same_window_sum(cumulative, pos + 1, TARGET_HOURS, 24)
    same_2d = base_gru.same_window_sum(cumulative, pos + 1, TARGET_HOURS, 48)
    same_7d = base_gru.same_window_sum(cumulative, pos + 1, TARGET_HOURS, 168)
    cum_total = cumulative[pos + 1]
    return np.array(
        [
            recent_1[0], recent_1[4], float(base_seq.safe_div(recent_1[4], recent_1[0])),
            recent_3[0], recent_3[4], float(base_seq.safe_div(recent_3[4], recent_3[0])),
            recent_6[0], recent_6[4], recent_6[3],
            recent_24[0], recent_24[4], recent_24[3],
            same_1d[0], same_1d[4], same_1d[3],
            same_2d[0], same_2d[4], same_2d[3],
            same_7d[0], same_7d[4], same_7d[3],
            cum_total[0], cum_total[4], float(pos + 1),
        ],
        dtype=np.float32,
    )


def build_samples(hourly: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seqs: list[np.ndarray] = []
    statics: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    anchor_dates: list[pd.Timestamp] = []

    for _, group_idx in hourly.groupby("ad_id", sort=False).groups.items():
        dense = dense_ad_group(hourly.loc[group_idx])
        if len(dense) < SEQ_HOURS + TARGET_HOURS:
            continue
        metric_arr = dense[base_gru.METRIC_COLS].to_numpy(dtype=np.float32)
        observed = dense["observed_hour"].to_numpy(dtype=np.int8)
        cumulative = np.vstack([np.zeros((1, len(base_gru.METRIC_COLS)), dtype=np.float32), np.cumsum(metric_arr, axis=0)])
        obs_cum = np.concatenate([[0], np.cumsum(observed)])

        for pos in range(SEQ_HOURS - 1, len(dense) - TARGET_HOURS):
            anchor_ts = pd.Timestamp(dense.loc[pos, "local_ts"])
            if anchor_ts.hour not in VALID_ANCHOR_HOURS:
                continue
            observed_history = int(obs_cum[pos + 1] - obs_cum[pos + 1 - SEQ_HOURS])
            if observed_history < MIN_OBSERVED_HISTORY_HOURS:
                continue

            future_start = pos + 1
            target = base_gru.future_sum(cumulative, future_start, TARGET_HOURS).astype(np.float32)
            if float(target[0] + target[1] + target[2] + target[3] + target[4]) <= 0:
                continue
            seq_sample = dense.loc[pos - SEQ_HOURS + 1:pos, base_seq.SEQ_FEATURES].to_numpy(dtype=np.float32)
            if np.isnan(seq_sample).any():
                continue
            statics.append(build_static(cumulative, pos))
            seqs.append(seq_sample)
            targets.append(target)
            anchor_dates.append(anchor_ts)
            if len(seqs) >= MAX_SAMPLES:
                return (
                    np.asarray(seqs, dtype=np.float32),
                    np.asarray(statics, dtype=np.float32),
                    np.asarray(targets, dtype=np.float32),
                    np.asarray(anchor_dates),
                )

    return (
        np.asarray(seqs, dtype=np.float32),
        np.asarray(statics, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(anchor_dates),
    )


def split_by_anchor_time(anchor_dates: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor_series = pd.Series(pd.to_datetime(anchor_dates)).sort_values().reset_index(drop=True)
    unique_times = np.array(anchor_series.unique())
    train_end = max(1, int(len(unique_times) * TRAIN_RATIO))
    valid_end = max(train_end + 1, int(len(unique_times) * (TRAIN_RATIO + VALID_RATIO)))
    valid_end = min(valid_end, len(unique_times) - 1)

    train_cutoff = pd.Timestamp(unique_times[train_end - 1])
    valid_cutoff = pd.Timestamp(unique_times[valid_end - 1])
    all_times = pd.to_datetime(anchor_dates)
    train_mask = all_times <= train_cutoff
    valid_mask = (all_times > train_cutoff) & (all_times <= valid_cutoff)
    test_mask = all_times > valid_cutoff
    return train_mask, valid_mask, test_mask


def train() -> None:
    hourly = base_seq.load_hourly()
    seq, static, target, anchor_dates = build_samples(hourly)
    if len(seq) == 0:
        raise RuntimeError("No padded 168h samples were built.")

    if np.isnan(seq).any() or np.isnan(static).any() or np.isnan(target).any():
        raise RuntimeError("Padded 6h samples still contain NaNs after preprocessing.")

    train_mask, valid_mask, test_mask = split_by_anchor_time(anchor_dates)
    seq_train, seq_valid, seq_test = seq[train_mask], seq[valid_mask], seq[test_mask]
    static_train, static_valid, static_test = static[train_mask], static[valid_mask], static[test_mask]
    target_train, target_valid, target_test = target[train_mask], target[valid_mask], target[test_mask]

    seq_scaler = StandardScaler()
    seq_scaler.fit(seq_train.reshape(-1, seq_train.shape[-1]))
    static_scaler = StandardScaler()
    static_scaler.fit(static_train)
    target_scaler = StandardScaler()
    target_train_log = np.log1p(np.maximum(target_train, 0.0)).astype(np.float32)
    target_valid_log = np.log1p(np.maximum(target_valid, 0.0)).astype(np.float32)
    target_scaler.fit(target_train_log)

    def scale_seq(x: np.ndarray) -> np.ndarray:
        return seq_scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)

    seq_train_s, seq_valid_s, seq_test_s = scale_seq(seq_train), scale_seq(seq_valid), scale_seq(seq_test)
    static_train_s = static_scaler.transform(static_train).astype(np.float32)
    static_valid_s = static_scaler.transform(static_valid).astype(np.float32)
    static_test_s = static_scaler.transform(static_test).astype(np.float32)
    target_train_s = target_scaler.transform(target_train_log).astype(np.float32)
    target_valid_s = target_scaler.transform(target_valid_log).astype(np.float32)

    model = base_gru.build_gru_model((SEQ_HOURS, seq.shape[2]), static.shape[1], target.shape[1], "6h")
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-5),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ]
    history = model.fit(
        [seq_train_s, static_train_s],
        target_train_s,
        validation_data=([seq_valid_s, static_valid_s], target_valid_s),
        epochs=TRAIN_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        verbose=0,
        callbacks=callbacks,
    )

    def inverse(pred_scaled: np.ndarray) -> np.ndarray:
        pred_log = np.clip(target_scaler.inverse_transform(pred_scaled), 0.0, 20.0)
        return np.expm1(pred_log).astype(np.float32)

    pred_valid = inverse(model.predict([seq_valid_s, static_valid_s], verbose=0))
    pred_test = inverse(model.predict([seq_test_s, static_test_s], verbose=0))

    rows: list[dict[str, object]] = []
    for split, y_true, y_pred in [("valid", target_valid, pred_valid), ("test", target_test, pred_test)]:
        for idx, target_name in enumerate(base_gru.TARGET_COLS):
            rows.append(
                {
                    "target": target_name,
                    "split": split,
                    "mae": float(mean_absolute_error(y_true[:, idx], y_pred[:, idx])),
                    "rmse": float(np.sqrt(mean_squared_error(y_true[:, idx], y_pred[:, idx]))),
                    "r2": float(r2_score(y_true[:, idx], y_pred[:, idx])),
                }
            )
    metrics = pd.DataFrame(rows)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)
    model.save(MODEL_DIR / "sequence_model.keras")
    joblib.dump(
        {
            "seq_scaler": seq_scaler,
            "static_scaler": static_scaler,
            "target_scaler": target_scaler,
            "seq_hours": SEQ_HOURS,
            "target_hours": TARGET_HOURS,
            "static_feature_names": base_gru.STATIC_FEATURES_6H,
            "sequence_feature_names": base_seq.SEQ_FEATURES,
            "training_basis": "entity_history_168h_padded",
            "min_observed_history_hours": MIN_OBSERVED_HISTORY_HOURS,
        },
        MODEL_DIR / "scalers.joblib",
    )

    test_view = metrics[metrics["split"] == "test"]
    lines = [
        "Adunbox Entity History GRU 168h Padded 6H Summary",
        "",
        f"Hourly input: {base_seq.HOURLY_INPUT_PATH}",
        f"Anchor count: {len(anchor_dates):,}",
        f"Sequence hours: {SEQ_HOURS}",
        f"Sequence feature count: {seq.shape[2]}",
        f"Static feature count: {static.shape[1]}",
        f"Train rows: {len(target_train):,}",
        f"Validation rows: {len(target_valid):,}",
        f"Test rows: {len(target_test):,}",
        f"Epochs used: {len(history.history.get('loss', []))}",
        f"Batch size: {TRAIN_BATCH_SIZE}",
        "",
        "Training basis:",
        "- last 168 local hourly rows are fed directly into GRU",
        "- missing hourly gaps are padded with zeros",
        "- samples require at least 24 observed hours inside the 168h window",
        f"- valid forecast anchors only: {sorted(VALID_ANCHOR_HOURS)} local hours",
        "- low-activity recent windows are retained instead of filtered out",
        "- target is next 6 local hourly rows",
        "",
        "Raw target test R2:",
    ]
    for row in test_view.itertuples(index=False):
        lines.append(f"- {row.target}: {float(row.r2):.6f}")
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(SUMMARY_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    train()
