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
import train_adunbox_entity_history_gru_168h_padded_6h as padded_6h
import train_adunbox_local_midnight_sequence_models as base_seq


BASE_DIR = Path(os.getenv("ADUNBOX_PROJECT_DIR", Path(__file__).resolve().parents[1]))
MODEL_DIR = BASE_DIR / "models" / "adunbox_entity_history_gru_168h_padded_6h_hybrid"
METRICS_PATH = BASE_DIR / "adunbox_entity_history_gru_168h_padded_6h_hybrid__metrics.csv"
SUMMARY_PATH = BASE_DIR / "adunbox_entity_history_gru_168h_padded_6h_hybrid__summary.txt"
BASELINE_MODEL_DIR = BASE_DIR / "models" / "adunbox_entity_history_gru_168h_padded_6h"

VOLUME_TARGET_COLS = [
    "target_spend",
    "target_impressions",
    "target_inline_link_clicks",
]
RATIO_TARGET_COLS = [
    "target_cvr",
    "target_roas",
]
RAW_TARGET_COLS = list(base_gru.TARGET_COLS)
TRAIN_EPOCHS = 200
TRAIN_BATCH_SIZE = 64


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_div_np(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numer, dtype=np.float32)
    mask = denom > 0
    out[mask] = numer[mask] / denom[mask]
    return out


def build_component_targets(raw_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spend = raw_target[:, 0]
    impressions = raw_target[:, 1]
    clicks = raw_target[:, 2]
    conversions = raw_target[:, 3]
    revenue = raw_target[:, 4]

    volume_target = np.stack([spend, impressions, clicks], axis=1).astype(np.float32)
    cvr = safe_div_np(conversions, clicks)
    roas = safe_div_np(revenue, spend)
    ratio_target = np.stack([cvr, roas], axis=1).astype(np.float32)
    return volume_target, ratio_target


def scale_inputs(
    seq_train: np.ndarray,
    seq_valid: np.ndarray,
    seq_test: np.ndarray,
    static_train: np.ndarray,
    static_valid: np.ndarray,
    static_test: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, StandardScaler]]:
    seq_scaler = StandardScaler()
    seq_scaler.fit(seq_train.reshape(-1, seq_train.shape[-1]))

    def scale_seq(x: np.ndarray) -> np.ndarray:
        return seq_scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)

    static_scaler = StandardScaler()
    static_scaler.fit(static_train)
    return (
        (scale_seq(seq_train), scale_seq(seq_valid), scale_seq(seq_test)),
        (
            static_scaler.transform(static_train).astype(np.float32),
            static_scaler.transform(static_valid).astype(np.float32),
            static_scaler.transform(static_test).astype(np.float32),
        ),
        {"seq_scaler": seq_scaler, "static_scaler": static_scaler},
    )


def fit_log_target_scaler(target_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(np.log1p(np.maximum(target_train, 0.0)).astype(np.float32))
    return scaler


def transform_log_target(target: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    target_log = np.log1p(np.maximum(target, 0.0)).astype(np.float32)
    return scaler.transform(target_log).astype(np.float32)


def inverse_log_target(pred_scaled: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    pred_log = scaler.inverse_transform(pred_scaled)
    pred_log = np.clip(pred_log, 0.0, 20.0)
    return np.expm1(pred_log).astype(np.float32)


def evaluate_targets(y_true: np.ndarray, y_pred: np.ndarray, target_cols: list[str], split: str, family: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, target_name in enumerate(target_cols):
        rows.append(
            {
                "family": family,
                "target": target_name,
                "split": split,
                "mae": float(mean_absolute_error(y_true[:, idx], y_pred[:, idx])),
                "rmse": rmse(y_true[:, idx], y_pred[:, idx]),
                "r2": float(r2_score(y_true[:, idx], y_pred[:, idx])),
            }
        )
    return pd.DataFrame(rows)


def reconstruct_raw_predictions(volume_pred: np.ndarray, ratio_pred: np.ndarray) -> np.ndarray:
    spend = np.maximum(volume_pred[:, 0], 0.0)
    impressions = np.maximum(volume_pred[:, 1], 0.0)
    clicks = np.maximum(volume_pred[:, 2], 0.0)
    cvr = np.maximum(ratio_pred[:, 0], 0.0)
    roas = np.maximum(ratio_pred[:, 1], 0.0)

    conversions = clicks * cvr
    revenue = spend * roas
    return np.stack([spend, impressions, clicks, conversions, revenue], axis=1).astype(np.float32)


def load_baseline_volume_predictions(
    seq_valid: np.ndarray,
    seq_test: np.ndarray,
    static_valid: np.ndarray,
    static_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaler_path = BASELINE_MODEL_DIR / "scalers.joblib"
    model_path = BASELINE_MODEL_DIR / "sequence_model.keras"
    if not scaler_path.exists() or not model_path.exists():
        raise RuntimeError("Baseline padded 6h GRU model is missing. Run train_adunbox_entity_history_gru_168h_padded_6h.py first.")

    artifacts = joblib.load(scaler_path)
    baseline_model = keras.models.load_model(model_path)

    def scale_seq(x: np.ndarray) -> np.ndarray:
        scaler = artifacts["seq_scaler"]
        return scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)

    seq_valid_s = scale_seq(seq_valid)
    seq_test_s = scale_seq(seq_test)
    static_valid_s = artifacts["static_scaler"].transform(static_valid).astype(np.float32)
    static_test_s = artifacts["static_scaler"].transform(static_test).astype(np.float32)

    def inverse(pred_scaled: np.ndarray) -> np.ndarray:
        pred_log = artifacts["target_scaler"].inverse_transform(pred_scaled)
        pred_log = np.clip(pred_log, 0.0, 20.0)
        return np.expm1(pred_log).astype(np.float32)

    valid_pred_full = inverse(baseline_model.predict([seq_valid_s, static_valid_s], verbose=0))
    test_pred_full = inverse(baseline_model.predict([seq_test_s, static_test_s], verbose=0))
    return valid_pred_full[:, :3], test_pred_full[:, :3]


def train() -> None:
    hourly = base_seq.load_hourly()
    seq, static, raw_target, anchor_dates = padded_6h.build_samples(hourly)
    if len(seq) == 0:
        raise RuntimeError("No final-rule hybrid 6h samples were built.")

    train_mask, valid_mask, test_mask = padded_6h.split_by_anchor_time(anchor_dates)
    seq_train, seq_valid, seq_test = seq[train_mask], seq[valid_mask], seq[test_mask]
    static_train, static_valid, static_test = static[train_mask], static[valid_mask], static[test_mask]
    raw_train, raw_valid, raw_test = raw_target[train_mask], raw_target[valid_mask], raw_target[test_mask]

    volume_train, ratio_train = build_component_targets(raw_train)
    volume_valid, ratio_valid = build_component_targets(raw_valid)
    volume_test, ratio_test = build_component_targets(raw_test)

    seq_scaled, static_scaled, input_scalers = scale_inputs(
        seq_train,
        seq_valid,
        seq_test,
        static_train,
        static_valid,
        static_test,
    )
    seq_train_s, seq_valid_s, seq_test_s = seq_scaled
    static_train_s, static_valid_s, static_test_s = static_scaled

    ratio_scaler = fit_log_target_scaler(ratio_train)
    ratio_train_s = transform_log_target(ratio_train, ratio_scaler)
    ratio_valid_s = transform_log_target(ratio_valid, ratio_scaler)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ratio_model = base_gru.build_gru_model((padded_6h.SEQ_HOURS, seq.shape[2]), static.shape[1], ratio_train.shape[1], "6h")
    ratio_history = ratio_model.fit(
        [seq_train_s, static_train_s],
        ratio_train_s,
        validation_data=([seq_valid_s, static_valid_s], ratio_valid_s),
        epochs=TRAIN_EPOCHS,
        batch_size=TRAIN_BATCH_SIZE,
        verbose=0,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, min_lr=1e-5),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ],
    )

    volume_pred_valid, volume_pred_test = load_baseline_volume_predictions(seq_valid, seq_test, static_valid, static_test)
    ratio_pred_valid = inverse_log_target(ratio_model.predict([seq_valid_s, static_valid_s], verbose=0), ratio_scaler)
    ratio_pred_test = inverse_log_target(ratio_model.predict([seq_test_s, static_test_s], verbose=0), ratio_scaler)

    raw_pred_valid = reconstruct_raw_predictions(volume_pred_valid, ratio_pred_valid)
    raw_pred_test = reconstruct_raw_predictions(volume_pred_test, ratio_pred_test)

    metrics = pd.concat(
        [
            evaluate_targets(volume_valid, volume_pred_valid, VOLUME_TARGET_COLS, "valid", "volume"),
            evaluate_targets(volume_test, volume_pred_test, VOLUME_TARGET_COLS, "test", "volume"),
            evaluate_targets(ratio_valid, ratio_pred_valid, RATIO_TARGET_COLS, "valid", "ratio"),
            evaluate_targets(ratio_test, ratio_pred_test, RATIO_TARGET_COLS, "test", "ratio"),
            evaluate_targets(raw_valid, raw_pred_valid, RAW_TARGET_COLS, "valid", "reconstructed_raw"),
            evaluate_targets(raw_test, raw_pred_test, RAW_TARGET_COLS, "test", "reconstructed_raw"),
        ],
        ignore_index=True,
    )
    metrics.to_csv(METRICS_PATH, index=False)

    ratio_model.save(MODEL_DIR / "ratio_model.keras")
    joblib.dump(
        {
            **input_scalers,
            "ratio_target_scaler": ratio_scaler,
            "seq_hours": padded_6h.SEQ_HOURS,
            "target_hours": padded_6h.TARGET_HOURS,
            "sequence_feature_names": base_seq.SEQ_FEATURES,
            "static_feature_names": base_gru.STATIC_FEATURES_6H,
            "training_basis": "entity_history_168h_padded_6h_final_hybrid",
            "volume_source_model_dir": str(BASELINE_MODEL_DIR),
            "volume_target_cols": VOLUME_TARGET_COLS,
            "ratio_target_cols": RATIO_TARGET_COLS,
            "raw_reconstruction": "conversions=clicks*cvr; revenue=spend*roas",
        },
        MODEL_DIR / "scalers.joblib",
    )

    raw_test_view = metrics[(metrics["family"] == "reconstructed_raw") & (metrics["split"] == "test")].copy()
    ratio_test_view = metrics[(metrics["family"] == "ratio") & (metrics["split"] == "test")].copy()
    lines = [
        "Adunbox Entity History GRU 168h Padded 6H Hybrid Summary",
        "",
        f"Hourly input: {base_seq.HOURLY_INPUT_PATH}",
        f"Anchor count: {len(anchor_dates):,}",
        f"Sequence hours: {padded_6h.SEQ_HOURS}",
        f"Sequence feature count: {seq.shape[2]}",
        f"Static feature count: {static.shape[1]}",
        f"Train rows: {len(raw_train):,}",
        f"Validation rows: {len(raw_valid):,}",
        f"Test rows: {len(raw_test):,}",
        f"Volume source model: {BASELINE_MODEL_DIR}",
        f"Ratio epochs used: {len(ratio_history.history.get('loss', []))}",
        "",
        "Training basis:",
        "- padded 6h baseline GRU provides spend, impressions, clicks",
        "- hybrid GRU predicts CVR and ROAS only",
        "- conversions reconstructed as clicks x CVR",
        "- revenue reconstructed as spend x ROAS",
        "- sample eligibility comes from train_adunbox_entity_history_gru_168h_padded_6h.py",
        "",
        "Reconstructed raw target test R2:",
    ]
    for _, row in raw_test_view.iterrows():
        lines.append(f"- {row['target']}: {row['r2']:.6f}")
    lines.append("")
    lines.append("Ratio target test R2:")
    for _, row in ratio_test_view.iterrows():
        lines.append(f"- {row['target']}: {row['r2']:.6f}")
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(SUMMARY_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    train()
