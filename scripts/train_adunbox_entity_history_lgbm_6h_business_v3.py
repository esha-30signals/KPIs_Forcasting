from __future__ import annotations

import os
import sys
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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_adunbox_entity_history_lgbm_6h_anchor_v2 as anchor_v2


BASE_DIR = Path(os.getenv("ADUNBOX_PROJECT_DIR", Path(__file__).resolve().parents[1]))
MODEL_DIR = BASE_DIR / "models" / "adunbox_entity_history_lgbm_6h_business_v3"
METRICS_PATH = BASE_DIR / "docs" / "adunbox_entity_history_lgbm_6h_business_v3__metrics.csv"
SUMMARY_PATH = BASE_DIR / "docs" / "adunbox_entity_history_lgbm_6h_business_v3__summary.txt"
CACHE_PATH = Path(os.getenv("ADUNBOX_6H_ANCHOR_FEATURE_CACHE", r"F:\adunbox_6h_anchor_v2_features_cache.pkl"))

BUSINESS_TARGETS = ["target_tracker_conversions", "target_tracker_revenue"]


def load_or_build_feature_cache() -> tuple[pd.DataFrame, pd.DataFrame, list[str], np.ndarray, np.ndarray, np.ndarray]:
    if CACHE_PATH.exists():
        print(f"loading feature cache: {CACHE_PATH}", flush=True)
        cached = joblib.load(CACHE_PATH)
        return (
            cached["features"],
            cached["targets"],
            cached["feature_cols"],
            cached["train_mask"],
            cached["valid_mask"],
            cached["test_mask"],
        )

    hourly = anchor_v2.load_hourly_aggregated()
    features, targets = anchor_v2.build_samples(hourly)
    if features.empty:
        raise RuntimeError("No 6h business-v3 samples were built.")
    train_mask, valid_mask, test_mask = anchor_v2.split_by_time(features)
    drop_cols = ["anchor_ts", *anchor_v2.ENTITY_COLS]
    feature_cols = [col for col in features.columns if col not in drop_cols]
    cached = {
        "features": features,
        "targets": targets,
        "feature_cols": feature_cols,
        "train_mask": train_mask,
        "valid_mask": valid_mask,
        "test_mask": test_mask,
        "hourly_input": str(anchor_v2.HOURLY_INPUT_PATH),
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cached, CACHE_PATH, compress=3)
    print(f"saved feature cache: {CACHE_PATH}", flush=True)
    return features, targets, feature_cols, train_mask, valid_mask, test_mask


def model_candidates(target: str) -> list[dict[str, object]]:
    if target == "target_tracker_conversions":
        return [
            {
                "name": "log_l1_balanced",
                "objective": "regression_l1",
                "transform": "log1p",
                "n_estimators": 1000,
                "learning_rate": 0.025,
                "num_leaves": 31,
                "min_child_samples": 70,
                "reg_alpha": 0.10,
                "reg_lambda": 0.45,
                "subsample": 0.90,
                "colsample_bytree": 0.86,
            },
            {
                "name": "log_huber_robust",
                "objective": "huber",
                "transform": "log1p",
                "alpha": 0.85,
                "n_estimators": 1100,
                "learning_rate": 0.022,
                "num_leaves": 31,
                "min_child_samples": 85,
                "reg_alpha": 0.12,
                "reg_lambda": 0.60,
                "subsample": 0.88,
                "colsample_bytree": 0.84,
            },
            {
                "name": "poisson_raw_count",
                "objective": "poisson",
                "transform": "raw",
                "n_estimators": 900,
                "learning_rate": 0.030,
                "num_leaves": 31,
                "min_child_samples": 80,
                "reg_alpha": 0.08,
                "reg_lambda": 0.50,
                "subsample": 0.90,
                "colsample_bytree": 0.86,
            },
        ]
    return [
        {
            "name": "log_l1_revenue",
            "objective": "regression_l1",
            "transform": "log1p",
            "n_estimators": 1100,
            "learning_rate": 0.022,
            "num_leaves": 31,
            "min_child_samples": 90,
            "reg_alpha": 0.18,
            "reg_lambda": 0.75,
            "subsample": 0.88,
            "colsample_bytree": 0.84,
        },
        {
            "name": "log_huber_revenue",
            "objective": "huber",
            "transform": "log1p",
            "alpha": 0.85,
            "n_estimators": 1200,
            "learning_rate": 0.020,
            "num_leaves": 31,
            "min_child_samples": 100,
            "reg_alpha": 0.20,
            "reg_lambda": 0.85,
            "subsample": 0.86,
            "colsample_bytree": 0.82,
        },
        {
            "name": "tweedie_raw_revenue",
            "objective": "tweedie",
            "transform": "raw",
            "tweedie_variance_power": 1.3,
            "n_estimators": 1000,
            "learning_rate": 0.025,
            "num_leaves": 31,
            "min_child_samples": 90,
            "reg_alpha": 0.16,
            "reg_lambda": 0.75,
            "subsample": 0.88,
            "colsample_bytree": 0.84,
        },
    ]


def make_model(params: dict[str, object]) -> LGBMRegressor:
    model_params = {k: v for k, v in params.items() if k not in {"name", "transform"}}
    return LGBMRegressor(
        random_state=42,
        n_jobs=1,
        verbosity=-1,
        subsample_freq=1,
        **model_params,
    )


def transform_y(y: np.ndarray, transform: str) -> np.ndarray:
    y = np.maximum(y.astype(np.float32), 0.0)
    if transform == "log1p":
        return np.log1p(y)
    return y


def inverse_pred(pred: np.ndarray, transform: str) -> np.ndarray:
    if transform == "log1p":
        pred = np.expm1(pred)
    return np.maximum(pred, 0.0)


def evaluate(target: str, candidate: str, split: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    denom = float(np.sum(np.abs(y_true)))
    return {
        "target": target,
        "candidate": candidate,
        "split": split,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "wmape": float(np.sum(np.abs(y_pred - y_true)) / denom) if denom else 0.0,
        "bias": float(np.sum(y_pred - y_true) / denom) if denom else 0.0,
    }


def main() -> None:
    features, targets, feature_cols, train_mask, valid_mask, test_mask = load_or_build_feature_cache()
    x = features[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float32")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []

    for target in BUSINESS_TARGETS:
        y_train_raw = targets.loc[train_mask, target].to_numpy(dtype=np.float32)
        y_valid_raw = targets.loc[valid_mask, target].to_numpy(dtype=np.float32)
        y_test_raw = targets.loc[test_mask, target].to_numpy(dtype=np.float32)
        best: tuple[float, str, LGBMRegressor, dict[str, object], np.ndarray] | None = None

        for params in model_candidates(target):
            name = str(params["name"])
            transform = str(params["transform"])
            print(f"training target={target}; candidate={name}; rows={len(x):,}; features={len(feature_cols):,}", flush=True)
            model = make_model(params)
            y_train = transform_y(y_train_raw, transform)
            y_valid = transform_y(y_valid_raw, transform)
            model.fit(
                x.loc[train_mask],
                y_train,
                eval_set=[(x.loc[valid_mask], y_valid)],
                eval_metric="l1",
                callbacks=[early_stopping(70, verbose=False), log_evaluation(0)],
                feature_name=feature_cols,
            )
            pred_valid = inverse_pred(model.predict(x.loc[valid_mask]), transform)
            pred_test = inverse_pred(model.predict(x.loc[test_mask]), transform)
            valid_row = evaluate(target, name, "valid", y_valid_raw, pred_valid)
            test_row = evaluate(target, name, "test", y_test_raw, pred_test)
            rows.extend([valid_row, test_row])
            if best is None or float(valid_row["r2"]) > best[0]:
                best = (float(valid_row["r2"]), name, model, params, pred_test)

        if best is None:
            continue
        _, best_name, best_model, best_params, best_pred_test = best
        joblib.dump(best_model, MODEL_DIR / f"{target}.joblib")
        joblib.dump(best_params, MODEL_DIR / f"{target}__selected_params.joblib")
        selected = evaluate(target, best_name, "selected_test", y_test_raw, best_pred_test)
        selected_rows.append(selected)

    metrics = pd.DataFrame(rows)
    selected = pd.DataFrame(selected_rows)
    metrics.to_csv(METRICS_PATH, index=False)
    selected.to_csv(BASE_DIR / "docs" / "adunbox_entity_history_lgbm_6h_business_v3__selected.csv", index=False)
    joblib.dump(
        {
            "feature_cols": feature_cols,
            "business_targets": BUSINESS_TARGETS,
            "hourly_input": str(anchor_v2.HOURLY_INPUT_PATH),
            "feature_cache": str(CACHE_PATH),
            "training_basis": "business_target_objective_search_6h_v3",
        },
        MODEL_DIR / "metadata.joblib",
    )

    lines = [
        "Adunbox Entity History LightGBM 6H Business V3 Summary",
        "",
        f"Hourly input: {anchor_v2.HOURLY_INPUT_PATH}",
        f"Anchor count: {len(features):,}",
        f"Feature count: {len(feature_cols):,}",
        f"Train rows: {int(train_mask.sum()):,}",
        f"Validation rows: {int(valid_mask.sum()):,}",
        f"Test rows: {int(test_mask.sum()):,}",
        "",
        "Selected business target test R2:",
    ]
    for row in selected.itertuples(index=False):
        lines.append(f"- {row.target}: {row.r2:.6f} ({row.candidate})")
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(SUMMARY_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
