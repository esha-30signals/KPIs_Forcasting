from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import train_adunbox_daily_24h_full_db_optimized as base


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


def load_multi_source_daily(paths: list[Path], sample_ads: int | None = None) -> tuple[pd.DataFrame, str]:
    frames = []
    sources = []
    for path in paths:
        if not path.exists():
            continue
        daily, source = base.load_ad_daily(path, sample_ads=None)
        frames.append(daily)
        sources.append(source)
    if not frames:
        raise RuntimeError("No daily input files found.")
    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.groupby(["local_date", "timezone", *base.ENTITY_COLS], as_index=False)[base.RAW_TARGETS]
        .sum()
        .sort_values(["ad_id", "local_date"])
        .reset_index(drop=True)
    )
    eligible = combined.groupby("ad_id")["local_date"].nunique()
    eligible_ads = eligible[eligible >= base.MIN_AD_DAYS].index.astype(str)
    if sample_ads:
        eligible_ads = eligible.loc[eligible_ads].sort_values(ascending=False).head(sample_ads).index.astype(str)
    combined = combined[combined["ad_id"].isin(eligible_ads)].copy()
    return combined, " + ".join(sources)


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
    base_cols = [*base.RAW_TARGETS, "kpi_ctr", "kpi_cpm", "kpi_cvr", "kpi_roas", "kpi_profit"]
    raw_set = set(base.RAW_TARGETS)

    for col in base_cols:
        for lag in [1, 2, 3, 7]:
            name = f"{col}_lag_{lag}d"
            feature_data[name] = grouped[col].shift(lag).fillna(0.0).astype("float32")
            feature_cols.append(name)
        shifted = grouped[col].shift(1)
        shifted_grouped = shifted.groupby(out["ad_id"], sort=False)
        for window in [3, 7]:
            mean_name = f"{col}_roll_mean_{window}d"
            std_name = f"{col}_roll_std_{window}d"
            feature_data[mean_name] = (
                shifted_grouped.rolling(window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
                .fillna(0.0)
                .astype("float32")
            )
            feature_data[std_name] = (
                shifted_grouped.rolling(window, min_periods=2)
                .std()
                .reset_index(level=0, drop=True)
                .fillna(0.0)
                .astype("float32")
            )
            feature_cols.extend([mean_name, std_name])
            if col in raw_set:
                sum_name = f"{col}_roll_sum_{window}d"
                zero_name = f"{col}_zero_count_{window}d"
                feature_data[sum_name] = (
                    shifted_grouped.rolling(window, min_periods=1)
                    .sum()
                    .reset_index(level=0, drop=True)
                    .fillna(0.0)
                    .astype("float32")
                )
                feature_data[zero_name] = (
                    shifted_grouped.rolling(window, min_periods=1)
                    .apply(lambda x: float(np.sum(x == 0)), raw=True)
                    .reset_index(level=0, drop=True)
                    .fillna(0.0)
                    .astype("float32")
                )
                feature_cols.extend([sum_name, zero_name])

    feature_frame = pd.DataFrame(feature_data, index=out.index)
    out = pd.concat([out, feature_frame], axis=1)

    for col in base.RAW_TARGETS:
        mean_3 = out.get(f"{col}_roll_mean_3d", 0.0)
        mean_7 = out.get(f"{col}_roll_mean_7d", 0.0)
        std_7 = out.get(f"{col}_roll_std_7d", 0.0)
        lag_1 = out.get(f"{col}_lag_1d", 0.0)
        out[f"{col}_trend_3d_vs_7d"] = base.safe_div(mean_3, pd.Series(mean_7).replace(0, np.nan)).fillna(0.0).astype("float32")
        out[f"{col}_cv_7d"] = base.safe_div(std_7, pd.Series(mean_7).replace(0, np.nan)).fillna(0.0).astype("float32")
        out[f"{col}_lag1_vs_mean_7d"] = base.safe_div(lag_1, pd.Series(mean_7).replace(0, np.nan)).fillna(0.0).astype("float32")
        feature_cols.extend([f"{col}_trend_3d_vs_7d", f"{col}_cv_7d", f"{col}_lag1_vs_mean_7d"])

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


def sample_weight(frame: pd.DataFrame) -> np.ndarray:
    weights = np.ones(len(frame), dtype=np.float32)
    weights[pd.to_datetime(frame["local_date"]) >= RECENCY_WEIGHT_START] = RECENCY_WEIGHT_MULTIPLIER
    return weights


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
        x_train = train[feature_cols].astype("float32")
        x_valid = valid[feature_cols].astype("float32")
        x_test = test[feature_cols].astype("float32")
        weights = sample_weight(train)
        for raw_target in base.RAW_TARGETS:
            target = f"target_24h_{raw_target}"
            y_train = train[target].astype("float32")
            y_valid = valid[target].astype("float32")
            y_test = test[target].astype("float32")
            model = base.build_model()
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
    global TRAIN_END, VALID_END, RECENCY_WEIGHT_START, RECENCY_WEIGHT_MULTIPLIER

    parser = argparse.ArgumentParser(description="Production retrain with original + recent daily data and recency weighting.")
    parser.add_argument("--original-daily", type=Path, default=ORIGINAL_DAILY)
    parser.add_argument("--recent-daily", type=Path, default=RECENT_DAILY)
    parser.add_argument("--train-end", type=str, default=str(TRAIN_END.date()), help="Last date included in train split, e.g. 2026-05-25.")
    parser.add_argument("--valid-end", type=str, default=str(VALID_END.date()), help="Last date included in validation split. Dates after this become test.")
    parser.add_argument("--recency-weight-start", type=str, default=str(RECENCY_WEIGHT_START.date()))
    parser.add_argument("--recency-weight-multiplier", type=float, default=RECENCY_WEIGHT_MULTIPLIER)
    parser.add_argument("--sample-ads", type=int, default=0)
    args = parser.parse_args()

    TRAIN_END = pd.Timestamp(args.train_end)
    VALID_END = pd.Timestamp(args.valid_end)
    RECENCY_WEIGHT_START = pd.Timestamp(args.recency_weight_start)
    RECENCY_WEIGHT_MULTIPLIER = float(args.recency_weight_multiplier)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily, source = load_multi_source_daily([args.original_daily, args.recent_daily], sample_ads=args.sample_ads or None)
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
