from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import score_adunbox_daily_24h_optimized_all_history as score_helpers
import train_adunbox_daily_24h_full_db_optimized as opt24


BASE_DIR = Path(r"G:\ml_model_historical_data")
RECENT_DAILY = Path(r"H:\adunbox_daily_breakdown_kpis.csv")
OUTPUT_DIR = BASE_DIR / "github_release" / "outputs"
PREDICTIONS_CSV = OUTPUT_DIR / "adunbox_daily_24h_recent_holdout_predictions.csv"
METRICS_CSV = OUTPUT_DIR / "adunbox_daily_24h_recent_holdout_metrics.csv"
SUMMARY_TXT = OUTPUT_DIR / "adunbox_daily_24h_recent_holdout_summary.txt"
HOLDOUT_START = pd.Timestamp("2026-05-13")

WINNERS = {
    "spend": "calibrated",
    "impressions": "base",
    "inline_link_clicks": "base",
    "tracker_conversions": "calibrated",
    "tracker_revenue": "base",
}


def wmape(actual: pd.Series, pred: pd.Series) -> float:
    denom = float(np.abs(actual).sum())
    return float(np.abs(actual - pred).sum() / denom) if denom else 0.0


def bias(actual: pd.Series, pred: pd.Series) -> float:
    denom = float(np.abs(actual).sum())
    return float((pred - actual).sum() / denom) if denom else 0.0


def metric_row(name: str, actual: pd.Series, pred: pd.Series) -> dict[str, object]:
    actual = pd.to_numeric(actual, errors="coerce").fillna(0.0).astype("float64")
    pred = pd.to_numeric(pred, errors="coerce").fillna(0.0).astype("float64")
    return {
        "metric": name,
        "rows": int(len(actual)),
        "actual_sum": float(actual.sum()),
        "pred_sum": float(pred.sum()),
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
        "r2": float(r2_score(actual, pred)) if len(actual) > 1 else 0.0,
        "wmape": wmape(actual, pred),
        "bias": bias(actual, pred),
        "underprediction_rate": float((pred.to_numpy() < actual.to_numpy()).mean()) if len(actual) else 0.0,
    }


def score_recent_holdout() -> pd.DataFrame:
    daily, source = opt24.load_ad_daily(RECENT_DAILY)
    dataset, feature_cols = opt24.build_features(daily)
    dataset = dataset[dataset["local_date"] >= HOLDOUT_START].copy()
    if dataset.empty:
        raise RuntimeError(f"No holdout feature rows found from {HOLDOUT_START.date()} onward.")

    metadata = joblib.load(opt24.MODEL_DIR / "metadata.joblib")
    feature_cols = metadata.get("feature_cols", feature_cols)
    calibration_specs = json.loads(opt24.CALIBRATION_JSON.read_text(encoding="utf-8"))

    scored = opt24.add_error_control_features(dataset.copy())
    x = scored[feature_cols].astype("float32")
    preds = pd.DataFrame(index=scored.index)

    for raw_target in opt24.RAW_TARGETS:
        target = f"target_24h_{raw_target}"
        model = joblib.load(opt24.MODEL_DIR / f"{target}.joblib")
        base = np.maximum(0.0, np.expm1(model.predict(x))).astype("float32")
        cal = opt24.apply_target_calibration(scored, raw_target, base, calibration_specs[raw_target])
        preds[f"pred_24h_{raw_target}"] = base
        preds[f"pred_calibrated_24h_{raw_target}"] = cal
        ranges = opt24.prediction_ranges(scored, raw_target, cal, calibration_specs[raw_target])
        for col in ranges.columns:
            preds[col] = ranges[col].to_numpy()
        source_choice = WINNERS[raw_target]
        preds[f"pred_final_24h_{raw_target}"] = cal if source_choice == "calibrated" else base
        preds[f"final_source_24h_{raw_target}"] = source_choice

    pred_base = opt24.derive_kpis(preds[[f"pred_24h_{t}" for t in opt24.RAW_TARGETS]])
    pred_final_input = preds[[f"pred_final_24h_{t}" for t in opt24.RAW_TARGETS]].rename(
        columns={f"pred_final_24h_{t}": f"pred_24h_{t}" for t in opt24.RAW_TARGETS}
    )
    pred_final = opt24.derive_kpis(pred_final_input).rename(
        columns={c: c.replace("pred_24h_", "pred_final_24h_") for c in opt24.derive_kpis(pred_final_input).columns}
    )
    preds = score_helpers.add_final_prediction_ranges(preds)

    out = scored[
        [
            "local_date",
            "timezone",
            *opt24.ENTITY_COLS,
            "days_active",
            *opt24.RAW_TARGETS,
            "kpi_roas",
            "kpi_profit",
            "kpi_ctr",
            "kpi_cvr",
            "kpi_cpc",
            "kpi_cpm",
        ]
    ].copy()
    out = out.rename(columns={col: f"actual_24h_{col}" for col in opt24.RAW_TARGETS})
    out["prediction_segment"] = opt24.classify_prediction_segment(scored).to_numpy()
    out["kpi_reliability_flag"] = np.select(
        [
            out["actual_24h_impressions"] < 100,
            out["actual_24h_inline_link_clicks"] < 5,
            out["actual_24h_spend"] < 1,
        ],
        ["LOW_IMPRESSIONS", "LOW_CLICKS", "LOW_SPEND"],
        default="OK",
    )
    for col in pred_base.columns:
        out[col] = pred_base[col].to_numpy()
    for col in pred_final.columns:
        out[col] = pred_final[col].to_numpy()
    for col in preds.columns:
        if col.startswith("final_source_") or col.startswith("pred_p") or col.startswith("pred_final_p"):
            out[col] = preds[col].to_numpy()
    out = score_helpers.add_production_flags(out)
    out["daily_source"] = str(source)
    out["model_source"] = "optimized_model_recent_out_of_time_holdout"
    return out


def write_metrics(out: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric_pairs = [
        ("spend", "actual_24h_spend", "pred_final_24h_spend"),
        ("impressions", "actual_24h_impressions", "pred_final_24h_impressions"),
        ("clicks", "actual_24h_inline_link_clicks", "pred_final_24h_inline_link_clicks"),
        ("conversions", "actual_24h_tracker_conversions", "pred_final_24h_tracker_conversions"),
        ("revenue", "actual_24h_tracker_revenue", "pred_final_24h_tracker_revenue"),
        ("roas", "kpi_roas", "pred_final_24h_roas"),
        ("profit", "kpi_profit", "pred_final_24h_profit"),
        ("ctr", "kpi_ctr", "pred_final_24h_ctr"),
        ("cvr", "kpi_cvr", "pred_final_24h_cvr"),
        ("cpm", "kpi_cpm", "pred_final_24h_cpm"),
    ]
    for name, actual_col, pred_col in metric_pairs:
        rows.append(metric_row(name, out[actual_col], out[pred_col]))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(METRICS_CSV, index=False)
    return metrics


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = score_recent_holdout()
    out.to_csv(PREDICTIONS_CSV, index=False)
    metrics = write_metrics(out)
    lines = [
        "Adunbox 24h Recent Out-of-Time Holdout",
        "",
        f"Recent source: {RECENT_DAILY}",
        f"Frozen model dir: {opt24.MODEL_DIR}",
        f"Holdout start: {HOLDOUT_START.date()}",
        f"Rows scored: {len(out):,}",
        f"Date range: {out['local_date'].min()} -> {out['local_date'].max()}",
        f"Accounts: {out['account_id'].nunique():,}",
        f"Ads: {out['ad_id'].nunique():,}",
        f"Predictions: {PREDICTIONS_CSV}",
        f"Metrics: {METRICS_CSV}",
        "",
        "Metrics:",
    ]
    for rec in metrics.itertuples(index=False):
        lines.append(f"- {rec.metric}: r2={rec.r2:.4f}, wmape={rec.wmape:.4f}, bias={rec.bias:.4f}")
    SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
