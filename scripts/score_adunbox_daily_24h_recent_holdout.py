from __future__ import annotations

import json
from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import score_adunbox_daily_24h_optimized_all_history as score_helpers
import train_adunbox_daily_24h_full_db_production as prod24
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


def active_model_context():
    """Prefer the production retrained model when it exists.

    The production model is trained in the current Python/sklearn environment and
    uses the memory-safe production feature builder. If it is not available, the
    scorer falls back to the older optimized model.
    """
    production_metadata = prod24.MODEL_DIR / "metadata.joblib"
    if production_metadata.exists() and prod24.FULL_READY_FLAG.exists():
        return {
            "name": "production_retrained",
            "module": prod24.base,
            "model_dir": prod24.MODEL_DIR,
            "calibration_json": prod24.CALIBRATION_JSON,
            "feature_builder": prod24.build_features_production_safe,
        }
    return {
        "name": "optimized_frozen",
        "module": opt24,
        "model_dir": opt24.MODEL_DIR,
        "calibration_json": opt24.CALIBRATION_JSON,
        "feature_builder": opt24.build_features,
    }


def load_ad_daily_flexible(path: Path, model_module, sample_ads: int | None = None) -> tuple[pd.DataFrame, str]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    # Some data exports keep real signal in conversions/conversions_value while
    # tracker_conversions/tracker_revenue are all zero. Prefer the real business
    # columns when present, then fall back to tracker columns.
    conversion_col = "conversions" if "conversions" in header else ("tracker_conversions" if "tracker_conversions" in header else "tracker_conversion")
    revenue_col = "conversions_value" if "conversions_value" in header else "tracker_revenue"
    usecols = ["entity_type", "date", "timezone", *model_module.ENTITY_COLS, "spend", "impressions", "inline_link_clicks", conversion_col, revenue_col]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=50_000, low_memory=False):
        chunk = chunk[chunk["entity_type"].astype(str).str.lower().eq("ad")].copy()
        if chunk.empty:
            continue
        rename_map = {}
        if conversion_col != "tracker_conversions":
            rename_map[conversion_col] = "tracker_conversions"
        if revenue_col != "tracker_revenue":
            rename_map[revenue_col] = "tracker_revenue"
        if rename_map:
            chunk = chunk.rename(columns=rename_map)
        for col in model_module.ENTITY_COLS:
            chunk[col] = model_module.normalize_id(chunk[col])
        chunk["timezone"] = chunk["timezone"].fillna("").astype(str)
        chunk["local_date"] = pd.to_datetime(chunk["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
        chunk = chunk[chunk["local_date"].notna()].copy()
        for col in model_module.RAW_TARGETS:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0.0).astype("float32")
        parts.append(chunk[["local_date", "timezone", *model_module.ENTITY_COLS, *model_module.RAW_TARGETS]])
    if not parts:
        raise RuntimeError(f"No ad-level rows found in {path}")
    daily = pd.concat(parts, ignore_index=True)
    daily = (
        daily.groupby(["local_date", "timezone", *model_module.ENTITY_COLS], as_index=False)[model_module.RAW_TARGETS]
        .sum()
        .sort_values(["ad_id", "local_date"])
        .reset_index(drop=True)
    )
    eligible = daily.groupby("ad_id")["local_date"].nunique()
    eligible_ads = eligible[eligible >= model_module.MIN_AD_DAYS].index.astype(str)
    if sample_ads:
        eligible_ads = eligible.loc[eligible_ads].sort_values(ascending=False).head(sample_ads).index.astype(str)
    daily = daily[daily["ad_id"].isin(eligible_ads)].copy()
    return daily, str(path)


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


def score_recent_holdout(recent_daily: Path = RECENT_DAILY, holdout_start: pd.Timestamp = HOLDOUT_START) -> pd.DataFrame:
    ctx = active_model_context()
    model_module = ctx["module"]
    daily, source = load_ad_daily_flexible(recent_daily, model_module)
    dataset, feature_cols = ctx["feature_builder"](daily)
    dataset = dataset[dataset["local_date"] >= holdout_start].copy()
    if dataset.empty:
        raise RuntimeError(f"No holdout feature rows found from {holdout_start.date()} onward.")

    metadata = joblib.load(ctx["model_dir"] / "metadata.joblib")
    feature_cols = metadata.get("feature_cols", feature_cols)
    calibration_specs = json.loads(ctx["calibration_json"].read_text(encoding="utf-8"))

    scored = model_module.add_error_control_features(dataset.copy())
    x = scored[feature_cols].astype("float32")
    preds = pd.DataFrame(index=scored.index)

    for raw_target in model_module.RAW_TARGETS:
        target = f"target_24h_{raw_target}"
        model = joblib.load(ctx["model_dir"] / f"{target}.joblib")
        base = np.maximum(0.0, np.expm1(model.predict(x))).astype("float32")
        cal = model_module.apply_target_calibration(scored, raw_target, base, calibration_specs[raw_target])
        preds[f"pred_24h_{raw_target}"] = base
        preds[f"pred_calibrated_24h_{raw_target}"] = cal
        ranges = model_module.prediction_ranges(scored, raw_target, cal, calibration_specs[raw_target])
        for col in ranges.columns:
            preds[col] = ranges[col].to_numpy()
        source_choice = WINNERS[raw_target]
        preds[f"pred_final_24h_{raw_target}"] = cal if source_choice == "calibrated" else base
        preds[f"final_source_24h_{raw_target}"] = source_choice

    pred_base = model_module.derive_kpis(preds[[f"pred_24h_{t}" for t in model_module.RAW_TARGETS]])
    pred_final_input = preds[[f"pred_final_24h_{t}" for t in model_module.RAW_TARGETS]].rename(
        columns={f"pred_final_24h_{t}": f"pred_24h_{t}" for t in model_module.RAW_TARGETS}
    )
    pred_final = model_module.derive_kpis(pred_final_input).rename(
        columns={c: c.replace("pred_24h_", "pred_final_24h_") for c in model_module.derive_kpis(pred_final_input).columns}
    )
    preds = score_helpers.add_final_prediction_ranges(preds)

    out = scored[
        [
            "local_date",
            "timezone",
            *model_module.ENTITY_COLS,
            "days_active",
            *model_module.RAW_TARGETS,
            "kpi_roas",
            "kpi_profit",
            "kpi_ctr",
            "kpi_cvr",
            "kpi_cpc",
            "kpi_cpm",
        ]
    ].copy()
    out = out.rename(columns={col: f"actual_24h_{col}" for col in model_module.RAW_TARGETS})
    out["prediction_segment"] = model_module.classify_prediction_segment(scored).to_numpy()
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
    out["model_source"] = f"{ctx['name']}_recent_out_of_time_holdout"
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
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a recent daily CSV with the trained 24h model.")
    parser.add_argument("--recent-daily", type=Path, default=RECENT_DAILY)
    parser.add_argument("--holdout-start", type=str, default=str(HOLDOUT_START.date()))
    parser.add_argument("--predictions-csv", type=Path, default=PREDICTIONS_CSV)
    parser.add_argument("--metrics-csv", type=Path, default=METRICS_CSV)
    parser.add_argument("--summary-txt", type=Path, default=SUMMARY_TXT)
    args = parser.parse_args()
    holdout_start = pd.Timestamp(args.holdout_start)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = score_recent_holdout(args.recent_daily, holdout_start)
    args.predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_txt.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.predictions_csv, index=False)
    metrics = write_metrics(out)
    metrics.to_csv(args.metrics_csv, index=False)
    lines = [
        "Adunbox 24h Recent Out-of-Time Holdout",
        "",
        f"Recent source: {args.recent_daily}",
        f"Model source: {out['model_source'].iloc[0] if len(out) else 'unknown'}",
        f"Holdout start: {holdout_start.date()}",
        f"Rows scored: {len(out):,}",
        f"Date range: {out['local_date'].min()} -> {out['local_date'].max()}",
        f"Accounts: {out['account_id'].nunique():,}",
        f"Ads: {out['ad_id'].nunique():,}",
        f"Predictions: {args.predictions_csv}",
        f"Metrics: {args.metrics_csv}",
        "",
        "Metrics:",
    ]
    for rec in metrics.itertuples(index=False):
        lines.append(f"- {rec.metric}: r2={rec.r2:.4f}, wmape={rec.wmape:.4f}, bias={rec.bias:.4f}")
    args.summary_txt.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
