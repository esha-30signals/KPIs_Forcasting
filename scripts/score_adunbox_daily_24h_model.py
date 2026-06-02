from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn._loss as sklearn_loss

import train_adunbox_daily_24h_model as daily24

sys.modules.setdefault("_loss", sklearn_loss)


BASE_DIR = Path(os.getenv("ADUNBOX_PROJECT_DIR", Path(__file__).resolve().parents[1]))
OUTPUT_PATH = BASE_DIR / "adunbox_daily_24h_latest_forecasts.csv"
SUMMARY_PATH = BASE_DIR / "adunbox_daily_24h_latest_forecasts__summary.txt"
MIN_24H_KPI_IMPRESSIONS = 100.0
MIN_24H_KPI_CLICKS = 5.0
MIN_24H_RECENT_SPEND_MEAN = 1.0
MIN_24H_HISTORY_DAYS = 14


def add_24h_confidence_and_ranges(out: pd.DataFrame) -> pd.DataFrame:
    scored = out.copy()
    spend_mean = pd.to_numeric(scored.get("spend_roll_mean_7d", 0.0), errors="coerce").fillna(0.0)
    impressions_mean = pd.to_numeric(scored.get("impressions_roll_mean_7d", 0.0), errors="coerce").fillna(0.0)
    clicks_mean = pd.to_numeric(scored.get("inline_link_clicks_roll_mean_7d", 0.0), errors="coerce").fillna(0.0)
    pred_impressions = pd.to_numeric(scored.get("pred_24h_impressions", 0.0), errors="coerce").fillna(0.0)
    pred_clicks = pd.to_numeric(scored.get("pred_24h_inline_link_clicks", 0.0), errors="coerce").fillna(0.0)
    days_active = pd.to_numeric(scored.get("days_active", 0), errors="coerce").fillna(0)

    low_history = days_active < MIN_24H_HISTORY_DAYS
    low_recent_volume = (
        (spend_mean < MIN_24H_RECENT_SPEND_MEAN)
        | (impressions_mean < MIN_24H_KPI_IMPRESSIONS)
        | (clicks_mean < 1.0)
    )
    low_kpi_volume = (pred_impressions < MIN_24H_KPI_IMPRESSIONS) | (pred_clicks < MIN_24H_KPI_CLICKS)

    scored["forecast_confidence"] = np.select(
        [low_history | low_recent_volume, low_kpi_volume],
        ["LOW", "MEDIUM"],
        default="HIGH",
    )
    scored["kpi_reliability_flag"] = np.select(
        [
            low_history,
            spend_mean < MIN_24H_RECENT_SPEND_MEAN,
            pred_impressions < MIN_24H_KPI_IMPRESSIONS,
            pred_clicks < MIN_24H_KPI_CLICKS,
        ],
        ["LOW_RECENT_HISTORY", "LOW_RECENT_SPEND", "LOW_IMPRESSIONS", "LOW_CLICKS"],
        default="OK",
    )
    scored["forecast_use_case"] = np.where(
        scored["forecast_confidence"].eq("LOW"),
        "benchmark_range",
        np.where(scored["forecast_confidence"].eq("MEDIUM"), "point_with_caution", "point_forecast"),
    )

    scored["benchmark_source"] = np.select(
        [
            scored["forecast_confidence"].eq("LOW") & scored.get("campaign_id", pd.Series("", index=scored.index)).astype(str).ne(""),
            scored["forecast_confidence"].eq("LOW") & scored.get("account_id", pd.Series("", index=scored.index)).astype(str).ne(""),
        ],
        ["campaign_recent_7d", "account_recent_7d"],
        default="model_point",
    )
    width = np.select(
        [scored["forecast_confidence"].eq("LOW"), scored["forecast_confidence"].eq("MEDIUM")],
        [0.75, 0.45],
        default=0.25,
    )
    for metric in daily24.RAW_TARGETS:
        pred_col = f"pred_24h_{metric}"
        if pred_col not in scored.columns:
            continue
        pred = pd.to_numeric(scored[pred_col], errors="coerce").fillna(0.0)
        recent_col = f"{metric}_roll_mean_7d"
        recent = pd.to_numeric(scored.get(recent_col, pred), errors="coerce").fillna(0.0)
        campaign_benchmark = recent.groupby(scored["campaign_id"].astype(str)).transform("mean") if "campaign_id" in scored.columns else recent
        account_benchmark = recent.groupby(scored["account_id"].astype(str)).transform("mean") if "account_id" in scored.columns else recent
        fallback = np.where(campaign_benchmark > 0.0, campaign_benchmark, account_benchmark)
        recommended = np.where(scored["forecast_confidence"].eq("LOW"), np.maximum(fallback, 0.0), pred)
        recommended = pd.Series(recommended, index=scored.index).fillna(pred)
        scored[f"recommended_24h_{metric}"] = recommended
        scored[f"{pred_col}_p50"] = recommended
        scored[f"{pred_col}_p10"] = np.maximum(0.0, recommended * (1.0 - width))
        scored[f"{pred_col}_p90"] = recommended * (1.0 + width)

    spend = pd.to_numeric(scored.get("recommended_24h_spend", scored.get("pred_24h_spend", 0.0)), errors="coerce").fillna(0.0)
    impressions = pd.to_numeric(scored.get("recommended_24h_impressions", scored.get("pred_24h_impressions", 0.0)), errors="coerce").fillna(0.0)
    clicks = pd.to_numeric(scored.get("recommended_24h_inline_link_clicks", scored.get("pred_24h_inline_link_clicks", 0.0)), errors="coerce").fillna(0.0)
    conversions = pd.to_numeric(scored.get("recommended_24h_tracker_conversions", scored.get("pred_24h_tracker_conversions", 0.0)), errors="coerce").fillna(0.0)
    revenue = pd.to_numeric(scored.get("recommended_24h_tracker_revenue", scored.get("pred_24h_tracker_revenue", 0.0)), errors="coerce").fillna(0.0)
    scored["recommended_24h_roas"] = np.where(spend > 0.0, revenue / spend, 0.0)
    scored["recommended_24h_profit"] = revenue - spend
    scored["recommended_24h_ctr"] = np.where(impressions > 0.0, clicks / impressions * 100.0, 0.0)
    scored["recommended_24h_cvr"] = np.where(clicks > 0.0, conversions / clicks * 100.0, 0.0)
    scored["recommended_24h_cpc"] = np.where(clicks > 0.0, spend / clicks, 0.0)
    scored["recommended_24h_cpm"] = np.where(impressions > 0.0, spend / impressions * 1000.0, 0.0)

    scored["forecast_note"] = np.select(
        [scored["kpi_reliability_flag"].ne("OK"), scored["forecast_confidence"].eq("MEDIUM")],
        ["KPI ratios are low-confidence; use raw metric range.", "Use point forecast with caution."],
        default="Forecast is stable enough for point use.",
    )
    return scored


def predict_latest(daily_input: Path) -> pd.DataFrame:
    daily, source = daily24.load_daily(force_rebuild_from_hourly=False, daily_input_path=daily_input)
    latest_actual = (
        daily.sort_values(["ad_id", "local_date"])
        .groupby("ad_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    future_rows = latest_actual[["local_date", "timezone", *daily24.ENTITY_COLS, *daily24.RAW_TARGETS]].copy()
    future_rows["history_cutoff_local_date"] = future_rows["local_date"]
    future_rows["local_date"] = future_rows["local_date"] + pd.Timedelta(days=1)
    for metric in daily24.RAW_TARGETS:
        future_rows[metric] = 0.0

    scoring_daily = pd.concat(
        [daily, future_rows[["local_date", "timezone", *daily24.ENTITY_COLS, *daily24.RAW_TARGETS]]],
        ignore_index=True,
    )
    dataset, feature_cols = daily24.add_features_and_targets(scoring_daily)
    metadata = joblib.load(daily24.MODEL_DIR / "metadata.joblib")
    feature_cols = metadata.get("feature_cols", feature_cols)

    future_keys = future_rows[["ad_id", "local_date", "history_cutoff_local_date"]].copy()
    latest = dataset.merge(future_keys, on=["ad_id", "local_date"], how="inner")
    latest = latest.sort_values(["ad_id", "local_date"]).drop_duplicates(["ad_id"], keep="last").reset_index(drop=True)
    X = latest[feature_cols].astype("float32")

    roll_cols = [col for col in latest.columns if col.endswith("_roll_mean_7d")]
    out = latest[["local_date", "history_cutoff_local_date", "timezone", *daily24.ENTITY_COLS, "days_active", *roll_cols]].copy()
    out = out.rename(columns={"local_date": "forecast_anchor_local_date"})
    for raw_target in daily24.RAW_TARGETS:
        target = f"target_24h_{raw_target}"
        model = joblib.load(daily24.MODEL_DIR / f"{target}.joblib")
        pred = np.expm1(model.predict(X))
        out[f"pred_24h_{raw_target}"] = np.maximum(0.0, pred).astype("float32")

    out = daily24.derive_kpis(out)
    out = add_24h_confidence_and_ranges(out)
    out["model_source"] = "daily_24h_histgb"
    out["daily_source"] = source
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score latest 24h forecasts from the daily 24h model.")
    parser.add_argument("--daily-input", type=Path, default=daily24.DEFAULT_DAILY_INPUT_PATH)
    args = parser.parse_args()

    forecasts = predict_latest(args.daily_input)
    forecasts.to_csv(OUTPUT_PATH, index=False)
    confidence_counts = forecasts["forecast_confidence"].value_counts(dropna=False).to_dict()
    reliability_counts = forecasts["kpi_reliability_flag"].value_counts(dropna=False).to_dict()
    lines = [
        "Adunbox Daily 24h Latest Forecasts",
        "",
        f"Rows scored: {len(forecasts):,}",
        f"Output: {OUTPUT_PATH}",
        f"Anchor date min: {forecasts['forecast_anchor_local_date'].min()}",
        f"Anchor date max: {forecasts['forecast_anchor_local_date'].max()}",
        f"Forecast confidence counts: {confidence_counts}",
        f"KPI reliability counts: {reliability_counts}",
        "",
        "Prediction columns:",
        "- pred_24h_spend",
        "- pred_24h_impressions",
        "- pred_24h_inline_link_clicks",
        "- pred_24h_tracker_conversions",
        "- pred_24h_tracker_revenue",
        "- pred_24h_*_p10 / p50 / p90 range columns",
        "- recommended_24h_* raw and KPI columns",
        "- pred_24h_roas",
        "- pred_24h_profit",
        "- forecast_confidence",
        "- kpi_reliability_flag",
        "- forecast_use_case",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
