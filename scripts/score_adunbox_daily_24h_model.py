from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import train_adunbox_daily_24h_model as daily24


BASE_DIR = Path(os.getenv("ADUNBOX_PROJECT_DIR", Path(__file__).resolve().parents[1]))
OUTPUT_PATH = BASE_DIR / "adunbox_daily_24h_latest_forecasts.csv"
SUMMARY_PATH = BASE_DIR / "adunbox_daily_24h_latest_forecasts__summary.txt"


def predict_latest(daily_input: Path) -> pd.DataFrame:
    daily, source = daily24.load_daily(force_rebuild_from_hourly=False, daily_input_path=daily_input)
    dataset, feature_cols = daily24.add_features_and_targets(daily)
    metadata = joblib.load(daily24.MODEL_DIR / "metadata.joblib")
    feature_cols = metadata.get("feature_cols", feature_cols)

    latest = (
        dataset.sort_values(["ad_id", "local_date"])
        .groupby("ad_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    X = latest[feature_cols].astype("float32")

    out = latest[["local_date", "timezone", *daily24.ENTITY_COLS, "days_active"]].copy()
    out["forecast_anchor_local_date"] = out["local_date"] + pd.Timedelta(days=1)
    for raw_target in daily24.RAW_TARGETS:
        target = f"target_24h_{raw_target}"
        model = joblib.load(daily24.MODEL_DIR / f"{target}.joblib")
        pred = np.expm1(model.predict(X))
        out[f"pred_24h_{raw_target}"] = np.maximum(0.0, pred).astype("float32")

    out = daily24.derive_kpis(out)
    out["model_source"] = "daily_24h_histgb"
    out["daily_source"] = source
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score latest 24h forecasts from the daily 24h model.")
    parser.add_argument("--daily-input", type=Path, default=daily24.DEFAULT_DAILY_INPUT_PATH)
    args = parser.parse_args()

    forecasts = predict_latest(args.daily_input)
    forecasts.to_csv(OUTPUT_PATH, index=False)
    lines = [
        "Adunbox Daily 24h Latest Forecasts",
        "",
        f"Rows scored: {len(forecasts):,}",
        f"Output: {OUTPUT_PATH}",
        f"Anchor date min: {forecasts['forecast_anchor_local_date'].min()}",
        f"Anchor date max: {forecasts['forecast_anchor_local_date'].max()}",
        "",
        "Prediction columns:",
        "- pred_24h_spend",
        "- pred_24h_impressions",
        "- pred_24h_inline_link_clicks",
        "- pred_24h_tracker_conversions",
        "- pred_24h_tracker_revenue",
        "- pred_24h_roas",
        "- pred_24h_profit",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
