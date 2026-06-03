from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import train_adunbox_daily_24h_full_db_optimized as opt24


BASE_DIR = Path(r"G:\ml_model_historical_data")
OUT = BASE_DIR / "github_release" / "outputs" / "adunbox_daily_24h_full_db_optimized__all_history_predictions.csv"
WINNERS = {
    "spend": "calibrated",
    "impressions": "base",
    "inline_link_clicks": "base",
    "tracker_conversions": "calibrated",
    "tracker_revenue": "base",
}


def safe_div(num: pd.Series, den: pd.Series, scale: float = 1.0) -> pd.Series:
    den = pd.to_numeric(den, errors="coerce").replace(0, np.nan)
    return (pd.to_numeric(num, errors="coerce") / den * scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_final_prediction_ranges(preds: pd.DataFrame) -> pd.DataFrame:
    """Expose production ranges around the final winner prediction.

    The optimized model chooses either base or calibrated output per raw metric.
    Calibration already creates p10/p50/p90 ranges, so this function carries those
    ranges forward to the final selected prediction and derives KPI ranges from
    consistent raw-metric bounds.
    """
    out = preds.copy()
    for raw_target in opt24.RAW_TARGETS:
        final_col = f"pred_final_24h_{raw_target}"
        range_p10 = f"pred_p10_24h_{raw_target}"
        range_p90 = f"pred_p90_24h_{raw_target}"
        out[f"pred_final_p10_24h_{raw_target}"] = np.minimum(out[final_col], out.get(range_p10, out[final_col])).astype("float32")
        out[f"pred_final_p50_24h_{raw_target}"] = out[final_col].astype("float32")
        out[f"pred_final_p90_24h_{raw_target}"] = np.maximum(out[final_col], out.get(range_p90, out[final_col])).astype("float32")

    eps = 1e-6
    p10_spend = out["pred_final_p10_24h_spend"]
    p90_spend = out["pred_final_p90_24h_spend"]
    p10_impr = out["pred_final_p10_24h_impressions"]
    p90_impr = out["pred_final_p90_24h_impressions"]
    p10_clicks = out["pred_final_p10_24h_inline_link_clicks"]
    p90_clicks = out["pred_final_p90_24h_inline_link_clicks"]
    p10_conv = out["pred_final_p10_24h_tracker_conversions"]
    p90_conv = out["pred_final_p90_24h_tracker_conversions"]
    p10_rev = out["pred_final_p10_24h_tracker_revenue"]
    p90_rev = out["pred_final_p90_24h_tracker_revenue"]

    out["pred_final_p10_24h_roas"] = safe_div(p10_rev, p90_spend.clip(lower=eps)).astype("float32")
    out["pred_final_p90_24h_roas"] = safe_div(p90_rev, p10_spend.clip(lower=eps)).astype("float32")
    out["pred_final_p10_24h_ctr"] = safe_div(p10_clicks, p90_impr.clip(lower=eps), 100.0).astype("float32")
    out["pred_final_p90_24h_ctr"] = safe_div(p90_clicks, p10_impr.clip(lower=eps), 100.0).astype("float32")
    out["pred_final_p10_24h_cvr"] = safe_div(p10_conv, p90_clicks.clip(lower=eps), 100.0).astype("float32")
    out["pred_final_p90_24h_cvr"] = safe_div(p90_conv, p10_clicks.clip(lower=eps), 100.0).astype("float32")
    out["pred_final_p10_24h_cpm"] = safe_div(p10_spend, p90_impr.clip(lower=eps), 1000.0).astype("float32")
    out["pred_final_p90_24h_cpm"] = safe_div(p90_spend, p10_impr.clip(lower=eps), 1000.0).astype("float32")
    return out


def add_production_flags(out: pd.DataFrame) -> pd.DataFrame:
    result = out.copy()
    roas_width = safe_div(
        result["pred_final_p90_24h_roas"] - result["pred_final_p10_24h_roas"],
        result["pred_final_24h_roas"].replace(0, np.nan),
    )
    revenue_width = safe_div(
        result["pred_final_p90_24h_tracker_revenue"] - result["pred_final_p10_24h_tracker_revenue"],
        result["pred_final_24h_tracker_revenue"].replace(0, np.nan),
    )
    result["prediction_interval_width_roas"] = roas_width.astype("float32")
    result["prediction_interval_width_revenue"] = revenue_width.astype("float32")

    segment = result["prediction_segment"].astype(str)
    reliability = result["kpi_reliability_flag"].astype(str)
    result["forecast_confidence"] = np.select(
        [
            segment.eq("stable") & reliability.eq("OK") & roas_width.le(1.25) & revenue_width.le(1.75),
            segment.isin(["stable", "spiky"]) & reliability.ne("LOW_SPEND") & roas_width.le(2.75),
        ],
        ["HIGH", "MEDIUM"],
        default="LOW",
    )
    result["production_guardrail_flag"] = np.select(
        [
            reliability.ne("OK"),
            segment.eq("new"),
            segment.eq("low_volume"),
            segment.eq("spiky") | roas_width.gt(2.75) | revenue_width.gt(3.50),
        ],
        ["LOW_DATA_RELIABILITY", "NEW_AD_HISTORY", "LOW_VOLUME_HISTORY", "SPIKY_OR_WIDE_RANGE"],
        default="OK",
    )
    result["forecast_usage_recommendation"] = np.select(
        [
            result["forecast_confidence"].eq("HIGH"),
            result["forecast_confidence"].eq("MEDIUM"),
        ],
        ["use_point_prediction", "use_point_plus_range"],
        default="use_range_only_review_manually",
    )
    return result


def main() -> None:
    daily, source = opt24.load_ad_daily(opt24.DEFAULT_DAILY_INPUT)
    dataset, feature_cols = opt24.build_features(daily)
    metadata = joblib.load(opt24.MODEL_DIR / "metadata.joblib")
    feature_cols = metadata.get("feature_cols", feature_cols)
    calibration_specs = json.loads(opt24.CALIBRATION_JSON.read_text(encoding="utf-8"))

    scored = opt24.add_error_control_features(dataset.copy())
    X = scored[feature_cols].astype("float32")
    preds = pd.DataFrame(index=scored.index)
    for raw_target in opt24.RAW_TARGETS:
        target = f"target_24h_{raw_target}"
        model = joblib.load(opt24.MODEL_DIR / f"{target}.joblib")
        base = np.maximum(0.0, np.expm1(model.predict(X))).astype("float32")
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
    preds = add_final_prediction_ranges(preds)

    out = scored[[
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
    ]].copy()
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
        if col.startswith("final_source_"):
            out[col] = preds[col].to_numpy()
        elif col.startswith("pred_p") or col.startswith("pred_final_p"):
            out[col] = preds[col].to_numpy()
    out = add_production_flags(out)

    out["daily_source"] = source
    out["model_source"] = "optimized_all_history"
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}")
    print(f"rows={len(out):,} dates={out['local_date'].min()} -> {out['local_date'].max()}")


if __name__ == "__main__":
    main()
