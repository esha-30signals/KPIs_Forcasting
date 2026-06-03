from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(r"G:\ml_model_historical_data")
BACKTEST = BASE_DIR / "github_release" / "outputs" / "adunbox_daily_24h_full_db_optimized__all_history_predictions.csv"
COMPARISON = BASE_DIR / "github_release" / "outputs" / "adunbox_daily_24h_previous_vs_optimized_comparison.csv"
DAILY_SRC = Path(r"C:\Users\eshaa\Downloads\adunbox_daily_breakdown_kpis.csv")
REFERENCE_HTML = Path(r"C:\Users\eshaa\Downloads\adunbox_24h_historical_actual_vs_predicted_kpi_dashboard (1).html")
OUT = Path(r"C:\Users\eshaa\Downloads\adunbox_24h_optimized_actual_vs_predicted_kpi_dashboard.html")
ACCOUNTS = ["7730708", "7738188", "36061656"]


RAW_COLS = [
    "local_date",
    "account_id",
    "campaign_id",
    "adset_id",
    "ad_id",
    "actual_24h_spend",
    "actual_24h_impressions",
    "actual_24h_inline_link_clicks",
    "actual_24h_tracker_conversions",
    "actual_24h_tracker_revenue",
    "kpi_roas",
    "kpi_profit",
    "kpi_ctr",
    "kpi_cvr",
    "kpi_cpm",
    "pred_24h_spend",
    "pred_24h_impressions",
    "pred_24h_inline_link_clicks",
    "pred_24h_tracker_conversions",
    "pred_24h_tracker_revenue",
    "pred_24h_roas",
    "pred_24h_ctr",
    "pred_24h_cvr",
    "pred_24h_cpm",
    "pred_final_24h_spend",
    "pred_final_24h_impressions",
    "pred_final_24h_inline_link_clicks",
    "pred_final_24h_tracker_conversions",
    "pred_final_24h_tracker_revenue",
    "pred_final_24h_roas",
    "pred_final_24h_ctr",
    "pred_final_24h_cvr",
    "pred_final_24h_cpm",
    "pred_final_p10_24h_spend",
    "pred_final_p90_24h_spend",
    "pred_final_p10_24h_impressions",
    "pred_final_p90_24h_impressions",
    "pred_final_p10_24h_inline_link_clicks",
    "pred_final_p90_24h_inline_link_clicks",
    "pred_final_p10_24h_tracker_conversions",
    "pred_final_p90_24h_tracker_conversions",
    "pred_final_p10_24h_tracker_revenue",
    "pred_final_p90_24h_tracker_revenue",
    "pred_final_p10_24h_roas",
    "pred_final_p90_24h_roas",
    "pred_final_p10_24h_ctr",
    "pred_final_p90_24h_ctr",
    "pred_final_p10_24h_cvr",
    "pred_final_p90_24h_cvr",
    "pred_final_p10_24h_cpm",
    "pred_final_p90_24h_cpm",
    "final_source_24h_spend",
    "final_source_24h_impressions",
    "final_source_24h_inline_link_clicks",
    "final_source_24h_tracker_conversions",
    "final_source_24h_tracker_revenue",
    "kpi_reliability_flag",
    "prediction_segment",
    "forecast_confidence",
    "production_guardrail_flag",
    "forecast_usage_recommendation",
    "prediction_interval_width_roas",
    "prediction_interval_width_revenue",
]


def clean_id(value) -> str:
    if pd.isna(value):
        return ""
    try:
        return str(int(float(value)))
    except Exception:
        return str(value)


def safe_div(num, den, mul=1.0):
    num = pd.Series(num, dtype="float64").fillna(0.0)
    den = pd.Series(den, dtype="float64").fillna(0.0)
    out = pd.Series(np.zeros(len(num)), index=num.index)
    mask = den != 0
    out.loc[mask] = num.loc[mask] / den.loc[mask] * mul
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def round_list(values, digits=4):
    return [round(float(v), digits) if pd.notna(v) else 0.0 for v in values]


def majority_label(values: pd.Series, default: str = "MIXED") -> str:
    counts = values.astype(str).value_counts()
    return str(counts.idxmax()) if len(counts) else default


def add_range_kpis(grouped: pd.DataFrame) -> pd.DataFrame:
    grouped["pred_final_p10_24h_roas"] = safe_div(grouped["pred_final_p10_24h_tracker_revenue"], grouped["pred_final_p90_24h_spend"])
    grouped["pred_final_p90_24h_roas"] = safe_div(grouped["pred_final_p90_24h_tracker_revenue"], grouped["pred_final_p10_24h_spend"])
    grouped["pred_final_p10_24h_ctr"] = safe_div(grouped["pred_final_p10_24h_inline_link_clicks"], grouped["pred_final_p90_24h_impressions"], 100.0)
    grouped["pred_final_p90_24h_ctr"] = safe_div(grouped["pred_final_p90_24h_inline_link_clicks"], grouped["pred_final_p10_24h_impressions"], 100.0)
    grouped["pred_final_p10_24h_cvr"] = safe_div(grouped["pred_final_p10_24h_tracker_conversions"], grouped["pred_final_p90_24h_inline_link_clicks"], 100.0)
    grouped["pred_final_p90_24h_cvr"] = safe_div(grouped["pred_final_p90_24h_tracker_conversions"], grouped["pred_final_p10_24h_inline_link_clicks"], 100.0)
    grouped["pred_final_p10_24h_cpm"] = safe_div(grouped["pred_final_p10_24h_spend"], grouped["pred_final_p90_24h_impressions"], 1000.0)
    grouped["pred_final_p90_24h_cpm"] = safe_div(grouped["pred_final_p90_24h_spend"], grouped["pred_final_p10_24h_impressions"], 1000.0)
    grouped["prediction_interval_width_roas"] = safe_div(
        grouped["pred_final_p90_24h_roas"] - grouped["pred_final_p10_24h_roas"],
        grouped["pred_final_24h_roas"].replace(0, np.nan),
    )
    grouped["prediction_interval_width_revenue"] = safe_div(
        grouped["pred_final_p90_24h_tracker_revenue"] - grouped["pred_final_p10_24h_tracker_revenue"],
        grouped["pred_final_24h_tracker_revenue"].replace(0, np.nan),
    )
    return grouped


def load_three_account_data() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(BACKTEST, usecols=lambda c: c in RAW_COLS, chunksize=50_000, low_memory=False):
        for col in ["account_id", "campaign_id", "adset_id", "ad_id"]:
            chunk[col] = chunk[col].map(clean_id)
        chunk = chunk[chunk["account_id"].isin(ACCOUNTS)].copy()
        if chunk.empty:
            continue
        chunk["local_date"] = pd.to_datetime(chunk["local_date"], errors="coerce")
        parts.append(chunk)
    if not parts:
        raise RuntimeError("No rows found for requested accounts in optimized winner backtest.")
    df = pd.concat(parts, ignore_index=True).sort_values(["account_id", "ad_id", "local_date"])
    if "split" not in df.columns:
        df["split"] = "all_history"
    df["date_label"] = df["local_date"].dt.strftime("%Y-%m-%d")
    return df


def load_reference_payload() -> dict[str, object]:
    import re

    text = REFERENCE_HTML.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"const RAW=(.*?);const COL=", text, re.S)
    if not match:
        raise RuntimeError(f"Could not parse RAW payload from {REFERENCE_HTML}")
    return json.loads(match.group(1))


def reference_series_fallback(ref_series: dict[str, object]) -> dict[str, object]:
    converted = dict(ref_series)
    converted["latest_date"] = str(ref_series.get("latest_d1", {}).get("date", ref_series.get("labels", [""])[-1]))
    converted["history"] = ref_series.get("history", {}).get("rows", [])
    converted["history_by_date"] = {
        label: ref_series.get("history", {}).get("rows", []) for label in ref_series.get("labels", [])
    }
    latest_d1 = ref_series.get("latest_d1", {})
    converted["latest_d1"] = {
        "date": latest_d1.get("date", converted["latest_date"]),
        "actual": latest_d1.get("actual", {}),
        "base": latest_d1.get("pred", {}),
        "final": latest_d1.get("cal", latest_d1.get("pred", {})),
        "range": {
            key: [float(val or 0), float(val or 0)]
            for key, val in latest_d1.get("cal", latest_d1.get("pred", {})).items()
        },
        "meta": {
            "confidence": "REFERENCE",
            "guardrail": "OPTIMIZED_ROW_UNAVAILABLE",
            "usage": "reference_filter_preserved",
            "roas_width": 0.0,
            "revenue_width": 0.0,
        },
    }
    converted["d1_by_date"] = {label: converted["latest_d1"] for label in ref_series.get("labels", [])}
    converted["reliability"] = "reference_fallback"
    converted["confidence"] = "REFERENCE"
    converted["guardrail"] = "OPTIMIZED_ROW_UNAVAILABLE"
    converted["usage"] = "reference_filter_preserved"
    converted["segment"] = "reference_fallback"
    converted["segment_rows"] = [
        {
            "date": row.get("date", ""),
            "segment": row.get("flag", "reference"),
            "reliability": row.get("score", ""),
            "note": row.get("note", "Reference fallback row; optimized model row unavailable."),
        }
        for row in ref_series.get("risk_rows", [])
    ]
    converted["sources"] = {
        "spend": "reference_fallback",
        "impressions": "reference_fallback",
        "clicks": "reference_fallback",
        "conversions": "reference_fallback",
        "revenue": "reference_fallback",
    }
    converted["kpis"] = {
        k: {"actual": v.get("actual", []), "base": v.get("pred", []), "final": v.get("cal", v.get("pred", []))}
        for k, v in ref_series.get("kpis", {}).items()
    }
    return converted


def load_daily_history() -> pd.DataFrame:
    usecols = [
        "entity_type",
        "date",
        "account_id",
        "campaign_id",
        "adset_id",
        "ad_id",
        "spend",
        "impressions",
        "inline_link_clicks",
        "tracker_conversions",
        "tracker_revenue",
    ]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(DAILY_SRC, usecols=lambda c: c in usecols, chunksize=50_000, low_memory=False):
        chunk = chunk[chunk["entity_type"].astype(str).str.lower().eq("ad")].copy()
        if chunk.empty:
            continue
        for col in ["account_id", "campaign_id", "adset_id", "ad_id"]:
            chunk[col] = chunk[col].map(clean_id)
        chunk = chunk[chunk["account_id"].isin(ACCOUNTS)].copy()
        if chunk.empty:
            continue
        chunk["local_date"] = pd.to_datetime(chunk["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
        for col in ["spend", "impressions", "inline_link_clicks", "tracker_conversions", "tracker_revenue"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0.0)
        parts.append(chunk[["local_date", "account_id", "campaign_id", "adset_id", "ad_id", "spend", "impressions", "inline_link_clicks", "tracker_conversions", "tracker_revenue"]])
    if not parts:
        raise RuntimeError("No daily history rows found for requested accounts.")
    daily = pd.concat(parts, ignore_index=True)
    daily = (
        daily.groupby(["local_date", "account_id", "campaign_id", "adset_id", "ad_id"], as_index=False)[
            ["spend", "impressions", "inline_link_clicks", "tracker_conversions", "tracker_revenue"]
        ]
        .sum()
        .sort_values(["account_id", "ad_id", "local_date"])
    )
    daily["kpi_ctr"] = safe_div(daily["inline_link_clicks"], daily["impressions"], 100.0)
    daily["kpi_cpm"] = safe_div(daily["spend"], daily["impressions"], 1000.0)
    daily["kpi_roas"] = safe_div(daily["tracker_revenue"], daily["spend"])
    daily["kpi_cvr"] = safe_div(daily["tracker_conversions"], daily["inline_link_clicks"], 100.0)
    daily["date_label"] = daily["local_date"].dt.strftime("%Y-%m-%d")
    return daily


def aggregate_account(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["account_id", "local_date"], as_index=False).agg(
        actual_24h_spend=("actual_24h_spend", "sum"),
        actual_24h_impressions=("actual_24h_impressions", "sum"),
        actual_24h_inline_link_clicks=("actual_24h_inline_link_clicks", "sum"),
        actual_24h_tracker_conversions=("actual_24h_tracker_conversions", "sum"),
        actual_24h_tracker_revenue=("actual_24h_tracker_revenue", "sum"),
        pred_24h_spend=("pred_24h_spend", "sum"),
        pred_24h_impressions=("pred_24h_impressions", "sum"),
        pred_24h_inline_link_clicks=("pred_24h_inline_link_clicks", "sum"),
        pred_24h_tracker_conversions=("pred_24h_tracker_conversions", "sum"),
        pred_24h_tracker_revenue=("pred_24h_tracker_revenue", "sum"),
        pred_final_24h_spend=("pred_final_24h_spend", "sum"),
        pred_final_24h_impressions=("pred_final_24h_impressions", "sum"),
        pred_final_24h_inline_link_clicks=("pred_final_24h_inline_link_clicks", "sum"),
        pred_final_24h_tracker_conversions=("pred_final_24h_tracker_conversions", "sum"),
        pred_final_24h_tracker_revenue=("pred_final_24h_tracker_revenue", "sum"),
        pred_final_p10_24h_spend=("pred_final_p10_24h_spend", "sum"),
        pred_final_p90_24h_spend=("pred_final_p90_24h_spend", "sum"),
        pred_final_p10_24h_impressions=("pred_final_p10_24h_impressions", "sum"),
        pred_final_p90_24h_impressions=("pred_final_p90_24h_impressions", "sum"),
        pred_final_p10_24h_inline_link_clicks=("pred_final_p10_24h_inline_link_clicks", "sum"),
        pred_final_p90_24h_inline_link_clicks=("pred_final_p90_24h_inline_link_clicks", "sum"),
        pred_final_p10_24h_tracker_conversions=("pred_final_p10_24h_tracker_conversions", "sum"),
        pred_final_p90_24h_tracker_conversions=("pred_final_p90_24h_tracker_conversions", "sum"),
        pred_final_p10_24h_tracker_revenue=("pred_final_p10_24h_tracker_revenue", "sum"),
        pred_final_p90_24h_tracker_revenue=("pred_final_p90_24h_tracker_revenue", "sum"),
        kpi_reliability_flag=("kpi_reliability_flag", lambda s: "OK" if (s == "OK").mean() >= 0.5 else "MIXED"),
        prediction_segment=("prediction_segment", majority_label),
        forecast_confidence=("forecast_confidence", majority_label),
        production_guardrail_flag=("production_guardrail_flag", majority_label),
        forecast_usage_recommendation=("forecast_usage_recommendation", majority_label),
    )
    grouped["campaign_id"] = "ALL"
    grouped["adset_id"] = "ALL"
    grouped["ad_id"] = "ALL"
    grouped["split"] = "aggregate"
    grouped["date_label"] = grouped["local_date"].dt.strftime("%Y-%m-%d")

    for prefix in ["actual_24h", "pred_24h", "pred_final_24h"]:
        grouped[f"{prefix}_ctr"] = safe_div(grouped[f"{prefix}_inline_link_clicks"], grouped[f"{prefix}_impressions"], 100.0)
        grouped[f"{prefix}_cpm"] = safe_div(grouped[f"{prefix}_spend"], grouped[f"{prefix}_impressions"], 1000.0)
        grouped[f"{prefix}_roas"] = safe_div(grouped[f"{prefix}_tracker_revenue"], grouped[f"{prefix}_spend"])
        grouped[f"{prefix}_cvr"] = safe_div(grouped[f"{prefix}_tracker_conversions"], grouped[f"{prefix}_inline_link_clicks"], 100.0)

    grouped["kpi_ctr"] = grouped["actual_24h_ctr"]
    grouped["kpi_cpm"] = grouped["actual_24h_cpm"]
    grouped["kpi_roas"] = grouped["actual_24h_roas"]
    grouped["kpi_cvr"] = grouped["actual_24h_cvr"]
    grouped["pred_24h_ctr"] = grouped["pred_24h_ctr"]
    grouped["pred_24h_cpm"] = grouped["pred_24h_cpm"]
    grouped["pred_24h_roas"] = grouped["pred_24h_roas"]
    grouped["pred_24h_cvr"] = grouped["pred_24h_cvr"]
    grouped = add_range_kpis(grouped)
    return grouped


def aggregate_ad_by_day(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["account_id", "campaign_id", "adset_id", "ad_id", "local_date"], as_index=False).agg(
        actual_24h_spend=("actual_24h_spend", "sum"),
        actual_24h_impressions=("actual_24h_impressions", "sum"),
        actual_24h_inline_link_clicks=("actual_24h_inline_link_clicks", "sum"),
        actual_24h_tracker_conversions=("actual_24h_tracker_conversions", "sum"),
        actual_24h_tracker_revenue=("actual_24h_tracker_revenue", "sum"),
        pred_24h_spend=("pred_24h_spend", "sum"),
        pred_24h_impressions=("pred_24h_impressions", "sum"),
        pred_24h_inline_link_clicks=("pred_24h_inline_link_clicks", "sum"),
        pred_24h_tracker_conversions=("pred_24h_tracker_conversions", "sum"),
        pred_24h_tracker_revenue=("pred_24h_tracker_revenue", "sum"),
        pred_final_24h_spend=("pred_final_24h_spend", "sum"),
        pred_final_24h_impressions=("pred_final_24h_impressions", "sum"),
        pred_final_24h_inline_link_clicks=("pred_final_24h_inline_link_clicks", "sum"),
        pred_final_24h_tracker_conversions=("pred_final_24h_tracker_conversions", "sum"),
        pred_final_24h_tracker_revenue=("pred_final_24h_tracker_revenue", "sum"),
        pred_final_p10_24h_spend=("pred_final_p10_24h_spend", "sum"),
        pred_final_p90_24h_spend=("pred_final_p90_24h_spend", "sum"),
        pred_final_p10_24h_impressions=("pred_final_p10_24h_impressions", "sum"),
        pred_final_p90_24h_impressions=("pred_final_p90_24h_impressions", "sum"),
        pred_final_p10_24h_inline_link_clicks=("pred_final_p10_24h_inline_link_clicks", "sum"),
        pred_final_p90_24h_inline_link_clicks=("pred_final_p90_24h_inline_link_clicks", "sum"),
        pred_final_p10_24h_tracker_conversions=("pred_final_p10_24h_tracker_conversions", "sum"),
        pred_final_p90_24h_tracker_conversions=("pred_final_p90_24h_tracker_conversions", "sum"),
        pred_final_p10_24h_tracker_revenue=("pred_final_p10_24h_tracker_revenue", "sum"),
        pred_final_p90_24h_tracker_revenue=("pred_final_p90_24h_tracker_revenue", "sum"),
        kpi_reliability_flag=("kpi_reliability_flag", lambda s: "OK" if (s == "OK").mean() >= 0.5 else "MIXED"),
        prediction_segment=("prediction_segment", majority_label),
        forecast_confidence=("forecast_confidence", majority_label),
        production_guardrail_flag=("production_guardrail_flag", majority_label),
        forecast_usage_recommendation=("forecast_usage_recommendation", majority_label),
    )
    grouped["split"] = "daily_ad_aggregate"
    grouped["date_label"] = grouped["local_date"].dt.strftime("%Y-%m-%d")
    for prefix in ["actual_24h", "pred_24h", "pred_final_24h"]:
        grouped[f"{prefix}_ctr"] = safe_div(grouped[f"{prefix}_inline_link_clicks"], grouped[f"{prefix}_impressions"], 100.0)
        grouped[f"{prefix}_cpm"] = safe_div(grouped[f"{prefix}_spend"], grouped[f"{prefix}_impressions"], 1000.0)
        grouped[f"{prefix}_roas"] = safe_div(grouped[f"{prefix}_tracker_revenue"], grouped[f"{prefix}_spend"])
        grouped[f"{prefix}_cvr"] = safe_div(grouped[f"{prefix}_tracker_conversions"], grouped[f"{prefix}_inline_link_clicks"], 100.0)
    grouped["kpi_ctr"] = grouped["actual_24h_ctr"]
    grouped["kpi_cpm"] = grouped["actual_24h_cpm"]
    grouped["kpi_roas"] = grouped["actual_24h_roas"]
    grouped["kpi_cvr"] = grouped["actual_24h_cvr"]
    grouped["pred_24h_ctr"] = grouped["pred_24h_ctr"]
    grouped["pred_24h_cpm"] = grouped["pred_24h_cpm"]
    grouped["pred_24h_roas"] = grouped["pred_24h_roas"]
    grouped["pred_24h_cvr"] = grouped["pred_24h_cvr"]
    grouped = add_range_kpis(grouped)
    return grouped


def build_history_table(history_df: pd.DataFrame, d1_date: pd.Timestamp) -> list[dict[str, object]]:
    days = [d1_date - pd.Timedelta(days=offset) for offset in range(7, 0, -1)]
    labels = ["D-6", "D-5", "D-4", "D-3", "D-2", "D-1", "D0"]
    output = []
    for label, day in zip(labels, days):
        match = history_df[history_df["local_date"].eq(day)]
        if match.empty:
            rec = {
                "date_label": day.strftime("%Y-%m-%d"),
                "spend": 0.0,
                "impressions": 0.0,
                "inline_link_clicks": 0.0,
                "tracker_conversions": 0.0,
                "tracker_revenue": 0.0,
                "kpi_ctr": 0.0,
                "kpi_cpm": 0.0,
                "kpi_roas": 0.0,
                "kpi_cvr": 0.0,
            }
        else:
            rec = match.iloc[0]
        output.append(
            {
                "label": label,
                "date": str(rec["date_label"]),
                "spend": float(rec["spend"]),
                "impressions": float(rec["impressions"]),
                "clicks": float(rec["inline_link_clicks"]),
                "conversions": float(rec["tracker_conversions"]),
                "revenue": float(rec["tracker_revenue"]),
                "ctr": float(rec["kpi_ctr"]),
                "cpm": float(rec["kpi_cpm"]),
                "roas": float(rec["kpi_roas"]),
                "cvr": float(rec["kpi_cvr"]),
            }
        )
    return output


def series_from_rows(rows: pd.DataFrame, history_df: pd.DataFrame) -> dict[str, object]:
    rows = rows.sort_values("local_date")
    latest = rows.iloc[-1]
    labels = rows["date_label"].astype(str).tolist()
    def d1_payload(row: pd.Series) -> dict[str, object]:
        return {
            "date": str(row["date_label"]),
            "actual": {
                "spend": float(row["actual_24h_spend"]),
                "impressions": float(row["actual_24h_impressions"]),
                "clicks": float(row["actual_24h_inline_link_clicks"]),
                "conversions": float(row["actual_24h_tracker_conversions"]),
                "revenue": float(row["actual_24h_tracker_revenue"]),
                "ctr": float(row["kpi_ctr"]),
                "cpm": float(row["kpi_cpm"]),
                "roas": float(row["kpi_roas"]),
                "cvr": float(row["kpi_cvr"]),
            },
            "base": {
                "spend": float(row["pred_24h_spend"]),
                "impressions": float(row["pred_24h_impressions"]),
                "clicks": float(row["pred_24h_inline_link_clicks"]),
                "conversions": float(row["pred_24h_tracker_conversions"]),
                "revenue": float(row["pred_24h_tracker_revenue"]),
                "ctr": float(row["pred_24h_ctr"]),
                "cpm": float(row["pred_24h_cpm"]),
                "roas": float(row["pred_24h_roas"]),
                "cvr": float(row["pred_24h_cvr"]),
            },
            "final": {
                "spend": float(row["pred_final_24h_spend"]),
                "impressions": float(row["pred_final_24h_impressions"]),
                "clicks": float(row["pred_final_24h_inline_link_clicks"]),
                "conversions": float(row["pred_final_24h_tracker_conversions"]),
                "revenue": float(row["pred_final_24h_tracker_revenue"]),
                "ctr": float(row["pred_final_24h_ctr"]),
                "cpm": float(row["pred_final_24h_cpm"]),
                "roas": float(row["pred_final_24h_roas"]),
                "cvr": float(row["pred_final_24h_cvr"]),
            },
            "range": {
                "spend": [float(row["pred_final_p10_24h_spend"]), float(row["pred_final_p90_24h_spend"])],
                "impressions": [float(row["pred_final_p10_24h_impressions"]), float(row["pred_final_p90_24h_impressions"])],
                "clicks": [float(row["pred_final_p10_24h_inline_link_clicks"]), float(row["pred_final_p90_24h_inline_link_clicks"])],
                "conversions": [float(row["pred_final_p10_24h_tracker_conversions"]), float(row["pred_final_p90_24h_tracker_conversions"])],
                "revenue": [float(row["pred_final_p10_24h_tracker_revenue"]), float(row["pred_final_p90_24h_tracker_revenue"])],
                "ctr": [float(row["pred_final_p10_24h_ctr"]), float(row["pred_final_p90_24h_ctr"])],
                "cpm": [float(row["pred_final_p10_24h_cpm"]), float(row["pred_final_p90_24h_cpm"])],
                "roas": [float(row["pred_final_p10_24h_roas"]), float(row["pred_final_p90_24h_roas"])],
                "cvr": [float(row["pred_final_p10_24h_cvr"]), float(row["pred_final_p90_24h_cvr"])],
            },
            "meta": {
                "confidence": str(row.get("forecast_confidence", "LOW")),
                "guardrail": str(row.get("production_guardrail_flag", "UNKNOWN")),
                "usage": str(row.get("forecast_usage_recommendation", "review_manually")),
                "roas_width": float(row.get("prediction_interval_width_roas", 0.0)),
                "revenue_width": float(row.get("prediction_interval_width_revenue", 0.0)),
            },
        }
    history_by_date = {
        str(row["date_label"]): build_history_table(history_df, pd.Timestamp(row["local_date"]))
        for _, row in rows.iterrows()
    }
    d1_by_date = {str(row["date_label"]): d1_payload(row) for _, row in rows.iterrows()}
    return {
        "labels": labels,
        "latest_date": str(latest["date_label"]),
        "history": build_history_table(history_df, pd.Timestamp(latest["local_date"])),
        "history_by_date": history_by_date,
        "latest_d1": d1_payload(latest),
        "d1_by_date": d1_by_date,
        "reliability": str(latest.get("kpi_reliability_flag", "OK")),
        "confidence": str(latest.get("forecast_confidence", "LOW")),
        "guardrail": str(latest.get("production_guardrail_flag", "UNKNOWN")),
        "usage": str(latest.get("forecast_usage_recommendation", "review_manually")),
        "segment": str(latest.get("prediction_segment", "stable")),
        "segment_rows": [
            {
                "date": str(row.date_label),
                "segment": str(getattr(row, "prediction_segment", "stable")),
                "reliability": str(getattr(row, "kpi_reliability_flag", "OK")),
                "confidence": str(getattr(row, "forecast_confidence", "LOW")),
                "guardrail": str(getattr(row, "production_guardrail_flag", "UNKNOWN")),
                "usage": str(getattr(row, "forecast_usage_recommendation", "review_manually")),
                "note": "Use prediction range / manual review" if str(getattr(row, "production_guardrail_flag", "OK")) != "OK" else "Point prediction usable",
            }
            for row in rows.tail(10).itertuples(index=False)
        ],
        "sources": {
            "spend": str(latest.get("final_source_24h_spend", "mixed")),
            "impressions": str(latest.get("final_source_24h_impressions", "mixed")),
            "clicks": str(latest.get("final_source_24h_inline_link_clicks", "mixed")),
            "conversions": str(latest.get("final_source_24h_tracker_conversions", "mixed")),
            "revenue": str(latest.get("final_source_24h_tracker_revenue", "mixed")),
        },
        "kpis": {
            "ctr": {
                "actual": round_list(rows["kpi_ctr"]),
                "base": round_list(rows["pred_24h_ctr"]),
                "final": round_list(rows["pred_final_24h_ctr"]),
            },
            "cpm": {
                "actual": round_list(rows["kpi_cpm"]),
                "base": round_list(rows["pred_24h_cpm"]),
                "final": round_list(rows["pred_final_24h_cpm"]),
            },
            "roas": {
                "actual": round_list(rows["kpi_roas"]),
                "base": round_list(rows["pred_24h_roas"]),
                "final": round_list(rows["pred_final_24h_roas"]),
            },
            "cvr": {
                "actual": round_list(rows["kpi_cvr"]),
                "base": round_list(rows["pred_24h_cvr"]),
                "final": round_list(rows["pred_final_24h_cvr"]),
            },
        },
        "raw": {
            "spend": {"actual": round_list(rows["actual_24h_spend"]), "base": round_list(rows["pred_24h_spend"]), "final": round_list(rows["pred_final_24h_spend"])},
            "impressions": {"actual": round_list(rows["actual_24h_impressions"]), "base": round_list(rows["pred_24h_impressions"]), "final": round_list(rows["pred_final_24h_impressions"])},
            "clicks": {"actual": round_list(rows["actual_24h_inline_link_clicks"]), "base": round_list(rows["pred_24h_inline_link_clicks"]), "final": round_list(rows["pred_final_24h_inline_link_clicks"])},
            "conversions": {"actual": round_list(rows["actual_24h_tracker_conversions"]), "base": round_list(rows["pred_24h_tracker_conversions"]), "final": round_list(rows["pred_final_24h_tracker_conversions"])},
            "revenue": {"actual": round_list(rows["actual_24h_tracker_revenue"]), "base": round_list(rows["pred_24h_tracker_revenue"]), "final": round_list(rows["pred_final_24h_tracker_revenue"])},
        },
    }


def comparison_cards() -> list[dict[str, object]]:
    comp = pd.read_csv(COMPARISON)
    keep = [
        "target_24h_spend",
        "target_24h_tracker_revenue",
        "target_24h_tracker_conversions",
        "target_24h_roas",
        "target_24h_profit",
        "target_24h_ctr",
        "target_24h_cvr",
        "target_24h_cpm",
    ]
    comp = comp[comp["target_base"].isin(keep)].copy()
    return [
        {
            "target": row.target_base.replace("target_24h_", ""),
            "previous": round(float(row.wmape_previous), 4),
            "optimized": round(float(row.wmape_optimized), 4),
            "delta": round(float(row.wmape_delta), 4),
        }
        for row in comp.itertuples(index=False)
    ]


def build_payload() -> dict[str, object]:
    df = load_three_account_data()
    daily_history = load_daily_history()
    reference_raw = load_reference_payload()
    reference_filters = {str(acc): reference_raw["data"][str(acc)]["ad_list"] for acc in reference_raw["accounts"]}
    payload = {"accounts": ACCOUNTS, "comparison": comparison_cards(), "data": {}}
    for acc in ACCOUNTS:
        acc_df = df[df["account_id"].eq(acc)].copy()
        acc_history = (
            daily_history[daily_history["account_id"].eq(acc)]
            .groupby(["local_date"], as_index=False)[["spend", "impressions", "inline_link_clicks", "tracker_conversions", "tracker_revenue"]]
            .sum()
        )
        acc_history["kpi_ctr"] = safe_div(acc_history["inline_link_clicks"], acc_history["impressions"], 100.0)
        acc_history["kpi_cpm"] = safe_div(acc_history["spend"], acc_history["impressions"], 1000.0)
        acc_history["kpi_roas"] = safe_div(acc_history["tracker_revenue"], acc_history["spend"])
        acc_history["kpi_cvr"] = safe_div(acc_history["tracker_conversions"], acc_history["inline_link_clicks"], 100.0)
        acc_history["date_label"] = acc_history["local_date"].dt.strftime("%Y-%m-%d")
        ad_list = reference_filters.get(acc, [{"id": "ALL", "label": "All Ads (account aggregate)"}])
        series = {"ALL": series_from_rows(aggregate_account(acc_df), acc_history)}
        output_ad_list = [ad_list[0]]
        for ad in ad_list[1:]:
            ad_id = str(ad["id"])
            ad_rows = acc_df[acc_df["ad_id"].eq(ad_id)].copy()
            if ad_rows.empty:
                output_ad_list.append({"id": ad_id, "label": str(ad["label"])})
                series[ad_id] = reference_series_fallback(reference_raw["data"][acc]["series"][ad_id])
                continue
            ad_history = daily_history[(daily_history["account_id"].eq(acc)) & (daily_history["ad_id"].eq(ad_id))].copy()
            output_ad_list.append({"id": ad_id, "label": str(ad["label"])})
            series[ad_id] = series_from_rows(aggregate_ad_by_day(ad_rows), ad_history)
        payload["data"][acc] = {"ad_list": output_ad_list, "series": series}
    return payload


def main() -> None:
    raw = build_payload()
    html = f"""<!doctype html><html lang="en"><head><meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Adunbox 24H Optimized KPI Backtest</title><script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}}.hdr{{background:linear-gradient(135deg,#111827,#172554 50%,#0f172a);padding:22px 30px;border-bottom:1px solid #263348}}.hdr h1{{font-size:1.45rem;color:#a5b4fc}}.hdr p{{color:#94a3b8;font-size:.84rem;margin-top:5px;max-width:1180px;line-height:1.45}}.wrap{{max-width:1500px;margin:0 auto;padding:18px 24px}}.tabs{{display:flex;gap:4px;margin:16px 0 0;flex-wrap:wrap}}.tab{{padding:15px 24px;border-radius:10px 10px 0 0;font-size:1rem;font-weight:800;cursor:pointer;color:#7f95bd;border:1px solid transparent;border-bottom:none;letter-spacing:.01em}}.tab:hover{{color:#dbeafe;background:#1e293b99}}.tab.active{{background:#1e293b;color:#dbeafe;border-color:#334155;border-bottom-color:#1e293b;box-shadow:0 -1px 0 #475569 inset}}.panel{{display:none;background:#1e293b;border:1px solid #334155;border-radius:0 12px 12px 12px;padding:18px 20px}}.panel.active{{display:block}}.bar{{display:flex;align-items:center;gap:14px;flex-wrap:wrap;background:#0f172a;border:1px solid #334155;border-radius:12px;padding:14px 18px;margin-bottom:14px}}label,.muted{{font-size:.82rem;color:#b7c4df;font-weight:800}}select{{background:#1e293b;border:1px solid #334155;color:#f8fafc;padding:10px 16px;border-radius:9px;min-width:325px;max-width:620px;font-size:1rem;box-shadow:0 0 0 1px #0f172a inset}}select:focus{{outline:none;border-color:#647fff;box-shadow:0 0 0 2px #2563eb55}}.btn{{background:#1e293b;border:1px solid #334155;color:#94a3b8;padding:7px 16px;border-radius:8px;cursor:pointer;font-size:.8rem;font-weight:700}}.btn.active{{background:#312e81;border-color:#6366f1;color:#c7d2fe}}.stats,.cmp{{display:grid;grid-template-columns:repeat(5,minmax(130px,1fr));gap:9px;margin:12px 0}}.stat,.metric{{background:#0f172a;border:1px solid #334155;border-radius:9px;padding:10px 13px}}.stat .lbl,.metric .lbl{{font-size:.67rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em}}.stat .val,.metric .val{{font-size:1.08rem;font-weight:800;margin-top:3px}}.metric .good{{color:#86efac}}.metric .bad{{color:#fecaca}}.alert{{background:#172554;border:1px solid #1d4ed8;color:#bfdbfe;border-radius:9px;padding:10px 13px;font-size:.82rem;line-height:1.45;margin-bottom:13px}}.legend{{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0 8px}}.li{{display:flex;align-items:center;gap:6px;font-size:.76rem;color:#94a3b8}}.dot{{width:9px;height:9px;border-radius:50%}}.gtitle{{font-size:.86rem;font-weight:900;color:#8b95ff;text-transform:uppercase;letter-spacing:.12em;margin:18px 0 10px;display:flex;align-items:center;gap:9px}}.gtitle::before{{content:'';width:4px;height:15px;border-radius:3px;background:#6366f1}}.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}}.card{{background:#0f172a;border:1px solid #334155;border-radius:11px;padding:13px 15px}}.card h3{{font-size:.78rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px}}.card canvas{{max-height:230px}}.tbl-wrap{{overflow-x:auto;border:1px solid #1e293b;border-radius:10px;margin-top:12px}}table{{width:100%;border-collapse:collapse;font-size:.79rem}}th{{background:#111827;color:#94a3b8;text-align:left;padding:9px 10px;white-space:nowrap}}td{{padding:8px 10px;border-top:1px solid #1e293b;white-space:nowrap}}tr:hover td{{background:#1e293b66}}@media(max-width:960px){{.grid{{grid-template-columns:1fr}}.stats,.cmp{{grid-template-columns:repeat(2,1fr)}}select{{min-width:180px}}}}
</style></head><body><div class="hdr"><h1>Adunbox - 24H Optimized Actual vs Predicted KPI Dashboard</h1><p>Same three-account visual review, now using the optimized full-dataset model output. Charts compare Actual vs Base Predicted vs Optimized Final for CTR, CPM, ROAS, and CVR.</p></div><div class="wrap"><div class="gtitle">Overall Previous vs Optimized WMAPE</div><div class="cmp" id="cmp"></div><div class="tabs" id="tabs"></div><div id="panels"></div></div>
<script>
const RAW={json.dumps(raw, separators=(",", ":"))};const COL={{actual:'#6366f1',base:'#a5b4fc',final:'#22d3ee'}};const CH={{}},STATE={{}};
function fmt(v,cur=false,dec=2){{if(v===null||v===undefined||Number.isNaN(Number(v)))return'N/A';const s=Number(v).toLocaleString('en-US',{{maximumFractionDigits:dec}});return cur?'$'+s:s}}function ds(label,data,color,dash=[]){{return{{label,data,borderColor:color,backgroundColor:color+'22',borderWidth:2,borderDash:dash,pointRadius:2.5,pointHoverRadius:5,tension:.25,spanGaps:true}}}}function slice(labels,n){{if(!n)return[0,labels.length];const take=Math.min(labels.length,n);return[labels.length-take,labels.length]}}function draw(id,labels,sets){{if(CH[id])CH[id].destroy();CH[id]=new Chart(document.getElementById(id),{{type:'line',data:{{labels,datasets:sets}},options:{{responsive:true,maintainAspectRatio:true,interaction:{{mode:'index',intersect:false}},plugins:{{legend:{{display:false}},tooltip:{{backgroundColor:'#1e293b',borderColor:'#334155',borderWidth:1,titleColor:'#cbd5e1',bodyColor:'#e2e8f0'}}}},scales:{{x:{{grid:{{color:'#1e293b'}},ticks:{{color:'#64748b',font:{{size:9}},maxRotation:45,minRotation:30}}}},y:{{grid:{{color:'#1e293b'}},ticks:{{color:'#64748b',font:{{size:9}}}}}}}}}}}})}}
function active(acc){{return RAW.data[acc].series[STATE[acc].ad]}}function selectedDate(acc){{const s=active(acc);return STATE[acc].date||s.latest_date}}function render(acc){{const s=active(acc),st=STATE[acc],[a,b]=slice(s.labels,st.days),labels=s.labels.slice(a,b);['ctr','cpm','roas','cvr'].forEach(k=>{{const kp=s.kpis[k],sets=[];if(st.mode==='both'||st.mode==='actual')sets.push(ds('Actual',kp.actual.slice(a,b),COL.actual));if(st.mode==='both'||st.mode==='base')sets.push(ds('Base Predicted',kp.base.slice(a,b),COL.base,[5,4]));if(st.mode==='both'||st.mode==='final')sets.push(ds('Optimized Final',kp.final.slice(a,b),COL.final,[2,3]));draw(`ch-${{k}}-${{acc}}`,labels,sets)}});stats(acc);tables(acc)}}function stats(acc){{const s=active(acc),dt=selectedDate(acc),i=Math.max(0,s.labels.indexOf(dt)),d=s.d1_by_date[dt]||s.latest_d1;document.getElementById(`rel-${{acc}}`).textContent=s.reliability;document.getElementById(`seg-${{acc}}`).textContent=s.segment;document.getElementById(`conf-${{acc}}`).textContent=s.confidence;document.getElementById(`guard-${{acc}}`).textContent=s.guardrail;document.getElementById(`usage-${{acc}}`).textContent=s.usage;document.getElementById(`ctr-${{acc}}`).textContent=fmt(s.kpis.ctr.final[i]);document.getElementById(`cpm-${{acc}}`).textContent=fmt(s.kpis.cpm.final[i],true);document.getElementById(`roas-${{acc}}`).textContent=fmt(s.kpis.roas.final[i]);document.getElementById(`cvr-${{acc}}`).textContent=fmt(s.kpis.cvr.final[i]);document.getElementById(`note-${{acc}}`).innerHTML=`<b>Selected forecast date:</b> ${{dt}}. <b>Winner sources:</b> spend=${{s.sources.spend}}, impressions=${{s.sources.impressions}}, clicks=${{s.sources.clicks}}, conversions=${{s.sources.conversions}}, revenue=${{s.sources.revenue}}. <b>Segment:</b> ${{s.segment}}. <b>Forecast confidence:</b> ${{s.confidence}}. <b>Guardrail:</b> ${{s.guardrail}}. <b>Usage:</b> ${{s.usage}}. <b>ROAS range:</b> ${{fmt(d.range.roas[0])}} - ${{fmt(d.range.roas[1])}}. <b>Revenue range:</b> ${{fmt(d.range.revenue[0],true)}} - ${{fmt(d.range.revenue[1],true)}}.`}}
function tables(acc){{const s=active(acc),dt=selectedDate(acc),h=s.history_by_date[dt]||s.history,d=s.d1_by_date[dt]||s.latest_d1;document.getElementById(`riskbody-${{acc}}`).innerHTML=s.segment_rows.map(r=>`<tr><td>${{r.date}}</td><td><b>${{r.segment}}</b></td><td>${{r.reliability}}</td><td>${{r.note}}</td></tr>`).join('');document.getElementById(`hhead-${{acc}}`).innerHTML='<tr><th>Day</th><th>Date</th><th>Value Type</th><th>Spend</th><th>Impressions</th><th>Clicks</th><th>Conversions</th><th>Revenue</th><th>CTR</th><th>CPM</th><th>ROAS</th><th>CVR</th></tr>';let histRows=h.map(r=>`<tr><td><b>${{r.label}}</b></td><td>${{r.date}}</td><td>Actual</td><td>${{fmt(r.spend,true)}}</td><td>${{fmt(r.impressions)}}</td><td>${{fmt(r.clicks)}}</td><td>${{fmt(r.conversions)}}</td><td>${{fmt(r.revenue,true)}}</td><td>${{fmt(r.ctr)}}</td><td>${{fmt(r.cpm,true)}}</td><td>${{fmt(r.roas)}}</td><td>${{fmt(r.cvr)}}</td></tr>`).join('');let d1Rows=['actual','base','final'].map(t=>`<tr><td><b>D1</b></td><td>${{d.date}}</td><td>${{t==='actual'?'Actual':t==='base'?'Base Predicted':'Optimized Final'}}</td><td>${{fmt(d[t].spend,true)}}</td><td>${{fmt(d[t].impressions)}}</td><td>${{fmt(d[t].clicks)}}</td><td>${{fmt(d[t].conversions)}}</td><td>${{fmt(d[t].revenue,true)}}</td><td>${{fmt(d[t].ctr)}}</td><td>${{fmt(d[t].cpm,true)}}</td><td>${{fmt(d[t].roas)}}</td><td>${{fmt(d[t].cvr)}}</td></tr>`).join('');let rangeRow=`<tr><td><b>D1</b></td><td>${{d.date}}</td><td>Range</td><td>${{fmt(d.range.spend[0],true)}} → ${{fmt(d.range.spend[1],true)}}</td><td>${{fmt(d.range.impressions[0])}} → ${{fmt(d.range.impressions[1])}}</td><td>${{fmt(d.range.clicks[0])}} → ${{fmt(d.range.clicks[1])}}</td><td>${{fmt(d.range.conversions[0])}} → ${{fmt(d.range.conversions[1])}}</td><td>${{fmt(d.range.revenue[0],true)}} → ${{fmt(d.range.revenue[1],true)}}</td><td>${{fmt(d.range.ctr[0])}} → ${{fmt(d.range.ctr[1])}}</td><td>${{fmt(d.range.cpm[0],true)}} → ${{fmt(d.range.cpm[1],true)}}</td><td>${{fmt(d.range.roas[0])}} → ${{fmt(d.range.roas[1])}}</td><td>${{fmt(d.range.cvr[0])}} → ${{fmt(d.range.cvr[1])}}</td></tr>`;document.getElementById(`hbody-${{acc}}`).innerHTML=histRows+d1Rows+rangeRow;let i=Math.max(0,s.labels.indexOf(dt));document.getElementById(`d1body-${{acc}}`).innerHTML=['ctr','cpm','roas','cvr'].map(k=>`<tr><td><b>${{k.toUpperCase()}}</b></td><td>${{s.labels[i]}}</td><td>${{fmt(s.kpis[k].actual[i],k==='cpm')}}</td><td>${{fmt(s.kpis[k].base[i],k==='cpm')}}</td><td>${{fmt(s.kpis[k].final[i],k==='cpm')}}</td><td>${{fmt(s.kpis[k].final[i]-s.kpis[k].actual[i],false,3)}}</td></tr>`).join('')}}
function switchAcc(acc,el){{document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));el.classList.add('active');document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));document.getElementById('p-'+acc).classList.add('active')}}function refreshDates(acc){{const s=active(acc),sel=document.getElementById(`date-${{acc}}`);if(!sel)return;sel.innerHTML=s.labels.map(d=>`<option value="${{d}}">${{d}}</option>`).join('');STATE[acc].date=s.latest_date;sel.value=s.latest_date}}function onAd(acc){{STATE[acc].ad=document.getElementById(`ad-${{acc}}`).value;refreshDates(acc);render(acc)}}function onDate(acc){{STATE[acc].date=document.getElementById(`date-${{acc}}`).value;stats(acc);tables(acc)}}function onMode(acc){{STATE[acc].mode=document.getElementById(`mode-${{acc}}`).value;render(acc)}}function onDays(acc,days,btn){{STATE[acc].days=days;document.getElementById(`days-${{acc}}`).querySelectorAll('.btn').forEach(x=>x.classList.remove('active'));btn.classList.add('active');render(acc)}}
function panel(acc){{STATE[acc]={{ad:'ALL',mode:'both',days:14,date:null}};const opts=RAW.data[acc].ad_list.map(a=>`<option value="${{a.id}}">${{a.label}}</option>`).join('');document.getElementById('p-'+acc).innerHTML=`<div class="bar"><label>Select Ad</label><select id="ad-${{acc}}" onchange="onAd('${{acc}}')">${{opts}}</select><label>Forecast Date</label><select id="date-${{acc}}" onchange="onDate('${{acc}}')"></select><label>View</label><select id="mode-${{acc}}" onchange="onMode('${{acc}}')"><option value="both">Actual + Base + Optimized</option><option value="actual">Actual Only</option><option value="base">Base Predicted Only</option><option value="final">Optimized Final Only</option></select><span class="muted">Chart Window</span><span id="days-${{acc}}"><button class="btn active" onclick="onDays('${{acc}}',14,this)">Last 14</button> <button class="btn" onclick="onDays('${{acc}}',0,this)">All</button></span></div><div class="stats"><div class="stat"><div class="lbl">Reliability</div><div class="val" id="rel-${{acc}}">-</div></div><div class="stat"><div class="lbl">Segment</div><div class="val" id="seg-${{acc}}">-</div></div><div class="stat"><div class="lbl">Forecast Confidence</div><div class="val" id="conf-${{acc}}">-</div></div><div class="stat"><div class="lbl">Production Guardrail</div><div class="val" id="guard-${{acc}}">-</div></div><div class="stat"><div class="lbl">Usage Recommendation</div><div class="val" id="usage-${{acc}}">-</div></div><div class="stat"><div class="lbl">Latest Opt CTR</div><div class="val" id="ctr-${{acc}}">-</div></div><div class="stat"><div class="lbl">Latest Opt CPM</div><div class="val" id="cpm-${{acc}}">-</div></div><div class="stat"><div class="lbl">Latest Opt ROAS</div><div class="val" id="roas-${{acc}}">-</div></div><div class="stat"><div class="lbl">Latest Opt CVR</div><div class="val" id="cvr-${{acc}}">-</div></div></div><div class="alert" id="note-${{acc}}"></div><div class="gtitle">Segment / Edge Risk Analysis - Recent Backtest Dates</div><div class="tbl-wrap"><table><thead><tr><th>Date</th><th>Segment</th><th>Reliability</th><th>Note</th></tr></thead><tbody id="riskbody-${{acc}}"></tbody></table></div><div class="legend"><div class="li"><span class="dot" style="background:${{COL.actual}}"></span>Actual</div><div class="li"><span class="dot"style="background:${{COL.base}}"></span>Base Predicted</div><div class="li"><span class="dot" style="background:${{COL.final}}"></span>Optimized Final</div></div><div class="gtitle">Optimized KPI Backtest Charts - CTR · CPM · ROAS · CVR</div><div class="grid"><div class="card"><h3>CTR (%)</h3><canvas id="ch-ctr-${{acc}}"></canvas></div><div class="card"><h3>CPM ($)</h3><canvas id="ch-cpm-${{acc}}"></canvas></div><div class="card"><h3>ROAS</h3><canvas id="ch-roas-${{acc}}"></canvas></div><div class="card"><h3>CVR (%)</h3><canvas id="ch-cvr-${{acc}}"></canvas></div></div><div class="gtitle">Latest D-6 to D0 + D1 Table Used For D1 Context</div><div class="tbl-wrap"><table><thead id="hhead-${{acc}}"></thead><tbody id="hbody-${{acc}}"></tbody></table></div><div class="gtitle">Latest D1 KPI Actual vs Base vs Optimized</div><div class="tbl-wrap"><table><thead><tr><th>KPI</th><th>D1 Date</th><th>Actual</th><th>Base Predicted</th><th>Optimized Final</th><th>Optimized Gap</th></tr></thead><tbody id="d1body-${{acc}}"></tbody></table></div>`;refreshDates(acc);render(acc)}}
function init(){{document.getElementById('cmp').innerHTML=RAW.comparison.map(r=>`<div class="metric"><div class="lbl">${{r.target}}</div><div class="val ${{r.delta<=0?'good':'bad'}}">${{fmt(r.previous)}} → ${{fmt(r.optimized)}}</div><div class="lbl">delta ${{fmt(r.delta,false,4)}}</div></div>`).join('');const tabs=document.getElementById('tabs'),panels=document.getElementById('panels');RAW.accounts.forEach((acc,i)=>{{tabs.insertAdjacentHTML('beforeend',`<div class="tab ${{i===0?'active':''}}" onclick="switchAcc('${{acc}}',this)">Account ${{acc}}</div>`);panels.insertAdjacentHTML('beforeend',`<div class="panel ${{i===0?'active':''}}" id="p-${{acc}}"></div>`);panel(acc)}})}}init();
</script></body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
