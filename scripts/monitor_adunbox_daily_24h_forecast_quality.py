from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE_DIR / "outputs" / "adunbox_daily_24h_full_db_production__backtest.csv"
DEFAULT_OUTPUT = BASE_DIR / "outputs" / "adunbox_daily_24h_forecast_quality_monitor.csv"
DEFAULT_SUMMARY = BASE_DIR / "outputs" / "adunbox_daily_24h_forecast_quality_monitor__summary.txt"

METRICS = {
    "spend": ("actual_24h_spend", "recommended_24h_spend", "pred_calibrated_24h_spend", "pred_24h_spend"),
    "impressions": ("actual_24h_impressions", "recommended_24h_impressions", "pred_calibrated_24h_impressions", "pred_24h_impressions"),
    "clicks": ("actual_24h_inline_link_clicks", "recommended_24h_inline_link_clicks", "pred_calibrated_24h_inline_link_clicks", "pred_24h_inline_link_clicks"),
    "conversions": ("actual_24h_tracker_conversions", "recommended_24h_tracker_conversions", "pred_calibrated_24h_tracker_conversions", "pred_24h_tracker_conversions"),
    "revenue": ("actual_24h_tracker_revenue", "recommended_24h_tracker_revenue", "pred_calibrated_24h_tracker_revenue", "pred_24h_tracker_revenue"),
    "roas": ("kpi_roas", "recommended_24h_roas", "pred_calibrated_24h_roas", "pred_24h_roas"),
    "ctr": ("kpi_ctr", "recommended_24h_ctr", "pred_calibrated_24h_ctr", "pred_24h_ctr"),
    "cvr": ("kpi_cvr", "recommended_24h_cvr", "pred_calibrated_24h_cvr", "pred_24h_cvr"),
    "cpm": ("kpi_cpm", "recommended_24h_cpm", "pred_calibrated_24h_cpm", "pred_24h_cpm"),
}


def first_existing(columns: pd.Index, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def r2_score_np(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) == 0:
        return 0.0
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    if ss_tot <= 0.0:
        return 0.0
    return float(1.0 - ((actual - pred) ** 2).sum() / ss_tot)


def prediction_candidates(serving_col: str, calibrated_col: str, base_col: str, mode: str) -> list[str]:
    if mode == "calibrated":
        return [calibrated_col, serving_col, base_col]
    if mode == "base":
        return [base_col, calibrated_col, serving_col]
    return [serving_col, calibrated_col, base_col]


def summarize_group(df: pd.DataFrame, group_name: str, group_value: str, mode: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric, (actual_col, serving_col, calibrated_col, base_col) in METRICS.items():
        pred_candidates = prediction_candidates(serving_col, calibrated_col, base_col, mode)
        pred_col = first_existing(df.columns, pred_candidates)
        if actual_col not in df.columns or pred_col is None:
            continue
        actual = pd.to_numeric(df[actual_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        pred = pd.to_numeric(df[pred_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        denom = max(float(np.abs(actual).sum()), 1.0)
        rows.append(
            {
                "group_name": group_name,
                "group_value": group_value,
                "metric": metric,
                "rows": int(len(df)),
                "actual_sum": float(actual.sum()),
                "predicted_sum": float(pred.sum()),
                "wmape": float(np.abs(pred - actual).sum() / denom),
                "bias": float((pred - actual).sum() / denom),
                "median_abs_pct_error": float(np.median(np.abs(pred - actual) / np.maximum(np.abs(actual), 1.0)) * 100.0),
                "p75_abs_pct_error": float(np.percentile(np.abs(pred - actual) / np.maximum(np.abs(actual), 1.0), 75) * 100.0),
                "r2": r2_score_np(actual, pred),
                "prediction_column": pred_col,
            }
        )
    return rows


def build_monitor(input_path: Path, mode: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    groups: list[tuple[str, str, pd.DataFrame]] = [("all", "all", df)]
    for col in ["split", "prediction_segment", "history_segment", "forecast_confidence", "spike_risk", "forecast_data_quality", "kpi_reliability_flag"]:
        if col not in df.columns:
            continue
        for value, part in df.groupby(df[col].fillna("UNKNOWN").astype(str), sort=True):
            groups.append((col, value, part))
    rows: list[dict[str, object]] = []
    for group_name, group_value, part in groups:
        rows.extend(summarize_group(part, group_name, group_value, mode))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize 24h forecast quality by metric and segment.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--mode",
        choices=["serving", "calibrated", "base"],
        default="serving",
        help="Which prediction family to prefer when multiple prediction columns exist.",
    )
    args = parser.parse_args()

    monitor = build_monitor(args.input, args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    monitor.to_csv(args.output, index=False)
    overall = monitor[monitor["group_name"].eq("all")].sort_values("metric")
    lines = [
        "Adunbox 24h Forecast Quality Monitor",
        "",
        f"Input: {args.input}",
        f"Output: {args.output}",
        f"Prediction mode: {args.mode}",
        "",
        overall[["metric", "rows", "wmape", "bias", "median_abs_pct_error", "r2", "prediction_column"]].to_string(index=False),
    ]
    args.summary.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
