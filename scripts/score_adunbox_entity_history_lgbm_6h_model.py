from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import adunbox_hierarchical_fallbacks as fallbacks
import train_adunbox_entity_history_lgbm_6h_anchor_v2 as anchor_v2


BASE_DIR = Path(os.getenv("ADUNBOX_PROJECT_DIR", Path(__file__).resolve().parents[1]))
ANCHOR_MODEL_DIR = BASE_DIR / "models" / "adunbox_entity_history_lgbm_6h_anchor_v2"
BUSINESS_MODEL_DIR = BASE_DIR / "models" / "adunbox_entity_history_lgbm_6h_business_v3"
OUTPUT_PATH = BASE_DIR / "outputs" / "adunbox_6h_latest_forecasts.csv"
SUMMARY_PATH = BASE_DIR / "outputs" / "adunbox_6h_latest_forecasts__summary.txt"

TARGET_ROUTE = {
    "target_spend": ANCHOR_MODEL_DIR,
    "target_impressions": ANCHOR_MODEL_DIR,
    "target_inline_link_clicks": ANCHOR_MODEL_DIR,
    "target_tracker_conversions": BUSINESS_MODEL_DIR,
    "target_tracker_revenue": BUSINESS_MODEL_DIR,
}
TARGET_TO_OUTPUT = {
    "target_spend": "pred_6h_spend",
    "target_impressions": "pred_6h_impressions",
    "target_inline_link_clicks": "pred_6h_inline_link_clicks",
    "target_tracker_conversions": "pred_6h_tracker_conversions",
    "target_tracker_revenue": "pred_6h_tracker_revenue",
}
RAW_TARGET_TO_HISTORY = {
    "target_spend": "spend_same_6h_7d_mean",
    "target_impressions": "impressions_same_6h_7d_mean",
    "target_inline_link_clicks": "inline_link_clicks_same_6h_7d_mean",
    "target_tracker_conversions": "tracker_conversions_same_6h_7d_mean",
    "target_tracker_revenue": "tracker_revenue_same_6h_7d_mean",
}
VALID_ANCHOR_HOURS = sorted(anchor_v2.VALID_ANCHOR_HOURS)
MIN_MODEL_HISTORY_HOURS = int(os.getenv("ADUNBOX_6H_MIN_MODEL_HISTORY_HOURS", "24"))
MIN_PEER_ADS = int(os.getenv("ADUNBOX_MIN_FALLBACK_PEER_ADS", "2"))
RECENT_UTC_DAYS = int(os.getenv("ADUNBOX_6H_RECENT_UTC_DAYS", "10"))
CHUNKSIZE = int(os.getenv("ADUNBOX_6H_SCORE_CHUNKSIZE", "25000"))
SCORE_ACCOUNT_IDS = {
    item.strip()
    for item in os.getenv("ADUNBOX_SCORE_ACCOUNT_IDS", "").split(",")
    if item.strip()
}
SCORE_MAX_ADS = int(os.getenv("ADUNBOX_SCORE_MAX_ADS", "0") or "0")
FEATURE_CACHE_PATH = Path(os.getenv("ADUNBOX_6H_FEATURE_CACHE", BASE_DIR / "outputs" / "adunbox_6h_latest_feature_cache.joblib"))
REUSE_FEATURE_CACHE = os.getenv("ADUNBOX_REUSE_FEATURE_CACHE", "false").strip().lower() in {"1", "true", "yes", "y"}
WRITE_FEATURE_CACHE = os.getenv("ADUNBOX_WRITE_FEATURE_CACHE", "true").strip().lower() in {"1", "true", "yes", "y"}


def _previous_anchor_hour(hour: int) -> int:
    candidates = [h for h in VALID_ANCHOR_HOURS if h <= hour]
    return candidates[-1] if candidates else VALID_ANCHOR_HOURS[-1]


def _anchor_ts_for_latest(latest_ts: pd.Timestamp) -> pd.Timestamp:
    anchor_hour = _previous_anchor_hour(int(latest_ts.hour))
    anchor_date = latest_ts.normalize()
    if anchor_hour > int(latest_ts.hour):
        anchor_date = anchor_date - pd.Timedelta(days=1)
    return anchor_date + pd.Timedelta(hours=anchor_hour)


def _load_recent_hourly(path: Path) -> pd.DataFrame:
    usecols = ["date", *anchor_v2.ENTITY_COLS, "timezone", *anchor_v2.RAW_FEATURES]
    max_input_ts: pd.Timestamp | None = None
    for date_chunk in pd.read_csv(path, usecols=["date"], chunksize=CHUNKSIZE, low_memory=False):
        date_values = pd.to_datetime(date_chunk["date"], errors="coerce", utc=True).dropna()
        if date_values.empty:
            continue
        chunk_max = date_values.max()
        max_input_ts = chunk_max if max_input_ts is None else max(max_input_ts, chunk_max)
    if max_input_ts is None:
        raise RuntimeError(f"No parseable hourly timestamps found in {path}")
    cutoff_utc = max_input_ts - pd.Timedelta(days=RECENT_UTC_DAYS)
    parts: list[pd.DataFrame] = []
    ad_spend: dict[str, float] = {}

    for chunk in pd.read_csv(path, usecols=lambda c: c in usecols, chunksize=CHUNKSIZE, low_memory=False):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce", utc=True)
        chunk = chunk[chunk["date"].notna() & (chunk["date"] >= cutoff_utc)].copy()
        if chunk.empty:
            continue
        for col in anchor_v2.ENTITY_COLS:
            chunk[col] = anchor_v2.normalize_id(chunk[col])
        if SCORE_ACCOUNT_IDS:
            chunk = chunk[chunk["account_id"].isin(SCORE_ACCOUNT_IDS)].copy()
            if chunk.empty:
                continue
        chunk["timezone"] = chunk["timezone"].fillna("").astype(str)
        for col in anchor_v2.RAW_FEATURES:
            chunk[col] = pd.to_numeric(chunk.get(col, 0.0), errors="coerce").fillna(0.0).astype("float32")
        chunk["local_ts"] = anchor_v2.to_local_ts(chunk["date"], chunk["timezone"])
        part = chunk.groupby(["local_ts", "timezone", *anchor_v2.ENTITY_COLS], as_index=False)[anchor_v2.RAW_FEATURES].sum()
        if SCORE_MAX_ADS > 0:
            spend_by_ad = part.groupby("ad_id")["spend"].sum()
            for ad_id, spend in spend_by_ad.items():
                ad_spend[str(ad_id)] = ad_spend.get(str(ad_id), 0.0) + float(spend)
        parts.append(part)

    if not parts:
        raise RuntimeError(f"No recent hourly rows found in {path}")
    hourly = pd.concat(parts, ignore_index=True)
    hourly = hourly.groupby(["local_ts", "timezone", *anchor_v2.ENTITY_COLS], as_index=False)[anchor_v2.RAW_FEATURES].sum()
    if SCORE_MAX_ADS > 0:
        allowed_ads = {
            ad for ad, _ in sorted(ad_spend.items(), key=lambda item: item[1], reverse=True)[:SCORE_MAX_ADS]
        }
        hourly = hourly[hourly["ad_id"].astype(str).isin(allowed_ads)].copy()
    return hourly.sort_values(["ad_id", "local_ts"]).reset_index(drop=True)


def _build_latest_feature_rows(hourly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, group in hourly.groupby("ad_id", sort=False):
        group = group.sort_values("local_ts").drop_duplicates("local_ts", keep="last")
        if group.empty:
            continue
        anchor_ts = _anchor_ts_for_latest(pd.Timestamp(group["local_ts"].max()))
        min_ts = anchor_ts - pd.Timedelta(hours=168)
        idx = pd.date_range(min_ts, anchor_ts, freq="1h")
        dense = group.set_index("local_ts").reindex(idx)
        observed = dense["ad_id"].notna().to_numpy(dtype=np.int8)
        for col in anchor_v2.RAW_FEATURES:
            dense[col] = dense[col].fillna(0.0)
        for col in ["timezone", *anchor_v2.ENTITY_COLS]:
            dense[col] = dense[col].ffill().bfill()
        if dense["ad_id"].isna().all():
            continue
        dense["local_ts"] = idx

        metric_arr = dense[anchor_v2.METRIC_COLS].to_numpy(dtype=np.float32)
        cumulative = np.vstack([np.zeros((1, len(anchor_v2.METRIC_COLS)), dtype=np.float32), np.cumsum(metric_arr, axis=0)])
        observed_history = int(observed.sum())
        pos = len(dense) - 1
        future_start = pos + 1
        row: dict[str, object] = {
            "anchor_ts": anchor_ts,
            "forecast_window_start": anchor_ts + pd.Timedelta(hours=1),
            "forecast_window_end": anchor_ts + pd.Timedelta(hours=6),
            "anchor_hour": anchor_ts.hour,
            "anchor_hour_0": int(anchor_ts.hour == 0),
            "anchor_hour_6": int(anchor_ts.hour == 6),
            "anchor_hour_12": int(anchor_ts.hour == 12),
            "anchor_hour_18": int(anchor_ts.hour == 18),
            "dow": anchor_ts.dayofweek,
            "is_weekend": int(anchor_ts.dayofweek >= 5),
            "hour_sin": float(np.sin(2 * np.pi * anchor_ts.hour / 24.0)),
            "hour_cos": float(np.cos(2 * np.pi * anchor_ts.hour / 24.0)),
            "dow_sin": float(np.sin(2 * np.pi * anchor_ts.dayofweek / 7.0)),
            "dow_cos": float(np.cos(2 * np.pi * anchor_ts.dayofweek / 7.0)),
            "observed_history_hours": observed_history,
            "observed_history_ratio": observed_history / 168.0,
            "observed_recent_24h": int(observed[-24:].sum()) if len(observed) >= 24 else int(observed.sum()),
            "hours_active": float(observed_history),
        }
        for col in anchor_v2.ENTITY_COLS:
            row[col] = dense.iloc[pos][col]
        row["timezone"] = dense.iloc[pos]["timezone"]

        for window in anchor_v2.WINDOWS:
            values = anchor_v2.window_sum(cumulative, pos + 1, window)
            for metric_idx, metric in enumerate(anchor_v2.METRIC_COLS):
                row[f"{metric}_sum_{window}h"] = float(values[metric_idx])
                row[f"{metric}_avg_{window}h"] = float(values[metric_idx]) / float(window)
            anchor_v2.add_ratio_features(row, f"kpi_{window}h", values)

        lag_values = []
        for lag in anchor_v2.SAME_WINDOW_LAGS:
            values = anchor_v2.lagged_target_window(cumulative, future_start, anchor_v2.TARGET_HOURS, lag)
            lag_values.append(values)
            days = lag // 24
            for metric_idx, metric in enumerate(anchor_v2.METRIC_COLS):
                row[f"{metric}_same_target_6h_d{days}"] = float(values[metric_idx])
            anchor_v2.add_ratio_features(row, f"kpi_same_target_6h_d{days}", values)
        anchor_v2.add_same_window_stats(row, np.vstack(lag_values))

        cum_values = cumulative[pos + 1]
        for metric_idx, metric in enumerate(anchor_v2.METRIC_COLS):
            row[f"{metric}_cum"] = float(cum_values[metric_idx])
        anchor_v2.add_ratio_features(row, "kpi_cum", cum_values)
        rows.append(row)

    return pd.DataFrame(rows)


def _derive_kpis(out: pd.DataFrame, prefix: str = "recommended_6h") -> pd.DataFrame:
    spend = pd.to_numeric(out.get(f"{prefix}_spend", 0.0), errors="coerce").fillna(0.0)
    impressions = pd.to_numeric(out.get(f"{prefix}_impressions", 0.0), errors="coerce").fillna(0.0)
    clicks = pd.to_numeric(out.get(f"{prefix}_inline_link_clicks", 0.0), errors="coerce").fillna(0.0)
    conversions = pd.to_numeric(out.get(f"{prefix}_tracker_conversions", 0.0), errors="coerce").fillna(0.0)
    revenue = pd.to_numeric(out.get(f"{prefix}_tracker_revenue", 0.0), errors="coerce").fillna(0.0)
    out[f"{prefix}_roas"] = np.where(spend > 0.0, revenue / spend, 0.0)
    out[f"{prefix}_profit"] = revenue - spend
    out[f"{prefix}_ctr"] = np.where(impressions > 0.0, clicks / impressions * 100.0, 0.0)
    out[f"{prefix}_cvr"] = np.where(clicks > 0.0, conversions / clicks * 100.0, 0.0)
    out[f"{prefix}_cpm"] = np.where(impressions > 0.0, spend / impressions * 1000.0, 0.0)
    return out


def score_latest(hourly_input: Path) -> pd.DataFrame:
    if REUSE_FEATURE_CACHE and FEATURE_CACHE_PATH.exists():
        payload = joblib.load(FEATURE_CACHE_PATH)
        features = payload["features"]
    else:
        hourly = _load_recent_hourly(hourly_input)
        features = _build_latest_feature_rows(hourly)
        if WRITE_FEATURE_CACHE:
            FEATURE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"features": features, "hourly_input": str(hourly_input)}, FEATURE_CACHE_PATH, compress=3)
    if features.empty:
        raise RuntimeError("No 6h scoring rows were built.")

    anchor_meta = joblib.load(ANCHOR_MODEL_DIR / "metadata.joblib")
    feature_cols = anchor_meta["feature_cols"]
    for col in feature_cols:
        if col not in features.columns:
            features[col] = 0.0
    x = features[feature_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0).astype("float32")

    out = features[
        [
            "anchor_ts",
            "forecast_window_start",
            "forecast_window_end",
            "timezone",
            *anchor_v2.ENTITY_COLS,
            "observed_history_hours",
            "observed_history_ratio",
            "observed_recent_24h",
        ]
    ].copy()
    enough_history = (
        pd.to_numeric(features["observed_history_hours"], errors="coerce").fillna(0.0) >= MIN_MODEL_HISTORY_HOURS
    ) & (
        pd.to_numeric(features.get("spend_sum_24h", 0.0), errors="coerce").fillna(0.0) > 0.0
    )
    out["forecast_confidence"] = np.where(enough_history, "HIGH", "LOW")
    out["forecast_use_case"] = np.where(enough_history, "model_point", "same_window_benchmark")

    for target, output_col in TARGET_TO_OUTPUT.items():
        model_dir = TARGET_ROUTE[target]
        model = joblib.load(model_dir / f"{target}.joblib")
        raw_log_pred = np.clip(model.predict(x), -20.0, 20.0)
        pred = np.maximum(0.0, np.expm1(raw_log_pred)).astype("float32")
        out[output_col] = pred
        history_col = RAW_TARGET_TO_HISTORY[target]
        history_value = pd.to_numeric(features.get(history_col, 0.0), errors="coerce").fillna(0.0)
        recommended, source = fallbacks.choose_hierarchical_forecast(
            features,
            pd.Series(pred, index=features.index),
            history_value,
            needs_fallback=~enough_history,
            min_peer_entities=MIN_PEER_ADS,
        )
        final_col = output_col.replace("pred_6h_", "recommended_6h_")
        out[final_col] = recommended.astype("float32")
        source_col = f"{final_col}_source"
        out[source_col] = np.where(enough_history, "model_point", source)

    source_cols = [col for col in out.columns if col.endswith("_source")]
    out["benchmark_source"] = out[source_cols].mode(axis=1)[0] if source_cols else "model_point"
    out["forecast_status"] = np.where(
        out["benchmark_source"].eq("insufficient_history"),
        "insufficient_history_monitoring",
        np.where(enough_history, "model_forecast", "hierarchical_benchmark_forecast"),
    )
    out = _derive_kpis(out, "recommended_6h")
    out["model_source"] = "entity_history_lgbm_6h_target_routed"
    out["hourly_source"] = f"feature_cache:{FEATURE_CACHE_PATH}" if REUSE_FEATURE_CACHE and FEATURE_CACHE_PATH.exists() else str(hourly_input)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score latest 6h forecasts from final target-routed LGBM models.")
    parser.add_argument("--hourly-input", type=Path, default=anchor_v2.HOURLY_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    forecasts = score_latest(args.hourly_input)
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, dir=str(output_path.parent), newline="", encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
        forecasts.to_csv(tmp, index=False)
    tmp_path.replace(output_path)
    SUMMARY_PATH.write_text(
        "\n".join(
            [
                "6h latest production scoring completed",
                f"rows={len(forecasts):,}",
                f"output={output_path}",
                f"status_counts={forecasts['forecast_status'].value_counts(dropna=False).to_dict()}",
                f"benchmark_source_counts={forecasts['benchmark_source'].value_counts(dropna=False).to_dict()}",
            ]
        ),
        encoding="utf-8",
    )
    print(SUMMARY_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
