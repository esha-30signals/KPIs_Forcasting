from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("ADUNBOX_PROJECT_DIR", str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import joblib
import numpy as np
import pandas as pd
from dagster import Definitions, ScheduleDefinition, asset, define_asset_job
from tensorflow import keras

import train_adunbox_daily_24h_model as daily24
import train_adunbox_entity_history_gru_168h_padded_6h as padded_6h
import train_adunbox_entity_history_gru_allhours as base_gru
import train_adunbox_local_midnight_sequence_models as base_seq


DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
HOURLY_INPUT = Path(os.getenv("ADUNBOX_HOURLY_INPUT", DATA_DIR / "traffic_reports.csv"))
DAILY_INPUT = Path(os.getenv("ADUNBOX_DAILY_INPUT", DATA_DIR / "adunbox_daily_breakdown_kpis.csv"))
HISTORICAL_6H_REVIEW_INPUT = Path(
    os.getenv("ADUNBOX_6H_REVIEW_INPUT", ROOT / "docs" / "adunbox_6h_final_prediction_history_benchmark_review.csv")
)

ENTITY_COLS = ["account_id", "campaign_id", "adset_id", "ad_id"]
RAW_6H_TARGETS = ["spend", "impressions", "inline_link_clicks", "tracker_conversions", "tracker_revenue"]
REVIEW_6H_RAW_TARGETS = [
    "spend",
    "impressions",
    "inline_link_clicks",
    "tracker_conversions",
    "tracker_revenue",
]
MIN_24H_KPI_IMPRESSIONS = 100.0
MIN_24H_KPI_CLICKS = 5.0
MIN_24H_RECENT_SPEND_MEAN = 1.0
MIN_24H_HISTORY_DAYS = 14


def _normalize_id(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    return text.str.replace(r"\.0$", "", regex=True).replace({"nan": "", "None": ""})


def _require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _write_csv(df: pd.DataFrame, name: str) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False)
    return str(path)


def _safe_mape(actual: pd.Series, pred: pd.Series) -> float:
    actual_num = pd.to_numeric(actual, errors="coerce").fillna(0.0)
    pred_num = pd.to_numeric(pred, errors="coerce").fillna(0.0)
    mask = actual_num.abs() > 1e-9
    if not mask.any():
        return 0.0
    return float(((actual_num[mask] - pred_num[mask]).abs() / actual_num[mask].abs()).mean())


def _latest_complete_6h_anchor(ts: pd.Timestamp) -> pd.Timestamp:
    hour = int(ts.hour // 6) * 6
    return ts.normalize() + pd.Timedelta(hours=hour)


def _inverse_log_scaled(pred_scaled: np.ndarray, scaler) -> np.ndarray:
    pred_log = np.clip(scaler.inverse_transform(pred_scaled), 0.0, 20.0)
    return np.expm1(pred_log).astype(np.float32)


def _add_24h_confidence_and_ranges(out: pd.DataFrame) -> pd.DataFrame:
    """Add production confidence flags and p10/p50/p90 ranges to 24h rows."""
    scored = out.copy()
    spend_mean = pd.to_numeric(scored.get("spend_roll_mean_7d", 0.0), errors="coerce").fillna(0.0)
    impressions_mean = pd.to_numeric(scored.get("impressions_roll_mean_7d", 0.0), errors="coerce").fillna(0.0)
    clicks_mean = pd.to_numeric(scored.get("inline_link_clicks_roll_mean_7d", 0.0), errors="coerce").fillna(0.0)
    pred_impressions = pd.to_numeric(scored.get("pred_24h_impressions", 0.0), errors="coerce").fillna(0.0)
    pred_clicks = pd.to_numeric(scored.get("pred_24h_inline_link_clicks", 0.0), errors="coerce").fillna(0.0)
    days_active = pd.to_numeric(scored.get("days_active", 0), errors="coerce").fillna(0)

    spike_cols = [col for col in scored.columns if col.endswith("_spike_flag")]
    any_spike_cap = scored[spike_cols].any(axis=1) if spike_cols else pd.Series(False, index=scored.index)
    low_history = days_active < MIN_24H_HISTORY_DAYS
    low_recent_volume = (
        (spend_mean < MIN_24H_RECENT_SPEND_MEAN)
        | (impressions_mean < MIN_24H_KPI_IMPRESSIONS)
        | (clicks_mean < 1.0)
    )
    low_kpi_volume = (pred_impressions < MIN_24H_KPI_IMPRESSIONS) | (pred_clicks < MIN_24H_KPI_CLICKS)

    scored["forecast_confidence"] = np.select(
        [low_history | low_recent_volume, any_spike_cap | low_kpi_volume],
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
        [
            scored["kpi_reliability_flag"].ne("OK"),
            any_spike_cap,
            scored["forecast_confidence"].eq("MEDIUM"),
        ],
        [
            "KPI ratios are low-confidence; use raw metric range.",
            "One or more raw predictions were capped by spike guardrail.",
            "Use point forecast with caution.",
        ],
        default="Forecast is stable enough for point use.",
    )
    return scored


@asset
def traffic_source_reports_raw() -> pd.DataFrame:
    """Load hourly traffic report rows from the local/prod-equivalent input CSV."""
    if not HOURLY_INPUT.exists():
        raise FileNotFoundError(f"Hourly input not found: {HOURLY_INPUT}")

    df = base_seq.load_hourly()
    _require_columns(df, ["date", "local_ts", "timezone", *ENTITY_COLS, *base_seq.SEQ_FEATURES], "hourly input")
    if df.empty:
        raise ValueError("Hourly input loaded successfully but contains zero usable rows.")
    return df


@asset
def traffic_source_accounts_raw(traffic_source_reports_raw: pd.DataFrame) -> pd.DataFrame:
    """Extract account/timezone metadata from hourly rows."""
    accounts = (
        traffic_source_reports_raw[["account_id", "timezone"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["account_id", "timezone"])
        .reset_index(drop=True)
    )
    if accounts.empty:
        raise ValueError("No account timezone metadata available from hourly input.")
    return accounts


@asset
def hourly_timezone_joined(
    traffic_source_reports_raw: pd.DataFrame,
    traffic_source_accounts_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate hourly rows to one local-hour row per ad/entity."""
    df = traffic_source_reports_raw.copy()
    if "timezone" not in df.columns or df["timezone"].eq("").all():
        df = df.drop(columns=["timezone"], errors="ignore").merge(
            traffic_source_accounts_raw, on="account_id", how="left"
        )

    grouped = (
        df.groupby(["local_ts", "timezone", *ENTITY_COLS], as_index=False)[base_seq.RAW_FEATURES]
        .sum()
        .sort_values(["ad_id", "local_ts"])
        .reset_index(drop=True)
    )
    # The padded GRU uses non-null date rows to distinguish observed hours from
    # padded missing hours after densification.
    grouped["date"] = grouped["local_ts"]
    grouped = base_seq.add_kpis(grouped)
    grouped["hour_of_day_sin"] = np.sin(2 * np.pi * grouped["local_ts"].dt.hour / 24.0).astype("float32")
    grouped["hour_of_day_cos"] = np.cos(2 * np.pi * grouped["local_ts"].dt.hour / 24.0).astype("float32")
    grouped["day_of_week_sin"] = np.sin(2 * np.pi * grouped["local_ts"].dt.dayofweek / 7.0).astype("float32")
    grouped["day_of_week_cos"] = np.cos(2 * np.pi * grouped["local_ts"].dt.dayofweek / 7.0).astype("float32")
    return grouped


@asset
def daily_ad_breakdown() -> pd.DataFrame:
    """Load daily ad-level rows for 24h forecasting."""
    if not DAILY_INPUT.exists():
        raise FileNotFoundError(f"Daily input not found: {DAILY_INPUT}")

    daily, _source = daily24.load_daily(force_rebuild_from_hourly=False, daily_input_path=DAILY_INPUT)
    if daily.empty:
        raise ValueError("Daily input has no usable ad-level rows after filtering.")
    return daily


@asset
def historical_6h_prediction_review() -> pd.DataFrame:
    """Load the larger historical 6h actual-vs-predicted review dataset."""
    if not HISTORICAL_6H_REVIEW_INPUT.exists():
        raise FileNotFoundError(f"6h historical review input not found: {HISTORICAL_6H_REVIEW_INPUT}")

    usecols = [
        "account_id",
        "campaign_id",
        "adset_id",
        "ad_id",
        "timezone",
        "anchor_local_ts",
        "forecast_anchor_date",
        "model_dataset_split",
        "model_source",
    ]
    for target in REVIEW_6H_RAW_TARGETS:
        usecols.extend([f"next_6h_actual_{target}", f"next_6h_pred_{target}"])

    review = pd.read_csv(HISTORICAL_6H_REVIEW_INPUT, usecols=lambda col: col in usecols, low_memory=False)
    _require_columns(review, ["ad_id", "anchor_local_ts", "model_dataset_split"], "6h historical review")
    review["anchor_local_ts"] = pd.to_datetime(review["anchor_local_ts"], errors="coerce")
    review = review[review["anchor_local_ts"].notna()].copy()
    if review.empty:
        raise ValueError("6h historical review loaded successfully but contains zero usable rows.")
    return review


@asset
def historical_6h_quality_summary(historical_6h_prediction_review: pd.DataFrame) -> pd.DataFrame:
    """Summarize 6h backtest coverage and raw-metric error by train/valid/test split."""
    review = historical_6h_prediction_review.copy()
    rows: list[dict[str, object]] = []
    for split, split_df in review.groupby("model_dataset_split", dropna=False):
        row: dict[str, object] = {
            "model_dataset_split": split,
            "rows": int(len(split_df)),
            "ads": int(split_df["ad_id"].nunique()),
            "accounts": int(split_df["account_id"].nunique()) if "account_id" in split_df.columns else 0,
            "first_anchor_local_ts": split_df["anchor_local_ts"].min(),
            "last_anchor_local_ts": split_df["anchor_local_ts"].max(),
        }
        for target in REVIEW_6H_RAW_TARGETS:
            actual_col = f"next_6h_actual_{target}"
            pred_col = f"next_6h_pred_{target}"
            if actual_col not in split_df.columns or pred_col not in split_df.columns:
                continue
            actual = pd.to_numeric(split_df[actual_col], errors="coerce").fillna(0.0)
            pred = pd.to_numeric(split_df[pred_col], errors="coerce").fillna(0.0)
            row[f"{target}_mae"] = float((actual - pred).abs().mean())
            row[f"{target}_mape"] = _safe_mape(actual, pred)
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("model_dataset_split").reset_index(drop=True)
    _write_csv(summary, "adunbox_6h_historical_quality_summary.csv")
    return summary


@asset
def features_6h_hourly_sequences(hourly_timezone_joined: pd.DataFrame) -> dict[str, object]:
    """Build latest eligible 168h padded GRU feature windows per ad."""
    seqs: list[np.ndarray] = []
    statics: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    rejected = {"short_history": 0, "bad_anchor": 0, "low_recent_activity": 0}

    for ad_id, group in hourly_timezone_joined.groupby("ad_id", sort=False):
        entity_meta = group[["timezone", *ENTITY_COLS]].dropna(how="all").tail(1)
        entity_meta = entity_meta.iloc[0].to_dict() if len(entity_meta) else {"ad_id": ad_id}
        dense = padded_6h.dense_ad_group(group)
        if len(dense) < padded_6h.MIN_OBSERVED_HISTORY_HOURS:
            rejected["short_history"] += 1
            continue

        latest_anchor = _latest_complete_6h_anchor(pd.Timestamp(dense["local_ts"].max()))
        candidates = dense.index[dense["local_ts"].eq(latest_anchor)].to_numpy()
        if len(candidates) == 0:
            rejected["bad_anchor"] += 1
            continue
        pos = int(candidates[-1])

        if pos + 1 < padded_6h.SEQ_HOURS:
            rejected["short_history"] += 1
            continue

        metric_arr = dense[base_gru.METRIC_COLS].to_numpy(dtype=np.float32)
        observed = dense["observed_hour"].to_numpy(dtype=np.int8)
        cumulative = np.vstack([np.zeros((1, len(base_gru.METRIC_COLS)), dtype=np.float32), np.cumsum(metric_arr, axis=0)])
        obs_cum = np.concatenate([[0], np.cumsum(observed)])
        observed_history = int(obs_cum[pos + 1] - obs_cum[pos + 1 - padded_6h.SEQ_HOURS])
        if observed_history < padded_6h.MIN_OBSERVED_HISTORY_HOURS:
            rejected["short_history"] += 1
            continue
        if not padded_6h.passes_recent_activity(cumulative, pos):
            rejected["low_recent_activity"] += 1
            continue

        seq_sample = dense.loc[pos - padded_6h.SEQ_HOURS + 1:pos, base_seq.SEQ_FEATURES].to_numpy(dtype=np.float32)
        if np.isnan(seq_sample).any():
            rejected["short_history"] += 1
            continue

        seqs.append(seq_sample)
        statics.append(padded_6h.build_static(cumulative, pos))
        meta = {**entity_meta, **dense.loc[pos, ["timezone", *ENTITY_COLS]].dropna().to_dict()}
        meta["ad_id"] = meta.get("ad_id", ad_id)
        meta["forecast_anchor_local_ts"] = latest_anchor
        meta["forecast_window_start_local_ts"] = latest_anchor + pd.Timedelta(hours=1)
        meta["forecast_window_end_local_ts"] = latest_anchor + pd.Timedelta(hours=padded_6h.TARGET_HOURS)
        meta["observed_history_hours"] = observed_history
        rows.append(meta)

    meta_df = pd.DataFrame(rows)
    return {
        "seq": np.asarray(seqs, dtype=np.float32),
        "static": np.asarray(statics, dtype=np.float32),
        "meta": meta_df,
        "rejected": rejected,
    }


@asset
def features_24h_daily_windows(daily_ad_breakdown: pd.DataFrame) -> pd.DataFrame:
    """Build latest feature row per ad for next-day 24h scoring."""
    dataset, feature_cols = daily24.add_features_and_targets(daily_ad_breakdown)
    if dataset.empty:
        raise ValueError("Daily feature builder produced zero rows; need at least 8 daily rows per ad.")
    latest = (
        dataset.sort_values(["ad_id", "local_date"])
        .groupby("ad_id", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    latest.attrs["feature_cols"] = feature_cols
    return latest


@asset
def score_6h_gru(features_6h_hourly_sequences: dict[str, object]) -> pd.DataFrame:
    """Score latest next-6h forecasts with the promoted hybrid GRU."""
    seq = features_6h_hourly_sequences["seq"]
    static = features_6h_hourly_sequences["static"]
    meta = features_6h_hourly_sequences["meta"].copy()

    if len(meta) == 0:
        empty = pd.DataFrame(columns=[*ENTITY_COLS, "forecast_anchor_local_ts", "status"])
        empty["status"] = "no_eligible_6h_rows"
        _write_csv(empty, "adunbox_6h_latest_forecasts.csv")
        return empty

    baseline_dir = ROOT / "models" / "adunbox_entity_history_gru_168h_padded_6h"
    hybrid_dir = ROOT / "models" / "adunbox_entity_history_gru_168h_padded_6h_hybrid"
    baseline_scalers = joblib.load(baseline_dir / "scalers.joblib")
    hybrid_scalers = joblib.load(hybrid_dir / "scalers.joblib")

    seq_volume = baseline_scalers["seq_scaler"].transform(seq.reshape(-1, seq.shape[-1])).reshape(seq.shape).astype(np.float32)
    static_volume = baseline_scalers["static_scaler"].transform(static).astype(np.float32)
    try:
        volume_model = keras.models.load_model(baseline_dir / "sequence_model.keras", compile=False)
    except Exception as exc:
        out = meta.copy()
        out["status"] = "model_load_failed"
        out["error"] = f"{type(exc).__name__}: {exc}"
        _write_csv(out, "adunbox_6h_latest_forecasts.csv")
        return out
    volume_full = _inverse_log_scaled(
        volume_model.predict([seq_volume, static_volume], verbose=0),
        baseline_scalers["target_scaler"],
    )
    volume_pred = volume_full[:, :3]

    seq_ratio = hybrid_scalers["seq_scaler"].transform(seq.reshape(-1, seq.shape[-1])).reshape(seq.shape).astype(np.float32)
    static_ratio = hybrid_scalers["static_scaler"].transform(static).astype(np.float32)
    try:
        ratio_model = keras.models.load_model(hybrid_dir / "ratio_model.keras", compile=False)
    except Exception as exc:
        out = meta.copy()
        out["status"] = "model_load_failed"
        out["error"] = f"{type(exc).__name__}: {exc}"
        _write_csv(out, "adunbox_6h_latest_forecasts.csv")
        return out
    ratio_pred = _inverse_log_scaled(
        ratio_model.predict([seq_ratio, static_ratio], verbose=0),
        hybrid_scalers["ratio_target_scaler"],
    )

    spend = np.maximum(volume_pred[:, 0], 0.0)
    impressions = np.maximum(volume_pred[:, 1], 0.0)
    clicks = np.maximum(volume_pred[:, 2], 0.0)
    cvr = np.maximum(ratio_pred[:, 0], 0.0)
    roas = np.maximum(ratio_pred[:, 1], 0.0)

    out = meta.copy()
    out["pred_6h_spend"] = spend
    out["pred_6h_impressions"] = impressions
    out["pred_6h_inline_link_clicks"] = clicks
    out["pred_6h_tracker_conversions"] = clicks * cvr
    out["pred_6h_tracker_revenue"] = spend * roas
    out["pred_6h_cvr"] = cvr
    out["pred_6h_roas"] = roas
    out["model_source"] = "hybrid_168h_gru_volume_plus_cvr_roas"
    _write_csv(out, "adunbox_6h_latest_forecasts.csv")
    return out


@asset
def score_24h_histgb(features_24h_daily_windows: pd.DataFrame) -> pd.DataFrame:
    """Score latest next-24h forecasts with the promoted daily HistGB models."""
    metadata = joblib.load(ROOT / "models" / "adunbox_daily_24h_histgb" / "metadata.joblib")
    feature_cols = metadata["feature_cols"]
    latest = features_24h_daily_windows.copy()
    X = latest[feature_cols].astype("float32")

    context_cols = ["local_date", "timezone", *daily24.ENTITY_COLS, "days_active"]
    roll_cols = [col for col in latest.columns if col.endswith("_roll_mean_7d")]
    out = latest[[*context_cols, *roll_cols]].copy()
    out["forecast_anchor_local_date"] = out["local_date"] + pd.Timedelta(days=1)
    for raw_target in daily24.RAW_TARGETS:
        target = f"target_24h_{raw_target}"
        model = joblib.load(ROOT / "models" / "adunbox_daily_24h_histgb" / f"{target}.joblib")
        pred = np.expm1(model.predict(X))
        out[f"pred_24h_{raw_target}"] = np.maximum(0.0, pred).astype("float32")

    out = daily24.derive_kpis(out)
    out = _add_24h_confidence_and_ranges(out)
    out["model_source"] = "daily_24h_histgb"
    _write_csv(out, "adunbox_24h_latest_forecasts.csv")
    return out


@asset
def apply_spike_calibration(score_24h_histgb: pd.DataFrame) -> pd.DataFrame:
    """Add production guardrails for spike-prone forecasts.

    The guardrail is intentionally conservative: it can cap risky spikes, but it
    must not inflate a prediction above the raw model output.
    """
    out = score_24h_histgb.copy()
    guardrails = {
        "spend": "pred_24h_spend",
        "impressions": "pred_24h_impressions",
        "inline_link_clicks": "pred_24h_inline_link_clicks",
        "tracker_conversions": "pred_24h_tracker_conversions",
        "tracker_revenue": "pred_24h_tracker_revenue",
    }
    for raw_name, pred_col in guardrails.items():
        lag_col = f"{raw_name}_roll_mean_7d"
        cap_col = f"{pred_col}_guardrail_cap"
        raw_pred_col = f"{pred_col}_raw"
        out[raw_pred_col] = out[pred_col]
        if lag_col in out.columns:
            history_cap = np.maximum(out[lag_col] * 3.0, 0.0)
            fallback_cap = np.maximum(out[pred_col] * 0.5, 0.0)
            out[cap_col] = np.where(history_cap > 0.0, history_cap, fallback_cap)
            out[f"{pred_col}_spike_flag"] = out[pred_col] > history_cap
            out[pred_col] = np.minimum(out[pred_col], out[cap_col])
        else:
            out[cap_col] = out[pred_col]
            out[f"{pred_col}_spike_flag"] = False
    out = daily24.derive_kpis(out)
    out = _add_24h_confidence_and_ranges(out)
    _write_csv(out, "adunbox_24h_latest_forecasts_calibrated.csv")
    return out


@asset
def publish_forecast_outputs(score_6h_gru: pd.DataFrame, apply_spike_calibration: pd.DataFrame) -> dict[str, object]:
    """Publish final forecast outputs to local CSVs; production can swap this for DB writes."""
    six_path = _write_csv(score_6h_gru, "adunbox_6h_latest_forecasts_published.csv")
    daily_path = _write_csv(apply_spike_calibration, "adunbox_24h_latest_forecasts_published.csv")
    confidence_counts = (
        apply_spike_calibration.get("forecast_confidence", pd.Series(dtype=str))
        .value_counts(dropna=False)
        .to_dict()
    )
    reliability_counts = (
        apply_spike_calibration.get("kpi_reliability_flag", pd.Series(dtype=str))
        .value_counts(dropna=False)
        .to_dict()
    )
    summary = {
        "status": "published",
        "six_hour_rows": int(len(score_6h_gru)),
        "six_hour_model_load_failed_rows": int(score_6h_gru.get("status", pd.Series(dtype=str)).eq("model_load_failed").sum()),
        "twenty_four_hour_rows": int(len(apply_spike_calibration)),
        "twenty_four_hour_confidence_counts": confidence_counts,
        "twenty_four_hour_kpi_reliability_counts": reliability_counts,
        "six_hour_output": six_path,
        "twenty_four_hour_output": daily_path,
    }
    (OUTPUT_DIR / "forecast_publish_summary.json").write_text(pd.Series(summary).to_json(indent=2), encoding="utf-8")
    return summary


@asset
def publish_orchestration_data_summary(
    traffic_source_reports_raw: pd.DataFrame,
    daily_ad_breakdown: pd.DataFrame,
    historical_6h_quality_summary: pd.DataFrame,
) -> dict[str, object]:
    """Publish a quick local summary of the datasets Dagster can see."""
    summary = {
        "hourly_rows": int(len(traffic_source_reports_raw)),
        "hourly_ads": int(traffic_source_reports_raw["ad_id"].nunique()),
        "hourly_first_utc": str(traffic_source_reports_raw["date"].min()),
        "hourly_last_utc": str(traffic_source_reports_raw["date"].max()),
        "daily_rows": int(len(daily_ad_breakdown)),
        "daily_ads": int(daily_ad_breakdown["ad_id"].nunique()),
        "historical_6h_review_rows": int(historical_6h_quality_summary["rows"].sum()),
        "historical_6h_review_ads": int(historical_6h_quality_summary["ads"].max()),
        "historical_6h_quality_summary_output": str(OUTPUT_DIR / "adunbox_6h_historical_quality_summary.csv"),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "orchestration_data_summary.json").write_text(
        pd.Series(summary).to_json(indent=2),
        encoding="utf-8",
    )
    return summary


forecast_job = define_asset_job(
    name="adunbox_forecast_job",
    selection=[
        traffic_source_reports_raw,
        traffic_source_accounts_raw,
        hourly_timezone_joined,
        daily_ad_breakdown,
        historical_6h_prediction_review,
        historical_6h_quality_summary,
        features_6h_hourly_sequences,
        features_24h_daily_windows,
        score_6h_gru,
        score_24h_histgb,
        apply_spike_calibration,
        publish_forecast_outputs,
        publish_orchestration_data_summary,
    ],
)


daily_forecast_schedule = ScheduleDefinition(
    job=forecast_job,
    cron_schedule="30 2 * * *",
)


defs = Definitions(
    assets=[
        traffic_source_reports_raw,
        traffic_source_accounts_raw,
        hourly_timezone_joined,
        daily_ad_breakdown,
        historical_6h_prediction_review,
        historical_6h_quality_summary,
        features_6h_hourly_sequences,
        features_24h_daily_windows,
        score_6h_gru,
        score_24h_histgb,
        apply_spike_calibration,
        publish_forecast_outputs,
        publish_orchestration_data_summary,
    ],
    schedules=[daily_forecast_schedule],
)
