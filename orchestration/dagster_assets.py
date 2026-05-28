from __future__ import annotations

from pathlib import Path

from dagster import Definitions, ScheduleDefinition, asset, define_asset_job


ROOT = Path(__file__).resolve().parents[1]


@asset
def traffic_source_reports_raw():
    """Fetch hourly traffic source report rows from the production database/API."""
    # Production implementation:
    # SELECT hourly report rows from Adunbox traffic source reports.
    # Persist to object storage or a warehouse staging table.
    return {"status": "placeholder"}


@asset
def traffic_source_accounts_raw():
    """Fetch traffic source account metadata, especially timezone."""
    # Production implementation:
    # SELECT account_id, traffic_source_id, timezone, active status.
    return {"status": "placeholder"}


@asset(deps=[traffic_source_reports_raw, traffic_source_accounts_raw])
def hourly_timezone_joined():
    """Join hourly reports with account timezone and build local timestamps."""
    return {"status": "placeholder"}


@asset
def daily_ad_breakdown():
    """Fetch daily ad-level data for 24h forecasting."""
    return {"status": "placeholder"}


@asset(deps=[hourly_timezone_joined])
def features_6h_hourly_sequences():
    """Build last-168-hour sequences for each active ad."""
    return {"status": "placeholder", "seq_hours": 168}


@asset(deps=[daily_ad_breakdown])
def features_24h_daily_windows():
    """Build last-7-day feature windows for each ad/date."""
    return {"status": "placeholder", "history_days": 7}


@asset(deps=[features_6h_hourly_sequences])
def score_6h_gru():
    """Load the trained 6h GRU and score next 6h raw metrics."""
    model_dir = ROOT / "models" / "adunbox_entity_history_gru_168h_padded_6h"
    return {"status": "placeholder", "model_dir": str(model_dir)}


@asset(deps=[features_24h_daily_windows])
def score_24h_histgb():
    """Load trained 24h daily HistGB models and score next-day raw metrics."""
    model_dir = ROOT / "models" / "adunbox_daily_24h_histgb"
    return {"status": "placeholder", "model_dir": str(model_dir)}


@asset(deps=[score_24h_histgb])
def apply_spike_calibration():
    """Apply spike-risk and calibrated forecast logic for edge cases."""
    return {"status": "placeholder"}


@asset(deps=[score_6h_gru, apply_spike_calibration])
def publish_forecast_outputs():
    """Publish final forecast outputs to DB tables, object storage, or dashboards."""
    return {"status": "placeholder"}


forecast_job = define_asset_job(
    name="adunbox_forecast_job",
    selection=[
        traffic_source_reports_raw,
        traffic_source_accounts_raw,
        hourly_timezone_joined,
        daily_ad_breakdown,
        features_6h_hourly_sequences,
        features_24h_daily_windows,
        score_6h_gru,
        score_24h_histgb,
        apply_spike_calibration,
        publish_forecast_outputs,
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
        features_6h_hourly_sequences,
        features_24h_daily_windows,
        score_6h_gru,
        score_24h_histgb,
        apply_spike_calibration,
        publish_forecast_outputs,
    ],
    schedules=[daily_forecast_schedule],
)

