from __future__ import annotations

import json
import os
import subprocess
import sys
import asyncio
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from dagster import AssetSelection, Definitions, ScheduleDefinition, asset, define_asset_job


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import postgresql

SCRIPTS = ROOT / "scripts"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"

FINAL_24H_MODEL_DIR = MODELS / "adunbox_daily_24h_histgb_full_db_production"
FINAL_6H_ANCHOR_MODEL_DIR = MODELS / "adunbox_entity_history_lgbm_6h_anchor_v2"
FINAL_6H_BUSINESS_MODEL_DIR = MODELS / "adunbox_entity_history_lgbm_6h_business_v3"

DEFAULT_24H_FORECAST = ROOT / "adunbox_daily_24h_latest_forecasts.csv"
DEFAULT_24H_SERVED = DEFAULT_24H_FORECAST
DEFAULT_24H_MONITOR = OUTPUTS / "adunbox_daily_24h_quality_monitor.csv"
DEFAULT_6H_FORECAST = OUTPUTS / "adunbox_6h_latest_forecasts.csv"
PRODUCTION_MANIFEST = OUTPUTS / "adunbox_production_model_manifest.json"
DB_6H_EXTRACT = OUTPUTS / "adunbox_6h_hourly_db_extract.csv"
DB_24H_EXTRACT = OUTPUTS / "adunbox_24h_daily_db_extract.csv"
FEATURE_CACHE_6H = OUTPUTS / "adunbox_6h_latest_feature_cache.joblib"
FEATURE_CACHE_24H = OUTPUTS / "adunbox_24h_latest_feature_cache.joblib"
FORECAST_PERSISTENCE_STATUS = OUTPUTS / "adunbox_forecast_persistence_status.json"


ADUNBOX_6H_REPORTS_SQL = """
SELECT
    r.id,
    r.report_id,
    r.date,
    r.company_id,
    r.traffic_source_id,
    r.traffic_source_config_id,
    r.account_id,
    r.campaign_id,
    r.adset_id,
    r.ad_id,
    r.impressions,
    r.inline_link_clicks,
    r.clicks,
    r.spend,
    r.inline_link_click_ctr,
    r.created_at,
    r.updated_at,
    r.site_id,
    r.results,
    r.tracker_revenue,
    r.tracker_conversions,
    r.synced_at,
    a.timezone
FROM adunbox_traffic_source_reports r
LEFT JOIN adunbox_traffic_source_accounts a
    ON r.account_id = a.id
   AND r.company_id = a.company_id
   AND r.traffic_source_id = a.traffic_source_id
   AND r.traffic_source_config_id = a.traffic_source_config_id
WHERE r.date >= (
    COALESCE($3::timestamptz, NOW())
) - (($1::int || ' days')::interval)
  AND r.date <= COALESCE($3::timestamptz, NOW())
ORDER BY r.date DESC
LIMIT COALESCE(NULLIF($2::int, 0), 2147483647)
"""


ADUNBOX_24H_DAILY_SQL = """
SELECT
    entity_type,
    date,
    timezone,
    account_id,
    campaign_id,
    adset_id,
    ad_id,
    spend,
    impressions,
    inline_link_clicks,
    tracker_conversions,
    tracker_revenue,
    conversions,
    conversions_value
FROM adunbox_daily_breakdown_kpis
WHERE entity_type = 'ad'
  AND ad_id IS NOT NULL
  AND date >= (
    COALESCE($3::timestamptz, NOW())
) - (($1::int || ' days')::interval)
  AND date <= COALESCE($3::timestamptz, NOW())
ORDER BY date DESC
LIMIT COALESCE(NULLIF($2::int, 0), 2147483647)
"""


def _run_python(script: Path, *args: str, env: dict[str, str] | None = None) -> None:
    cmd = [sys.executable, str(script), *args]
    result = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nSTDOUT:\n"
            + result.stdout
            + "\n\nSTDERR:\n"
            + result.stderr
        )


def _require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _use_database_extract() -> bool:
    return os.getenv("ADUNBOX_USE_DATABASE", "false").strip().lower() in {"1", "true", "yes", "y"}


def _write_forecasts_to_db_enabled() -> bool:
    return os.getenv("ADUNBOX_WRITE_FORECASTS_TO_DB", "false").strip().lower() in {"1", "true", "yes", "y"}


def _write_features_to_db_enabled() -> bool:
    return os.getenv("ADUNBOX_WRITE_FEATURES_TO_DB", "false").strip().lower() in {"1", "true", "yes", "y"}


def _read_sql_override(env_name: str, default_sql: str) -> str:
    sql_path = os.getenv(env_name)
    if sql_path:
        return Path(sql_path).read_text(encoding="utf-8")
    return default_sql


def _parse_anchor_date(value: str | None):
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="raise")
    if getattr(parsed, "tzinfo", None) is None:
        return parsed.to_pydatetime().replace(tzinfo=timezone.utc)
    return parsed.to_pydatetime()


async def _query_postgres_to_frame(sql: str) -> pd.DataFrame:
    rows = await postgresql.query(sql)
    return pd.DataFrame(rows)


async def _query_postgres_to_frame_with_args(sql: str, *args) -> pd.DataFrame:
    rows = await postgresql.query(sql, *args)
    return pd.DataFrame(rows)


def _query_postgres_to_csv(sql: str, output_path: Path, *args) -> str:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    df = asyncio.run(_query_postgres_to_frame_with_args(sql, *args)) if args else asyncio.run(_query_postgres_to_frame(sql))
    if df.empty:
        raise ValueError(f"Database query returned zero rows for output: {output_path}")
    df.to_csv(output_path, index=False)
    return str(output_path)


def _query_postgres_to_csv_with_retries(
    sql: str,
    output_path: Path,
    attempts: list[tuple[int, int, object]],
) -> str:
    last_error: Exception | None = None
    for lookback_days, row_limit, anchor_date in attempts:
        try:
            return _query_postgres_to_csv(sql, output_path, lookback_days, row_limit, anchor_date)
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"No database extract attempts configured for {output_path}")


def _json_safe(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _db_datetime(value):
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if hasattr(value, "isoformat"):
        if getattr(value, "tzinfo", None) is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    try:
        parsed = pd.to_datetime(value, errors="raise")
    except Exception:
        return None
    dt = parsed.to_pydatetime()
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _first_existing(row: pd.Series, names: list[str]):
    for name in names:
        if name in row and not pd.isna(row[name]):
            return row[name]
    return None


async def _persist_forecast_csv_to_postgres(csv_path: Path, horizon: str, table_name: str) -> int:
    if not csv_path.exists():
        return 0

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id BIGSERIAL PRIMARY KEY,
        forecast_horizon TEXT NOT NULL,
        account_id TEXT,
        campaign_id TEXT,
        adset_id TEXT,
        ad_id TEXT,
        forecast_anchor TIMESTAMPTZ,
        forecast_window_start TIMESTAMPTZ,
        forecast_window_end TIMESTAMPTZ,
        forecast_confidence TEXT,
        forecast_status TEXT,
        benchmark_source TEXT,
        model_source TEXT,
        payload JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """
    await postgresql.execute(create_sql)
    await postgresql.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ad_created ON {table_name}(ad_id, created_at DESC)")
    await postgresql.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_account_created ON {table_name}(account_id, created_at DESC)")

    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty:
        return 0

    rows: list[tuple] = []
    for _, row in df.iterrows():
        payload = {col: _json_safe(row[col]) for col in df.columns}
        rows.append(
            (
                horizon,
                str(_first_existing(row, ["account_id"]) or ""),
                str(_first_existing(row, ["campaign_id"]) or ""),
                str(_first_existing(row, ["adset_id"]) or ""),
                str(_first_existing(row, ["ad_id"]) or ""),
                _db_datetime(_first_existing(row, ["forecast_anchor_local_date", "anchor_ts", "local_date"])),
                _db_datetime(_first_existing(row, ["forecast_window_start"])),
                _db_datetime(_first_existing(row, ["forecast_window_end"])),
                str(_first_existing(row, ["forecast_confidence"]) or ""),
                str(_first_existing(row, ["forecast_status"]) or ""),
                str(_first_existing(row, ["benchmark_source"]) or ""),
                str(_first_existing(row, ["model_source"]) or ""),
                json.dumps(payload),
            )
        )

    insert_sql = f"""
    INSERT INTO {table_name} (
        forecast_horizon, account_id, campaign_id, adset_id, ad_id,
        forecast_anchor, forecast_window_start, forecast_window_end,
        forecast_confidence, forecast_status, benchmark_source, model_source, payload
    )
    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13::jsonb)
    """
    batch_size = int(os.getenv("ADUNBOX_DB_WRITE_BATCH_SIZE", "1000"))
    for start in range(0, len(rows), batch_size):
        await postgresql.executemany(insert_sql, rows[start : start + batch_size])
    return len(rows)


async def _persist_all_forecasts_to_postgres(path_24h: Path, path_6h: Path, table_name: str) -> tuple[int, int]:
    rows_24h = await _persist_forecast_csv_to_postgres(path_24h, "24h", table_name)
    rows_6h = await _persist_forecast_csv_to_postgres(path_6h, "6h", table_name)
    return rows_24h, rows_6h


async def _persist_feature_cache_to_postgres(cache_path: Path, horizon: str, table_name: str) -> int:
    if not cache_path.exists():
        return 0

    payload = joblib.load(cache_path)
    frame = payload.get("latest")
    if frame is None:
        frame = payload.get("features")
    if frame is None or len(frame) == 0:
        return 0
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)

    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id BIGSERIAL PRIMARY KEY,
        feature_horizon TEXT NOT NULL,
        account_id TEXT,
        campaign_id TEXT,
        adset_id TEXT,
        ad_id TEXT,
        feature_anchor TIMESTAMPTZ,
        feature_source TEXT,
        payload JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """
    await postgresql.execute(create_sql)
    await postgresql.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ad_created ON {table_name}(ad_id, created_at DESC)")
    await postgresql.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_account_created ON {table_name}(account_id, created_at DESC)")
    await postgresql.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_horizon_created ON {table_name}(feature_horizon, created_at DESC)")

    source = str(payload.get("source") or payload.get("hourly_input") or cache_path)
    rows: list[tuple] = []
    for _, row in frame.iterrows():
        row_payload = {col: _json_safe(row[col]) for col in frame.columns}
        rows.append(
            (
                horizon,
                str(_first_existing(row, ["account_id"]) or ""),
                str(_first_existing(row, ["campaign_id"]) or ""),
                str(_first_existing(row, ["adset_id"]) or ""),
                str(_first_existing(row, ["ad_id"]) or ""),
                _db_datetime(_first_existing(row, ["forecast_anchor_local_date", "anchor_ts", "local_date"])),
                source,
                json.dumps(row_payload),
            )
        )

    insert_sql = f"""
    INSERT INTO {table_name} (
        feature_horizon, account_id, campaign_id, adset_id, ad_id,
        feature_anchor, feature_source, payload
    )
    VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb)
    """
    batch_size = int(os.getenv("ADUNBOX_DB_WRITE_BATCH_SIZE", "1000"))
    for start in range(0, len(rows), batch_size):
        await postgresql.executemany(insert_sql, rows[start : start + batch_size])
    return len(rows)


async def _persist_all_features_to_postgres(table_name: str) -> tuple[int, int]:
    rows_24h = await _persist_feature_cache_to_postgres(FEATURE_CACHE_24H, "24h", table_name)
    rows_6h = await _persist_feature_cache_to_postgres(FEATURE_CACHE_6H, "6h", table_name)
    return rows_24h, rows_6h


async def _persist_all_outputs_to_postgres(
    path_24h: Path,
    path_6h: Path,
    forecast_table: str,
    feature_table: str,
    write_forecasts: bool,
    write_features: bool,
) -> tuple[int, int, int, int]:
    rows_24h = rows_6h = feature_rows_24h = feature_rows_6h = 0
    if write_forecasts:
        rows_24h, rows_6h = await _persist_all_forecasts_to_postgres(path_24h, path_6h, forecast_table)
    if write_features:
        feature_rows_24h, feature_rows_6h = await _persist_all_features_to_postgres(feature_table)
    return rows_24h, rows_6h, feature_rows_24h, feature_rows_6h


@asset(group_name="production_model_registry")
def final_model_registry(context) -> str:
    """Validate that only the final 6h and 24h production model folders are used."""
    required = {
        "24h_model": FINAL_24H_MODEL_DIR,
        "6h_anchor_model": FINAL_6H_ANCHOR_MODEL_DIR,
        "6h_business_model": FINAL_6H_BUSINESS_MODEL_DIR,
    }
    for label, path in required.items():
        _require_path(path, label)
        _require_path(path / "metadata.joblib", f"{label} metadata")

    model_24h_meta = joblib.load(FINAL_24H_MODEL_DIR / "metadata.joblib")
    model_6h_anchor_meta = joblib.load(FINAL_6H_ANCHOR_MODEL_DIR / "metadata.joblib")
    model_6h_business_meta = joblib.load(FINAL_6H_BUSINESS_MODEL_DIR / "metadata.joblib")

    manifest = {
        "24h_model_dir": str(FINAL_24H_MODEL_DIR),
        "6h_anchor_model_dir": str(FINAL_6H_ANCHOR_MODEL_DIR),
        "6h_business_model_dir": str(FINAL_6H_BUSINESS_MODEL_DIR),
        "6h_target_routing": {
            "spend": "6h_anchor_model",
            "impressions": "6h_anchor_model",
            "inline_link_clicks": "6h_anchor_model",
            "tracker_conversions": "6h_business_model",
            "tracker_revenue": "6h_business_model",
        },
        "24h_prediction_mode": "raw_p50",
        "24h_feature_count": len(model_24h_meta.get("feature_cols", [])),
        "6h_anchor_feature_count": len(model_6h_anchor_meta.get("feature_cols", [])),
        "6h_business_feature_count": len(model_6h_business_meta.get("feature_cols", [])),
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    PRODUCTION_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    context.log.info(f"Wrote production manifest: {PRODUCTION_MANIFEST}")
    return str(PRODUCTION_MANIFEST)


@asset(group_name="source_extract")
def postgres_6h_hourly_extract(final_model_registry: str) -> str:
    """Extract 6h source rows.

    Production source:
      Join adunbox traffic source reports with traffic source accounts to attach
      timezone/account context, then export hourly ad-level rows.

    Local smoke source:
      Set ADUNBOX_HOURLY_INPUT to a joined hourly CSV.
    """
    if _use_database_extract():
        sql = _read_sql_override("ADUNBOX_6H_SQL_PATH", ADUNBOX_6H_REPORTS_SQL)
        lookback_days = int(os.getenv("ADUNBOX_6H_DB_LOOKBACK_DAYS", "2"))
        row_limit = int(os.getenv("ADUNBOX_6H_DB_ROW_LIMIT", "1000") or "1000")
        anchor_date = _parse_anchor_date(os.getenv("ADUNBOX_6H_DB_ANCHOR_DATE"))
        return _query_postgres_to_csv(sql, DB_6H_EXTRACT, lookback_days, row_limit, anchor_date)

    hourly_input = Path(os.getenv("ADUNBOX_HOURLY_INPUT", ROOT / "data" / "traffic_reports.csv"))
    _require_path(hourly_input, "6h hourly input")
    return str(hourly_input)


@asset(group_name="source_extract")
def postgres_24h_daily_extract(final_model_registry: str, adunbox_6h_production_ready_manifest: str) -> str:
    """Extract 24h daily source rows.

    Production source:
      Read adunbox daily breakdown KPI table at ad-level grain.

    Local smoke source:
      Set ADUNBOX_DAILY_INPUT to the daily CSV export.
    """
    if _use_database_extract():
        sql = _read_sql_override("ADUNBOX_24H_SQL_PATH", ADUNBOX_24H_DAILY_SQL)
        lookback_days = int(os.getenv("ADUNBOX_24H_DB_LOOKBACK_DAYS", "21"))
        row_limit = int(os.getenv("ADUNBOX_24H_DB_ROW_LIMIT", "5000") or "5000")
        anchor_date = _parse_anchor_date(os.getenv("ADUNBOX_24H_DB_ANCHOR_DATE"))
        retry_enabled = os.getenv("ADUNBOX_24H_DB_RETRY_ON_TIMEOUT", "true").strip().lower() in {"1", "true", "yes", "y"}
        if not retry_enabled:
            return _query_postgres_to_csv(sql, DB_24H_EXTRACT, lookback_days, row_limit, anchor_date)
        attempts = [
            (lookback_days, row_limit, anchor_date),
            (min(lookback_days, 7), min(row_limit, 500), anchor_date),
            (min(lookback_days, 3), min(row_limit, 200), anchor_date),
        ]
        return _query_postgres_to_csv_with_retries(sql, DB_24H_EXTRACT, attempts)

    daily_input = Path(os.getenv("ADUNBOX_DAILY_INPUT", ROOT / "data" / "adunbox_daily_breakdown_kpis.csv"))
    _require_path(daily_input, "24h daily input")
    return str(daily_input)


@asset(group_name="forecast_24h")
def adunbox_24h_raw_forecast(context, postgres_24h_daily_extract: str) -> str:
    """Score 24h daily forecasts using the final production model."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["ADUNBOX_DAILY_INPUT"] = postgres_24h_daily_extract
    env["ADUNBOX_24H_FEATURE_CACHE"] = str(FEATURE_CACHE_24H)
    _run_python(SCRIPTS / "score_adunbox_daily_24h_model.py", "--daily-input", postgres_24h_daily_extract, env=env)
    _require_path(DEFAULT_24H_FORECAST, "24h latest forecast output")
    context.log.info("24h raw forecast scoring completed.")
    return str(DEFAULT_24H_FORECAST)


@asset(group_name="forecast_24h")
def adunbox_24h_served_forecast(context, adunbox_24h_raw_forecast: str) -> str:
    """Return latest 24h forecast with confidence/range columns.

    The production scorer already applies the confidence/range layer for latest
    forecasts. The separate serving-layer script is retained for historical
    backtest files where actual columns are present.
    """
    _require_path(Path(adunbox_24h_raw_forecast), "24h latest forecast")
    context.log.info("24h latest forecast already includes production serving columns.")
    return str(DEFAULT_24H_SERVED)


@asset(group_name="forecast_24h")
def adunbox_24h_quality_monitor(context, adunbox_24h_served_forecast: str) -> str:
    """Write a lightweight production monitor placeholder for latest forecasts.

    Full actual-vs-predicted quality monitoring should run once D1 actuals have
    landed. For live forecasts, this asset records that the forecast file exists.
    """
    forecast_path = Path(adunbox_24h_served_forecast)
    _require_path(forecast_path, "24h latest forecast")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    DEFAULT_24H_MONITOR.write_text(
        json.dumps({"latest_forecast": str(forecast_path), "status": "forecast_written_waiting_for_d1_actuals"}, indent=2),
        encoding="utf-8",
    )
    context.log.info("24h latest forecast monitor placeholder completed.")
    return str(DEFAULT_24H_MONITOR)


@asset(group_name="forecast_6h")
def adunbox_6h_production_ready_manifest(context, postgres_6h_hourly_extract: str) -> str:
    """Score latest 6h forecasts using the final target-routed 6h ensemble.

    The final 6h production model is target-wise:
      spend/impressions/clicks -> anchor_v2
      conversions/revenue -> business_v3
    """
    _require_path(Path(postgres_6h_hourly_extract), "6h hourly source")
    _require_path(FINAL_6H_ANCHOR_MODEL_DIR / "metadata.joblib", "6h anchor metadata")
    _require_path(FINAL_6H_BUSINESS_MODEL_DIR / "metadata.joblib", "6h business metadata")
    env = os.environ.copy()
    env["ADUNBOX_HOURLY_INPUT"] = postgres_6h_hourly_extract
    env["ADUNBOX_6H_FEATURE_CACHE"] = str(FEATURE_CACHE_6H)
    env.setdefault("ADUNBOX_6H_SCORE_CHUNKSIZE", "25000")
    _run_python(SCRIPTS / "score_adunbox_entity_history_lgbm_6h_model.py", "--hourly-input", postgres_6h_hourly_extract, env=env)
    _require_path(DEFAULT_6H_FORECAST, "6h latest forecast output")
    context.log.info("6h production scoring completed.")
    return str(DEFAULT_6H_FORECAST)


@asset(group_name="forecast_persistence")
def adunbox_forecast_postgres_sink(
    context,
    adunbox_24h_served_forecast: str,
    adunbox_6h_production_ready_manifest: str,
) -> str:
    """Optionally persist latest forecasts back into PostgreSQL.

    Enable with:
      ADUNBOX_WRITE_FORECASTS_TO_DB=true

    The table stores core IDs as columns and the complete forecast row in JSONB,
    so schema changes in model output do not break production inserts.
    """
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    if not _write_forecasts_to_db_enabled() and not _write_features_to_db_enabled():
        status = {
            "enabled": False,
            "status": "skipped",
            "reason": "Set ADUNBOX_WRITE_FORECASTS_TO_DB=true and/or ADUNBOX_WRITE_FEATURES_TO_DB=true to persist outputs to PostgreSQL.",
        }
        FORECAST_PERSISTENCE_STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
        context.log.info("Forecast/feature DB persistence skipped.")
        return str(FORECAST_PERSISTENCE_STATUS)

    forecast_table = os.getenv("ADUNBOX_FORECAST_TABLE", "adunbox_model_forecasts")
    feature_table = os.getenv("ADUNBOX_FEATURE_TABLE", "adunbox_model_feature_cache")
    write_forecasts = _write_forecasts_to_db_enabled()
    write_features = _write_features_to_db_enabled()
    rows_24h, rows_6h, feature_rows_24h, feature_rows_6h = asyncio.run(
        _persist_all_outputs_to_postgres(
            Path(adunbox_24h_served_forecast),
            Path(adunbox_6h_production_ready_manifest),
            forecast_table,
            feature_table,
            write_forecasts,
            write_features,
        )
    )
    status = {
        "forecast_write_enabled": write_forecasts,
        "feature_write_enabled": write_features,
        "forecast_table": forecast_table,
        "feature_table": feature_table,
        "rows_24h": rows_24h,
        "rows_6h": rows_6h,
        "feature_rows_24h": feature_rows_24h,
        "feature_rows_6h": feature_rows_6h,
        "status": "written",
    }
    FORECAST_PERSISTENCE_STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
    context.log.info(f"Persisted outputs to PostgreSQL: {status}")
    return str(FORECAST_PERSISTENCE_STATUS)


production_job = define_asset_job(
    "adunbox_production_forecast_job",
    selection=AssetSelection.all(),
)

daily_schedule = ScheduleDefinition(
    job=production_job,
    cron_schedule="15 0 * * *",
    execution_timezone="Asia/Kolkata",
)

defs = Definitions(
    assets=[
        final_model_registry,
        postgres_6h_hourly_extract,
        postgres_24h_daily_extract,
        adunbox_24h_raw_forecast,
        adunbox_24h_served_forecast,
        adunbox_24h_quality_monitor,
        adunbox_6h_production_ready_manifest,
        adunbox_forecast_postgres_sink,
    ],
    jobs=[production_job],
    schedules=[daily_schedule],
)
