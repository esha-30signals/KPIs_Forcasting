import json
import os
import re
import subprocess
import sys
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import joblib
import pandas as pd
from dagster import (
    AssetSelection,
    Definitions,
    DynamicPartitionsDefinition,
    AssetExecutionContext,
    ScheduleDefinition,
    asset,
    define_asset_job,
)


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


account_partitions = DynamicPartitionsDefinition(name="account_id")


ADUNBOX_6H_REPORTS_SQL = """
WITH active_rows AS (
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
        NULL::text AS results,
        {reports_revenue_expr} AS tracker_revenue,
        {reports_conversions_expr} AS tracker_conversions,
        NULL::timestamptz AS synced_at,
        a.timezone
    FROM {traffic_source_reports_table} r
    LEFT JOIN {traffic_source_accounts_table} a
        ON r.account_id = a.id
       AND r.company_id = a.company_id
       AND r.traffic_source_id = a.traffic_source_id
       AND r.traffic_source_config_id = a.traffic_source_config_id
    WHERE r.date >= (
        COALESCE($3::timestamptz, NOW())
    ) - (($1::int || ' days')::interval)
      AND r.date <= COALESCE($3::timestamptz, NOW())
      AND r.ad_id IS NOT NULL
      AND {active_hierarchy_filter_6h}
      AND (
          COALESCE($4::text, '') = ''
          OR r.ad_id::text = ANY(string_to_array($4::text, ','))
      )
      AND (
          COALESCE($5::text, '') = ''
          OR r.account_id::text = ANY(string_to_array($5::text, ','))
      )
),
selected_ads AS (
    SELECT ad_id
    FROM active_rows
    GROUP BY ad_id
    ORDER BY SUM(COALESCE(spend, 0)) DESC, MAX(date) DESC
    LIMIT COALESCE(NULLIF($2::int, 0), 2147483647)
)
SELECT active_rows.*
FROM active_rows
JOIN selected_ads USING (ad_id)
ORDER BY active_rows.ad_id, active_rows.date ASC
"""


ADUNBOX_24H_DAILY_SQL = """
WITH active_rows AS (
    SELECT
        d.entity_type,
        d.date,
        d.timezone,
        d.account_id,
        d.campaign_id,
        d.adset_id,
        d.ad_id,
        d.spend,
        d.impressions,
        d.inline_link_clicks,
        {daily_conversions_expr} AS tracker_conversions,
        {daily_revenue_expr} AS tracker_revenue,
        d.conversions,
        d.conversions_value
    FROM {daily_breakdown_kpis_table} d
    WHERE d.entity_type = 'ad'
      AND d.ad_id IS NOT NULL
      AND d.date >= (
        COALESCE($3::timestamptz, NOW())
    ) - (($1::int || ' days')::interval)
      AND d.date <= COALESCE($3::timestamptz, NOW())
      AND {active_hierarchy_filter_24h}
      AND (
          COALESCE($4::text, '') = ''
          OR d.ad_id::text = ANY(string_to_array($4::text, ','))
      )
      AND (
          COALESCE($5::text, '') = ''
          OR d.account_id::text = ANY(string_to_array($5::text, ','))
      )
),
selected_ads AS (
    SELECT ad_id
    FROM active_rows
    GROUP BY ad_id
    ORDER BY SUM(COALESCE(spend, 0)) DESC, MAX(date) DESC
    LIMIT COALESCE(NULLIF($2::int, 0), 2147483647)
)
SELECT active_rows.*
FROM active_rows
JOIN selected_ads USING (ad_id)
ORDER BY active_rows.ad_id, active_rows.date ASC
"""


ADUNBOX_6H_REPORTS_ELIGIBILITY_VIEW_SQL = """
WITH active_rows AS (
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
        NULL::text AS results,
        {reports_revenue_expr} AS tracker_revenue,
        {reports_conversions_expr} AS tracker_conversions,
        NULL::timestamptz AS synced_at,
        a.timezone,
        e.eligibility_status,
        e.scheduled_active_from,
        e.scheduled_active_until
    FROM {traffic_source_reports_table} r
    LEFT JOIN {traffic_source_accounts_table} a
        ON r.account_id = a.id
       AND r.company_id = a.company_id
       AND r.traffic_source_id = a.traffic_source_id
       AND r.traffic_source_config_id = a.traffic_source_config_id
    JOIN {forecast_eligible_ads_table} e
      ON e.account_id = r.account_id
     AND e.campaign_id = r.campaign_id
     AND e.adset_id = r.adset_id
     AND e.ad_id = r.ad_id
    WHERE r.date >= (
        COALESCE($3::timestamptz, NOW())
    ) - (($1::int || ' days')::interval)
      AND r.date <= COALESCE($3::timestamptz, NOW())
      AND r.ad_id IS NOT NULL
      AND (
          e.is_currently_active = TRUE
          OR (
              e.scheduled_active_from IS NOT NULL
              AND e.scheduled_active_from <= COALESCE($3::timestamptz, NOW()) + interval '6 hours'
              AND COALESCE(e.scheduled_active_until, COALESCE($3::timestamptz, NOW()) + interval '6 hours')
                  >= COALESCE($3::timestamptz, NOW())
          )
      )
      AND (
          COALESCE($4::text, '') = ''
          OR r.ad_id::text = ANY(string_to_array($4::text, ','))
      )
      AND (
          COALESCE($5::text, '') = ''
          OR r.account_id::text = ANY(string_to_array($5::text, ','))
      )
),
selected_ads AS (
    SELECT ad_id
    FROM active_rows
    GROUP BY ad_id
    ORDER BY SUM(COALESCE(spend, 0)) DESC, MAX(date) DESC
    LIMIT COALESCE(NULLIF($2::int, 0), 2147483647)
)
SELECT active_rows.*
FROM active_rows
JOIN selected_ads USING (ad_id)
ORDER BY active_rows.ad_id, active_rows.date ASC
"""


ADUNBOX_24H_DAILY_ELIGIBILITY_VIEW_SQL = """
WITH active_rows AS (
    SELECT
        d.entity_type,
        d.date,
        d.timezone,
        d.account_id,
        d.campaign_id,
        d.adset_id,
        d.ad_id,
        d.spend,
        d.impressions,
        d.inline_link_clicks,
        {daily_conversions_expr} AS tracker_conversions,
        {daily_revenue_expr} AS tracker_revenue,
        d.conversions,
        d.conversions_value,
        e.eligibility_status,
        e.scheduled_active_from,
        e.scheduled_active_until
    FROM {daily_breakdown_kpis_table} d
    JOIN {forecast_eligible_ads_table} e
      ON e.account_id = d.account_id
     AND e.campaign_id = d.campaign_id
     AND e.adset_id = d.adset_id
     AND e.ad_id = d.ad_id
    WHERE d.entity_type = 'ad'
      AND d.ad_id IS NOT NULL
      AND d.date >= (
        COALESCE($3::timestamptz, NOW())
    ) - (($1::int || ' days')::interval)
      AND d.date <= COALESCE($3::timestamptz, NOW())
      AND (
          e.is_currently_active = TRUE
          OR (
              e.scheduled_active_from IS NOT NULL
              AND e.scheduled_active_from <= COALESCE($3::timestamptz, NOW()) + interval '24 hours'
              AND COALESCE(e.scheduled_active_until, COALESCE($3::timestamptz, NOW()) + interval '24 hours')
                  >= COALESCE($3::timestamptz, NOW())
          )
      )
      AND (
          COALESCE($4::text, '') = ''
          OR d.ad_id::text = ANY(string_to_array($4::text, ','))
      )
      AND (
          COALESCE($5::text, '') = ''
          OR d.account_id::text = ANY(string_to_array($5::text, ','))
      )
),
selected_ads AS (
    SELECT ad_id
    FROM active_rows
    GROUP BY ad_id
    ORDER BY SUM(COALESCE(spend, 0)) DESC, MAX(date) DESC
    LIMIT COALESCE(NULLIF($2::int, 0), 2147483647)
)
SELECT active_rows.*
FROM active_rows
JOIN selected_ads USING (ad_id)
ORDER BY active_rows.ad_id, active_rows.date ASC
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


def _clean_local_outputs_after_db_write() -> bool:
    return os.getenv("ADUNBOX_CLEAN_LOCAL_OUTPUTS_AFTER_DB_WRITE", "false").strip().lower() in {"1", "true", "yes", "y"}


def _use_eligibility_view() -> bool:
    return os.getenv("ADUNBOX_USE_FORECAST_ELIGIBILITY_VIEW", "false").strip().lower() in {"1", "true", "yes", "y"}


def _safe_table_identifier(value: str) -> str:
    """Allow plain or schema-qualified table names without allowing SQL fragments."""
    text = value.strip()
    if not text:
        raise ValueError("Database table name cannot be empty")
    parts = text.split(".")
    for part in parts:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
            raise ValueError(f"Unsafe database table identifier: {value!r}")
    return ".".join(parts)


def _safe_column_expr(env_name: str, default_column: str, alias: str = "r") -> str:
    """Render a safe aliased column expression, or NULL when explicitly disabled."""
    configured = os.getenv(env_name)
    column = default_column if configured is None else configured.strip()
    if not column:
        return "NULL::double precision"
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", column):
        raise ValueError(f"Unsafe database column identifier for {env_name}: {column!r}")
    return f"{alias}.{column}"


async def _table_columns(table_name: str) -> set[str]:
    rows = await postgresql.query(
        """
        SELECT a.attname AS column_name
        FROM pg_attribute a
        WHERE a.attrelid = to_regclass($1)
          AND a.attnum > 0
          AND NOT a.attisdropped
        """,
        table_name,
    )
    return {str(row["column_name"]) for row in rows}


def _resolve_column_name(columns: set[str], requested: str | None, fallbacks: list[str]) -> str:
    candidates = []
    if requested and requested.strip():
        candidates.append(requested.strip())
    candidates.extend(fallbacks)
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate):
            continue
        if candidate in columns:
            return candidate
    raise ValueError(
        "None of the configured metric columns exist on the source table. "
        f"Tried: {', '.join(candidates)}. Available columns: {', '.join(sorted(columns))}"
    )


async def _resolve_database_metric_columns() -> dict[str, str]:
    """Resolve revenue/conversion column aliases against the connected DB.

    This keeps the pipeline stable across Adunbox/Vibelets table variants where
    revenue may be stored as conversions_value, conversion_values, or tracker_revenue.
    """
    try:
        ctx = _db_table_context_base()
        reports_columns = await _table_columns(ctx["traffic_source_reports_table"])
        daily_columns = await _table_columns(ctx["daily_breakdown_kpis_table"])
        return {
            "ADUNBOX_6H_REPORTS_REVENUE_COLUMN": _resolve_column_name(
                reports_columns,
                os.getenv("ADUNBOX_6H_REPORTS_REVENUE_COLUMN"),
                ["conversions_value", "conversion_values", "tracker_revenue"],
            ),
            "ADUNBOX_6H_REPORTS_CONVERSIONS_COLUMN": _resolve_column_name(
                reports_columns,
                os.getenv("ADUNBOX_6H_REPORTS_CONVERSIONS_COLUMN"),
                ["conversions", "tracker_conversions"],
            ),
            "ADUNBOX_24H_DAILY_REVENUE_COLUMN": _resolve_column_name(
                daily_columns,
                os.getenv("ADUNBOX_24H_DAILY_REVENUE_COLUMN"),
                ["conversions_value", "conversion_values", "tracker_revenue"],
            ),
            "ADUNBOX_24H_DAILY_CONVERSIONS_COLUMN": _resolve_column_name(
                daily_columns,
                os.getenv("ADUNBOX_24H_DAILY_CONVERSIONS_COLUMN"),
                ["conversions", "tracker_conversions"],
            ),
        }
    finally:
        # asyncpg pools are bound to the event loop that created them. The
        # extract itself runs in a separate asyncio.run(), so close this
        # short-lived preflight pool before the real query starts.
        await postgresql.disconnect()


def _apply_resolved_database_metric_columns(context: AssetExecutionContext | None = None) -> dict[str, str]:
    resolved = asyncio.run(_resolve_database_metric_columns())
    for env_name, column in resolved.items():
        os.environ[env_name] = column
    if context is not None:
        context.log.info(f"Resolved DB metric columns: {resolved}")
    return resolved


def _require_active_hierarchy() -> bool:
    return os.getenv("ADUNBOX_REQUIRE_ACTIVE_HIERARCHY", "true").strip().lower() in {"1", "true", "yes", "y"}


def _active_hierarchy_filter_6h() -> str:
    if not _require_active_hierarchy():
        return "TRUE"
    ctx = _db_table_context_base()
    return f"""
      UPPER(COALESCE(a.status, '')) = 'ACTIVE'
      AND EXISTS (
          SELECT 1
          FROM {ctx["traffic_source_campaigns_table"]} c
          WHERE c.id = r.campaign_id
            AND UPPER(COALESCE(c.status, '')) = 'ACTIVE'
      )
      AND EXISTS (
          SELECT 1
          FROM {ctx["traffic_source_adsets_table"]} s
          WHERE s.id = r.adset_id
            AND UPPER(COALESCE(s.status, '')) = 'ACTIVE'
      )
      AND EXISTS (
          SELECT 1
          FROM {ctx["traffic_source_ads_table"]} ad
          WHERE ad.id = r.ad_id
            AND UPPER(COALESCE(ad.status, '')) = 'ACTIVE'
      )
    """.strip()


def _active_hierarchy_filter_24h() -> str:
    if not _require_active_hierarchy():
        return "TRUE"
    ctx = _db_table_context_base()
    return f"""
      EXISTS (
          SELECT 1
          FROM {ctx["traffic_source_accounts_table"]} a
          WHERE a.id = d.account_id
            AND UPPER(COALESCE(a.status, '')) = 'ACTIVE'
      )
      AND EXISTS (
          SELECT 1
          FROM {ctx["traffic_source_campaigns_table"]} c
          WHERE c.id = d.campaign_id
            AND UPPER(COALESCE(c.status, '')) = 'ACTIVE'
      )
      AND EXISTS (
          SELECT 1
          FROM {ctx["traffic_source_adsets_table"]} s
          WHERE s.id = d.adset_id
            AND UPPER(COALESCE(s.status, '')) = 'ACTIVE'
      )
      AND EXISTS (
          SELECT 1
          FROM {ctx["traffic_source_ads_table"]} ad
          WHERE ad.id = d.ad_id
            AND UPPER(COALESCE(ad.status, '')) = 'ACTIVE'
      )
    """.strip()


def _db_table_context_base() -> dict[str, str]:
    prefix = os.getenv("ADUNBOX_DB_TABLE_PREFIX", "adunbox_")

    def table(env_name: str, suffix: str) -> str:
        configured = os.getenv(env_name)
        return _safe_table_identifier(configured if configured else f"{prefix}{suffix}")

    return {
        "traffic_source_reports_table": table("ADUNBOX_TRAFFIC_SOURCE_REPORTS_TABLE", "traffic_source_reports"),
        "traffic_source_accounts_table": table("ADUNBOX_TRAFFIC_SOURCE_ACCOUNTS_TABLE", "traffic_source_accounts"),
        "traffic_source_campaigns_table": table("ADUNBOX_TRAFFIC_SOURCE_CAMPAIGNS_TABLE", "traffic_source_campaigns"),
        "traffic_source_adsets_table": table("ADUNBOX_TRAFFIC_SOURCE_ADSETS_TABLE", "traffic_source_adsets"),
        "traffic_source_ads_table": table("ADUNBOX_TRAFFIC_SOURCE_ADS_TABLE", "traffic_source_ads"),
        "daily_breakdown_kpis_table": table("ADUNBOX_DAILY_BREAKDOWN_KPIS_TABLE", "daily_breakdown_kpis"),
        "forecast_eligible_ads_table": table("ADUNBOX_FORECAST_ELIGIBLE_ADS_TABLE", "forecast_eligible_ads"),
    }


def _db_table_context() -> dict[str, str]:
    ctx = _db_table_context_base()
    ctx.update(
        {
            "reports_revenue_expr": _safe_column_expr("ADUNBOX_6H_REPORTS_REVENUE_COLUMN", "tracker_revenue"),
            "reports_conversions_expr": _safe_column_expr("ADUNBOX_6H_REPORTS_CONVERSIONS_COLUMN", "tracker_conversions"),
            "daily_revenue_expr": _safe_column_expr("ADUNBOX_24H_DAILY_REVENUE_COLUMN", "tracker_revenue", alias="d"),
            "daily_conversions_expr": _safe_column_expr("ADUNBOX_24H_DAILY_CONVERSIONS_COLUMN", "tracker_conversions", alias="d"),
            "active_hierarchy_filter_6h": _active_hierarchy_filter_6h(),
            "active_hierarchy_filter_24h": _active_hierarchy_filter_24h(),
        }
    )
    return ctx


def _render_db_sql(sql: str) -> str:
    return sql.format(**_db_table_context())


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
        debug_path = output_path.with_name(f"{output_path.stem}__zero_rows_debug.json")
        debug_path.write_text(
            json.dumps(
                {
                    "output_path": str(output_path),
                    "reason": "Database query returned zero rows",
                    "args": [_json_safe(arg) for arg in args],
                    "table_context": _db_table_context(),
                    "require_active_hierarchy": _require_active_hierarchy(),
                    "debug_ad_ids": os.getenv("ADUNBOX_DEBUG_AD_IDS", ""),
                    "debug_account_ids": os.getenv("ADUNBOX_DEBUG_ACCOUNT_IDS", ""),
                    "rendered_sql": sql,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise ValueError(f"Database query returned zero rows for output: {output_path}. Debug written to: {debug_path}")
    df.to_csv(output_path, index=False)
    return str(output_path)


def _query_postgres_to_csv_with_retries(
    sql: str,
    output_path: Path,
    attempts: list[tuple],
) -> str:
    last_error: Exception | None = None
    for args in attempts:
        try:
            return _query_postgres_to_csv(sql, output_path, *args)
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


def _forecast_window(anchor, horizon: str) -> tuple[object | None, object | None]:
    start = _db_datetime(anchor)
    if start is None:
        return None, None
    hours = 24 if horizon == "24h" else 6 if horizon == "6h" else None
    if hours is None:
        return start, None
    return start, start + pd.Timedelta(hours=hours)


def _local_datetime(value) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value, errors="raise")
    except Exception:
        return None
    dt = parsed.to_pydatetime()
    if getattr(dt, "tzinfo", None) is not None:
        dt = dt.replace(tzinfo=None)
    return dt


def _local_date(value):
    dt = _local_datetime(value)
    return dt.date() if dt is not None else None


def _utc_to_account_local(value, tz_name: str | None) -> datetime | None:
    dt = _db_datetime(value)
    if dt is None:
        return None
    if not tz_name:
        return dt.replace(tzinfo=None)
    try:
        return dt.astimezone(ZoneInfo(str(tz_name))).replace(tzinfo=None)
    except (ZoneInfoNotFoundError, ValueError):
        return dt.replace(tzinfo=None)


def _account_local_datetime(value, tz_name: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value, errors="raise")
    except Exception:
        return None
    dt = parsed.to_pydatetime()
    if getattr(dt, "tzinfo", None) is None:
        return dt
    if not tz_name:
        return dt.replace(tzinfo=None)
    try:
        return dt.astimezone(ZoneInfo(str(tz_name))).replace(tzinfo=None)
    except (ZoneInfoNotFoundError, ValueError):
        return dt.replace(tzinfo=None)


def _forecast_status(row: pd.Series) -> str:
    existing = _first_existing(row, ["forecast_status"])
    if existing:
        return str(existing)
    benchmark_source = str(_first_existing(row, ["benchmark_source"]) or "").strip()
    confidence = str(_first_existing(row, ["forecast_confidence"]) or "").strip().upper()
    if benchmark_source == "insufficient_history":
        return "insufficient_history_monitoring"
    if benchmark_source and benchmark_source != "model_point":
        return "hierarchical_benchmark_forecast"
    if confidence == "LOW":
        return "low_confidence_forecast"
    return "model_forecast"


def _forecast_confidence_score(row: pd.Series) -> float | None:
    existing = _first_existing(row, ["forecast_confidence_score", "confidence_score"])
    if existing is not None and not pd.isna(existing):
        try:
            return float(existing)
        except (TypeError, ValueError):
            pass
    confidence = str(_first_existing(row, ["forecast_confidence"]) or "").strip().upper()
    return {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}.get(confidence)


def _forecast_results(payload: dict, horizon: str) -> dict:
    prefix = f"{horizon}_"
    raw_metrics = {
        "spend": f"recommended_{prefix}spend",
        "impressions": f"recommended_{prefix}impressions",
        "clicks": f"recommended_{prefix}inline_link_clicks",
        "conversions": f"recommended_{prefix}tracker_conversions",
        "revenue": f"recommended_{prefix}tracker_revenue",
        "roas": f"recommended_{prefix}roas",
        "profit": f"recommended_{prefix}profit",
        "ctr": f"recommended_{prefix}ctr",
        "cvr": f"recommended_{prefix}cvr",
        "cpc": f"recommended_{prefix}cpc",
        "cpm": f"recommended_{prefix}cpm",
    }
    fallback_metrics = {
        "spend": f"pred_{prefix}spend",
        "impressions": f"pred_{prefix}impressions",
        "clicks": f"pred_{prefix}inline_link_clicks",
        "conversions": f"pred_{prefix}tracker_conversions",
        "revenue": f"pred_{prefix}tracker_revenue",
        "roas": f"pred_{prefix}roas",
        "profit": f"pred_{prefix}profit",
        "ctr": f"pred_{prefix}ctr",
        "cvr": f"pred_{prefix}cvr",
        "cpc": f"pred_{prefix}cpc",
        "cpm": f"pred_{prefix}cpm",
    }
    results: dict[str, object] = {}
    for label, key in raw_metrics.items():
        value = payload.get(key)
        if value is None:
            value = payload.get(fallback_metrics[label])
        results[label] = _json_safe(value)
    for label in ["spend", "impressions", "inline_link_clicks", "tracker_conversions", "tracker_revenue", "roas"]:
        p10_key = f"pred_{prefix}{label}_p10"
        p50_key = f"pred_{prefix}{label}_p50"
        p90_key = f"pred_{prefix}{label}_p90"
        if p10_key in payload or p50_key in payload or p90_key in payload:
            output_label = "clicks" if label == "inline_link_clicks" else "conversions" if label == "tracker_conversions" else "revenue" if label == "tracker_revenue" else label
            results[f"{output_label}_range"] = {
                "p10": _json_safe(payload.get(p10_key)),
                "p50": _json_safe(payload.get(p50_key)),
                "p90": _json_safe(payload.get(p90_key)),
            }
    return _derive_ratio_metrics(results)


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _derive_ratio_metrics(results: dict) -> dict:
    spend = _safe_float(results.get("spend"))
    impressions = _safe_float(results.get("impressions"))
    clicks = _safe_float(results.get("clicks"))
    conversions = _safe_float(results.get("conversions"))
    revenue = _safe_float(results.get("revenue"))

    if results.get("roas") is None and spend and spend > 0:
        results["roas"] = (revenue or 0.0) / spend
    if results.get("profit") is None and revenue is not None and spend is not None:
        results["profit"] = revenue - spend
    if results.get("ctr") is None and impressions and impressions > 0:
        results["ctr"] = (clicks or 0.0) / impressions * 100.0
    if results.get("cvr") is None and clicks and clicks > 0:
        results["cvr"] = (conversions or 0.0) / clicks * 100.0
    if results.get("cpc") is None and clicks and clicks > 0:
        results["cpc"] = (spend or 0.0) / clicks
    if results.get("cpm") is None and impressions and impressions > 0:
        results["cpm"] = (spend or 0.0) / impressions * 1000.0
    return results


def _raw_forecast_results(payload: dict, horizon: str) -> dict:
    prefix = f"{horizon}_"
    raw_metrics = {
        "spend": f"pred_{prefix}spend",
        "impressions": f"pred_{prefix}impressions",
        "clicks": f"pred_{prefix}inline_link_clicks",
        "conversions": f"pred_{prefix}tracker_conversions",
        "revenue": f"pred_{prefix}tracker_revenue",
        "roas": f"pred_{prefix}roas",
        "profit": f"pred_{prefix}profit",
        "ctr": f"pred_{prefix}ctr",
        "cvr": f"pred_{prefix}cvr",
        "cpc": f"pred_{prefix}cpc",
        "cpm": f"pred_{prefix}cpm",
    }
    results = {label: _json_safe(payload.get(key)) for label, key in raw_metrics.items()}
    return _derive_ratio_metrics(results)


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
        timezone TEXT,
        forecast_anchor_local TIMESTAMP,
        forecast_window_start_local TIMESTAMP,
        forecast_window_end_local TIMESTAMP,
        forecast_local_date DATE,
        forecast_confidence TEXT,
        forecast_confidence_score DOUBLE PRECISION,
        forecast_status TEXT,
        benchmark_source TEXT,
        model_source TEXT,
        result_spend DOUBLE PRECISION,
        result_impressions DOUBLE PRECISION,
        result_clicks DOUBLE PRECISION,
        result_conversions DOUBLE PRECISION,
        result_revenue DOUBLE PRECISION,
        result_roas DOUBLE PRECISION,
        result_profit DOUBLE PRECISION,
        result_ctr DOUBLE PRECISION,
        result_cvr DOUBLE PRECISION,
        result_cpc DOUBLE PRECISION,
        result_cpm DOUBLE PRECISION,
        raw_pred_spend DOUBLE PRECISION,
        raw_pred_impressions DOUBLE PRECISION,
        raw_pred_clicks DOUBLE PRECISION,
        raw_pred_conversions DOUBLE PRECISION,
        raw_pred_revenue DOUBLE PRECISION,
        raw_pred_roas DOUBLE PRECISION,
        raw_pred_profit DOUBLE PRECISION,
        raw_pred_ctr DOUBLE PRECISION,
        raw_pred_cvr DOUBLE PRECISION,
        raw_pred_cpc DOUBLE PRECISION,
        raw_pred_cpm DOUBLE PRECISION,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """
    await postgresql.execute(create_sql)
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS timezone TEXT")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS forecast_anchor_local TIMESTAMP")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS forecast_window_start_local TIMESTAMP")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS forecast_window_end_local TIMESTAMP")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS forecast_local_date DATE")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS forecast_confidence_score DOUBLE PRECISION")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS result_spend DOUBLE PRECISION")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS result_impressions DOUBLE PRECISION")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS result_clicks DOUBLE PRECISION")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS result_conversions DOUBLE PRECISION")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS result_revenue DOUBLE PRECISION")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS result_roas DOUBLE PRECISION")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS result_profit DOUBLE PRECISION")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS result_ctr DOUBLE PRECISION")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS result_cvr DOUBLE PRECISION")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS result_cpc DOUBLE PRECISION")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS result_cpm DOUBLE PRECISION")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS raw_pred_spend DOUBLE PRECISION")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS raw_pred_impressions DOUBLE PRECISION")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS raw_pred_clicks DOUBLE PRECISION")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS raw_pred_conversions DOUBLE PRECISION")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS raw_pred_revenue DOUBLE PRECISION")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS raw_pred_roas DOUBLE PRECISION")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS raw_pred_profit DOUBLE PRECISION")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS raw_pred_ctr DOUBLE PRECISION")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS raw_pred_cvr DOUBLE PRECISION")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS raw_pred_cpc DOUBLE PRECISION")
    await postgresql.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS raw_pred_cpm DOUBLE PRECISION")
    # Older local test tables may still have payload JSONB NOT NULL from a previous schema.
    # We no longer write JSON into the forecast table, so make that legacy column nullable if it exists.
    await postgresql.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                  AND column_name = 'payload'
            ) THEN
                EXECUTE 'ALTER TABLE {table_name} ALTER COLUMN payload DROP NOT NULL';
            END IF;
        END $$;
        """
    )
    await postgresql.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ad_created ON {table_name}(ad_id, created_at DESC)")
    await postgresql.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_account_created ON {table_name}(account_id, created_at DESC)")

    df = pd.read_csv(csv_path, low_memory=False)
    if df.empty:
        return 0

    rows: list[tuple] = []
    for _, row in df.iterrows():
        payload = {col: _json_safe(row[col]) for col in df.columns}
        results = _forecast_results(payload, horizon)
        raw_results = _raw_forecast_results(payload, horizon)
        forecast_anchor = _first_existing(row, ["forecast_anchor_local_date", "anchor_ts", "local_date"])
        window_start = _first_existing(row, ["forecast_window_start"]) or forecast_anchor
        window_end = _first_existing(row, ["forecast_window_end"])
        derived_window_start, derived_window_end = _forecast_window(window_start, horizon)
        if window_end is not None:
            derived_window_end = _db_datetime(window_end)
        tz_name = str(_first_existing(row, ["timezone"]) or "")
        local_anchor = _account_local_datetime(forecast_anchor, tz_name)
        local_window_start = _account_local_datetime(window_start, tz_name)
        local_window_end = _account_local_datetime(window_end, tz_name) if window_end is not None else None
        if local_window_start is not None and local_window_end is None:
            local_window_end = local_window_start + pd.Timedelta(hours=24 if horizon == "24h" else 6)
        local_date = _local_date(local_window_start or local_anchor)
        rows.append(
            (
                horizon,
                str(_first_existing(row, ["account_id"]) or ""),
                str(_first_existing(row, ["campaign_id"]) or ""),
                str(_first_existing(row, ["adset_id"]) or ""),
                str(_first_existing(row, ["ad_id"]) or ""),
                _db_datetime(forecast_anchor),
                derived_window_start,
                derived_window_end,
                tz_name,
                local_anchor,
                local_window_start,
                local_window_end,
                local_date,
                str(_first_existing(row, ["forecast_confidence"]) or ""),
                _forecast_confidence_score(row),
                _forecast_status(row),
                str(_first_existing(row, ["benchmark_source"]) or ""),
                str(_first_existing(row, ["model_source"]) or ""),
                results.get("spend"),
                results.get("impressions"),
                results.get("clicks"),
                results.get("conversions"),
                results.get("revenue"),
                results.get("roas"),
                results.get("profit"),
                results.get("ctr"),
                results.get("cvr"),
                results.get("cpc"),
                results.get("cpm"),
                raw_results.get("spend"),
                raw_results.get("impressions"),
                raw_results.get("clicks"),
                raw_results.get("conversions"),
                raw_results.get("revenue"),
                raw_results.get("roas"),
                raw_results.get("profit"),
                raw_results.get("ctr"),
                raw_results.get("cvr"),
                raw_results.get("cpc"),
                raw_results.get("cpm"),
            )
        )

    insert_sql = f"""
    INSERT INTO {table_name} (
        forecast_horizon, account_id, campaign_id, adset_id, ad_id,
        forecast_anchor, forecast_window_start, forecast_window_end,
        timezone, forecast_anchor_local, forecast_window_start_local,
        forecast_window_end_local, forecast_local_date,
        forecast_confidence, forecast_confidence_score, forecast_status, benchmark_source, model_source,
        result_spend, result_impressions, result_clicks,
        result_conversions, result_revenue, result_roas, result_profit,
        result_ctr, result_cvr, result_cpc, result_cpm,
        raw_pred_spend, raw_pred_impressions, raw_pred_clicks,
        raw_pred_conversions, raw_pred_revenue, raw_pred_roas, raw_pred_profit,
        raw_pred_ctr, raw_pred_cvr, raw_pred_cpc, raw_pred_cpm
    )
    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31,$32,$33,$34,$35,$36,$37,$38,$39,$40)
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


async def _persist_single_horizon_outputs_to_postgres(
    csv_path: Path,
    horizon: str,
    feature_cache_path: Path,
    forecast_table: str,
    feature_table: str,
    write_forecasts: bool,
    write_features: bool,
) -> tuple[int, int]:
    forecast_rows = 0
    feature_rows = 0
    if write_forecasts:
        forecast_rows = await _persist_forecast_csv_to_postgres(csv_path, horizon, forecast_table)
    if write_features:
        feature_rows = await _persist_feature_cache_to_postgres(feature_cache_path, horizon, feature_table)
    return forecast_rows, feature_rows


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


@asset(group_name="partition_management")
def discover_account_partitions(context: AssetExecutionContext) -> int:
    """Query active account IDs from the DB and register them as dynamic partitions.

    Run this asset once before triggering any partitioned forecast jobs so that
    Dagster knows which account_id partition keys exist.
    """
    ctx = _db_table_context_base()
    accounts_table = ctx["traffic_source_accounts_table"]
    sql = f"SELECT DISTINCT id::text AS account_id FROM {accounts_table} WHERE UPPER(COALESCE(status, '')) = 'ACTIVE' ORDER BY 1"
    df = asyncio.run(_query_postgres_to_frame(sql))
    if df.empty:
        context.log.warning("No active accounts found — no partitions registered.")
        return 0
    account_ids = df["account_id"].dropna().astype(str).tolist()
    context.instance.add_dynamic_partitions(account_partitions.name, account_ids)
    context.log.info(f"Registered {len(account_ids)} account_id partitions: {account_ids[:10]}{'...' if len(account_ids) > 10 else ''}")
    return len(account_ids)


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


@asset(group_name="source_extract", partitions_def=account_partitions)
def postgres_6h_hourly_extract(context: AssetExecutionContext, final_model_registry: str) -> str:
    """Extract 6h source rows.

    When run as a partition, restricts the extract to the single account_id
    given by context.partition_key. When run without a partition (legacy all-
    accounts mode), falls back to ADUNBOX_DEBUG_ACCOUNT_IDS (empty = all).

    Production source:
      Join adunbox traffic source reports with traffic source accounts to attach
      timezone/account context, then export hourly ad-level rows.

    Local smoke source:
      Set ADUNBOX_HOURLY_INPUT to a joined hourly CSV.
    """
    if _use_database_extract():
        account_id = context.partition_key if context.has_partition_key else os.getenv("ADUNBOX_DEBUG_ACCOUNT_IDS", "")
        output_path = (OUTPUTS / account_id / "adunbox_6h_hourly_db_extract.csv") if account_id else DB_6H_EXTRACT
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _apply_resolved_database_metric_columns(context)
        default_sql = ADUNBOX_6H_REPORTS_ELIGIBILITY_VIEW_SQL if _use_eligibility_view() else ADUNBOX_6H_REPORTS_SQL
        sql = _render_db_sql(_read_sql_override("ADUNBOX_6H_SQL_PATH", default_sql))
        lookback_days = int(os.getenv("ADUNBOX_6H_DB_LOOKBACK_DAYS", "8"))
        row_limit = int(os.getenv("ADUNBOX_6H_DB_ROW_LIMIT", "1000") or "1000")
        anchor_date = _parse_anchor_date(os.getenv("ADUNBOX_6H_DB_ANCHOR_DATE"))
        debug_ad_ids = os.getenv("ADUNBOX_DEBUG_AD_IDS", "")
        return _query_postgres_to_csv(
            sql,
            output_path,
            lookback_days,
            row_limit,
            anchor_date,
            debug_ad_ids,
            account_id,
        )

    hourly_input = Path(os.getenv("ADUNBOX_HOURLY_INPUT", ROOT / "data" / "traffic_reports.csv"))
    _require_path(hourly_input, "6h hourly input")
    return str(hourly_input)


@asset(group_name="source_extract", partitions_def=account_partitions)
def postgres_24h_daily_extract(context: AssetExecutionContext, final_model_registry: str) -> str:
    """Extract 24h daily source rows.

    When run as a partition, restricts the extract to the single account_id
    given by context.partition_key. When run without a partition (legacy all-
    accounts mode), falls back to ADUNBOX_DEBUG_ACCOUNT_IDS (empty = all).

    Production source:
      Read adunbox daily breakdown KPI table at ad-level grain.

    Local smoke source:
      Set ADUNBOX_DAILY_INPUT to the daily CSV export.
    """
    if _use_database_extract():
        account_id = context.partition_key if context.has_partition_key else os.getenv("ADUNBOX_DEBUG_ACCOUNT_IDS", "")
        output_path = (OUTPUTS / account_id / "adunbox_24h_daily_db_extract.csv") if account_id else DB_24H_EXTRACT
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _apply_resolved_database_metric_columns(context)
        default_sql = ADUNBOX_24H_DAILY_ELIGIBILITY_VIEW_SQL if _use_eligibility_view() else ADUNBOX_24H_DAILY_SQL
        sql = _render_db_sql(_read_sql_override("ADUNBOX_24H_SQL_PATH", default_sql))
        lookback_days = int(os.getenv("ADUNBOX_24H_DB_LOOKBACK_DAYS", "21"))
        row_limit = int(os.getenv("ADUNBOX_24H_DB_ROW_LIMIT", "5000") or "5000")
        anchor_date = _parse_anchor_date(os.getenv("ADUNBOX_24H_DB_ANCHOR_DATE"))
        debug_ad_ids = os.getenv("ADUNBOX_DEBUG_AD_IDS", "")
        retry_enabled = os.getenv("ADUNBOX_24H_DB_RETRY_ON_TIMEOUT", "true").strip().lower() in {"1", "true", "yes", "y"}
        if not retry_enabled:
            return _query_postgres_to_csv(
                sql,
                output_path,
                lookback_days,
                row_limit,
                anchor_date,
                debug_ad_ids,
                account_id,
            )
        attempts = [
            (lookback_days, row_limit, anchor_date, debug_ad_ids, account_id),
            (min(lookback_days, 7), min(row_limit, 500), anchor_date, debug_ad_ids, account_id),
            (min(lookback_days, 3), min(row_limit, 200), anchor_date, debug_ad_ids, account_id),
        ]
        return _query_postgres_to_csv_with_retries(sql, output_path, attempts)

    daily_input = Path(os.getenv("ADUNBOX_DAILY_INPUT", ROOT / "data" / "adunbox_daily_breakdown_kpis.csv"))
    _require_path(daily_input, "24h daily input")
    return str(daily_input)


@asset(group_name="forecast_24h", partitions_def=account_partitions)
def adunbox_24h_raw_forecast(context: AssetExecutionContext, postgres_24h_daily_extract: str) -> str:
    """Score 24h daily forecasts using the final production model."""
    account_id = context.partition_key if context.has_partition_key else ""
    output_dir = OUTPUTS / account_id if account_id else OUTPUTS
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "adunbox_24h_latest_forecasts.csv"
    feature_cache = output_dir / "adunbox_24h_latest_feature_cache.joblib"

    env = os.environ.copy()
    env["ADUNBOX_DAILY_INPUT"] = postgres_24h_daily_extract
    env["ADUNBOX_24H_OUTPUT_PATH"] = str(output_path)
    env["ADUNBOX_24H_FEATURE_CACHE"] = str(feature_cache)
    if account_id:
        env["ADUNBOX_SCORE_ACCOUNT_IDS"] = account_id
    _run_python(SCRIPTS / "score_adunbox_daily_24h_model.py", "--daily-input", postgres_24h_daily_extract, env=env)
    _require_path(output_path, "24h latest forecast output")
    context.log.info(f"24h raw forecast scoring completed{' for account ' + account_id if account_id else ''}.")
    return str(output_path)


@asset(group_name="forecast_24h", partitions_def=account_partitions)
def adunbox_24h_served_forecast(context: AssetExecutionContext, adunbox_24h_raw_forecast: str) -> str:
    """Return latest 24h forecast with confidence/range columns.

    The production scorer already applies the confidence/range layer for latest
    forecasts. The separate serving-layer script is retained for historical
    backtest files where actual columns are present.
    """
    _require_path(Path(adunbox_24h_raw_forecast), "24h latest forecast")
    context.log.info("24h latest forecast already includes production serving columns.")
    return adunbox_24h_raw_forecast


@asset(group_name="forecast_24h", partitions_def=account_partitions)
def adunbox_24h_quality_monitor(context: AssetExecutionContext, adunbox_24h_served_forecast: str) -> str:
    """Write a lightweight production monitor placeholder for latest forecasts.

    Full actual-vs-predicted quality monitoring should run once D1 actuals have
    landed. For live forecasts, this asset records that the forecast file exists.
    """
    account_id = context.partition_key if context.has_partition_key else ""
    output_dir = OUTPUTS / account_id if account_id else OUTPUTS
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor_path = output_dir / "adunbox_daily_24h_quality_monitor.csv"

    forecast_path = Path(adunbox_24h_served_forecast)
    _require_path(forecast_path, "24h latest forecast")
    monitor_path.write_text(
        json.dumps({"latest_forecast": str(forecast_path), "status": "forecast_written_waiting_for_d1_actuals"}, indent=2),
        encoding="utf-8",
    )
    context.log.info("24h latest forecast monitor placeholder completed.")
    return str(monitor_path)


@asset(group_name="forecast_6h", partitions_def=account_partitions)
def adunbox_6h_production_ready_manifest(context: AssetExecutionContext, postgres_6h_hourly_extract: str) -> str:
    """Score latest 6h forecasts using the final target-routed 6h ensemble.

    The final 6h production model is target-wise:
      spend/impressions/clicks -> anchor_v2
      conversions/revenue -> business_v3
    """
    account_id = context.partition_key if context.has_partition_key else ""
    output_dir = OUTPUTS / account_id if account_id else OUTPUTS
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "adunbox_6h_latest_forecasts.csv"
    feature_cache = output_dir / "adunbox_6h_latest_feature_cache.joblib"

    _require_path(Path(postgres_6h_hourly_extract), "6h hourly source")
    _require_path(FINAL_6H_ANCHOR_MODEL_DIR / "metadata.joblib", "6h anchor metadata")
    _require_path(FINAL_6H_BUSINESS_MODEL_DIR / "metadata.joblib", "6h business metadata")
    env = os.environ.copy()
    env["ADUNBOX_HOURLY_INPUT"] = postgres_6h_hourly_extract
    env["ADUNBOX_6H_OUTPUT_PATH"] = str(output_path)
    env["ADUNBOX_6H_FEATURE_CACHE"] = str(feature_cache)
    if account_id:
        env["ADUNBOX_SCORE_ACCOUNT_IDS"] = account_id
    env.setdefault("ADUNBOX_6H_SCORE_CHUNKSIZE", "25000")
    _run_python(SCRIPTS / "score_adunbox_entity_history_lgbm_6h_model.py", "--hourly-input", postgres_6h_hourly_extract, env=env)
    _require_path(output_path, "6h latest forecast output")
    context.log.info(f"6h production scoring completed{' for account ' + account_id if account_id else ''}.")
    return str(output_path)


def _write_single_horizon_sink_status(status_path: Path, status: dict) -> str:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return str(status_path)


def _cleanup_local_working_files(paths: list[Path]) -> list[str]:
    removed: list[str] = []
    if not _clean_local_outputs_after_db_write():
        return removed
    for path in paths:
        try:
            if path.exists() and path.is_file():
                path.unlink()
                removed.append(str(path))
        except OSError:
            continue
    return removed


@asset(group_name="forecast_persistence", partitions_def=account_partitions)
def adunbox_6h_forecast_postgres_sink(context: AssetExecutionContext, adunbox_6h_production_ready_manifest: str) -> str:
    """Optionally persist only the latest 6h forecast output to PostgreSQL."""
    account_id = context.partition_key if context.has_partition_key else ""
    output_dir = OUTPUTS / account_id if account_id else OUTPUTS
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "adunbox_6h_forecast_persistence_status.json"
    feature_cache = output_dir / "adunbox_6h_latest_feature_cache.joblib"
    if not _write_forecasts_to_db_enabled() and not _write_features_to_db_enabled():
        return _write_single_horizon_sink_status(
            status_path,
            {
                "horizon": "6h",
                "enabled": False,
                "status": "skipped",
                "reason": "Set ADUNBOX_WRITE_FORECASTS_TO_DB=true and/or ADUNBOX_WRITE_FEATURES_TO_DB=true.",
            },
        )

    forecast_table = os.getenv("ADUNBOX_FORECAST_TABLE", "adunbox_model_forecasts")
    feature_table = os.getenv("ADUNBOX_FEATURE_TABLE", "adunbox_model_feature_cache")
    db_extract = output_dir / "adunbox_6h_hourly_db_extract.csv" if account_id else DB_6H_EXTRACT
    forecast_rows, feature_rows = asyncio.run(
        _persist_single_horizon_outputs_to_postgres(
            Path(adunbox_6h_production_ready_manifest),
            "6h",
            feature_cache,
            forecast_table,
            feature_table,
            _write_forecasts_to_db_enabled(),
            _write_features_to_db_enabled(),
        )
    )
    status = {
        "horizon": "6h",
        "account_id": account_id or "all",
        "forecast_write_enabled": _write_forecasts_to_db_enabled(),
        "feature_write_enabled": _write_features_to_db_enabled(),
        "forecast_table": forecast_table,
        "feature_table": feature_table,
        "forecast_rows": forecast_rows,
        "feature_rows": feature_rows,
        "status": "written",
    }
    status["local_cleanup_removed"] = _cleanup_local_working_files(
        [Path(adunbox_6h_production_ready_manifest), db_extract, feature_cache]
    )
    context.log.info(f"Persisted 6h outputs to PostgreSQL: {status}")
    return _write_single_horizon_sink_status(status_path, status)


@asset(group_name="forecast_persistence", partitions_def=account_partitions)
def adunbox_24h_forecast_postgres_sink(context: AssetExecutionContext, adunbox_24h_served_forecast: str) -> str:
    """Optionally persist only the latest 24h forecast output to PostgreSQL."""
    account_id = context.partition_key if context.has_partition_key else ""
    output_dir = OUTPUTS / account_id if account_id else OUTPUTS
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "adunbox_24h_forecast_persistence_status.json"
    feature_cache = output_dir / "adunbox_24h_latest_feature_cache.joblib"
    if not _write_forecasts_to_db_enabled() and not _write_features_to_db_enabled():
        return _write_single_horizon_sink_status(
            status_path,
            {
                "horizon": "24h",
                "enabled": False,
                "status": "skipped",
                "reason": "Set ADUNBOX_WRITE_FORECASTS_TO_DB=true and/or ADUNBOX_WRITE_FEATURES_TO_DB=true.",
            },
        )

    forecast_table = os.getenv("ADUNBOX_FORECAST_TABLE", "adunbox_model_forecasts")
    feature_table = os.getenv("ADUNBOX_FEATURE_TABLE", "adunbox_model_feature_cache")
    db_extract = output_dir / "adunbox_24h_daily_db_extract.csv" if account_id else DB_24H_EXTRACT
    forecast_rows, feature_rows = asyncio.run(
        _persist_single_horizon_outputs_to_postgres(
            Path(adunbox_24h_served_forecast),
            "24h",
            feature_cache,
            forecast_table,
            feature_table,
            _write_forecasts_to_db_enabled(),
            _write_features_to_db_enabled(),
        )
    )
    status = {
        "horizon": "24h",
        "account_id": account_id or "all",
        "forecast_write_enabled": _write_forecasts_to_db_enabled(),
        "feature_write_enabled": _write_features_to_db_enabled(),
        "forecast_table": forecast_table,
        "feature_table": feature_table,
        "forecast_rows": forecast_rows,
        "feature_rows": feature_rows,
        "status": "written",
    }
    status["local_cleanup_removed"] = _cleanup_local_working_files(
        [Path(adunbox_24h_served_forecast), db_extract, feature_cache]
    )
    context.log.info(f"Persisted 24h outputs to PostgreSQL: {status}")
    return _write_single_horizon_sink_status(status_path, status)


@asset(group_name="forecast_persistence", partitions_def=account_partitions)
def adunbox_forecast_postgres_sink(
    context: AssetExecutionContext,
    adunbox_24h_served_forecast: str,
    adunbox_6h_production_ready_manifest: str,
) -> str:
    """Optionally persist latest forecasts back into PostgreSQL.

    Enable with:
      ADUNBOX_WRITE_FORECASTS_TO_DB=true

    The table stores core IDs as columns and the complete forecast row in JSONB,
    so schema changes in model output do not break production inserts.
    """
    account_id = context.partition_key if context.has_partition_key else ""
    output_dir = OUTPUTS / account_id if account_id else OUTPUTS
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "adunbox_forecast_persistence_status.json"
    feature_cache_6h = output_dir / "adunbox_6h_latest_feature_cache.joblib"
    feature_cache_24h = output_dir / "adunbox_24h_latest_feature_cache.joblib"
    db_extract_6h = output_dir / "adunbox_6h_hourly_db_extract.csv" if account_id else DB_6H_EXTRACT
    db_extract_24h = output_dir / "adunbox_24h_daily_db_extract.csv" if account_id else DB_24H_EXTRACT

    if not _write_forecasts_to_db_enabled() and not _write_features_to_db_enabled():
        status = {
            "enabled": False,
            "status": "skipped",
            "reason": "Set ADUNBOX_WRITE_FORECASTS_TO_DB=true and/or ADUNBOX_WRITE_FEATURES_TO_DB=true to persist outputs to PostgreSQL.",
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        context.log.info("Forecast/feature DB persistence skipped.")
        return str(status_path)

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
        "account_id": account_id or "all",
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
    status["local_cleanup_removed"] = _cleanup_local_working_files(
        [
            Path(adunbox_24h_served_forecast),
            Path(adunbox_6h_production_ready_manifest),
            db_extract_24h,
            db_extract_6h,
            feature_cache_24h,
            feature_cache_6h,
        ]
    )
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    context.log.info(f"Persisted outputs to PostgreSQL: {status}")
    return str(status_path)


# ── All-accounts jobs (non-partitioned, legacy mode) ──────────────────────────

production_6h_job = define_asset_job(
    "adunbox_6h_forecast_job",
    selection=AssetSelection.keys(
        "final_model_registry",
        "postgres_6h_hourly_extract",
        "adunbox_6h_production_ready_manifest",
        "adunbox_6h_forecast_postgres_sink",
    ),
)

production_24h_job = define_asset_job(
    "adunbox_24h_forecast_job",
    selection=AssetSelection.keys(
        "final_model_registry",
        "postgres_24h_daily_extract",
        "adunbox_24h_raw_forecast",
        "adunbox_24h_served_forecast",
        "adunbox_24h_quality_monitor",
        "adunbox_24h_forecast_postgres_sink",
    ),
)

production_job = define_asset_job(
    "adunbox_production_forecast_job",
    selection=AssetSelection.keys(
        "final_model_registry",
        "postgres_6h_hourly_extract",
        "adunbox_6h_production_ready_manifest",
        "postgres_24h_daily_extract",
        "adunbox_24h_raw_forecast",
        "adunbox_24h_served_forecast",
        "adunbox_24h_quality_monitor",
        "adunbox_forecast_postgres_sink",
    ),
)

# ── Per-account partitioned jobs ───────────────────────────────────────────────

partitioned_6h_job = define_asset_job(
    "adunbox_6h_partitioned_forecast_job",
    selection=AssetSelection.keys(
        "final_model_registry",
        "postgres_6h_hourly_extract",
        "adunbox_6h_production_ready_manifest",
        "adunbox_6h_forecast_postgres_sink",
    ),
    partitions_def=account_partitions,
)

partitioned_24h_job = define_asset_job(
    "adunbox_24h_partitioned_forecast_job",
    selection=AssetSelection.keys(
        "final_model_registry",
        "postgres_24h_daily_extract",
        "adunbox_24h_raw_forecast",
        "adunbox_24h_served_forecast",
        "adunbox_24h_quality_monitor",
        "adunbox_24h_forecast_postgres_sink",
    ),
    partitions_def=account_partitions,
)

partitioned_job = define_asset_job(
    "adunbox_partitioned_forecast_job",
    selection=AssetSelection.keys(
        "final_model_registry",
        "postgres_6h_hourly_extract",
        "adunbox_6h_production_ready_manifest",
        "postgres_24h_daily_extract",
        "adunbox_24h_raw_forecast",
        "adunbox_24h_served_forecast",
        "adunbox_24h_quality_monitor",
        "adunbox_forecast_postgres_sink",
    ),
    partitions_def=account_partitions,
)

# ── Schedules ──────────────────────────────────────────────────────────────────

daily_6h_schedule = ScheduleDefinition(
    job=production_6h_job,
    cron_schedule="15 */6 * * *",
    execution_timezone="Asia/Kolkata",
)

daily_24h_schedule = ScheduleDefinition(
    job=production_24h_job,
    cron_schedule="15 0 * * *",
    execution_timezone="Asia/Kolkata",
)

daily_schedule = ScheduleDefinition(
    job=production_job,
    cron_schedule="15 0 * * *",
    execution_timezone="Asia/Kolkata",
)

defs = Definitions(
    assets=[
        discover_account_partitions,
        final_model_registry,
        postgres_6h_hourly_extract,
        postgres_24h_daily_extract,
        adunbox_24h_raw_forecast,
        adunbox_24h_served_forecast,
        adunbox_24h_quality_monitor,
        adunbox_6h_production_ready_manifest,
        adunbox_6h_forecast_postgres_sink,
        adunbox_24h_forecast_postgres_sink,
        adunbox_forecast_postgres_sink,
    ],
    jobs=[
        production_6h_job,
        production_24h_job,
        production_job,
        partitioned_6h_job,
        partitioned_24h_job,
        partitioned_job,
    ],
    schedules=[daily_6h_schedule, daily_24h_schedule, daily_schedule],
)
