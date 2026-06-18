import asyncio
import json
import os
from pathlib import Path

import joblib
import pandas as pd

from ._helpers import (
    _account_local_datetime,
    _db_datetime,
    _first_existing,
    _forecast_results,
    _forecast_status,
    _forecast_confidence_score,
    _forecast_window,
    _json_safe,
    _local_date,
    _raw_forecast_results,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"

if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from db import postgresql


async def _query_postgres_to_frame(sql: str, *args) -> pd.DataFrame:
    rows = await postgresql.query(sql, *args) if args else await postgresql.query(sql)
    return pd.DataFrame(rows)


def _query_postgres_to_csv(sql: str, output_path: Path, *args) -> str:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    df = asyncio.run(_query_postgres_to_frame(sql, *args))
    if df.empty:
        debug_path = output_path.with_name(f"{output_path.stem}__zero_rows_debug.json")
        debug_path.write_text(
            json.dumps(
                {
                    "output_path": str(output_path),
                    "reason": "Database query returned zero rows",
                    "args": [str(a) for a in args],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise ValueError(
            f"Database query returned zero rows for output: {output_path}. Debug written to: {debug_path}"
        )
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
        recently_active BOOLEAN,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """
    await postgresql.execute(create_sql)

    optional_columns = [
        "timezone TEXT", "forecast_anchor_local TIMESTAMP", "forecast_window_start_local TIMESTAMP",
        "forecast_window_end_local TIMESTAMP", "forecast_local_date DATE",
        "forecast_confidence_score DOUBLE PRECISION",
        "result_spend DOUBLE PRECISION", "result_impressions DOUBLE PRECISION",
        "result_clicks DOUBLE PRECISION", "result_conversions DOUBLE PRECISION",
        "result_revenue DOUBLE PRECISION", "result_roas DOUBLE PRECISION",
        "result_profit DOUBLE PRECISION", "result_ctr DOUBLE PRECISION",
        "result_cvr DOUBLE PRECISION", "result_cpc DOUBLE PRECISION", "result_cpm DOUBLE PRECISION",
        "raw_pred_spend DOUBLE PRECISION", "raw_pred_impressions DOUBLE PRECISION",
        "raw_pred_clicks DOUBLE PRECISION", "raw_pred_conversions DOUBLE PRECISION",
        "raw_pred_revenue DOUBLE PRECISION", "raw_pred_roas DOUBLE PRECISION",
        "raw_pred_profit DOUBLE PRECISION", "raw_pred_ctr DOUBLE PRECISION",
        "raw_pred_cvr DOUBLE PRECISION", "raw_pred_cpc DOUBLE PRECISION",
        "raw_pred_cpm DOUBLE PRECISION", "recently_active BOOLEAN",
    ]
    for col_def in optional_columns:
        col_name = col_def.split()[0]
        await postgresql.execute(
            f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {col_name} {' '.join(col_def.split()[1:])}"
        )

    await postgresql.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = '{table_name}' AND column_name = 'payload'
            ) THEN
                EXECUTE 'ALTER TABLE {table_name} ALTER COLUMN payload DROP NOT NULL';
            END IF;
        END $$;
        """
    )
    await postgresql.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ad_created ON {table_name}(ad_id, created_at DESC)"
    )
    await postgresql.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_account_created ON {table_name}(account_id, created_at DESC)"
    )

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
        rows.append((
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
            results.get("spend"), results.get("impressions"), results.get("clicks"),
            results.get("conversions"), results.get("revenue"), results.get("roas"),
            results.get("profit"), results.get("ctr"), results.get("cvr"),
            results.get("cpc"), results.get("cpm"),
            raw_results.get("spend"), raw_results.get("impressions"), raw_results.get("clicks"),
            raw_results.get("conversions"), raw_results.get("revenue"), raw_results.get("roas"),
            raw_results.get("profit"), raw_results.get("ctr"), raw_results.get("cvr"),
            raw_results.get("cpc"), raw_results.get("cpm"),
            bool(_first_existing(row, ["recently_active"]))
            if _first_existing(row, ["recently_active"]) is not None else None,
        ))

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
        raw_pred_ctr, raw_pred_cvr, raw_pred_cpc, raw_pred_cpm,
        recently_active
    )
    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31,$32,$33,$34,$35,$36,$37,$38,$39,$40,$41)
    """
    batch_size = int(os.getenv("ADUNBOX_DB_WRITE_BATCH_SIZE", "1000"))
    for start in range(0, len(rows), batch_size):
        await postgresql.executemany(insert_sql, rows[start: start + batch_size])
    return len(rows)


async def _persist_single_horizon_to_postgres(
    csv_path: Path,
    horizon: str,
    forecast_table: str,
) -> int:
    return await _persist_forecast_csv_to_postgres(csv_path, horizon, forecast_table)
