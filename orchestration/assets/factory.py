"""
Factory functions that generate Dagster assets for a given platform + horizon.

Each call to make_6h_assets() or make_24h_assets() returns a list of 3 assets
namespaced under [platform]:

  make_6h_assets("adunbox", ADUNBOX_CONFIG, adunbox_account_partition)
    → adunbox/extract_6h, adunbox/score_6h, adunbox/sink_6h

  make_6h_assets("vibelets", VIBELETS_CONFIG, vibelets_account_partition)
    → vibelets/extract_6h, vibelets/score_6h, vibelets/sink_6h

Platform config (table prefix, column names, forecast table) is baked into the
closure at factory call time — no env-var switching needed at runtime.
"""
import asyncio
import os
from pathlib import Path

from dagster import AssetExecutionContext, DynamicPartitionsDefinition, asset

from ._config import ForecastConfig
from ._db_io import _persist_single_horizon_to_postgres, _query_postgres_to_csv, _query_postgres_to_csv_with_retries
from ._helpers import (
    _cleanup_local_working_files,
    _require_path,
    _run_python,
    _write_single_horizon_sink_status,
    render_24h_sql,
    render_6h_sql,
)
from ._sql import DAILY_24H_SQL, REPORTS_6H_SQL

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"

FINAL_24H_MODEL_DIR = MODELS / "adunbox_daily_24h_histgb_full_db_production"
FINAL_6H_ANCHOR_MODEL_DIR = MODELS / "adunbox_entity_history_lgbm_6h_anchor_v2"
FINAL_6H_BUSINESS_MODEL_DIR = MODELS / "adunbox_entity_history_lgbm_6h_business_v3"

_6H_LOOKBACK_DAYS = 8
_6H_ROW_LIMIT = 500
_24H_LOOKBACK_DAYS = 21
_24H_ROW_LIMIT = 500


def make_6h_assets(
    platform: str,
    config: ForecastConfig,
    partitions_def: DynamicPartitionsDefinition,
) -> list:
    """Return [extract_6h, score_6h, sink_6h] assets namespaced under `platform`."""
    prefix = [platform]
    group = f"{platform}_6h_forecast"
    sql = render_6h_sql(REPORTS_6H_SQL, config)

    @asset(key_prefix=prefix, group_name=group, partitions_def=partitions_def)
    def extract_6h(context: AssetExecutionContext) -> str:
        account_id = context.partition_key if context.has_partition_key else ""
        output_dir = OUTPUTS / account_id if account_id else OUTPUTS
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{platform}_6h_hourly_db_extract.csv"

        lookback = int(os.getenv("FORECAST_6H_DB_LOOKBACK_DAYS", str(_6H_LOOKBACK_DAYS)))
        row_limit = int(os.getenv("FORECAST_6H_DB_ROW_LIMIT", str(_6H_ROW_LIMIT)) or str(_6H_ROW_LIMIT))
        return _query_postgres_to_csv(sql, output_path, lookback, row_limit, None, "", account_id)

    @asset(key_prefix=prefix, group_name=group, partitions_def=partitions_def)
    def score_6h(context: AssetExecutionContext, extract_6h: str) -> str:
        account_id = context.partition_key if context.has_partition_key else ""
        output_dir = OUTPUTS / account_id if account_id else OUTPUTS
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{platform}_6h_latest_forecasts.csv"
        feature_cache = output_dir / f"{platform}_6h_latest_feature_cache.joblib"

        _require_path(Path(extract_6h), "6h hourly source")
        _require_path(FINAL_6H_ANCHOR_MODEL_DIR / "metadata.joblib", "6h anchor model")
        _require_path(FINAL_6H_BUSINESS_MODEL_DIR / "metadata.joblib", "6h business model")

        env = os.environ.copy()
        env["ADUNBOX_HOURLY_INPUT"] = extract_6h
        env["ADUNBOX_6H_OUTPUT_PATH"] = str(output_path)
        env["ADUNBOX_6H_FEATURE_CACHE"] = str(feature_cache)
        if account_id:
            env["ADUNBOX_SCORE_ACCOUNT_IDS"] = account_id
        env.setdefault("ADUNBOX_6H_SCORE_CHUNKSIZE", "25000")
        _run_python(SCRIPTS / "score_adunbox_entity_history_lgbm_6h_model.py", "--hourly-input", extract_6h, env=env)
        _require_path(output_path, "6h forecast output")
        context.log.info(f"[{platform}] 6h scoring done{' — account ' + account_id if account_id else ''}.")
        return str(output_path)

    @asset(key_prefix=prefix, group_name=group, partitions_def=partitions_def)
    def sink_6h(context: AssetExecutionContext, score_6h: str) -> str:
        account_id = context.partition_key if context.has_partition_key else ""
        output_dir = OUTPUTS / account_id if account_id else OUTPUTS
        output_dir.mkdir(parents=True, exist_ok=True)
        status_path = output_dir / f"{platform}_6h_persistence_status.json"
        feature_cache = output_dir / f"{platform}_6h_latest_feature_cache.joblib"
        db_extract = output_dir / f"{platform}_6h_hourly_db_extract.csv"

        forecast_rows = asyncio.run(
            _persist_single_horizon_to_postgres(Path(score_6h), "6h", config.forecast_table)
        )
        status = {
            "horizon": "6h",
            "platform": platform,
            "account_id": account_id or "all",
            "forecast_table": config.forecast_table,
            "forecast_rows": forecast_rows,
            "status": "written",
        }
        status["local_cleanup_removed"] = _cleanup_local_working_files(
            [Path(score_6h), db_extract, feature_cache]
        )
        context.log.info(f"[{platform}] Persisted 6h → {config.forecast_table}: {status}")
        return _write_single_horizon_sink_status(status_path, status)

    return [extract_6h, score_6h, sink_6h]


def make_24h_assets(
    platform: str,
    config: ForecastConfig,
    partitions_def: DynamicPartitionsDefinition,
) -> list:
    """Return [extract_24h, score_24h, sink_24h] assets namespaced under `platform`."""
    prefix = [platform]
    group = f"{platform}_24h_forecast"
    sql = render_24h_sql(DAILY_24H_SQL, config)

    @asset(key_prefix=prefix, group_name=group, partitions_def=partitions_def)
    def extract_24h(context: AssetExecutionContext) -> str:
        account_id = context.partition_key if context.has_partition_key else ""
        output_dir = OUTPUTS / account_id if account_id else OUTPUTS
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{platform}_24h_daily_db_extract.csv"

        lookback = int(os.getenv("FORECAST_24H_DB_LOOKBACK_DAYS", str(_24H_LOOKBACK_DAYS)))
        row_limit = int(os.getenv("FORECAST_24H_DB_ROW_LIMIT", str(_24H_ROW_LIMIT)) or str(_24H_ROW_LIMIT))

        attempts = [
            (lookback, row_limit, None, "", account_id),
            (min(lookback, 7), min(row_limit, 500), None, "", account_id),
            (min(lookback, 3), min(row_limit, 200), None, "", account_id),
        ]
        return _query_postgres_to_csv_with_retries(sql, output_path, attempts)

    @asset(key_prefix=prefix, group_name=group, partitions_def=partitions_def)
    def score_24h(context: AssetExecutionContext, extract_24h: str) -> str:
        account_id = context.partition_key if context.has_partition_key else ""
        output_dir = OUTPUTS / account_id if account_id else OUTPUTS
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{platform}_24h_latest_forecasts.csv"
        feature_cache = output_dir / f"{platform}_24h_latest_feature_cache.joblib"

        _require_path(Path(extract_24h), "24h daily source")
        _require_path(FINAL_24H_MODEL_DIR / "metadata.joblib", "24h model")

        env = os.environ.copy()
        env["ADUNBOX_DAILY_INPUT"] = extract_24h
        env["ADUNBOX_24H_OUTPUT_PATH"] = str(output_path)
        env["ADUNBOX_24H_FEATURE_CACHE"] = str(feature_cache)
        if account_id:
            env["ADUNBOX_SCORE_ACCOUNT_IDS"] = account_id
        _run_python(SCRIPTS / "score_adunbox_daily_24h_model.py", "--daily-input", extract_24h, env=env)
        _require_path(output_path, "24h forecast output")
        context.log.info(f"[{platform}] 24h scoring done{' — account ' + account_id if account_id else ''}.")
        return str(output_path)

    @asset(key_prefix=prefix, group_name=group, partitions_def=partitions_def)
    def sink_24h(context: AssetExecutionContext, score_24h: str) -> str:
        account_id = context.partition_key if context.has_partition_key else ""
        output_dir = OUTPUTS / account_id if account_id else OUTPUTS
        output_dir.mkdir(parents=True, exist_ok=True)
        status_path = output_dir / f"{platform}_24h_persistence_status.json"
        feature_cache = output_dir / f"{platform}_24h_latest_feature_cache.joblib"
        db_extract = output_dir / f"{platform}_24h_daily_db_extract.csv"

        forecast_rows = asyncio.run(
            _persist_single_horizon_to_postgres(Path(score_24h), "24h", config.forecast_table)
        )
        status = {
            "horizon": "24h",
            "platform": platform,
            "account_id": account_id or "all",
            "forecast_table": config.forecast_table,
            "forecast_rows": forecast_rows,
            "status": "written",
        }
        status["local_cleanup_removed"] = _cleanup_local_working_files(
            [Path(score_24h), db_extract, feature_cache]
        )
        context.log.info(f"[{platform}] Persisted 24h → {config.forecast_table}: {status}")
        return _write_single_horizon_sink_status(status_path, status)

    return [extract_24h, score_24h, sink_24h]
