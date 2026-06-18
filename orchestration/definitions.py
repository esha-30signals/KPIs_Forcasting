"""
Dagster definitions for the KPI forecasting pipeline.

4 jobs:
  adunbox_6h_forecast_job   — 6h forecasts for Adunbox accounts
  adunbox_24h_forecast_job  — 24h forecasts for Adunbox accounts
  vibelets_6h_forecast_job  — 6h forecasts for Vibelets accounts
  vibelets_24h_forecast_job — 24h forecasts for Vibelets accounts

Platform config (table prefix, column names, forecast table) lives in
assets/_config.py as Python constants — no env-var switching needed.
"""
import asyncio
import sys
from pathlib import Path

from dagster import (
    AssetSelection,
    Definitions,
    RunRequest,
    ScheduleDefinition,
    ScheduleEvaluationContext,
    asset,
    define_asset_job,
    schedule,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import postgresql

from orchestration.assets._config import ADUNBOX_CONFIG, VIBELETS_CONFIG
from orchestration.assets._db_io import _query_postgres_to_frame
from orchestration.assets.factory import make_24h_assets, make_6h_assets
from orchestration.partitions import adunbox_account_partition, vibelets_account_partition

# ── Generate assets via factory ────────────────────────────────────────────────

adunbox_6h_assets = make_6h_assets("adunbox", ADUNBOX_CONFIG, adunbox_account_partition)
adunbox_24h_assets = make_24h_assets("adunbox", ADUNBOX_CONFIG, adunbox_account_partition)
vibelets_6h_assets = make_6h_assets("vibelets", VIBELETS_CONFIG, vibelets_account_partition)
vibelets_24h_assets = make_24h_assets("vibelets", VIBELETS_CONFIG, vibelets_account_partition)

# ── 4 Jobs ─────────────────────────────────────────────────────────────────────

adunbox_6h_job = define_asset_job(
    "adunbox_6h_forecast_job",
    selection=AssetSelection.assets(*adunbox_6h_assets),
    partitions_def=adunbox_account_partition,
)

adunbox_24h_job = define_asset_job(
    "adunbox_24h_forecast_job",
    selection=AssetSelection.assets(*adunbox_24h_assets),
    partitions_def=adunbox_account_partition,
)

vibelets_6h_job = define_asset_job(
    "vibelets_6h_forecast_job",
    selection=AssetSelection.assets(*vibelets_6h_assets),
    partitions_def=vibelets_account_partition,
)

vibelets_24h_job = define_asset_job(
    "vibelets_24h_forecast_job",
    selection=AssetSelection.assets(*vibelets_24h_assets),
    partitions_def=vibelets_account_partition,
)

# ── Schedules with inline partition discovery ──────────────────────────────────


def _discover_and_run(
    context: ScheduleEvaluationContext,
    accounts_table: str,
    partition_def: object,
) -> list[RunRequest]:
    sql = f"SELECT DISTINCT id::text AS account_id FROM {accounts_table} WHERE UPPER(COALESCE(status, '')) = 'ACTIVE' ORDER BY 1"
    df = asyncio.run(_query_postgres_to_frame(sql))
    asyncio.run(postgresql.disconnect())
    if df.empty:
        context.log.warning(f"No active accounts found in {accounts_table}.")
        return []
    account_ids = df["account_id"].dropna().astype(str).tolist()
    context.instance.add_dynamic_partitions(partition_def.name, account_ids)
    context.log.info(f"Registered {len(account_ids)} partitions from {accounts_table}.")
    return [RunRequest(partition_key=aid) for aid in account_ids]


@schedule(
    name="adunbox_6h_schedule",
    cron_schedule="15 */6 * * *",
    job=adunbox_6h_job,
    execution_timezone="Asia/Kolkata",
)
def adunbox_6h_schedule(context: ScheduleEvaluationContext):
    yield from _discover_and_run(
        context,
        accounts_table=ADUNBOX_CONFIG.get_table("traffic_source_accounts"),
        partition_def=adunbox_account_partition,
    )


@schedule(
    name="adunbox_24h_schedule",
    cron_schedule="15 0 * * *",
    job=adunbox_24h_job,
    execution_timezone="Asia/Kolkata",
)
def adunbox_24h_schedule(context: ScheduleEvaluationContext):
    yield from _discover_and_run(
        context,
        accounts_table=ADUNBOX_CONFIG.get_table("traffic_source_accounts"),
        partition_def=adunbox_account_partition,
    )


@schedule(
    name="vibelets_6h_schedule",
    cron_schedule="15 */6 * * *",
    job=vibelets_6h_job,
    execution_timezone="Asia/Kolkata",
)
def vibelets_6h_schedule(context: ScheduleEvaluationContext):
    yield from _discover_and_run(
        context,
        accounts_table=VIBELETS_CONFIG.get_table("traffic_source_accounts"),
        partition_def=vibelets_account_partition,
    )


@schedule(
    name="vibelets_24h_schedule",
    cron_schedule="15 0 * * *",
    job=vibelets_24h_job,
    execution_timezone="Asia/Kolkata",
)
def vibelets_24h_schedule(context: ScheduleEvaluationContext):
    yield from _discover_and_run(
        context,
        accounts_table=VIBELETS_CONFIG.get_table("traffic_source_accounts"),
        partition_def=vibelets_account_partition,
    )


# ── Definitions ────────────────────────────────────────────────────────────────

defs = Definitions(
    assets=[
        *adunbox_6h_assets,
        *adunbox_24h_assets,
        *vibelets_6h_assets,
        *vibelets_24h_assets,
    ],
    jobs=[
        adunbox_6h_job,
        adunbox_24h_job,
        vibelets_6h_job,
        vibelets_24h_job,
    ],
    schedules=[
        adunbox_6h_schedule,
        adunbox_24h_schedule,
        vibelets_6h_schedule,
        vibelets_24h_schedule,
    ],
)
