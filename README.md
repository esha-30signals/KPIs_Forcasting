# Adunbox Forecasting Production Release

This folder contains the production-ready Adunbox forecasting pipeline for:

- `6h` forecasts from hourly ad data.
- `24h` forecasts from daily ad data.
- Dagster orchestration for local CSV testing or direct PostgreSQL testing.

## Production Models

### 6h Model

Target-routed LightGBM:

```text
spend / impressions / clicks:
  models/adunbox_entity_history_lgbm_6h_anchor_v2/

conversions / revenue:
  models/adunbox_entity_history_lgbm_6h_business_v3/
```

### 24h Model

Daily LightGBM/HistGB production model:

```text
models/adunbox_daily_24h_histgb_full_db_production/
```

## Pipeline Flow

```text
PostgreSQL or local CSV
  -> source extract
  -> feature engineering / feature cache
  -> model scoring
  -> confidence + fallback layer
  -> CSV forecast outputs
  -> optional PostgreSQL forecast table
```

Dagster exposes separate model pipelines:

```text
adunbox_6h_forecast_job:
  hourly extract -> active hierarchy filter -> 6h model scoring -> optional 6h DB sink

adunbox_24h_forecast_job:
  daily extract -> active hierarchy filter -> 24h model scoring -> monitor -> optional 24h DB sink

adunbox_production_forecast_job:
  full combined run, including optional forecast DB sink
```

By default, only ads whose account, campaign, adset, and ad are all `ACTIVE` are sent to forecasting in database mode.

If production needs to include ads that are paused now but scheduled to become active inside the forecast window, create the optional `public.adunbox_forecast_eligible_ads` view from `sql/adunbox_forecast_eligible_ads_view_template.sql` and set:

```bash
export ADUNBOX_USE_FORECAST_ELIGIBILITY_VIEW=true
```

With this enabled, an ad is forecasted when it is currently active or scheduled to become active during the relevant `6h` / `24h` forecast window.

## Key Scripts

```text
scripts/score_adunbox_daily_24h_model.py
scripts/score_adunbox_entity_history_lgbm_6h_model.py
scripts/adunbox_hierarchical_fallbacks.py
scripts/train_adunbox_daily_24h_full_db_production.py
scripts/train_adunbox_entity_history_lgbm_6h_anchor_v2.py
scripts/train_adunbox_entity_history_lgbm_6h_business_v3.py
```

Dagster entrypoint:

```text
orchestration/production_dagster_assets.py
```

## Install

From `github_release/`:

```bash
pip install -r requirements.txt
```

## Local CSV Smoke Test

Use this when you do not want to connect to the database.

```bash
cd /github_release

export ADUNBOX_USE_DATABASE=false
export ADUNBOX_DAILY_INPUT="adunbox_daily_breakdown_kpis.csv"
export ADUNBOX_HOURLY_INPUT="adunbox_joined_traffic_reports_with_timezone.csv"

dagster dev -f orchestration/production_dagster_assets.py -h 127.0.0.1 -p 3000
```

Open:

```text
http://127.0.0.1:3000
```

Click:

```text
Materialize all
```

For lower RAM testing, use the Jobs page and launch only:

```text
adunbox_6h_forecast_job
```

or:

```text
adunbox_24h_forecast_job
```

## Direct PostgreSQL Smoke Test

Use small row limits first. This is designed for laptops with around `16 GB RAM`.

Before production DB runs, ask the DB/backend owner to apply:

```text
sql/adunbox_production_indexes.sql
```

Then verify query plans with:

```text
sql/adunbox_verify_query_plans.sql
```

```bash
cd /g/ml_model_historical_data/github_release

export ADUNBOX_USE_DATABASE=true
export ADUNBOX_WRITE_FORECASTS_TO_DB=false

export POSTGRES_HOST="your_host"
export POSTGRES_PORT="5432"
export POSTGRES_DB="your_db"
export POSTGRES_USER="your_user"
export POSTGRES_PASSWORD="your_password"

export POSTGRES_POOL_MIN_SIZE=1
export POSTGRES_POOL_MAX_SIZE=2
export POSTGRES_CONNECT_TIMEOUT=15
export POSTGRES_QUERY_TIMEOUT=600
export POSTGRES_COMMAND_TIMEOUT=600

# Table prefix mapping:
# adunbox_ -> adunbox_traffic_source_reports / adunbox_daily_breakdown_kpis
# empty    -> traffic_source_reports / daily_breakdown_kpis
export ADUNBOX_DB_TABLE_PREFIX=adunbox_

export ADUNBOX_6H_DB_LOOKBACK_DAYS=8
export ADUNBOX_6H_DB_ROW_LIMIT=500
export ADUNBOX_24H_DB_LOOKBACK_DAYS=7
export ADUNBOX_24H_DB_ROW_LIMIT=500
export ADUNBOX_24H_DB_RETRY_ON_TIMEOUT=true
export ADUNBOX_6H_SCORE_CHUNKSIZE=25000
export ADUNBOX_USE_FORECAST_ELIGIBILITY_VIEW=false
export ADUNBOX_DEBUG_AD_IDS=
export ADUNBOX_DEBUG_ACCOUNT_IDS=

export ADUNBOX_WRITE_FEATURE_CACHE=true
export ADUNBOX_REUSE_FEATURE_CACHE=false
export ADUNBOX_CLEAN_LOCAL_OUTPUTS_AFTER_DB_WRITE=false

dagster dev -f orchestration/production_dagster_assets.py -h 127.0.0.1 -p 3000
```

If the same schema exists without the `adunbox_` prefix, switch only the prefix:

```bash
export ADUNBOX_DB_TABLE_PREFIX=
```

### Vibelets Direct PostgreSQL Run

Use this for the newer Vibelets database where source tables are named
`traffic_source_reports`, `traffic_source_accounts`, and `daily_breakdown_kpis`
without the `adunbox_` prefix. Do not commit real passwords.

```bash
cd /g/ml_model_historical_data/github_release

export ADUNBOX_USE_DATABASE=true
export ADUNBOX_DB_TABLE_PREFIX=

# Option A: one-line DSN
export PG_DATABASE_URL="postgres://username:your_password_here@host:5432/db_name"

# Option B: individual connection fields. Use this only if PG_DATABASE_URL is empty.
export POSTGRES_HOST="host"
export POSTGRES_PORT="5432"
export POSTGRES_DB="db_name"
export POSTGRES_USER="username"
export POSTGRES_PASSWORD="your_password_here"

export ADUNBOX_REQUIRE_ACTIVE_HIERARCHY=true
export ADUNBOX_6H_REPORTS_REVENUE_COLUMN=conversions_value
export ADUNBOX_6H_REPORTS_CONVERSIONS_COLUMN=conversions
export ADUNBOX_24H_DAILY_REVENUE_COLUMN=conversions_value
export ADUNBOX_24H_DAILY_CONVERSIONS_COLUMN=conversions

export ADUNBOX_6H_DB_LOOKBACK_DAYS=8
export ADUNBOX_6H_DB_ROW_LIMIT=500
export ADUNBOX_24H_DB_LOOKBACK_DAYS=21
export ADUNBOX_24H_DB_ROW_LIMIT=500
export ADUNBOX_24H_DB_RETRY_ON_TIMEOUT=true

export ADUNBOX_WRITE_FORECASTS_TO_DB=true
export ADUNBOX_FORECAST_TABLE=vibelets_model_forecasts
export ADUNBOX_WRITE_FEATURE_CACHE=false
export ADUNBOX_CLEAN_LOCAL_OUTPUTS_AFTER_DB_WRITE=true

dagster dev -f orchestration/production_dagster_assets.py -h 127.0.0.1 -p 3000
```

In Dagster, run either `adunbox_6h_forecast_job`, `adunbox_24h_forecast_job`, or
`adunbox_production_forecast_job`. The forecast rows will be written to:

```text
vibelets_model_forecasts
```

If production has custom table names, override individual tables:

```bash
export ADUNBOX_TRAFFIC_SOURCE_REPORTS_TABLE=public.traffic_source_reports
export ADUNBOX_TRAFFIC_SOURCE_ACCOUNTS_TABLE=public.traffic_source_accounts
export ADUNBOX_DAILY_BREAKDOWN_KPIS_TABLE=public.daily_breakdown_kpis
```

If the 6h reports table stores revenue/conversions under different column names, map them:

```bash
export ADUNBOX_6H_REPORTS_REVENUE_COLUMN=conversions_value
export ADUNBOX_6H_REPORTS_CONVERSIONS_COLUMN=conversions
```

If the 24h daily table also stores the populated revenue/conversion fields under
`conversions_value` and `conversions`, map those too:

```bash
export ADUNBOX_24H_DAILY_REVENUE_COLUMN=conversions_value
export ADUNBOX_24H_DAILY_CONVERSIONS_COLUMN=conversions
```

If the source only has reports/accounts and you do not want campaign/adset/ad status filtering during a Vibelets-style DB test:

```bash
export ADUNBOX_REQUIRE_ACTIVE_HIERARCHY=false
```

To force-test one or more specific ads in the 6h database extract:

```bash
export ADUNBOX_DEBUG_AD_IDS="1968522420"
export ADUNBOX_6H_DB_ROW_LIMIT=1
```

To force-test one account and forecast only active/scheduled eligible ads inside that account:

```bash
export ADUNBOX_DEBUG_ACCOUNT_IDS="35044270"
export ADUNBOX_DEBUG_AD_IDS=
export ADUNBOX_6H_DB_ROW_LIMIT=500
```

Unset debug scope for normal production:

```bash
unset ADUNBOX_DEBUG_AD_IDS
unset ADUNBOX_DEBUG_ACCOUNT_IDS
```

If the first run succeeds and you want to write forecasts back to PostgreSQL:

```bash
export ADUNBOX_WRITE_FORECASTS_TO_DB=true
export ADUNBOX_FORECAST_TABLE=adunbox_model_forecasts
export ADUNBOX_WRITE_FEATURES_TO_DB=true
export ADUNBOX_FEATURE_TABLE=adunbox_model_feature_cache
dagster dev -f orchestration/production_dagster_assets.py -h 127.0.0.1 -p 3000
```

## Fast Repeat Run

After one successful feature-building run, reuse the feature cache:

```bash
export ADUNBOX_REUSE_FEATURE_CACHE=true
dagster dev -f orchestration/production_dagster_assets.py -h 127.0.0.1 -p 3000
```

This avoids rebuilding feature rows from raw extracts and is the recommended quick validation path.

## Local Storage Notes

Dagster still needs a small local `DAGSTER_HOME` for run metadata/event logs.
The model scripts also need short-lived CSV/joblib working files while scoring.
For production-style runs where PostgreSQL is the final store, enable cleanup:

```bash
export ADUNBOX_WRITE_FEATURE_CACHE=false
export ADUNBOX_CLEAN_LOCAL_OUTPUTS_AFTER_DB_WRITE=true
```

With this enabled, successful DB sink steps remove bulky local extract/forecast/cache
files after rows are written to the forecast table. The final forecast table remains:

```text
vibelets_model_forecasts
```

## Outputs

```text
adunbox_daily_24h_latest_forecasts.csv
outputs/adunbox_6h_latest_forecasts.csv
outputs/adunbox_production_model_manifest.json
outputs/adunbox_forecast_persistence_status.json
```

If database persistence is enabled:

```text
Final forecast table: adunbox_model_forecasts
Feature review/debug table: adunbox_model_feature_cache
```

`adunbox_model_forecasts` contains the final 6h/24h forecast rows that business users should screen. It uses separate scalar columns for final served values (`result_*`) and raw model values (`raw_pred_*`), not JSON payload columns.

The separated `6h` and `24h` jobs can also write their own horizon rows to the same forecast table when `ADUNBOX_WRITE_FORECASTS_TO_DB=true`.

`adunbox_model_feature_cache` contains the exact feature rows used by the model, stored as JSONB payloads for debugging and audit.

## Why Runs Can Fail Locally

Most failures are not model failures. They usually come from:

```text
1. 24h DB query timeout due to large daily table scan.
2. Missing or tiny 24h ad-level history slice.
3. Laptop memory pressure from running Dagster + Python scoring together.
4. Wrong shell syntax for env vars.
```

Fixes:

```text
Use Git Bash export syntax.
Set small row limits for smoke tests.
For both 6h and 24h, the row limit means selected active ads; each selected ad still gets its full bounded history window.
Set POSTGRES_QUERY_TIMEOUT=600.
Set ADUNBOX_24H_DB_LOOKBACK_DAYS=7-14 for enough daily history.
Apply sql/adunbox_production_indexes.sql before production DB runs.
Use ADUNBOX_REUSE_FEATURE_CACHE=true after the first successful run.
```

## New / Low-History Ads

The scoring layer uses hierarchical fallback:

```text
If ad has enough history:
  use model prediction
Else if adset/campaign/account peers exist:
  use same-window benchmark fallback
Else:
  mark insufficient_history / monitoring
```

This prevents new ads from receiving overconfident model predictions.
