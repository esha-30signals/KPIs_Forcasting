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

export ADUNBOX_6H_DB_LOOKBACK_DAYS=7
export ADUNBOX_6H_DB_ROW_LIMIT=500
export ADUNBOX_24H_DB_LOOKBACK_DAYS=7
export ADUNBOX_24H_DB_ROW_LIMIT=500
export ADUNBOX_24H_DB_RETRY_ON_TIMEOUT=true
export ADUNBOX_6H_SCORE_CHUNKSIZE=25000

export ADUNBOX_WRITE_FEATURE_CACHE=true
export ADUNBOX_REUSE_FEATURE_CACHE=false

dagster dev -f orchestration/production_dagster_assets.py -h 127.0.0.1 -p 3000
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

`adunbox_model_forecasts` contains the final 6h/24h forecast rows that business users should screen.

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
