# Adunbox Dagster Productionization Plan

## Goal

Run the final 6h and 24h forecasting models through Dagster using either local CSV inputs or direct PostgreSQL extracts.

```text
source extract
  -> feature engineering / feature cache
  -> model scoring
  -> confidence + fallback
  -> forecast files
  -> optional PostgreSQL forecast sink
```

## Final Production Models

### 6h

Target-routed LightGBM setup:

```text
spend / impressions / clicks:
  models/adunbox_entity_history_lgbm_6h_anchor_v2/

conversions / revenue:
  models/adunbox_entity_history_lgbm_6h_business_v3/
```

### 24h

```text
models/adunbox_daily_24h_histgb_full_db_production/
```

## Dagster Entrypoint

```text
orchestration/production_dagster_assets.py
```

Main jobs:

```text
adunbox_6h_forecast_job
adunbox_24h_forecast_job
adunbox_production_forecast_job
```

Use the 6h/24h jobs separately for normal production scheduling or local low-RAM testing. When `ADUNBOX_WRITE_FORECASTS_TO_DB=true`, each separated job writes only its own horizon rows. Use the combined job only when both horizons plus optional persistence should run together.

Assets:

```text
final_model_registry
postgres_6h_hourly_extract
postgres_24h_daily_extract
adunbox_6h_production_ready_manifest
adunbox_24h_raw_forecast
adunbox_24h_served_forecast
adunbox_24h_quality_monitor
adunbox_6h_forecast_postgres_sink
adunbox_24h_forecast_postgres_sink
adunbox_forecast_postgres_sink
```

## Database Sources

### 6h Source

Reads hourly reports joined with account timezone:

```text
adunbox_traffic_source_reports
LEFT JOIN adunbox_traffic_source_accounts
```

The query filters recent rows using:

```text
ADUNBOX_6H_DB_LOOKBACK_DAYS
ADUNBOX_6H_DB_ROW_LIMIT
ADUNBOX_6H_DB_ANCHOR_DATE optional
```

For `6h`, `ADUNBOX_6H_DB_ROW_LIMIT` limits the number of selected active ads, not the number of raw hourly rows. Once an ad is selected, the extract keeps the full bounded hourly history for that ad so the 168-hour feature window is not accidentally truncated.

Database mode also requires active hierarchy:

```text
account.status = ACTIVE
campaign.status = ACTIVE
adset.status = ACTIVE
ad.status = ACTIVE
```

For scheduled activation use cases, enable the optional eligibility view:

```text
ADUNBOX_USE_FORECAST_ELIGIBILITY_VIEW=true
```

Then Dagster reads `public.adunbox_forecast_eligible_ads` instead of only checking current status. The view must return one row per ad hierarchy with:

```text
account_id
campaign_id
adset_id
ad_id
is_currently_active
scheduled_active_from
scheduled_active_until
eligibility_status
```

Eligibility logic:

```text
include ad if currently active
OR scheduled_active_from is inside the forecast window
```

The starter template is:

```text
sql/adunbox_forecast_eligible_ads_view_template.sql
```

### 24h Source

Reads ad-level rows from:

```text
adunbox_daily_breakdown_kpis
```

The default query filters:

```sql
WHERE entity_type = 'ad'
  AND ad_id IS NOT NULL
  AND date >= COALESCE(anchor_date, NOW()) - lookback_days
```

Database mode also filters to ads where account, campaign, adset, and ad are all `ACTIVE`.

This avoids a slow `MAX(date)` table scan during laptop/local production testing.

## Safe Laptop Defaults

These defaults are intentionally conservative for machines around `16 GB RAM`:

```text
ADUNBOX_6H_DB_LOOKBACK_DAYS=8
ADUNBOX_6H_DB_ROW_LIMIT=500
ADUNBOX_24H_DB_LOOKBACK_DAYS=7
ADUNBOX_24H_DB_ROW_LIMIT=500
ADUNBOX_24H_DB_RETRY_ON_TIMEOUT=true
ADUNBOX_6H_SCORE_CHUNKSIZE=25000
ADUNBOX_USE_FORECAST_ELIGIBILITY_VIEW=false
ADUNBOX_DEBUG_AD_IDS=
POSTGRES_POOL_MAX_SIZE=2
POSTGRES_QUERY_TIMEOUT=600
```

`ADUNBOX_DEBUG_AD_IDS` is only for debugging specific ads in the 6h DB extract. Example:

```bash
export ADUNBOX_DEBUG_AD_IDS="1968522420"
export ADUNBOX_6H_DB_ROW_LIMIT=1
```

Keep it empty/unset for normal production runs.

Why `24h` needs a larger lookback:

```text
The 24h model uses lag/rolling daily features.
A very small same-day slice can produce zero eligible scoring rows.
```

## Git Bash DB Test Commands

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

export ADUNBOX_6H_DB_LOOKBACK_DAYS=8
export ADUNBOX_6H_DB_ROW_LIMIT=500
export ADUNBOX_24H_DB_LOOKBACK_DAYS=7
export ADUNBOX_24H_DB_ROW_LIMIT=500
export ADUNBOX_24H_DB_RETRY_ON_TIMEOUT=true
export ADUNBOX_6H_SCORE_CHUNKSIZE=25000

export ADUNBOX_WRITE_FEATURE_CACHE=true
export ADUNBOX_REUSE_FEATURE_CACHE=false

dagster dev -f orchestration/production_dagster_assets.py -h 127.0.0.1 -p 3000
```

Then open:

```text
http://127.0.0.1:3000
```

Click:

```text
Materialize all
```

## Fast Repeat Test

After the first successful run:

```bash
export ADUNBOX_REUSE_FEATURE_CACHE=true
dagster dev -f orchestration/production_dagster_assets.py -h 127.0.0.1 -p 3000
```

This reuses:

```text
outputs/adunbox_24h_latest_feature_cache.joblib
outputs/adunbox_6h_latest_feature_cache.joblib
```

## Forecast Persistence

To write forecasts back into PostgreSQL:

```bash
export ADUNBOX_WRITE_FORECASTS_TO_DB=true
export ADUNBOX_FORECAST_TABLE=adunbox_model_forecasts
```

To also write the exact model feature rows used for scoring:

```bash
export ADUNBOX_WRITE_FEATURES_TO_DB=true
export ADUNBOX_FEATURE_TABLE=adunbox_model_feature_cache
```

The sink creates table `adunbox_model_forecasts` if missing and stores final forecast results:

```text
forecast_horizon
account_id / campaign_id / adset_id / ad_id
forecast timestamps
confidence/status/source columns
final served metric columns: result_spend, result_revenue, result_roas, etc.
raw model metric columns: raw_pred_spend, raw_pred_revenue, raw_pred_roas, etc.
```

The forecast table intentionally avoids JSON columns so business users can query/filter every metric directly.

It also creates `adunbox_model_feature_cache` when feature persistence is enabled:

```text
feature_horizon
account_id / campaign_id / adset_id / ad_id
feature_anchor
feature_source
complete model feature row as JSONB payload
```

Recommended screening query:

```sql
SELECT
    forecast_horizon,
    account_id,
    campaign_id,
    adset_id,
    ad_id,
    forecast_anchor,
    forecast_confidence,
    forecast_status,
    benchmark_source,
    model_source,
    result_spend,
    result_impressions,
    result_clicks,
    result_conversions,
    result_revenue,
    result_roas,
    result_ctr,
    result_cvr,
    result_cpm,
    raw_pred_spend,
    raw_pred_impressions,
    raw_pred_clicks,
    raw_pred_conversions,
    raw_pred_revenue,
    created_at
FROM adunbox_model_forecasts
ORDER BY created_at DESC
LIMIT 100;
```

## Why Local Runs Were Failing

Observed issues:

```text
1. 24h extract timed out because the daily table query was too heavy.
2. Small row limits could pull only latest rows and truncate per-ad history.
3. PowerShell/Dagster processes created RAM pressure on a 16 GB laptop.
4. Git Bash and PowerShell env syntax were mixed.
```

Fixes now applied:

```text
1. 24h SQL filters ad-level rows before scoring.
2. DB anchor defaults to NOW(), not MAX(date), reducing table scans.
3. Safe row-limit/lookback defaults are set in Dagster.
4. 6h DB extract now selects active ads first, then pulls their full bounded hourly history.
5. 24h scorer handles empty eligible slices without crashing.
6. Feature cache supports faster repeat runs.
7. .gitignore excludes local Dagster state and secrets.
7. Production index SQL is provided in sql/adunbox_production_indexes.sql.
```

## Low-History / New Ads

Scoring uses hierarchical fallback:

```text
If ad has enough history:
  model prediction
Else if enough peer data exists:
  adset/campaign/account same-window benchmark
Else:
  insufficient_history / monitoring
```

This keeps production forecasts safe for new ads and avoids overconfident predictions.

## What To Push

Push:

```text
db/
docs/
models/adunbox_daily_24h_histgb_full_db_production/
models/adunbox_entity_history_lgbm_6h_anchor_v2/
models/adunbox_entity_history_lgbm_6h_business_v3/
orchestration/
scripts/
.env.example
.gitignore
README.md
requirements.txt
```

Do not push:

```text
outputs/
data/
.tmp_dagster_home*/
.env
raw CSV/JSON/parquet dumps
```
