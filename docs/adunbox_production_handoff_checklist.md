# Adunbox Production Handoff Checklist

## What Is Ready

```text
Dagster assets are wired.
6h production model is wired.
24h production model is wired.
PostgreSQL source extracts are wired.
Forecast DB sink is wired.
Feature-cache DB sink is optional and wired.
Local CSV smoke mode is available.
```

## What Must Be Done Before Production

1. Configure secrets outside Git:

```text
POSTGRES_HOST
POSTGRES_PORT
POSTGRES_DB
POSTGRES_USER
POSTGRES_PASSWORD
ADUNBOX_USE_DATABASE=true
```

2. Add/verify DB indexes:

```sql
-- See sql/adunbox_production_indexes.sql
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_adunbox_dbk_ad_rows_date
ON public.adunbox_daily_breakdown_kpis (date DESC)
WHERE entity_type = 'ad' AND ad_id IS NOT NULL;
```

3. Keep scoring bounded:

```text
ADUNBOX_24H_DB_LOOKBACK_DAYS=7 to 14
ADUNBOX_24H_DB_ROW_LIMIT=500 to 5000
ADUNBOX_6H_DB_LOOKBACK_DAYS=7 to 14
ADUNBOX_6H_DB_ROW_LIMIT=500 to 5000
```

4. Run one DB-read dry run:

```text
ADUNBOX_WRITE_FORECASTS_TO_DB=false
ADUNBOX_WRITE_FEATURES_TO_DB=false
```

5. Run one DB-write dry run:

```text
ADUNBOX_WRITE_FORECASTS_TO_DB=true
ADUNBOX_FORECAST_TABLE=adunbox_model_forecasts
```

6. Verify output:

```sql
SELECT forecast_horizon, account_id, campaign_id, adset_id, ad_id,
       forecast_anchor, forecast_confidence, forecast_status, created_at
FROM adunbox_model_forecasts
ORDER BY created_at DESC
LIMIT 50;
```

## Current Known Risks

```text
24h DB extract can timeout if the date-first ad-row index is missing.
Low-RAM laptops can fail if large CSV fallback is used.
Tiny lookback windows can return zero rows if DB data is not current.
Feature/materialized tables are not required for phase 1, but are recommended later.
```

## Phase 2 Optimization

```text
Move feature generation into scheduled DB/materialized feature tables.
Then Dagster scoring reads compact feature rows instead of raw source tables.
```
