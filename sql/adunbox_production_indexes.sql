-- Adunbox forecasting production indexes.
--
-- Run these in PostgreSQL before production Dagster scoring.
-- Use CONCURRENTLY so normal reads/writes are not blocked while the index builds.

-- 24h daily model extract:
-- The production query pulls latest ad-level rows across all ads by date.
-- Existing (ad_id, date) indexes help one-ad lookups, but this date-first
-- partial index is better for batch scoring.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_adunbox_dbk_ad_rows_date
ON public.adunbox_daily_breakdown_kpis (date DESC)
WHERE entity_type = 'ad' AND ad_id IS NOT NULL;

-- 6h hourly model extract:
-- The production query pulls recent hourly rows and later groups by ad/entity.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_adunbox_reports_date_ad
ON public.adunbox_traffic_source_reports (date DESC, ad_id);

-- 6h timezone join support:
-- Verify this already exists before creating. It helps the reports -> accounts
-- join used to attach timezone.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_adunbox_accounts_join_keys
ON public.adunbox_traffic_source_accounts (
  id,
  company_id,
  traffic_source_id,
  traffic_source_config_id
);

ANALYZE public.adunbox_daily_breakdown_kpis;
ANALYZE public.adunbox_traffic_source_reports;
ANALYZE public.adunbox_traffic_source_accounts;
