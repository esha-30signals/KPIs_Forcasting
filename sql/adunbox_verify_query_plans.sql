-- Use these checks after indexes are created.
-- The ideal plan should use an Index Scan / Index Only Scan, not a full Seq Scan.

EXPLAIN ANALYZE
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
FROM public.adunbox_daily_breakdown_kpis
WHERE entity_type = 'ad'
  AND ad_id IS NOT NULL
  AND date >= NOW() - INTERVAL '7 days'
  AND date <= NOW()
ORDER BY date DESC
LIMIT 500;

EXPLAIN ANALYZE
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
FROM public.adunbox_traffic_source_reports r
LEFT JOIN public.adunbox_traffic_source_accounts a
    ON r.account_id = a.id
   AND r.company_id = a.company_id
   AND r.traffic_source_id = a.traffic_source_id
   AND r.traffic_source_config_id = a.traffic_source_config_id
WHERE r.date >= NOW() - INTERVAL '7 days'
  AND r.date <= NOW()
ORDER BY r.date DESC
LIMIT 500;
