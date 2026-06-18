REPORTS_6H_SQL = """
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


DAILY_24H_SQL = """
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
