-- Optional production view for scheduled activation support.
--
-- Use this when forecasts should include:
--   1. ads whose account/campaign/adset/ad hierarchy is currently ACTIVE, or
--   2. ads scheduled to become active inside the forecast window.
--
-- IMPORTANT:
-- Replace the scheduled_* placeholder expressions below with the real schedule
-- columns/tables used by production. Keep the output column names unchanged.

CREATE OR REPLACE VIEW public.adunbox_forecast_eligible_ads AS
SELECT
    acc.id AS account_id,
    c.id AS campaign_id,
    s.id AS adset_id,
    ad.id AS ad_id,

    (
        UPPER(COALESCE(acc.status, '')) = 'ACTIVE'
        AND UPPER(COALESCE(c.status, '')) = 'ACTIVE'
        AND UPPER(COALESCE(s.status, '')) = 'ACTIVE'
        AND UPPER(COALESCE(ad.status, '')) = 'ACTIVE'
    ) AS is_currently_active,

    -- Replace NULL with the earliest timestamp when the full hierarchy is
    -- scheduled to become active. Example:
    -- GREATEST(acc.scheduled_active_from, c.scheduled_active_from,
    --          s.scheduled_active_from, ad.scheduled_active_from)
    NULL::timestamptz AS scheduled_active_from,

    -- Replace NULL with the timestamp when the scheduled active period ends,
    -- if available. Leave NULL when the active period is open-ended.
    NULL::timestamptz AS scheduled_active_until,

    CASE
        WHEN (
            UPPER(COALESCE(acc.status, '')) = 'ACTIVE'
            AND UPPER(COALESCE(c.status, '')) = 'ACTIVE'
            AND UPPER(COALESCE(s.status, '')) = 'ACTIVE'
            AND UPPER(COALESCE(ad.status, '')) = 'ACTIVE'
        ) THEN 'currently_active'
        ELSE 'not_currently_active'
    END AS eligibility_status
FROM public.adunbox_traffic_source_ads ad
JOIN public.adunbox_traffic_source_adsets s
  ON s.id = ad.adset_id
JOIN public.adunbox_traffic_source_campaigns c
  ON c.id = ad.campaign_id
JOIN public.adunbox_traffic_source_accounts acc
  ON acc.id = ad.account_id;

CREATE INDEX IF NOT EXISTS idx_adunbox_forecast_eligible_ads_ad_id
ON public.adunbox_traffic_source_ads (id);
