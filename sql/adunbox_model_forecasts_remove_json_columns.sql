-- Optional cleanup for old local/production forecast tables.
-- New pipeline writes scalar columns only and does not use these JSON columns.
--
-- Run only after confirming no downstream dashboard/query depends on these columns.

ALTER TABLE public.adunbox_model_forecasts
    DROP COLUMN IF EXISTS results,
    DROP COLUMN IF EXISTS payload;
