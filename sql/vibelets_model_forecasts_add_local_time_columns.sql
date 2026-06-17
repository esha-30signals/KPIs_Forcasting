ALTER TABLE vibelets_model_forecasts
    ADD COLUMN IF NOT EXISTS timezone TEXT,
    ADD COLUMN IF NOT EXISTS forecast_anchor_local TIMESTAMP,
    ADD COLUMN IF NOT EXISTS forecast_window_start_local TIMESTAMP,
    ADD COLUMN IF NOT EXISTS forecast_window_end_local TIMESTAMP,
    ADD COLUMN IF NOT EXISTS forecast_local_date DATE,
    ADD COLUMN IF NOT EXISTS forecast_confidence_score DOUBLE PRECISION;
