# Adunbox 24h Forecast Deployment Readiness

## Production Model To Use

Use the current trained LightGBM 24h daily model artifacts in:

```text
github_release/models/adunbox_daily_24h_histgb_full_db_production/
```

Required files:

```text
metadata.joblib
target_24h_spend.joblib
target_24h_impressions.joblib
target_24h_inline_link_clicks.joblib
target_24h_tracker_conversions.joblib
target_24h_tracker_revenue.joblib
```

This is the corrected no-overlap model trained from:

```text
C:\Users\eshaa\Downloads\adunbox_daily_breakdown_kpis.csv
H:\adunbox_daily_breakdown_kpis.csv
```

Overlap handling:

```text
2026-05-04 to 2026-05-12 overlap was deduped.
For duplicate same ad/date rows across sources, the later/recent source row was kept.
Rows were not summed across overlapping files.
```

## Final Forecast Rule

For production forecasting, use raw model p50 predictions as the point forecast:

```text
pred_24h_spend
pred_24h_impressions
pred_24h_inline_link_clicks
pred_24h_tracker_conversions
pred_24h_tracker_revenue
```

Then derive KPIs:

```text
ROAS = predicted_revenue / predicted_spend
CTR  = predicted_clicks / predicted_impressions * 100
CVR  = predicted_conversions / predicted_clicks * 100
CPM  = predicted_spend / predicted_impressions * 1000
Profit = predicted_revenue - predicted_spend
```

Do not use calibrated columns as the main production point forecast. Calibration is useful as context/range, but the latest out-of-time tests showed raw p50 performs better overall.

## Current Best Backtest Quality

Latest raw model backtest monitor:

```text
Clicks R2:       0.433
Impressions R2:  0.235
Conversions R2:  0.200
Spend R2:        0.173
Revenue R2:      0.121
ROAS R2:         0.018
```

Final weak-metric selector test:

```text
Raw p50 remained better than final-selected/blended output on test R2.

Test raw:
  Spend R2:   0.256
  Revenue R2: 0.049
  ROAS R2:    0.005

Test final-selected:
  Spend R2:   0.250
  Revenue R2: 0.047
  ROAS R2:    0.005
```

Decision:

```text
Do not replace production point forecasts with final-selected/blended values.
Use raw p50 as the official production forecast.
Use final-selected experiments only as analysis artifacts.
```

Important interpretation:

```text
Revenue and ROAS are still the hardest metrics because revenue is sparse, zero-heavy, and spiky.
The model is useful for forecasting direction and raw metric context, but spiky/new/low-volume ads should be treated as lower confidence.
```

## Deployment Scripts

Compile-checked scripts:

```text
github_release/scripts/score_adunbox_daily_24h_model.py
github_release/scripts/monitor_adunbox_daily_24h_forecast_quality.py
github_release/scripts/train_adunbox_daily_24h_full_db_production.py
```

Recommended deployment flow:

```powershell
python github_release\scripts\score_adunbox_daily_24h_model.py
python github_release\scripts\monitor_adunbox_daily_24h_forecast_quality.py --mode base
```

## Final HTML Prototype

Use this dashboard for review:

```text
H:\adunbox_24h_improved_last7_history_dashboard.html
```

It shows last-7-day history plus actual-vs-predicted KPI/raw metric behavior using the latest safe forecast files.

## Production Guardrails

Use these forecast confidence rules:

```text
stable_history:
  Use point forecast.

mixed_history or spiky_history:
  Show forecast range and warning.

low_volume, mostly_zero_history, new_ad:
  Use forecast as directional only.

inactive:
  Treat as inactive/no-action forecast.
```

## Final Note

Further improvement is possible, but it should be done through a memory-safe retraining job or a larger machine. Experimental longer-history feature attempts hit local RAM limits during full-data feature building, so they should not be considered deployed until a full retrain completes successfully.
