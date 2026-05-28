# Production Orchestration Plan

This is the proposed production flow for Adunbox forecasting using Dagster-style orchestration.

The user called the orchestration tool `Dextor/Dapster`; the Python orchestration framework commonly used for this pattern is `Dagster`. If your team uses an internal wrapper with a different name, the same asset/job structure still applies.

## Goal

Run the forecasting system automatically so every active ad gets:

- 6h forecast from hourly sequence model.
- 24h forecast from daily model.
- raw metric predictions.
- KPI calculations.
- spike-risk / calibration context.
- dashboard or downstream recommendation output.

## Asset Flow

```text
traffic_source_reports
traffic_source_accounts
        ↓
hourly timezone join
        ↓
6h hourly feature table
        ↓
6h GRU scoring
        ↓
6h predictions

daily ad breakdown table
        ↓
24h daily feature table
        ↓
24h HistGB scoring
        ↓
24h predictions
        ↓
spike-risk + calibration layer
        ↓
KPI output / dashboard / recommendation layer
```

## 6h Production Logic

Use the hourly data pipeline.

1. Pull hourly traffic source report rows.
2. Join traffic source accounts to get timezone.
3. Convert UTC timestamp to local timestamp.
4. Build one dense hourly sequence per ad.
5. For each scoring anchor, collect last `168` hours.
6. Fill missing hours with zero but keep enough-history flags.
7. Feed sequence into GRU model.
8. Predict next 6h raw totals:
   - spend
   - impressions
   - clicks
   - conversions
   - revenue
9. Derive KPIs:
   - ROAS = revenue / spend
   - CTR = clicks / impressions * 100
   - CPM = spend / impressions * 1000
   - CVR = conversions / clicks * 100

## 24h Production Logic

Use the daily data pipeline.

1. Pull daily ad-level breakdown rows.
2. For each ad/date, collect previous 7 daily rows.
3. Build daily features from raw metrics and KPI history.
4. Score five raw metric models:
   - spend
   - impressions
   - clicks
   - conversions
   - revenue
5. Derive KPIs from predicted raw metrics.
6. Apply spike-aware / calibration layer.
7. Save final 24h forecast output.

## Edge Cases

### New Account With Less History

If a new account appears after the original training window:

```text
If account has 7+ daily rows:
  Use current production model immediately.
  Score using the account/ad recent history.

If account has 1-6 daily rows:
  Score with partial-history fallback.
  Mark confidence LOW.
  Use account/campaign benchmark if available.

If account has 0 usable rows:
  Do not forecast.
  Mark status = insufficient_history.
  Fetch last 30 days from source API/database.
```

Important: a new account does not always require retraining. The trained model can score it if the feature schema is the same. Retraining is needed when new data distribution is very different or when scheduled retraining runs.

### Missing Hourly History For 6h

```text
>= 168 hours available:
  full-history forecast

24-167 hours available:
  padded forecast
  confidence LOW/MEDIUM depending on observed hours

< 24 hours available:
  insufficient history
  monitor only or conservative fallback
```

### Missing Daily History For 24h

```text
>= 7 days:
  full-history forecast

3-6 days:
  partial-history forecast or benchmark fallback

< 3 days:
  do not make aggressive decision
  fetch more data or mark monitoring
```

## Suggested Dagster Assets

```python
traffic_source_reports_raw
traffic_source_accounts_raw
hourly_timezone_joined
daily_ad_breakdown
features_6h_hourly_sequences
features_24h_daily_windows
score_6h_gru
score_24h_histgb
apply_spike_calibration
build_forecast_dashboard
publish_forecast_outputs
```

## Suggested Schedule

```text
Hourly:
  refresh hourly traffic reports
  score 6h forecast for active ads

Daily after reporting completes:
  refresh daily breakdown
  score 24h forecast
  rebuild dashboard/output tables

Weekly:
  retrain 6h model
  retrain 24h model
  compare validation metrics
  promote only if test metrics improve
```

## Production Outputs

Recommended output tables:

```text
ad_forecasts_6h
ad_forecasts_24h
ad_forecast_kpis
ad_forecast_spike_risk
ad_forecast_backtest_examples
```

## Monitoring

Track these daily:

- prediction coverage percentage
- LOW / MEDIUM / HIGH spike-risk distribution
- negative prediction count
- 24h prediction less than 6h count
- ROAS error
- revenue error
- spend error
- large-ad underprediction rate
- new-account partial-history count

## Promotion Rule

Do not auto-promote a retrained model only because it exists.

Promote only when:

```text
test R2 improves or remains stable
key business metric error does not degrade
large-ad underprediction does not worsen
coverage remains acceptable
manual dashboard review passes
```

