# Adunbox Forecasting Models

This repo contains the cleaned production-ready subset for Adunbox forecasting:

- `6h` prediction: final hybrid GRU using last `168` hourly rows per ad.
- `24h` prediction: daily HistGradientBoosting models using last `7` daily rows per ad.
- Review dashboard: historical actual vs predicted KPI backtest for `CTR`, `CPM`, `ROAS`, and `CVR`.

## What Is Included

- `models/adunbox_entity_history_gru_168h_padded_6h/`
  - 6h baseline volume GRU used for spend, impressions, and clicks.
- `models/adunbox_entity_history_gru_168h_padded_6h_hybrid/`
  - Final promoted 6h hybrid layer used for CVR/ROAS, with conversions and revenue reconstructed from predicted clicks/spend.
- `models/adunbox_daily_24h_histgb/`
  - Trained 24h daily regressors for spend, impressions, clicks, conversions, and revenue.
- `scripts/`
  - Training, scoring, and dashboard-generation scripts.
- `docs/`
  - Current model metrics and summaries.
- `dashboards/`
  - Shareable HTML review dashboard.

## What Is Intentionally Excluded

Large local datasets, historical CSV dumps, JSON exports, parquet files, logs, and experimental model folders are not included. Keep those in storage or a data lake, not GitHub.

## Data Expected By The Models

### 6h GRU Input

The 6h model expects hourly ad-level data joined with account timezone metadata.

Required grain:

```text
one row per account_id / campaign_id / adset_id / ad_id / hour
```

Core fields:

```text
date or timestamp UTC
timezone
account_id
campaign_id
adset_id
ad_id
spend
impressions
inline_link_clicks
tracker_conversions
tracker_revenue
```

The pipeline converts UTC to local time, builds a dense 168-hour sequence, fills missing hours with zero, and predicts the next 6-hour totals. The promoted 6h production setup is the four-rule hybrid path:

```text
Rule 1: baseline 168h GRU predicts spend / impressions / clicks
Rule 2: hybrid GRU predicts CVR
Rule 3: hybrid GRU predicts ROAS
Rule 4: reconstruct conversions and revenue
        conversions = predicted_clicks * predicted_CVR
        revenue = predicted_spend * predicted_ROAS
```

Do not use the later `rule3_removed` experiment results. Those were not promoted because conversions/revenue degraded.

### 24h Daily Model Input

The 24h model expects daily ad-level rows.

Required grain:

```text
one row per account_id / campaign_id / adset_id / ad_id / day
```

Core fields:

```text
date
account_id
campaign_id
adset_id
ad_id
spend
impressions
inline_link_clicks
tracker_conversions
tracker_revenue
```

The model uses the previous 7 daily rows to forecast the next 24-hour day.

## Local Smoke Test

Before pushing or handing to a teammate:

```bash
python orchestration/local_smoke_run.py
python -m compileall -q scripts orchestration
```

This confirms the selected promoted model artifacts are present. Full scoring still requires real hourly/daily input data.

## Run Dagster Locally

From inside `github_release`:

```bash
pip install -r requirements.txt
dagster dev -f orchestration/dagster_assets.py
```

Open:

```text
http://localhost:3000
```

In the Dagster UI you should see the asset graph and the job named `adunbox_forecast_job`. Materialize the full graph or launch that job. Outputs are written to `outputs/`, including:

```text
adunbox_6h_latest_forecasts_published.csv
adunbox_24h_latest_forecasts_published.csv
adunbox_6h_historical_quality_summary.csv
orchestration_data_summary.json
forecast_publish_summary.json
```

To point Dagster at a larger local dataset without changing code:

```powershell
$env:ADUNBOX_HOURLY_INPUT="G:\path\to\traffic_reports.csv"
$env:ADUNBOX_DAILY_INPUT="G:\path\to\adunbox_daily_breakdown_kpis.csv"
$env:ADUNBOX_6H_REVIEW_INPUT="G:\path\to\adunbox_6h_final_prediction_history_benchmark_review.csv"
dagster dev -f orchestration/dagster_assets.py
```

The bundled orchestration now also reads the larger 6h historical review file from `docs/` and publishes a quality summary, so the local Dagster graph has more data context than only the latest scoring CSVs.

## Production Note

Use this repo as the code/model package. Raw data should come from database queries or object storage at runtime, not from committed CSV files.
