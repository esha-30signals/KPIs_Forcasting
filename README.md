# Adunbox Forecasting Models

This repo contains the cleaned production-ready subset for Adunbox forecasting:

- `6h` prediction: GRU sequence model using last `168` hourly rows per ad.
- `24h` prediction: daily HistGradientBoosting models using last `7` daily rows per ad.
- Review dashboard: historical actual vs predicted KPI backtest for `CTR`, `CPM`, `ROAS`, and `CVR`.

## What Is Included

- `models/adunbox_entity_history_gru_168h_padded_6h/`
  - Trained 6h GRU model and scalers.
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

The pipeline converts UTC to local time, builds a dense 168-hour sequence, fills missing hours with zero, and predicts the next 6-hour totals.

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

## GitHub Commands

Set Git identity:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

Create and push a new repo:

```bash
cd G:/ml_model_historical_data/github_release
git init
git add .
git commit -m "Initial Adunbox forecasting production release"
git branch -M main
git remote add origin https://github.com/<your-org-or-user>/<repo-name>.git
git push -u origin main
```

If the remote already exists:

```bash
git remote -v
git remote set-url origin https://github.com/<your-org-or-user>/<repo-name>.git
git push -u origin main
```

## Production Note

Use this repo as code/model package. Raw data should come from database queries or object storage at runtime, not from committed CSV files.

