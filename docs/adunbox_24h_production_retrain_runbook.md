# Adunbox 24h Model Retrain / Holdout Runbook

## What Is Included

The `github_release/scripts` folder now contains the complete 24h process:

- `train_adunbox_daily_24h_full_db_optimized.py`
- `score_adunbox_daily_24h_optimized_all_history.py`
- `build_adunbox_24h_optimized_three_account_dashboard.py`
- `score_adunbox_daily_24h_recent_holdout.py`
- `build_adunbox_24h_recent_holdout_dashboard.py`
- `train_adunbox_daily_24h_full_db_production.py`

## Current Model Flow

1. Train optimized 24h daily model from daily ad-level data.
2. Predict raw metrics: spend, impressions, clicks, conversions, revenue.
3. Derive KPIs from raw predictions: ROAS, profit, CTR, CVR, CPM.
4. Apply calibration, segment flags, prediction ranges, confidence, and guardrails.
5. Build actual-vs-predicted dashboards.

## Run Optimized Historical Model

```powershell
python github_release\scripts\train_adunbox_daily_24h_full_db_optimized.py
python github_release\scripts\score_adunbox_daily_24h_optimized_all_history.py
python github_release\scripts\build_adunbox_24h_optimized_three_account_dashboard.py
```

## Run Recent Out-of-Time Holdout

This uses the frozen optimized model and tests it on recent unseen data.

```powershell
python github_release\scripts\score_adunbox_daily_24h_recent_holdout.py
python github_release\scripts\build_adunbox_24h_recent_holdout_dashboard.py
```

Expected recent input:

```text
H:\adunbox_daily_breakdown_kpis.csv
```

## Run Production Retrain With Recent Data

This combines the original daily file plus the recent H-drive daily file, applies recency weighting, and retrains a new production model.

```powershell
python github_release\scripts\train_adunbox_daily_24h_full_db_production.py
```

Production retrain behavior:

- Uses original plus recent daily data.
- Uses May/June rows with higher recency weight.
- Keeps segment-specific calibration.
- Keeps prediction ranges and guardrails.
- Writes a separate production model folder so the current optimized model is not overwritten.

## Key Outputs

```text
github_release\outputs\adunbox_daily_24h_full_db_optimized__metrics.csv
github_release\outputs\adunbox_daily_24h_full_db_optimized__all_history_predictions.csv
github_release\outputs\adunbox_daily_24h_recent_holdout_predictions.csv
github_release\outputs\adunbox_daily_24h_recent_holdout_metrics.csv
github_release\outputs\adunbox_daily_24h_recent_holdout_summary.txt
```

Recent holdout dashboard:

```text
H:\adunbox_24h_recent_holdout_actual_vs_predicted_dashboard.html
```

## Notes

- The 24h model predicts raw metrics first.
- KPIs are derived from the raw predictions.
- Revenue and profit are currently the strongest 24h outputs.
- CTR, CVR, and CPM are weaker because they are ratio metrics and should be interpreted with guardrails/ranges.
