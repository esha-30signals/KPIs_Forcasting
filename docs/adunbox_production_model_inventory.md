# Adunbox Production Model Inventory

## Keep For Production

### 6h Forecasting

Production model is a target-wise LightGBM ensemble:

```text
models/adunbox_entity_history_lgbm_6h_anchor_v2/
models/adunbox_entity_history_lgbm_6h_business_v3/
```

Target routing:

```text
spend       -> adunbox_entity_history_lgbm_6h_anchor_v2
impressions -> adunbox_entity_history_lgbm_6h_anchor_v2
clicks      -> adunbox_entity_history_lgbm_6h_anchor_v2
conversions -> adunbox_entity_history_lgbm_6h_business_v3
revenue     -> adunbox_entity_history_lgbm_6h_business_v3
```

Selected 6h test R2:

```text
spend:       0.357898
impressions: 0.424732
clicks:      0.495425
conversions: 0.423132
revenue:     0.316992
```

Reference:

```text
docs/adunbox_6h_production_model_selection__summary.txt
docs/adunbox_6h_production_model_selection__metrics.csv
```

### 24h Forecasting

Production model:

```text
models/adunbox_daily_24h_histgb_full_db_production/
```

Use raw p50 prediction columns as the official point forecast:

```text
pred_24h_spend
pred_24h_impressions
pred_24h_inline_link_clicks
pred_24h_tracker_conversions
pred_24h_tracker_revenue
```

Derive KPIs after raw prediction:

```text
ROAS   = revenue / spend
CTR    = clicks / impressions * 100
CPM    = spend / impressions * 1000
CVR    = conversions / clicks * 100
Profit = revenue - spend
```

Current 24h raw backtest monitor:

```text
clicks:      0.432811
impressions: 0.234713
conversions: 0.199714
spend:       0.173075
revenue:     0.121168
ROAS:        0.017755
```

Reference:

```text
docs/adunbox_24h_deployment_readiness_report.md
models/adunbox_daily_24h_histgb_full_db_production/deployment_manifest.json
```

## Archive / Do Not Push As Production

These are research/trial folders and should be moved to archive or excluded from the production repo unless needed for audit:

```text
models/adunbox_daily_24h_histgb/
models/adunbox_daily_24h_histgb_full_db_improved/
models/adunbox_daily_24h_histgb_full_db_optimized/
models/adunbox_entity_history_gru_168h_padded_6h/
models/adunbox_entity_history_gru_168h_padded_6h_hybrid/
models/adunbox_entity_history_lgbm_168h_padded_6h/
models/adunbox_entity_history_lgbm_6h_full_fast/
```

Keep `vibelets_daily_24h_histgb_production/` only if Vibelets is part of the same deployment. Otherwise keep it in a separate Vibelets release.

## Recommended Repo Policy

```text
Push production model folders only.
Push metrics and deployment docs.
Do not push large trial dashboards and experiment CSVs.
Move old trials to H:/model_archive or a separate experiment branch.
```

