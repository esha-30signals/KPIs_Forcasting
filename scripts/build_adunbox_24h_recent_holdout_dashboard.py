from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(r"G:\ml_model_historical_data")
PREDICTIONS = BASE_DIR / "github_release" / "outputs" / "adunbox_daily_24h_recent_holdout_predictions.csv"
METRICS = BASE_DIR / "github_release" / "outputs" / "adunbox_daily_24h_recent_holdout_metrics.csv"
RECENT_DAILY = Path(r"H:\adunbox_daily_breakdown_kpis.csv")
OUT = Path(r"H:\adunbox_24h_recent_holdout_actual_vs_predicted_dashboard.html")

RAW_TARGETS = ["spend", "impressions", "inline_link_clicks", "tracker_conversions", "tracker_revenue"]
KPI_KEYS = ["ctr", "cpm", "roas", "cvr"]
RAW_KEYS = ["spend", "impressions", "clicks", "conversions", "revenue"]


def clean_id(value) -> str:
    if pd.isna(value):
        return ""
    try:
        return str(int(float(value)))
    except Exception:
        return str(value)


def safe_div(num, den, mul=1.0):
    num = pd.Series(num, dtype="float64").fillna(0.0)
    den = pd.Series(den, dtype="float64").fillna(0.0)
    out = pd.Series(np.zeros(len(num)), index=num.index)
    mask = den != 0
    out.loc[mask] = num.loc[mask] / den.loc[mask] * mul
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def round_list(values, digits=4):
    return [round(float(v), digits) if pd.notna(v) else 0.0 for v in values]


def majority_label(values: pd.Series, default: str = "MIXED") -> str:
    counts = values.astype(str).value_counts()
    return str(counts.idxmax()) if len(counts) else default


def add_kpis(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df[f"{prefix}_ctr"] = safe_div(df[f"{prefix}_inline_link_clicks"], df[f"{prefix}_impressions"], 100.0)
    df[f"{prefix}_cpm"] = safe_div(df[f"{prefix}_spend"], df[f"{prefix}_impressions"], 1000.0)
    df[f"{prefix}_roas"] = safe_div(df[f"{prefix}_tracker_revenue"], df[f"{prefix}_spend"])
    df[f"{prefix}_cvr"] = safe_div(df[f"{prefix}_tracker_conversions"], df[f"{prefix}_inline_link_clicks"], 100.0)
    return df


def normalize_actual_kpis(df: pd.DataFrame) -> pd.DataFrame:
    if "actual_24h_ctr" not in df.columns:
        df["actual_24h_ctr"] = df["kpi_ctr"]
        df["actual_24h_cpm"] = df["kpi_cpm"]
        df["actual_24h_roas"] = df["kpi_roas"]
        df["actual_24h_cvr"] = df["kpi_cvr"]
    return df


def load_prediction_data() -> pd.DataFrame:
    df = pd.read_csv(PREDICTIONS, low_memory=False)
    for col in ["account_id", "campaign_id", "adset_id", "ad_id"]:
        df[col] = df[col].map(clean_id)
    df["local_date"] = pd.to_datetime(df["local_date"], errors="coerce").dt.normalize()
    df["date_label"] = df["local_date"].dt.strftime("%Y-%m-%d")
    df = normalize_actual_kpis(df)
    return df


def choose_accounts(df: pd.DataFrame, n: int = 3) -> list[str]:
    account_summary = (
        df.groupby("account_id", as_index=False)
        .agg(
            actual_revenue=("actual_24h_tracker_revenue", "sum"),
            actual_spend=("actual_24h_spend", "sum"),
            rows=("ad_id", "size"),
        )
        .sort_values(["actual_revenue", "actual_spend", "rows"], ascending=False)
    )
    return account_summary.head(n)["account_id"].astype(str).tolist()


def load_recent_daily_history(accounts: list[str]) -> pd.DataFrame:
    usecols = [
        "entity_type",
        "date",
        "account_id",
        "campaign_id",
        "adset_id",
        "ad_id",
        *RAW_TARGETS,
    ]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(RECENT_DAILY, usecols=lambda c: c in usecols, chunksize=200_000, low_memory=False):
        chunk = chunk[chunk["entity_type"].astype(str).str.lower().eq("ad")].copy()
        if chunk.empty:
            continue
        for col in ["account_id", "campaign_id", "adset_id", "ad_id"]:
            chunk[col] = chunk[col].map(clean_id)
        chunk = chunk[chunk["account_id"].isin(accounts)].copy()
        if chunk.empty:
            continue
        chunk["local_date"] = pd.to_datetime(chunk["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
        for col in RAW_TARGETS:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0.0).astype("float32")
        parts.append(chunk[["local_date", "account_id", "campaign_id", "adset_id", "ad_id", *RAW_TARGETS]])
    if not parts:
        raise RuntimeError("No recent daily history rows found for selected accounts.")
    hist = pd.concat(parts, ignore_index=True)
    hist = hist.groupby(["local_date", "account_id", "campaign_id", "adset_id", "ad_id"], as_index=False)[RAW_TARGETS].sum()
    hist["kpi_ctr"] = safe_div(hist["inline_link_clicks"], hist["impressions"], 100.0)
    hist["kpi_cpm"] = safe_div(hist["spend"], hist["impressions"], 1000.0)
    hist["kpi_roas"] = safe_div(hist["tracker_revenue"], hist["spend"])
    hist["kpi_cvr"] = safe_div(hist["tracker_conversions"], hist["inline_link_clicks"], 100.0)
    hist["date_label"] = hist["local_date"].dt.strftime("%Y-%m-%d")
    return hist


def aggregate_rows(rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = rows.groupby([*keys, "local_date"], as_index=False).agg(
        actual_24h_spend=("actual_24h_spend", "sum"),
        actual_24h_impressions=("actual_24h_impressions", "sum"),
        actual_24h_inline_link_clicks=("actual_24h_inline_link_clicks", "sum"),
        actual_24h_tracker_conversions=("actual_24h_tracker_conversions", "sum"),
        actual_24h_tracker_revenue=("actual_24h_tracker_revenue", "sum"),
        pred_24h_spend=("pred_24h_spend", "sum"),
        pred_24h_impressions=("pred_24h_impressions", "sum"),
        pred_24h_inline_link_clicks=("pred_24h_inline_link_clicks", "sum"),
        pred_24h_tracker_conversions=("pred_24h_tracker_conversions", "sum"),
        pred_24h_tracker_revenue=("pred_24h_tracker_revenue", "sum"),
        pred_final_24h_spend=("pred_final_24h_spend", "sum"),
        pred_final_24h_impressions=("pred_final_24h_impressions", "sum"),
        pred_final_24h_inline_link_clicks=("pred_final_24h_inline_link_clicks", "sum"),
        pred_final_24h_tracker_conversions=("pred_final_24h_tracker_conversions", "sum"),
        pred_final_24h_tracker_revenue=("pred_final_24h_tracker_revenue", "sum"),
        pred_final_p10_24h_spend=("pred_final_p10_24h_spend", "sum"),
        pred_final_p90_24h_spend=("pred_final_p90_24h_spend", "sum"),
        pred_final_p10_24h_impressions=("pred_final_p10_24h_impressions", "sum"),
        pred_final_p90_24h_impressions=("pred_final_p90_24h_impressions", "sum"),
        pred_final_p10_24h_inline_link_clicks=("pred_final_p10_24h_inline_link_clicks", "sum"),
        pred_final_p90_24h_inline_link_clicks=("pred_final_p90_24h_inline_link_clicks", "sum"),
        pred_final_p10_24h_tracker_conversions=("pred_final_p10_24h_tracker_conversions", "sum"),
        pred_final_p90_24h_tracker_conversions=("pred_final_p90_24h_tracker_conversions", "sum"),
        pred_final_p10_24h_tracker_revenue=("pred_final_p10_24h_tracker_revenue", "sum"),
        pred_final_p90_24h_tracker_revenue=("pred_final_p90_24h_tracker_revenue", "sum"),
        forecast_confidence=("forecast_confidence", majority_label),
        production_guardrail_flag=("production_guardrail_flag", majority_label),
        forecast_usage_recommendation=("forecast_usage_recommendation", majority_label),
        prediction_segment=("prediction_segment", majority_label),
        kpi_reliability_flag=("kpi_reliability_flag", majority_label),
    )
    grouped = add_kpis(grouped, "actual_24h")
    grouped = add_kpis(grouped, "pred_24h")
    grouped = add_kpis(grouped, "pred_final_24h")
    grouped["pred_final_p10_24h_roas"] = safe_div(grouped["pred_final_p10_24h_tracker_revenue"], grouped["pred_final_p90_24h_spend"])
    grouped["pred_final_p90_24h_roas"] = safe_div(grouped["pred_final_p90_24h_tracker_revenue"], grouped["pred_final_p10_24h_spend"])
    grouped["pred_final_p10_24h_ctr"] = safe_div(grouped["pred_final_p10_24h_inline_link_clicks"], grouped["pred_final_p90_24h_impressions"], 100.0)
    grouped["pred_final_p90_24h_ctr"] = safe_div(grouped["pred_final_p90_24h_inline_link_clicks"], grouped["pred_final_p10_24h_impressions"], 100.0)
    grouped["pred_final_p10_24h_cvr"] = safe_div(grouped["pred_final_p10_24h_tracker_conversions"], grouped["pred_final_p90_24h_inline_link_clicks"], 100.0)
    grouped["pred_final_p90_24h_cvr"] = safe_div(grouped["pred_final_p90_24h_tracker_conversions"], grouped["pred_final_p10_24h_inline_link_clicks"], 100.0)
    grouped["pred_final_p10_24h_cpm"] = safe_div(grouped["pred_final_p10_24h_spend"], grouped["pred_final_p90_24h_impressions"], 1000.0)
    grouped["pred_final_p90_24h_cpm"] = safe_div(grouped["pred_final_p90_24h_spend"], grouped["pred_final_p10_24h_impressions"], 1000.0)
    grouped["kpi_ctr"] = grouped["actual_24h_ctr"]
    grouped["kpi_cpm"] = grouped["actual_24h_cpm"]
    grouped["kpi_roas"] = grouped["actual_24h_roas"]
    grouped["kpi_cvr"] = grouped["actual_24h_cvr"]
    grouped["date_label"] = grouped["local_date"].dt.strftime("%Y-%m-%d")
    return grouped


def history_rows(history_df: pd.DataFrame, d1_date: pd.Timestamp) -> list[dict[str, object]]:
    labels = ["D-6", "D-5", "D-4", "D-3", "D-2", "D-1", "D0"]
    days = [d1_date - pd.Timedelta(days=offset) for offset in range(7, 0, -1)]
    rows = []
    for label, day in zip(labels, days):
        match = history_df[history_df["local_date"].eq(day)]
        if match.empty:
            rec = {
                "date_label": day.strftime("%Y-%m-%d"),
                "spend": 0.0,
                "impressions": 0.0,
                "inline_link_clicks": 0.0,
                "tracker_conversions": 0.0,
                "tracker_revenue": 0.0,
                "kpi_ctr": 0.0,
                "kpi_cpm": 0.0,
                "kpi_roas": 0.0,
                "kpi_cvr": 0.0,
            }
        else:
            rec = match.iloc[0]
        rows.append(
            {
                "label": label,
                "date": str(rec["date_label"]),
                "spend": float(rec["spend"]),
                "impressions": float(rec["impressions"]),
                "clicks": float(rec["inline_link_clicks"]),
                "conversions": float(rec["tracker_conversions"]),
                "revenue": float(rec["tracker_revenue"]),
                "ctr": float(rec["kpi_ctr"]),
                "cpm": float(rec["kpi_cpm"]),
                "roas": float(rec["kpi_roas"]),
                "cvr": float(rec["kpi_cvr"]),
            }
        )
    return rows


def d1_payload(row: pd.Series) -> dict[str, object]:
    return {
        "date": str(row["date_label"]),
        "actual": {
            "spend": float(row["actual_24h_spend"]),
            "impressions": float(row["actual_24h_impressions"]),
            "clicks": float(row["actual_24h_inline_link_clicks"]),
            "conversions": float(row["actual_24h_tracker_conversions"]),
            "revenue": float(row["actual_24h_tracker_revenue"]),
            "ctr": float(row["actual_24h_ctr"]),
            "cpm": float(row["actual_24h_cpm"]),
            "roas": float(row["actual_24h_roas"]),
            "cvr": float(row["actual_24h_cvr"]),
        },
        "base": {
            "spend": float(row["pred_24h_spend"]),
            "impressions": float(row["pred_24h_impressions"]),
            "clicks": float(row["pred_24h_inline_link_clicks"]),
            "conversions": float(row["pred_24h_tracker_conversions"]),
            "revenue": float(row["pred_24h_tracker_revenue"]),
            "ctr": float(row["pred_24h_ctr"]),
            "cpm": float(row["pred_24h_cpm"]),
            "roas": float(row["pred_24h_roas"]),
            "cvr": float(row["pred_24h_cvr"]),
        },
        "final": {
            "spend": float(row["pred_final_24h_spend"]),
            "impressions": float(row["pred_final_24h_impressions"]),
            "clicks": float(row["pred_final_24h_inline_link_clicks"]),
            "conversions": float(row["pred_final_24h_tracker_conversions"]),
            "revenue": float(row["pred_final_24h_tracker_revenue"]),
            "ctr": float(row["pred_final_24h_ctr"]),
            "cpm": float(row["pred_final_24h_cpm"]),
            "roas": float(row["pred_final_24h_roas"]),
            "cvr": float(row["pred_final_24h_cvr"]),
        },
        "range": {
            "spend": [float(row["pred_final_p10_24h_spend"]), float(row["pred_final_p90_24h_spend"])],
            "impressions": [float(row["pred_final_p10_24h_impressions"]), float(row["pred_final_p90_24h_impressions"])],
            "clicks": [float(row["pred_final_p10_24h_inline_link_clicks"]), float(row["pred_final_p90_24h_inline_link_clicks"])],
            "conversions": [float(row["pred_final_p10_24h_tracker_conversions"]), float(row["pred_final_p90_24h_tracker_conversions"])],
            "revenue": [float(row["pred_final_p10_24h_tracker_revenue"]), float(row["pred_final_p90_24h_tracker_revenue"])],
            "ctr": [float(row["pred_final_p10_24h_ctr"]), float(row["pred_final_p90_24h_ctr"])],
            "cpm": [float(row["pred_final_p10_24h_cpm"]), float(row["pred_final_p90_24h_cpm"])],
            "roas": [float(row["pred_final_p10_24h_roas"]), float(row["pred_final_p90_24h_roas"])],
            "cvr": [float(row["pred_final_p10_24h_cvr"]), float(row["pred_final_p90_24h_cvr"])],
        },
        "meta": {
            "confidence": str(row["forecast_confidence"]),
            "guardrail": str(row["production_guardrail_flag"]),
            "usage": str(row["forecast_usage_recommendation"]),
            "segment": str(row["prediction_segment"]),
            "reliability": str(row["kpi_reliability_flag"]),
        },
    }


def series_from_rows(rows: pd.DataFrame, history: pd.DataFrame) -> dict[str, object]:
    rows = rows.sort_values("local_date")
    latest = rows.iloc[-1]
    labels = rows["date_label"].astype(str).tolist()
    history_by_date = {
        str(row["date_label"]): history_rows(history, pd.Timestamp(row["local_date"]))
        for _, row in rows.iterrows()
    }
    d1_by_date = {str(row["date_label"]): d1_payload(row) for _, row in rows.iterrows()}
    return {
        "labels": labels,
        "latest_date": str(latest["date_label"]),
        "history": history_rows(history, pd.Timestamp(latest["local_date"])),
        "history_by_date": history_by_date,
        "latest_d1": d1_payload(latest),
        "d1_by_date": d1_by_date,
        "confidence": str(latest["forecast_confidence"]),
        "guardrail": str(latest["production_guardrail_flag"]),
        "usage": str(latest["forecast_usage_recommendation"]),
        "segment": str(latest["prediction_segment"]),
        "reliability": str(latest["kpi_reliability_flag"]),
        "kpis": {
            "ctr": {"actual": round_list(rows["actual_24h_ctr"]), "base": round_list(rows["pred_24h_ctr"]), "final": round_list(rows["pred_final_24h_ctr"])},
            "cpm": {"actual": round_list(rows["actual_24h_cpm"]), "base": round_list(rows["pred_24h_cpm"]), "final": round_list(rows["pred_final_24h_cpm"])},
            "roas": {"actual": round_list(rows["actual_24h_roas"]), "base": round_list(rows["pred_24h_roas"]), "final": round_list(rows["pred_final_24h_roas"])},
            "cvr": {"actual": round_list(rows["actual_24h_cvr"]), "base": round_list(rows["pred_24h_cvr"]), "final": round_list(rows["pred_final_24h_cvr"])},
        },
    }


def metrics_payload() -> list[dict[str, object]]:
    metrics = pd.read_csv(METRICS)
    keep = ["spend", "impressions", "clicks", "conversions", "revenue", "roas", "profit", "ctr", "cvr", "cpm"]
    metrics = metrics[metrics["metric"].isin(keep)].copy()
    return [
        {
            "metric": str(row.metric),
            "r2": round(float(row.r2), 4),
            "wmape": round(float(row.wmape), 4),
            "bias": round(float(row.bias), 4),
        }
        for row in metrics.itertuples(index=False)
    ]


def build_payload() -> dict[str, object]:
    df = load_prediction_data()
    accounts = choose_accounts(df)
    df = df[df["account_id"].isin(accounts)].copy()
    history = load_recent_daily_history(accounts)
    payload = {"accounts": accounts, "metrics": metrics_payload(), "data": {}}
    for account in accounts:
        acc_rows = df[df["account_id"].eq(account)].copy()
        acc_history = (
            history[history["account_id"].eq(account)]
            .groupby("local_date", as_index=False)[RAW_TARGETS]
            .sum()
        )
        acc_history["kpi_ctr"] = safe_div(acc_history["inline_link_clicks"], acc_history["impressions"], 100.0)
        acc_history["kpi_cpm"] = safe_div(acc_history["spend"], acc_history["impressions"], 1000.0)
        acc_history["kpi_roas"] = safe_div(acc_history["tracker_revenue"], acc_history["spend"])
        acc_history["kpi_cvr"] = safe_div(acc_history["tracker_conversions"], acc_history["inline_link_clicks"], 100.0)
        acc_history["date_label"] = acc_history["local_date"].dt.strftime("%Y-%m-%d")
        account_series = series_from_rows(aggregate_rows(acc_rows, ["account_id"]), acc_history)

        ad_summary = (
            acc_rows.groupby("ad_id", as_index=False)
            .agg(days=("local_date", "nunique"), actual_revenue=("actual_24h_tracker_revenue", "sum"), actual_spend=("actual_24h_spend", "sum"))
            .sort_values(["actual_revenue", "actual_spend", "days"], ascending=False)
            .head(30)
        )
        ad_list = [{"id": "ALL", "label": "All Ads (account aggregate)"}]
        series = {"ALL": account_series}
        for rec in ad_summary.itertuples(index=False):
            ad_id = str(rec.ad_id)
            ad_rows = acc_rows[acc_rows["ad_id"].eq(ad_id)].copy()
            ad_history = history[(history["account_id"].eq(account)) & (history["ad_id"].eq(ad_id))].copy()
            ad_list.append({"id": ad_id, "label": f"Ad {ad_id} ({int(rec.days)} days)"})
            series[ad_id] = series_from_rows(aggregate_rows(ad_rows, ["account_id", "campaign_id", "adset_id", "ad_id"]), ad_history)
        payload["data"][account] = {"ad_list": ad_list, "series": series}
    return payload


def main() -> None:
    raw = build_payload()
    html = f"""<!doctype html><html lang="en"><head><meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Adunbox 24H Recent Holdout Dashboard</title><script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}}.hdr{{background:linear-gradient(135deg,#111827,#172554 50%,#0f172a);padding:22px 30px;border-bottom:1px solid #263348}}.hdr h1{{font-size:1.45rem;color:#a5b4fc}}.hdr p{{color:#94a3b8;font-size:.84rem;margin-top:5px;max-width:1180px;line-height:1.45}}.wrap{{max-width:1500px;margin:0 auto;padding:18px 24px}}.tabs{{display:flex;gap:4px;margin:16px 0 0;flex-wrap:wrap}}.tab{{padding:15px 24px;border-radius:10px 10px 0 0;font-size:1rem;font-weight:800;cursor:pointer;color:#7f95bd;border:1px solid transparent;border-bottom:none;letter-spacing:.01em}}.tab.active{{background:#1e293b;color:#dbeafe;border-color:#334155;border-bottom-color:#1e293b}}.panel{{display:none;background:#1e293b;border:1px solid #334155;border-radius:0 12px 12px 12px;padding:18px 20px}}.panel.active{{display:block}}.bar{{display:flex;align-items:center;gap:14px;flex-wrap:wrap;background:#0f172a;border:1px solid #334155;border-radius:12px;padding:14px 18px;margin-bottom:14px}}label,.muted{{font-size:.82rem;color:#b7c4df;font-weight:800}}select{{background:#1e293b;border:1px solid #334155;color:#f8fafc;padding:10px 16px;border-radius:9px;min-width:300px;max-width:620px;font-size:1rem}}.btn{{background:#1e293b;border:1px solid #334155;color:#94a3b8;padding:7px 16px;border-radius:8px;cursor:pointer;font-size:.8rem;font-weight:700}}.btn.active{{background:#312e81;border-color:#6366f1;color:#c7d2fe}}.stats,.cmp{{display:grid;grid-template-columns:repeat(5,minmax(130px,1fr));gap:9px;margin:12px 0}}.stat,.metric{{background:#0f172a;border:1px solid #334155;border-radius:9px;padding:10px 13px}}.lbl{{font-size:.67rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em}}.val{{font-size:1.08rem;font-weight:800;margin-top:3px}}.good{{color:#86efac}}.bad{{color:#fecaca}}.alert{{background:#172554;border:1px solid #1d4ed8;color:#bfdbfe;border-radius:9px;padding:10px 13px;font-size:.82rem;line-height:1.45;margin-bottom:13px}}.legend{{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0 8px}}.li{{display:flex;align-items:center;gap:6px;font-size:.76rem;color:#94a3b8}}.dot{{width:9px;height:9px;border-radius:50%}}.gtitle{{font-size:.86rem;font-weight:900;color:#8b95ff;text-transform:uppercase;letter-spacing:.12em;margin:18px 0 10px;display:flex;align-items:center;gap:9px}}.gtitle::before{{content:'';width:4px;height:15px;border-radius:3px;background:#6366f1}}.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}}.card{{background:#0f172a;border:1px solid #334155;border-radius:11px;padding:13px 15px}}.card h3{{font-size:.78rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px}}.card canvas{{max-height:230px}}.tbl-wrap{{overflow-x:auto;border:1px solid #1e293b;border-radius:10px;margin-top:12px}}table{{width:100%;border-collapse:collapse;font-size:.79rem}}th{{background:#111827;color:#94a3b8;text-align:left;padding:9px 10px;white-space:nowrap}}td{{padding:8px 10px;border-top:1px solid #1e293b;white-space:nowrap}}tr:hover td{{background:#1e293b66}}@media(max-width:960px){{.grid{{grid-template-columns:1fr}}.stats,.cmp{{grid-template-columns:repeat(2,1fr)}}select{{min-width:180px}}}}
</style></head><body><div class="hdr"><h1>Adunbox - 24H Recent Out-of-Time Holdout</h1><p>Frozen optimized 24h model tested on H-drive recent daily data from May 13 onward. Charts compare actual vs base prediction vs optimized final prediction. Tables show D-6 to D0 context and D1 actual/predicted/range.</p></div><div class="wrap"><div class="gtitle">Recent Holdout Metrics</div><div class="cmp" id="cmp"></div><div class="tabs" id="tabs"></div><div id="panels"></div></div>
<script>
const RAW={json.dumps(raw, separators=(",", ":"))};const COL={{actual:'#6366f1',base:'#a5b4fc',final:'#22d3ee'}};const CH={{}},STATE={{}};
function fmt(v,cur=false,dec=2){{if(v===null||v===undefined||Number.isNaN(Number(v)))return'N/A';const s=Number(v).toLocaleString('en-US',{{maximumFractionDigits:dec}});return cur?'$'+s:s}}function ds(label,data,color,dash=[]){{return{{label,data,borderColor:color,backgroundColor:color+'22',borderWidth:2,borderDash:dash,pointRadius:2.5,pointHoverRadius:5,tension:.25,spanGaps:true}}}}function slice(labels,n){{if(!n)return[0,labels.length];const take=Math.min(labels.length,n);return[labels.length-take,labels.length]}}function draw(id,labels,sets){{if(CH[id])CH[id].destroy();CH[id]=new Chart(document.getElementById(id),{{type:'line',data:{{labels,datasets:sets}},options:{{responsive:true,maintainAspectRatio:true,interaction:{{mode:'index',intersect:false}},plugins:{{legend:{{display:false}},tooltip:{{backgroundColor:'#1e293b',borderColor:'#334155',borderWidth:1,titleColor:'#cbd5e1',bodyColor:'#e2e8f0'}}}},scales:{{x:{{grid:{{color:'#1e293b'}},ticks:{{color:'#64748b',font:{{size:9}},maxRotation:45,minRotation:30}}}},y:{{grid:{{color:'#1e293b'}},ticks:{{color:'#64748b',font:{{size:9}}}}}}}}}}}})}}
function active(acc){{return RAW.data[acc].series[STATE[acc].ad]}}function selectedDate(acc){{const s=active(acc);return STATE[acc].date||s.latest_date}}function render(acc){{const s=active(acc),st=STATE[acc],[a,b]=slice(s.labels,st.days),labels=s.labels.slice(a,b);['ctr','cpm','roas','cvr'].forEach(k=>{{const kp=s.kpis[k],sets=[];if(st.mode==='both'||st.mode==='actual')sets.push(ds('Actual',kp.actual.slice(a,b),COL.actual));if(st.mode==='both'||st.mode==='base')sets.push(ds('Base Predicted',kp.base.slice(a,b),COL.base,[5,4]));if(st.mode==='both'||st.mode==='final')sets.push(ds('Optimized Final',kp.final.slice(a,b),COL.final,[2,3]));draw(`ch-${{k}}-${{acc}}`,labels,sets)}});stats(acc);tables(acc)}}function stats(acc){{const s=active(acc),dt=selectedDate(acc),d=s.d1_by_date[dt]||s.latest_d1;document.getElementById(`conf-${{acc}}`).textContent=d.meta.confidence;document.getElementById(`guard-${{acc}}`).textContent=d.meta.guardrail;document.getElementById(`seg-${{acc}}`).textContent=d.meta.segment;document.getElementById(`use-${{acc}}`).textContent=d.meta.usage;document.getElementById(`note-${{acc}}`).innerHTML=`<b>Selected D1:</b> ${{dt}}. <b>Meaning:</b> model used previous 7 daily rows as context, predicted this D1, and actual D1 exists in recent file for gap check.`}}
function tables(acc){{const s=active(acc),dt=selectedDate(acc),h=s.history_by_date[dt]||s.history,d=s.d1_by_date[dt]||s.latest_d1;document.getElementById(`hhead-${{acc}}`).innerHTML='<tr><th>Day</th><th>Date</th><th>Value Type</th><th>Spend</th><th>Impressions</th><th>Clicks</th><th>Conversions</th><th>Revenue</th><th>CTR</th><th>CPM</th><th>ROAS</th><th>CVR</th></tr>';let histRows=h.map(r=>`<tr><td><b>${{r.label}}</b></td><td>${{r.date}}</td><td>Actual</td><td>${{fmt(r.spend,true)}}</td><td>${{fmt(r.impressions)}}</td><td>${{fmt(r.clicks)}}</td><td>${{fmt(r.conversions)}}</td><td>${{fmt(r.revenue,true)}}</td><td>${{fmt(r.ctr)}}</td><td>${{fmt(r.cpm,true)}}</td><td>${{fmt(r.roas)}}</td><td>${{fmt(r.cvr)}}</td></tr>`).join('');let d1Rows=['actual','base','final'].map(t=>`<tr><td><b>D1</b></td><td>${{d.date}}</td><td>${{t==='actual'?'Actual':t==='base'?'Base Predicted':'Optimized Final'}}</td><td>${{fmt(d[t].spend,true)}}</td><td>${{fmt(d[t].impressions)}}</td><td>${{fmt(d[t].clicks)}}</td><td>${{fmt(d[t].conversions)}}</td><td>${{fmt(d[t].revenue,true)}}</td><td>${{fmt(d[t].ctr)}}</td><td>${{fmt(d[t].cpm,true)}}</td><td>${{fmt(d[t].roas)}}</td><td>${{fmt(d[t].cvr)}}</td></tr>`).join('');document.getElementById(`hbody-${{acc}}`).innerHTML=histRows+d1Rows;document.getElementById(`rangebody-${{acc}}`).innerHTML=['spend','impressions','clicks','conversions','revenue','ctr','cpm','roas','cvr'].map(k=>`<tr><td><b>${{k.toUpperCase()}}</b></td><td>${{fmt(d.actual[k],['spend','revenue','cpm'].includes(k))}}</td><td>${{fmt(d.final[k],['spend','revenue','cpm'].includes(k))}}</td><td>${{fmt(d.range[k][0],['spend','revenue','cpm'].includes(k))}} - ${{fmt(d.range[k][1],['spend','revenue','cpm'].includes(k))}}</td><td>${{fmt(d.final[k]-d.actual[k],false,3)}}</td></tr>`).join('')}}
function switchAcc(acc,el){{document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));el.classList.add('active');document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));document.getElementById('p-'+acc).classList.add('active')}}function refreshDates(acc){{const s=active(acc),sel=document.getElementById(`date-${{acc}}`);sel.innerHTML=s.labels.map(d=>`<option value="${{d}}">${{d}}</option>`).join('');STATE[acc].date=s.latest_date;sel.value=s.latest_date}}function onAd(acc){{STATE[acc].ad=document.getElementById(`ad-${{acc}}`).value;refreshDates(acc);render(acc)}}function onDate(acc){{STATE[acc].date=document.getElementById(`date-${{acc}}`).value;render(acc)}}function onMode(acc){{STATE[acc].mode=document.getElementById(`mode-${{acc}}`).value;render(acc)}}function onDays(acc,days,btn){{STATE[acc].days=days;document.getElementById(`days-${{acc}}`).querySelectorAll('.btn').forEach(x=>x.classList.remove('active'));btn.classList.add('active');render(acc)}}
function panel(acc){{STATE[acc]={{ad:'ALL',mode:'both',days:14,date:null}};const opts=RAW.data[acc].ad_list.map(a=>`<option value="${{a.id}}">${{a.label}}</option>`).join('');document.getElementById('p-'+acc).innerHTML=`<div class="bar"><label>Select Ad</label><select id="ad-${{acc}}" onchange="onAd('${{acc}}')">${{opts}}</select><label>D1 Date</label><select id="date-${{acc}}" onchange="onDate('${{acc}}')"></select><label>View</label><select id="mode-${{acc}}" onchange="onMode('${{acc}}')"><option value="both">Actual + Base + Optimized</option><option value="actual">Actual Only</option><option value="base">Base Predicted Only</option><option value="final">Optimized Final Only</option></select><span class="muted">Chart Window</span><span id="days-${{acc}}"><button class="btn active" onclick="onDays('${{acc}}',14,this)">Last 14</button> <button class="btn" onclick="onDays('${{acc}}',0,this)">All</button></span></div><div class="stats"><div class="stat"><div class="lbl">Confidence</div><div class="val" id="conf-${{acc}}">-</div></div><div class="stat"><div class="lbl">Guardrail</div><div class="val" id="guard-${{acc}}">-</div></div><div class="stat"><div class="lbl">Segment</div><div class="val" id="seg-${{acc}}">-</div></div><div class="stat"><div class="lbl">Usage</div><div class="val" id="use-${{acc}}">-</div></div></div><div class="alert" id="note-${{acc}}"></div><div class="legend"><div class="li"><span class="dot" style="background:${{COL.actual}}"></span>Actual</div><div class="li"><span class="dot" style="background:${{COL.base}}"></span>Base Predicted</div><div class="li"><span class="dot" style="background:${{COL.final}}"></span>Optimized Final</div></div><div class="gtitle">Recent Holdout KPI Charts - CTR, CPM, ROAS, CVR</div><div class="grid"><div class="card"><h3>CTR (%)</h3><canvas id="ch-ctr-${{acc}}"></canvas></div><div class="card"><h3>CPM ($)</h3><canvas id="ch-cpm-${{acc}}"></canvas></div><div class="card"><h3>ROAS</h3><canvas id="ch-roas-${{acc}}"></canvas></div><div class="card"><h3>CVR (%)</h3><canvas id="ch-cvr-${{acc}}"></canvas></div></div><div class="gtitle">D-6 to D0 Actual History + D1 Actual/Predicted</div><div class="tbl-wrap"><table><thead id="hhead-${{acc}}"></thead><tbody id="hbody-${{acc}}"></tbody></table></div><div class="gtitle">D1 Final Prediction Range And Gap</div><div class="tbl-wrap"><table><thead><tr><th>Metric</th><th>Actual</th><th>Optimized Final</th><th>Final Range p10-p90</th><th>Gap</th></tr></thead><tbody id="rangebody-${{acc}}"></tbody></table></div>`;refreshDates(acc);render(acc)}}
function init(){{document.getElementById('cmp').innerHTML=RAW.metrics.map(r=>`<div class="metric"><div class="lbl">${{r.metric}}</div><div class="val ${{r.r2>=0?'good':'bad'}}">R2 ${{fmt(r.r2,false,3)}}</div><div class="lbl">WMAPE ${{fmt(r.wmape,false,3)}} | Bias ${{fmt(r.bias,false,3)}}</div></div>`).join('');const tabs=document.getElementById('tabs'),panels=document.getElementById('panels');RAW.accounts.forEach((acc,i)=>{{tabs.insertAdjacentHTML('beforeend',`<div class="tab ${{i===0?'active':''}}" onclick="switchAcc('${{acc}}',this)">Account ${{acc}}</div>`);panels.insertAdjacentHTML('beforeend',`<div class="panel ${{i===0?'active':''}}" id="p-${{acc}}"></div>`);panel(acc)}})}}init();
</script></body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
