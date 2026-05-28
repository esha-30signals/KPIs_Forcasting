from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(os.getenv("ADUNBOX_PROJECT_DIR", Path(__file__).resolve().parents[1]))
PRIMARY = Path(os.getenv("ADUNBOX_KPI_BACKTEST_CSV", BASE_DIR / "outputs" / "adunbox_24h_three_account_ctr_cpm_roas_cvr_predicted_review.csv"))
SPIKE = Path(os.getenv("ADUNBOX_SPIKE_REVIEW_CSV", BASE_DIR / "outputs" / "adunbox_daily_24h_spike_aware_prediction_review.csv"))
OUT = Path(os.getenv("ADUNBOX_DASHBOARD_OUT", BASE_DIR / "dashboards" / "adunbox_24h_historical_actual_vs_predicted_kpi_dashboard.html"))
ACCOUNTS = ["7730708", "7738188", "36061656"]
HIST_LABELS = ["D-6", "D-5", "D-4", "D-3", "D-2", "D-1", "D0"]


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


def add_rolling_impression_calibration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["account_id", "ad_id", "date"]).copy()
    factors = []
    for _, group in df.groupby(["account_id", "ad_id"], sort=False):
        prior = []
        for _, row in group.iterrows():
            factor = float(np.median(prior)) if prior else 1.0
            factors.append(min(max(factor, 0.25), 4.0))
            pred = float(row.get("pred_impressions", 0) or 0)
            actual = float(row.get("actual_impressions", 0) or 0)
            if pred > 0 and actual > 0:
                prior.append(actual / pred)
    df["calibration__impressions_factor"] = factors
    df["d1_pred_calibrated__impressions"] = df["pred_impressions"] * df["calibration__impressions_factor"]
    return df


def load_data() -> pd.DataFrame:
    df = pd.read_csv(PRIMARY, low_memory=False)
    for col in ["account_id", "campaign_id", "adset_id", "ad_id"]:
        df[col] = df[col].map(clean_id)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["account_id"].isin(ACCOUNTS)].copy()

    spike_cols = [
        "identity__account_id",
        "identity__campaign_id",
        "identity__adset_id",
        "identity__ad_id",
        "prediction__d1_date",
        "spike_risk_score",
        "spike_risk_flag",
        "review_note",
        "d1_pred_calibrated__spend",
        "d1_pred_calibrated__revenue",
        "d1_pred_calibrated__clicks",
        "d1_pred_calibrated__conversions",
        "d1_pred_calibrated__roas",
    ]
    spike = pd.read_csv(SPIKE, usecols=lambda c: c in spike_cols, low_memory=False)
    spike = spike.rename(
        columns={
            "identity__account_id": "account_id",
            "identity__campaign_id": "campaign_id",
            "identity__adset_id": "adset_id",
            "identity__ad_id": "ad_id",
            "prediction__d1_date": "date",
        }
    )
    for col in ["account_id", "campaign_id", "adset_id", "ad_id"]:
        spike[col] = spike[col].map(clean_id)
    spike["date"] = pd.to_datetime(spike["date"], errors="coerce")

    df = df.merge(spike, on=["account_id", "campaign_id", "adset_id", "ad_id", "date"], how="left")
    df["spike_risk_score"] = pd.to_numeric(df["spike_risk_score"], errors="coerce").fillna(0).astype(int)
    df["spike_risk_flag"] = df["spike_risk_flag"].fillna("LOW")
    df["review_note"] = df["review_note"].fillna("No spike-aware review row matched this date.")
    for base in ["spend", "revenue", "clicks", "conversions"]:
        cal = f"d1_pred_calibrated__{base}"
        pred = f"pred_{base}" if base != "conversions" else "pred_conversions"
        df[cal] = pd.to_numeric(df[cal], errors="coerce").fillna(df[pred])

    df = add_rolling_impression_calibration(df)
    df["cal_ctr_pct"] = safe_div(df["d1_pred_calibrated__clicks"], df["d1_pred_calibrated__impressions"], 100.0)
    df["cal_cpm"] = safe_div(df["d1_pred_calibrated__spend"], df["d1_pred_calibrated__impressions"], 1000.0)
    df["cal_roas"] = safe_div(df["d1_pred_calibrated__revenue"], df["d1_pred_calibrated__spend"])
    df["cal_cvr_pct"] = safe_div(df["d1_pred_calibrated__conversions"], df["d1_pred_calibrated__clicks"], 100.0)
    df["date_key"] = df["date"].dt.strftime("%Y-%m-%d")
    return df.sort_values(["account_id", "ad_id", "date"])


def build_daily_lookup(rows: pd.DataFrame, keys: list[str]):
    grouped = rows.groupby(keys + ["date"], as_index=False).agg(
        actual_spend=("actual_spend", "sum"),
        actual_impressions=("actual_impressions", "sum"),
        actual_clicks=("actual_clicks", "sum"),
        actual_conversions=("actual_conversions", "sum"),
        actual_revenue=("actual_revenue", "sum"),
    )
    grouped["actual_ctr_pct"] = safe_div(grouped["actual_clicks"], grouped["actual_impressions"], 100.0)
    grouped["actual_cpm"] = safe_div(grouped["actual_spend"], grouped["actual_impressions"], 1000.0)
    grouped["actual_roas"] = safe_div(grouped["actual_revenue"], grouped["actual_spend"])
    grouped["actual_cvr_pct"] = safe_div(grouped["actual_conversions"], grouped["actual_clicks"], 100.0)
    return grouped


def history_window(row: pd.Series, lookup: pd.DataFrame, keys: list[str]) -> dict:
    d1 = pd.to_datetime(row["date"])
    start = d1 - pd.Timedelta(days=7)
    days = [start + pd.Timedelta(days=i) for i in range(7)]
    subset = lookup.copy()
    for key in keys:
        subset = subset[subset[key].astype(str).eq(str(row[key]))]
    values = []
    for label, day in zip(HIST_LABELS, days):
        match = subset[subset["date"].eq(day)]
        rec = match.iloc[0] if len(match) else {}
        values.append(
            {
                "label": label,
                "date": day.strftime("%Y-%m-%d"),
                "spend": float(rec.get("actual_spend", 0) or 0),
                "impressions": float(rec.get("actual_impressions", 0) or 0),
                "clicks": float(rec.get("actual_clicks", 0) or 0),
                "conversions": float(rec.get("actual_conversions", 0) or 0),
                "revenue": float(rec.get("actual_revenue", 0) or 0),
                "ctr": float(rec.get("actual_ctr_pct", 0) or 0),
                "cpm": float(rec.get("actual_cpm", 0) or 0),
                "roas": float(rec.get("actual_roas", 0) or 0),
                "cvr": float(rec.get("actual_cvr_pct", 0) or 0),
            }
        )
    return {"d1_date": d1.strftime("%Y-%m-%d"), "rows": values}


def series_from_rows(rows: pd.DataFrame, history_lookup: pd.DataFrame, keys: list[str]) -> dict:
    rows = rows.sort_values("date").copy()
    if rows.empty:
        return {}
    latest = rows.iloc[-1].copy()
    max_risk_score = int(rows["spike_risk_score"].max())
    latest_risk_score = int(latest.get("spike_risk_score", 0) or 0)
    latest_risk_flag = "HIGH" if latest_risk_score >= 70 else "MEDIUM" if latest_risk_score >= 40 else "LOW"
    max_risk_flag = "HIGH" if max_risk_score >= 70 else "MEDIUM" if max_risk_score >= 40 else "LOW"
    risk_rows = [
        {
            "date": str(row.get("date_key", "")),
            "score": int(row.get("spike_risk_score", 0) or 0),
            "flag": "HIGH"
            if int(row.get("spike_risk_score", 0) or 0) >= 70
            else "MEDIUM"
            if int(row.get("spike_risk_score", 0) or 0) >= 40
            else "LOW",
            "note": str(row.get("review_note", "")),
        }
        for _, row in rows.tail(14).iterrows()
    ]
    return {
        "labels": rows["date_key"].astype(str).tolist(),
        "risk_score": latest_risk_score,
        "risk_flag": latest_risk_flag,
        "max_risk_score": max_risk_score,
        "max_risk_flag": max_risk_flag,
        "review_note": str(latest.get("review_note", "")),
        "risk_rows": risk_rows,
        "history": history_window(latest, history_lookup, keys),
        "kpis": {
            "ctr": {"actual": round_list(rows["actual_ctr_pct"]), "pred": round_list(rows["pred_ctr_pct"]), "cal": round_list(rows["cal_ctr_pct"])},
            "cpm": {"actual": round_list(rows["actual_cpm"]), "pred": round_list(rows["pred_cpm"]), "cal": round_list(rows["cal_cpm"])},
            "roas": {"actual": round_list(rows["actual_roas"]), "pred": round_list(rows["pred_roas"]), "cal": round_list(rows["cal_roas"])},
            "cvr": {"actual": round_list(rows["actual_cvr_pct"]), "pred": round_list(rows["pred_cvr_pct"]), "cal": round_list(rows["cal_cvr_pct"])},
        },
        "raw": {
            "spend": {"actual": round_list(rows["actual_spend"]), "pred": round_list(rows["pred_spend"]), "cal": round_list(rows["d1_pred_calibrated__spend"])},
            "impressions": {"actual": round_list(rows["actual_impressions"]), "pred": round_list(rows["pred_impressions"]), "cal": round_list(rows["d1_pred_calibrated__impressions"])},
            "clicks": {"actual": round_list(rows["actual_clicks"]), "pred": round_list(rows["pred_clicks"]), "cal": round_list(rows["d1_pred_calibrated__clicks"])},
            "conversions": {"actual": round_list(rows["actual_conversions"]), "pred": round_list(rows["pred_conversions"]), "cal": round_list(rows["d1_pred_calibrated__conversions"])},
            "revenue": {"actual": round_list(rows["actual_revenue"]), "pred": round_list(rows["pred_revenue"]), "cal": round_list(rows["d1_pred_calibrated__revenue"])},
        },
        "latest_d1": {
            "date": str(latest.get("date_key", "")),
            "actual": {
                "spend": float(latest.get("actual_spend", 0) or 0),
                "impressions": float(latest.get("actual_impressions", 0) or 0),
                "clicks": float(latest.get("actual_clicks", 0) or 0),
                "conversions": float(latest.get("actual_conversions", 0) or 0),
                "revenue": float(latest.get("actual_revenue", 0) or 0),
                "ctr": float(latest.get("actual_ctr_pct", 0) or 0),
                "cpm": float(latest.get("actual_cpm", 0) or 0),
                "roas": float(latest.get("actual_roas", 0) or 0),
                "cvr": float(latest.get("actual_cvr_pct", 0) or 0),
            },
            "pred": {
                "spend": float(latest.get("pred_spend", 0) or 0),
                "impressions": float(latest.get("pred_impressions", 0) or 0),
                "clicks": float(latest.get("pred_clicks", 0) or 0),
                "conversions": float(latest.get("pred_conversions", 0) or 0),
                "revenue": float(latest.get("pred_revenue", 0) or 0),
                "ctr": float(latest.get("pred_ctr_pct", 0) or 0),
                "cpm": float(latest.get("pred_cpm", 0) or 0),
                "roas": float(latest.get("pred_roas", 0) or 0),
                "cvr": float(latest.get("pred_cvr_pct", 0) or 0),
            },
            "cal": {
                "spend": float(latest.get("d1_pred_calibrated__spend", 0) or 0),
                "impressions": float(latest.get("d1_pred_calibrated__impressions", 0) or 0),
                "clicks": float(latest.get("d1_pred_calibrated__clicks", 0) or 0),
                "conversions": float(latest.get("d1_pred_calibrated__conversions", 0) or 0),
                "revenue": float(latest.get("d1_pred_calibrated__revenue", 0) or 0),
                "ctr": float(latest.get("cal_ctr_pct", 0) or 0),
                "cpm": float(latest.get("cal_cpm", 0) or 0),
                "roas": float(latest.get("cal_roas", 0) or 0),
                "cvr": float(latest.get("cal_cvr_pct", 0) or 0),
            },
        },
    }


def aggregate_account(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["account_id", "date"], as_index=False).agg(
        actual_spend=("actual_spend", "sum"),
        pred_spend=("pred_spend", "sum"),
        d1_pred_calibrated__spend=("d1_pred_calibrated__spend", "sum"),
        actual_impressions=("actual_impressions", "sum"),
        pred_impressions=("pred_impressions", "sum"),
        d1_pred_calibrated__impressions=("d1_pred_calibrated__impressions", "sum"),
        actual_clicks=("actual_clicks", "sum"),
        pred_clicks=("pred_clicks", "sum"),
        d1_pred_calibrated__clicks=("d1_pred_calibrated__clicks", "sum"),
        actual_conversions=("actual_conversions", "sum"),
        pred_conversions=("pred_conversions", "sum"),
        d1_pred_calibrated__conversions=("d1_pred_calibrated__conversions", "sum"),
        actual_revenue=("actual_revenue", "sum"),
        pred_revenue=("pred_revenue", "sum"),
        d1_pred_calibrated__revenue=("d1_pred_calibrated__revenue", "sum"),
        spike_risk_score=("spike_risk_score", "max"),
        review_note=("review_note", "last"),
    )
    grouped["campaign_id"] = "ALL"
    grouped["adset_id"] = "ALL"
    grouped["ad_id"] = "ALL"
    grouped["date_key"] = grouped["date"].dt.strftime("%Y-%m-%d")
    grouped["actual_ctr_pct"] = safe_div(grouped["actual_clicks"], grouped["actual_impressions"], 100.0)
    grouped["pred_ctr_pct"] = safe_div(grouped["pred_clicks"], grouped["pred_impressions"], 100.0)
    grouped["cal_ctr_pct"] = safe_div(grouped["d1_pred_calibrated__clicks"], grouped["d1_pred_calibrated__impressions"], 100.0)
    grouped["actual_cpm"] = safe_div(grouped["actual_spend"], grouped["actual_impressions"], 1000.0)
    grouped["pred_cpm"] = safe_div(grouped["pred_spend"], grouped["pred_impressions"], 1000.0)
    grouped["cal_cpm"] = safe_div(grouped["d1_pred_calibrated__spend"], grouped["d1_pred_calibrated__impressions"], 1000.0)
    grouped["actual_roas"] = safe_div(grouped["actual_revenue"], grouped["actual_spend"])
    grouped["pred_roas"] = safe_div(grouped["pred_revenue"], grouped["pred_spend"])
    grouped["cal_roas"] = safe_div(grouped["d1_pred_calibrated__revenue"], grouped["d1_pred_calibrated__spend"])
    grouped["actual_cvr_pct"] = safe_div(grouped["actual_conversions"], grouped["actual_clicks"], 100.0)
    grouped["pred_cvr_pct"] = safe_div(grouped["pred_conversions"], grouped["pred_clicks"], 100.0)
    grouped["cal_cvr_pct"] = safe_div(grouped["d1_pred_calibrated__conversions"], grouped["d1_pred_calibrated__clicks"], 100.0)
    return grouped


def build_payload() -> dict:
    df = load_data()
    account_lookup = build_daily_lookup(df, ["account_id"])
    ad_lookup = build_daily_lookup(df, ["account_id", "ad_id"])
    payload = {"accounts": ACCOUNTS, "data": {}}
    for acc in ACCOUNTS:
        acc_df = df[df["account_id"].eq(acc)].copy()
        ad_rank = acc_df.groupby("ad_id", as_index=False).agg(actual_revenue=("actual_revenue", "sum"), rows=("ad_id", "size")).sort_values(["actual_revenue", "rows"], ascending=False).head(30)
        ad_list = [{"id": "ALL", "label": "All Ads (account aggregate)"}]
        series = {"ALL": series_from_rows(aggregate_account(acc_df), account_lookup, ["account_id"])}
        for _, rec in ad_rank.iterrows():
            ad_id = str(rec["ad_id"])
            ad_rows = acc_df[acc_df["ad_id"].eq(ad_id)].copy()
            ad_list.append({"id": ad_id, "label": f"Ad {ad_id} ({len(ad_rows)} days)"})
            series[ad_id] = series_from_rows(ad_rows, ad_lookup, ["account_id", "ad_id"])
        payload["data"][acc] = {"ad_list": ad_list, "series": series}
    return payload


def main() -> None:
    raw = build_payload()
    html = f"""<!doctype html><html lang="en"><head><meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Adunbox 24H Historical KPI Backtest</title><script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh}}.hdr{{background:linear-gradient(135deg,#111827,#172554 50%,#0f172a);padding:22px 30px;border-bottom:1px solid #263348}}.hdr h1{{font-size:1.45rem;color:#a5b4fc}}.hdr p{{color:#94a3b8;font-size:.84rem;margin-top:5px;max-width:1180px;line-height:1.45}}.wrap{{max-width:1500px;margin:0 auto;padding:18px 24px}}.tabs{{display:flex;gap:4px;margin:16px 0 0;flex-wrap:wrap}}.tab{{padding:9px 18px;border-radius:9px 9px 0 0;font-size:.86rem;font-weight:700;cursor:pointer;color:#64748b;border:1px solid transparent;border-bottom:none}}.tab.active{{background:#1e293b;color:#c7d2fe;border-color:#334155;border-bottom-color:#1e293b}}.panel{{display:none;background:#1e293b;border:1px solid #334155;border-radius:0 12px 12px 12px;padding:18px 20px}}.panel.active{{display:block}}.bar{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;background:#0f172a;border:1px solid #334155;border-radius:10px;padding:12px 16px;margin-bottom:10px}}label,.muted{{font-size:.76rem;color:#94a3b8;font-weight:700}}select{{background:#1e293b;border:1px solid #334155;color:#e2e8f0;padding:7px 10px;border-radius:7px;min-width:260px;max-width:620px}}.btn{{background:#1e293b;border:1px solid #334155;color:#94a3b8;padding:7px 16px;border-radius:8px;cursor:pointer;font-size:.8rem;font-weight:700}}.btn.active{{background:#312e81;border-color:#6366f1;color:#c7d2fe}}.stats{{display:grid;grid-template-columns:repeat(5,minmax(130px,1fr));gap:9px;margin:12px 0}}.stat{{background:#0f172a;border:1px solid #334155;border-radius:9px;padding:10px 13px}}.stat .lbl{{font-size:.67rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em}}.stat .val{{font-size:1.08rem;font-weight:800;margin-top:3px}}.alert{{background:#172554;border:1px solid #1d4ed8;color:#bfdbfe;border-radius:9px;padding:10px 13px;font-size:.82rem;line-height:1.45;margin-bottom:13px}}.legend{{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0 8px}}.li{{display:flex;align-items:center;gap:6px;font-size:.76rem;color:#94a3b8}}.dot{{width:9px;height:9px;border-radius:50%}}.gtitle{{font-size:.72rem;font-weight:900;color:#818cf8;text-transform:uppercase;letter-spacing:.08em;margin:16px 0 9px;display:flex;align-items:center;gap:8px}}.gtitle::before{{content:'';width:4px;height:15px;border-radius:3px;background:#6366f1}}.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}}.card{{background:#0f172a;border:1px solid #334155;border-radius:11px;padding:13px 15px}}.card h3{{font-size:.78rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px}}.card canvas{{max-height:230px}}.tbl-wrap{{overflow-x:auto;border:1px solid #1e293b;border-radius:10px;margin-top:12px}}table{{width:100%;border-collapse:collapse;font-size:.79rem}}th{{background:#111827;color:#94a3b8;text-align:left;padding:9px 10px;white-space:nowrap}}td{{padding:8px 10px;border-top:1px solid #1e293b;white-space:nowrap}}tr:hover td{{background:#1e293b66}}.pill{{padding:3px 8px;border-radius:999px;font-weight:900;font-size:.7rem}}.LOW{{background:#064e3b;color:#bbf7d0}}.MEDIUM{{background:#78350f;color:#fde68a}}.HIGH{{background:#7f1d1d;color:#fecaca}}@media(max-width:960px){{.grid{{grid-template-columns:1fr}}.stats{{grid-template-columns:repeat(2,1fr)}}select{{min-width:180px}}}}
</style></head><body><div class="hdr"><h1>Adunbox - 24H Historical Actual vs Predicted KPI Backtest</h1><p>Charts show date-wise actual vs predicted vs calibrated KPI values. The lower table stays focused on one latest D-6 to D0 history window for the selected account/ad, plus D1 actual/predicted/calibrated context.</p></div><div class="wrap"><div class="tabs" id="tabs"></div><div id="panels"></div></div>
<script>
const RAW={json.dumps(raw, separators=(",", ":"))};const COL={{actual:'#6366f1',pred:'#a5b4fc',cal:'#22d3ee'}};const CH={{}},STATE={{}};
function fmt(v,cur=false,dec=2){{if(v===null||v===undefined||Number.isNaN(Number(v)))return'N/A';const s=Number(v).toLocaleString('en-US',{{maximumFractionDigits:dec}});return cur?'$'+s:s}}function ds(label,data,color,dash=[]){{return{{label,data,borderColor:color,backgroundColor:color+'22',borderWidth:2,borderDash:dash,pointRadius:2.5,pointHoverRadius:5,tension:.25,spanGaps:true}}}}function slice(labels,n){{if(!n)return[0,labels.length];const take=Math.min(labels.length,n);return[labels.length-take,labels.length]}}function draw(id,labels,sets){{if(CH[id])CH[id].destroy();CH[id]=new Chart(document.getElementById(id),{{type:'line',data:{{labels,datasets:sets}},options:{{responsive:true,maintainAspectRatio:true,interaction:{{mode:'index',intersect:false}},plugins:{{legend:{{display:false}},tooltip:{{backgroundColor:'#1e293b',borderColor:'#334155',borderWidth:1,titleColor:'#cbd5e1',bodyColor:'#e2e8f0'}}}},scales:{{x:{{grid:{{color:'#1e293b'}},ticks:{{color:'#64748b',font:{{size:9}},maxRotation:45,minRotation:30}}}},y:{{grid:{{color:'#1e293b'}},ticks:{{color:'#64748b',font:{{size:9}}}}}}}}}}}})}}
function active(acc){{return RAW.data[acc].series[STATE[acc].ad]}}function render(acc){{const s=active(acc),st=STATE[acc],[a,b]=slice(s.labels,st.days),labels=s.labels.slice(a,b);['ctr','cpm','roas','cvr'].forEach(k=>{{const kp=s.kpis[k],sets=[];if(st.mode==='both'||st.mode==='actual')sets.push(ds('Actual',kp.actual.slice(a,b),COL.actual));if(st.mode==='both'||st.mode==='pred')sets.push(ds('Predicted',kp.pred.slice(a,b),COL.pred,[5,4]));if(st.mode==='both'||st.mode==='cal')sets.push(ds('Calibrated',kp.cal.slice(a,b),COL.cal,[2,3]));draw(`ch-${{k}}-${{acc}}`,labels,sets)}});stats(acc);tables(acc)}}function stats(acc){{const s=active(acc),i=s.labels.length-1;document.getElementById(`risk-${{acc}}`).innerHTML=`<span class="pill ${{s.risk_flag}}">Latest ${{s.risk_flag}} (${{s.risk_score}})</span><br><span class="pill ${{s.max_risk_flag}}">Max ${{s.max_risk_flag}} (${{s.max_risk_score}})</span>`;document.getElementById(`ctr-${{acc}}`).textContent=fmt(s.kpis.ctr.cal[i]);document.getElementById(`cpm-${{acc}}`).textContent=fmt(s.kpis.cpm.cal[i],true);document.getElementById(`roas-${{acc}}`).textContent=fmt(s.kpis.roas.cal[i]);document.getElementById(`cvr-${{acc}}`).textContent=fmt(s.kpis.cvr.cal[i]);document.getElementById(`note-${{acc}}`).innerHTML=`<b>Latest spike note:</b> ${{s.review_note}}<br><b>How to read:</b> Latest risk belongs to the selected D1 row. Max risk highlights whether this ad/account had any recent edge-case spike in the chart window.`}}
function tables(acc){{const s=active(acc),h=s.history.rows,d=s.latest_d1;if(!document.getElementById(`riskbody-${{acc}}`)){{document.getElementById(`note-${{acc}}`).insertAdjacentHTML('afterend',`<div class="gtitle">Spike Risk Analysis - Recent Backtest Dates</div><div class="tbl-wrap"><table><thead><tr><th>Date</th><th>Risk</th><th>Score</th><th>Reason / Review Note</th></tr></thead><tbody id="riskbody-${{acc}}"></tbody></table></div>`)}}document.getElementById(`hhead-${{acc}}`).innerHTML='<tr><th>Day</th><th>Date</th><th>Value Type</th><th>Spend</th><th>Impressions</th><th>Clicks</th><th>Conversions</th><th>Revenue</th><th>CTR</th><th>CPM</th><th>ROAS</th><th>CVR</th></tr>';let histRows=h.map(r=>`<tr><td><b>${{r.label}}</b></td><td>${{r.date}}</td><td>Actual</td><td>${{fmt(r.spend,true)}}</td><td>${{fmt(r.impressions)}}</td><td>${{fmt(r.clicks)}}</td><td>${{fmt(r.conversions)}}</td><td>${{fmt(r.revenue,true)}}</td><td>${{fmt(r.ctr)}}</td><td>${{fmt(r.cpm,true)}}</td><td>${{fmt(r.roas)}}</td><td>${{fmt(r.cvr)}}</td></tr>`).join('');let d1Rows=['actual','pred','cal'].map(t=>`<tr><td><b>D1</b></td><td>${{d.date}}</td><td>${{t==='actual'?'Actual':t==='pred'?'Base Predicted':'Calibrated'}}</td><td>${{fmt(d[t].spend,true)}}</td><td>${{fmt(d[t].impressions)}}</td><td>${{fmt(d[t].clicks)}}</td><td>${{fmt(d[t].conversions)}}</td><td>${{fmt(d[t].revenue,true)}}</td><td>${{fmt(d[t].ctr)}}</td><td>${{fmt(d[t].cpm,true)}}</td><td>${{fmt(d[t].roas)}}</td><td>${{fmt(d[t].cvr)}}</td></tr>`).join('');document.getElementById(`hbody-${{acc}}`).innerHTML=histRows+d1Rows;let i=s.labels.length-1;document.getElementById(`d1body-${{acc}}`).innerHTML=['ctr','cpm','roas','cvr'].map(k=>`<tr><td><b>${{k.toUpperCase()}}</b></td><td>${{s.labels[i]}}</td><td>${{fmt(s.kpis[k].actual[i],k==='cpm')}}</td><td>${{fmt(s.kpis[k].pred[i],k==='cpm')}}</td><td>${{fmt(s.kpis[k].cal[i],k==='cpm')}}</td><td>${{fmt(s.kpis[k].cal[i]-s.kpis[k].actual[i],false,3)}}</td></tr>`).join('');document.getElementById(`riskbody-${{acc}}`).innerHTML=s.risk_rows.map(r=>`<tr><td>${{r.date}}</td><td><span class="pill ${{r.flag}}">${{r.flag}}</span></td><td>${{r.score}}</td><td>${{r.note}}</td></tr>`).join('')}}
function switchAcc(acc,el){{document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));el.classList.add('active');document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));document.getElementById('p-'+acc).classList.add('active')}}function onAd(acc){{STATE[acc].ad=document.getElementById(`ad-${{acc}}`).value;render(acc)}}function onMode(acc){{STATE[acc].mode=document.getElementById(`mode-${{acc}}`).value;render(acc)}}function onDays(acc,days,btn){{STATE[acc].days=days;document.getElementById(`days-${{acc}}`).querySelectorAll('.btn').forEach(x=>x.classList.remove('active'));btn.classList.add('active');render(acc)}}
function panel(acc){{STATE[acc]={{ad:'ALL',mode:'both',days:14}};const opts=RAW.data[acc].ad_list.map(a=>`<option value="${{a.id}}">${{a.label}}</option>`).join('');document.getElementById('p-'+acc).innerHTML=`<div class="bar"><label>Select Ad</label><select id="ad-${{acc}}" onchange="onAd('${{acc}}')">${{opts}}</select><label>View</label><select id="mode-${{acc}}" onchange="onMode('${{acc}}')"><option value="both">Actual + Predicted + Calibrated</option><option value="actual">Actual Only</option><option value="pred">Predicted Only</option><option value="cal">Calibrated Only</option></select><span class="muted">Chart Window</span><span id="days-${{acc}}"><button class="btn active" onclick="onDays('${{acc}}',14,this)">Last 14</button> <button class="btn" onclick="onDays('${{acc}}',0,this)">All</button></span></div><div class="stats"><div class="stat"><div class="lbl">Spike Risk</div><div class="val" id="risk-${{acc}}">-</div></div><div class="stat"><div class="lbl">Latest Cal CTR</div><div class="val" id="ctr-${{acc}}">-</div></div><div class="stat"><div class="lbl">Latest Cal CPM</div><div class="val" id="cpm-${{acc}}">-</div></div><div class="stat"><div class="lbl">Latest Cal ROAS</div><div class="val" id="roas-${{acc}}">-</div></div><div class="stat"><div class="lbl">Latest Cal CVR</div><div class="val" id="cvr-${{acc}}">-</div></div></div><div class="alert" id="note-${{acc}}"></div><div class="legend"><div class="li"><span class="dot" style="background:${{COL.actual}}"></span>Actual</div><div class="li"><span class="dot" style="background:${{COL.pred}}"></span>Predicted</div><div class="li"><span class="dot" style="background:${{COL.cal}}"></span>Calibrated / spike-aware</div></div><div class="gtitle">Historical KPI Backtest Charts - CTR · CPM · ROAS · CVR</div><div class="grid"><div class="card"><h3>CTR (%)</h3><canvas id="ch-ctr-${{acc}}"></canvas></div><div class="card"><h3>CPM ($)</h3><canvas id="ch-cpm-${{acc}}"></canvas></div><div class="card"><h3>ROAS</h3><canvas id="ch-roas-${{acc}}"></canvas></div><div class="card"><h3>CVR (%)</h3><canvas id="ch-cvr-${{acc}}"></canvas></div></div><div class="gtitle">Latest D-6 to D0 + D1 Table Used For D1 Context</div><div class="tbl-wrap"><table><thead id="hhead-${{acc}}"></thead><tbody id="hbody-${{acc}}"></tbody></table></div><div class="gtitle">Latest D1 KPI Actual vs Predicted vs Calibrated</div><div class="tbl-wrap"><table><thead><tr><th>KPI</th><th>D1 Date</th><th>Actual</th><th>Base Predicted</th><th>Calibrated</th><th>Calibrated Gap</th></tr></thead><tbody id="d1body-${{acc}}"></tbody></table></div>`;render(acc)}}
function init(){{const tabs=document.getElementById('tabs'),panels=document.getElementById('panels');RAW.accounts.forEach((acc,i)=>{{tabs.insertAdjacentHTML('beforeend',`<div class="tab ${{i===0?'active':''}}" onclick="switchAcc('${{acc}}',this)">Account ${{acc}}</div>`);panels.insertAdjacentHTML('beforeend',`<div class="panel ${{i===0?'active':''}}" id="p-${{acc}}"></div>`);panel(acc)}})}}init();
</script></body></html>"""
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
