from __future__ import annotations

import base64
import gzip
import json
import os
from pathlib import Path

import pandas as pd


BASE_DIR = Path(os.getenv("ADUNBOX_PROJECT_DIR", Path(__file__).resolve().parents[1]))
INPUT_CSV = Path(os.getenv("ADUNBOX_6H_REVIEW_CSV", BASE_DIR / "outputs" / "adunbox_6h_spike_aware_prediction_review_slim.csv"))
METRICS_CSV = BASE_DIR / "adunbox_6h_final_model__metrics.csv"
PATTERN_CSV = BASE_DIR / "adunbox_6h_final_prediction_history_6h_window_pattern.csv"
MISMATCH_CSV = BASE_DIR / "adunbox_6h_spike_aware_mismatch_examples.csv"
SUMMARY_CSV = BASE_DIR / "adunbox_6h_spike_aware_summary.csv"
OUTPUT_HTML = BASE_DIR / "adunbox_6h_visual_review.html"
OUTPUT_FULL_HTML = BASE_DIR / "adunbox_6h_visual_review_full.html"
OUTPUT_BOSS_HTML = BASE_DIR / "adunbox_6h_boss_review.html"
OUTPUT_ALL_COLUMNS_HTML = Path(os.getenv("ADUNBOX_6H_ALL_COLUMNS_HTML", BASE_DIR / "dashboards" / "adunbox_6h_review.html"))


def num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col, 0.0), errors="coerce").fillna(0.0)


def risk_color(flag: str) -> str:
    return {
        "LOW": "#2c7a4b",
        "MEDIUM": "#b7791f",
        "HIGH": "#c53030",
    }.get(str(flag).upper(), "#4a5568")


def sparkline(points: list[float], color: str, width: int = 220, height: int = 70) -> str:
    if not points:
        return ""
    min_v = min(points)
    max_v = max(points)
    spread = max(max_v - min_v, 1e-9)
    coords = []
    for idx, value in enumerate(points):
        x = idx * (width / max(len(points) - 1, 1))
        y = height - ((value - min_v) / spread) * (height - 10) - 5
        coords.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(coords)
    return f'<svg viewBox="0 0 {width} {height}" class="spark"><polyline fill="none" stroke="{color}" stroke-width="3" points="{poly}"/></svg>'


def bar_chart(data: dict[str, int], color_map: dict[str, str]) -> str:
    total = max(sum(data.values()), 1)
    parts = []
    for key, value in data.items():
        width = (value / total) * 100.0
        color = color_map.get(key, "#718096")
        parts.append(
            f'<div class="stack-seg" style="width:{width:.2f}%;background:{color}"><span>{key} {value:,}</span></div>'
        )
    return "".join(parts)


def to_str(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def gzip_b64(path: Path) -> str:
    return base64.b64encode(gzip.compress(path.read_bytes(), compresslevel=9)).decode("ascii")


def build_rows_payload(df: pd.DataFrame, limit: int | None) -> list[dict[str, object]]:
    keep = [
        "account_id",
        "campaign_id",
        "adset_id",
        "ad_id",
        "forecast_anchor_date",
        "next_6h_window_start",
        "next_6h_window_end",
        "next_6h_window_range",
        "model_dataset_split",
        "spike_risk_flag",
        "spike_risk_score",
        "next_6h_actual_spend",
        "next_6h_pred_spend",
        "next_6h_pred_corrected_spend",
        "next_6h_gap_spend",
        "next_6h_gap_corrected_spend",
        "next_6h_actual_inline_link_clicks",
        "next_6h_pred_inline_link_clicks",
        "next_6h_pred_corrected_inline_link_clicks",
        "next_6h_gap_inline_link_clicks",
        "next_6h_gap_corrected_inline_link_clicks",
        "next_6h_actual_tracker_conversions",
        "next_6h_pred_tracker_conversions",
        "next_6h_pred_corrected_tracker_conversions",
        "next_6h_gap_tracker_conversions",
        "next_6h_gap_corrected_tracker_conversions",
        "next_6h_actual_tracker_revenue",
        "next_6h_pred_tracker_revenue",
        "next_6h_pred_corrected_tracker_revenue",
        "next_6h_gap_tracker_revenue",
        "next_6h_gap_corrected_tracker_revenue",
        "history_same_6h_spend_max_7d",
        "history_same_6h_clicks_max_7d",
        "history_same_6h_conversions_max_7d",
        "history_same_6h_revenue_max_7d",
        "history_same_6h_roas_max_7d",
        "spike_guardrail_note",
    ]
    out = df[keep].copy()
    for col in out.columns:
        if "date" in col or "window" in col or col.endswith("_note"):
            out[col] = out[col].map(to_str)
    if limit is not None:
        out = out.head(limit)
    return out.to_dict(orient="records")


def merge_full_review_df(pattern_df: pd.DataFrame, spike_df: pd.DataFrame) -> pd.DataFrame:
    join_keys = [
        "account_id",
        "campaign_id",
        "adset_id",
        "ad_id",
        "forecast_anchor_date",
        "next_6h_window_start",
        "next_6h_window_end",
        "next_6h_window_range",
        "model_dataset_split",
    ]
    extra_cols = [col for col in spike_df.columns if col not in pattern_df.columns and col not in join_keys]
    merged = pattern_df.merge(spike_df[join_keys + extra_cols], on=join_keys, how="left", validate="one_to_one")
    return merged


def to_serializable_df(df: pd.DataFrame) -> dict[str, object]:
    safe_df = df.copy()
    for col in safe_df.columns:
        if pd.api.types.is_datetime64_any_dtype(safe_df[col]):
            safe_df[col] = safe_df[col].astype(str)
        elif safe_df[col].dtype == object:
            safe_df[col] = safe_df[col].map(to_str)
    safe_df = safe_df.where(pd.notna(safe_df), "")
    return {
        "columns": list(safe_df.columns),
        "rows": safe_df.values.tolist(),
    }


def build_main_html(
    *,
    df: pd.DataFrame,
    metric_map: dict[str, float],
    risk_counts: dict[str, int],
    split_counts: dict[str, int],
    top_revenue: pd.DataFrame,
    top_conv: pd.DataFrame,
    spike_summary: pd.DataFrame,
    rows_payload: list[dict[str, object]],
    title: str,
    description: str,
    detail_note: str,
    output_name: str,
) -> str:
    page_data = {
        "rows": rows_payload,
        "links": {
            "pattern_csv": str(PATTERN_CSV),
            "spike_csv": str(INPUT_CSV),
            "mismatch_csv": str(MISMATCH_CSV),
            "boss_html": str(OUTPUT_BOSS_HTML),
            "full_html": str(OUTPUT_FULL_HTML),
            "lite_html": str(OUTPUT_HTML),
            "all_columns_html": str(OUTPUT_ALL_COLUMNS_HTML),
        },
    }

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --card: #fffaf2;
      --ink: #1e293b;
      --muted: #6b7280;
      --line: #d9cbb6;
      --accent: #0f766e;
      --accent-2: #c2410c;
      --low: #2c7a4b;
      --med: #b7791f;
      --high: #c53030;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(194,65,12,.08), transparent 32%),
        radial-gradient(circle at left center, rgba(15,118,110,.08), transparent 28%),
        var(--bg);
    }}
    .wrap {{ max-width: 1480px; margin: 0 auto; padding: 28px; }}
    .hero {{
      display: grid;
      grid-template-columns: 1.3fr .7fr;
      gap: 22px;
      margin-bottom: 22px;
    }}
    .panel {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 22px;
      box-shadow: 0 18px 40px rgba(30,41,59,.06);
    }}
    h1 {{ margin: 0 0 8px; font-size: 34px; }}
    h2 {{ margin: 0 0 16px; font-size: 20px; }}
    p {{ margin: 0; line-height: 1.5; }}
    .subtle {{ color: var(--muted); }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 14px;
      margin: 22px 0;
    }}
    .card {{
      background: rgba(255,255,255,.7);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
    }}
    .k {{ font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }}
    .v {{ font-size: 28px; margin-top: 6px; font-weight: 700; }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 22px;
      margin-bottom: 22px;
    }}
    .stack {{
      display: flex;
      overflow: hidden;
      border-radius: 999px;
      height: 34px;
      border: 1px solid var(--line);
      background: #efe6d9;
      margin-top: 14px;
    }}
    .stack-seg {{
      display: flex;
      align-items: center;
      justify-content: center;
      color: #fff;
      font-size: 12px;
      white-space: nowrap;
      overflow: hidden;
    }}
    .stack-seg span {{ padding: 0 8px; }}
    .spark {{ width: 100%; height: 70px; margin-top: 10px; }}
    .mini-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .mini-table th, .mini-table td {{
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    .pill {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      color: white;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: 1.5fr .7fr .7fr .7fr;
      gap: 12px;
      margin-bottom: 14px;
    }}
    input, select {{
      width: 100%;
      padding: 12px 14px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: white;
      font: inherit;
    }}
    .table-wrap {{
      max-height: 720px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: white;
    }}
    table.data {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    table.data th {{
      position: sticky;
      top: 0;
      background: #f8f2e9;
      z-index: 2;
    }}
    table.data th, table.data td {{
      padding: 9px 10px;
      border-bottom: 1px solid #ece3d6;
      text-align: left;
      white-space: nowrap;
    }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .good {{ color: var(--low); }}
    .bad {{ color: var(--high); }}
    .links a {{
      color: var(--accent);
      text-decoration: none;
      margin-right: 16px;
    }}
    @media (max-width: 1100px) {{
      .hero, .grid-2, .cards, .toolbar {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="panel">
        <h1>{title}</h1>
        <p class="subtle">{description}</p>
        <div class="cards">
          <div class="card"><div class="k">Rows</div><div class="v">{len(df):,}</div></div>
          <div class="card"><div class="k">Spend R2</div><div class="v">{metric_map.get('target_spend', 0.0):.3f}</div></div>
          <div class="card"><div class="k">Impr R2</div><div class="v">{metric_map.get('target_impressions', 0.0):.3f}</div></div>
          <div class="card"><div class="k">Clicks R2</div><div class="v">{metric_map.get('target_inline_link_clicks', 0.0):.3f}</div></div>
          <div class="card"><div class="k">Revenue R2</div><div class="v">{metric_map.get('target_tracker_revenue', 0.0):.3f}</div></div>
        </div>
        <div class="links">
          <a href="{PATTERN_CSV.name}">Detailed 6h Window Pattern CSV</a>
          <a href="{INPUT_CSV.name}">Spike-Aware Slim CSV</a>
          <a href="{MISMATCH_CSV.name}">Top Mismatch CSV</a>
          <a href="{OUTPUT_BOSS_HTML.name}">Boss Page</a>
          <a href="{OUTPUT_ALL_COLUMNS_HTML.name}">All Columns Page</a>
          <a href="{OUTPUT_FULL_HTML.name}">Full Page</a>
          <a href="{OUTPUT_HTML.name}">Lite Page</a>
        </div>
      </div>
      <div class="panel">
        <h2>Risk Mix</h2>
        <p class="subtle">Spike-risk flags based on same-window 7-day history caps.</p>
        <div class="stack">{bar_chart(risk_counts, {'LOW': '#2c7a4b', 'MEDIUM': '#b7791f', 'HIGH': '#c53030'})}</div>
        <h2 style="margin-top:18px;">Dataset Split Mix</h2>
        <div class="stack">{bar_chart(split_counts, {'train': '#0f766e', 'valid': '#c2410c', 'test': '#4338ca'})}</div>
        <h2 style="margin-top:18px;">Revenue Trace</h2>
        {sparkline(num(df, "next_6h_actual_tracker_revenue").head(240).tolist(), "#0f766e")}
      </div>
    </section>

    <section class="grid-2">
      <div class="panel">
        <h2>Top Revenue Mismatches</h2>
        <table class="mini-table">
          <thead><tr><th>Ad</th><th>Window</th><th>Risk</th><th>Actual</th><th>Pred</th><th>Corrected</th></tr></thead>
          <tbody>
            {''.join(f"<tr><td>{row.ad_id}</td><td>{row.next_6h_window_range}</td><td><span class='pill' style='background:{risk_color(row.spike_risk_flag)}'>{row.spike_risk_flag}</span></td><td>{row.next_6h_actual_tracker_revenue:.2f}</td><td>{row.next_6h_pred_tracker_revenue:.2f}</td><td>{row.next_6h_pred_corrected_tracker_revenue:.2f}</td></tr>" for row in top_revenue.itertuples(index=False))}
          </tbody>
        </table>
      </div>
      <div class="panel">
        <h2>Top Conversion Mismatches</h2>
        <table class="mini-table">
          <thead><tr><th>Ad</th><th>Window</th><th>Risk</th><th>Actual</th><th>Pred</th><th>Corrected</th></tr></thead>
          <tbody>
            {''.join(f"<tr><td>{row.ad_id}</td><td>{row.next_6h_window_range}</td><td><span class='pill' style='background:{risk_color(row.spike_risk_flag)}'>{row.spike_risk_flag}</span></td><td>{row.next_6h_actual_tracker_conversions:.2f}</td><td>{row.next_6h_pred_tracker_conversions:.2f}</td><td>{row.next_6h_pred_corrected_tracker_conversions:.2f}</td></tr>" for row in top_conv.itertuples(index=False))}
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel" style="margin-bottom:22px;">
      <h2>Spike Summary</h2>
      <table class="mini-table">
        <thead><tr><th>Risk</th><th>Rows</th><th>Avg Rev Gap Before</th><th>Avg Rev Gap After</th><th>Avg Conv Gap Before</th><th>Avg Conv Gap After</th></tr></thead>
        <tbody>
          {''.join(f"<tr><td><span class='pill' style='background:{risk_color(row.spike_risk_flag)}'>{row.spike_risk_flag}</span></td><td>{int(row.rows):,}</td><td>{float(row.avg_abs_rev_gap_before):.2f}</td><td>{float(row.avg_abs_rev_gap_after):.2f}</td><td>{float(row.avg_abs_conv_gap_before):.2f}</td><td>{float(row.avg_abs_conv_gap_after):.2f}</td></tr>" for row in spike_summary.itertuples(index=False))}
        </tbody>
      </table>
    </section>

    <section class="panel">
      <h2>Interactive Row Review</h2>
      <p class="subtle">{detail_note}</p>
      <div class="toolbar">
        <input id="search" placeholder="Search ad_id / account_id / campaign_id / window">
        <select id="risk">
          <option value="">All Risk Flags</option>
          <option value="LOW">LOW</option>
          <option value="MEDIUM">MEDIUM</option>
          <option value="HIGH">HIGH</option>
        </select>
        <select id="split">
          <option value="">All Splits</option>
          <option value="train">train</option>
          <option value="valid">valid</option>
          <option value="test">test</option>
        </select>
        <select id="metric">
          <option value="revenue">Sort by Revenue Gap</option>
          <option value="conversions">Sort by Conversion Gap</option>
          <option value="spend">Sort by Spend Gap</option>
        </select>
      </div>
      <div class="table-wrap">
        <table class="data" id="data-table">
          <thead>
            <tr>
              <th>ad_id</th>
              <th>split</th>
              <th>risk</th>
              <th>window</th>
              <th>actual spend</th>
              <th>pred spend</th>
              <th>corrected spend</th>
              <th>actual conv</th>
              <th>pred conv</th>
              <th>corrected conv</th>
              <th>actual rev</th>
              <th>pred rev</th>
              <th>corrected rev</th>
              <th>7d max spend</th>
              <th>7d max conv</th>
              <th>7d max rev</th>
              <th>note</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </section>
  </div>

  <script id="page-data" type="application/json">{json.dumps(page_data)}</script>
  <script>
    const pageData = JSON.parse(document.getElementById('page-data').textContent);
    const rows = pageData.rows;
    const tbody = document.querySelector('#data-table tbody');
    const searchEl = document.getElementById('search');
    const riskEl = document.getElementById('risk');
    const splitEl = document.getElementById('split');
    const metricEl = document.getElementById('metric');

    function fmt(value) {{
      const n = Number(value || 0);
      return Number.isFinite(n) ? n.toFixed(2) : '';
    }}

    function metricField(metric) {{
      if (metric === 'conversions') return 'next_6h_gap_corrected_tracker_conversions';
      if (metric === 'spend') return 'next_6h_gap_corrected_spend';
      return 'next_6h_gap_corrected_tracker_revenue';
    }}

    function render() {{
      const q = searchEl.value.trim().toLowerCase();
      const risk = riskEl.value;
      const split = splitEl.value;
      const field = metricField(metricEl.value);
      let filtered = rows.filter(row => {{
        const text = [row.ad_id, row.account_id, row.campaign_id, row.next_6h_window_range].join(' ').toLowerCase();
        if (q && !text.includes(q)) return false;
        if (risk && row.spike_risk_flag !== risk) return false;
        if (split && row.model_dataset_split !== split) return false;
        return true;
      }});
      filtered.sort((a, b) => Math.abs(Number(b[field] || 0)) - Math.abs(Number(a[field] || 0)));

      tbody.innerHTML = filtered.map(row => `
        <tr>
          <td>${{row.ad_id}}</td>
          <td>${{row.model_dataset_split}}</td>
          <td><span class="pill" style="background:${{row.spike_risk_flag === 'HIGH' ? '#c53030' : row.spike_risk_flag === 'MEDIUM' ? '#b7791f' : '#2c7a4b'}}">${{row.spike_risk_flag}}</span></td>
          <td>${{row.next_6h_window_range}}</td>
          <td class="num">${{fmt(row.next_6h_actual_spend)}}</td>
          <td class="num bad">${{fmt(row.next_6h_pred_spend)}}</td>
          <td class="num good">${{fmt(row.next_6h_pred_corrected_spend)}}</td>
          <td class="num">${{fmt(row.next_6h_actual_tracker_conversions)}}</td>
          <td class="num bad">${{fmt(row.next_6h_pred_tracker_conversions)}}</td>
          <td class="num good">${{fmt(row.next_6h_pred_corrected_tracker_conversions)}}</td>
          <td class="num">${{fmt(row.next_6h_actual_tracker_revenue)}}</td>
          <td class="num bad">${{fmt(row.next_6h_pred_tracker_revenue)}}</td>
          <td class="num good">${{fmt(row.next_6h_pred_corrected_tracker_revenue)}}</td>
          <td class="num">${{fmt(row.history_same_6h_spend_max_7d)}}</td>
          <td class="num">${{fmt(row.history_same_6h_conversions_max_7d)}}</td>
          <td class="num">${{fmt(row.history_same_6h_revenue_max_7d)}}</td>
          <td>${{row.spike_guardrail_note || ''}}</td>
        </tr>
      `).join('');
    }}

    [searchEl, riskEl, splitEl, metricEl].forEach(el => el.addEventListener('input', render));
    [riskEl, splitEl, metricEl].forEach(el => el.addEventListener('change', render));
    render();
  </script>
</body>
</html>"""


def build_boss_html(
    *,
    df: pd.DataFrame,
    metric_map: dict[str, float],
    risk_counts: dict[str, int],
    top_revenue: pd.DataFrame,
    top_conv: pd.DataFrame,
    spike_summary: pd.DataFrame,
) -> str:
    total_rows = len(df)
    high_risk = int(risk_counts.get("HIGH", 0))
    medium_risk = int(risk_counts.get("MEDIUM", 0))
    low_risk = int(risk_counts.get("LOW", 0))
    avg_rev_gap_before = float(num(df, "next_6h_gap_tracker_revenue").abs().mean())
    avg_rev_gap_after = float(num(df, "next_6h_gap_corrected_tracker_revenue").abs().mean())
    avg_conv_gap_before = float(num(df, "next_6h_gap_tracker_conversions").abs().mean())
    avg_conv_gap_after = float(num(df, "next_6h_gap_corrected_tracker_conversions").abs().mean())

    top_revenue_rows = "".join(
        f"<tr><td>{row.ad_id}</td><td>{row.next_6h_window_range}</td><td>{row.next_6h_actual_tracker_revenue:.2f}</td><td>{row.next_6h_pred_tracker_revenue:.2f}</td><td>{row.next_6h_pred_corrected_tracker_revenue:.2f}</td><td><span class='pill' style='background:{risk_color(row.spike_risk_flag)}'>{row.spike_risk_flag}</span></td></tr>"
        for row in top_revenue.head(8).itertuples(index=False)
    )
    top_conv_rows = "".join(
        f"<tr><td>{row.ad_id}</td><td>{row.next_6h_window_range}</td><td>{row.next_6h_actual_tracker_conversions:.2f}</td><td>{row.next_6h_pred_tracker_conversions:.2f}</td><td>{row.next_6h_pred_corrected_tracker_conversions:.2f}</td><td><span class='pill' style='background:{risk_color(row.spike_risk_flag)}'>{row.spike_risk_flag}</span></td></tr>"
        for row in top_conv.head(8).itertuples(index=False)
    )
    spike_rows = "".join(
        f"<tr><td><span class='pill' style='background:{risk_color(row.spike_risk_flag)}'>{row.spike_risk_flag}</span></td><td>{int(row.rows):,}</td><td>{float(row.avg_abs_rev_gap_before):.2f}</td><td>{float(row.avg_abs_rev_gap_after):.2f}</td><td>{float(row.avg_abs_conv_gap_before):.2f}</td><td>{float(row.avg_abs_conv_gap_after):.2f}</td></tr>"
        for row in spike_summary.itertuples(index=False)
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Adunbox 6H Boss Review</title>
  <style>
    :root {{
      --bg: #f7f1e8;
      --card: rgba(255, 251, 245, 0.92);
      --ink: #172554;
      --muted: #5b6472;
      --line: #dfd0be;
      --sea: #0f766e;
      --sand: #d97706;
      --rose: #be123c;
      --sky: #1d4ed8;
      --shadow: 0 24px 50px rgba(23,37,84,.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: Cambria, Georgia, serif;
      background:
        radial-gradient(circle at top left, rgba(29,78,216,.08), transparent 26%),
        radial-gradient(circle at 85% 8%, rgba(217,119,6,.10), transparent 22%),
        linear-gradient(180deg, #fbf7f1 0%, #f2ebe2 100%);
    }}
    .wrap {{ max-width: 1320px; margin: 0 auto; padding: 34px 26px 40px; }}
    .hero {{
      display: grid;
      grid-template-columns: 1.2fr .8fr;
      gap: 24px;
      margin-bottom: 24px;
    }}
    .panel {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 26px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(6px);
    }}
    h1 {{ margin: 0 0 12px; font-size: 42px; line-height: 1.05; }}
    h2 {{ margin: 0 0 14px; font-size: 22px; }}
    p {{ margin: 0; line-height: 1.55; }}
    .muted {{ color: var(--muted); }}
    .ribbon {{
      display: inline-block;
      padding: 8px 14px;
      border-radius: 999px;
      background: rgba(15,118,110,.10);
      color: var(--sea);
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .08em;
      margin-bottom: 16px;
    }}
    .score-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 14px;
      margin-top: 22px;
    }}
    .score {{
      background: rgba(255,255,255,.72);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
    }}
    .score .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }}
    .score .value {{ margin-top: 8px; font-size: 34px; font-weight: 700; }}
    .callouts {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-top: 20px;
    }}
    .callout {{
      border-radius: 20px;
      padding: 18px;
      border: 1px solid var(--line);
      background: linear-gradient(135deg, rgba(255,255,255,.78), rgba(250,242,232,.96));
    }}
    .big {{
      font-size: 38px;
      font-weight: 700;
      line-height: 1;
      margin-top: 10px;
    }}
    .strip {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 18px;
      margin: 24px 0;
    }}
    .tile {{
      padding: 20px;
      border-radius: 24px;
      color: white;
      min-height: 160px;
    }}
    .tile h3 {{ margin: 0 0 10px; font-size: 18px; }}
    .tile p {{ opacity: .95; }}
    .sea {{ background: linear-gradient(135deg, #0f766e, #115e59); }}
    .sand {{ background: linear-gradient(135deg, #d97706, #b45309); }}
    .rose {{ background: linear-gradient(135deg, #be123c, #9f1239); }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-top: 24px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
    }}
    .pill {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      color: white;
    }}
    .footer-links {{
      margin-top: 24px;
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .footer-links a {{
      text-decoration: none;
      color: var(--sky);
      padding: 10px 14px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255,255,255,.8);
    }}
    @media (max-width: 1100px) {{
      .hero, .grid-2, .score-grid, .strip, .callouts {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="panel">
        <div class="ribbon">6H Forecast Review</div>
        <h1>Production review page for the final 6-hour Adunbox model</h1>
        <p class="muted">This page is meant for quick stakeholder review. It focuses on model quality, spike-risk behavior, and where the corrected guardrail output is helping most.</p>
        <div class="score-grid">
          <div class="score"><div class="label">Rows Reviewed</div><div class="value">{total_rows:,}</div></div>
          <div class="score"><div class="label">Spend R2</div><div class="value">{metric_map.get('target_spend', 0.0):.3f}</div></div>
          <div class="score"><div class="label">Impressions R2</div><div class="value">{metric_map.get('target_impressions', 0.0):.3f}</div></div>
          <div class="score"><div class="label">Clicks R2</div><div class="value">{metric_map.get('target_inline_link_clicks', 0.0):.3f}</div></div>
          <div class="score"><div class="label">Revenue R2</div><div class="value">{metric_map.get('target_tracker_revenue', 0.0):.3f}</div></div>
        </div>
      </div>
      <div class="panel">
        <h2>Guardrail impact</h2>
        <div class="callouts">
          <div class="callout">
            <div class="muted">Avg revenue abs gap</div>
            <div class="big">{avg_rev_gap_after:.2f}</div>
            <p class="muted">down from {avg_rev_gap_before:.2f} after spike-aware correction</p>
          </div>
          <div class="callout">
            <div class="muted">Avg conversion abs gap</div>
            <div class="big">{avg_conv_gap_after:.2f}</div>
            <p class="muted">down from {avg_conv_gap_before:.2f} after correction</p>
          </div>
        </div>
        <div class="callouts">
          <div class="callout">
            <div class="muted">High risk rows</div>
            <div class="big">{high_risk:,}</div>
            <p class="muted">rows needing the most caution in production</p>
          </div>
          <div class="callout">
            <div class="muted">Low / medium risk rows</div>
            <div class="big">{low_risk + medium_risk:,}</div>
            <p class="muted">rows where the model is more stable</p>
          </div>
        </div>
      </div>
    </section>

    <section class="strip">
      <div class="tile sea">
        <h3>What is working</h3>
        <p>Volume targets stay usable and the corrected review output reduces many medium-risk blowups without changing the audited actual values.</p>
      </div>
      <div class="tile sand">
        <h3>What needs caution</h3>
        <p>Revenue and conversion spikes are still the main operational risk. They are concentrated in a smaller subset of ads and windows.</p>
      </div>
      <div class="tile rose">
        <h3>Production recommendation</h3>
        <p>Serve the corrected output with spike flags, keep fallback logic for high-risk rows, and log live forecasts until fresh actuals arrive six hours later.</p>
      </div>
    </section>

    <section class="grid-2">
      <div class="panel">
        <h2>Top revenue outliers</h2>
        <table>
          <thead><tr><th>Ad</th><th>Window</th><th>Actual</th><th>Pred</th><th>Corrected</th><th>Risk</th></tr></thead>
          <tbody>{top_revenue_rows}</tbody>
        </table>
      </div>
      <div class="panel">
        <h2>Top conversion outliers</h2>
        <table>
          <thead><tr><th>Ad</th><th>Window</th><th>Actual</th><th>Pred</th><th>Corrected</th><th>Risk</th></tr></thead>
          <tbody>{top_conv_rows}</tbody>
        </table>
      </div>
    </section>

    <section class="panel" style="margin-top:24px;">
      <h2>Spike summary by risk bucket</h2>
      <table>
        <thead><tr><th>Risk</th><th>Rows</th><th>Avg Rev Gap Before</th><th>Avg Rev Gap After</th><th>Avg Conv Gap Before</th><th>Avg Conv Gap After</th></tr></thead>
        <tbody>{spike_rows}</tbody>
      </table>
      <div class="footer-links">
        <a href="{OUTPUT_FULL_HTML.name}">Open full row-level page</a>
        <a href="{OUTPUT_ALL_COLUMNS_HTML.name}">Open all-column page</a>
        <a href="{OUTPUT_HTML.name}">Open lite technical page</a>
        <a href="{PATTERN_CSV.name}">Open detailed 6h pattern CSV</a>
        <a href="{INPUT_CSV.name}">Open spike-aware slim CSV</a>
      </div>
    </section>
  </div>
</body>
</html>"""


def build_all_columns_html(
    *,
    merged_df: pd.DataFrame,
    metric_map: dict[str, float],
) -> str:
    payload = to_serializable_df(merged_df)
    key_columns = [
        "account_id",
        "campaign_id",
        "adset_id",
        "ad_id",
        "forecast_anchor_date",
        "next_6h_window_range",
        "model_dataset_split",
        "spike_risk_flag",
    ]
    default_sort_col = "next_6h_abs_gap_tracker_revenue" if "next_6h_abs_gap_tracker_revenue" in merged_df.columns else ""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Adunbox 6H Full Column Review</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --card: #fffaf2;
      --ink: #1e293b;
      --muted: #6b7280;
      --line: #d9cbb6;
      --accent: #0f766e;
      --accent-2: #c2410c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(194,65,12,.08), transparent 32%),
        radial-gradient(circle at left center, rgba(15,118,110,.08), transparent 28%),
        var(--bg);
    }}
    .wrap {{ max-width: 100%; padding: 22px; }}
    .panel {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 20px;
      box-shadow: 0 18px 40px rgba(30,41,59,.06);
      margin-bottom: 18px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 32px; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    p {{ margin: 0; line-height: 1.5; }}
    .subtle {{ color: var(--muted); }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .card {{
      background: rgba(255,255,255,.75);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
    }}
    .k {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }}
    .v {{ font-size: 24px; margin-top: 6px; font-weight: 700; }}
    .links a {{
      color: var(--accent);
      text-decoration: none;
      margin-right: 14px;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: 2fr 1fr 1fr 1fr 1fr;
      gap: 12px;
      margin-top: 16px;
    }}
    input, select {{
      width: 100%;
      padding: 12px 14px;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: white;
      font: inherit;
    }}
    .table-wrap {{
      max-height: 76vh;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: white;
    }}
    table {{
      width: max-content;
      min-width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: #f8f2e9;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid #ece3d6;
      text-align: left;
      white-space: nowrap;
      max-width: 320px;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    td.num {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    .summary {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 1200px) {{
      .cards, .toolbar {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="panel">
      <h1>Adunbox 6H Full Column Review</h1>
      <p class="subtle">This page includes every row and every available merged column from the detailed 6h pattern CSV plus the spike-aware corrected review CSV.</p>
      <div class="cards">
        <div class="card"><div class="k">Rows</div><div class="v">{len(merged_df):,}</div></div>
        <div class="card"><div class="k">Columns</div><div class="v">{len(merged_df.columns):,}</div></div>
        <div class="card"><div class="k">Spend R2</div><div class="v">{metric_map.get('target_spend', 0.0):.3f}</div></div>
        <div class="card"><div class="k">Impr R2</div><div class="v">{metric_map.get('target_impressions', 0.0):.3f}</div></div>
        <div class="card"><div class="k">Clicks R2</div><div class="v">{metric_map.get('target_inline_link_clicks', 0.0):.3f}</div></div>
        <div class="card"><div class="k">Revenue R2</div><div class="v">{metric_map.get('target_tracker_revenue', 0.0):.3f}</div></div>
      </div>
      <div class="summary">
        <span>Source CSV 1: {PATTERN_CSV.name}</span>
        <span>Source CSV 2: {INPUT_CSV.name}</span>
        <span>Main review file: {OUTPUT_ALL_COLUMNS_HTML.name}</span>
      </div>
      <div class="links" style="margin-top:14px;">
        <a href="{OUTPUT_BOSS_HTML.name}">Boss Page</a>
        <a href="{OUTPUT_FULL_HTML.name}">Focused Full Page</a>
        <a href="{OUTPUT_HTML.name}">Lite Page</a>
        <a href="{PATTERN_CSV.name}">Pattern CSV</a>
        <a href="{INPUT_CSV.name}">Spike-Aware CSV</a>
      </div>
    </section>

    <section class="panel">
      <h2>Interactive Wide Table</h2>
      <p class="subtle">Use search, split, and risk filters. The table renders the top matching rows after filtering so the browser stays usable even with the full merged column set.</p>
      <div class="toolbar">
        <input id="search" placeholder="Search IDs, dates, window text, notes">
        <select id="split"><option value="">All Splits</option></select>
        <select id="risk"><option value="">All Risk Flags</option></select>
        <select id="sort"></select>
        <select id="limit">
          <option value="100">Top 100 rows</option>
          <option value="250" selected>Top 250 rows</option>
          <option value="500">Top 500 rows</option>
          <option value="1000">Top 1000 rows</option>
        </select>
      </div>
      <div class="summary">
        <span id="match-count">Preparing rows...</span>
      </div>
      <div class="table-wrap">
        <table id="wide-table">
          <thead></thead>
          <tbody></tbody>
        </table>
      </div>
    </section>
  </div>

  <script id="page-data" type="application/json">{json.dumps({"payload": payload, "keyColumns": key_columns, "defaultSortCol": default_sort_col})}</script>
  <script>
    const pageData = JSON.parse(document.getElementById('page-data').textContent);
    const columns = pageData.payload.columns;
    const rows = pageData.payload.rows;
    const keyColumns = new Set(pageData.keyColumns);
    const defaultSortCol = pageData.defaultSortCol;

    const searchEl = document.getElementById('search');
    const splitEl = document.getElementById('split');
    const riskEl = document.getElementById('risk');
    const sortEl = document.getElementById('sort');
    const limitEl = document.getElementById('limit');
    const matchCountEl = document.getElementById('match-count');
    const thead = document.querySelector('#wide-table thead');
    const tbody = document.querySelector('#wide-table tbody');

    const splitIdx = columns.indexOf('model_dataset_split');
    const riskIdx = columns.indexOf('spike_risk_flag');
    const searchIndices = columns
      .map((col, idx) => keyColumns.has(col) || col.endsWith('_note') ? idx : -1)
      .filter(idx => idx >= 0);

    function isNumericLike(value) {{
      if (value === '' || value === null || value === undefined) return false;
      return !Number.isNaN(Number(value));
    }}

    function formatValue(value) {{
      if (value === null || value === undefined || value === '') return '';
      if (isNumericLike(value)) {{
        const n = Number(value);
        return Number.isInteger(n) ? String(n) : n.toFixed(4);
      }}
      return String(value);
    }}

    function buildHeader() {{
      thead.innerHTML = '<tr>' + columns.map(col => `<th title="${{col}}">${{col}}</th>`).join('') + '</tr>';
    }}

    function initFilters() {{
      const splits = [...new Set(rows.map(r => splitIdx >= 0 ? String(r[splitIdx] || '') : '').filter(Boolean))].sort();
      const risks = [...new Set(rows.map(r => riskIdx >= 0 ? String(r[riskIdx] || '') : '').filter(Boolean))].sort();
      splitEl.innerHTML = '<option value="">All Splits</option>' + splits.map(v => `<option value="${{v}}">${{v}}</option>`).join('');
      riskEl.innerHTML = '<option value="">All Risk Flags</option>' + risks.map(v => `<option value="${{v}}">${{v}}</option>`).join('');

      const preferredSorts = [
        'next_6h_abs_gap_tracker_revenue',
        'next_6h_gap_corrected_tracker_revenue',
        'next_6h_abs_gap_tracker_conversions',
        'next_6h_gap_corrected_tracker_conversions',
        'next_6h_abs_gap_spend',
        'spike_risk_score'
      ].filter(col => columns.includes(col));
      const otherNumeric = columns.filter(col => !preferredSorts.includes(col));
      sortEl.innerHTML = preferredSorts.concat(otherNumeric).map(col => `<option value="${{col}}">${{col}}</option>`).join('');
      if (defaultSortCol && columns.includes(defaultSortCol)) {{
        sortEl.value = defaultSortCol;
      }}
    }}

    function render() {{
      const q = searchEl.value.trim().toLowerCase();
      const split = splitEl.value;
      const risk = riskEl.value;
      const sortCol = sortEl.value;
      const limit = Number(limitEl.value || 250);
      const sortIdx = columns.indexOf(sortCol);

      let filtered = rows.filter(row => {{
        if (split && splitIdx >= 0 && String(row[splitIdx] || '') !== split) return false;
        if (risk && riskIdx >= 0 && String(row[riskIdx] || '') !== risk) return false;
        if (q) {{
          const text = searchIndices.map(idx => String(row[idx] || '')).join(' ').toLowerCase();
          if (!text.includes(q)) return false;
        }}
        return true;
      }});

      if (sortIdx >= 0) {{
        filtered.sort((a, b) => {{
          const av = a[sortIdx];
          const bv = b[sortIdx];
          if (isNumericLike(av) || isNumericLike(bv)) {{
            return Math.abs(Number(bv || 0)) - Math.abs(Number(av || 0));
          }}
          return String(av || '').localeCompare(String(bv || ''));
        }});
      }}

      matchCountEl.textContent = `${{filtered.length.toLocaleString()}} matching rows. Showing ${{Math.min(filtered.length, limit).toLocaleString()}} rows in the table.`;
      filtered = filtered.slice(0, limit);

      tbody.innerHTML = filtered.map(row => '<tr>' + row.map(value => {{
        const cls = isNumericLike(value) ? ' class="num"' : '';
        const text = formatValue(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
        return `<td${{cls}} title="${{text}}">${{text}}</td>`;
      }}).join('') + '</tr>').join('');
    }}

    buildHeader();
    initFilters();
    [searchEl, splitEl, riskEl, sortEl, limitEl].forEach(el => {{
      el.addEventListener('input', render);
      el.addEventListener('change', render);
    }});
    render();
  </script>
</body>
</html>"""


def build_all_columns_loader_html(
    *,
    pattern_columns: list[str],
    spike_columns: list[str],
    pattern_rows: int,
    spike_rows: int,
    pattern_gzip_b64: str,
    spike_gzip_b64: str,
    metric_map: dict[str, float],
) -> str:
    sources = {
        "pattern": {
            "label": "Detailed 6h Window Pattern CSV",
            "file": PATTERN_CSV.name,
            "columns": pattern_columns,
            "rows": pattern_rows,
            "gzipBase64": pattern_gzip_b64,
            "gzipBytes": len(pattern_gzip_b64) * 3 // 4,
        },
        "spike": {
            "label": "Spike-Aware Prediction Review CSV",
            "file": INPUT_CSV.name,
            "columns": spike_columns,
            "rows": spike_rows,
            "gzipBase64": spike_gzip_b64,
            "gzipBytes": len(spike_gzip_b64) * 3 // 4,
        },
    }

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Adunbox 6H All Columns Review</title>
  <style>
    :root {{
      --bg: #f6f7f3;
      --panel: #ffffff;
      --ink: #18212f;
      --muted: #657084;
      --line: #d8ddd2;
      --accent: #0f766e;
      --accent-2: #b45309;
      --danger: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        linear-gradient(135deg, rgba(15,118,110,.08), transparent 32%),
        linear-gradient(315deg, rgba(180,83,9,.08), transparent 28%),
        var(--bg);
    }}
    .wrap {{ max-width: 100%; padding: 22px; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 16px 32px rgba(24,33,47,.06);
      margin-bottom: 16px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    p {{ margin: 0; line-height: 1.45; }}
    .muted {{ color: var(--muted); }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin-top: 16px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfa;
    }}
    .k {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }}
    .v {{ margin-top: 5px; font-size: 22px; font-weight: 700; }}
    .toolbar {{
      display: grid;
      grid-template-columns: 1fr 1fr 2fr 1fr 1fr 1fr;
      gap: 10px;
      margin-top: 14px;
      align-items: end;
    }}
    label {{ display: block; font-size: 12px; color: var(--muted); margin-bottom: 5px; }}
    input, select, button {{
      width: 100%;
      min-height: 42px;
      padding: 10px 12px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      font: inherit;
    }}
    button {{
      cursor: pointer;
      background: var(--accent);
      color: white;
      border-color: var(--accent);
      font-weight: 700;
    }}
    .links {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-top: 12px;
    }}
    .links a {{ color: var(--accent); text-decoration: none; }}
    .status {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .column-box {{
      max-height: 130px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfcfa;
      font-size: 12px;
      line-height: 1.55;
      word-break: break-word;
    }}
    .table-wrap {{
      max-height: 76vh;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
    }}
    table {{
      width: max-content;
      min-width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: #eef3ec;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid #edf0ea;
      text-align: left;
      white-space: nowrap;
      max-width: 320px;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .warn {{ color: var(--danger); }}
    @media (max-width: 1100px) {{
      .cards, .toolbar {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="panel">
      <h1>Adunbox 6H All Columns Review</h1>
      <p class="muted">This page exposes every column from the selected CSV. The CSV data is embedded directly inside this HTML in compressed form, so this file can be shared by itself.</p>
      <div class="cards">
        <div class="card"><div class="k">Pattern Rows</div><div class="v">{pattern_rows:,}</div></div>
        <div class="card"><div class="k">Pattern Columns</div><div class="v">{len(pattern_columns):,}</div></div>
        <div class="card"><div class="k">Spike Rows</div><div class="v">{spike_rows:,}</div></div>
        <div class="card"><div class="k">Spike Columns</div><div class="v">{len(spike_columns):,}</div></div>
        <div class="card"><div class="k">Spend R2</div><div class="v">{metric_map.get('target_spend', 0.0):.3f}</div></div>
        <div class="card"><div class="k">Revenue R2</div><div class="v">{metric_map.get('target_tracker_revenue', 0.0):.3f}</div></div>
      </div>
      <div class="links">
        <a href="{OUTPUT_BOSS_HTML.name}">Boss page</a>
        <a href="{OUTPUT_FULL_HTML.name}">Focused full page</a>
        <a href="{OUTPUT_HTML.name}">Lite page</a>
      </div>
    </section>

    <section class="panel">
      <h2>Load Data</h2>
      <div class="toolbar">
        <div>
          <label for="source">CSV source</label>
          <select id="source">
            <option value="pattern">Detailed pattern CSV</option>
            <option value="spike">Spike-aware CSV</option>
          </select>
        </div>
        <div>
          <label for="search">Search</label>
          <input id="search" placeholder="Search IDs, windows, dates, notes">
        </div>
        <div>
          <label for="split">Split</label>
          <select id="split"><option value="">All Splits</option></select>
        </div>
        <div>
          <label for="risk">Risk</label>
          <select id="risk"><option value="">All Risk Flags</option></select>
        </div>
        <div>
          <label for="limit">Rows shown</label>
          <select id="limit">
            <option value="100">100</option>
            <option value="250" selected>250</option>
            <option value="500">500</option>
            <option value="1000">1000</option>
            <option value="2000">2000</option>
          </select>
        </div>
      </div>
      <div class="toolbar" style="grid-template-columns: 2fr 1fr 1fr;">
        <div>
          <label for="sort">Sort by</label>
          <select id="sort"></select>
        </div>
        <div>
          <label>&nbsp;</label>
          <button id="load-auto">Load selected CSV</button>
        </div>
        <div>
          <label>&nbsp;</label>
          <button id="clear">Clear filters</button>
        </div>
      </div>
      <div class="status">
        <span id="status">Loading embedded detailed CSV automatically. The first load can take a little time because the CSV is decompressed in the browser.</span>
        <span id="match-count"></span>
      </div>
    </section>

    <section class="panel">
      <h2>Available Columns</h2>
      <div id="column-box" class="column-box"></div>
    </section>

    <section class="panel">
      <h2>All-Column Table</h2>
      <div class="table-wrap">
        <table id="wide-table">
          <thead></thead>
          <tbody></tbody>
        </table>
      </div>
    </section>
  </div>

  <script id="source-data" type="application/json">{json.dumps(sources)}</script>
  <script>
    const sources = JSON.parse(document.getElementById('source-data').textContent);
    let columns = [];
    let rows = [];

    const sourceEl = document.getElementById('source');
    const searchEl = document.getElementById('search');
    const splitEl = document.getElementById('split');
    const riskEl = document.getElementById('risk');
    const limitEl = document.getElementById('limit');
    const sortEl = document.getElementById('sort');
    const loadAutoEl = document.getElementById('load-auto');
    const clearEl = document.getElementById('clear');
    const statusEl = document.getElementById('status');
    const matchCountEl = document.getElementById('match-count');
    const columnBoxEl = document.getElementById('column-box');
    const thead = document.querySelector('#wide-table thead');
    const tbody = document.querySelector('#wide-table tbody');

    function parseCSVLine(line) {{
      const out = [];
      let current = '';
      let inQuotes = false;
      for (let i = 0; i < line.length; i++) {{
        const ch = line[i];
        const next = line[i + 1];
        if (ch === '"' && inQuotes && next === '"') {{
          current += '"';
          i += 1;
        }} else if (ch === '"') {{
          inQuotes = !inQuotes;
        }} else if (ch === ',' && !inQuotes) {{
          out.push(current);
          current = '';
        }} else {{
          current += ch;
        }}
      }}
      out.push(current);
      return out;
    }}

    function parseCSV(text) {{
      const lines = text.replace(/^\\uFEFF/, '').split(/\\r?\\n/).filter(line => line.length > 0);
      if (!lines.length) return {{ columns: [], rows: [] }};
      const parsedColumns = parseCSVLine(lines[0]);
      const parsedRows = lines.slice(1).map(line => {{
        const values = parseCSVLine(line);
        while (values.length < parsedColumns.length) values.push('');
        return values.slice(0, parsedColumns.length);
      }});
      return {{ columns: parsedColumns, rows: parsedRows }};
    }}

    function base64ToBytes(base64) {{
      const chunkSize = 32768;
      const chunks = [];
      for (let offset = 0; offset < base64.length; offset += chunkSize) {{
        const chunk = base64.slice(offset, offset + chunkSize);
        const binary = atob(chunk);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        chunks.push(bytes);
      }}
      const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
      const out = new Uint8Array(total);
      let cursor = 0;
      for (const chunk of chunks) {{
        out.set(chunk, cursor);
        cursor += chunk.length;
      }}
      return out;
    }}

    async function decompressGzipBase64(base64) {{
      if (!('DecompressionStream' in window)) {{
        throw new Error('This browser does not support built-in gzip decompression. Please open in a recent Chrome or Edge browser.');
      }}
      const bytes = base64ToBytes(base64);
      const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
      return await new Response(stream).text();
    }}

    function isNumericLike(value) {{
      if (value === '' || value === null || value === undefined) return false;
      return !Number.isNaN(Number(value));
    }}

    function formatValue(value) {{
      if (value === null || value === undefined || value === '') return '';
      if (isNumericLike(value)) {{
        const n = Number(value);
        return Number.isInteger(n) ? String(n) : n.toFixed(4);
      }}
      return String(value);
    }}

    function htmlEscape(value) {{
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;');
    }}

    function selectedSource() {{
      return sources[sourceEl.value];
    }}

    function setKnownColumns() {{
      const source = selectedSource();
      columns = source.columns;
      rows = [];
      renderColumnBox();
      buildHeader();
      initControls();
      tbody.innerHTML = '';
      matchCountEl.textContent = '';
      statusEl.textContent = `${{source.label}} selected. ${{source.columns.length.toLocaleString()}} columns are embedded and available.`;
    }}

    function renderColumnBox() {{
      columnBoxEl.textContent = columns.join(', ');
    }}

    function buildHeader() {{
      thead.innerHTML = '<tr>' + columns.map(col => `<th title="${{htmlEscape(col)}}">${{htmlEscape(col)}}</th>`).join('') + '</tr>';
    }}

    function initControls() {{
      const splitIdx = columns.indexOf('model_dataset_split');
      const riskIdx = columns.indexOf('spike_risk_flag');
      const splits = rows.length && splitIdx >= 0 ? [...new Set(rows.map(r => String(r[splitIdx] || '')).filter(Boolean))].sort() : [];
      const risks = rows.length && riskIdx >= 0 ? [...new Set(rows.map(r => String(r[riskIdx] || '')).filter(Boolean))].sort() : [];
      splitEl.innerHTML = '<option value="">All Splits</option>' + splits.map(v => `<option value="${{htmlEscape(v)}}">${{htmlEscape(v)}}</option>`).join('');
      riskEl.innerHTML = '<option value="">All Risk Flags</option>' + risks.map(v => `<option value="${{htmlEscape(v)}}">${{htmlEscape(v)}}</option>`).join('');

      const preferred = [
        'next_6h_abs_gap_tracker_revenue',
        'next_6h_gap_corrected_tracker_revenue',
        'next_6h_abs_gap_tracker_conversions',
        'next_6h_gap_corrected_tracker_conversions',
        'next_6h_abs_gap_spend',
        'spike_risk_score'
      ].filter(col => columns.includes(col));
      const rest = columns.filter(col => !preferred.includes(col));
      sortEl.innerHTML = preferred.concat(rest).map(col => `<option value="${{htmlEscape(col)}}">${{htmlEscape(col)}}</option>`).join('');
    }}

    function render() {{
      if (!columns.length) return;
      const q = searchEl.value.trim().toLowerCase();
      const split = splitEl.value;
      const risk = riskEl.value;
      const limit = Number(limitEl.value || 250);
      const splitIdx = columns.indexOf('model_dataset_split');
      const riskIdx = columns.indexOf('spike_risk_flag');
      const sortIdx = columns.indexOf(sortEl.value);

      let filtered = rows.filter(row => {{
        if (split && splitIdx >= 0 && String(row[splitIdx] || '') !== split) return false;
        if (risk && riskIdx >= 0 && String(row[riskIdx] || '') !== risk) return false;
        if (q && !row.join(' ').toLowerCase().includes(q)) return false;
        return true;
      }});

      if (sortIdx >= 0) {{
        filtered.sort((a, b) => {{
          const av = a[sortIdx];
          const bv = b[sortIdx];
          if (isNumericLike(av) || isNumericLike(bv)) {{
            return Math.abs(Number(bv || 0)) - Math.abs(Number(av || 0));
          }}
          return String(av || '').localeCompare(String(bv || ''));
        }});
      }}

      matchCountEl.textContent = `${{filtered.length.toLocaleString()}} matching rows. Showing ${{Math.min(filtered.length, limit).toLocaleString()}}.`;
      filtered = filtered.slice(0, limit);

      tbody.innerHTML = filtered.map(row => '<tr>' + row.map(value => {{
        const text = htmlEscape(formatValue(value));
        const cls = isNumericLike(value) ? ' class="num"' : '';
        return `<td${{cls}} title="${{text}}">${{text}}</td>`;
      }}).join('') + '</tr>').join('');
    }}

    async function loadSelectedCSV() {{
      const source = selectedSource();
      statusEl.textContent = `Decompressing embedded ${{source.label}}...`;
      try {{
        const text = await decompressGzipBase64(source.gzipBase64);
        const parsed = parseCSV(text);
        columns = parsed.columns;
        rows = parsed.rows;
        renderColumnBox();
        buildHeader();
        initControls();
        render();
        statusEl.textContent = `Loaded embedded ${{source.label}} with ${{rows.length.toLocaleString()}} rows and ${{columns.length.toLocaleString()}} columns.`;
      }} catch (err) {{
        statusEl.innerHTML = `<span class="warn">${{htmlEscape(err.message || String(err))}}</span>`;
      }}
    }}

    sourceEl.addEventListener('change', () => {{
      setKnownColumns();
      loadSelectedCSV();
    }});
    loadAutoEl.addEventListener('click', loadSelectedCSV);
    clearEl.addEventListener('click', () => {{
      searchEl.value = '';
      splitEl.value = '';
      riskEl.value = '';
      render();
    }});
    [searchEl, splitEl, riskEl, limitEl, sortEl].forEach(el => {{
      el.addEventListener('input', render);
      el.addEventListener('change', render);
    }});

    setKnownColumns();
    loadSelectedCSV();
  </script>
</body>
</html>"""


def main() -> None:
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    pattern_columns = list(pd.read_csv(PATTERN_CSV, nrows=0, low_memory=False).columns)
    metrics = pd.read_csv(METRICS_CSV, low_memory=False)
    spike_summary = pd.read_csv(SUMMARY_CSV, low_memory=False) if SUMMARY_CSV.exists() else pd.DataFrame()

    risk_counts = df["spike_risk_flag"].astype(str).value_counts().reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0).to_dict()
    split_counts = df["model_dataset_split"].astype(str).value_counts().to_dict()
    metric_test = metrics[metrics["split"].eq("test")].copy()
    metric_map = {row["target"]: float(row["r2"]) for _, row in metric_test.iterrows()}

    top_revenue = df.assign(abs_gap=num(df, "next_6h_gap_tracker_revenue").abs()).sort_values("abs_gap", ascending=False).head(12)
    top_conv = df.assign(abs_gap=num(df, "next_6h_gap_tracker_conversions").abs()).sort_values("abs_gap", ascending=False).head(12)

    lite_rows = build_rows_payload(df, limit=1200)
    full_rows = build_rows_payload(df, limit=None)

    lite_html = build_main_html(
        df=df,
        metric_map=metric_map,
        risk_counts=risk_counts,
        split_counts=split_counts,
        top_revenue=top_revenue,
        top_conv=top_conv,
        spike_summary=spike_summary,
        rows_payload=lite_rows,
        title="Adunbox 6H Prediction Review",
        description="Visual review page for the final 6-hour model outputs, actual next-6h values, corrected spike-aware predictions, and a lightweight interactive slice of row-level inspection.",
        detail_note="This page embeds a lightweight 1,200-row slice for fast browsing; use the full page or linked CSVs for the complete 30,439-row dataset.",
        output_name=OUTPUT_HTML.name,
    )
    full_html = build_main_html(
        df=df,
        metric_map=metric_map,
        risk_counts=risk_counts,
        split_counts=split_counts,
        top_revenue=top_revenue,
        top_conv=top_conv,
        spike_summary=spike_summary,
        rows_payload=full_rows,
        title="Adunbox 6H Prediction Review Full",
        description="Full embedded row-level page for the final 6-hour model outputs. All modeled rows are included directly in this page for deep review.",
        detail_note="This page embeds the complete 30,439-row dataset. Filtering can take a moment because every row is loaded directly in the browser.",
        output_name=OUTPUT_FULL_HTML.name,
    )
    boss_html = build_boss_html(
        df=df,
        metric_map=metric_map,
        risk_counts=risk_counts,
        top_revenue=top_revenue,
        top_conv=top_conv,
        spike_summary=spike_summary,
    )
    all_columns_html = build_all_columns_loader_html(
        pattern_columns=pattern_columns,
        spike_columns=list(df.columns),
        pattern_rows=30439,
        spike_rows=len(df),
        pattern_gzip_b64=gzip_b64(PATTERN_CSV),
        spike_gzip_b64=gzip_b64(INPUT_CSV),
        metric_map=metric_map,
    )

    OUTPUT_HTML.write_text(lite_html, encoding="utf-8")
    OUTPUT_FULL_HTML.write_text(full_html, encoding="utf-8")
    OUTPUT_BOSS_HTML.write_text(boss_html, encoding="utf-8")
    OUTPUT_ALL_COLUMNS_HTML.write_text(all_columns_html, encoding="utf-8")

    print(OUTPUT_HTML)
    print(OUTPUT_FULL_HTML)
    print(OUTPUT_BOSS_HTML)
    print(OUTPUT_ALL_COLUMNS_HTML)


if __name__ == "__main__":
    main()
