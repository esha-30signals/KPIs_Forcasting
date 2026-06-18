import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from ._config import ForecastConfig

ROOT = Path(__file__).resolve().parents[2]


# ── SQL rendering ──────────────────────────────────────────────────────────────

def _safe_table_identifier(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("Database table name cannot be empty")
    parts = text.split(".")
    for part in parts:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
            raise ValueError(f"Unsafe database table identifier: {value!r}")
    return ".".join(parts)


def _table_context(config: ForecastConfig) -> dict[str, str]:
    p = config.table_prefix
    return {
        "traffic_source_reports_table": _safe_table_identifier(f"{p}traffic_source_reports"),
        "traffic_source_accounts_table": _safe_table_identifier(f"{p}traffic_source_accounts"),
        "traffic_source_campaigns_table": _safe_table_identifier(f"{p}traffic_source_campaigns"),
        "traffic_source_adsets_table": _safe_table_identifier(f"{p}traffic_source_adsets"),
        "traffic_source_ads_table": _safe_table_identifier(f"{p}traffic_source_ads"),
        "daily_breakdown_kpis_table": _safe_table_identifier(f"{p}daily_breakdown_kpis"),
        "forecast_eligible_ads_table": _safe_table_identifier(f"{p}forecast_eligible_ads"),
    }


def _active_hierarchy_filter_6h(tables: dict[str, str]) -> str:
    return f"""
      UPPER(COALESCE(a.status, '')) = 'ACTIVE'
      AND EXISTS (
          SELECT 1 FROM {tables["traffic_source_campaigns_table"]} c
          WHERE c.id = r.campaign_id AND UPPER(COALESCE(c.status, '')) = 'ACTIVE'
      )
      AND EXISTS (
          SELECT 1 FROM {tables["traffic_source_adsets_table"]} s
          WHERE s.id = r.adset_id AND UPPER(COALESCE(s.status, '')) = 'ACTIVE'
      )
      AND EXISTS (
          SELECT 1 FROM {tables["traffic_source_ads_table"]} ad
          WHERE ad.id = r.ad_id AND UPPER(COALESCE(ad.status, '')) = 'ACTIVE'
      )
    """.strip()


def _active_hierarchy_filter_24h(tables: dict[str, str]) -> str:
    return f"""
      EXISTS (
          SELECT 1 FROM {tables["traffic_source_accounts_table"]} a
          WHERE a.id = d.account_id AND UPPER(COALESCE(a.status, '')) = 'ACTIVE'
      )
      AND EXISTS (
          SELECT 1 FROM {tables["traffic_source_campaigns_table"]} c
          WHERE c.id = d.campaign_id AND UPPER(COALESCE(c.status, '')) = 'ACTIVE'
      )
      AND EXISTS (
          SELECT 1 FROM {tables["traffic_source_adsets_table"]} s
          WHERE s.id = d.adset_id AND UPPER(COALESCE(s.status, '')) = 'ACTIVE'
      )
      AND EXISTS (
          SELECT 1 FROM {tables["traffic_source_ads_table"]} ad
          WHERE ad.id = d.ad_id AND UPPER(COALESCE(ad.status, '')) = 'ACTIVE'
      )
    """.strip()


def render_6h_sql(sql_template: str, config: ForecastConfig) -> str:
    tables = _table_context(config)
    ctx = dict(tables)
    ctx["reports_revenue_expr"] = f"r.{config.revenue_col_6h}"
    ctx["reports_conversions_expr"] = f"r.{config.conversions_col_6h}"
    ctx["active_hierarchy_filter_6h"] = _active_hierarchy_filter_6h(tables)
    return sql_template.format(**ctx)


def render_24h_sql(sql_template: str, config: ForecastConfig) -> str:
    tables = _table_context(config)
    ctx = dict(tables)
    ctx["daily_revenue_expr"] = f"d.{config.revenue_col_24h}"
    ctx["daily_conversions_expr"] = f"d.{config.conversions_col_24h}"
    ctx["active_hierarchy_filter_24h"] = _active_hierarchy_filter_24h(tables)
    return sql_template.format(**ctx)


# ── Subprocess ─────────────────────────────────────────────────────────────────

def _run_python(script: Path, *args: str, env: dict[str, str] | None = None) -> None:
    cmd = [sys.executable, str(script), *args]
    result = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nSTDOUT:\n"
            + result.stdout
            + "\n\nSTDERR:\n"
            + result.stderr
        )


def _require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


# ── Datetime helpers ───────────────────────────────────────────────────────────

def _parse_anchor_date(value: str | None):
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="raise")
    if getattr(parsed, "tzinfo", None) is None:
        return parsed.to_pydatetime().replace(tzinfo=timezone.utc)
    return parsed.to_pydatetime()


def _db_datetime(value):
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if hasattr(value, "isoformat"):
        if getattr(value, "tzinfo", None) is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    try:
        parsed = pd.to_datetime(value, errors="raise")
    except Exception:
        return None
    dt = parsed.to_pydatetime()
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _local_datetime(value) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value, errors="raise")
    except Exception:
        return None
    dt = parsed.to_pydatetime()
    if getattr(dt, "tzinfo", None) is not None:
        dt = dt.replace(tzinfo=None)
    return dt


def _local_date(value):
    dt = _local_datetime(value)
    return dt.date() if dt is not None else None


def _account_local_datetime(value, tz_name: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value, errors="raise")
    except Exception:
        return None
    dt = parsed.to_pydatetime()
    if getattr(dt, "tzinfo", None) is None:
        return dt
    if not tz_name:
        return dt.replace(tzinfo=None)
    try:
        return dt.astimezone(ZoneInfo(str(tz_name))).replace(tzinfo=None)
    except (ZoneInfoNotFoundError, ValueError):
        return dt.replace(tzinfo=None)


def _forecast_window(anchor, horizon: str) -> tuple[object | None, object | None]:
    start = _db_datetime(anchor)
    if start is None:
        return None, None
    hours = 24 if horizon == "24h" else 6 if horizon == "6h" else None
    if hours is None:
        return start, None
    return start, start + pd.Timedelta(hours=hours)


# ── JSON / value helpers ───────────────────────────────────────────────────────

def _json_safe(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _first_existing(row: pd.Series, names: list[str]):
    for name in names:
        if name in row and not pd.isna(row[name]):
            return row[name]
    return None


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


# ── Forecast result processing ─────────────────────────────────────────────────

def _derive_ratio_metrics(results: dict) -> dict:
    spend = _safe_float(results.get("spend"))
    impressions = _safe_float(results.get("impressions"))
    clicks = _safe_float(results.get("clicks"))
    conversions = _safe_float(results.get("conversions"))
    revenue = _safe_float(results.get("revenue"))
    if results.get("roas") is None and spend and spend > 0:
        results["roas"] = (revenue or 0.0) / spend
    if results.get("profit") is None and revenue is not None and spend is not None:
        results["profit"] = revenue - spend
    if results.get("ctr") is None and impressions and impressions > 0:
        results["ctr"] = (clicks or 0.0) / impressions * 100.0
    if results.get("cvr") is None and clicks and clicks > 0:
        results["cvr"] = (conversions or 0.0) / clicks * 100.0
    if results.get("cpc") is None and clicks and clicks > 0:
        results["cpc"] = (spend or 0.0) / clicks
    if results.get("cpm") is None and impressions and impressions > 0:
        results["cpm"] = (spend or 0.0) / impressions * 1000.0
    return results


def _forecast_status(row: pd.Series) -> str:
    existing = _first_existing(row, ["forecast_status"])
    if existing:
        return str(existing)
    benchmark_source = str(_first_existing(row, ["benchmark_source"]) or "").strip()
    confidence = str(_first_existing(row, ["forecast_confidence"]) or "").strip().upper()
    if benchmark_source == "insufficient_history":
        return "insufficient_history_monitoring"
    if benchmark_source and benchmark_source != "model_point":
        return "hierarchical_benchmark_forecast"
    if confidence == "LOW":
        return "low_confidence_forecast"
    return "model_forecast"


def _forecast_confidence_score(row: pd.Series) -> float | None:
    existing = _first_existing(row, ["forecast_confidence_score", "confidence_score"])
    if existing is not None and not pd.isna(existing):
        try:
            return float(existing)
        except (TypeError, ValueError):
            pass
    confidence = str(_first_existing(row, ["forecast_confidence"]) or "").strip().upper()
    return {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}.get(confidence)


def _forecast_results(payload: dict, horizon: str) -> dict:
    prefix = f"{horizon}_"
    raw_metrics = {
        "spend": f"recommended_{prefix}spend",
        "impressions": f"recommended_{prefix}impressions",
        "clicks": f"recommended_{prefix}inline_link_clicks",
        "conversions": f"recommended_{prefix}tracker_conversions",
        "revenue": f"recommended_{prefix}tracker_revenue",
        "roas": f"recommended_{prefix}roas",
        "profit": f"recommended_{prefix}profit",
        "ctr": f"recommended_{prefix}ctr",
        "cvr": f"recommended_{prefix}cvr",
        "cpc": f"recommended_{prefix}cpc",
        "cpm": f"recommended_{prefix}cpm",
    }
    fallback_metrics = {
        "spend": f"pred_{prefix}spend",
        "impressions": f"pred_{prefix}impressions",
        "clicks": f"pred_{prefix}inline_link_clicks",
        "conversions": f"pred_{prefix}tracker_conversions",
        "revenue": f"pred_{prefix}tracker_revenue",
        "roas": f"pred_{prefix}roas",
        "profit": f"pred_{prefix}profit",
        "ctr": f"pred_{prefix}ctr",
        "cvr": f"pred_{prefix}cvr",
        "cpc": f"pred_{prefix}cpc",
        "cpm": f"pred_{prefix}cpm",
    }
    results: dict[str, object] = {}
    for label, key in raw_metrics.items():
        value = payload.get(key)
        if value is None:
            value = payload.get(fallback_metrics[label])
        results[label] = _json_safe(value)
    for label in ["spend", "impressions", "inline_link_clicks", "tracker_conversions", "tracker_revenue", "roas"]:
        p10_key = f"pred_{prefix}{label}_p10"
        p50_key = f"pred_{prefix}{label}_p50"
        p90_key = f"pred_{prefix}{label}_p90"
        if p10_key in payload or p50_key in payload or p90_key in payload:
            out_label = (
                "clicks" if label == "inline_link_clicks"
                else "conversions" if label == "tracker_conversions"
                else "revenue" if label == "tracker_revenue"
                else label
            )
            results[f"{out_label}_range"] = {
                "p10": _json_safe(payload.get(p10_key)),
                "p50": _json_safe(payload.get(p50_key)),
                "p90": _json_safe(payload.get(p90_key)),
            }
    return _derive_ratio_metrics(results)


def _raw_forecast_results(payload: dict, horizon: str) -> dict:
    prefix = f"{horizon}_"
    raw_metrics = {
        "spend": f"pred_{prefix}spend",
        "impressions": f"pred_{prefix}impressions",
        "clicks": f"pred_{prefix}inline_link_clicks",
        "conversions": f"pred_{prefix}tracker_conversions",
        "revenue": f"pred_{prefix}tracker_revenue",
        "roas": f"pred_{prefix}roas",
        "profit": f"pred_{prefix}profit",
        "ctr": f"pred_{prefix}ctr",
        "cvr": f"pred_{prefix}cvr",
        "cpc": f"pred_{prefix}cpc",
        "cpm": f"pred_{prefix}cpm",
    }
    results = {label: _json_safe(payload.get(key)) for label, key in raw_metrics.items()}
    return _derive_ratio_metrics(results)


# ── I/O helpers ────────────────────────────────────────────────────────────────

def _write_single_horizon_sink_status(status_path: Path, status: dict) -> str:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return str(status_path)


def _cleanup_local_working_files(paths: list[Path]) -> list[str]:
    removed: list[str] = []
    for path in paths:
        try:
            if path.exists() and path.is_file():
                path.unlink()
                removed.append(str(path))
        except OSError:
            continue
    return removed
