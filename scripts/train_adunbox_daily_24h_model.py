from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import train_adunbox_local_midnight_sequence_models as base_seq


BASE_DIR = Path(os.getenv("ADUNBOX_PROJECT_DIR", Path(__file__).resolve().parents[1]))
DEFAULT_DAILY_INPUT_PATH = Path(os.getenv("ADUNBOX_DAILY_INPUT", BASE_DIR / "data" / "adunbox_daily_breakdown_kpis.csv"))
DAILY_FALLBACK_PATH = BASE_DIR / "adunbox_ad_daily_from_hourly_full.csv"
MODEL_DIR = BASE_DIR / "models" / "adunbox_daily_24h_histgb"
METRICS_PATH = BASE_DIR / "adunbox_daily_24h_histgb__metrics.csv"
SUMMARY_PATH = BASE_DIR / "adunbox_daily_24h_histgb__summary.txt"
PREDICTION_EXAMPLES_PATH = BASE_DIR / "adunbox_daily_24h_histgb__prediction_examples.csv"

RAW_TARGETS = [
    "spend",
    "impressions",
    "inline_link_clicks",
    "tracker_conversions",
    "tracker_revenue",
]

ENTITY_COLS = ["account_id", "campaign_id", "adset_id", "ad_id"]
HOURLY_USECOLS = ["date", "timezone", *ENTITY_COLS, *RAW_TARGETS]

TRAIN_END = pd.Timestamp("2026-04-30")
VALID_START = pd.Timestamp("2026-05-01")
VALID_END = pd.Timestamp("2026-05-07")
TEST_START = pd.Timestamp("2026-05-08")
TEST_END = pd.Timestamp("2026-05-12")


def normalize_id(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    return text.str.replace(r"\.0$", "", regex=True)


def safe_div(numer: pd.Series | np.ndarray, denom: pd.Series | np.ndarray, multiplier: float = 1.0):
    numer_s = pd.Series(numer, copy=False)
    denom_s = pd.Series(denom, copy=False)
    out = pd.Series(np.zeros(len(numer_s), dtype=np.float32), index=numer_s.index)
    mask = denom_s != 0
    out.loc[mask] = (numer_s.loc[mask] / denom_s.loc[mask]) * multiplier
    return out.astype("float32")


def has_daily_rows(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 400:
        return False
    try:
        return len(pd.read_csv(path, nrows=5, low_memory=False)) > 0
    except Exception:
        return False


def to_local_date(utc_ts: pd.Series, timezone: pd.Series) -> pd.Series:
    local_date = pd.Series(pd.NaT, index=utc_ts.index, dtype="datetime64[ns]")
    tz_values = timezone.fillna("").astype(str)
    for tz_name, idx in tz_values.groupby(tz_values).groups.items():
        utc_slice = utc_ts.loc[idx]
        try:
            local_slice = utc_slice.dt.tz_convert(str(tz_name).strip()) if str(tz_name).strip() else utc_slice
        except Exception:
            local_slice = utc_slice
        local_date.loc[idx] = local_slice.dt.tz_localize(None).dt.normalize().to_numpy()
    return local_date


def build_daily_from_hourly(output_path: Path) -> pd.DataFrame:
    grouped_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(base_seq.HOURLY_INPUT_PATH, usecols=HOURLY_USECOLS, chunksize=250_000, low_memory=False):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce", utc=True)
        chunk = chunk[chunk["date"].notna()].copy()
        for col in ENTITY_COLS:
            chunk[col] = normalize_id(chunk[col])
        chunk["timezone"] = chunk["timezone"].fillna("").astype(str)
        for col in RAW_TARGETS:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0.0).astype("float32")
        chunk["local_date"] = to_local_date(chunk["date"], chunk["timezone"])
        part = (
            chunk.groupby(["local_date", "timezone", *ENTITY_COLS], as_index=False)[RAW_TARGETS]
            .sum()
        )
        grouped_parts.append(part)

    daily = pd.concat(grouped_parts, ignore_index=True)
    daily = (
        daily.groupby(["local_date", "timezone", *ENTITY_COLS], as_index=False)[RAW_TARGETS]
        .sum()
        .sort_values(["ad_id", "local_date"])
        .reset_index(drop=True)
    )
    daily.to_csv(output_path, index=False)
    return daily


def load_daily(force_rebuild_from_hourly: bool, daily_input_path: Path) -> tuple[pd.DataFrame, str]:
    if not force_rebuild_from_hourly and has_daily_rows(daily_input_path):
        usecols = ["entity_type", "date", "timezone", *ENTITY_COLS, *RAW_TARGETS]
        daily = pd.read_csv(daily_input_path, usecols=lambda c: c in usecols, low_memory=False)
        if "entity_type" in daily.columns:
            daily = daily[daily["entity_type"].astype(str).str.lower().eq("ad")].copy()
            daily = daily.drop(columns=["entity_type"])
        daily = daily.rename(columns={"date": "local_date"})
        source = str(daily_input_path)
    elif DAILY_FALLBACK_PATH.exists() and not force_rebuild_from_hourly:
        daily = pd.read_csv(DAILY_FALLBACK_PATH, low_memory=False)
        source = str(DAILY_FALLBACK_PATH)
    else:
        daily = build_daily_from_hourly(DAILY_FALLBACK_PATH)
        source = f"{DAILY_FALLBACK_PATH} (built from full hourly data)"

    daily["local_date"] = pd.to_datetime(daily["local_date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    daily = daily[daily["local_date"].notna()].copy()
    for col in ENTITY_COLS:
        daily[col] = normalize_id(daily[col])
    daily["timezone"] = daily.get("timezone", "").fillna("").astype(str)
    for col in RAW_TARGETS:
        daily[col] = pd.to_numeric(daily.get(col, 0.0), errors="coerce").fillna(0.0).astype("float32")
    return daily.sort_values(["ad_id", "local_date"]).reset_index(drop=True), source


def add_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["kpi_ctr"] = safe_div(out["inline_link_clicks"], out["impressions"], 100.0)
    out["kpi_cpc"] = safe_div(out["spend"], out["inline_link_clicks"])
    out["kpi_cpm"] = safe_div(out["spend"], out["impressions"], 1000.0)
    out["kpi_cvr"] = safe_div(out["tracker_conversions"], out["inline_link_clicks"], 100.0)
    out["kpi_roas"] = safe_div(out["tracker_revenue"], out["spend"])
    out["kpi_profit"] = (out["tracker_revenue"] - out["spend"]).astype("float32")
    return out


def densify_by_ad(daily: pd.DataFrame) -> pd.DataFrame:
    dense_parts: list[pd.DataFrame] = []
    for ad_id, grp in daily.groupby("ad_id", sort=False):
        grp = grp.sort_values("local_date").drop_duplicates(["local_date"], keep="last")
        date_index = pd.date_range(grp["local_date"].min(), grp["local_date"].max(), freq="1D")
        dense = pd.DataFrame({"local_date": date_index})
        dense = dense.merge(grp, on="local_date", how="left")
        dense["ad_id"] = str(ad_id)
        for col in ["timezone", "account_id", "campaign_id", "adset_id"]:
            dense[col] = dense[col].ffill().bfill().fillna("")
        dense[RAW_TARGETS] = dense[RAW_TARGETS].fillna(0.0)
        dense_parts.append(dense)
    return pd.concat(dense_parts, ignore_index=True)


def add_features_and_targets(daily: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = add_kpis(densify_by_ad(daily))
    out = out.sort_values(["ad_id", "local_date"]).reset_index(drop=True)
    grouped = out.groupby("ad_id", sort=False)

    day_of_week = out["local_date"].dt.dayofweek.astype("int16")
    feature_data: dict[str, pd.Series | np.ndarray] = {
        "day_of_week": day_of_week,
        "dow_sin": np.sin(2.0 * np.pi * day_of_week / 7.0).astype("float32"),
        "dow_cos": np.cos(2.0 * np.pi * day_of_week / 7.0).astype("float32"),
        "days_active": (grouped.cumcount() + 1).astype("int32"),
        "cum_spend": grouped["spend"].cumsum().astype("float32"),
        "cum_clicks": grouped["inline_link_clicks"].cumsum().astype("float32"),
        "cum_conversions": grouped["tracker_conversions"].cumsum().astype("float32"),
        "cum_revenue": grouped["tracker_revenue"].cumsum().astype("float32"),
        "cum_profit": grouped["kpi_profit"].cumsum().astype("float32"),
    }
    feature_cols = list(feature_data.keys())
    base_cols = [*RAW_TARGETS, "kpi_ctr", "kpi_cpc", "kpi_cpm", "kpi_cvr", "kpi_roas", "kpi_profit"]

    raw_set = set(RAW_TARGETS)
    for col in base_cols:
        for lag in [1, 2, 3, 7]:
            name = f"{col}_lag_{lag}d"
            feature_data[name] = grouped[col].shift(lag).fillna(0.0).astype("float32")
            feature_cols.append(name)
        shifted = grouped[col].shift(1)
        shifted_grouped = shifted.groupby(out["ad_id"], sort=False)
        for window in [3, 7, 14]:
            mean_name = f"{col}_roll_mean_{window}d"
            feature_data[mean_name] = (
                shifted_grouped.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0.0).astype("float32")
            )
            feature_cols.append(mean_name)
            if col in raw_set:
                sum_name = f"{col}_roll_sum_{window}d"
                feature_data[sum_name] = (
                    shifted_grouped.rolling(window, min_periods=1).sum().reset_index(level=0, drop=True).fillna(0.0).astype("float32")
                )
                feature_cols.append(sum_name)

    target_data: dict[str, pd.Series] = {}
    for target in RAW_TARGETS:
        target_data[f"target_24h_{target}"] = out[target].astype("float32")

    target_data["target_24h_roas"] = out["kpi_roas"].astype("float32")
    target_data["target_24h_profit"] = out["kpi_profit"].astype("float32")
    target_data["target_24h_ctr"] = out["kpi_ctr"].astype("float32")
    target_data["target_24h_cvr"] = out["kpi_cvr"].astype("float32")
    target_data["target_24h_cpc"] = out["kpi_cpc"].astype("float32")
    target_data["target_24h_cpm"] = out["kpi_cpm"].astype("float32")

    feature_frame = pd.DataFrame(feature_data, index=out.index)
    target_frame = pd.DataFrame(target_data, index=out.index)
    out = pd.concat([out, feature_frame, target_frame], axis=1)
    out = out[out["days_active"] >= 8].replace([np.inf, -np.inf], np.nan).fillna(0.0).copy()
    return out, feature_cols


def build_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_iter=80,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        max_bins=64,
        l2_regularization=0.02,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=10,
        random_state=42,
    )


def split_by_fixed_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["local_date"] <= TRAIN_END].copy()
    valid = df[(df["local_date"] >= VALID_START) & (df["local_date"] <= VALID_END)].copy()
    test = df[(df["local_date"] >= TEST_START) & (df["local_date"] <= TEST_END)].copy()
    if len(train) and len(valid) and len(test):
        return train, valid, test

    unique_dates = np.array(sorted(df["local_date"].unique()))
    train_end = int(len(unique_dates) * 0.70)
    valid_end = int(len(unique_dates) * 0.85)
    return (
        df[df["local_date"] <= pd.Timestamp(unique_dates[train_end - 1])].copy(),
        df[(df["local_date"] > pd.Timestamp(unique_dates[train_end - 1])) & (df["local_date"] <= pd.Timestamp(unique_dates[valid_end - 1]))].copy(),
        df[df["local_date"] > pd.Timestamp(unique_dates[valid_end - 1])].copy(),
    )


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(target: str, split: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, object]:
    return {
        "target": target,
        "split": split,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_target(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], target: str) -> tuple[object, list[dict[str, object]], np.ndarray]:
    X_train = train[feature_cols].astype("float32")
    X_valid = valid[feature_cols].astype("float32")
    X_test = test[feature_cols].astype("float32")
    y_train = pd.to_numeric(train[target], errors="coerce").fillna(0.0).astype("float32")
    y_valid = pd.to_numeric(valid[target], errors="coerce").fillna(0.0).astype("float32")
    y_test = pd.to_numeric(test[target], errors="coerce").fillna(0.0).astype("float32")

    model = build_model()
    model.fit(X_train, np.log1p(np.maximum(0.0, y_train)))
    pred_valid = np.expm1(model.predict(X_valid))
    pred_test = np.expm1(model.predict(X_test))
    pred_valid = np.maximum(0.0, pred_valid)
    pred_test = np.maximum(0.0, pred_test)
    rows = [evaluate(target, "valid", y_valid, pred_valid), evaluate(target, "test", y_test, pred_test)]
    return model, rows, pred_test


def derive_kpis(preds: pd.DataFrame) -> pd.DataFrame:
    out = preds.copy()
    out["pred_24h_roas"] = safe_div(out["pred_24h_tracker_revenue"], out["pred_24h_spend"])
    out["pred_24h_profit"] = (out["pred_24h_tracker_revenue"] - out["pred_24h_spend"]).astype("float32")
    out["pred_24h_ctr"] = safe_div(out["pred_24h_inline_link_clicks"], out["pred_24h_impressions"], 100.0)
    out["pred_24h_cvr"] = safe_div(out["pred_24h_tracker_conversions"], out["pred_24h_inline_link_clicks"], 100.0)
    out["pred_24h_cpc"] = safe_div(out["pred_24h_spend"], out["pred_24h_inline_link_clicks"])
    out["pred_24h_cpm"] = safe_div(out["pred_24h_spend"], out["pred_24h_impressions"], 1000.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train daily-data 24h ad forecast model.")
    parser.add_argument("--daily-input", type=Path, default=DEFAULT_DAILY_INPUT_PATH, help="Daily breakdown CSV to use for 24h training.")
    parser.add_argument("--rebuild-daily-from-hourly", action="store_true", help="Ignore cached/found daily rows and rebuild ad daily table from full hourly data.")
    args = parser.parse_args()

    daily, source = load_daily(force_rebuild_from_hourly=args.rebuild_daily_from_hourly, daily_input_path=args.daily_input)
    dataset, feature_cols = add_features_and_targets(daily)
    train, valid, test = split_by_fixed_dates(dataset)
    if train.empty or valid.empty or test.empty:
        raise RuntimeError(f"Bad split sizes: train={len(train):,}, valid={len(valid):,}, test={len(test):,}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, object]] = []
    test_preds = pd.DataFrame(index=test.index)
    target_cols = [f"target_24h_{target}" for target in RAW_TARGETS]
    for target in target_cols:
        model, rows, pred_test = train_target(train, valid, test, feature_cols, target)
        joblib.dump(model, MODEL_DIR / f"{target}.joblib")
        metrics.extend(rows)
        raw_name = target.replace("target_24h_", "")
        test_preds[f"pred_24h_{raw_name}"] = pred_test.astype("float32")

    pred_kpis = derive_kpis(test_preds)
    for target, pred_col in [
        ("target_24h_roas", "pred_24h_roas"),
        ("target_24h_profit", "pred_24h_profit"),
        ("target_24h_ctr", "pred_24h_ctr"),
        ("target_24h_cvr", "pred_24h_cvr"),
        ("target_24h_cpc", "pred_24h_cpc"),
        ("target_24h_cpm", "pred_24h_cpm"),
    ]:
        metrics.append(evaluate(target, "test", test[target].astype("float32"), pred_kpis[pred_col].to_numpy()))

    pd.DataFrame(metrics).to_csv(METRICS_PATH, index=False)
    joblib.dump({"feature_cols": feature_cols, "raw_targets": RAW_TARGETS, "source": source}, MODEL_DIR / "metadata.joblib")

    examples = test[["local_date", "timezone", *ENTITY_COLS, *[f"target_24h_{c}" for c in RAW_TARGETS], "target_24h_roas", "target_24h_profit"]].copy()
    for col in pred_kpis.columns:
        examples[col] = pred_kpis[col].to_numpy()
    examples["abs_revenue_gap"] = (examples["pred_24h_tracker_revenue"] - examples["target_24h_tracker_revenue"]).abs()
    examples["abs_roas_gap"] = (examples["pred_24h_roas"] - examples["target_24h_roas"]).abs()
    examples.sort_values(["abs_roas_gap", "abs_revenue_gap"]).head(25).to_csv(PREDICTION_EXAMPLES_PATH, index=False)

    test_metrics = pd.DataFrame(metrics)
    test_metrics = test_metrics[test_metrics["split"] == "test"]
    lines = [
        "Adunbox Daily 24h HistGradientBoosting Model",
        "",
        f"Daily source: {source}",
        f"Rows after feature build: {len(dataset):,}",
        f"Train rows: {len(train):,}",
        f"Valid rows: {len(valid):,}",
        f"Test rows: {len(test):,}",
        f"Date range: {dataset['local_date'].min().date()} -> {dataset['local_date'].max().date()}",
        f"Feature count: {len(feature_cols):,}",
        "",
        "Test R2:",
    ]
    for row in test_metrics.itertuples(index=False):
        lines.append(f"- {row.target}: {float(row.r2):.6f}")
    lines.extend([
        "",
        "Implementation note:",
        "- This model predicts a full local-day total from prior daily history.",
        "- It is intended to replace/compete with weak hourly-sequence 24h revenue predictions.",
        "- The 6h model should remain hourly-anchor or multi-anchor because the target is intraday.",
    ])
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
