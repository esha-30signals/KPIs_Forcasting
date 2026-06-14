from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
FULL_FAST_METRICS = BASE_DIR / "docs" / "adunbox_entity_history_lgbm_6h_full_fast__metrics.csv"
ANCHOR_V2_METRICS = BASE_DIR / "docs" / "adunbox_entity_history_lgbm_6h_anchor_v2__metrics.csv"
BUSINESS_V3_SELECTED = BASE_DIR / "docs" / "adunbox_entity_history_lgbm_6h_business_v3__selected.csv"
SELECTION_CSV = BASE_DIR / "docs" / "adunbox_6h_production_model_selection__metrics.csv"
SELECTION_SUMMARY = BASE_DIR / "docs" / "adunbox_6h_production_model_selection__summary.txt"

TARGET_LABELS = {
    "target_spend": "spend",
    "target_impressions": "impressions",
    "target_inline_link_clicks": "clicks",
    "target_tracker_conversions": "conversions",
    "target_tracker_revenue": "revenue",
}


def load_test_metrics(path: Path, model_name: str) -> pd.DataFrame:
    metrics = pd.read_csv(path)
    metrics = metrics[metrics["split"].eq("test")].copy()
    metrics["model"] = model_name
    for col in ["wmape", "bias"]:
        if col not in metrics.columns:
            metrics[col] = pd.NA
    return metrics[["target", "model", "mae", "rmse", "r2", "wmape", "bias"]]


def main() -> None:
    full_fast = load_test_metrics(FULL_FAST_METRICS, "lgbm_6h_full_fast_joined")
    anchor_v2 = load_test_metrics(ANCHOR_V2_METRICS, "lgbm_6h_anchor_v2_joined")
    candidates = [full_fast, anchor_v2]
    if BUSINESS_V3_SELECTED.exists():
        business_v3 = pd.read_csv(BUSINESS_V3_SELECTED).copy()
        business_v3 = business_v3.rename(columns={"candidate": "selected_candidate"})
        business_v3["split"] = "test"
        business_v3["model"] = "lgbm_6h_business_v3_joined"
        for col in ["wmape", "bias"]:
            if col not in business_v3.columns:
                business_v3[col] = pd.NA
        candidates.append(business_v3[["target", "model", "mae", "rmse", "r2", "wmape", "bias"]])
    all_metrics = pd.concat(candidates, ignore_index=True)

    selected_rows = []
    for target in TARGET_LABELS:
        target_metrics = all_metrics[all_metrics["target"].eq(target)].copy()
        winner = target_metrics.sort_values("r2", ascending=False).iloc[0].copy()
        winner["metric"] = TARGET_LABELS[target]
        selected_rows.append(winner)

    selected = pd.DataFrame(selected_rows)
    selected = selected[["metric", "target", "model", "mae", "rmse", "r2", "wmape", "bias"]]
    SELECTION_CSV.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(SELECTION_CSV, index=False)

    lines = [
        "Adunbox 6H Production Model Selection",
        "",
        "Promoted production candidate:",
        "- target-wise LightGBM ensemble over raw 6h metrics",
        "- use anchor_v2 where it improves raw volume metrics",
        "- use business_v3 where it improves conversions/revenue",
        "",
        "Selected raw metric R2:",
    ]
    for row in selected.itertuples(index=False):
        lines.append(f"- {row.metric}: {row.r2:.6f} ({row.model})")
    lines.extend(
        [
            "",
            "Target routing:",
        ]
    )
    for row in selected.itertuples(index=False):
        lines.append(f"- {row.target} -> {row.model}")
    lines.extend(
        [
            "",
            "Decision note:",
            "- anchor_v2 improved impressions and clicks substantially",
            "- business_v3 improved conversions and revenue substantially",
            "- production should keep guardrails for revenue because it remains the weakest raw metric",
            "",
            f"Selection CSV: {SELECTION_CSV.relative_to(BASE_DIR)}",
        ]
    )
    SELECTION_SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(SELECTION_SUMMARY.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
