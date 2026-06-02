from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def assert_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    print(f"OK: {label}: {path}")


def main() -> None:
    print("Adunbox local orchestration smoke check")
    print(f"Project root: {ROOT}")

    assert_exists(ROOT / "models" / "adunbox_daily_24h_histgb" / "metadata.joblib", "24h metadata")
    for target in [
        "target_24h_spend.joblib",
        "target_24h_impressions.joblib",
        "target_24h_inline_link_clicks.joblib",
        "target_24h_tracker_conversions.joblib",
        "target_24h_tracker_revenue.joblib",
    ]:
        assert_exists(ROOT / "models" / "adunbox_daily_24h_histgb" / target, f"24h model {target}")

    assert_exists(ROOT / "models" / "adunbox_entity_history_gru_168h_padded_6h" / "sequence_model.keras", "6h baseline volume GRU")
    assert_exists(ROOT / "models" / "adunbox_entity_history_gru_168h_padded_6h" / "scalers.joblib", "6h baseline scalers")
    assert_exists(ROOT / "models" / "adunbox_entity_history_gru_168h_padded_6h_hybrid" / "target_cvr_model.keras", "6h hybrid CVR model")
    assert_exists(ROOT / "models" / "adunbox_entity_history_gru_168h_padded_6h_hybrid" / "target_roas_model.keras", "6h hybrid ROAS model")
    assert_exists(ROOT / "models" / "adunbox_entity_history_gru_168h_padded_6h_hybrid" / "scalers.joblib", "6h hybrid scalers")

    assert_exists(ROOT / "docs" / "adunbox_6h_final_model__metrics.csv", "final 6h metrics")
    assert_exists(ROOT / "docs" / "adunbox_daily_24h_histgb__metrics.csv", "final 24h metrics")
    assert_exists(ROOT / "orchestration" / "dagster_assets.py", "orchestration asset graph")

    print("\nSmoke check passed. Repo has the promoted 6h/24h model artifacts and orchestration skeleton.")
    print("Note: this smoke test validates packaging only. Full scoring requires production data inputs.")


if __name__ == "__main__":
    main()

