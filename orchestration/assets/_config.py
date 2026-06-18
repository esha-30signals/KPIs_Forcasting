from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastConfig:
    table_prefix: str
    revenue_col_6h: str
    conversions_col_6h: str
    revenue_col_24h: str
    conversions_col_24h: str
    forecast_table: str
    feature_table: str

    def get_table(self, suffix: str) -> str:
        return f"{self.table_prefix}{suffix}"


ADUNBOX_CONFIG = ForecastConfig(
    table_prefix="adunbox_",
    revenue_col_6h="tracker_revenue",
    conversions_col_6h="tracker_conversions",
    revenue_col_24h="tracker_revenue",
    conversions_col_24h="tracker_conversions",
    forecast_table="adunbox_model_forecasts",
    feature_table="adunbox_model_feature_cache",
)

VIBELETS_CONFIG = ForecastConfig(
    table_prefix="",
    revenue_col_6h="conversions_value",
    conversions_col_6h="conversions",
    revenue_col_24h="conversions_value",
    conversions_col_24h="conversions",
    forecast_table="vibelets_model_forecasts",
    feature_table="vibelets_model_feature_cache",
)
