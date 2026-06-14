from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_REPORTS = Path(r"F:\adunbox_traffic_source_reports.csv")
DEFAULT_ACCOUNTS = Path(r"F:\adunbox_traffic_source_accounts.csv")
DEFAULT_OUTPUT = Path(r"H:\adunbox_joined_traffic_reports_with_timezone.csv")

JOIN_LEFT = ["account_id", "company_id", "traffic_source_id", "traffic_source_config_id"]
JOIN_RIGHT = ["id", "company_id", "traffic_source_id", "traffic_source_config_id"]

REPORT_COLUMNS = [
    "id",
    "report_id",
    "date",
    "company_id",
    "traffic_source_id",
    "traffic_source_config_id",
    "account_id",
    "campaign_id",
    "adset_id",
    "ad_id",
    "impressions",
    "inline_link_clicks",
    "clicks",
    "spend",
    "inline_link_click_ctr",
    "created_at",
    "updated_at",
    "site_id",
    "results",
    "tracker_revenue",
    "tracker_conversions",
    "synced_at",
]

OUTPUT_COLUMNS = [*REPORT_COLUMNS, "timezone"]


def normalize_join_key(series: pd.Series) -> pd.Series:
    text = series.astype("string").fillna("").str.strip()
    return text.str.replace(r"\.0$", "", regex=True)


def load_accounts(path: Path) -> pd.DataFrame:
    accounts = pd.read_csv(
        path,
        usecols=["id", "company_id", "traffic_source_id", "traffic_source_config_id", "timezone"],
        low_memory=False,
    )
    for col in JOIN_RIGHT:
        accounts[col] = normalize_join_key(accounts[col])
    accounts["timezone"] = accounts["timezone"].astype("string").fillna("").str.strip()
    accounts = accounts.drop_duplicates(JOIN_RIGHT, keep="last")
    return accounts


def build_joined_reports(reports_path: Path, accounts_path: Path, output_path: Path, chunksize: int) -> None:
    accounts = load_accounts(accounts_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    total_rows = 0
    matched_rows = 0
    first_chunk = True

    for chunk_idx, chunk in enumerate(
        pd.read_csv(reports_path, usecols=REPORT_COLUMNS, chunksize=chunksize, low_memory=False),
        start=1,
    ):
        for col in JOIN_LEFT:
            chunk[col] = normalize_join_key(chunk[col])

        joined = chunk.merge(
            accounts,
            how="left",
            left_on=JOIN_LEFT,
            right_on=JOIN_RIGHT,
            suffixes=("", "_account"),
        )
        joined["timezone"] = joined["timezone"].astype("string").fillna("").str.strip()
        matched = int(joined["timezone"].ne("").sum())
        total = len(joined)
        total_rows += total
        matched_rows += matched

        joined[OUTPUT_COLUMNS].to_csv(output_path, mode="w" if first_chunk else "a", header=first_chunk, index=False)
        first_chunk = False
        print(
            f"chunk={chunk_idx:,} rows={total:,} matched_timezone={matched:,} "
            f"missing_timezone={total - matched:,}"
        )

    missing_rows = total_rows - matched_rows
    match_rate = matched_rows / total_rows if total_rows else 0.0
    print("")
    print("Joined hourly dataset created")
    print(f"reports: {reports_path}")
    print(f"accounts: {accounts_path}")
    print(f"output: {output_path}")
    print(f"rows: {total_rows:,}")
    print(f"matched_timezone_rows: {matched_rows:,}")
    print(f"missing_timezone_rows: {missing_rows:,}")
    print(f"timezone_match_rate: {match_rate:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the hourly training CSV using the production join: "
            "reports.account_id = accounts.id plus company/source/config keys."
        )
    )
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--accounts", type=Path, default=DEFAULT_ACCOUNTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()

    build_joined_reports(args.reports, args.accounts, args.output, args.chunksize)


if __name__ == "__main__":
    main()
