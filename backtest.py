"""
Backtest - Evaluate forecast accuracy against historical actuals.

Holds out recent months, generates forecasts using only prior data,
and compares to what actually happened. Zero changes to core code.

Usage:
    python backtest.py
    python backtest.py --months 12
    python backtest.py --months 6 --output backtest_results.xlsx
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from database import FuelDatabase
from forecaster import FuelForecaster

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def get_actual_monthly_volume(db, site_id, month_str):
    """Get actual total volume for a site in a given month."""
    start = f"{month_str}-01"
    end = (pd.to_datetime(start) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    df = db.get_sales_data(
        start_date=start, end_date=end,
        site_ids=[site_id], exclude_estimated=True,
    )
    if df.empty:
        return None
    return float(df["volume"].sum())


def run_backtest(db_path="fuel_sales.db", months=6, output=None, min_months=24):
    db = FuelDatabase(db_path)
    forecaster = FuelForecaster(db, min_months_data=min_months)

    # Find the date range in the database
    stats = db.get_summary_stats()
    max_date = pd.to_datetime(stats["date_range"].split(" to ")[1])

    # Build list of test months (most recent complete months)
    # A "complete" month is one that ended before the last date in the DB
    last_complete = (max_date.to_period("M") - 1).to_timestamp()
    test_months = []
    for i in range(months):
        m = (last_complete.to_period("M") - i).to_timestamp()
        test_months.append(m)
    test_months.sort()

    # Get sites with enough data
    sites_df = db.get_distinct_sites()
    print(f"Backtest: {months} months, {len(sites_df)} sites total\n")
    print(f"Test period: {test_months[0].strftime('%Y-%m')} to {test_months[-1].strftime('%Y-%m')}")
    print(f"Checking data sufficiency...\n")

    # Pre-filter sites: need enough history before the earliest test month
    earliest_test = test_months[0]
    qualified_sites = []
    for _, row in sites_df.iterrows():
        site_id = row["site_id"]
        # Check months of data before earliest test month
        cutoff = (earliest_test - pd.DateOffset(days=1)).strftime("%Y-%m-%d")
        data = forecaster.prepare_monthly_data(
            site_id=site_id, end_date=cutoff,
            handle_outliers=False, fill_gaps=False,
        )
        if len(data) >= min_months:
            qualified_sites.append(site_id)

    print(f"Sites with >= {min_months} months of pre-test data: {len(qualified_sites)}")
    if not qualified_sites:
        print("No sites have enough data for backtesting.")
        db.close()
        return

    # Run forecasts for each site x test month
    results = []
    total_combos = len(qualified_sites) * len(test_months)
    done = 0

    for site_id in qualified_sites:
        for test_month in test_months:
            done += 1
            if done % 50 == 0 or done == total_combos:
                print(f"  Progress: {done}/{total_combos}", end="\r")

            target = test_month.strftime("%Y-%m")

            # Get actual volume for this month
            actual = get_actual_monthly_volume(db, site_id, target)
            if actual is None or actual <= 0:
                continue

            # Cutoff: last day of the month before the test month
            cutoff = (test_month - pd.DateOffset(days=1)).strftime("%Y-%m-%d")

            try:
                monthly_data = forecaster.prepare_monthly_data(
                    site_id=site_id, end_date=cutoff,
                    handle_outliers=True, fill_gaps=True,
                )
                monthly_data_raw = forecaster.prepare_monthly_data(
                    site_id=site_id, end_date=cutoff,
                    handle_outliers=False, fill_gaps=False,
                )

                if len(monthly_data) < 12:
                    continue

                forecast_df = forecaster.generate_forecast(
                    target_month=target,
                    site_id=site_id,
                    monthly_data=monthly_data,
                    monthly_data_raw=monthly_data_raw,
                    show_yoy=False,
                )

                ensemble = forecast_df[forecast_df["model"] == "ENSEMBLE"]
                if ensemble.empty:
                    continue

                forecast_vol = float(ensemble["forecast_volume"].iloc[0])
                error_pct = abs(forecast_vol - actual) / actual * 100

                results.append({
                    "site_id": site_id,
                    "month": target,
                    "forecast": round(forecast_vol, 1),
                    "actual": round(actual, 1),
                    "error_pct": round(error_pct, 2),
                })

            except Exception as e:
                logger.debug(f"Site {site_id}, month {target}: {e}")
                continue

    print()  # clear progress line

    if not results:
        print("No results generated. Check that test months have actual data.")
        db.close()
        return

    results_df = pd.DataFrame(results)

    # Calculate MAPE per site
    site_mape = (
        results_df.groupby("site_id")["error_pct"]
        .mean()
        .reset_index()
        .rename(columns={"error_pct": "mape_pct"})
        .sort_values("mape_pct", ascending=False)
    )

    # Add rating
    site_mape["rating"] = pd.cut(
        site_mape["mape_pct"],
        bins=[-np.inf, 5, 10, np.inf],
        labels=["Good", "Acceptable", "Review"],
    )

    # Print summary
    overall_mape = results_df["error_pct"].mean()
    good = (site_mape["mape_pct"] < 5).sum()
    acceptable = ((site_mape["mape_pct"] >= 5) & (site_mape["mape_pct"] < 10)).sum()
    review = (site_mape["mape_pct"] >= 10).sum()
    total_sites = len(site_mape)

    print(f"Backtest: {months} months, {total_sites} sites\n")
    print(f"Overall MAPE: {overall_mape:.1f}%\n")
    print(f"  MAPE < 5%:  {good:>4} sites ({good*100//total_sites}%) - Good")
    print(f"  MAPE 5-10%: {acceptable:>4} sites ({acceptable*100//total_sites}%) - Acceptable")
    print(f"  MAPE > 10%: {review:>4} sites ({review*100//total_sites}%) - Review these")

    # Save to Excel if requested
    if output:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            results_df.to_excel(writer, sheet_name="Results", index=False)
            site_mape.round(2).to_excel(writer, sheet_name="Site MAPE", index=False)
        print(f"\nSaved detail to: {output}")

    db.close()
    return results_df, site_mape


def main():
    parser = argparse.ArgumentParser(
        description="Backtest forecast accuracy against historical actuals"
    )
    parser.add_argument(
        "--months", type=int, default=6,
        help="Number of recent months to test (default: 6)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to Excel file (e.g., backtest_results.xlsx)",
    )
    parser.add_argument(
        "--database", type=str, default="fuel_sales.db",
        help="Path to SQLite database (default: fuel_sales.db)",
    )
    parser.add_argument(
        "--min-months", type=int, default=24,
        help="Minimum months of pre-test data required (default: 24)",
    )
    args = parser.parse_args()

    run_backtest(
        db_path=args.database,
        months=args.months,
        output=args.output,
        min_months=args.min_months,
    )


if __name__ == "__main__":
    main()
