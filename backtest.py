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

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# MAPE thresholds for site accuracy ratings.
# <5% is considered "Good", 5-10% "Acceptable", >=10% "Review".
MAPE_GOOD_THRESHOLD = 5
MAPE_ACCEPTABLE_THRESHOLD = 10


def _validate_backtest_params(months: int, min_months: int, horizon: int) -> None:
    """Validate backtest configuration values."""
    if months < 1:
        raise ValueError("months must be >= 1")
    if min_months < 1:
        raise ValueError("min_months must be >= 1")
    if horizon < 0:
        raise ValueError("horizon must be >= 0")


def get_actual_monthly_volume(db, site_id, month_str, grade=None):
    """Get actual total volume for a site (optionally per-grade) in a given month."""
    start = f"{month_str}-01"
    end = (pd.to_datetime(start) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
    grades = [grade] if grade else None
    df = db.get_sales_data(
        start_date=start, end_date=end,
        site_ids=[site_id], grades=grades, exclude_estimated=True,
    )
    if df.empty:
        return None
    return float(df["volume"].sum())


def _trimmed_mean_drop_worst(values: pd.Series) -> float:
    """Mean after dropping the single worst month (highest APE) per site."""
    if values.empty:
        return np.nan
    if len(values) <= 1:
        return float(values.mean())
    sorted_vals = values.sort_values()
    return float(sorted_vals.iloc[:-1].mean())


def build_site_error_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-site error metrics including robust measures.

    Returns columns:
      site_id, mape_pct, median_ape_pct, trimmed_mape_pct, max_ape_pct, n_months, rating
    """
    if results_df.empty:
        return pd.DataFrame(
            columns=[
                "site_id",
                "mape_pct",
                "median_ape_pct",
                "trimmed_mape_pct",
                "max_ape_pct",
                "n_months",
                "rating",
            ]
        )

    clean = results_df.copy()
    clean["error_pct"] = pd.to_numeric(clean["error_pct"], errors="coerce")
    clean = clean[np.isfinite(clean["error_pct"])].copy()
    if clean.empty:
        return pd.DataFrame(
            columns=[
                "site_id",
                "mape_pct",
                "median_ape_pct",
                "trimmed_mape_pct",
                "max_ape_pct",
                "n_months",
                "rating",
            ]
        )

    grouped = clean.groupby("site_id")["error_pct"]
    site_metrics = (
        grouped.agg(
            mape_pct="mean",
            median_ape_pct="median",
            max_ape_pct="max",
            n_months="count",
        )
        .reset_index()
    )

    trimmed = (
        grouped.apply(_trimmed_mean_drop_worst)
        .reset_index(name="trimmed_mape_pct")
    )
    site_metrics = site_metrics.merge(trimmed, on="site_id", how="left")
    site_metrics = site_metrics.sort_values("mape_pct", ascending=False)

    site_metrics["rating"] = pd.cut(
        site_metrics["mape_pct"],
        bins=[-np.inf, MAPE_GOOD_THRESHOLD, MAPE_ACCEPTABLE_THRESHOLD, np.inf],
        labels=["Good", "Acceptable", "Review"],
    )
    return site_metrics


def _parse_max_date(stats: dict) -> pd.Timestamp:
    """Extract the end date from summary stats, raising a clear error on failure."""
    date_range = stats.get("date_range", "")
    if not isinstance(date_range, str) or " to " not in date_range:
        raise RuntimeError(
            f"Unexpected date_range format in summary stats: {date_range!r}"
        )
    return pd.to_datetime(date_range.split(" to ")[1])


def build_per_model_site_metrics(per_model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-site per-model error metrics.

    Returns columns: site_id, model, mape_pct, n_months
    """
    if per_model_df.empty:
        return pd.DataFrame(columns=["site_id", "model", "mape_pct", "n_months"])

    clean = per_model_df.copy()
    clean["error_pct"] = pd.to_numeric(clean["error_pct"], errors="coerce")
    clean = clean[np.isfinite(clean["error_pct"])].copy()
    if clean.empty:
        return pd.DataFrame(columns=["site_id", "model", "mape_pct", "n_months"])

    return (
        clean.groupby(["site_id", "model"])["error_pct"]
        .agg(mape_pct="mean", n_months="count")
        .reset_index()
    )


def run_backtest(db_path="fuel_sales.db", months=6, output=None, min_months=24,
                 horizon=2, return_per_model=False):
    _validate_backtest_params(months, min_months, horizon)

    db = FuelDatabase(db_path)
    try:
        return _run_backtest_inner(db, months, output, min_months, horizon,
                                   return_per_model=return_per_model)
    finally:
        db.close()


def _run_backtest_inner(db, months, output, min_months, horizon,
                        return_per_model=False):
    """Core backtest logic, called inside a try/finally that guarantees db.close()."""
    _validate_backtest_params(months, min_months, horizon)

    # Disable adaptive weights during backtest to get clean, unbiased error
    # measurements using fixed default weights.
    forecaster = FuelForecaster(db, min_months_data=min_months,
                                use_adaptive_weights=False)

    stats = db.get_summary_stats()
    if not stats.get("total_records"):
        raise RuntimeError("Database is empty. Load data before running backtest.")
    max_date = _parse_max_date(stats)

    # Build list of test months (most recent complete months)
    last_complete = (max_date.to_period("M") - 1).to_timestamp()
    test_months = sorted(
        (last_complete.to_period("M") - i).to_timestamp() for i in range(months)
    )

    # Get site/grade combos with enough data — backtesting at the same grain
    # as production forecasting (per-site-per-grade) so calibrated weights and
    # residuals reflect the actual unit of prediction.
    site_grades_df = db.get_distinct_site_grades()
    print(f"Backtest: {months} months, {len(site_grades_df)} site-grade combos, horizon={horizon} months ahead\n")
    print(f"Test period: {test_months[0].strftime('%Y-%m')} to {test_months[-1].strftime('%Y-%m')}")
    print(f"Checking data sufficiency...\n")

    earliest_test = test_months[0]
    qualified_combos = []  # list of (site_id, grade)
    for _, row in site_grades_df.iterrows():
        site_id = row["site_id"]
        grade = row["grade"]
        cutoff = (earliest_test - pd.DateOffset(months=horizon, days=1)).strftime("%Y-%m-%d")
        try:
            data = forecaster.prepare_monthly_data(
                site_id=site_id, grade=grade, end_date=cutoff,
                handle_outliers=False, fill_gaps=False,
            )
        except ValueError:
            continue
        if len(data) >= min_months:
            qualified_combos.append((site_id, grade))

    print(f"Site-grade combos with >= {min_months} months of pre-test data: {len(qualified_combos)}")
    if not qualified_combos:
        print("No site-grade combos have enough data for backtesting.")
        if return_per_model:
            return None, None, None
        return None, None

    # Run forecasts for each (site, grade) x test month
    results = []
    per_model_results = []
    skipped_count = 0
    total_combos = len(qualified_combos) * len(test_months)
    done = 0

    for site_id, grade in qualified_combos:
        for test_month in test_months:
            done += 1
            if done % 50 == 0 or done == total_combos:
                print(f"  Progress: {done}/{total_combos}", end="\r")

            target = test_month.strftime("%Y-%m")

            actual = get_actual_monthly_volume(db, site_id, target, grade=grade)
            if actual is None or actual <= 0:
                continue

            cutoff = (test_month - pd.DateOffset(months=horizon, days=1)).strftime("%Y-%m-%d")

            try:
                monthly_data = forecaster.prepare_monthly_data(
                    site_id=site_id, grade=grade, end_date=cutoff,
                    handle_outliers=True, fill_gaps=True,
                )
                monthly_data_raw = forecaster.prepare_monthly_data(
                    site_id=site_id, grade=grade, end_date=cutoff,
                    handle_outliers=False, fill_gaps=False,
                )

                if len(monthly_data) < min_months:
                    continue

                forecast_df = forecaster.generate_forecast(
                    target_month=target,
                    site_id=site_id,
                    grade=grade,
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
                    "grade": grade,
                    "month": target,
                    "forecast": round(forecast_vol, 1),
                    "actual": round(actual, 1),
                    "error_pct": round(error_pct, 2),
                })

                # Collect per-model results when requested
                if return_per_model:
                    for _, row in forecast_df.iterrows():
                        model_name = row["model"]
                        if model_name in ("ENSEMBLE", "FALLBACK"):
                            continue
                        model_vol = float(row["forecast_volume"])
                        model_error = abs(model_vol - actual) / actual * 100
                        residual = (model_vol - actual) / actual
                        per_model_results.append({
                            "site_id": site_id,
                            "grade": grade,
                            "month": target,
                            "model": model_name,
                            "forecast": round(model_vol, 1),
                            "actual": round(actual, 1),
                            "error_pct": round(model_error, 2),
                            "residual": round(residual, 4),
                        })

            except (ValueError, KeyError) as e:
                # Expected data-related errors - count them so the user sees the skip rate
                skipped_count += 1
                logger.info(f"Skipped site {site_id}, grade {grade}, month {target}: {e}")
                continue
            except Exception as e:
                # Unexpected errors - log at warning so they surface
                skipped_count += 1
                logger.warning(f"Site {site_id}, grade {grade}, month {target}: {e}")
                continue

    print()  # clear progress line

    if skipped_count:
        print(f"  Skipped {skipped_count} site-month combos (see log for details)")

    if not results:
        print("No results generated. Check that test months have actual data.")
        if return_per_model:
            return None, None, None
        return None, None

    results_df = pd.DataFrame(results)
    per_model_df = pd.DataFrame(per_model_results) if return_per_model else None

    site_mape = build_site_error_metrics(results_df)

    overall_mape = results_df["error_pct"].mean()
    overall_median_ape = results_df["error_pct"].median()
    overall_trimmed_mape = site_mape["trimmed_mape_pct"].mean()
    good = (site_mape["mape_pct"] < MAPE_GOOD_THRESHOLD).sum()
    acceptable = (
        (site_mape["mape_pct"] >= MAPE_GOOD_THRESHOLD)
        & (site_mape["mape_pct"] < MAPE_ACCEPTABLE_THRESHOLD)
    ).sum()
    review = (site_mape["mape_pct"] >= MAPE_ACCEPTABLE_THRESHOLD).sum()
    total_sites = len(site_mape)
    pct = lambda n: round(n * 100 / total_sites) if total_sites else 0

    print(f"Backtest: {months} months, {total_sites} sites\n")
    print(f"Overall MAPE: {overall_mape:.1f}%\n")
    print(f"Median APE (all site-month rows): {overall_median_ape:.1f}%")
    print(f"Trimmed MAPE (drop each site's worst month): {overall_trimmed_mape:.1f}%\n")
    print(f"  MAPE < 5%:  {good:>4} sites ({pct(good)}%) - Good")
    print(f"  MAPE 5-10%: {acceptable:>4} sites ({pct(acceptable)}%) - Acceptable")
    print(f"  MAPE > 10%: {review:>4} sites ({pct(review)}%) - Review these")

    if output:
        summary_df = pd.DataFrame(
            [
                {"metric": "overall_mape_pct", "value": round(float(overall_mape), 4)},
                {"metric": "overall_median_ape_pct", "value": round(float(overall_median_ape), 4)},
                {"metric": "overall_trimmed_mape_pct", "value": round(float(overall_trimmed_mape), 4)},
                {"metric": "sites_total", "value": int(total_sites)},
                {"metric": "sites_good_lt_5_count", "value": int(good)},
                {"metric": "sites_acceptable_5_to_10_count", "value": int(acceptable)},
                {"metric": "sites_review_ge_10_count", "value": int(review)},
            ]
        )
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            results_df.to_excel(writer, sheet_name="Results", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            site_mape.round(2).to_excel(writer, sheet_name="Site MAPE", index=False)
        print(f"\nSaved detail to: {output}")

    if return_per_model:
        return results_df, site_mape, per_model_df
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
    parser.add_argument(
        "--horizon", type=int, default=2,
        help="Forecast horizon in months ahead (default: 2, e.g. Feb submission for April target)",
    )
    args = parser.parse_args()

    run_backtest(
        db_path=args.database,
        months=args.months,
        output=args.output,
        min_months=args.min_months,
        horizon=args.horizon,
    )


if __name__ == "__main__":
    main()

