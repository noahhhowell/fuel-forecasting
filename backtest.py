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
SITE_RATING_METRIC = "trimmed_mape_pct"


def _finite_mean(values: pd.Series) -> float:
    """Return the mean of finite numeric values, or NaN when none exist."""
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return np.nan
    return float(finite.mean())


def _round_finite(value: float, digits: int) -> float:
    """Round finite values while preserving NaN/inf sentinels."""
    if not np.isfinite(value):
        return value
    return round(float(value), digits)


def _compute_error_fields(forecast: float, actual: float) -> dict:
    """Compute row-level signed and absolute error metrics."""
    forecast = float(forecast)
    actual = float(actual)
    signed_error = forecast - actual
    if actual == 0:
        return {
            "signed_error": signed_error,
            "signed_error_pct": np.nan,
            "error_pct": np.nan,
            "residual": np.nan,
        }

    signed_error_pct = signed_error / actual * 100
    return {
        "signed_error": signed_error,
        "signed_error_pct": signed_error_pct,
        "error_pct": abs(signed_error_pct),
        "residual": signed_error / actual,
    }


def _coerce_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce or derive columns needed for MAPE, ME, and MPE metrics."""
    clean = df.copy()

    derived_error = None
    derived_pct = None
    if {"forecast", "actual"}.issubset(clean.columns):
        forecast = pd.to_numeric(clean["forecast"], errors="coerce")
        actual = pd.to_numeric(clean["actual"], errors="coerce")
        derived_error = forecast - actual
        derived_pct = pd.Series(np.nan, index=clean.index, dtype=float)
        pct_mask = actual.notna() & (actual != 0)
        derived_pct.loc[pct_mask] = (
            derived_error.loc[pct_mask] / actual.loc[pct_mask] * 100
        )

    residual_pct = None
    if "residual" in clean.columns:
        residual_pct = pd.to_numeric(clean["residual"], errors="coerce") * 100

    for col in ("error_pct", "signed_error", "signed_error_pct"):
        if col not in clean.columns:
            clean[col] = np.nan
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    if "error" in clean.columns:
        clean["signed_error"] = clean["signed_error"].fillna(
            pd.to_numeric(clean["error"], errors="coerce")
        )
    if "pct_error" in clean.columns:
        clean["signed_error_pct"] = clean["signed_error_pct"].fillna(
            pd.to_numeric(clean["pct_error"], errors="coerce")
        )

    if derived_error is not None:
        clean["signed_error"] = clean["signed_error"].fillna(derived_error)
    if derived_pct is not None:
        clean["signed_error_pct"] = clean["signed_error_pct"].fillna(derived_pct)
    if residual_pct is not None:
        clean["signed_error_pct"] = clean["signed_error_pct"].fillna(residual_pct)

    clean["error_pct"] = clean["error_pct"].fillna(
        clean["signed_error_pct"].abs()
    ).abs()

    # Treat ±inf as missing so downstream NaN-skipping aggregations ignore it.
    metric_cols = ["error_pct", "signed_error", "signed_error_pct"]
    clean[metric_cols] = clean[metric_cols].replace([np.inf, -np.inf], np.nan)
    return clean


def _validate_backtest_params(months: int, min_months: int, horizon: int) -> None:
    """Validate backtest configuration values."""
    if months < 1:
        raise ValueError("months must be >= 1")
    if min_months < 1:
        raise ValueError("min_months must be >= 1")
    if horizon < 0:
        raise ValueError("horizon must be >= 0")


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


def _trimmed_mean_drop_worst(values: pd.Series) -> float:
    """Mean after dropping the single worst month (highest APE) per site."""
    values = values.dropna()
    if values.empty:
        return np.nan
    if len(values) <= 1:
        return float(values.mean())
    sorted_vals = values.sort_values()
    return float(sorted_vals.iloc[:-1].mean())


def build_site_error_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-site error metrics including robust measures.

    ME/MPE are signed bias metrics. Positive values mean over-forecasting.

    Returns columns:
      site_id, mape_pct, me, mpe, median_ape_pct, trimmed_mape_pct,
      max_ape_pct, n_months, rating
    """
    columns = [
        "site_id",
        "mape_pct",
        "me",
        "mpe",
        "median_ape_pct",
        "trimmed_mape_pct",
        "max_ape_pct",
        "n_months",
        "rating",
    ]
    if results_df.empty:
        return pd.DataFrame(columns=columns)

    clean = _coerce_error_columns(results_df)
    clean = clean[
        clean[["error_pct", "signed_error", "signed_error_pct"]].notna().any(axis=1)
    ].copy()
    if clean.empty:
        return pd.DataFrame(columns=columns)

    # Aggregations skip NaN, so each metric uses only the rows where its
    # source column is present.
    grouped = clean.groupby("site_id")
    site_metrics = grouped.agg(
        mape_pct=("error_pct", "mean"),
        me=("signed_error", "mean"),
        mpe=("signed_error_pct", "mean"),
        median_ape_pct=("error_pct", "median"),
        max_ape_pct=("error_pct", "max"),
        n_months=("error_pct", "count"),
    ).reset_index()
    trimmed = (
        grouped["error_pct"]
        .apply(_trimmed_mean_drop_worst)
        .reset_index(name="trimmed_mape_pct")
    )
    site_metrics = site_metrics.merge(trimmed, on="site_id", how="left")

    site_metrics["n_months"] = site_metrics["n_months"].astype(int)
    site_metrics = site_metrics.sort_values(
        SITE_RATING_METRIC, ascending=False, na_position="last"
    )

    site_metrics["rating"] = pd.cut(
        site_metrics[SITE_RATING_METRIC],
        bins=[-np.inf, MAPE_GOOD_THRESHOLD, MAPE_ACCEPTABLE_THRESHOLD, np.inf],
        labels=["Good", "Acceptable", "Review"],
    )
    return site_metrics[columns]


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

    ME/MPE are signed bias metrics. Positive values mean over-forecasting.

    Returns columns: site_id, model, mape_pct, me, mpe, n_months
    """
    columns = ["site_id", "model", "mape_pct", "me", "mpe", "n_months"]
    if per_model_df.empty:
        return pd.DataFrame(columns=columns)

    clean = _coerce_error_columns(per_model_df)
    clean = clean[
        clean[["error_pct", "signed_error", "signed_error_pct"]].notna().any(axis=1)
    ].copy()
    if clean.empty:
        return pd.DataFrame(columns=columns)

    # Aggregations skip NaN, so each metric uses only the rows where its
    # source column is present.
    metrics = (
        clean.groupby(["site_id", "model"])
        .agg(
            mape_pct=("error_pct", "mean"),
            me=("signed_error", "mean"),
            mpe=("signed_error_pct", "mean"),
            n_months=("error_pct", "count"),
        )
        .reset_index()
    )
    metrics["n_months"] = metrics["n_months"].astype(int)
    return metrics[columns]


def run_backtest(db_path="fuel_sales.db", months=6, output=None, min_months=24,
                 horizon=2, return_per_model=False, use_adaptive_weights=False):
    _validate_backtest_params(months, min_months, horizon)

    db = FuelDatabase(db_path)
    try:
        return _run_backtest_inner(db, months, output, min_months, horizon,
                                   return_per_model=return_per_model,
                                   use_adaptive_weights=use_adaptive_weights)
    finally:
        db.close()


def _run_backtest_inner(db, months, output, min_months, horizon,
                        return_per_model=False, use_adaptive_weights=False):
    """Core backtest logic, called inside a try/finally that guarantees db.close()."""
    _validate_backtest_params(months, min_months, horizon)

    # Default: disable adaptive weights to get clean, unbiased error measurements
    # using fixed default weights.  The --adaptive flag flips this on so calibrated
    # per-site weights can be evaluated out-of-sample against the fixed baseline.
    forecaster = FuelForecaster(db, min_months_data=min_months,
                                use_adaptive_weights=use_adaptive_weights)

    stats = db.get_summary_stats()
    if not stats.get("total_records"):
        raise RuntimeError("Database is empty. Load data before running backtest.")
    max_date = _parse_max_date(stats)

    # Build list of test months (most recent complete months)
    last_complete = (max_date.to_period("M") - 1).to_timestamp()
    test_months = sorted(
        (last_complete.to_period("M") - i).to_timestamp() for i in range(months)
    )

    # Get sites with enough data
    sites_df = db.get_distinct_sites()
    print(f"Backtest: {months} months, {len(sites_df)} sites total, horizon={horizon} months ahead\n")
    print(f"Test period: {test_months[0].strftime('%Y-%m')} to {test_months[-1].strftime('%Y-%m')}")
    print(f"Checking data sufficiency...\n")

    earliest_test = test_months[0]
    qualified_sites = []
    for _, row in sites_df.iterrows():
        site_id = row["site_id"]
        cutoff = (earliest_test - pd.DateOffset(months=horizon, days=1)).strftime("%Y-%m-%d")
        try:
            data = forecaster.prepare_monthly_data(
                site_id=site_id, end_date=cutoff,
                handle_outliers=False, fill_gaps=False,
            )
        except ValueError:
            continue
        if len(data) >= min_months:
            qualified_sites.append(site_id)

    print(f"Sites with >= {min_months} months of pre-test data: {len(qualified_sites)}")
    if not qualified_sites:
        print("No sites have enough data for backtesting.")
        if return_per_model:
            return None, None, None
        return None, None

    # Run forecasts for each site x test month
    results = []
    per_model_results = []
    skipped_count = 0
    total_combos = len(qualified_sites) * len(test_months)
    done = 0

    for site_id in qualified_sites:
        for test_month in test_months:
            done += 1
            if done % 50 == 0 or done == total_combos:
                print(f"  Progress: {done}/{total_combos}", end="\r")

            target = test_month.strftime("%Y-%m")

            actual = get_actual_monthly_volume(db, site_id, target)
            if actual is None or actual <= 0:
                continue

            cutoff = (test_month - pd.DateOffset(months=horizon, days=1)).strftime("%Y-%m-%d")

            try:
                monthly_data = forecaster.prepare_monthly_data(
                    site_id=site_id, end_date=cutoff,
                    handle_outliers=True, fill_gaps=True,
                )
                monthly_data_raw = forecaster.prepare_monthly_data(
                    site_id=site_id, end_date=cutoff,
                    handle_outliers=False, fill_gaps=False,
                )

                if len(monthly_data) < min_months:
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
                error_fields = _compute_error_fields(forecast_vol, actual)

                results.append({
                    "site_id": site_id,
                    "month": target,
                    "forecast": round(forecast_vol, 1),
                    "actual": round(actual, 1),
                    "signed_error": round(error_fields["signed_error"], 1),
                    "signed_error_pct": _round_finite(
                        error_fields["signed_error_pct"], 2
                    ),
                    "error_pct": _round_finite(error_fields["error_pct"], 2),
                })

                # Collect per-model results when requested
                if return_per_model:
                    for _, row in forecast_df.iterrows():
                        model_name = row["model"]
                        if model_name in ("ENSEMBLE", "FALLBACK"):
                            continue
                        model_vol = float(row["forecast_volume"])
                        model_errors = _compute_error_fields(model_vol, actual)
                        per_model_results.append({
                            "site_id": site_id,
                            "month": target,
                            "model": model_name,
                            "forecast": round(model_vol, 1),
                            "actual": round(actual, 1),
                            "signed_error": round(model_errors["signed_error"], 1),
                            "signed_error_pct": _round_finite(
                                model_errors["signed_error_pct"], 2
                            ),
                            "error_pct": _round_finite(model_errors["error_pct"], 2),
                            "residual": _round_finite(model_errors["residual"], 4),
                        })

            except (ValueError, KeyError) as e:
                # Expected data-related errors - count them so the user sees the skip rate
                skipped_count += 1
                logger.info(f"Skipped site {site_id}, month {target}: {e}")
                continue
            except Exception as e:
                # Unexpected errors - log at warning so they surface
                skipped_count += 1
                logger.warning(f"Site {site_id}, month {target}: {e}")
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

    overall_mape = _finite_mean(results_df["error_pct"])
    overall_median_ape = pd.to_numeric(
        results_df["error_pct"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).median()
    overall_me = _finite_mean(results_df["signed_error"])
    overall_mpe = _finite_mean(results_df["signed_error_pct"])
    overall_trimmed_mape = _finite_mean(site_mape["trimmed_mape_pct"])
    rating_values = site_mape[SITE_RATING_METRIC]
    good = (rating_values < MAPE_GOOD_THRESHOLD).sum()
    acceptable = (
        (rating_values >= MAPE_GOOD_THRESHOLD)
        & (rating_values < MAPE_ACCEPTABLE_THRESHOLD)
    ).sum()
    review = (rating_values >= MAPE_ACCEPTABLE_THRESHOLD).sum()
    total_sites = len(site_mape)
    pct = lambda n: round(n * 100 / total_sites) if total_sites else 0

    print(f"Backtest: {months} months, {total_sites} sites\n")
    print(f"Overall MAPE: {overall_mape:.1f}%")
    print(f"Overall ME: {overall_me:+,.1f} gallons")
    print(f"Overall MPE: {overall_mpe:+.1f}%\n")
    print(f"Median APE (all site-month rows): {overall_median_ape:.1f}%")
    print(f"Trimmed MAPE (drop each site's worst month): {overall_trimmed_mape:.1f}%\n")
    print("Site ratings use trimmed MAPE:")
    print(f"  < 5%:   {good:>4} sites ({pct(good)}%) - Good")
    print(f"  5-10%:  {acceptable:>4} sites ({pct(acceptable)}%) - Acceptable")
    print(f"  >= 10%: {review:>4} sites ({pct(review)}%) - Review these")

    if output:
        summary_df = pd.DataFrame(
            [
                {"metric": "overall_mape_pct", "value": round(float(overall_mape), 4)},
                {"metric": "overall_me", "value": round(float(overall_me), 4)},
                {"metric": "overall_mpe_pct", "value": round(float(overall_mpe), 4)},
                {"metric": "overall_median_ape_pct", "value": round(float(overall_median_ape), 4)},
                {"metric": "overall_trimmed_mape_pct", "value": round(float(overall_trimmed_mape), 4)},
                {"metric": "site_rating_metric", "value": SITE_RATING_METRIC},
                {"metric": "sites_total", "value": int(total_sites)},
                {"metric": "sites_good_trimmed_lt_5_count", "value": int(good)},
                {"metric": "sites_acceptable_trimmed_5_to_10_count", "value": int(acceptable)},
                {"metric": "sites_review_trimmed_ge_10_count", "value": int(review)},
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
    parser.add_argument(
        "--adaptive", dest="adaptive", action="store_true", default=False,
        help="Use calibrated per-site adaptive weights instead of fixed 70/30 "
             "(requires a prior `cli.py calibrate` run). Default: fixed weights.",
    )
    args = parser.parse_args()

    run_backtest(
        db_path=args.database,
        months=args.months,
        output=args.output,
        min_months=args.min_months,
        horizon=args.horizon,
        use_adaptive_weights=args.adaptive,
    )


if __name__ == "__main__":
    main()
