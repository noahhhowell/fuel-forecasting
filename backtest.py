"""
Backtest - Evaluate forecast accuracy against historical actuals.

Holds out recent months, generates forecasts using only prior data,
applies the same post-processing guardrails used in production exports,
and compares to what actually happened.

Usage:
    python backtest.py
    python backtest.py --months 12
    python backtest.py --months 6 --output backtest_results.xlsx
"""

import argparse
import logging

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
    """
    Compute row-level forecast error fields.

    error is signed volume error: forecast - actual. pct_error is the signed
    percentage error, so positive error/ME/MPE means over-forecasting. When
    actual is zero, pct_error and error_pct are undefined and left as NaN; the
    signed volume error is still valid and should be included in ME.
    """
    forecast = float(forecast)
    actual = float(actual)
    error = forecast - actual

    if actual == 0:
        return {
            "error": error,
            "pct_error": np.nan,
            "error_pct": np.nan,
            "residual": np.nan,
        }

    pct_error = error / actual * 100
    return {
        "error": error,
        "pct_error": pct_error,
        "error_pct": abs(pct_error),
        "residual": error / actual,
    }


def _coerce_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce or derive metric columns needed for MAPE, ME, and MPE."""
    clean = df.copy()

    derived_error = None
    derived_pct_error = None
    if {"forecast", "actual"}.issubset(clean.columns):
        forecast = pd.to_numeric(clean["forecast"], errors="coerce")
        actual = pd.to_numeric(clean["actual"], errors="coerce")
        derived_error = forecast - actual
        derived_pct_error = pd.Series(np.nan, index=clean.index, dtype=float)
        pct_mask = actual.notna() & (actual != 0)
        derived_pct_error.loc[pct_mask] = (
            derived_error.loc[pct_mask] / actual.loc[pct_mask] * 100
        )

    residual_pct_error = None
    if "residual" in clean.columns:
        residual_pct_error = pd.to_numeric(clean["residual"], errors="coerce") * 100

    for col in ("error_pct", "error", "pct_error"):
        if col not in clean.columns:
            clean[col] = np.nan
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    if derived_error is not None:
        clean["error"] = clean["error"].fillna(derived_error)

    if derived_pct_error is not None:
        clean["pct_error"] = clean["pct_error"].fillna(derived_pct_error)
    if residual_pct_error is not None:
        clean["pct_error"] = clean["pct_error"].fillna(residual_pct_error)

    clean["error_pct"] = clean["error_pct"].fillna(clean["pct_error"].abs()).abs()
    return clean


def _validate_month_params(months: int, min_months: int, horizon: int) -> None:
    """Shared month/horizon validation used by backtest and calibration."""
    if months < 1:
        raise ValueError("months must be >= 1")
    if min_months < 1:
        raise ValueError("min_months must be >= 1")
    if horizon < 0:
        raise ValueError("horizon must be >= 0")


def _validate_backtest_params(months: int, min_months: int, horizon: int) -> None:
    """Validate backtest configuration values."""
    _validate_month_params(months, min_months, horizon)


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


def _prior_year_known_by_cutoff(month_str: str, cutoff: str) -> bool:
    """Return True if prior-year same-month actuals exist before forecast cutoff."""
    target_date = pd.to_datetime(f"{month_str}-01")
    prior_year_start = target_date - pd.DateOffset(years=1)
    prior_year_end = prior_year_start + pd.offsets.MonthEnd(0)
    return prior_year_end <= pd.to_datetime(cutoff)


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

    ME/MPE are signed bias metrics: positive values mean over-forecasting.
    MAPE and MPE exclude rows where actual is zero; ME includes them.

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
        np.isfinite(clean[["error_pct", "error", "pct_error"]]).any(axis=1)
    ].copy()
    if clean.empty:
        return pd.DataFrame(columns=columns)

    site_metrics = clean.groupby("site_id").size().reset_index(name="_rows")
    site_metrics = site_metrics.drop(columns="_rows")

    pct_clean = clean[np.isfinite(clean["error_pct"])]
    if not pct_clean.empty:
        grouped = pct_clean.groupby("site_id")["error_pct"]
        mape_metrics = (
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
        site_metrics = site_metrics.merge(mape_metrics, on="site_id", how="left")
        site_metrics = site_metrics.merge(trimmed, on="site_id", how="left")
    else:
        site_metrics["mape_pct"] = np.nan
        site_metrics["median_ape_pct"] = np.nan
        site_metrics["max_ape_pct"] = np.nan
        site_metrics["n_months"] = 0
        site_metrics["trimmed_mape_pct"] = np.nan

    error_clean = clean[np.isfinite(clean["error"])]
    if not error_clean.empty:
        me = error_clean.groupby("site_id")["error"].mean().reset_index(name="me")
        site_metrics = site_metrics.merge(me, on="site_id", how="left")
    else:
        site_metrics["me"] = np.nan

    signed_pct_clean = clean[np.isfinite(clean["pct_error"])]
    if not signed_pct_clean.empty:
        mpe = (
            signed_pct_clean.groupby("site_id")["pct_error"]
            .mean()
            .reset_index(name="mpe")
        )
        site_metrics = site_metrics.merge(mpe, on="site_id", how="left")
    else:
        site_metrics["mpe"] = np.nan

    site_metrics["n_months"] = site_metrics["n_months"].fillna(0).astype(int)
    site_metrics = site_metrics.sort_values(
        "mape_pct", ascending=False, na_position="last"
    )

    site_metrics["rating"] = pd.cut(
        site_metrics["mape_pct"],
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
    Build per-site(/grade) per-model error metrics.

    ME/MPE are signed bias metrics: positive values mean over-forecasting.
    MAPE and MPE exclude rows where actual is zero; ME includes them.

    Returns columns: site_id, [grade,] model, mape_pct, me, mpe, n_months
    """
    base_columns = ["site_id", "model", "mape_pct", "me", "mpe", "n_months"]
    if "grade" in per_model_df.columns:
        base_columns = [
            "site_id",
            "grade",
            "model",
            "mape_pct",
            "me",
            "mpe",
            "n_months",
        ]
    if per_model_df.empty:
        return pd.DataFrame(columns=base_columns)

    clean = _coerce_error_columns(per_model_df)
    clean = clean[
        np.isfinite(clean[["error_pct", "error", "pct_error"]]).any(axis=1)
    ].copy()
    if clean.empty:
        return pd.DataFrame(columns=base_columns)

    group_cols = ["site_id", "model"]
    if "grade" in clean.columns:
        group_cols = ["site_id", "grade", "model"]

    metrics = clean.groupby(group_cols).size().reset_index(name="_rows")
    metrics = metrics.drop(columns="_rows")

    pct_clean = clean[np.isfinite(clean["error_pct"])]
    if not pct_clean.empty:
        mape = (
            pct_clean.groupby(group_cols)["error_pct"]
            .agg(mape_pct="mean", n_months="count")
            .reset_index()
        )
        metrics = metrics.merge(mape, on=group_cols, how="left")
    else:
        metrics["mape_pct"] = np.nan
        metrics["n_months"] = 0

    error_clean = clean[np.isfinite(clean["error"])]
    if not error_clean.empty:
        me = error_clean.groupby(group_cols)["error"].mean().reset_index(name="me")
        metrics = metrics.merge(me, on=group_cols, how="left")
    else:
        metrics["me"] = np.nan

    signed_pct_clean = clean[np.isfinite(clean["pct_error"])]
    if not signed_pct_clean.empty:
        mpe = (
            signed_pct_clean.groupby(group_cols)["pct_error"]
            .mean()
            .reset_index(name="mpe")
        )
        metrics = metrics.merge(mpe, on=group_cols, how="left")
    else:
        metrics["mpe"] = np.nan

    metrics["n_months"] = metrics["n_months"].fillna(0).astype(int)
    return metrics[base_columns]


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
    sanity_bounds_cache = {}

    for site_id, grade in qualified_combos:
        for test_month in test_months:
            done += 1
            if done % 50 == 0 or done == total_combos:
                print(f"  Progress: {done}/{total_combos}", end="\r")

            target = test_month.strftime("%Y-%m")

            actual = get_actual_monthly_volume(db, site_id, target, grade=grade)
            if actual is None or not np.isfinite(actual) or actual < 0:
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

                show_yoy = _prior_year_known_by_cutoff(target, cutoff)
                forecast_df = forecaster.generate_forecast(
                    target_month=target,
                    site_id=site_id,
                    grade=grade,
                    monthly_data=monthly_data,
                    monthly_data_raw=monthly_data_raw,
                    show_yoy=show_yoy,
                )
                if cutoff not in sanity_bounds_cache:
                    sanity_bounds_cache[cutoff] = forecaster._precompute_sanity_bounds(
                        end_date=cutoff
                    )
                forecast_df = forecaster._apply_caps_to_forecast_df(
                    forecast_df,
                    sanity_bounds_cache[cutoff],
                    monthly_data_raw,
                )

                ensemble = forecast_df[forecast_df["model"] == "ENSEMBLE"]
                if ensemble.empty:
                    continue

                forecast_vol = float(ensemble["forecast_volume"].iloc[0])
                error_fields = _compute_error_fields(forecast_vol, actual)

                results.append({
                    "site_id": site_id,
                    "grade": grade,
                    "model": "ENSEMBLE",
                    "month": target,
                    "forecast": round(forecast_vol, 1),
                    "actual": round(actual, 1),
                    "error": _round_finite(error_fields["error"], 1),
                    "pct_error": _round_finite(error_fields["pct_error"], 2),
                    "error_pct": _round_finite(error_fields["error_pct"], 2),
                })

                # Collect per-model results when requested
                if return_per_model:
                    for _, row in forecast_df.iterrows():
                        model_name = row["model"]
                        if model_name in ("ENSEMBLE", "FALLBACK"):
                            continue
                        model_vol = float(row["forecast_volume"])
                        model_error_fields = _compute_error_fields(model_vol, actual)
                        per_model_results.append({
                            "site_id": site_id,
                            "grade": grade,
                            "month": target,
                            "model": model_name,
                            "forecast": round(model_vol, 1),
                            "actual": round(actual, 1),
                            "error": _round_finite(model_error_fields["error"], 1),
                            "pct_error": _round_finite(
                                model_error_fields["pct_error"], 2
                            ),
                            "error_pct": _round_finite(
                                model_error_fields["error_pct"], 2
                            ),
                            "residual": _round_finite(
                                model_error_fields["residual"], 4
                            ),
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

    overall_mape = _finite_mean(results_df["error_pct"])
    overall_me = _finite_mean(results_df["error"])
    overall_mpe = _finite_mean(results_df["pct_error"])
    overall_median_ape = pd.to_numeric(
        results_df["error_pct"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).median()
    overall_trimmed_mape = _finite_mean(site_mape["trimmed_mape_pct"])
    good = (site_mape["mape_pct"] < MAPE_GOOD_THRESHOLD).sum()
    acceptable = (
        (site_mape["mape_pct"] >= MAPE_GOOD_THRESHOLD)
        & (site_mape["mape_pct"] < MAPE_ACCEPTABLE_THRESHOLD)
    ).sum()
    review = (site_mape["mape_pct"] >= MAPE_ACCEPTABLE_THRESHOLD).sum()
    total_sites = len(site_mape)
    pct = lambda n: round(n * 100 / total_sites) if total_sites else 0

    print(f"Backtest: {months} months, {total_sites} sites\n")
    print(f"Overall MAPE: {overall_mape:.1f}%")
    print(f"Overall ME: {overall_me:+.1f} volume")
    print(f"Overall MPE: {overall_mpe:+.1f}%\n")
    print(f"Median APE (all site-month rows): {overall_median_ape:.1f}%")
    print(f"Trimmed MAPE (drop each site's worst month): {overall_trimmed_mape:.1f}%\n")
    print(f"  MAPE < 5%:  {good:>4} sites ({pct(good)}%) - Good")
    print(f"  MAPE 5-10%: {acceptable:>4} sites ({pct(acceptable)}%) - Acceptable")
    print(f"  MAPE > 10%: {review:>4} sites ({pct(review)}%) - Review these")

    if output:
        summary_df = pd.DataFrame(
            [
                {"metric": "overall_mape_pct", "value": round(float(overall_mape), 4)},
                {"metric": "overall_me", "value": round(float(overall_me), 4)},
                {"metric": "overall_mpe_pct", "value": round(float(overall_mpe), 4)},
                {
                    "metric": "overall_median_ape_pct",
                    "value": round(float(overall_median_ape), 4),
                },
                {
                    "metric": "overall_trimmed_mape_pct",
                    "value": round(float(overall_trimmed_mape), 4),
                },
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
