"""
Rigorous investigation of high-error backtest sites.

Multi-dimensional analysis covering:
  1. Data quality & coverage profiling
  2. Statistical characterization (CV, trend, seasonality strength)
  3. Structural break detection (rolling Welch's t-tests on adjacent windows)
  4. Individual model decomposition (ETS vs snaive vs ensemble)
  5. Predictive feature analysis (what predicts high MAPE?)
  6. Error taxonomy (classify root causes)
"""

import warnings
import logging

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from database import FuelDatabase
from forecaster import FuelForecaster

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

DB_PATH = "fuel_sales.db"
BACKTEST_MONTHS = 6
HORIZON = 2
MIN_MONTHS = 24
HIGH_ERROR_THRESHOLD = 10  # MAPE % to classify as "high error"

# ── helpers ──────────────────────────────────────────────────────────────────

def coefficient_of_variation(series):
    """CV = std/mean, measures relative dispersion."""
    m = series.mean()
    if m == 0:
        return np.nan
    return series.std() / m * 100


def seasonality_strength(dates, volumes):
    """
    Strength of seasonality via ratio of seasonal variance to total variance.
    Uses month-of-year means as the seasonal component.
    Returns 0-1 where 1 = perfectly seasonal.

    Args:
        dates: Series of datetime values (used to extract calendar month)
        volumes: Series of volume values
    """
    if len(volumes) < 13:
        return np.nan
    df = pd.DataFrame({"v": volumes.values, "month": dates.dt.month.values})
    monthly_means = df.groupby("month")["v"].transform("mean")
    seasonal_var = monthly_means.var()
    total_var = df["v"].var()
    if total_var == 0:
        return 0.0
    return min(1.0, seasonal_var / total_var)


def trend_strength(series):
    """
    Linear trend R² — how much variance is explained by a linear time trend.
    """
    if len(series) < 6:
        return np.nan
    x = np.arange(len(series))
    slope, intercept, r, p, se = sp_stats.linregress(x, series.values)
    return r ** 2


def trend_slope_pct(series):
    """Monthly slope as % of mean volume."""
    if len(series) < 6:
        return np.nan
    x = np.arange(len(series))
    slope, *_ = sp_stats.linregress(x, series.values)
    m = series.mean()
    if m == 0:
        return np.nan
    return slope / m * 100


def detect_structural_breaks(series, window=12):
    """
    Detect structural breaks using rolling Welch's t-tests.
    Splits the series at each point and tests if adjacent windows have
    significantly different means.
    Returns: number of significant breaks (p < 0.01), most significant break point.
    Note: overlapping windows can overcount breaks; use counts for relative ranking only.
    """
    if len(series) < window * 2:
        return 0, None, np.nan

    values = series.values
    n_breaks = 0
    best_p = 1.0
    best_point = None

    for i in range(window, len(values) - window):
        left = values[i - window : i]
        right = values[i : i + window]
        t_stat, p_val = sp_stats.ttest_ind(left, right, equal_var=False)
        if p_val < 0.01:
            n_breaks += 1
        if p_val < best_p:
            best_p = p_val
            best_point = i

    return n_breaks, best_point, best_p


def count_missing_months(db, site_id, end_date):
    """Count months with zero or missing data in the series."""
    df = db.get_sales_data(
        end_date=end_date, site_ids=[site_id], exclude_estimated=True
    )
    if df.empty:
        return np.nan, np.nan
    df["date"] = pd.to_datetime(df["day"])
    df["ym"] = df["date"].dt.to_period("M")
    monthly = df.groupby("ym")["volume"].sum()

    # Full range
    full_range = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    missing = len(full_range) - len(monthly)
    zero_months = (monthly <= 0).sum()
    return missing, zero_months


def count_estimated_pct(db, site_id, end_date):
    """Percentage of daily records flagged as estimated."""
    all_data = db.get_sales_data(end_date=end_date, site_ids=[site_id])
    if all_data.empty:
        return np.nan
    if "is_estimated" not in all_data.columns:
        return 0.0
    return all_data["is_estimated"].sum() / len(all_data) * 100


def recent_volume_change(series, months=6):
    """YoY % change in recent vs prior period average."""
    if len(series) < months * 2:
        return np.nan
    recent = series.iloc[-months:].mean()
    prior = series.iloc[-months * 2 : -months].mean()
    if prior == 0:
        return np.nan
    return (recent - prior) / prior * 100


# ── main analysis ────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("HIGH-ERROR SITE INVESTIGATION")
    print("Rigorous multi-dimensional analysis")
    print("=" * 80)

    db = FuelDatabase(DB_PATH)
    forecaster = FuelForecaster(db, min_months_data=MIN_MONTHS)

    # ── Phase 1: Re-run backtest capturing per-model results ─────────────
    print("\n[Phase 1] Re-running backtest with per-model decomposition...")

    stats_info = db.get_summary_stats()
    max_date = pd.to_datetime(stats_info["date_range"].split(" to ")[1])
    last_complete = (max_date.to_period("M") - 1).to_timestamp()

    test_months = []
    for i in range(BACKTEST_MONTHS):
        m = (last_complete.to_period("M") - i).to_timestamp()
        test_months.append(m)
    test_months.sort()

    sites_df = db.get_distinct_sites()
    earliest_test = test_months[0]

    # Pre-filter sites
    qualified_sites = []
    for _, row in sites_df.iterrows():
        sid = row["site_id"]
        cutoff = (earliest_test - pd.DateOffset(months=HORIZON, days=1)).strftime("%Y-%m-%d")
        try:
            data = forecaster.prepare_monthly_data(
                site_id=sid, end_date=cutoff, handle_outliers=False, fill_gaps=False
            )
        except ValueError:
            continue
        if len(data) >= MIN_MONTHS:
            qualified_sites.append(sid)

    print(f"  Qualified sites: {len(qualified_sites)}")
    print(f"  Test period: {test_months[0].strftime('%Y-%m')} to {test_months[-1].strftime('%Y-%m')}")

    # Collect per-model results
    all_results = []
    total = len(qualified_sites) * len(test_months)
    done = 0

    for sid in qualified_sites:
        for tm in test_months:
            done += 1
            if done % 100 == 0:
                print(f"  Progress: {done}/{total}", end="\r")

            target = tm.strftime("%Y-%m")

            # Actual
            start = f"{target}-01"
            end = (pd.to_datetime(start) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
            actual_df = db.get_sales_data(
                start_date=start, end_date=end, site_ids=[sid], exclude_estimated=True
            )
            if actual_df.empty:
                continue
            actual = float(actual_df["volume"].sum())
            if actual <= 0:
                continue

            cutoff = (tm - pd.DateOffset(months=HORIZON, days=1)).strftime("%Y-%m-%d")

            try:
                md = forecaster.prepare_monthly_data(
                    site_id=sid, end_date=cutoff, handle_outliers=True, fill_gaps=True
                )
                md_raw = forecaster.prepare_monthly_data(
                    site_id=sid, end_date=cutoff, handle_outliers=False, fill_gaps=False
                )

                if len(md) < MIN_MONTHS:
                    continue

                fc = forecaster.generate_forecast(
                    target_month=target, site_id=sid,
                    monthly_data=md, monthly_data_raw=md_raw, show_yoy=False,
                )

                for _, frow in fc.iterrows():
                    model = frow["model"]
                    fv = float(frow["forecast_volume"])
                    err = abs(fv - actual) / actual * 100
                    bias = (fv - actual) / actual * 100
                    all_results.append({
                        "site_id": sid, "month": target, "model": model,
                        "forecast": round(fv, 1), "actual": round(actual, 1),
                        "error_pct": round(err, 2), "bias_pct": round(bias, 2),
                    })
            except Exception:
                continue

    print(f"  Progress: {done}/{total} - done")

    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        print("No results. Exiting.")
        db.close()
        return

    # Site-level MAPE for ensemble
    ensemble = results_df[results_df["model"] == "ENSEMBLE"]
    site_mape = (
        ensemble.groupby("site_id")
        .agg(
            mape=("error_pct", "mean"),
            bias=("bias_pct", "mean"),
            n_months=("error_pct", "count"),
            median_error=("error_pct", "median"),
            max_error=("error_pct", "max"),
        )
        .reset_index()
    )

    high_error = site_mape[site_mape["mape"] >= HIGH_ERROR_THRESHOLD].copy()
    low_error = site_mape[site_mape["mape"] < 5].copy()

    print(f"\n  Total sites: {len(site_mape)}")
    print(f"  High-error sites (MAPE >= {HIGH_ERROR_THRESHOLD}%): {len(high_error)}")
    print(f"  Low-error sites (MAPE < 5%): {len(low_error)}")

    # ── Phase 2: Per-model performance comparison ────────────────────────
    print("\n" + "=" * 80)
    print("[Phase 2] Per-model performance comparison")
    print("=" * 80)

    model_perf = (
        results_df.groupby("model")
        .agg(
            mean_mape=("error_pct", "mean"),
            median_mape=("error_pct", "median"),
            mean_bias=("bias_pct", "mean"),
            n=("error_pct", "count"),
        )
        .round(2)
    )
    print(f"\n{model_perf.to_string()}\n")

    # Per-model MAPE by site group
    for group_name, group_sites in [("HIGH-ERROR", high_error), ("LOW-ERROR", low_error)]:
        group_ids = set(group_sites["site_id"])
        group_data = results_df[results_df["site_id"].isin(group_ids)]
        gperf = (
            group_data.groupby("model")
            .agg(mean_mape=("error_pct", "mean"), mean_bias=("bias_pct", "mean"))
            .round(2)
        )
        print(f"  {group_name} sites ({len(group_ids)} sites):")
        print(f"  {gperf.to_string()}\n")

    # Which model "wins" per site?
    model_wins = {}
    for model in ["ets", "snaive", "ENSEMBLE"]:
        mdata = results_df[results_df["model"] == model]
        m_site_mape = mdata.groupby("site_id")["error_pct"].mean()
        model_wins[model] = m_site_mape

    wins_df = pd.DataFrame(model_wins)
    wins_df["best_model"] = wins_df[["ets", "snaive"]].idxmin(axis=1)
    print(f"  Model that would have been best (lowest MAPE):")
    print(f"    ETS wins: {(wins_df['best_model'] == 'ets').sum()} sites")
    print(f"    Snaive wins: {(wins_df['best_model'] == 'snaive').sum()} sites")
    ensemble_gap = (wins_df['ENSEMBLE'] - wins_df[['ets', 'snaive']].min(axis=1)).mean()
    print(f"    Ensemble vs oracle per-site picker: {ensemble_gap:+.2f}pp "
          f"({'ensemble is worse' if ensemble_gap > 0 else 'ensemble is better'})")

    # ── Phase 3: Data quality & statistical profiling ────────────────────
    print("\n" + "=" * 80)
    print("[Phase 3] Data quality & statistical profiling")
    print("=" * 80)
    print("  Building site feature matrix...")

    all_site_features = []
    cutoff_for_profiling = (earliest_test - pd.DateOffset(months=HORIZON, days=1)).strftime("%Y-%m-%d")

    for idx, sid in enumerate(qualified_sites):
        if (idx + 1) % 50 == 0:
            print(f"  Profiling: {idx + 1}/{len(qualified_sites)}", end="\r")

        try:
            raw = forecaster.prepare_monthly_data(
                site_id=sid, end_date=cutoff_for_profiling,
                handle_outliers=False, fill_gaps=False,
            )
            vol = raw["volume"]

            missing, zero_m = count_missing_months(db, sid, cutoff_for_profiling)
            est_pct = count_estimated_pct(db, sid, cutoff_for_profiling)

            feats = {
                "site_id": sid,
                "n_months": len(raw),
                "mean_volume": vol.mean(),
                "median_volume": vol.median(),
                "cv_pct": coefficient_of_variation(vol),
                "seasonality": seasonality_strength(raw["date"], vol),
                "trend_r2": trend_strength(vol),
                "trend_slope_pct": trend_slope_pct(vol),
                "missing_months": missing,
                "zero_months": zero_m,
                "estimated_pct": est_pct,
                "recent_change_pct": recent_volume_change(vol),
                "min_volume": vol.min(),
                "max_volume": vol.max(),
                "range_ratio": vol.max() / vol.min() if vol.min() > 0 else np.nan,
            }

            # Structural break detection
            n_breaks, break_point, break_p = detect_structural_breaks(vol)
            feats["n_structural_breaks"] = n_breaks
            feats["break_p_value"] = break_p

            # Recent volatility (last 6 months CV)
            if len(vol) >= 6:
                recent = vol.tail(6)
                feats["recent_cv_pct"] = coefficient_of_variation(recent)
            else:
                feats["recent_cv_pct"] = np.nan

            all_site_features.append(feats)
        except Exception:
            continue

    print(f"  Profiling: {len(qualified_sites)}/{len(qualified_sites)} - done")

    features_df = pd.DataFrame(all_site_features)

    # Merge with MAPE
    analysis = features_df.merge(
        site_mape[["site_id", "mape", "bias", "median_error", "max_error"]],
        on="site_id", how="inner"
    )
    analysis["is_high_error"] = analysis["mape"] >= HIGH_ERROR_THRESHOLD

    # ── Phase 4: Statistical comparison of high vs low error groups ──────
    print("\n" + "=" * 80)
    print("[Phase 4] Statistical comparison: High-error vs Low-error sites")
    print("=" * 80)

    comparison_features = [
        "n_months", "mean_volume", "cv_pct", "seasonality", "trend_r2",
        "trend_slope_pct", "missing_months", "zero_months", "estimated_pct",
        "recent_change_pct", "range_ratio", "n_structural_breaks", "recent_cv_pct",
    ]

    high = analysis[analysis["mape"] >= HIGH_ERROR_THRESHOLD]
    low = analysis[analysis["mape"] < 5]

    print(f"\n  Comparing: {len(high)} high-error (MAPE >= {HIGH_ERROR_THRESHOLD}%) vs "
          f"{len(low)} low-error (MAPE < 5%) sites\n")
    print(f"  {'Feature':<25} {'High-Err Mean':>14} {'Low-Err Mean':>14} {'Diff':>10} {'p-value':>10} {'Sig':>5}")
    print("  " + "-" * 80)

    significant_features = []
    for feat in comparison_features:
        h_vals = high[feat].dropna()
        l_vals = low[feat].dropna()
        if len(h_vals) < 5 or len(l_vals) < 5:
            continue

        # Mann-Whitney U test (non-parametric, no normality assumption)
        try:
            u_stat, p_val = sp_stats.mannwhitneyu(h_vals, l_vals, alternative="two-sided")
        except Exception:
            p_val = np.nan

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""

        print(f"  {feat:<25} {h_vals.mean():>14.2f} {l_vals.mean():>14.2f} "
              f"{h_vals.mean() - l_vals.mean():>10.2f} {p_val:>10.4f} {sig:>5}")

        if p_val < 0.05:
            significant_features.append((feat, p_val, h_vals.mean() - l_vals.mean()))

    # Effect sizes (Cohen's d)
    print(f"\n  Effect sizes (Cohen's d) for significant features:")
    for feat, p, diff in sorted(significant_features, key=lambda x: x[1]):
        h_vals = high[feat].dropna()
        l_vals = low[feat].dropna()
        pooled_std = np.sqrt((h_vals.std() ** 2 + l_vals.std() ** 2) / 2)
        d = diff / pooled_std if pooled_std > 0 else np.nan
        magnitude = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"
        print(f"    {feat:<25} d={d:>7.3f} ({magnitude})")

    # ── Phase 5: Correlation analysis ────────────────────────────────────
    print("\n" + "=" * 80)
    print("[Phase 5] Correlation with MAPE (Spearman rank)")
    print("=" * 80)

    print(f"\n  {'Feature':<25} {'rho':>8} {'p-value':>10} {'Sig':>5}")
    print("  " + "-" * 50)

    corr_results = []
    for feat in comparison_features:
        vals = analysis[[feat, "mape"]].dropna()
        if len(vals) < 10:
            continue
        # Skip constant columns (e.g., all zeros) — spearmanr returns NaN
        if vals[feat].nunique() < 2:
            print(f"  {feat:<25} {'(constant)':>8} {'':>10} {'':>5}")
            continue
        rho, p = sp_stats.spearmanr(vals[feat], vals["mape"])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {feat:<25} {rho:>8.3f} {p:>10.4f} {sig:>5}")
        corr_results.append((feat, rho, p))

    # ── Phase 6: Error taxonomy ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("[Phase 6] Error taxonomy — classifying root causes")
    print("=" * 80)

    # Classify each high-error site
    taxonomy = {
        "high_volatility": [],      # CV > 20%
        "structural_break": [],     # Significant structural breaks
        "strong_trend": [],         # Trend R² > 0.3 and slope > 2%/month
        "data_quality": [],         # Missing months or high estimated %
        "low_volume": [],           # Mean volume < 10th percentile
        "recent_level_shift": [],   # Recent change > 20%
        "unexplained": [],          # None of the above
    }

    vol_10pct = analysis["mean_volume"].quantile(0.10)

    for _, row in high.iterrows():
        sid = row["site_id"]
        classified = False

        if row.get("cv_pct", 0) > 20:
            taxonomy["high_volatility"].append(sid)
            classified = True
        if row.get("n_structural_breaks", 0) >= 3:
            taxonomy["structural_break"].append(sid)
            classified = True
        if row.get("trend_r2", 0) > 0.3 and abs(row.get("trend_slope_pct", 0)) > 2:
            taxonomy["strong_trend"].append(sid)
            classified = True
        if row.get("missing_months", 0) > 2 or row.get("estimated_pct", 0) > 10:
            taxonomy["data_quality"].append(sid)
            classified = True
        if row.get("mean_volume", np.inf) < vol_10pct:
            taxonomy["low_volume"].append(sid)
            classified = True
        if abs(row.get("recent_change_pct", 0)) > 20:
            taxonomy["recent_level_shift"].append(sid)
            classified = True
        if not classified:
            taxonomy["unexplained"].append(sid)

    print(f"\n  Root cause categories (sites can appear in multiple):\n")
    total_high = len(high)
    for cause, sites in sorted(taxonomy.items(), key=lambda x: -len(x[1])):
        pct = len(sites) / total_high * 100 if total_high > 0 else 0
        print(f"    {cause:<25} {len(sites):>4} sites ({pct:>5.1f}%)")
        if sites and cause != "unexplained":
            # Show top 3 worst in this category
            cat_mape = high[high["site_id"].isin(sites)].nlargest(3, "mape")
            for _, r in cat_mape.iterrows():
                print(f"      -> Site {r['site_id']}: MAPE={r['mape']:.1f}%, "
                      f"CV={r.get('cv_pct', 0):.1f}%, "
                      f"breaks={int(r.get('n_structural_breaks', 0))}, "
                      f"trend_slope={r.get('trend_slope_pct', 0):.2f}%/mo")

    # ── Phase 7: Deep dive on worst 10 sites ─────────────────────────────
    print("\n" + "=" * 80)
    print("[Phase 7] Deep dive — 10 worst sites")
    print("=" * 80)

    worst_10 = site_mape.nlargest(10, "mape")

    for _, srow in worst_10.iterrows():
        sid = srow["site_id"]
        mape_val = srow["mape"]
        bias_val = srow["bias"]

        print(f"\n  --- Site {sid} (MAPE={mape_val:.1f}%, Bias={bias_val:+.1f}%) ---")

        # Get features
        sf = analysis[analysis["site_id"] == sid]
        if not sf.empty:
            sf = sf.iloc[0]
            print(f"    Months of data:     {sf['n_months']}")
            print(f"    Mean volume:        {sf['mean_volume']:,.0f}")
            print(f"    CV:                 {sf['cv_pct']:.1f}%")
            print(f"    Seasonality:        {sf['seasonality']:.3f}")
            print(f"    Trend R²:           {sf['trend_r2']:.3f}")
            print(f"    Trend slope:        {sf['trend_slope_pct']:.2f}%/month")
            print(f"    Structural breaks:  {int(sf['n_structural_breaks'])}")
            print(f"    Missing months:     {int(sf['missing_months'])}")
            print(f"    Recent change:      {sf['recent_change_pct']:+.1f}%")
            print(f"    Range ratio:        {sf['range_ratio']:.1f}x")

        # Per-model performance for this site
        site_results = results_df[results_df["site_id"] == sid]
        for model in ["ets", "snaive", "ENSEMBLE"]:
            mr = site_results[site_results["model"] == model]
            if not mr.empty:
                print(f"    {model:<10} MAPE={mr['error_pct'].mean():.1f}%, "
                      f"Bias={mr['bias_pct'].mean():+.1f}%")

        # Month-by-month errors
        ens_results = site_results[site_results["model"] == "ENSEMBLE"].sort_values("month")
        print(f"    Month-by-month:")
        for _, er in ens_results.iterrows():
            direction = "OVER " if er["bias_pct"] > 0 else "UNDER"
            print(f"      {er['month']}: forecast={er['forecast']:>10,.0f}  "
                  f"actual={er['actual']:>10,.0f}  "
                  f"err={er['error_pct']:>5.1f}%  {direction}")

    # ── Phase 8: Actionable summary ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("[Phase 8] ACTIONABLE SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    # Overall stats
    print(f"\n  Overall backtest: {len(site_mape)} sites, "
          f"Mean MAPE={site_mape['mape'].mean():.1f}%, "
          f"Median MAPE={site_mape['mape'].median():.1f}%")

    # Key findings
    print(f"\n  KEY FINDINGS:")

    # 1. Model comparison
    ets_mape = results_df[results_df["model"] == "ets"]["error_pct"].mean()
    snaive_mape = results_df[results_df["model"] == "snaive"]["error_pct"].mean()
    ens_mape = results_df[results_df["model"] == "ENSEMBLE"]["error_pct"].mean()
    print(f"\n  1. Model performance: ETS={ets_mape:.1f}%, Snaive={snaive_mape:.1f}%, Ensemble={ens_mape:.1f}%")
    better = "ETS" if ets_mape < snaive_mape else "Snaive"
    print(f"     {better} is the stronger individual model overall.")

    # 2. Dominant error drivers
    print(f"\n  2. Dominant error drivers (sorted by prevalence):")
    for cause, sites in sorted(taxonomy.items(), key=lambda x: -len(x[1])):
        if sites:
            print(f"     - {cause}: {len(sites)} sites")

    # 3. Bias
    overall_bias = ensemble["bias_pct"].mean()
    print(f"\n  3. Systematic bias: {overall_bias:+.1f}% "
          f"({'under-forecasting' if overall_bias < 0 else 'over-forecasting'})")

    # 4. Top correlates of error
    print(f"\n  4. Strongest predictors of forecast error:")
    top_corrs = sorted(corr_results, key=lambda x: abs(x[1]), reverse=True)[:5]
    for feat, rho, p in top_corrs:
        print(f"     - {feat}: rho={rho:+.3f} (p={p:.4f})")

    # 5. Recommendations
    print(f"\n  RECOMMENDATIONS:")

    if len(taxonomy["high_volatility"]) > len(high) * 0.3:
        print(f"  -> High volatility is the #1 issue ({len(taxonomy['high_volatility'])} sites).")
        print(f"     Consider: wider prediction intervals, volatility-weighted ensemble,")
        print(f"     or GARCH-style variance modeling for these sites.")

    if len(taxonomy["recent_level_shift"]) > 10:
        print(f"  -> {len(taxonomy['recent_level_shift'])} sites had recent level shifts (>20% change).")
        print(f"     Consider: shorter training window or regime-switching detection")
        print(f"     to down-weight stale history.")

    if len(taxonomy["structural_break"]) > 10:
        print(f"  -> {len(taxonomy['structural_break'])} sites have structural breaks in their history.")
        print(f"     Consider: post-break-only training or structural break pre-filtering.")

    if abs(overall_bias) > 2:
        direction = "under" if overall_bias < 0 else "over"
        print(f"  -> Systematic {direction}-forecasting bias of {abs(overall_bias):.1f}%.")
        print(f"     Consider: bias correction term or asymmetric loss function.")

    if len(taxonomy["data_quality"]) > 10:
        print(f"  -> {len(taxonomy['data_quality'])} sites have data quality issues.")
        print(f"     Consider: data completeness checks before forecasting,")
        print(f"     flagging sites with >2 missing months for manual review.")

    ets_better_count = (wins_df["best_model"] == "ets").sum()
    snaive_better_count = (wins_df["best_model"] == "snaive").sum()
    print(f"  -> Model selection: ETS is better for {ets_better_count} sites, "
          f"Snaive for {snaive_better_count}.")
    print(f"     Consider: per-site model selection based on CV/seasonality features,")
    print(f"     or a weighted ensemble where weights adapt to site characteristics.")

    # ── Save detailed results ────────────────────────────────────────────
    output_path = "investigation_results.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Full feature matrix with MAPE
        analysis.sort_values("mape", ascending=False).to_excel(
            writer, sheet_name="Site Features", index=False
        )
        # Per-model results
        results_df.to_excel(writer, sheet_name="Per-Model Results", index=False)
        # Taxonomy
        tax_rows = []
        for cause, sites in taxonomy.items():
            for s in sites:
                tax_rows.append({"site_id": s, "root_cause": cause})
        if tax_rows:
            pd.DataFrame(tax_rows).to_excel(
                writer, sheet_name="Error Taxonomy", index=False
            )
        # Model wins
        wins_df.reset_index().to_excel(writer, sheet_name="Model Comparison", index=False)

    print(f"\n  Detailed results saved to: {output_path}")

    db.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
