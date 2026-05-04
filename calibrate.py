"""
Calibration Module - Learn per-site optimal ensemble weights and prediction intervals.

Runs backtests, computes inverse-MAPE weights with shrinkage fallback,
builds calibrated residual distributions, and stores everything in the database.

Usage:
    python cli.py calibrate [--months 12] [--min-months 24] [--horizon 2]
"""

import logging

import numpy as np
import pandas as pd

from backtest import _run_backtest_inner, _validate_month_params, build_per_model_site_metrics
from database import FuelDatabase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

WEIGHT_FLOOR = 0.10
DEFAULT_MAPE_FOR_MISSING = 50.0
SHRINKAGE_MIN_OBSERVATIONS = 6

# Sites with >= this weight for one model are classified as "dominant"
DOMINANCE_THRESHOLD = 0.65


def _validate_calibration_params(months, min_months, horizon, weight_floor):
    """Validate user-provided calibration parameters."""
    _validate_month_params(months, min_months, horizon)
    if not np.isfinite(weight_floor):
        raise ValueError("weight_floor must be a finite number")
    if weight_floor < 0:
        raise ValueError("weight_floor must be >= 0")


def compute_optimal_weights(site_model_mapes, weight_floor=WEIGHT_FLOOR):
    """
    Inverse-MAPE weighting with floor and normalization.

    Guarantees each model gets at least weight_floor after normalization.

    Args:
        site_model_mapes: dict of {model_name: mape_pct}
        weight_floor: minimum weight per model

    Returns:
        dict of {model_name: weight} summing to 1.0
    """
    n = len(site_model_mapes)
    if n == 0:
        return {}
    if n == 1:
        return {m: 1.0 for m in site_model_mapes}

    # If floors consume all budget, return equal weights
    total_floor = weight_floor * n
    if total_floor >= 1.0:
        return {m: 1.0 / n for m in site_model_mapes}

    def _safe_mape(value):
        # NaN flows through max() unchanged, so guard explicitly before clamping.
        if not np.isfinite(value):
            return DEFAULT_MAPE_FOR_MISSING
        return max(value, 0.1)

    raw = {m: 1.0 / _safe_mape(mape) for m, mape in site_model_mapes.items()}
    total = sum(raw.values())
    normed = {m: v / total for m, v in raw.items()}

    # Distribute remaining budget (1 - total_floor) proportionally
    remaining = 1.0 - total_floor
    return {m: weight_floor + remaining * normed[m] for m in normed}


def _blend_weights(site_weights, fallback_weights, site_n_months,
                   min_obs=SHRINKAGE_MIN_OBSERVATIONS):
    """
    Shrinkage blend: if site has few observations, pull toward fallback.

    Uses linear interpolation: alpha = min(1, n_months / min_obs).
    """
    if site_n_months >= min_obs:
        return site_weights

    alpha = site_n_months / min_obs
    blended = {}
    all_models = set(site_weights) | set(fallback_weights)
    for m in all_models:
        sw = site_weights.get(m, 0.5)
        fw = fallback_weights.get(m, 0.5)
        blended[m] = alpha * sw + (1 - alpha) * fw
    s = sum(blended.values())
    return {m: w / s for m, w in blended.items()}


def _compute_residual_stats(residuals):
    """Compute residual distribution stats from an array of residual values."""
    residuals = np.array(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 2:
        return None
    return {
        "residual_std": float(np.std(residuals, ddof=1)),
        "residual_p10": float(np.percentile(residuals, 10)),
        "residual_p90": float(np.percentile(residuals, 90)),
        "n_observations": len(residuals),
    }


def _compute_ensemble_residuals(per_model_df, weights_lookup):
    """
    Compute ensemble residuals using calibrated weights.

    For each (site, [grade,] month), blends per-model forecasts with their
    calibrated weights, then computes residual = (ensemble - actual) / actual.

    When per_model_df has a "grade" column, the result is keyed at
    (site_id, grade, month) so interval calibration can segment by grade.
    Otherwise returns (site_id, month, residual) for backward compatibility.
    """
    has_grade = (
        per_model_df is not None
        and not per_model_df.empty
        and "grade" in per_model_df.columns
    )
    columns = (
        ["site_id", "grade", "month", "residual"]
        if has_grade
        else ["site_id", "month", "residual"]
    )
    if per_model_df is None or per_model_df.empty:
        return pd.DataFrame(columns=columns)

    group_keys = ["site_id", "grade", "month"] if has_grade else ["site_id", "month"]

    records = []
    grouped = per_model_df.groupby(group_keys)
    for key, group in grouped:
        if has_grade:
            site_id, grade, month = key
        else:
            site_id, month = key
            grade = None

        ensemble_vol = 0.0
        total_weight = 0.0
        actual = None
        for _, row in group.iterrows():
            if has_grade:
                w = weights_lookup.get(
                    (row["site_id"], row["grade"], row["model"]),
                    weights_lookup.get(
                        (row["site_id"], "ALL", row["model"]),
                        weights_lookup.get((row["site_id"], row["model"]), 0.0),
                    ),
                )
            else:
                w = weights_lookup.get(
                    (row["site_id"], "ALL", row["model"]),
                    weights_lookup.get((row["site_id"], row["model"]), 0.0),
                )
            forecast = row["forecast"]
            if not np.isfinite(forecast):
                continue
            ensemble_vol += w * forecast
            total_weight += w
            actual = row["actual"]

        if total_weight > 0 and actual is not None and np.isfinite(actual) and actual > 0:
            ensemble_vol /= total_weight
            record = {
                "site_id": site_id,
                "month": month,
                "residual": (ensemble_vol - actual) / actual,
            }
            if has_grade:
                record["grade"] = grade
            records.append(record)
    return pd.DataFrame(records, columns=columns)


def run_calibration(db_path="fuel_sales.db", months=12, min_months=24,
                    horizon=2, weight_floor=WEIGHT_FLOOR, output=None):
    """
    Main calibration entry point.

    1. Run backtest with per-model tracking
    2. Compute per-site optimal weights with shrinkage
    3. Compute residual distributions for interval calibration
    4. Store everything in the database
    5. Print summary

    Returns:
        run_id of the stored calibration run, or None on failure
    """
    _validate_calibration_params(months, min_months, horizon, weight_floor)

    db = FuelDatabase(db_path)
    try:
        return _run_calibration_inner(db, months, min_months, horizon,
                                      weight_floor, output)
    finally:
        db.close()


def _run_calibration_inner(db, months, min_months, horizon, weight_floor, output):
    """Core calibration logic."""
    print("Running backtest for calibration...")
    result = _run_backtest_inner(db, months, None, min_months, horizon,
                                 return_per_model=True)

    results_df, site_mape, per_model_df = result

    if results_df is None or per_model_df is None or per_model_df.empty:
        print("Calibration failed: no backtest results.")
        return None

    overall_mape = float(results_df["error_pct"].mean())

    # -- Step 1: Per-site per-model MAPE --
    pm_metrics = build_per_model_site_metrics(per_model_df)
    if pm_metrics.empty:
        print("Calibration failed: no per-model metrics.")
        return None

    # -- Step 2: Compute optimal weights per site/grade forecast unit --
    has_grade_metrics = "grade" in pm_metrics.columns
    group_cols = ["site_id", "grade"] if has_grade_metrics else ["site_id"]
    calibration_units = list(pm_metrics.groupby(group_cols, dropna=False))
    site_ids = pm_metrics["site_id"].unique()

    # Global average MAPE per model (fallback)
    global_mapes = (
        pm_metrics.groupby("model")["mape_pct"].mean().to_dict()
    )
    global_weights = compute_optimal_weights(global_mapes, weight_floor)

    weight_records = []
    for key, site_data in calibration_units:
        if has_grade_metrics:
            site_id, grade = key
            grade = str(grade)
        else:
            site_id = key[0] if isinstance(key, tuple) else key
            grade = "ALL"
        site_mapes = dict(zip(site_data["model"], site_data["mape_pct"]))
        # Use the best-observed model's n_months as the site maturity signal,
        # rather than penalizing the whole site for the least-observed model.
        site_n = int(site_data["n_months"].max()) if not site_data.empty else 0

        # Fill missing models with pessimistic default
        for model in global_mapes:
            if model not in site_mapes:
                site_mapes[model] = DEFAULT_MAPE_FOR_MISSING

        raw_weights = compute_optimal_weights(site_mapes, weight_floor)

        # Shrinkage toward global
        final_weights = _blend_weights(raw_weights, global_weights, site_n)

        for model_name, weight in final_weights.items():
            mape_val = site_mapes.get(model_name)
            n_val = int(site_data.loc[
                site_data["model"] == model_name, "n_months"
            ].iloc[0]) if model_name in site_data["model"].values else 0
            weight_records.append({
                "site_id": site_id,
                "grade": grade,
                "model_name": model_name,
                "weight": round(weight, 4),
                "mape_pct": round(mape_val, 2) if mape_val is not None else None,
                "n_months": n_val,
            })

    # -- Step 3: Interval calibration (residual distributions) --
    # Compute ensemble residuals using the calibrated weights, rather than
    # pooling raw per-model residuals (which would inflate intervals for
    # models with high error but low weight).
    weights_lookup = {
        (r["site_id"], r.get("grade", "ALL"), r["model_name"]): r["weight"]
        for r in weight_records
    }
    ensemble_residuals = _compute_ensemble_residuals(per_model_df, weights_lookup)
    if ensemble_residuals.empty:
        logger.warning("No ensemble residuals produced; skipping interval calibration.")

    segments = []
    has_grade_residuals = (
        not ensemble_residuals.empty and "grade" in ensemble_residuals.columns
    )

    if has_grade_residuals:
        # Most specific: per (site, grade)
        for (site_id, grade), group in ensemble_residuals.groupby(["site_id", "grade"]):
            stats = _compute_residual_stats(group["residual"].values)
            if stats:
                stats["segment"] = f"site:{site_id}:grade:{grade}"
                segments.append(stats)

        # Per grade (pooled across sites) — fallback when a (site, grade)
        # pair has too few observations for its own distribution.
        for grade, group in ensemble_residuals.groupby("grade"):
            stats = _compute_residual_stats(group["residual"].values)
            if stats:
                stats["segment"] = f"grade:{grade}"
                segments.append(stats)

    # Per site (aggregate across grades) — kept for sites with many grades
    # where the pooled site-level distribution is still informative.
    for site_id, group in ensemble_residuals.groupby("site_id"):
        stats = _compute_residual_stats(group["residual"].values)
        if stats:
            stats["segment"] = f"site:{site_id}"
            segments.append(stats)

    # Global residuals — final fallback.
    global_stats = _compute_residual_stats(
        ensemble_residuals["residual"].values
    )
    if global_stats:
        global_stats["segment"] = "global"
        segments.append(global_stats)

    # -- Step 4: Store in DB --
    run_id = db.save_calibration_run({
        "backtest_months": months,
        "horizon": horizon,
        "min_months": min_months,
        "sites_calibrated": len(site_ids),
        "overall_mape": round(overall_mape, 2),
    })
    db.save_site_weights(run_id, weight_records)
    if segments:
        db.save_interval_calibration(run_id, segments)

    # -- Step 5: Summary --
    weights_df = pd.DataFrame(weight_records)
    _print_summary(weights_df, overall_mape, len(site_ids), run_id)

    # Optional Excel export
    if output:
        _export_calibration_report(weights_df, pm_metrics, segments, output)

    return run_id


def _print_summary(weights_df, overall_mape, n_sites, run_id):
    """Print calibration summary to console."""
    print(f"\nCalibration complete (run_id={run_id})")
    print(f"  Sites calibrated: {n_sites}")
    if "grade" in weights_df.columns:
        combos = weights_df[["site_id", "grade"]].drop_duplicates()
        print(f"  Site-grade weights calibrated: {len(combos)}")
    print(f"  Overall MAPE: {overall_mape:.1f}%\n")

    # Classify sites
    ets_dominant = 0
    snaive_dominant = 0
    balanced = 0

    unit_cols = ["site_id", "grade"] if "grade" in weights_df.columns else ["site_id"]
    for _, site_w in weights_df.groupby(unit_cols):
        ets_row = site_w[site_w["model_name"] == "ets"]
        snaive_row = site_w[site_w["model_name"] == "snaive"]
        ets_w = float(ets_row["weight"].iloc[0]) if not ets_row.empty else 0.5
        snaive_w = float(snaive_row["weight"].iloc[0]) if not snaive_row.empty else 0.5

        if ets_w >= DOMINANCE_THRESHOLD:
            ets_dominant += 1
        elif snaive_w >= DOMINANCE_THRESHOLD:
            snaive_dominant += 1
        else:
            balanced += 1

    print("  Site classification:")
    print(f"    ETS-dominant (>={DOMINANCE_THRESHOLD*100:.0f}%): {ets_dominant}")
    print(f"    SNaive-dominant (>={DOMINANCE_THRESHOLD*100:.0f}%): {snaive_dominant}")
    print(f"    Balanced: {balanced}")


def _export_calibration_report(weights_df, pm_metrics, segments, output):
    """Write calibration report to Excel."""
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        weights_df.to_excel(writer, sheet_name="Weights", index=False)
        pm_metrics.round(2).to_excel(writer, sheet_name="Per-Model MAPE", index=False)
        if segments:
            seg_df = pd.DataFrame(segments)
            seg_df.to_excel(writer, sheet_name="Interval Calibration", index=False)
    print(f"\nCalibration report saved to: {output}")
