"""
Forecaster Module - High-level forecasting interface with progress tracking
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
import logging
from pathlib import Path

# Handle both relative and absolute imports
try:
    from .models import get_available_models
except ImportError:
    from models import get_available_models

logger = logging.getLogger(__name__)


class FuelForecaster:
    """Main forecasting interface"""

    def __init__(self, database, min_months_data: int = 24):
        """
        Initialize forecaster

        Args:
            database: FuelDatabase instance
            min_months_data: Minimum months of data required (default: 24)
        """
        self.db = database
        self.min_months_data = min_months_data
        self.models = {}
        # Softer floor for sparse site/grade combos; allows forecasting with fewer months
        self.soft_min_months = max(6, min(12, self.min_months_data))

    def prepare_monthly_data(
        self,
        site_id: Optional[str] = None,
        grade: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        handle_outliers: bool = True,
        fill_gaps: bool = True,
    ) -> pd.DataFrame:
        """
        Prepare monthly aggregated data with outlier detection

        Args:
            site_id: Optional site ID filter
            grade: Optional grade filter
            start_date: Optional start date
            end_date: Optional end date
            handle_outliers: Apply outlier detection/handling (default: True)
            fill_gaps: Fill missing months via interpolation (default: True).
                       Set False for snaive to preserve exact historical values.

        Returns:
            DataFrame with monthly volume data
        """
        site_ids = [site_id] if site_id else None
        grades = [grade] if grade else None

        df = self.db.get_sales_data(
            start_date=start_date,
            end_date=end_date,
            site_ids=site_ids,
            grades=grades,
            exclude_estimated=True,
        )

        if df.empty:
            raise ValueError("No data available")

        # Aggregate to monthly
        df["date"] = pd.to_datetime(df["day"])
        df["year_month"] = df["date"].dt.to_period("M")

        monthly = (
            df.groupby("year_month")
            .agg({"volume": "sum"})
            .reset_index()
        )

        monthly["date"] = monthly["year_month"].dt.to_timestamp()
        monthly = monthly.sort_values("date").reset_index(drop=True)

        result = monthly[["date", "volume"]].copy()

        # Fill missing months to keep seasonality aligned on sparse series
        # Skip for snaive which needs exact historical values without interpolation
        if fill_gaps:
            result = self._fill_monthly_gaps(result, site_id=site_id, grade=grade)

        # Outlier detection and handling (critical for individual site accuracy)
        if handle_outliers and len(result) >= 12:
            result = self._handle_outliers(result, site_id)

        return result

    def _fill_monthly_gaps(
        self, data: pd.DataFrame, site_id: Optional[str] = None, grade: Optional[str] = None
    ) -> pd.DataFrame:
        """Reindex to full monthly range and fill missing months."""
        if data.empty:
            return data

        start = data["date"].min()
        end = data["date"].max()
        full_range = pd.date_range(start=start, end=end, freq="MS")

        reindexed = data.set_index("date").reindex(full_range)
        missing = int(reindexed["volume"].isna().sum())

        if missing > 0:
            reindexed["volume"] = (
                reindexed["volume"]
                .interpolate(limit_direction="both")
                .ffill()
                .bfill()
            )
            reindexed["volume"] = reindexed["volume"].clip(lower=0.0)

            site_label = f"{site_id}" if site_id else "ALL"
            grade_label = f"{grade}" if grade else "ALL"
            logger.info(
                f"Filled {missing} missing month(s) for site {site_label}, grade {grade_label}"
            )

        reindexed = reindexed.reset_index().rename(columns={"index": "date"})
        return reindexed

    def _handle_outliers(
        self, data: pd.DataFrame, site_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Detect and handle outliers using rolling-window MAD method.

        Uses a rolling window (last 18 months) to compute statistics, which adapts
        to legitimate business changes (growth, new customers, etc.) while still
        catching true data errors (typos, system glitches).

        Outliers are capped at boundaries rather than removed to preserve time series continuity.
        """
        df = data.copy()

        # Use rolling window for statistics (adapts to business changes)
        # 18 months captures seasonality while being responsive to trends
        window_size = min(18, len(df))
        recent_data = df.tail(window_size)
        recent_volumes = recent_data["volume"].values

        # Clip extreme spikes relative to rolling median to avoid single-month blowups
        if len(df) >= 4:
            rolling_median = (
                df["volume"].rolling(window=6, min_periods=1).median().replace(0, np.nan)
            )
            # More permissive: 8x rolling median instead of 5x
            spike_cap = rolling_median * 8
            spike_mask = df["volume"] > spike_cap.fillna(df["volume"].max() * 2)
            if spike_mask.any():
                capped = int(spike_mask.sum())
                df.loc[spike_mask, "volume"] = spike_cap[spike_mask].fillna(
                    df["volume"].median()
                )
                logger.info(
                    f"  {site_id or 'Dataset'}: Capped {capped} spike(s) at 8x rolling median"
                )

        # Recompute recent volumes after spike capping so MAD uses cleaned data
        recent_data = df.tail(window_size)
        recent_volumes = recent_data["volume"].values

        # Use MAD method on RECENT data (more robust to business changes)
        median = np.median(recent_volumes)
        MAD = np.median(np.abs(recent_volumes - median))

        # Guard against degenerate series (MAD==0 or NaN)
        if not (MAD > 0 and np.isfinite(MAD)):
            # Degenerate series: use a wider band
            lower_bound = max(0.0, median * 0.5)
            upper_bound = median * 2.0
        else:
            # Define outlier boundaries using recent statistics
            # Lower: catch data errors (zeros, negative-like entries)
            # Upper: use MAD-based bound that adapts to recent variance
            lower_bound = max(0.0, median - 4.0 * MAD)
            upper_bound = median + 6.0 * MAD  # MAD-based, adapts to recent variance

        # Identify outliers
        volumes = df["volume"].values
        outlier_mask = (volumes < lower_bound) | (volumes > upper_bound)
        outlier_indices = df.index[outlier_mask]
        outlier_count = len(outlier_indices)

        if outlier_count > 0:
            # Clip outliers at boundaries (preserves time series structure)
            df["volume"] = df["volume"].clip(lower=lower_bound, upper=upper_bound)

            # Log details for audit trail
            outlier_dates = (
                df.loc[outlier_indices, "date"].dt.strftime("%Y-%m").tolist()
            )
            site_label = f"Site {site_id}" if site_id else "Dataset"
            logger.info(
                f"  {site_label}: Capped {outlier_count} outlier(s) at boundaries "
                f"[{lower_bound:.0f}, {upper_bound:.0f}] in months: {outlier_dates[:5]}"
                f"{'...' if len(outlier_dates) > 5 else ''}"
            )

        return df

    def _assess_data_quality(self, months_available: int) -> Tuple[str, Optional[str]]:
        """
        Categorize data sufficiency for transparency in outputs.
        Returns (label, note).
        """
        if months_available >= self.min_months_data:
            return "ok", None
        if months_available >= self.soft_min_months:
            return (
                "sparse",
                f"{months_available} months (<{self.min_months_data})",
            )
        return (
            "very_sparse",
            f"{months_available} months (<{self.min_months_data}); high uncertainty",
        )

    def _get_prior_year_actual(
        self,
        target_month: str,
        site_id: Optional[str] = None,
        grade: Optional[str] = None,
    ) -> Optional[float]:
        """
        Get the actual volume for the same month in the prior year.

        Args:
            target_month: Target month in 'YYYY-MM' format
            site_id: Optional site ID filter
            grade: Optional grade filter

        Returns:
            Actual volume for prior year same month, or None if not available
        """
        target_date = pd.to_datetime(target_month)
        prior_year_month = (target_date - pd.DateOffset(years=1)).strftime("%Y-%m")
        prior_year_start = f"{prior_year_month}-01"
        prior_year_end = (
            pd.to_datetime(prior_year_start) + pd.offsets.MonthEnd(0)
        ).strftime("%Y-%m-%d")

        site_ids = [site_id] if site_id else None
        grades = [grade] if grade else None

        try:
            df = self.db.get_sales_data(
                start_date=prior_year_start,
                end_date=prior_year_end,
                site_ids=site_ids,
                grades=grades,
                exclude_estimated=True,
            )

            if df.empty:
                return None

            return float(df["volume"].sum())
        except Exception as e:
            logger.debug(f"Could not fetch prior year actual: {e}")
            return None

    def _calculate_yoy_change(
        self, forecast_volume: float, prior_year_volume: Optional[float]
    ) -> Optional[float]:
        """
        Calculate year-over-year percentage change.

        Args:
            forecast_volume: Forecasted volume for target month
            prior_year_volume: Actual volume from same month prior year

        Returns:
            Percentage change (e.g., 5.2 for 5.2% increase), or None if prior year unavailable
        """
        # Handle None, NaN, and zero values to avoid division errors
        if prior_year_volume is None or pd.isna(prior_year_volume) or prior_year_volume == 0:
            return None
        return ((forecast_volume - prior_year_volume) / prior_year_volume) * 100

    def _get_prior_year_actual_by_grade(
        self, target_month: str, grade: str
    ) -> Optional[float]:
        """
        Get actual aggregate volume for a specific grade across all sites for prior year month.

        This queries the database directly for the true historical aggregate,
        ensuring accurate YoY comparisons in summary reports.

        Args:
            target_month: Target month in 'YYYY-MM' format
            grade: Fuel grade to filter by

        Returns:
            Actual total volume for the grade in prior year same month, or None if unavailable
        """
        # Handle "ALL" grade (used in site-level forecasts) by fetching total across all grades
        if grade == "ALL":
            return self._get_prior_year_actual_total(target_month)
        return self._get_prior_year_actual(target_month, site_id=None, grade=grade)

    def _get_prior_year_actual_total(self, target_month: str) -> Optional[float]:
        """
        Get actual total volume across all sites and grades for prior year month.

        This queries the database directly for the true historical total,
        ensuring accurate YoY comparisons in BU-level summary reports.

        Args:
            target_month: Target month in 'YYYY-MM' format

        Returns:
            Actual total volume for prior year same month, or None if unavailable
        """
        return self._get_prior_year_actual(target_month, site_id=None, grade=None)

    def _fallback_forecast_value(
        self, data: pd.DataFrame, months_ahead: int, note: Optional[str]
    ) -> float:
        """
        Generate a conservative fallback forecast using median and simple trend.
        """
        recent = data.tail(min(6, len(data)))
        median_recent = float(recent["volume"].median())

        trend = 0.0
        if len(recent) >= 3:
            x = np.arange(len(recent))
            try:
                trend = float(np.polyfit(x, recent["volume"].values, 1)[0])
            except Exception:
                trend = 0.0

        fallback = median_recent + trend * months_ahead
        fallback = max(0.0, fallback)

        logger.info(
            f"Using fallback forecast (median/trend). Months={len(data)}, Note={note or 'none'}"
        )
        return fallback

    def check_data_sufficiency(
        self, site_id: Optional[str] = None, grade: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if there's sufficient data for forecasting

        Returns:
            Dictionary with sufficiency info
        """
        try:
            monthly_data = self.prepare_monthly_data(
                site_id=site_id,
                grade=grade,
                handle_outliers=False,
                fill_gaps=False,
            )
            months_available = len(monthly_data)
            is_sufficient = months_available >= self.min_months_data

            return {
                "sufficient": is_sufficient,
                "months_available": months_available,
                "months_required": self.min_months_data,
                "date_range": f"{monthly_data['date'].min().strftime('%Y-%m')} to {monthly_data['date'].max().strftime('%Y-%m')}",
            }
        except Exception as e:
            return {
                "sufficient": False,
                "months_available": 0,
                "months_required": self.min_months_data,
                "error": str(e),
            }

    def train_models(
        self,
        data: pd.DataFrame,
        models_to_use: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """
        Train forecasting models

        Args:
            data: Prepared monthly time series data
            models_to_use: List of model names (default: all available)

        Returns:
            Dictionary of successfully trained models
        """
        available_models = get_available_models()

        if models_to_use is None:
            models_to_use = list(available_models.keys())

        trained_models = {}

        for model_name in models_to_use:
            if model_name not in available_models:
                logger.warning(f"Model {model_name} not available")
                continue

            try:
                model = available_models[model_name]()
                model.fit(data)
                trained_models[model_name] = model
                logger.debug(f"Trained model {model_name}")
            except Exception as e:
                logger.warning(f"✗ {model_name}: {e}")

        self.models = trained_models
        return trained_models

    def generate_forecast(
        self,
        target_month: str,
        site_id: Optional[str] = None,
        grade: Optional[str] = None,
        models_to_use: Optional[List[str]] = None,
        monthly_data: Optional[pd.DataFrame] = None,
        monthly_data_raw: Optional[pd.DataFrame] = None,
        show_yoy: bool = True,
    ) -> pd.DataFrame:
        """
        Generate forecast for a specific month

        Args:
            target_month: Target month in 'YYYY-MM' format
            site_id: Specific site (None for all)
            grade: Specific grade (None for all)
            models_to_use: Specific models to use
            monthly_data: Pre-computed monthly data WITH outlier handling (for ETS)
            monthly_data_raw: Pre-computed monthly data WITHOUT outlier handling (for snaive)
            show_yoy: Include year-over-year comparison columns (default: True)

        Returns:
            DataFrame with forecasts from all models
        """
        # Normalize target month to first day of month
        target_date = pd.to_datetime(target_month).to_period("M").to_timestamp()

        # Get prior year actual for YoY comparison (if enabled)
        prior_year_volume = None
        prior_year_month = None
        if show_yoy:
            prior_year_volume = self._get_prior_year_actual(
                target_month, site_id=site_id, grade=grade
            )
            prior_year_month = (target_date - pd.DateOffset(years=1)).strftime("%Y-%m")

        # Use cached data or prepare fresh
        if monthly_data is None:
            # Check data sufficiency (log warning if low, but don't block)
            check = self.check_data_sufficiency(site_id=site_id, grade=grade)
            if not check["sufficient"]:
                logger.warning(
                    f"Low data warning: {check['months_available']} months "
                    f"(recommended: {check['months_required']}). Forecasting anyway."
                )

            # Prepare data with outlier handling (for ETS)
            monthly_data = self.prepare_monthly_data(site_id=site_id, grade=grade, handle_outliers=True)

        # Prepare raw data for snaive (no outlier handling, no gap filling - use exact historical values)
        if monthly_data_raw is None:
            monthly_data_raw = self.prepare_monthly_data(site_id=site_id, grade=grade, handle_outliers=False, fill_gaps=False)

        last_date = monthly_data["date"].max()

        # Calculate periods ahead
        months_ahead = (target_date.year - last_date.year) * 12 + (
            target_date.month - last_date.month
        )

        if months_ahead <= 0:
            raise ValueError(f"Target month {target_month} is not in the future")

        months_available = len(monthly_data)
        data_quality, quality_note = self._assess_data_quality(months_available)

        # Get available models
        available_models = get_available_models()
        if models_to_use is None:
            models_to_use = list(available_models.keys())

        # Generate predictions - train each model with appropriate data
        results = []
        forecasts_for_ensemble = []
        note = quality_note
        fallback_value = None

        for model_name in models_to_use:
            if model_name not in available_models:
                logger.warning(f"Model {model_name} not available")
                continue

            try:
                model = available_models[model_name]()

                # Use raw data for snaive (preserves actual historical values)
                # Use outlier-handled data for ETS (benefits from cleaned data)
                if model_name == "snaive":
                    model.fit(monthly_data_raw)
                else:
                    model.fit(monthly_data)

                predictions = model.predict(periods=months_ahead)

                # Get forecast for target month
                target_pred = predictions[predictions["date"] == target_date]

                if target_pred.empty:
                    # Use last available prediction if exact date doesn't match
                    target_pred = predictions.iloc[-1:].copy()

                forecast_value = target_pred["forecast"].values[0]
                forecasts_for_ensemble.append(forecast_value)

                # Check if snaive used fallback (missing same-month data)
                model_note = note
                used_fallback = False
                if model_name == "snaive" and hasattr(model, "last_prediction_used_fallback"):
                    if model.last_prediction_used_fallback:
                        used_fallback = True
                        fallback_note = f"SNAIVE used fallback (no same-month data for: {', '.join(model.fallback_months)})"
                        model_note = f"{note}; {fallback_note}" if note else fallback_note
                        logger.debug(fallback_note)

                result_row = {
                    "model": model_name,
                    "target_month": target_month,
                    "forecast_volume": forecast_value,
                    "site_id": site_id or "ALL",
                    "grade": grade or "ALL",
                    "months_available": months_available,
                    "data_quality": data_quality,
                    "note": model_note,
                    "snaive_used_fallback": used_fallback if model_name == "snaive" else None,
                }
                # Add YoY columns if enabled
                if show_yoy:
                    result_row["prior_year_month"] = prior_year_month
                    result_row["prior_year_volume"] = prior_year_volume
                    result_row["yoy_change_pct"] = self._calculate_yoy_change(
                        forecast_value, prior_year_volume
                    )
                results.append(result_row)
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")

        if not forecasts_for_ensemble:
            fallback_value = self._fallback_forecast_value(
                monthly_data, months_ahead, quality_note
            )
            forecasts_for_ensemble.append(fallback_value)
            fallback_row = {
                "model": "FALLBACK",
                "target_month": target_month,
                "forecast_volume": fallback_value,
                "site_id": site_id or "ALL",
                "grade": grade or "ALL",
                "months_available": months_available,
                "data_quality": data_quality,
                "note": (note or "Using fallback due to model failure"),
                "snaive_used_fallback": None,
            }
            # Add YoY columns if enabled
            if show_yoy:
                fallback_row["prior_year_month"] = prior_year_month
                fallback_row["prior_year_volume"] = prior_year_volume
                fallback_row["yoy_change_pct"] = self._calculate_yoy_change(
                    fallback_value, prior_year_volume
                )
            results.append(fallback_row)

        results_df = pd.DataFrame(results)

        ensemble_note = note or ("Used fallback forecast" if fallback_value is not None else None)

        # Add ensemble (robust median instead of mean)
        if forecasts_for_ensemble:
            # Median is more robust to outlier models
            ensemble_forecast = float(np.median(forecasts_for_ensemble))

            # Check if any snaive result used fallback
            snaive_fallback_in_results = any(
                r.get("snaive_used_fallback") for r in results if r.get("model") == "snaive"
            )

            ensemble_row_data = {
                "model": "ENSEMBLE",
                "target_month": target_month,
                "forecast_volume": ensemble_forecast,
                "site_id": site_id or "ALL",
                "grade": grade or "ALL",
                "months_available": months_available,
                "data_quality": data_quality,
                "note": ensemble_note,
                "snaive_used_fallback": snaive_fallback_in_results if snaive_fallback_in_results else None,
            }
            # Add YoY columns if enabled
            if show_yoy:
                ensemble_row_data["prior_year_month"] = prior_year_month
                ensemble_row_data["prior_year_volume"] = prior_year_volume
                ensemble_row_data["yoy_change_pct"] = self._calculate_yoy_change(
                    ensemble_forecast, prior_year_volume
                )
            ensemble_row = pd.DataFrame([ensemble_row_data])
            results_df = pd.concat([results_df, ensemble_row], ignore_index=True)

        return results_df

    def generate_bulk_forecasts(
        self,
        target_month: str,
        by: str = "site",
        models_to_use: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        skip_insufficient: bool = True,
        show_yoy: bool = True,
    ) -> pd.DataFrame:
        """
        Generate forecasts for multiple configurations with progress tracking

        Args:
            target_month: Target month 'YYYY-MM'
            by: 'grade', 'site', or 'site_grade'
            models_to_use: Specific models to use
            output_path: Output Excel file path
            skip_insufficient: Skip items with insufficient data
            show_yoy: Include year-over-year comparison columns (default: True)

        Returns:
            DataFrame with all forecasts
        """
        all_forecasts = []
        skipped = []

        if by == "grade":
            stats = self.db.get_summary_stats()
            grades = stats["fuel_grades"]

            logger.info(f"Generating forecasts for {len(grades)} grades")

            for i, grade in enumerate(grades, 1):
                logger.info(f"  [{i}/{len(grades)}] Grade: {grade}")
                try:
                    forecast = self.generate_forecast(
                        target_month, grade=grade, models_to_use=models_to_use, show_yoy=show_yoy
                    )
                    all_forecasts.append(forecast)
                except Exception as e:
                    logger.warning(f"  ✗ Failed: {e}")
                    skipped.append({"grade": grade, "reason": str(e)})

        elif by == "site":
            sites_df = self.db.get_distinct_sites()

            total = len(sites_df)
            logger.info(f"Generating forecasts for {total} sites")

            for i, row in sites_df.iterrows():
                if (i + 1) % 20 == 0 or (i + 1) == total:
                    logger.info(f"  Progress: {i+1}/{total} sites")

                try:
                    # Cache monthly data - both processed (for ETS) and raw (for snaive)
                    site_monthly_data = self.prepare_monthly_data(
                        site_id=row["site_id"], handle_outliers=True
                    )
                    site_monthly_data_raw = self.prepare_monthly_data(
                        site_id=row["site_id"], handle_outliers=False, fill_gaps=False
                    )
                    months_available = len(site_monthly_data)

                    if months_available < self.min_months_data and skip_insufficient:
                        skipped.append(
                            {
                                "site_id": row["site_id"],
                                "site": row["site"],
                                "reason": f"Only {months_available} months",
                            }
                        )
                        continue

                    # Use cached data in bulk forecasts
                    forecast = self.generate_forecast(
                        target_month,
                        site_id=row["site_id"],
                        models_to_use=models_to_use,
                        monthly_data=site_monthly_data,
                        monthly_data_raw=site_monthly_data_raw,
                        show_yoy=show_yoy,
                    )
                    forecast["site_name"] = row["site"]
                    all_forecasts.append(forecast)

                except Exception as e:
                    logger.warning(f"  ✗ Site {row['site_id']}: {e}")
                    skipped.append(
                        {
                            "site_id": row["site_id"],
                            "site": row["site"],
                            "reason": str(e),
                        }
                    )

        elif by == "site_grade":
            combos = self.db.get_distinct_site_grades()

            total = len(combos)
            logger.info(
                f"Generating forecasts for {total} site-grade combinations"
            )

            for i, row in combos.iterrows():
                if (i + 1) % 50 == 0 or (i + 1) == total:
                    logger.info(f"  Progress: {i+1}/{total} combinations")

                try:
                    # Cache monthly data - both processed (for ETS) and raw (for snaive)
                    combo_monthly_data = self.prepare_monthly_data(
                        site_id=row["site_id"], grade=row["grade"], handle_outliers=True
                    )
                    combo_monthly_data_raw = self.prepare_monthly_data(
                        site_id=row["site_id"], grade=row["grade"], handle_outliers=False, fill_gaps=False
                    )
                    months_available = len(combo_monthly_data)

                    if months_available < self.soft_min_months and skip_insufficient:
                        skipped.append(
                            {
                                "site_id": row["site_id"],
                                "site": row["site"],
                                "grade": row["grade"],
                                "reason": f"Only {months_available} months",
                            }
                        )
                        continue

                    # Use cached data in bulk forecasts
                    forecast = self.generate_forecast(
                        target_month,
                        site_id=row["site_id"],
                        grade=row["grade"],
                        models_to_use=models_to_use,
                        monthly_data=combo_monthly_data,
                        monthly_data_raw=combo_monthly_data_raw,
                        show_yoy=show_yoy,
                    )
                    forecast["site_name"] = row["site"]
                    all_forecasts.append(forecast)

                except Exception as e:
                    logger.warning(
                        f"  ✗ Site {row['site_id']}, Grade {row['grade']}: {e}"
                    )
                    skipped.append(
                        {
                            "site_id": row["site_id"],
                            "site": row["site"],
                            "grade": row["grade"],
                            "reason": str(e),
                        }
                    )

        else:
            raise ValueError("by must be 'grade', 'site', or 'site_grade'")

        if not all_forecasts:
            raise ValueError("No forecasts were successfully generated")

        # Combine results
        combined = pd.concat(all_forecasts, ignore_index=True)

        # Log summary
        logger.info("\nForecast Summary:")
        logger.info(f"  ✓ Generated: {len(all_forecasts)} forecasts")
        if skipped:
            logger.info(f"  ⊘ Skipped: {len(skipped)} items")

        # Export to Excel
        if output_path:
            self._export_results(combined, skipped, output_path)

        return combined

    def _export_results(
        self, forecasts: pd.DataFrame, skipped: List[Dict], output_path: str
    ):
        """Export forecasts to Excel or CSV based on output extension"""
        suffix = Path(output_path).suffix.lower()
        if suffix == ".csv":
            self._export_to_csv(forecasts, skipped, output_path)
        else:
            self._export_to_excel(forecasts, skipped, output_path)
        logger.info(f"  → Saved to: {output_path}")

    def _create_site_summary(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        """
        Create site-level summary by summing grade-level forecasts.

        This ensures site totals are reconciled (exactly equal to sum of grades).
        Only applicable when forecasts contain grade-level detail.

        Args:
            forecasts: DataFrame with site_grade level forecasts

        Returns:
            DataFrame with site-level totals derived from summing grades
        """
        # Check if this is grade-level data (has non-ALL grade values)
        if "grade" not in forecasts.columns:
            return pd.DataFrame()

        has_grade_detail = (forecasts["grade"] != "ALL").any()
        if not has_grade_detail:
            return pd.DataFrame()

        # Filter to only rows with actual grade values (not ALL)
        grade_data = forecasts[forecasts["grade"] != "ALL"].copy()

        if grade_data.empty:
            return pd.DataFrame()

        # Columns to sum
        sum_cols = ["forecast_volume"]
        if "prior_year_volume" in grade_data.columns:
            sum_cols.append("prior_year_volume")

        # Group by site and model, sum the volumes
        group_cols = ["site_id", "target_month", "model"]
        if "site_name" in grade_data.columns:
            group_cols.insert(1, "site_name")

        # Build aggregation dict
        agg_dict = {col: "sum" for col in sum_cols if col in grade_data.columns}
        agg_dict["grade"] = "count"  # Count of grades included

        site_summary = grade_data.groupby(group_cols, as_index=False).agg(agg_dict)
        site_summary = site_summary.rename(columns={"grade": "grades_included"})

        # Recalculate YoY change based on summed values
        if "prior_year_volume" in site_summary.columns:
            site_summary["yoy_change_pct"] = site_summary.apply(
                lambda row: self._calculate_yoy_change(
                    row["forecast_volume"], row["prior_year_volume"]
                ),
                axis=1
            )

        # Reorder columns for clarity
        desired_order = [
            "site_id",
            "site_name",
            "target_month",
            "model",
            "forecast_volume",
            "prior_year_volume",
            "yoy_change_pct",
            "grades_included",
        ]
        ordered_cols = [c for c in desired_order if c in site_summary.columns]
        remaining_cols = [c for c in site_summary.columns if c not in ordered_cols]
        site_summary = site_summary[ordered_cols + remaining_cols]

        return site_summary

    def _create_product_summary(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        """
        Create product/grade-level summary with YoY % change.

        Aggregates forecasts by grade (fuel product type) across all sites,
        showing total volumes and YoY change at the product level.

        IMPORTANT: Prior year volumes are fetched directly from the database
        (not summed from individual forecasts) to ensure accurate YoY comparisons
        even when individual site forecasts used imputed/historical data.

        Args:
            forecasts: DataFrame with forecast data

        Returns:
            DataFrame with product-level totals and YoY %
        """
        if "grade" not in forecasts.columns:
            return pd.DataFrame()

        # Filter to ENSEMBLE model for clean aggregation (avoid double-counting)
        if "model" in forecasts.columns:
            ensemble_data = forecasts[forecasts["model"] == "ENSEMBLE"].copy()
            if ensemble_data.empty:
                # Fallback to all data if no ENSEMBLE
                ensemble_data = forecasts.copy()
        else:
            ensemble_data = forecasts.copy()

        if ensemble_data.empty:
            return pd.DataFrame()

        # Group by grade and target_month, sum forecast volumes
        group_cols = ["grade", "target_month"]
        agg_dict = {"forecast_volume": "sum"}
        if "site_id" in ensemble_data.columns:
            agg_dict["site_id"] = "nunique"  # Count of sites included

        product_summary = ensemble_data.groupby(group_cols, as_index=False).agg(agg_dict)
        if "site_id" in product_summary.columns:
            product_summary = product_summary.rename(columns={"site_id": "sites_included"})

        # Fetch actual prior year volumes directly from database for each grade
        # This ensures accurate YoY comparison regardless of imputation in individual forecasts
        prior_year_volumes = []
        prior_year_months = []
        for _, row in product_summary.iterrows():
            target_month = row["target_month"]
            grade = row["grade"]
            prior_year_vol = self._get_prior_year_actual_by_grade(target_month, grade)
            prior_year_volumes.append(prior_year_vol)
            # Calculate prior year month for display
            target_date = pd.to_datetime(target_month)
            prior_year_months.append(
                (target_date - pd.DateOffset(years=1)).strftime("%Y-%m")
            )

        product_summary["prior_year_month"] = prior_year_months
        product_summary["prior_year_volume"] = prior_year_volumes

        # Calculate YoY change based on actual prior year aggregates
        product_summary["yoy_change_pct"] = product_summary.apply(
            lambda row: self._calculate_yoy_change(
                row["forecast_volume"], row["prior_year_volume"]
            ),
            axis=1
        )

        # Reorder columns for clarity
        desired_order = [
            "grade",
            "target_month",
            "forecast_volume",
            "prior_year_month",
            "prior_year_volume",
            "yoy_change_pct",
            "sites_included",
        ]
        ordered_cols = [c for c in desired_order if c in product_summary.columns]
        remaining_cols = [c for c in product_summary.columns if c not in ordered_cols]
        product_summary = product_summary[ordered_cols + remaining_cols]

        return product_summary

    def _create_bu_summary(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        """
        Create BU (Business Unit) level summary with YoY % change.

        Aggregates all forecasts to show overall totals for the entire BU,
        including total volumes and YoY change.

        IMPORTANT: Prior year volumes are fetched directly from the database
        (not summed from individual forecasts) to ensure accurate YoY comparisons
        even when individual site forecasts used imputed/historical data.

        Args:
            forecasts: DataFrame with forecast data

        Returns:
            DataFrame with BU-level totals and YoY %
        """
        # Filter to ENSEMBLE model for clean aggregation (avoid double-counting)
        if "model" in forecasts.columns:
            ensemble_data = forecasts[forecasts["model"] == "ENSEMBLE"].copy()
            if ensemble_data.empty:
                # Fallback to all data if no ENSEMBLE
                ensemble_data = forecasts.copy()
        else:
            ensemble_data = forecasts.copy()

        if ensemble_data.empty:
            return pd.DataFrame()

        # Group by target_month only (BU-level = all sites, all grades)
        group_cols = ["target_month"]
        agg_dict = {"forecast_volume": "sum"}

        # Count unique sites and grades
        if "site_id" in ensemble_data.columns:
            agg_dict["site_id"] = "nunique"
        if "grade" in ensemble_data.columns:
            agg_dict["grade"] = "nunique"

        bu_summary = ensemble_data.groupby(group_cols, as_index=False).agg(agg_dict)

        # Rename count columns
        if "site_id" in bu_summary.columns:
            bu_summary = bu_summary.rename(columns={"site_id": "sites_included"})
        if "grade" in bu_summary.columns:
            bu_summary = bu_summary.rename(columns={"grade": "grades_included"})

        # Fetch actual prior year total volumes directly from database
        # This ensures accurate YoY comparison regardless of imputation in individual forecasts
        prior_year_volumes = []
        prior_year_months = []
        for _, row in bu_summary.iterrows():
            target_month = row["target_month"]
            prior_year_vol = self._get_prior_year_actual_total(target_month)
            prior_year_volumes.append(prior_year_vol)
            # Calculate prior year month for display
            target_date = pd.to_datetime(target_month)
            prior_year_months.append(
                (target_date - pd.DateOffset(years=1)).strftime("%Y-%m")
            )

        bu_summary["prior_year_month"] = prior_year_months
        bu_summary["prior_year_volume"] = prior_year_volumes

        # Calculate YoY change based on actual prior year total
        bu_summary["yoy_change_pct"] = bu_summary.apply(
            lambda row: self._calculate_yoy_change(
                row["forecast_volume"], row["prior_year_volume"]
            ),
            axis=1
        )

        # Reorder columns for clarity
        desired_order = [
            "target_month",
            "forecast_volume",
            "prior_year_month",
            "prior_year_volume",
            "yoy_change_pct",
            "sites_included",
            "grades_included",
        ]
        ordered_cols = [c for c in desired_order if c in bu_summary.columns]
        remaining_cols = [c for c in bu_summary.columns if c not in ordered_cols]
        bu_summary = bu_summary[ordered_cols + remaining_cols]

        return bu_summary

    def _export_to_excel(
        self, forecasts: pd.DataFrame, skipped: List[Dict], output_path: str
    ):
        """Export forecasts to Excel with multiple sheets"""
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Main forecasts
            desired_order = [
                "site_id",
                "grade",
                "target_month",
                "model",
                "forecast_volume",
                "prior_year_month",
                "prior_year_volume",
                "yoy_change_pct",
            ]
            ordered_cols = [c for c in desired_order if c in forecasts.columns]

            # Keep any extra columns (e.g., site_name) after the requested order
            remaining_cols = [
                c for c in forecasts.columns if c not in ordered_cols
            ]
            export_cols = ordered_cols + remaining_cols

            forecasts[export_cols].to_excel(
                writer, sheet_name="Forecasts", index=False
            )

            # Site Summary (reconciled site totals from summing grades)
            site_summary = self._create_site_summary(forecasts)
            if not site_summary.empty:
                site_summary.to_excel(writer, sheet_name="Site Summary", index=False)
                logger.info(
                    f"  → Site Summary: {len(site_summary)} reconciled site-level forecasts"
                )

            # Product Summary (grade-level aggregation with YoY %)
            product_summary = self._create_product_summary(forecasts)
            if not product_summary.empty:
                product_summary.to_excel(writer, sheet_name="Product Summary", index=False)
                logger.info(
                    f"  → Product Summary: {len(product_summary)} product-level forecasts with YoY %"
                )

            # BU Summary (overall business unit totals with YoY %)
            bu_summary = self._create_bu_summary(forecasts)
            if not bu_summary.empty:
                bu_summary.to_excel(writer, sheet_name="BU Summary", index=False)
                logger.info(
                    f"  → BU Summary: Overall business unit forecast with YoY %"
                )

            # Skipped items (if any)
            if skipped:
                skipped_df = pd.DataFrame(skipped)
                skipped_df.to_excel(writer, sheet_name="Skipped", index=False)

            # Summary by model
            summary = (
                forecasts.groupby("model")["forecast_volume"]
                .agg(["count", "sum", "mean", "min", "max"])
                .reset_index()
            )
            summary.columns = ["Model", "Count", "Total", "Average", "Min", "Max"]
            summary.to_excel(writer, sheet_name="Summary", index=False)

    def _export_to_csv(
        self, forecasts: pd.DataFrame, skipped: List[Dict], output_path: str
    ):
        """Export forecasts to CSV; skipped/summary/site_summary go to sibling files"""
        desired_order = [
            "site_id",
            "grade",
            "target_month",
            "model",
            "forecast_volume",
            "prior_year_month",
            "prior_year_volume",
            "yoy_change_pct",
        ]
        ordered_cols = [c for c in desired_order if c in forecasts.columns]
        remaining_cols = [c for c in forecasts.columns if c not in ordered_cols]
        export_cols = ordered_cols + remaining_cols

        forecasts[export_cols].to_csv(output_path, index=False)

        base = Path(output_path)

        # Site Summary (reconciled site totals from summing grades)
        site_summary = self._create_site_summary(forecasts)
        if not site_summary.empty:
            site_summary_path = base.with_name(f"{base.stem}_site_summary.csv")
            site_summary.to_csv(site_summary_path, index=False)
            logger.info(
                f"  → Site Summary ({len(site_summary)} reconciled forecasts) saved to: {site_summary_path}"
            )

        # Product Summary (grade-level aggregation with YoY %)
        product_summary = self._create_product_summary(forecasts)
        if not product_summary.empty:
            product_summary_path = base.with_name(f"{base.stem}_product_summary.csv")
            product_summary.to_csv(product_summary_path, index=False)
            logger.info(
                f"  → Product Summary ({len(product_summary)} product-level forecasts with YoY %) saved to: {product_summary_path}"
            )

        # BU Summary (overall business unit totals with YoY %)
        bu_summary = self._create_bu_summary(forecasts)
        if not bu_summary.empty:
            bu_summary_path = base.with_name(f"{base.stem}_bu_summary.csv")
            bu_summary.to_csv(bu_summary_path, index=False)
            logger.info(
                f"  → BU Summary (overall business unit forecast with YoY %) saved to: {bu_summary_path}"
            )

        if skipped:
            skipped_df = pd.DataFrame(skipped)
            skipped_path = base.with_name(f"{base.stem}_skipped.csv")
            skipped_df.to_csv(skipped_path, index=False)
            logger.info(f"  → Skipped items saved to: {skipped_path}")

        summary = (
            forecasts.groupby("model")["forecast_volume"]
            .agg(["count", "sum", "mean", "min", "max"])
            .reset_index()
        )
        summary.columns = ["Model", "Count", "Total", "Average", "Min", "Max"]
        summary_path = base.with_name(f"{base.stem}_summary.csv")
        summary.to_csv(summary_path, index=False)
        logger.info(f"  → Summary saved to: {summary_path}")
