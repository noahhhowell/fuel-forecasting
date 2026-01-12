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
            monthly_data = self.prepare_monthly_data(site_id=site_id, grade=grade)
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

        Returns:
            DataFrame with forecasts from all models
        """
        # Normalize target month to first day of month
        target_date = pd.to_datetime(target_month).to_period("M").to_timestamp()

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

                results.append(
                    {
                        "model": model_name,
                        "target_month": target_month,
                        "forecast_volume": forecast_value,
                        "site_id": site_id or "ALL",
                        "grade": grade or "ALL",
                        "months_available": months_available,
                        "data_quality": data_quality,
                        "note": note,
                    }
                )
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")

        if not forecasts_for_ensemble:
            fallback_value = self._fallback_forecast_value(
                monthly_data, months_ahead, quality_note
            )
            forecasts_for_ensemble.append(fallback_value)
            results.append(
                {
                    "model": "FALLBACK",
                    "target_month": target_month,
                    "forecast_volume": fallback_value,
                    "site_id": site_id or "ALL",
                    "grade": grade or "ALL",
                    "months_available": months_available,
                    "data_quality": data_quality,
                    "note": (note or "Using fallback due to model failure"),
                }
            )

        results_df = pd.DataFrame(results)

        ensemble_note = note or ("Used fallback forecast" if fallback_value is not None else None)

        # Add ensemble (robust median instead of mean)
        if forecasts_for_ensemble:
            # Median is more robust to outlier models
            ensemble_forecast = float(np.median(forecasts_for_ensemble))
            ensemble_row = pd.DataFrame(
                [
                    {
                        "model": "ENSEMBLE",
                        "target_month": target_month,
                        "forecast_volume": ensemble_forecast,
                        "site_id": site_id or "ALL",
                        "grade": grade or "ALL",
                        "months_available": months_available,
                        "data_quality": data_quality,
                        "note": ensemble_note,
                    }
                ]
            )
            results_df = pd.concat([results_df, ensemble_row], ignore_index=True)

        return results_df

    def generate_bulk_forecasts(
        self,
        target_month: str,
        by: str = "site",
        models_to_use: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        skip_insufficient: bool = True,
    ) -> pd.DataFrame:
        """
        Generate forecasts for multiple configurations with progress tracking

        Args:
            target_month: Target month 'YYYY-MM'
            by: 'total', 'grade', 'site', or 'site_grade'
            models_to_use: Specific models to use
            output_path: Output Excel file path
            skip_insufficient: Skip items with insufficient data

        Returns:
            DataFrame with all forecasts
        """
        all_forecasts = []
        skipped = []

        if by == "total":
            logger.info(f"Generating total forecast for {target_month}")
            forecast = self.generate_forecast(target_month, models_to_use=models_to_use)
            all_forecasts.append(forecast)

        elif by == "grade":
            stats = self.db.get_summary_stats()
            grades = stats["fuel_grades"]

            logger.info(f"Generating forecasts for {len(grades)} grades")

            for i, grade in enumerate(grades, 1):
                logger.info(f"  [{i}/{len(grades)}] Grade: {grade}")
                try:
                    forecast = self.generate_forecast(
                        target_month, grade=grade, models_to_use=models_to_use
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
            raise ValueError("by must be 'total', 'grade', 'site', or 'site_grade'")

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
        """Export forecasts to CSV; skipped/summary go to sibling files"""
        desired_order = [
            "site_id",
            "grade",
            "target_month",
            "model",
            "forecast_volume",
        ]
        ordered_cols = [c for c in desired_order if c in forecasts.columns]
        remaining_cols = [c for c in forecasts.columns if c not in ordered_cols]
        export_cols = ordered_cols + remaining_cols

        forecasts[export_cols].to_csv(output_path, index=False)

        base = Path(output_path)

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
