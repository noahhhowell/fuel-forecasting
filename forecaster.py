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

    def prepare_monthly_data(
        self,
        site_id: Optional[str] = None,
        grade: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        handle_outliers: bool = True,
    ) -> pd.DataFrame:
        """
        Prepare monthly aggregated data with outlier detection

        Args:
            site_id: Optional site ID filter
            grade: Optional grade filter
            start_date: Optional start date
            end_date: Optional end date
            handle_outliers: Apply outlier detection/handling (default: True)

        Returns:
            DataFrame with monthly volume data
        """
        site_ids = [site_id] if site_id else None
        grades = [grade] if grade else None
        start_date_str = (
            pd.to_datetime(start_date).strftime("%Y-%m-%d")
            if start_date is not None
            else None
        )
        end_date_str = (
            pd.to_datetime(end_date).strftime("%Y-%m-%d")
            if end_date is not None
            else None
        )

        df = self.db.get_sales_data(
            start_date=start_date_str,
            end_date=end_date_str,
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

        # Outlier detection and handling (critical for individual site accuracy)
        if handle_outliers and len(result) >= 12:
            result = self._handle_outliers(result, site_id)

        return result

    def _handle_outliers(
        self, data: pd.DataFrame, site_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Detect and handle outliers using MAD method with site-specific logging

        MAD (Median Absolute Deviation) is more robust than IQR for asymmetric distributions
        Outliers are capped at boundaries rather than removed to preserve time series continuity
        """
        df = data.copy()
        volumes = df["volume"].values

        # Use MAD method (more robust than IQR, especially for asymmetric distributions)
        median = np.median(volumes)
        MAD = np.median(np.abs(volumes - median))

        # Guard against degenerate series (MAD==0 or NaN)
        if not (MAD > 0 and np.isfinite(MAD)):
            # Degenerate series: use a simple ±30% band around median
            lower_bound = max(0.0, median * 0.7)
            upper_bound = median * 1.3
        else:
            # Define outlier boundaries with zero floor (volume cannot be negative)
            # Using 3.0 * MAD for lower bound to catch data errors
            # Using 5.0 * median for upper bound to allow valid high demand (holidays, etc.)
            lower_bound = max(0.0, median - 3.0 * MAD)
            upper_bound = median * 5.0  # Allow valid high demand, only clip data errors

        # Identify outliers
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

    def check_data_sufficiency(
        self,
        site_id: Optional[str] = None,
        grade: Optional[str] = None,
        history_end_month: Optional[str] = None,
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
                end_date=self._get_history_end_date(history_end_month),
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
                logger.debug(f"Trained model: {model_name}")
            except Exception as e:
                logger.warning(f"Training failed for {model_name}: {e}")

        self.models = trained_models
        return trained_models

    def generate_forecast(
        self,
        target_month: str,
        site_id: Optional[str] = None,
        grade: Optional[str] = None,
        models_to_use: Optional[List[str]] = None,
        monthly_data: Optional[pd.DataFrame] = None,
        history_end_month: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate forecast for a specific month

        Args:
            target_month: Target month in 'YYYY-MM' format
            site_id: Specific site (None for all)
            grade: Specific grade (None for all)
            models_to_use: Specific models to use
            monthly_data: Pre-computed monthly data (for caching in bulk forecasts)
            history_end_month: Only use history up to and including this month (YYYY-MM)

        Returns:
            DataFrame with forecasts from all models
        """
        # Normalize target month to first day of month
        target_date = pd.to_datetime(target_month).to_period("M").to_timestamp()
        history_end_date = self._get_history_end_date(history_end_month)

        # Use cached data or prepare fresh
        if monthly_data is None:
            # Check data sufficiency (log warning if low, but don't block)
            check = self.check_data_sufficiency(
                site_id=site_id, grade=grade, history_end_month=history_end_month
            )
            if not check["sufficient"]:
                logger.warning(
                    f"Low data warning: {check['months_available']} months "
                    f"(recommended: {check['months_required']}). Forecasting anyway."
                )

            # Prepare data with outlier handling
            monthly_data = self.prepare_monthly_data(
                site_id=site_id, grade=grade, end_date=history_end_date
            )

        last_date = monthly_data["date"].max()

        # Calculate periods ahead
        months_ahead = (target_date.year - last_date.year) * 12 + (
            target_date.month - last_date.month
        )

        if months_ahead <= 0:
            raise ValueError(f"Target month {target_month} is not in the future")

        # Train models
        trained_models = self.train_models(
            monthly_data, models_to_use=models_to_use
        )

        if not trained_models:
            raise ValueError("No models were successfully trained")

        # Generate predictions
        results = []
        forecasts_for_ensemble = []

        for model_name, model in trained_models.items():
            try:
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
                    }
                )
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")

        results_df = pd.DataFrame(results)

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
        history_end_month: Optional[str] = None,
        include_actuals: bool = False,
    ) -> pd.DataFrame:
        """
        Generate forecasts for multiple configurations with progress tracking

        Args:
            target_month: Target month 'YYYY-MM'
            by: 'total', 'grade', 'site', or 'site_grade'
            models_to_use: Specific models to use
            output_path: Output Excel file path
            skip_insufficient: Skip items with insufficient data
            history_end_month: Only use history up to and including this month (YYYY-MM)
            include_actuals: Attach actuals/error columns for target month if available

        Returns:
            DataFrame with all forecasts
        """
        all_forecasts = []
        skipped = []
        history_end_date = self._get_history_end_date(history_end_month)

        if by == "total":
            logger.info(f"Generating total forecast for {target_month}")
            forecast = self.generate_forecast(
                target_month,
                models_to_use=models_to_use,
                history_end_month=history_end_month,
            )
            all_forecasts.append(forecast)

        elif by == "grade":
            stats = self.db.get_summary_stats()
            grades = stats["fuel_grades"]

            logger.info(f"Generating forecasts for {len(grades)} grades")

            for i, grade in enumerate(grades, 1):
                logger.info(f"  [{i}/{len(grades)}] Grade: {grade}")
                try:
                    forecast = self.generate_forecast(
                        target_month,
                        grade=grade,
                        models_to_use=models_to_use,
                        history_end_month=history_end_month,
                    )
                    all_forecasts.append(forecast)
                except Exception as e:
                    logger.warning(f"  - Failed: {e}")
                    skipped.append({"grade": grade, "reason": str(e)})

        elif by == "site":
            sites_df = self.db.get_distinct_sites()

            total = len(sites_df)
            logger.info(f"Generating forecasts for {total} sites")

            for i, row in sites_df.iterrows():
                if (i + 1) % 20 == 0 or (i + 1) == total:
                    logger.info(f"  Progress: {i+1}/{total} sites")

                try:
                    # Cache monthly data (avoid recomputing in check + forecast)
                    site_monthly_data = self.prepare_monthly_data(
                        site_id=row["site_id"], end_date=history_end_date
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
                        history_end_month=history_end_month,
                    )
                    forecast["site_name"] = row["site"]
                    all_forecasts.append(forecast)

                except Exception as e:
                    logger.warning(f"  - Site {row['site_id']}: {e}")
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
                    # Cache monthly data (avoid recomputing in check + forecast)
                    combo_monthly_data = self.prepare_monthly_data(
                        site_id=row["site_id"],
                        grade=row["grade"],
                        end_date=history_end_date,
                    )
                    months_available = len(combo_monthly_data)

                    if months_available < self.min_months_data and skip_insufficient:
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
                        history_end_month=history_end_month,
                    )
                    forecast["site_name"] = row["site"]
                    all_forecasts.append(forecast)

                except Exception as e:
                    logger.warning(
                        f"  - Site {row['site_id']}, Grade {row['grade']}: {e}"
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
        if include_actuals:
            combined, summary = self._attach_actuals(
                combined, by=by, target_month=target_month
            )
            combined.attrs["actuals_summary"] = summary

        # Log summary
        logger.info("\nForecast Summary:")
        logger.info(f"  Generated: {len(all_forecasts)} forecasts")
        if skipped:
            logger.info(f"  Skipped: {len(skipped)} items")

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
        logger.info(f"  Saved to: {output_path}")

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
            logger.info(f"  Skipped items saved to: {skipped_path}")

        summary = (
            forecasts.groupby("model")["forecast_volume"]
            .agg(["count", "sum", "mean", "min", "max"])
            .reset_index()
        )
        summary.columns = ["Model", "Count", "Total", "Average", "Min", "Max"]
        summary_path = base.with_name(f"{base.stem}_summary.csv")
        summary.to_csv(summary_path, index=False)
        logger.info(f"  Summary saved to: {summary_path}")

    def _get_history_end_date(
        self, history_end_month: Optional[str]
    ) -> Optional[pd.Timestamp]:
        """Convert YYYY-MM to the last day of that month"""
        if not history_end_month:
            return None
        period = pd.to_datetime(history_end_month).to_period("M")
        return period.to_timestamp(how="end")

    def _get_actuals_for_month(self, target_month: str, by: str) -> pd.DataFrame:
        """Aggregate actual volumes for the target month at the requested level"""
        period = pd.to_datetime(target_month).to_period("M")
        start = period.to_timestamp()
        end = period.to_timestamp(how="end")

        data = self.db.get_sales_data(
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            exclude_estimated=True,
        )

        if data.empty:
            return pd.DataFrame()

        if by == "total":
            actuals = pd.DataFrame(
                [
                    {
                        "site_id": "ALL",
                        "grade": "ALL",
                        "target_month": target_month,
                        "actual_volume": data["volume"].sum(),
                    }
                ]
            )
        elif by == "grade":
            actuals = (
                data.groupby("grade")["volume"]
                .sum()
                .reset_index()
                .rename(columns={"volume": "actual_volume"})
            )
            actuals["target_month"] = target_month
            actuals["site_id"] = "ALL"
        elif by == "site":
            actuals = (
                data.groupby("site_id")["volume"]
                .sum()
                .reset_index()
                .rename(columns={"volume": "actual_volume"})
            )
            actuals["target_month"] = target_month
            actuals["grade"] = "ALL"
        elif by == "site_grade":
            actuals = (
                data.groupby(["site_id", "grade"])["volume"]
                .sum()
                .reset_index()
                .rename(columns={"volume": "actual_volume"})
            )
            actuals["target_month"] = target_month
        else:
            raise ValueError("Invalid 'by' value for actuals aggregation")

        return actuals[["site_id", "grade", "target_month", "actual_volume"]]

    def _attach_actuals(
        self, forecasts: pd.DataFrame, by: str, target_month: str
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """Merge actuals and compute simple error metrics"""
        actuals = self._get_actuals_for_month(target_month, by)

        if actuals.empty:
            logger.info("  No actuals available for target month; skipping comparison")
            return forecasts, {}

        merged = forecasts.merge(
            actuals,
            on=["site_id", "grade", "target_month"],
            how="left",
        )

        if "actual_volume" not in merged.columns:
            return merged, {}

        merged["error"] = merged["forecast_volume"] - merged["actual_volume"]
        merged["abs_error"] = merged["error"].abs()
        merged["ape"] = merged["abs_error"] / merged["actual_volume"].replace(
            0, np.nan
        )

        metrics: Dict[str, Dict[str, float]] = {}
        valid_mask = merged["actual_volume"].notna()

        for model, group in merged[valid_mask].groupby("model"):
            ape_series = group["ape"].dropna()
            metrics[model] = {
                "count": int(len(group)),
                "mae": float(group["abs_error"].mean()) if not group.empty else None,
                "mape": float(ape_series.mean()) if not ape_series.empty else None,
                "median_ape": float(ape_series.median()) if not ape_series.empty else None,
            }

        return merged, metrics
