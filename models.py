"""
Forecasting Models Module - ETS (Holt-Winters) and Seasonal Naive
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from abc import ABC, abstractmethod
from dateutil.relativedelta import relativedelta

# Core models
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# No optional models needed - using only ETS and Seasonal Naive


logger = logging.getLogger(__name__)


class ForecastModel(ABC):
    """Base class for forecast models"""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False

    def _validate_input(self, data: pd.DataFrame):
        """Validate input data structure and contents"""
        if data is None or not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")

        if data.empty:
            raise ValueError("Input data is empty - need at least 1 month of data")

        # Check required columns exist
        required_cols = ["volume", "date"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Data must have 'date' and 'volume' columns."
            )

        # Validate data contains valid values
        if data["volume"].isna().all():
            raise ValueError("All volume values are NaN - cannot train model")

        if data["date"].isna().all():
            raise ValueError("All date values are NaN - cannot train model")

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        """Fit model to training data"""
        pass

    @abstractmethod
    def predict(self, periods: int) -> pd.DataFrame:
        """Generate predictions"""
        pass

    def get_name(self) -> str:
        return self.name


class SeasonalNaiveModel(ForecastModel):
    """Seasonal Naive - Fast baseline using same-month-last-year logic"""

    def __init__(self):
        super().__init__("SeasonalNaive")
        self.period = 12

    def fit(self, data: pd.DataFrame):
        """Fit Seasonal Naive model (stores training data with dates for month lookup)"""
        try:
            self._validate_input(data)

            # Ensure date column is datetime
            data = data.copy()
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values('date').reset_index(drop=True)

            # Store both volumes and dates for same-month lookup
            self.train_volumes = data["volume"].values
            self.train_dates = data["date"].values
            self.last_date = pd.Timestamp(data["date"].max())
            self.is_fitted = True
            logger.debug(f"SeasonalNaive: Trained on {len(self.train_volumes)} months")
        except Exception as e:
            logger.error(f"SeasonalNaive fit failed: {e}")
            raise

    def predict(self, periods: int) -> pd.DataFrame:
        """Generate Seasonal Naive predictions using same-month-last-year values"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        # Safety check: ensure we have training data
        if len(self.train_volumes) == 0:
            raise ValueError(
                "No training data available - cannot generate predictions. "
                "This should not happen if model was properly fitted."
            )

        # Generate future dates first
        future_dates = [
            self.last_date + relativedelta(months=i + 1) for i in range(periods)
        ]

        # Build forecast by looking up same-month values from training data
        forecast_values = []
        for future_date in future_dates:
            target_month = future_date.month

            # Find the most recent occurrence of this month in training data
            # Search backwards through training data for same month
            found_value = None
            for i in range(len(self.train_dates) - 1, -1, -1):
                train_date = pd.Timestamp(self.train_dates[i])
                if train_date.month == target_month:
                    found_value = self.train_volumes[i]
                    break

            if found_value is not None:
                forecast_values.append(found_value)
            else:
                # No same-month data available - fall back to last known value
                forecast_values.append(self.train_volumes[-1])

        # Clamp to non-negative
        forecast_values = np.maximum(0.0, forecast_values)

        return pd.DataFrame(
            {"date": future_dates, "forecast": forecast_values, "model": self.name}
        )


class ETSModel(ForecastModel):
    """Exponential Smoothing (Holt-Winters) with adaptive seasonality and damped trend"""

    def __init__(self):
        super().__init__("ETS")

    def _pick_seasonal_mode(self, y: np.ndarray) -> str:
        """Pick multiplicative vs additive seasonality using CV heuristic"""
        mean = float(np.mean(y))
        std = float(np.std(y))
        cv = std / mean if mean > 1e-9 else 0.0
        return "mul" if cv > 0.3 else "add"

    def fit(self, data: pd.DataFrame):
        """Fit ETS model with adaptive parameters"""
        try:
            self._validate_input(data)

            y = data["volume"].astype(float).to_numpy()
            data_length = len(y)

            # Determine if we have enough data for seasonality
            self.has_seasonality = data_length >= 24

            seasonal = None
            seasonal_periods = None
            if self.has_seasonality:
                seasonal = self._pick_seasonal_mode(y)
                seasonal_periods = 12
                logger.debug(f"ETS: Using {seasonal} seasonality")
            else:
                logger.debug(
                    f"ETS: Disabling seasonality due to limited data ({data_length} months)"
                )

            # Store training statistics for validation bounds
            self.training_mean = float(np.mean(y))
            self.training_std = float(np.std(y))

            # Guard against flat/degenerate series
            if not (self.training_std > 0 and np.isfinite(self.training_std)):
                self.training_std = max(1e-6, self.training_mean * 0.05)

            # Fit ETS with damped trend
            self.model = ExponentialSmoothing(
                y,
                trend="add",
                damped_trend=True,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                initialization_method="estimated",
            ).fit(optimized=True, use_brute=False)

            self.is_fitted = True
            self.last_date = data["date"].max()

            logger.debug(
                f"ETS fitted: trend=add (damped), seasonal={seasonal}, "
                f"mean={self.training_mean:.0f}, std={self.training_std:.0f}"
            )

        except Exception as e:
            logger.error(f"ETS fit failed: {e}")
            raise

    def predict(self, periods: int) -> pd.DataFrame:
        """Generate ETS predictions"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        forecast = self.model.forecast(steps=periods)

        # Generate future dates
        future_dates = [
            self.last_date + relativedelta(months=i + 1) for i in range(periods)
        ]

        # Ensure non-negative (volumes cannot be negative)
        forecast_values = np.maximum(0.0, forecast)

        return pd.DataFrame(
            {"date": future_dates, "forecast": forecast_values, "model": self.name}
        )


def get_available_models() -> Dict[str, type]:
    """Get dictionary of available models"""
    models = {
        "ets": ETSModel,
        "snaive": SeasonalNaiveModel,
    }

    return models


# Log available models on import
logger.info("Models available: ETS, SeasonalNaive")
