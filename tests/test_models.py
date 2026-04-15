"""
Tests for models.py — ETS and Seasonal Naive forecast models.

These tests verify that models:
  - Accept valid data and reject bad data
  - Produce forecasts in the right shape
  - Give sensible values (non-negative, in the right ballpark)
"""

import numpy as np
import pandas as pd
import pytest

from models import ETSModel, SeasonalNaiveModel, get_available_models


def _make_monthly_series(months=36, base=1000.0, seasonal_amp=0.15, seed=42):
    """
    Build a synthetic monthly time series with trend + seasonality.

    This is the shape that our models expect: a DataFrame with 'date'
    and 'volume' columns, one row per month.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=months, freq="MS")
    volumes = []
    for i, d in enumerate(dates):
        seasonal = 1.0 + seasonal_amp * np.sin(2 * np.pi * (d.month - 1) / 12)
        trend = 1.0 + 0.002 * i  # slight upward trend
        noise = rng.normal(0, 0.03)
        volumes.append(base * seasonal * trend + noise * base)

    return pd.DataFrame({"date": dates, "volume": np.maximum(0.1, volumes)})


# ---------------------------------------------------------------------------
# Seasonal Naive
# ---------------------------------------------------------------------------

class TestSeasonalNaive:

    def test_predict_uses_same_month_last_year(self):
        """The core snaive idea: forecast = same month from the prior year."""
        data = _make_monthly_series(months=24)
        model = SeasonalNaiveModel()
        model.fit(data)
        preds = model.predict(periods=1)

        # The first prediction should equal the volume from 12 months back
        # (same month last year)
        target_month = preds["date"].iloc[0].month
        same_month_rows = data[data["date"].dt.month == target_month]
        expected = same_month_rows["volume"].iloc[-1]

        assert preds["forecast"].iloc[0] == pytest.approx(expected)

    def test_predict_returns_correct_shape(self):
        """Predicting N periods should give N rows."""
        data = _make_monthly_series(months=24)
        model = SeasonalNaiveModel()
        model.fit(data)
        preds = model.predict(periods=6)

        assert len(preds) == 6
        assert "date" in preds.columns
        assert "forecast" in preds.columns

    def test_predict_non_negative(self):
        """Forecasts should never be negative."""
        data = _make_monthly_series(months=24)
        model = SeasonalNaiveModel()
        model.fit(data)
        preds = model.predict(periods=12)

        assert (preds["forecast"] >= 0).all()

    def test_fallback_when_month_missing(self):
        """If a target month has no history, snaive uses fallback."""
        # Create data with only 6 months (Jan-Jun) — no Jul-Dec history
        dates = pd.date_range("2023-01-01", periods=6, freq="MS")
        data = pd.DataFrame({
            "date": dates,
            "volume": [100, 200, 300, 400, 500, 600],
        })
        model = SeasonalNaiveModel()
        model.fit(data)

        # Predict into Jul (month 7) — no history for month 7
        preds = model.predict(periods=1)
        assert model.last_prediction_used_fallback is True
        assert len(model.fallback_months) == 1

    def test_rejects_empty_data(self):
        """Model should raise on empty input."""
        model = SeasonalNaiveModel()
        with pytest.raises(ValueError):
            model.fit(pd.DataFrame({"date": [], "volume": []}))

    def test_rejects_missing_columns(self):
        """Model should raise if required columns are missing."""
        model = SeasonalNaiveModel()
        with pytest.raises(ValueError, match="Missing required columns"):
            model.fit(pd.DataFrame({"timestamp": [1], "value": [100]}))


# ---------------------------------------------------------------------------
# ETS (Holt-Winters)
# ---------------------------------------------------------------------------

class TestETS:

    def test_fit_and_predict_basic(self):
        """ETS should fit and produce reasonable forecasts."""
        data = _make_monthly_series(months=36)
        model = ETSModel()
        model.fit(data)
        preds = model.predict(periods=3)

        assert len(preds) == 3
        # Forecasts should be in the same order of magnitude as training data
        train_mean = data["volume"].mean()
        for val in preds["forecast"]:
            assert val > train_mean * 0.3, "Forecast way too low"
            assert val < train_mean * 3.0, "Forecast way too high"

    def test_predict_non_negative(self):
        """ETS forecasts should never be negative."""
        data = _make_monthly_series(months=36)
        model = ETSModel()
        model.fit(data)
        preds = model.predict(periods=12)

        assert (preds["forecast"] >= 0).all()

    def test_additive_seasonality_when_zeros_present(self):
        """If data contains zeros, ETS should pick additive (not crash)."""
        data = _make_monthly_series(months=36)
        # Inject a zero
        data.loc[5, "volume"] = 0.0

        model = ETSModel()
        # This should NOT raise — the fix forces additive seasonality
        model.fit(data)
        preds = model.predict(periods=1)
        assert len(preds) == 1

    def test_short_series_disables_seasonality(self):
        """With <24 months, ETS should disable seasonality gracefully."""
        data = _make_monthly_series(months=18)
        model = ETSModel()
        model.fit(data)
        assert model.has_seasonality is False

        preds = model.predict(periods=3)
        assert len(preds) == 3

    def test_dates_are_sequential_months(self):
        """Predicted dates should be consecutive months after training data."""
        data = _make_monthly_series(months=36)
        model = ETSModel()
        model.fit(data)
        preds = model.predict(periods=3)

        last_train = data["date"].max()
        for i, row in preds.iterrows():
            expected_month = last_train.month + (i + 1)
            expected_year = last_train.year + (expected_month - 1) // 12
            expected_month = ((expected_month - 1) % 12) + 1
            assert row["date"].month == expected_month
            assert row["date"].year == expected_year


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class TestModelRegistry:

    def test_available_models_returns_both(self):
        """get_available_models should list ets and snaive."""
        models = get_available_models()
        assert "ets" in models
        assert "snaive" in models

    def test_models_are_instantiable(self):
        """Each registered model class should be instantiable."""
        for name, cls in get_available_models().items():
            instance = cls()
            assert instance.name  # should return a non-empty string
