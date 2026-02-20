"""
Tests for forecaster.py — the core forecasting pipeline.

These are integration-style tests: they wire real models to a real
(temporary) database and verify the full pipeline produces correct
output structures and reasonable values.

The 'db' and 'forecaster' fixtures come from conftest.py.
"""

import numpy as np
import pandas as pd
import pytest

from forecaster import FuelForecaster, _normalize_filter, _reorder_columns


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

class TestNormalizeFilter:

    def test_empty_string_becomes_none(self):
        assert _normalize_filter("") is None

    def test_whitespace_becomes_none(self):
        assert _normalize_filter("   ") is None

    def test_none_stays_none(self):
        assert _normalize_filter(None) is None

    def test_valid_string_unchanged(self):
        assert _normalize_filter("100") == "100"


class TestReorderColumns:

    def test_puts_desired_first(self):
        df = pd.DataFrame({"c": [1], "a": [2], "b": [3]})
        result = _reorder_columns(df, ["a", "b"])
        assert list(result.columns) == ["a", "b", "c"]

    def test_missing_desired_cols_ignored(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        result = _reorder_columns(df, ["z", "x"])
        assert list(result.columns) == ["x", "y"]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

class TestDataPreparation:

    def test_prepare_monthly_data_returns_monthly(self, forecaster):
        """Output should have one row per month, not per day."""
        data = forecaster.prepare_monthly_data(site_id="100")
        assert (data["date"].dt.day == 1).all()
        assert data["date"].is_unique

    def test_prepare_monthly_data_for_specific_site(self, forecaster):
        """Passing a site_id should only aggregate that site's data."""
        data_100 = forecaster.prepare_monthly_data(site_id="100")
        data_200 = forecaster.prepare_monthly_data(site_id="200")
        assert data_100["volume"].sum() != data_200["volume"].sum()

    def test_fill_gaps_creates_continuous_range(self, forecaster):
        """After gap filling, months should be contiguous with no holes."""
        data = forecaster.prepare_monthly_data(site_id="100", fill_gaps=True)
        diffs = data["date"].diff().dropna()
        assert (diffs.dt.days >= 28).all()
        assert (diffs.dt.days <= 31).all()

    def test_no_fill_gaps_preserves_original(self, forecaster):
        """fill_gaps=False should return data as-is from aggregation."""
        filled = forecaster.prepare_monthly_data(site_id="100", fill_gaps=True)
        raw = forecaster.prepare_monthly_data(site_id="100", fill_gaps=False)
        assert len(raw) <= len(filled)

    def test_empty_string_site_id_treated_as_none(self, forecaster):
        """Empty-string site_id should return data for ALL sites."""
        all_data = forecaster.prepare_monthly_data(site_id=None)
        empty_str = forecaster.prepare_monthly_data(site_id="")
        assert len(all_data) == len(empty_str)
        assert all_data["volume"].sum() == pytest.approx(empty_str["volume"].sum())


# ---------------------------------------------------------------------------
# Outlier handling
# ---------------------------------------------------------------------------

class TestOutlierHandling:

    def test_extreme_spike_is_capped(self, forecaster):
        """A single extreme spike should be capped, not left as-is."""
        data = forecaster.prepare_monthly_data(
            site_id="100", handle_outliers=False
        )
        original_max = data["volume"].max()
        data.loc[len(data) // 2, "volume"] = original_max * 100

        cleaned = forecaster._handle_outliers(data.copy(), site_id="100")
        assert cleaned["volume"].max() < original_max * 100

    def test_outlier_handling_preserves_length(self, forecaster):
        """Outlier handling should cap values, not remove rows."""
        data = forecaster.prepare_monthly_data(
            site_id="100", handle_outliers=False
        )
        cleaned = forecaster._handle_outliers(data.copy(), site_id="100")
        assert len(cleaned) == len(data)

    def test_volumes_stay_non_negative(self, forecaster):
        """After outlier handling, no volume should be negative."""
        data = forecaster.prepare_monthly_data(
            site_id="100", handle_outliers=True
        )
        assert (data["volume"] >= 0).all()

    def test_short_series_skips_spike_detection(self):
        """_detect_spikes should be a no-op on very short series."""
        fc = FuelForecaster(database=None, min_months_data=24)
        df = pd.DataFrame({
            "date": pd.date_range("2024-01", periods=3, freq="MS"),
            "volume": [100.0, 200.0, 9999.0],
        })
        result = fc._detect_spikes(df.copy(), "test")
        # With < 4 rows, spikes are not touched
        assert result["volume"].iloc[-1] == 9999.0


# ---------------------------------------------------------------------------
# Forecast generation
# ---------------------------------------------------------------------------

class TestForecastGeneration:

    def test_generate_forecast_returns_all_models(self, forecaster):
        """Forecast should include ets, snaive, and ENSEMBLE rows."""
        forecast = forecaster.generate_forecast(
            target_month="2025-03", site_id="100"
        )
        models_in_output = set(forecast["model"].unique())
        assert "ets" in models_in_output
        assert "snaive" in models_in_output
        assert "ENSEMBLE" in models_in_output

    def test_forecast_has_expected_columns(self, forecaster):
        """Output should include key columns."""
        forecast = forecaster.generate_forecast(
            target_month="2025-03", site_id="100"
        )
        required = {"model", "target_month", "forecast_volume", "site_id", "grade"}
        assert required.issubset(forecast.columns)

    def test_forecast_volumes_are_positive(self, forecaster):
        """All forecast volumes should be positive."""
        forecast = forecaster.generate_forecast(
            target_month="2025-03", site_id="100"
        )
        assert (forecast["forecast_volume"] > 0).all()

    def test_ensemble_is_weighted_average_of_models(self, forecaster):
        """ENSEMBLE should be weighted toward ETS by configured weights."""
        forecast = forecaster.generate_forecast(
            target_month="2025-03", site_id="100"
        )
        individual = forecast.set_index("model")["forecast_volume"]
        ensemble = forecast[forecast["model"] == "ENSEMBLE"]

        ets_weight = forecaster.ensemble_weights["ets"]
        snaive_weight = forecaster.ensemble_weights["snaive"]
        expected_weighted = (
            individual["ets"] * ets_weight + individual["snaive"] * snaive_weight
        ) / (ets_weight + snaive_weight)
        actual_ensemble = ensemble["forecast_volume"].iloc[0]
        assert actual_ensemble == pytest.approx(expected_weighted)

    def test_yoy_columns_present_by_default(self, forecaster):
        """YoY comparison columns should be included when show_yoy=True."""
        forecast = forecaster.generate_forecast(
            target_month="2025-03", site_id="100", show_yoy=True
        )
        assert "prior_year_volume" in forecast.columns
        assert "yoy_change_pct" in forecast.columns

    def test_yoy_columns_absent_when_disabled(self, forecaster):
        """YoY columns should not appear when show_yoy=False."""
        forecast = forecaster.generate_forecast(
            target_month="2025-03", site_id="100", show_yoy=False
        )
        assert "prior_year_volume" not in forecast.columns
        assert "yoy_change_pct" not in forecast.columns

    def test_target_month_in_past_raises(self, forecaster):
        """Requesting a forecast for a month already in the data should raise."""
        with pytest.raises(ValueError, match="not in the future"):
            forecaster.generate_forecast(
                target_month="2022-06", site_id="100"
            )

    def test_snaive_fallback_flag_is_target_month_specific(self):
        """Fallback flag should reflect target month only, not earlier horizon steps."""
        model_input = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2023-08-01",
                        "2024-01-01",
                        "2024-02-01",
                        "2024-03-01",
                        "2024-04-01",
                        "2024-05-01",
                        "2024-06-01",
                    ]
                ),
                "volume": [80.0, 11.0, 21.0, 31.0, 41.0, 51.0, 61.0],
            }
        )

        local_forecaster = FuelForecaster(database=None, min_months_data=24)
        forecast = local_forecaster.generate_forecast(
            target_month="2024-08",
            models_to_use=["snaive"],
            monthly_data=model_input,
            monthly_data_raw=model_input,
            show_yoy=False,
        )

        snaive_row = forecast[forecast["model"] == "snaive"].iloc[0]
        assert bool(snaive_row["snaive_used_fallback"]) is False
        note = "" if pd.isna(snaive_row["note"]) else str(snaive_row["note"])
        assert "SNAIVE used fallback" not in note

    def test_forecast_exact_value_range(self, forecaster):
        """Forecast volumes should be within a reasonable range of training data."""
        data = forecaster.prepare_monthly_data(site_id="100", handle_outliers=False)
        train_median = data["volume"].median()

        forecast = forecaster.generate_forecast(
            target_month="2025-03", site_id="100", show_yoy=False,
        )
        ensemble_vol = forecast[forecast["model"] == "ENSEMBLE"]["forecast_volume"].iloc[0]
        # Should be within 5x of training median (sanity check)
        assert ensemble_vol < train_median * 5
        assert ensemble_vol > train_median * 0.2


# ---------------------------------------------------------------------------
# YoY calculation edge cases
# ---------------------------------------------------------------------------

class TestYoYCalculation:

    def test_yoy_normal(self, forecaster):
        """Standard YoY: 10% increase."""
        result = forecaster._calculate_yoy_change(110.0, 100.0)
        assert result == pytest.approx(10.0)

    def test_yoy_decrease(self, forecaster):
        """YoY should be negative for a decrease."""
        result = forecaster._calculate_yoy_change(90.0, 100.0)
        assert result == pytest.approx(-10.0)

    def test_yoy_zero_prior(self, forecaster):
        """Zero prior year should return None (avoid division by zero)."""
        assert forecaster._calculate_yoy_change(100.0, 0) is None

    def test_yoy_none_prior(self, forecaster):
        """None prior year should return None."""
        assert forecaster._calculate_yoy_change(100.0, None) is None

    def test_yoy_nan_prior(self, forecaster):
        """NaN prior year should return None."""
        assert forecaster._calculate_yoy_change(100.0, float("nan")) is None


# ---------------------------------------------------------------------------
# Bulk forecasts
# ---------------------------------------------------------------------------

class TestBulkForecasts:

    def test_bulk_by_site(self, forecaster):
        """Bulk by-site should produce forecasts for both test sites."""
        result = forecaster.generate_bulk_forecasts(
            target_month="2025-03",
            by="site",
            show_yoy=False,
        )
        assert not result.empty
        sites = result["site_id"].unique()
        # The fixture has 2 sites; both should appear
        assert len(sites) == 2

    def test_bulk_by_grade(self, forecaster):
        """Bulk by-grade should produce forecasts for UNL and DSL."""
        result = forecaster.generate_bulk_forecasts(
            target_month="2025-03",
            by="grade",
            show_yoy=False,
        )
        assert not result.empty
        grades = result["grade"].unique()
        assert set(grades) >= {"UNL", "DSL"}

    def test_bulk_by_site_grade(self, forecaster):
        """Bulk by-site_grade should produce per-site-per-grade forecasts."""
        result = forecaster.generate_bulk_forecasts(
            target_month="2025-03",
            by="site_grade",
            show_yoy=False,
        )
        assert not result.empty
        ensemble = result[result["model"] == "ENSEMBLE"]
        combos = ensemble.groupby("site_id")["grade"].nunique()
        assert (combos >= 2).all()

    def test_invalid_by_raises(self, forecaster):
        """Invalid 'by' parameter should raise ValueError."""
        with pytest.raises(ValueError, match="must be"):
            forecaster.generate_bulk_forecasts(
                target_month="2025-03", by="invalid"
            )

    def test_bulk_ensemble_count_matches_items(self, forecaster):
        """Number of ENSEMBLE rows should equal number of items forecast."""
        result = forecaster.generate_bulk_forecasts(
            target_month="2025-03", by="site", show_yoy=False,
        )
        ensemble_count = (result["model"] == "ENSEMBLE").sum()
        sites = result["site_id"].unique()
        assert ensemble_count == len(sites)


# ---------------------------------------------------------------------------
# Summary sheets
# ---------------------------------------------------------------------------

class TestSummarySheets:

    def _get_site_grade_forecasts(self, forecaster):
        """Helper: generate site_grade forecasts for summary testing."""
        return forecaster.generate_bulk_forecasts(
            target_month="2025-03",
            by="site_grade",
            show_yoy=True,
        )

    def test_site_summary_one_row_per_site(self, forecaster):
        """Site summary should have exactly one row per site (ENSEMBLE only)."""
        forecasts = self._get_site_grade_forecasts(forecaster)
        summary = forecaster._create_site_summary(forecasts)

        # The fixture always has grade-level detail, so summary should not be empty
        assert not summary.empty, "Site summary should not be empty with test data"
        assert "model" not in summary.columns
        assert summary["site_id"].is_unique

    def test_site_summary_matches_ensemble_sum(self, forecaster):
        """Site summary totals should equal sum of ENSEMBLE grade forecasts."""
        forecasts = self._get_site_grade_forecasts(forecaster)
        summary = forecaster._create_site_summary(forecasts)

        assert not summary.empty, "Site summary should not be empty with test data"

        ensemble_grades = forecasts[
            (forecasts["model"] == "ENSEMBLE")
            & (forecasts["grade"] != "ALL")
        ]
        for site_id in summary["site_id"].unique():
            expected_sum = ensemble_grades[
                ensemble_grades["site_id"] == site_id
            ]["forecast_volume"].sum()
            actual_sum = summary[
                summary["site_id"] == site_id
            ]["forecast_volume"].iloc[0]
            assert actual_sum == pytest.approx(expected_sum, rel=1e-6)

    def test_product_summary_groups_by_grade(self, forecaster):
        """Product summary should have one row per grade."""
        forecasts = self._get_site_grade_forecasts(forecaster)
        summary = forecaster._create_product_summary(forecasts)

        assert not summary.empty, "Product summary should not be empty with test data"
        assert summary["grade"].is_unique

    def test_bu_summary_is_single_row(self, forecaster):
        """BU summary should have exactly one row (grand total)."""
        forecasts = self._get_site_grade_forecasts(forecaster)
        summary = forecaster._create_bu_summary(forecasts)

        assert len(summary) == 1
        assert summary["forecast_volume"].iloc[0] > 0


# ---------------------------------------------------------------------------
# Export (file output)
# ---------------------------------------------------------------------------

class TestExport:

    def test_export_to_excel(self, forecaster, tmp_path):
        """Excel export should create a file with expected sheets."""
        forecasts = forecaster.generate_bulk_forecasts(
            target_month="2025-03",
            by="site_grade",
            show_yoy=True,
        )
        output = str(tmp_path / "test_output.xlsx")
        forecaster._export_results(forecasts, [], output)

        xl = pd.ExcelFile(output)
        assert "Forecasts" in xl.sheet_names
        assert "Model Detail" in xl.sheet_names

    def test_export_to_csv(self, forecaster, tmp_path):
        """CSV export should create the main file."""
        forecasts = forecaster.generate_bulk_forecasts(
            target_month="2025-03",
            by="site",
            show_yoy=False,
        )
        output = str(tmp_path / "test_output.csv")
        forecaster._export_results(forecasts, [], output)

        df = pd.read_csv(output)
        assert not df.empty
        assert "forecast_volume" in df.columns


# ---------------------------------------------------------------------------
# Edge-case: single site / sparse data
# ---------------------------------------------------------------------------

class TestEdgeCases:

    @pytest.fixture
    def sparse_forecaster(self, tmp_path):
        """A forecaster with only 12 months of data for a single site."""
        from conftest import _build_sample_sales
        from database import FuelDatabase

        db_path = str(tmp_path / "sparse.db")
        database = FuelDatabase(db_path)

        df = _build_sample_sales(
            site_ids=("999",), grades=("UNL",), start="2024-01-01", months=12,
        )
        records = df[
            [
                "site_id", "grade", "day", "brand", "site", "address",
                "city", "state", "owner", "b_unit", "stock", "delivered",
                "volume", "is_estimated", "total_sales", "target",
            ]
        ].itertuples(index=False, name=None)

        sql = """
            INSERT OR IGNORE INTO sales (
                site_id, grade, day, brand, site, address, city, state,
                owner, b_unit, stock, delivered, volume, is_estimated,
                total_sales, target
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with database.conn:
            database.conn.executemany(sql, records)

        fc = FuelForecaster(database, min_months_data=24)
        yield fc
        database.close()

    def test_sparse_data_still_forecasts(self, sparse_forecaster):
        """Even with only 12 months, a forecast should succeed."""
        forecast = sparse_forecaster.generate_forecast(
            target_month="2025-03", site_id="999", grade="UNL", show_yoy=False,
        )
        assert not forecast.empty
        assert (forecast["forecast_volume"] > 0).all()
        # Should flag sparse data quality
        assert forecast["data_quality"].iloc[0] in ("sparse", "very_sparse")

    def test_single_site_bulk(self, sparse_forecaster):
        """Bulk forecast with a single site should still succeed."""
        result = sparse_forecaster.generate_bulk_forecasts(
            target_month="2025-03", by="site", show_yoy=False,
            skip_insufficient=False,
        )
        assert not result.empty
        assert result["site_id"].nunique() == 1
