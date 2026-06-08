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

from forecaster import (
    FuelForecaster,
    _normalize_filter,
    _reorder_columns,
    SANITY_CAP_UPPER_MULT,
    SANITY_CAP_LOWER_MULT,
    YOY_GUARD_UPPER_RATIO,
    YOY_GUARD_LOWER_RATIO,
    TREND_EXTRAP_CEILING_MULT,
    MATURE_SITE_MONTHS,
)


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

    def test_short_series_skips_outlier_detection(self):
        """Outlier detection should be a no-op on very short series."""
        fc = FuelForecaster(database=None, min_months_data=24)
        df = pd.DataFrame({
            "date": pd.date_range("2024-01", periods=3, freq="MS"),
            "volume": [100.0, 200.0, 9999.0],
        })
        result = fc._handle_outliers(df.copy(), site_id="100")
        # With < 4 rows, outliers are not touched
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

    def test_bulk_forecasts(self, forecaster):
        """Bulk forecasts should produce per-site-per-grade forecasts."""
        result = forecaster.generate_bulk_forecasts(
            target_month="2025-03",
            show_yoy=False,
        )
        assert not result.empty
        ensemble = result[result["model"] == "ENSEMBLE"]
        combos = ensemble.groupby("site_id")["grade"].nunique()
        assert (combos >= 2).all()

    def test_bulk_ensemble_count_matches_items(self, forecaster):
        """Number of ENSEMBLE rows should equal number of site-grade combos in DB."""
        result = forecaster.generate_bulk_forecasts(
            target_month="2025-03", show_yoy=False,
        )
        ensemble_count = (result["model"] == "ENSEMBLE").sum()
        expected = len(forecaster.db.get_distinct_site_grades())
        assert ensemble_count == expected


# ---------------------------------------------------------------------------
# Summary sheets
# ---------------------------------------------------------------------------

class TestSummarySheets:

    def _get_site_grade_forecasts(self, forecaster):
        """Helper: generate site_grade forecasts for summary testing."""
        return forecaster.generate_bulk_forecasts(
            target_month="2025-03",
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
            target_month="2025-03", show_yoy=False,
            skip_insufficient=False,
        )
        assert not result.empty
        assert result["site_id"].nunique() == 1


# ---------------------------------------------------------------------------
# Adaptive weights and prediction intervals
# ---------------------------------------------------------------------------

class TestAdaptiveWeights:

    def test_no_calibration_uses_default_weights(self, forecaster):
        """Without calibration data, default 70/30 weights should be used."""
        forecast = forecaster.generate_forecast(
            target_month="2025-03", site_id="100", show_yoy=False,
        )
        individual = forecast.set_index("model")["forecast_volume"]
        ensemble_vol = individual["ENSEMBLE"]
        expected = (
            individual["ets"] * 0.7 + individual["snaive"] * 0.3
        )
        assert ensemble_vol == pytest.approx(expected)

    def test_adaptive_weights_used_when_calibrated(self, db):
        """When calibration data exists, site-specific weights should be used."""
        # Store calibration: site 100 gets 50/50
        run_id = db.save_calibration_run({
            "backtest_months": 6, "horizon": 2,
            "min_months": 24, "sites_calibrated": 1, "overall_mape": 5.0,
        })
        db.save_site_weights(run_id, [
            {"site_id": "100", "model_name": "ets", "weight": 0.50,
             "mape_pct": 5.0, "n_months": 6},
            {"site_id": "100", "model_name": "snaive", "weight": 0.50,
             "mape_pct": 5.0, "n_months": 6},
        ])

        fc = FuelForecaster(db, min_months_data=24, use_adaptive_weights=True)
        forecast = fc.generate_forecast(
            target_month="2025-03", site_id="100", show_yoy=False,
        )
        individual = forecast.set_index("model")["forecast_volume"]
        ensemble_vol = individual["ENSEMBLE"]
        expected = (individual["ets"] * 0.50 + individual["snaive"] * 0.50)
        assert ensemble_vol == pytest.approx(expected)

    def test_no_calibration_flag_forces_defaults(self, db):
        """use_adaptive_weights=False should ignore stored calibration."""
        run_id = db.save_calibration_run({
            "backtest_months": 6, "horizon": 2,
            "min_months": 24, "sites_calibrated": 1, "overall_mape": 5.0,
        })
        db.save_site_weights(run_id, [
            {"site_id": "100", "model_name": "ets", "weight": 0.50,
             "mape_pct": 5.0, "n_months": 6},
            {"site_id": "100", "model_name": "snaive", "weight": 0.50,
             "mape_pct": 5.0, "n_months": 6},
        ])

        fc = FuelForecaster(db, min_months_data=24, use_adaptive_weights=False)
        forecast = fc.generate_forecast(
            target_month="2025-03", site_id="100", show_yoy=False,
        )
        individual = forecast.set_index("model")["forecast_volume"]
        ensemble_vol = individual["ENSEMBLE"]
        expected_default = (individual["ets"] * 0.7 + individual["snaive"] * 0.3)
        assert ensemble_vol == pytest.approx(expected_default)


class TestPredictionIntervals:

    def test_intervals_null_without_calibration(self, forecaster):
        """Without calibration data, interval columns should be null."""
        forecast = forecaster.generate_forecast(
            target_month="2025-03", site_id="100", show_yoy=False,
        )
        ensemble = forecast[forecast["model"] == "ENSEMBLE"].iloc[0]
        assert pd.isna(ensemble.get("forecast_p10"))
        assert pd.isna(ensemble.get("forecast_p90"))

    def test_intervals_populated_with_calibration(self, db):
        """With calibration data, intervals should be computed."""
        run_id = db.save_calibration_run({
            "backtest_months": 6, "horizon": 2,
            "min_months": 24, "sites_calibrated": 1, "overall_mape": 5.0,
        })
        db.save_interval_calibration(run_id, [
            {"segment": "site:100", "residual_std": 0.08,
             "residual_p10": -0.10, "residual_p90": 0.12, "n_observations": 12},
        ])

        fc = FuelForecaster(db, min_months_data=24, use_adaptive_weights=True)
        forecast = fc.generate_forecast(
            target_month="2025-03", site_id="100", show_yoy=False,
        )
        ensemble = forecast[forecast["model"] == "ENSEMBLE"].iloc[0]
        assert not pd.isna(ensemble["forecast_p10"])
        assert not pd.isna(ensemble["forecast_p90"])
        assert ensemble["forecast_p10"] < ensemble["forecast_volume"]
        assert ensemble["forecast_p90"] > ensemble["forecast_volume"]

    def test_interval_fallback_to_global(self, db):
        """When site-level intervals are missing, should fall back to global."""
        run_id = db.save_calibration_run({
            "backtest_months": 6, "horizon": 2,
            "min_months": 24, "sites_calibrated": 1, "overall_mape": 5.0,
        })
        db.save_interval_calibration(run_id, [
            {"segment": "global", "residual_std": 0.10,
             "residual_p10": -0.15, "residual_p90": 0.15, "n_observations": 50},
        ])

        fc = FuelForecaster(db, min_months_data=24, use_adaptive_weights=True)
        forecast = fc.generate_forecast(
            target_month="2025-03", site_id="100", show_yoy=False,
        )
        ensemble = forecast[forecast["model"] == "ENSEMBLE"].iloc[0]
        assert not pd.isna(ensemble["forecast_p10"])

    def test_risk_flag_high_for_wide_interval(self, db):
        """Wide intervals should trigger HIGH risk flag."""
        run_id = db.save_calibration_run({
            "backtest_months": 6, "horizon": 2,
            "min_months": 24, "sites_calibrated": 1, "overall_mape": 5.0,
        })
        # Very wide interval: -40% to +40%
        db.save_interval_calibration(run_id, [
            {"segment": "site:100", "residual_std": 0.30,
             "residual_p10": -0.40, "residual_p90": 0.40, "n_observations": 12},
        ])

        fc = FuelForecaster(db, min_months_data=24, use_adaptive_weights=True)
        forecast = fc.generate_forecast(
            target_month="2025-03", site_id="100", show_yoy=False,
        )
        ensemble = forecast[forecast["model"] == "ENSEMBLE"].iloc[0]
        assert ensemble["risk_flag"] == "HIGH"

    def test_forecast_p10_non_negative(self, db):
        """forecast_p10 should be clamped to 0 (volumes are non-negative)."""
        run_id = db.save_calibration_run({
            "backtest_months": 6, "horizon": 2,
            "min_months": 24, "sites_calibrated": 1, "overall_mape": 5.0,
        })
        # Extreme negative residual that would make p10 negative
        db.save_interval_calibration(run_id, [
            {"segment": "site:100", "residual_std": 1.5,
             "residual_p10": -2.0, "residual_p90": 0.50, "n_observations": 12},
        ])

        fc = FuelForecaster(db, min_months_data=24, use_adaptive_weights=True)
        forecast = fc.generate_forecast(
            target_month="2025-03", site_id="100", show_yoy=False,
        )
        ensemble = forecast[forecast["model"] == "ENSEMBLE"].iloc[0]
        assert ensemble["forecast_p10"] >= 0


# ---------------------------------------------------------------------------
# Sanity caps & YoY guardrails
# ---------------------------------------------------------------------------

class TestSanityCaps:

    def _make_forecaster(self):
        """Create a standalone FuelForecaster with no database."""
        return FuelForecaster(database=None, min_months_data=24)

    def _make_monthly_data(self, volumes):
        """Build a simple monthly DataFrame for testing."""
        return pd.DataFrame({
            "date": pd.date_range("2024-01", periods=len(volumes), freq="MS"),
            "volume": [float(v) for v in volumes],
        })

    def test_mature_site_upper_cap(self):
        """24-month site whose forecast exceeds 3x max should be capped."""
        fc = self._make_forecaster()
        bounds = {
            ("S1", "UNL"): {
                "upper_cap": SANITY_CAP_UPPER_MULT * 1000,  # 3000
                "lower_floor": SANITY_CAP_LOWER_MULT * 200,  # 20
                "is_mature": True,
            }
        }
        raw = self._make_monthly_data([500] * 24)
        capped, unclamped, label, upper, lower, _yoy = fc._apply_bounds(
            5000.0, "S1", "UNL", bounds, raw, prior_year_volume=None,
        )
        assert capped == 3000.0
        assert unclamped == 5000.0
        assert label == "upper_cap"

    def test_mature_site_lower_floor(self):
        """Forecast below 0.1x min should be floored."""
        fc = self._make_forecaster()
        bounds = {
            ("S1", "UNL"): {
                "upper_cap": 3000.0,
                "lower_floor": SANITY_CAP_LOWER_MULT * 200,  # 20
                "is_mature": True,
            }
        }
        raw = self._make_monthly_data([500] * 24)
        capped, unclamped, label, upper, lower, _yoy = fc._apply_bounds(
            5.0, "S1", "UNL", bounds, raw, prior_year_volume=None,
        )
        assert capped == 20.0
        assert unclamped == 5.0
        assert label == "lower_floor"

    def test_new_site_uses_grade_cohort(self):
        """6-month site should use cohort bounds, not its own history."""
        fc = self._make_forecaster()
        # Cohort-derived bounds (not site-specific)
        bounds = {
            ("NEW1", "DSL"): {
                "upper_cap": SANITY_CAP_UPPER_MULT * 2000,  # 6000
                "lower_floor": SANITY_CAP_LOWER_MULT * 100,  # 10
                "is_mature": False,
            }
        }
        raw = self._make_monthly_data([300] * 6)
        capped, unclamped, label, upper, lower, _yoy = fc._apply_bounds(
            8000.0, "NEW1", "DSL", bounds, raw, prior_year_volume=None,
        )
        assert capped == 6000.0
        assert label == "upper_cap"

    def test_trend_adjusted_cap(self):
        """Growing new site gets widened cap via trend extrapolation."""
        fc = self._make_forecaster()
        # cohort-based upper_cap = 3 * 500 = 1500
        bounds = {
            ("NEW2", "UNL"): {
                "upper_cap": SANITY_CAP_UPPER_MULT * 500,  # 1500
                "lower_floor": 10.0,
                "is_mature": False,
            }
        }
        # Ramping volumes: trend is clearly upward
        raw = self._make_monthly_data([200, 400, 600, 800, 1000])
        most_recent = 1000.0
        # Extrapolation ceiling: 2x most recent = 2000
        # Trend-widened cap: 3 * min(extrapolated, 2000)
        # polyfit slope is ~200/month, extrapolated forward by ~5 months = 2000
        # ceiling clips to 2000, so trend upper = 3*2000 = 6000
        capped, unclamped, label, upper, lower, _yoy = fc._apply_bounds(
            2000.0, "NEW2", "UNL", bounds, raw, prior_year_volume=None,
        )
        # 2000 < trend-widened cap, so no cap should fire
        assert label == ""
        assert capped == 2000.0
        # Verify trend widening actually raised the cap above base 1500
        assert upper > 1500.0

    def test_yoy_cap_applied(self):
        """Forecast > 3x prior year should be capped to 3x."""
        fc = self._make_forecaster()
        # Empty bounds -> no sanity clamp, isolating the YoY guardrail.
        capped, _unclamped, _sanity, _upper, _lower, yoy_label = fc._apply_bounds(
            4000.0, "S1", "UNL", {}, None, prior_year_volume=1000.0,
        )
        assert capped == 3000.0
        assert yoy_label == "yoy_upper"

    def test_yoy_skipped_no_prior_year(self):
        """No prior year data -> YoY skipped."""
        fc = self._make_forecaster()
        capped, _, _, _, _, yoy_label = fc._apply_bounds(
            5000.0, "S1", "UNL", {}, None, prior_year_volume=None,
        )
        assert capped == 5000.0
        assert yoy_label == ""

        capped2, _, _, _, _, yoy_label2 = fc._apply_bounds(
            5000.0, "S1", "UNL", {}, None, prior_year_volume=0.0,
        )
        assert capped2 == 5000.0
        assert yoy_label2 == ""

    def test_no_cap_normal_forecast(self):
        """Forecast within bounds: untouched, unclamped == clamped."""
        fc = self._make_forecaster()
        bounds = {
            ("S1", "UNL"): {
                "upper_cap": 3000.0,
                "lower_floor": 20.0,
                "is_mature": True,
            }
        }
        raw = self._make_monthly_data([500] * 24)
        capped, unclamped, label, upper, lower, _yoy = fc._apply_bounds(
            1500.0, "S1", "UNL", bounds, raw, prior_year_volume=None,
        )
        assert capped == 1500.0
        assert unclamped == 1500.0
        assert label == ""

    def test_both_caps_fire(self):
        """When sanity cap fires and then YoY also fires, both labels populated."""
        fc = self._make_forecaster()

        # Build a forecast DF with an ENSEMBLE row
        forecast_df = pd.DataFrame([{
            "model": "ENSEMBLE",
            "forecast_volume": 10000.0,
            "site_id": "S1",
            "grade": "UNL",
            "target_month": "2025-03",
            "prior_year_volume": 1500.0,
            "yoy_change_pct": None,
            "forecast_p10": pd.NA,
            "forecast_p90": pd.NA,
            "interval_width_pct": pd.NA,
        }])

        bounds = {
            ("S1", "UNL"): {
                "upper_cap": 5000.0,  # sanity caps to 5000
                "lower_floor": 10.0,
                "is_mature": True,
            }
        }
        raw = self._make_monthly_data([500] * 24)

        result = fc._apply_caps_to_forecast_df(forecast_df, bounds, raw)
        row = result.iloc[0]

        assert row["sanity_cap_applied"] == "upper_cap"
        # After sanity cap: 5000. YoY upper = 1500 * 3 = 4500
        assert row["yoy_cap_applied"] == "yoy_upper"
        assert float(row["forecast_volume"]) == pytest.approx(4500.0)
        assert float(row["forecast_volume_unclamped"]) == pytest.approx(10000.0)


class TestReviewSheet:

    def _make_forecaster(self):
        return FuelForecaster(database=None, min_months_data=24)

    def test_includes_capped_and_sparse_sites(self):
        """Review sheet should include capped rows and sparse rows."""
        fc = self._make_forecaster()
        forecasts = pd.DataFrame([
            {
                "model": "ENSEMBLE", "site_id": "A", "grade": "UNL",
                "sanity_cap_applied": "upper_cap", "yoy_cap_applied": "",
                "data_quality": "good", "risk_flag": "LOW",
                "forecast_volume": 100, "forecast_volume_unclamped": 200,
                "upper_cap_value": 100, "lower_floor_value": 10,
                "interval_width_pct": 5.0, "note": None,
                "months_available": 36,
            },
            {
                "model": "ENSEMBLE", "site_id": "B", "grade": "DSL",
                "sanity_cap_applied": "", "yoy_cap_applied": "",
                "data_quality": "sparse", "risk_flag": "LOW",
                "forecast_volume": 50, "forecast_volume_unclamped": 50,
                "upper_cap_value": 300, "lower_floor_value": 5,
                "interval_width_pct": 10.0, "note": None,
                "months_available": 8,
            },
            {
                "model": "ENSEMBLE", "site_id": "C", "grade": "UNL",
                "sanity_cap_applied": "", "yoy_cap_applied": "",
                "data_quality": "good", "risk_flag": "LOW",
                "forecast_volume": 80, "forecast_volume_unclamped": 80,
                "upper_cap_value": 300, "lower_floor_value": 5,
                "interval_width_pct": 8.0, "note": None,
                "months_available": 36,
            },
        ])
        review = fc._create_review_sheet(forecasts)
        assert len(review) == 2
        assert "A" in review["site_id"].values
        assert "B" in review["site_id"].values
        assert "C" not in review["site_id"].values

    def test_sort_order(self):
        """Capped should come before sparse, sparse before HIGH risk only."""
        fc = self._make_forecaster()
        forecasts = pd.DataFrame([
            {
                "model": "ENSEMBLE", "site_id": "RISK", "grade": "UNL",
                "sanity_cap_applied": "", "yoy_cap_applied": "",
                "data_quality": "good", "risk_flag": "HIGH",
                "forecast_volume": 90, "forecast_volume_unclamped": 90,
                "upper_cap_value": 300, "lower_floor_value": 5,
                "interval_width_pct": 35.0, "note": None,
                "months_available": 36,
            },
            {
                "model": "ENSEMBLE", "site_id": "SPARSE", "grade": "UNL",
                "sanity_cap_applied": "", "yoy_cap_applied": "",
                "data_quality": "sparse", "risk_flag": "LOW",
                "forecast_volume": 50, "forecast_volume_unclamped": 50,
                "upper_cap_value": 300, "lower_floor_value": 5,
                "interval_width_pct": 10.0, "note": None,
                "months_available": 8,
            },
            {
                "model": "ENSEMBLE", "site_id": "CAPPED", "grade": "UNL",
                "sanity_cap_applied": "upper_cap", "yoy_cap_applied": "",
                "data_quality": "good", "risk_flag": "LOW",
                "forecast_volume": 100, "forecast_volume_unclamped": 200,
                "upper_cap_value": 100, "lower_floor_value": 10,
                "interval_width_pct": 5.0, "note": None,
                "months_available": 36,
            },
        ])
        review = fc._create_review_sheet(forecasts)
        assert len(review) == 3
        assert review.iloc[0]["site_id"] == "CAPPED"
        assert review.iloc[1]["site_id"] == "SPARSE"
        assert review.iloc[2]["site_id"] == "RISK"

    def test_empty_when_all_clean(self):
        """No review rows when all forecasts are clean."""
        fc = self._make_forecaster()
        forecasts = pd.DataFrame([
            {
                "model": "ENSEMBLE", "site_id": "A", "grade": "UNL",
                "sanity_cap_applied": "", "yoy_cap_applied": "",
                "data_quality": "good", "risk_flag": "LOW",
                "forecast_volume": 100, "forecast_volume_unclamped": 100,
                "upper_cap_value": 300, "lower_floor_value": 5,
                "interval_width_pct": 5.0, "note": None,
                "months_available": 36,
            },
        ])
        review = fc._create_review_sheet(forecasts)
        assert review.empty
