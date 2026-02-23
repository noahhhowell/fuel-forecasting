"""
Tests for database.py — data loading, querying, and deduplication.

Each test function name starts with test_ so pytest can discover it.
The 'db' parameter is a fixture defined in conftest.py — pytest
automatically injects it.
"""

import pandas as pd
import pytest

from database import FuelDatabase


# ---------------------------------------------------------------------------
# Schema & basic queries
# ---------------------------------------------------------------------------

class TestDatabaseQueries:
    """Tests for querying data out of the database."""

    def test_summary_stats_has_expected_keys(self, db):
        """get_summary_stats should return a dict with the right structure."""
        stats = db.get_summary_stats()
        assert stats["total_records"] > 0
        assert stats["unique_sites"] == 2  # conftest creates sites 100, 200
        assert "UNL" in stats["fuel_grades"]
        assert "DSL" in stats["fuel_grades"]
        assert " to " in stats["date_range"]

    def test_get_sales_data_returns_dataframe(self, db):
        """Basic query should return a non-empty DataFrame."""
        df = db.get_sales_data()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "volume" in df.columns
        assert "day" in df.columns

    def test_filter_by_site_id(self, db):
        """Filtering by site_id should return only that site's data."""
        df = db.get_sales_data(site_ids=["100"])
        assert (df["site_id"] == "100").all()

    def test_filter_by_grade(self, db):
        """Filtering by grade should return only that grade's data."""
        df = db.get_sales_data(grades=["DSL"])
        assert (df["grade"] == "DSL").all()

    def test_filter_by_date_range(self, db):
        """Date filters should narrow results correctly."""
        df = db.get_sales_data(start_date="2023-06-01", end_date="2023-06-30")
        dates = pd.to_datetime(df["day"])
        assert dates.min() >= pd.Timestamp("2023-06-01")
        assert dates.max() <= pd.Timestamp("2023-06-30")

    def test_exclude_estimated_default(self, db):
        """By default, estimated rows are excluded."""
        # Our sample data has is_estimated=0 for all rows, so all should appear
        df_excl = db.get_sales_data(exclude_estimated=True)
        df_incl = db.get_sales_data(exclude_estimated=False)
        assert len(df_excl) == len(df_incl)  # no estimated rows in sample

    def test_distinct_sites(self, db):
        """get_distinct_sites should return both test sites."""
        sites = db.get_distinct_sites()
        assert len(sites) == 2
        assert set(sites["site_id"]) == {"100", "200"}

    def test_distinct_site_grades(self, db):
        """Should return all site-grade combinations."""
        combos = db.get_distinct_site_grades()
        # 2 sites x 2 grades = 4 combinations
        assert len(combos) == 4


# ---------------------------------------------------------------------------
# Data loading & deduplication
# ---------------------------------------------------------------------------

class TestDataLoading:
    """Tests for loading data and handling duplicates."""

    def test_load_csv_inserts_rows(self, tmp_path):
        """Loading a CSV should insert records into the database."""
        # Create a tiny CSV
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "site_id,grade,day,volume,is_estimated\n"
            "999,UNL,2024-01-01,500.0,0\n"
            "999,UNL,2024-01-02,510.0,0\n"
        )

        db = FuelDatabase(str(tmp_path / "load_test.db"))
        try:
            stats = db.load_from_csv(str(csv_path))
            assert stats["inserted"] == 2
            assert stats["duplicates"] == 0
        finally:
            db.close()

    def test_duplicate_rows_are_skipped(self, tmp_path):
        """Loading the same data twice should skip duplicates."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "site_id,grade,day,volume,is_estimated\n"
            "999,UNL,2024-01-01,500.0,0\n"
        )

        db = FuelDatabase(str(tmp_path / "dedup_test.db"))
        try:
            first = db.load_from_csv(str(csv_path))
            second = db.load_from_csv(str(csv_path))
            assert first["inserted"] == 1
            assert second["inserted"] == 0
            assert second["duplicates"] == 1
        finally:
            db.close()

    def test_column_mapping_handles_variants(self, tmp_path):
        """Column mapping should handle common header variations."""
        csv_path = tmp_path / "variant.csv"
        # "Site ID" instead of "site_id", "Grade" instead of "grade", etc.
        csv_path.write_text(
            "Site ID,Grade,Day,Volume,Estimated\n"
            "888,DSL,2024-03-01,750.0,0\n"
        )

        db = FuelDatabase(str(tmp_path / "mapping_test.db"))
        try:
            stats = db.load_from_csv(str(csv_path))
            assert stats["inserted"] == 1

            df = db.get_sales_data(site_ids=["888"])
            assert len(df) == 1
            assert df.iloc[0]["grade"] == "DSL"
        finally:
            db.close()


# ---------------------------------------------------------------------------
# NULL is_estimated handling (the COALESCE fix)
# ---------------------------------------------------------------------------

class TestNullEstimated:
    """Verify that NULL is_estimated rows are treated as non-estimated."""

    def test_null_is_estimated_included_when_excluding_estimated(self, tmp_path):
        """Rows with NULL is_estimated should NOT be dropped."""
        db = FuelDatabase(str(tmp_path / "null_est.db"))
        try:
            # Insert a row with NULL is_estimated directly via SQL
            db.conn.execute(
                "INSERT INTO sales (site_id, grade, day, volume, is_estimated) "
                "VALUES ('777', 'UNL', '2024-01-01', 100.0, NULL)"
            )
            db.conn.commit()

            df = db.get_sales_data(exclude_estimated=True)
            assert len(df) == 1, "NULL is_estimated row should be included"
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Calibration table CRUD
# ---------------------------------------------------------------------------

class TestCalibrationCRUD:
    """Tests for calibration-related database methods."""

    def test_save_and_get_calibration_run(self, db):
        """Should save a calibration run and retrieve it."""
        run_id = db.save_calibration_run({
            "backtest_months": 12,
            "horizon": 2,
            "min_months": 24,
            "sites_calibrated": 5,
            "overall_mape": 7.5,
        })
        assert run_id is not None
        assert run_id > 0

        latest = db.get_latest_calibration_run()
        assert latest is not None
        assert latest["run_id"] == run_id
        assert latest["backtest_months"] == 12
        assert latest["overall_mape"] == 7.5

    def test_save_and_get_site_weights(self, db):
        """Should save weights and retrieve them in bulk."""
        run_id = db.save_calibration_run({
            "backtest_months": 6, "horizon": 2,
            "min_months": 24, "sites_calibrated": 2, "overall_mape": 8.0,
        })
        weights = [
            {"site_id": "100", "model_name": "ets", "weight": 0.65,
             "mape_pct": 4.2, "n_months": 6},
            {"site_id": "100", "model_name": "snaive", "weight": 0.35,
             "mape_pct": 8.1, "n_months": 6},
            {"site_id": "200", "model_name": "ets", "weight": 0.50,
             "mape_pct": 6.0, "n_months": 6},
            {"site_id": "200", "model_name": "snaive", "weight": 0.50,
             "mape_pct": 6.0, "n_months": 6},
        ]
        rows_written = db.save_site_weights(run_id, weights)
        assert rows_written == 4

        bulk = db.get_site_weights_bulk()
        assert "100" in bulk
        assert bulk["100"]["ets"] == 0.65
        assert bulk["100"]["snaive"] == 0.35
        assert "200" in bulk

    def test_save_and_get_interval_calibration(self, db):
        """Should save and retrieve interval calibration factors."""
        run_id = db.save_calibration_run({
            "backtest_months": 6, "horizon": 2,
            "min_months": 24, "sites_calibrated": 1, "overall_mape": 5.0,
        })
        segments = [
            {"segment": "site:100", "residual_std": 0.08,
             "residual_p10": -0.12, "residual_p90": 0.10, "n_observations": 12},
            {"segment": "global", "residual_std": 0.10,
             "residual_p10": -0.15, "residual_p90": 0.13, "n_observations": 50},
        ]
        db.save_interval_calibration(run_id, segments)

        factors = db.get_interval_factors("site:100")
        assert factors is not None
        assert factors["residual_std"] == 0.08
        assert factors["residual_p10"] == -0.12

        global_f = db.get_interval_factors("global")
        assert global_f is not None
        assert global_f["n_observations"] == 50

    def test_get_interval_factors_missing_returns_none(self, db):
        """Missing segment should return None."""
        assert db.get_interval_factors("site:nonexistent") is None

    def test_get_latest_calibration_run_empty(self, tmp_path):
        """Empty DB should return None for latest calibration."""
        empty_db = FuelDatabase(str(tmp_path / "empty_cal.db"))
        try:
            assert empty_db.get_latest_calibration_run() is None
        finally:
            empty_db.close()

    def test_site_weights_upsert(self, db):
        """Saving weights for the same site should overwrite (upsert)."""
        run_id = db.save_calibration_run({
            "backtest_months": 6, "horizon": 2,
            "min_months": 24, "sites_calibrated": 1, "overall_mape": 5.0,
        })
        db.save_site_weights(run_id, [
            {"site_id": "100", "model_name": "ets", "weight": 0.60,
             "mape_pct": 5.0, "n_months": 6},
        ])
        # Overwrite with new weight
        db.save_site_weights(run_id, [
            {"site_id": "100", "model_name": "ets", "weight": 0.80,
             "mape_pct": 3.0, "n_months": 8},
        ])
        bulk = db.get_site_weights_bulk()
        assert bulk["100"]["ets"] == 0.80

    def test_get_site_weights_bulk_latest_run_only(self, db):
        """Weight lookup should only return rows from the latest calibration run."""
        run1 = db.save_calibration_run({
            "backtest_months": 6, "horizon": 2,
            "min_months": 24, "sites_calibrated": 1, "overall_mape": 6.0,
        })
        db.save_site_weights(run1, [
            {"site_id": "100", "model_name": "ets", "weight": 0.75,
             "mape_pct": 4.0, "n_months": 6},
            {"site_id": "100", "model_name": "snaive", "weight": 0.25,
             "mape_pct": 8.0, "n_months": 6},
        ])

        run2 = db.save_calibration_run({
            "backtest_months": 3, "horizon": 2,
            "min_months": 24, "sites_calibrated": 1, "overall_mape": 5.0,
        })
        db.save_site_weights(run2, [
            {"site_id": "200", "model_name": "ets", "weight": 0.60,
             "mape_pct": 5.0, "n_months": 3},
            {"site_id": "200", "model_name": "snaive", "weight": 0.40,
             "mape_pct": 6.0, "n_months": 3},
        ])

        bulk = db.get_site_weights_bulk()
        assert "200" in bulk
        assert "100" not in bulk

    def test_get_interval_factors_latest_run_only(self, db):
        """Interval lookup should only read factors from the latest run."""
        run1 = db.save_calibration_run({
            "backtest_months": 6, "horizon": 2,
            "min_months": 24, "sites_calibrated": 1, "overall_mape": 6.0,
        })
        db.save_interval_calibration(run1, [
            {"segment": "site:100", "residual_std": 0.08,
             "residual_p10": -0.12, "residual_p90": 0.11, "n_observations": 10},
        ])

        run2 = db.save_calibration_run({
            "backtest_months": 3, "horizon": 2,
            "min_months": 24, "sites_calibrated": 1, "overall_mape": 5.0,
        })
        db.save_interval_calibration(run2, [
            {"segment": "global", "residual_std": 0.10,
             "residual_p10": -0.15, "residual_p90": 0.13, "n_observations": 20},
        ])

        assert db.get_interval_factors("site:100") is None
        assert db.get_interval_factors("global") is not None
