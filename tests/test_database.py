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
