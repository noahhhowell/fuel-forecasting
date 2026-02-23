"""Tests for backtest.py — metric helpers and run_backtest integration."""

import numpy as np
import pandas as pd
import pytest

from backtest import (
    build_site_error_metrics,
    build_per_model_site_metrics,
    run_backtest,
    _run_backtest_inner,
    _parse_max_date,
    MAPE_GOOD_THRESHOLD,
    MAPE_ACCEPTABLE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# build_site_error_metrics
# ---------------------------------------------------------------------------

class TestBuildSiteErrorMetrics:

    def test_has_expected_columns(self):
        """Per-site metrics should include robust error columns."""
        results = pd.DataFrame(
            [
                {"site_id": "100", "month": "2025-01", "error_pct": 5.0},
                {"site_id": "100", "month": "2025-02", "error_pct": 6.0},
                {"site_id": "200", "month": "2025-01", "error_pct": 12.0},
                {"site_id": "200", "month": "2025-02", "error_pct": 8.0},
            ]
        )
        site_metrics = build_site_error_metrics(results)

        expected_cols = {
            "site_id", "mape_pct", "median_ape_pct",
            "trimmed_mape_pct", "max_ape_pct", "n_months", "rating",
        }
        assert expected_cols.issubset(site_metrics.columns)

    def test_trimmed_mape_drops_single_worst_month(self):
        """Trimmed MAPE should reduce sensitivity to one catastrophic month."""
        results = pd.DataFrame(
            [
                {"site_id": "100", "month": "2025-01", "error_pct": 5.0},
                {"site_id": "100", "month": "2025-02", "error_pct": 5.0},
                {"site_id": "100", "month": "2025-03", "error_pct": 100.0},
                {"site_id": "200", "month": "2025-01", "error_pct": 10.0},
                {"site_id": "200", "month": "2025-02", "error_pct": 20.0},
            ]
        )
        site_metrics = build_site_error_metrics(results).set_index("site_id")

        assert site_metrics.loc["100", "mape_pct"] == pytest.approx(
            (5.0 + 5.0 + 100.0) / 3.0
        )
        assert site_metrics.loc["100", "median_ape_pct"] == pytest.approx(5.0)
        assert site_metrics.loc["100", "trimmed_mape_pct"] == pytest.approx(5.0)
        assert site_metrics.loc["200", "mape_pct"] == pytest.approx(15.0)
        assert site_metrics.loc["200", "trimmed_mape_pct"] == pytest.approx(10.0)

    def test_ignores_non_finite_values(self):
        """Non-finite error rows should be excluded before aggregation."""
        results = pd.DataFrame(
            [
                {"site_id": "100", "month": "2025-01", "error_pct": 8.0},
                {"site_id": "100", "month": "2025-02", "error_pct": float("inf")},
                {"site_id": "100", "month": "2025-03", "error_pct": float("nan")},
                {"site_id": "200", "month": "2025-01", "error_pct": "bad"},
            ]
        )
        site_metrics = build_site_error_metrics(results)

        assert len(site_metrics) == 1
        assert site_metrics.iloc[0]["site_id"] == "100"
        assert site_metrics.iloc[0]["mape_pct"] == pytest.approx(8.0)

    def test_empty_input_returns_empty_with_schema(self):
        """Empty input should return an empty DataFrame with the correct columns."""
        site_metrics = build_site_error_metrics(pd.DataFrame())
        assert site_metrics.empty
        assert "rating" in site_metrics.columns

    def test_rating_uses_threshold_constants(self):
        """Rating column should respect the MAPE threshold constants."""
        results = pd.DataFrame(
            [
                {"site_id": "A", "month": "2025-01", "error_pct": MAPE_GOOD_THRESHOLD - 1},
                {"site_id": "B", "month": "2025-01", "error_pct": MAPE_GOOD_THRESHOLD + 1},
                {"site_id": "C", "month": "2025-01", "error_pct": MAPE_ACCEPTABLE_THRESHOLD + 1},
            ]
        )
        metrics = build_site_error_metrics(results).set_index("site_id")
        assert str(metrics.loc["A", "rating"]) == "Good"
        assert str(metrics.loc["B", "rating"]) == "Acceptable"
        assert str(metrics.loc["C", "rating"]) == "Review"


# ---------------------------------------------------------------------------
# _parse_max_date
# ---------------------------------------------------------------------------

class TestParseMaxDate:

    def test_valid_date_range(self):
        result = _parse_max_date({"date_range": "2022-01-01 to 2024-12-31"})
        assert result == pd.Timestamp("2024-12-31")

    def test_missing_key_raises(self):
        with pytest.raises(RuntimeError, match="Unexpected date_range format"):
            _parse_max_date({})

    def test_bad_format_raises(self):
        with pytest.raises(RuntimeError, match="Unexpected date_range format"):
            _parse_max_date({"date_range": "2024-12-31"})


# ---------------------------------------------------------------------------
# run_backtest integration
# ---------------------------------------------------------------------------

class TestRunBacktest:

    def test_returns_results_and_site_mape(self, db):
        """run_backtest with the test DB should produce valid results."""
        results_df, site_mape = run_backtest(
            db_path=db.db_path, months=2, min_months=12, horizon=1,
        )

        assert results_df is not None
        assert not results_df.empty
        assert {"site_id", "month", "forecast", "actual", "error_pct"}.issubset(
            results_df.columns
        )

        assert site_mape is not None
        assert not site_mape.empty
        assert "rating" in site_mape.columns

    def test_all_error_pcts_non_negative(self, db):
        """Absolute percentage errors must be >= 0."""
        results_df, _ = run_backtest(
            db_path=db.db_path, months=2, min_months=12, horizon=1,
        )
        if results_df is not None:
            assert (results_df["error_pct"] >= 0).all()

    def test_excel_output(self, db, tmp_path):
        """Excel export should create a file with expected sheets."""
        out = str(tmp_path / "bt.xlsx")
        run_backtest(
            db_path=db.db_path, months=2, min_months=12, horizon=1, output=out,
        )
        xl = pd.ExcelFile(out)
        assert "Results" in xl.sheet_names
        assert "Summary" in xl.sheet_names
        assert "Site MAPE" in xl.sheet_names

    def test_invalid_months_raises(self):
        with pytest.raises(ValueError, match="months must be >= 1"):
            run_backtest(months=0)

    def test_invalid_min_months_raises(self):
        with pytest.raises(ValueError, match="min_months must be >= 1"):
            run_backtest(min_months=0)

    def test_inner_invalid_months_raises(self, db):
        with pytest.raises(ValueError, match="months must be >= 1"):
            _run_backtest_inner(
                db, months=0, output=None, min_months=12, horizon=1,
            )

    def test_db_closed_on_error(self, tmp_path):
        """Database should be closed even if an error occurs during backtest."""
        bad_db = str(tmp_path / "empty.db")
        # Create an empty database
        from database import FuelDatabase
        empty = FuelDatabase(bad_db)
        empty.close()

        with pytest.raises(RuntimeError, match="empty"):
            run_backtest(db_path=bad_db)

    def test_return_per_model(self, db):
        """return_per_model=True should return a 3-tuple with per-model data."""
        result = run_backtest(
            db_path=db.db_path, months=2, min_months=12, horizon=1,
            return_per_model=True,
        )
        assert len(result) == 3
        results_df, site_mape, per_model_df = result
        assert results_df is not None
        assert per_model_df is not None
        assert not per_model_df.empty
        assert {"site_id", "month", "model", "forecast", "actual",
                "error_pct", "residual"}.issubset(per_model_df.columns)

    def test_return_per_model_false_returns_pair(self, db):
        """return_per_model=False (default) should return a 2-tuple."""
        result = run_backtest(
            db_path=db.db_path, months=2, min_months=12, horizon=1,
            return_per_model=False,
        )
        assert len(result) == 2


# ---------------------------------------------------------------------------
# build_per_model_site_metrics
# ---------------------------------------------------------------------------

class TestBuildPerModelSiteMetrics:

    def test_groups_by_site_and_model(self):
        """Should produce one row per (site_id, model) pair."""
        df = pd.DataFrame([
            {"site_id": "100", "model": "ets", "error_pct": 5.0},
            {"site_id": "100", "model": "ets", "error_pct": 7.0},
            {"site_id": "100", "model": "snaive", "error_pct": 10.0},
            {"site_id": "200", "model": "ets", "error_pct": 3.0},
        ])
        metrics = build_per_model_site_metrics(df)
        assert len(metrics) == 3
        assert {"site_id", "model", "mape_pct", "n_months"}.issubset(metrics.columns)

        ets_100 = metrics[(metrics["site_id"] == "100") & (metrics["model"] == "ets")]
        assert ets_100["mape_pct"].iloc[0] == pytest.approx(6.0)
        assert ets_100["n_months"].iloc[0] == 2

    def test_empty_input(self):
        """Empty input should return empty DataFrame with correct schema."""
        metrics = build_per_model_site_metrics(pd.DataFrame())
        assert metrics.empty
        assert "mape_pct" in metrics.columns
