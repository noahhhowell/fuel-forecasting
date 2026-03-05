"""
Tests for calibrate.py — weight computation, shrinkage, and interval calibration.
"""

import numpy as np
import pytest

import pandas as pd

from calibrate import (
    compute_optimal_weights,
    _blend_weights,
    _compute_ensemble_residuals,
    _compute_residual_stats,
    run_calibration,
    WEIGHT_FLOOR,
    DEFAULT_MAPE_FOR_MISSING,
    SHRINKAGE_MIN_OBSERVATIONS,
)


# ---------------------------------------------------------------------------
# compute_optimal_weights
# ---------------------------------------------------------------------------

class TestComputeOptimalWeights:

    def test_weights_sum_to_one(self):
        """Weights must sum to 1.0."""
        mapes = {"ets": 5.0, "snaive": 10.0}
        weights = compute_optimal_weights(mapes)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_lower_mape_gets_higher_weight(self):
        """Model with lower MAPE should get higher weight."""
        mapes = {"ets": 3.0, "snaive": 15.0}
        weights = compute_optimal_weights(mapes)
        assert weights["ets"] > weights["snaive"]

    def test_equal_mape_gives_equal_weights(self):
        """Equal MAPEs should produce equal weights."""
        mapes = {"ets": 10.0, "snaive": 10.0}
        weights = compute_optimal_weights(mapes)
        assert weights["ets"] == pytest.approx(weights["snaive"])
        assert weights["ets"] == pytest.approx(0.5)

    def test_weight_floor_enforced(self):
        """Even a terrible model should get at least WEIGHT_FLOOR weight."""
        mapes = {"ets": 1.0, "snaive": 100.0}
        weights = compute_optimal_weights(mapes, weight_floor=0.10)
        assert weights["snaive"] >= 0.10

    def test_zero_mape_clamped(self):
        """MAPE of 0 should not cause division by zero (clamped to 0.1)."""
        mapes = {"ets": 0.0, "snaive": 10.0}
        weights = compute_optimal_weights(mapes)
        assert all(np.isfinite(v) for v in weights.values())
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_single_model(self):
        """Single model should get weight 1.0."""
        mapes = {"ets": 5.0}
        weights = compute_optimal_weights(mapes)
        assert weights["ets"] == pytest.approx(1.0)

    def test_custom_floor(self):
        """Custom weight floor should be respected."""
        mapes = {"ets": 1.0, "snaive": 200.0}
        weights = compute_optimal_weights(mapes, weight_floor=0.20)
        assert weights["snaive"] >= 0.20


# ---------------------------------------------------------------------------
# Shrinkage blending
# ---------------------------------------------------------------------------

class TestBlendWeights:

    def test_sufficient_obs_returns_site_weights(self):
        """With enough observations, site weights are returned as-is."""
        site_w = {"ets": 0.8, "snaive": 0.2}
        fallback = {"ets": 0.5, "snaive": 0.5}
        result = _blend_weights(site_w, fallback, SHRINKAGE_MIN_OBSERVATIONS)
        assert result["ets"] == pytest.approx(0.8)
        assert result["snaive"] == pytest.approx(0.2)

    def test_zero_obs_returns_fallback(self):
        """With 0 observations, should return pure fallback."""
        site_w = {"ets": 0.9, "snaive": 0.1}
        fallback = {"ets": 0.5, "snaive": 0.5}
        result = _blend_weights(site_w, fallback, 0)
        assert result["ets"] == pytest.approx(0.5)
        assert result["snaive"] == pytest.approx(0.5)

    def test_partial_obs_blends(self):
        """With partial observations, should blend between site and fallback."""
        site_w = {"ets": 1.0, "snaive": 0.0}
        fallback = {"ets": 0.5, "snaive": 0.5}
        # 3 out of 6 min observations → alpha = 0.5
        result = _blend_weights(site_w, fallback, 3, min_obs=6)
        # After blend + renormalize
        assert result["ets"] > 0.5
        assert result["snaive"] > 0.0
        assert sum(result.values()) == pytest.approx(1.0)

    def test_blend_sums_to_one(self):
        """Blended weights should always sum to 1.0."""
        site_w = {"ets": 0.7, "snaive": 0.3}
        fallback = {"ets": 0.6, "snaive": 0.4}
        result = _blend_weights(site_w, fallback, 2, min_obs=6)
        assert sum(result.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Residual stats
# ---------------------------------------------------------------------------

class TestResidualStats:

    def test_normal_residuals(self):
        """Should compute std, p10, p90 for a set of residuals."""
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 0.1, size=100).tolist()
        stats = _compute_residual_stats(residuals)
        assert stats is not None
        assert stats["residual_std"] > 0
        assert stats["residual_p10"] < 0
        assert stats["residual_p90"] > 0
        assert stats["n_observations"] == 100

    def test_too_few_returns_none(self):
        """With fewer than 2 residuals, should return None."""
        assert _compute_residual_stats([0.05]) is None
        assert _compute_residual_stats([]) is None

    def test_non_finite_filtered(self):
        """Non-finite values should be filtered out."""
        residuals = [0.1, -0.1, float("inf"), float("nan"), 0.05]
        stats = _compute_residual_stats(residuals)
        assert stats is not None
        assert stats["n_observations"] == 3


# ---------------------------------------------------------------------------
# Ensemble residuals
# ---------------------------------------------------------------------------

class TestComputeEnsembleResiduals:

    def _make_per_model_df(self):
        """Two models, two months, one site. Actual=100 each month."""
        return pd.DataFrame([
            {"site_id": "S1", "month": "2025-01", "model": "ets",
             "forecast": 110.0, "actual": 100.0, "error_pct": 10.0, "residual": 0.10},
            {"site_id": "S1", "month": "2025-01", "model": "snaive",
             "forecast": 90.0, "actual": 100.0, "error_pct": 10.0, "residual": -0.10},
            {"site_id": "S1", "month": "2025-02", "model": "ets",
             "forecast": 105.0, "actual": 100.0, "error_pct": 5.0, "residual": 0.05},
            {"site_id": "S1", "month": "2025-02", "model": "snaive",
             "forecast": 95.0, "actual": 100.0, "error_pct": 5.0, "residual": -0.05},
        ])

    def test_equal_weights_gives_zero_residual(self):
        """Equal weights on symmetric forecasts should produce ~0 residual."""
        df = self._make_per_model_df()
        weights = {("S1", "ets"): 0.5, ("S1", "snaive"): 0.5}
        result = _compute_ensemble_residuals(df, weights)
        assert len(result) == 2
        # (0.5*110 + 0.5*90) = 100, residual = 0
        assert result["residual"].iloc[0] == pytest.approx(0.0)
        assert result["residual"].iloc[1] == pytest.approx(0.0)

    def test_high_weight_on_overforecast_gives_positive_residual(self):
        """Weighting the over-forecasting model more should give positive residual."""
        df = self._make_per_model_df()
        weights = {("S1", "ets"): 0.8, ("S1", "snaive"): 0.2}
        result = _compute_ensemble_residuals(df, weights)
        # Month 1: (0.8*110 + 0.2*90)/1.0 = 106, residual = 0.06
        assert result["residual"].iloc[0] == pytest.approx(0.06)

    def test_missing_weight_excluded(self):
        """Model with no weight entry (weight=0) should not contribute."""
        df = self._make_per_model_df()
        # Only ets has a weight; snaive missing from lookup → weight=0
        weights = {("S1", "ets"): 1.0}
        result = _compute_ensemble_residuals(df, weights)
        # ensemble = ets only → 110/100 - 1 = 0.10
        assert result["residual"].iloc[0] == pytest.approx(0.10)

    def test_returns_one_row_per_site_month(self):
        """Should collapse per-model rows into one ensemble row per site-month."""
        df = self._make_per_model_df()
        weights = {("S1", "ets"): 0.6, ("S1", "snaive"): 0.4}
        result = _compute_ensemble_residuals(df, weights)
        assert len(result) == 2
        assert list(result.columns) == ["site_id", "month", "residual"]

    def test_empty_input_preserves_expected_columns(self):
        """Empty inputs should return an empty frame with the stable schema."""
        result = _compute_ensemble_residuals(pd.DataFrame(), {})
        assert result.empty
        assert list(result.columns) == ["site_id", "month", "residual"]


# ---------------------------------------------------------------------------
# run_calibration validation
# ---------------------------------------------------------------------------

class TestRunCalibrationValidation:

    def test_invalid_months_raises(self, tmp_path):
        with pytest.raises(ValueError, match="months must be >= 1"):
            run_calibration(db_path=str(tmp_path / "cal.db"), months=0)

    def test_invalid_min_months_raises(self, tmp_path):
        with pytest.raises(ValueError, match="min_months must be >= 1"):
            run_calibration(db_path=str(tmp_path / "cal.db"), min_months=0)

    def test_invalid_horizon_raises(self, tmp_path):
        with pytest.raises(ValueError, match="horizon must be >= 0"):
            run_calibration(db_path=str(tmp_path / "cal.db"), horizon=-1)

    def test_negative_weight_floor_raises(self, tmp_path):
        with pytest.raises(ValueError, match="weight_floor must be >= 0"):
            run_calibration(db_path=str(tmp_path / "cal.db"), weight_floor=-0.01)

    def test_non_finite_weight_floor_raises(self, tmp_path):
        with pytest.raises(ValueError, match="weight_floor must be a finite number"):
            run_calibration(db_path=str(tmp_path / "cal.db"), weight_floor=float("nan"))
