"""Tests for model comparison and selection utilities.

These tests verify the functionality of the ModelComparison class and
related helper functions using both simple mock-like models and real
sklearn models.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from snowforecast.models.comparison import (
    ComparisonResults,
    ModelComparison,
    compare_models,
    create_comparison_report,
    rank_models,
)


class SimpleModel:
    """Simple model for testing that returns constant or scaled predictions."""

    def __init__(self, predictions: np.ndarray | None = None, scale: float = 1.0):
        """Initialize with fixed predictions or scale factor.

        Args:
            predictions: Fixed predictions to return. If None, returns input mean * scale.
            scale: Scale factor to apply to input mean if predictions is None.
        """
        self.predictions = predictions
        self.scale = scale
        self._last_X = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions."""
        self._last_X = X
        if self.predictions is not None:
            return self.predictions
        # Return scaled version of input (simple linear transform)
        return np.full(len(X), X.values.mean() * self.scale)


class BiasedModel:
    """Model that adds a constant bias to true values."""

    def __init__(self, bias: float, y_true: np.ndarray):
        """Initialize with bias and true values to base predictions on.

        Args:
            bias: Constant to add to predictions.
            y_true: True values to base predictions on.
        """
        self.bias = bias
        self.y_true = y_true

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return biased predictions."""
        return self.y_true + self.bias


class NoisyModel:
    """Model that adds random noise to true values."""

    def __init__(self, noise_std: float, y_true: np.ndarray, seed: int = 42):
        """Initialize with noise level and true values.

        Args:
            noise_std: Standard deviation of noise.
            y_true: True values to add noise to.
            seed: Random seed for reproducibility.
        """
        self.noise_std = noise_std
        self.y_true = y_true
        self.seed = seed

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return noisy predictions."""
        rng = np.random.RandomState(self.seed)
        noise = rng.normal(0, self.noise_std, len(self.y_true))
        return self.y_true + noise


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_data():
    """Create sample test data."""
    np.random.seed(42)
    n_samples = 100

    X = pd.DataFrame({
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "feature3": np.random.randn(n_samples),
    })

    # Create y with some relationship to features
    y = pd.Series(
        3.0 * X["feature1"] + 2.0 * X["feature2"] + np.random.randn(n_samples) * 2.0,
        name="target"
    )

    return X, y


@pytest.fixture
def sample_data_with_groups():
    """Create sample test data with grouping columns."""
    np.random.seed(42)
    n_samples = 120

    X = pd.DataFrame({
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "elevation_band": np.repeat(["low", "mid", "high"], n_samples // 3),
        "region": np.tile(["north", "south"], n_samples // 2),
    })

    y = pd.Series(
        3.0 * X["feature1"] + 2.0 * X["feature2"] + np.random.randn(n_samples) * 2.0,
        name="target"
    )

    return X, y


@pytest.fixture
def simple_models(sample_data):
    """Create simple test models with known behaviors."""
    X, y = sample_data
    y_array = y.values

    return {
        "perfect": SimpleModel(predictions=y_array),  # Perfect predictions
        "biased_up": BiasedModel(bias=5.0, y_true=y_array),  # Over-predicts by 5
        "biased_down": BiasedModel(bias=-3.0, y_true=y_array),  # Under-predicts by 3
        "noisy": NoisyModel(noise_std=10.0, y_true=y_array),  # High noise
    }


@pytest.fixture
def sklearn_models(sample_data):
    """Create fitted sklearn models."""
    X, y = sample_data

    linear = LinearRegression()
    linear.fit(X, y)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)

    ridge_strong = Ridge(alpha=100.0)
    ridge_strong.fit(X, y)

    return {
        "linear_regression": linear,
        "ridge": ridge,
        "ridge_strong_reg": ridge_strong,
    }


# =============================================================================
# ComparisonResults Tests
# =============================================================================


class TestComparisonResults:
    """Tests for the ComparisonResults dataclass."""

    def test_empty_results(self):
        """Test behavior with empty results."""
        results = ComparisonResults()
        assert results.to_dataframe().empty
        assert results.summary() == "No comparison results available."
        assert results.get_best_model_name("rmse") is None

    def test_add_model_metrics(self):
        """Test adding model metrics."""
        results = ComparisonResults()
        metrics = {"rmse": 10.0, "mae": 8.0, "f1": 0.8}

        results.add_model_metrics("model_a", metrics)

        assert "model_a" in results.model_metrics
        assert results.model_metrics["model_a"]["rmse"] == 10.0

    def test_add_model_metrics_with_predictions(self):
        """Test adding metrics with predictions."""
        results = ComparisonResults()
        predictions = np.array([1.0, 2.0, 3.0])

        results.add_model_metrics(
            "model_a",
            {"rmse": 5.0},
            predictions=predictions,
        )

        assert "model_a" in results.predictions
        np.testing.assert_array_equal(results.predictions["model_a"], predictions)

    def test_add_stratified_metrics(self):
        """Test adding stratified metrics."""
        results = ComparisonResults()
        strat_df = pd.DataFrame({
            "group": ["A", "B"],
            "rmse": [10.0, 12.0],
        })

        results.add_stratified_metrics("model_a", strat_df)

        assert "model_a" in results.stratified_metrics
        pd.testing.assert_frame_equal(
            results.stratified_metrics["model_a"], strat_df
        )

    def test_to_dataframe(self):
        """Test converting results to DataFrame."""
        results = ComparisonResults()
        results.add_model_metrics("model_a", {"rmse": 10.0, "mae": 8.0})
        results.add_model_metrics("model_b", {"rmse": 12.0, "mae": 9.0})

        df = results.to_dataframe()

        assert len(df) == 2
        assert "model" in df.columns
        assert "rmse" in df.columns
        assert "mae" in df.columns
        assert df[df["model"] == "model_a"]["rmse"].iloc[0] == 10.0

    def test_get_best_model_name_lower_is_better(self):
        """Test getting best model for metrics where lower is better."""
        results = ComparisonResults()
        results.add_model_metrics("model_a", {"rmse": 10.0, "mae": 8.0})
        results.add_model_metrics("model_b", {"rmse": 5.0, "mae": 4.0})
        results.add_model_metrics("model_c", {"rmse": 15.0, "mae": 12.0})

        assert results.get_best_model_name("rmse") == "model_b"
        assert results.get_best_model_name("mae") == "model_b"

    def test_get_best_model_name_higher_is_better(self):
        """Test getting best model for metrics where higher is better."""
        results = ComparisonResults()
        results.add_model_metrics("model_a", {"f1": 0.7, "precision": 0.8})
        results.add_model_metrics("model_b", {"f1": 0.9, "precision": 0.85})
        results.add_model_metrics("model_c", {"f1": 0.5, "precision": 0.6})

        assert results.get_best_model_name("f1") == "model_b"
        assert results.get_best_model_name("precision") == "model_b"

    def test_summary_format(self):
        """Test that summary produces formatted output."""
        results = ComparisonResults()
        results.add_model_metrics("model_a", {"rmse": 10.0, "f1": 0.8})
        results.add_model_metrics("model_b", {"rmse": 12.0, "f1": 0.9})

        summary = results.summary()

        assert "MODEL COMPARISON SUMMARY" in summary
        assert "model_a" in summary
        assert "model_b" in summary
        assert "Best Model by Metric" in summary

    def test_repr(self):
        """Test string representation."""
        results = ComparisonResults()
        assert "empty" in repr(results)

        results.add_model_metrics("model_a", {"rmse": 10.0})
        assert "model_a" in repr(results)


# =============================================================================
# ModelComparison Tests
# =============================================================================


class TestModelComparison:
    """Tests for the ModelComparison class."""

    def test_init_empty_models_raises(self):
        """Test that empty models dict raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ModelComparison({})

    def test_evaluate_all_simple_models(self, sample_data, simple_models):
        """Test evaluate_all with simple mock-like models."""
        X, y = sample_data

        comparison = ModelComparison(simple_models)
        results_df = comparison.evaluate_all(X, y)

        assert len(results_df) == len(simple_models)
        assert "model" in results_df.columns
        assert "rmse" in results_df.columns
        assert "mae" in results_df.columns
        assert "f1" in results_df.columns

        # Perfect model should have RMSE close to 0
        perfect_rmse = results_df[results_df["model"] == "perfect"]["rmse"].iloc[0]
        assert perfect_rmse < 1e-10

    def test_evaluate_all_with_sklearn_models(self, sample_data, sklearn_models):
        """Test evaluate_all with real sklearn models."""
        X, y = sample_data

        comparison = ModelComparison(sklearn_models)
        results_df = comparison.evaluate_all(X, y)

        assert len(results_df) == len(sklearn_models)

        # Linear regression should perform better than heavily regularized Ridge
        linear_rmse = results_df[
            results_df["model"] == "linear_regression"
        ]["rmse"].iloc[0]
        ridge_strong_rmse = results_df[
            results_df["model"] == "ridge_strong_reg"
        ]["rmse"].iloc[0]

        # With default data, linear should be better
        assert linear_rmse < ridge_strong_rmse

    def test_evaluate_all_empty_data_raises(self, simple_models):
        """Test that empty data raises ValueError."""
        X_empty = pd.DataFrame()
        y_empty = pd.Series([], dtype=float)

        comparison = ModelComparison(simple_models)

        with pytest.raises(ValueError, match="cannot be empty"):
            comparison.evaluate_all(X_empty, y_empty)

    def test_get_best_model_before_evaluate_raises(self, simple_models):
        """Test that get_best_model before evaluate raises."""
        comparison = ModelComparison(simple_models)

        with pytest.raises(ValueError, match="Must call evaluate_all"):
            comparison.get_best_model()

    def test_get_best_model(self, sample_data, simple_models):
        """Test getting the best model."""
        X, y = sample_data

        comparison = ModelComparison(simple_models)
        comparison.evaluate_all(X, y)

        # Best by RMSE should be perfect model
        name, model = comparison.get_best_model(metric="rmse")
        assert name == "perfect"
        assert model is simple_models["perfect"]

    def test_get_best_model_invalid_metric(self, sample_data, simple_models):
        """Test that invalid metric raises ValueError."""
        X, y = sample_data

        comparison = ModelComparison(simple_models)
        comparison.evaluate_all(X, y)

        with pytest.raises(ValueError, match="Unknown metric"):
            comparison.get_best_model(metric="invalid_metric")

    def test_compare_by_group(self, sample_data_with_groups):
        """Test stratified metrics by group."""
        X, y = sample_data_with_groups

        # Create models
        perfect_model = SimpleModel(predictions=y.values)
        noisy_model = NoisyModel(noise_std=5.0, y_true=y.values)

        models = {"perfect": perfect_model, "noisy": noisy_model}
        comparison = ModelComparison(models)

        # First evaluate
        comparison.evaluate_all(X[["feature1", "feature2"]], y)

        # Create DataFrame with predictions
        df = X.copy()
        df["y_true"] = y.values
        df["perfect_pred"] = perfect_model.predict(X)
        df["noisy_pred"] = noisy_model.predict(X)

        # Compare by elevation band
        stratified = comparison.compare_by_group(
            df, group_col="elevation_band", y_true_col="y_true"
        )

        assert "perfect" in stratified
        assert "noisy" in stratified
        assert "elevation_band" in stratified["perfect"].columns

    def test_compare_by_group_missing_column(self, sample_data_with_groups):
        """Test that missing group column raises ValueError."""
        X, y = sample_data_with_groups

        models = {"model": SimpleModel(predictions=y.values)}
        comparison = ModelComparison(models)
        comparison.evaluate_all(X[["feature1", "feature2"]], y)

        df = X.copy()
        df["y_true"] = y.values

        with pytest.raises(ValueError, match="not found"):
            comparison.compare_by_group(df, group_col="nonexistent", y_true_col="y_true")

    def test_statistical_significance_test(self, sample_data):
        """Test paired t-test for model comparison."""
        X, y = sample_data
        y_array = y.values

        # Create two models with different noise levels
        model1 = NoisyModel(noise_std=2.0, y_true=y_array, seed=42)
        model2 = NoisyModel(noise_std=10.0, y_true=y_array, seed=43)

        models = {"low_noise": model1, "high_noise": model2}
        comparison = ModelComparison(models)
        comparison.evaluate_all(X, y)

        result = comparison.statistical_significance_test(
            "low_noise", "high_noise"
        )

        assert "t_statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "model1_mean_se" in result
        assert "model2_mean_se" in result
        assert "better_model" in result

        # Low noise model should have lower mean squared error
        assert result["model1_mean_se"] < result["model2_mean_se"]

    def test_statistical_significance_identical_models(self, sample_data):
        """Test t-test with identical models (not significant)."""
        X, y = sample_data
        y_array = y.values

        # Same predictions
        predictions = y_array + 1.0  # Small constant error

        model1 = SimpleModel(predictions=predictions)
        model2 = SimpleModel(predictions=predictions)

        models = {"model1": model1, "model2": model2}
        comparison = ModelComparison(models)
        comparison.evaluate_all(X, y)

        result = comparison.statistical_significance_test("model1", "model2")

        # Should not be significant (identical predictions)
        assert not result["significant"]
        assert result["better_model"] is None

    def test_statistical_significance_with_explicit_predictions(self, sample_data):
        """Test t-test with explicitly provided predictions."""
        X, y = sample_data

        models = {"model1": SimpleModel(), "model2": SimpleModel()}
        comparison = ModelComparison(models)

        y_true = y.values
        y_pred1 = y_true + np.random.randn(len(y_true)) * 2  # Low noise
        y_pred2 = y_true + np.random.randn(len(y_true)) * 20  # High noise

        result = comparison.statistical_significance_test(
            "model1", "model2",
            y_true=y_true,
            y_pred1=y_pred1,
            y_pred2=y_pred2,
        )

        assert "t_statistic" in result
        assert "p_value" in result

    def test_get_results(self, sample_data, simple_models):
        """Test getting ComparisonResults object."""
        X, y = sample_data

        comparison = ModelComparison(simple_models)
        comparison.evaluate_all(X, y)

        results = comparison.get_results()

        assert isinstance(results, ComparisonResults)
        assert len(results.model_metrics) == len(simple_models)


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestCompareModelsFunction:
    """Tests for the compare_models convenience function."""

    def test_compare_models(self, sample_data, sklearn_models):
        """Test the compare_models convenience function."""
        X, y = sample_data

        results = compare_models(sklearn_models, X, y)

        assert isinstance(results, ComparisonResults)
        assert len(results.model_metrics) == len(sklearn_models)

    def test_compare_models_with_threshold(self, sample_data):
        """Test compare_models with custom threshold."""
        X, y = sample_data

        model = SimpleModel(predictions=y.values)
        results = compare_models(
            {"model": model}, X, y, snowfall_threshold_cm=5.0
        )

        assert isinstance(results, ComparisonResults)


class TestRankModels:
    """Tests for the rank_models function."""

    def test_rank_models_rmse(self):
        """Test ranking models by RMSE (lower is better)."""
        results = ComparisonResults()
        results.add_model_metrics("model_a", {"rmse": 15.0})
        results.add_model_metrics("model_b", {"rmse": 10.0})
        results.add_model_metrics("model_c", {"rmse": 20.0})

        rankings = rank_models(results, metric="rmse")

        assert len(rankings) == 3
        assert rankings[0][0] == "model_b"  # Best (lowest)
        assert rankings[1][0] == "model_a"
        assert rankings[2][0] == "model_c"  # Worst (highest)

    def test_rank_models_f1(self):
        """Test ranking models by F1 (higher is better)."""
        results = ComparisonResults()
        results.add_model_metrics("model_a", {"f1": 0.7})
        results.add_model_metrics("model_b", {"f1": 0.9})
        results.add_model_metrics("model_c", {"f1": 0.5})

        rankings = rank_models(results, metric="f1")

        assert len(rankings) == 3
        assert rankings[0][0] == "model_b"  # Best (highest)
        assert rankings[1][0] == "model_a"
        assert rankings[2][0] == "model_c"  # Worst (lowest)

    def test_rank_models_empty(self):
        """Test ranking with empty results."""
        results = ComparisonResults()
        rankings = rank_models(results, metric="rmse")
        assert rankings == []

    def test_rank_models_with_nan(self):
        """Test ranking excludes models with NaN values."""
        results = ComparisonResults()
        results.add_model_metrics("model_a", {"rmse": 10.0})
        results.add_model_metrics("model_b", {"rmse": np.nan})
        results.add_model_metrics("model_c", {"rmse": 15.0})

        rankings = rank_models(results, metric="rmse")

        assert len(rankings) == 2
        model_names = [r[0] for r in rankings]
        assert "model_b" not in model_names


class TestCreateComparisonReport:
    """Tests for the create_comparison_report function."""

    def test_create_report(self):
        """Test report generation."""
        results = ComparisonResults()
        results.add_model_metrics("model_a", {
            "rmse": 10.0, "mae": 8.0, "bias": 0.5, "f1": 0.8
        })
        results.add_model_metrics("model_b", {
            "rmse": 12.0, "mae": 9.0, "bias": -0.3, "f1": 0.85
        })

        report = create_comparison_report(results)

        assert "SNOWFALL PREDICTION MODEL COMPARISON REPORT" in report
        assert "OVERALL METRICS" in report
        assert "MODEL RANKINGS" in report
        assert "BEST MODELS" in report
        assert "model_a" in report
        assert "model_b" in report

    def test_create_report_with_stratified(self):
        """Test report includes stratified metrics."""
        results = ComparisonResults()
        results.add_model_metrics("model_a", {"rmse": 10.0})

        strat_df = pd.DataFrame({
            "group": ["A", "B"],
            "rmse": [8.0, 12.0],
        })
        results.add_stratified_metrics("model_a", strat_df)

        report = create_comparison_report(results, include_stratified=True)

        assert "STRATIFIED METRICS" in report

    def test_create_report_empty(self):
        """Test report with empty results."""
        results = ComparisonResults()
        report = create_comparison_report(results)

        assert "No model comparison results" in report


# =============================================================================
# Integration Tests with Real sklearn Models
# =============================================================================


class TestSklearnIntegration:
    """Integration tests using real sklearn models."""

    def test_linear_vs_ridge_comparison(self):
        """Test comparing LinearRegression vs Ridge regression."""
        np.random.seed(42)
        n_samples = 200

        # Create data with some collinearity
        X = pd.DataFrame({
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
        })
        X["feature3"] = X["feature1"] * 0.9 + np.random.randn(n_samples) * 0.1

        y = pd.Series(
            2.0 * X["feature1"] + 1.5 * X["feature2"] +
            np.random.randn(n_samples) * 0.5
        )

        # Fit models
        linear = LinearRegression()
        linear.fit(X, y)

        ridge = Ridge(alpha=1.0)
        ridge.fit(X, y)

        # Compare
        comparison = ModelComparison({
            "linear": linear,
            "ridge": ridge,
        })
        results_df = comparison.evaluate_all(X, y)

        # Both should perform reasonably well on training data
        assert results_df["rmse"].max() < 2.0  # Reasonable for noise level

        # Get best model
        best_name, best_model = comparison.get_best_model(metric="rmse")
        assert best_name in ["linear", "ridge"]

    def test_significance_linear_vs_biased(self):
        """Test significance between unbiased and biased models."""
        np.random.seed(42)
        n_samples = 100

        X = pd.DataFrame({"feature": np.random.randn(n_samples)})
        y = pd.Series(2.0 * X["feature"] + np.random.randn(n_samples) * 0.5)

        # Fit linear regression (unbiased)
        linear = LinearRegression()
        linear.fit(X, y)

        # Create biased predictions
        y_array = y.values
        biased_model = BiasedModel(bias=5.0, y_true=y_array)

        models = {"linear": linear, "biased": biased_model}
        comparison = ModelComparison(models)
        comparison.evaluate_all(X, y)

        # Test significance
        result = comparison.statistical_significance_test("linear", "biased")

        # Should be significant - biased model has systematic error
        assert result["significant"]
        assert result["better_model"] == "linear"

    def test_full_workflow(self):
        """Test complete comparison workflow."""
        np.random.seed(42)
        n_samples = 150

        # Create data
        X = pd.DataFrame({
            "temp": np.random.randn(n_samples),
            "precip": np.random.exponential(5, n_samples),
            "elevation": np.random.uniform(1000, 3000, n_samples),
        })

        # Snowfall depends on temp and precip
        y = pd.Series(
            np.maximum(0, -2.0 * X["temp"] + 0.5 * X["precip"] + 5.0) +
            np.random.randn(n_samples) * 2.0
        )

        # Train multiple models
        linear = LinearRegression()
        linear.fit(X, y)

        ridge_light = Ridge(alpha=0.1)
        ridge_light.fit(X, y)

        ridge_heavy = Ridge(alpha=10.0)
        ridge_heavy.fit(X, y)

        models = {
            "linear": linear,
            "ridge_light": ridge_light,
            "ridge_heavy": ridge_heavy,
        }

        # Run comparison
        comparison = ModelComparison(models)
        results_df = comparison.evaluate_all(X, y)

        # Check results
        assert len(results_df) == 3
        assert all(col in results_df.columns for col in ["rmse", "mae", "f1"])

        # Get results object
        results = comparison.get_results()

        # Generate report
        report = create_comparison_report(results)
        assert "linear" in report
        assert "ridge_light" in report

        # Rank models
        rankings = rank_models(results, metric="rmse")
        assert len(rankings) == 3

        # Get best
        best_name, _ = comparison.get_best_model(metric="rmse")
        assert best_name == rankings[0][0]


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_model_with_nan_predictions(self, sample_data):
        """Test handling of models that return NaN predictions."""
        X, y = sample_data

        nan_predictions = np.full(len(y), np.nan)
        nan_model = SimpleModel(predictions=nan_predictions)

        models = {"nan_model": nan_model}
        comparison = ModelComparison(models)
        results_df = comparison.evaluate_all(X, y)

        # Should have NaN metrics
        assert np.isnan(results_df["rmse"].iloc[0])

    def test_model_with_partial_nan_predictions(self, sample_data):
        """Test handling of partial NaN predictions."""
        X, y = sample_data
        y_array = y.values

        # Some NaN predictions
        predictions = y_array.copy()
        predictions[:10] = np.nan

        partial_nan_model = SimpleModel(predictions=predictions)

        models = {"partial_nan": partial_nan_model}
        comparison = ModelComparison(models)
        results_df = comparison.evaluate_all(X, y)

        # Should still compute metrics (excluding NaN samples)
        assert not np.isnan(results_df["rmse"].iloc[0])

    def test_single_sample(self):
        """Test with single sample (edge case)."""
        X = pd.DataFrame({"feature": [1.0]})
        y = pd.Series([5.0])

        model = SimpleModel(predictions=np.array([5.0]))

        comparison = ModelComparison({"model": model})
        results_df = comparison.evaluate_all(X, y)

        # Should work with single sample
        assert results_df["rmse"].iloc[0] == 0.0

    def test_all_zeros(self):
        """Test with all zero values."""
        n = 50
        X = pd.DataFrame({"feature": np.zeros(n)})
        y = pd.Series(np.zeros(n))

        model = SimpleModel(predictions=np.zeros(n))

        comparison = ModelComparison({"model": model})
        results_df = comparison.evaluate_all(X, y)

        assert results_df["rmse"].iloc[0] == 0.0
        assert results_df["mae"].iloc[0] == 0.0

    def test_significance_missing_predictions(self, sample_data):
        """Test significance test with missing predictions."""
        X, y = sample_data

        models = {"model1": SimpleModel(), "model2": SimpleModel()}
        comparison = ModelComparison(models)

        # Don't call evaluate_all, predictions won't be stored

        with pytest.raises(ValueError, match="No predictions available"):
            comparison.statistical_significance_test(
                "model1", "model2", y_true=y.values
            )

    def test_significance_shape_mismatch(self, sample_data):
        """Test significance test with shape mismatch."""
        X, y = sample_data

        models = {"model1": SimpleModel(), "model2": SimpleModel()}
        comparison = ModelComparison(models)

        y_true = y.values
        y_pred1 = y_true.copy()
        y_pred2 = y_true[:50]  # Wrong length

        with pytest.raises(ValueError, match="Shape mismatch"):
            comparison.statistical_significance_test(
                "model1", "model2",
                y_true=y_true,
                y_pred1=y_pred1,
                y_pred2=y_pred2,
            )
