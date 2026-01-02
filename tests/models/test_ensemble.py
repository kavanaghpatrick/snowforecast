"""Tests for ensemble methods.

This module tests SimpleEnsemble and StackingEnsemble using realistic
synthetic snow data with multiple base models.
"""


import numpy as np
import pandas as pd
import pytest

from snowforecast.models.ensemble import (
    SimpleEnsemble,
    StackingEnsemble,
    create_ensemble,
    get_model_weights,
)
from snowforecast.models.linear import LinearRegressionModel

# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture
def realistic_snow_features():
    """Create realistic features for snow depth prediction."""
    np.random.seed(42)
    n_samples = 200

    elevation_m = np.random.uniform(2000, 3500, n_samples)
    temperature_c = np.random.uniform(-25, 5, n_samples)
    precipitation_mm = np.random.exponential(10, n_samples)
    wind_speed_mps = np.random.uniform(0, 25, n_samples)
    humidity_pct = np.random.uniform(40, 100, n_samples)

    return pd.DataFrame({
        "elevation_m": elevation_m,
        "temperature_c": temperature_c,
        "precipitation_mm": precipitation_mm,
        "wind_speed_mps": wind_speed_mps,
        "humidity_pct": humidity_pct,
    })


@pytest.fixture
def realistic_snow_target(realistic_snow_features):
    """Create realistic snow depth target based on features."""
    np.random.seed(42)
    X = realistic_snow_features

    snow_depth = (
        0.05 * X["elevation_m"]
        - 3.0 * X["temperature_c"]
        + 0.8 * X["precipitation_mm"]
        - 0.2 * X["wind_speed_mps"]
        + np.random.normal(0, 10, len(X))
    )

    snow_depth = np.maximum(snow_depth, 0)
    return pd.Series(snow_depth, name="snow_depth_cm")


@pytest.fixture
def train_test_data(realistic_snow_features, realistic_snow_target):
    """Split data into train and test sets."""
    n_train = 150

    X_train = realistic_snow_features.iloc[:n_train].reset_index(drop=True)
    X_test = realistic_snow_features.iloc[n_train:].reset_index(drop=True)
    y_train = realistic_snow_target.iloc[:n_train].reset_index(drop=True)
    y_test = realistic_snow_target.iloc[n_train:].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


@pytest.fixture
def fitted_models(train_test_data):
    """Return a list of fitted models for ensemble testing."""
    X_train, _, y_train, _ = train_test_data

    model1 = LinearRegressionModel(normalize=True)
    model1.fit(X_train, y_train)

    model2 = LinearRegressionModel(normalize=False)
    model2.fit(X_train, y_train)

    return [model1, model2]


@pytest.fixture
def simple_ensemble(fitted_models):
    """Return a simple averaging ensemble."""
    return SimpleEnsemble(models=fitted_models)


@pytest.fixture
def weighted_ensemble(fitted_models):
    """Return a weighted averaging ensemble."""
    return SimpleEnsemble(models=fitted_models, weights=[0.7, 0.3])


# =============================================================================
# Test SimpleEnsemble initialization
# =============================================================================


class TestSimpleEnsembleInit:
    """Tests for SimpleEnsemble initialization."""

    def test_init_with_models(self, fitted_models):
        """Should initialize with models."""
        ensemble = SimpleEnsemble(models=fitted_models)
        assert ensemble.is_fitted is True
        assert len(ensemble.models) == 2

    def test_init_with_weights(self, fitted_models):
        """Should initialize with custom weights."""
        weights = [0.6, 0.4]
        ensemble = SimpleEnsemble(models=fitted_models, weights=weights)
        np.testing.assert_array_almost_equal(ensemble.weights, [0.6, 0.4])

    def test_init_weights_normalized(self, fitted_models):
        """Weights should be normalized to sum to 1."""
        weights = [2.0, 3.0]  # Sum = 5
        ensemble = SimpleEnsemble(models=fitted_models, weights=weights)
        np.testing.assert_array_almost_equal(ensemble.weights, [0.4, 0.6])

    def test_init_no_models(self):
        """Should initialize without models."""
        ensemble = SimpleEnsemble()
        assert ensemble.is_fitted is False
        assert len(ensemble.models) == 0

    def test_init_unfitted_model_raises(self, train_test_data):
        """Should raise if any model is not fitted."""
        _, _, _, _ = train_test_data
        unfitted_model = LinearRegressionModel()
        fitted_model = LinearRegressionModel()

        with pytest.raises(ValueError, match="not fitted"):
            SimpleEnsemble(models=[unfitted_model, fitted_model])

    def test_init_weights_length_mismatch_raises(self, fitted_models):
        """Should raise if weights don't match models."""
        with pytest.raises(ValueError, match="Number of weights"):
            SimpleEnsemble(models=fitted_models, weights=[0.5])

    def test_repr(self, simple_ensemble):
        """Repr should show useful info."""
        repr_str = repr(simple_ensemble)
        assert "SimpleEnsemble" in repr_str
        assert "n_models=2" in repr_str
        assert "fitted" in repr_str


# =============================================================================
# Test SimpleEnsemble predict
# =============================================================================


class TestSimpleEnsemblePredict:
    """Tests for SimpleEnsemble prediction."""

    def test_predict_returns_array(self, simple_ensemble, train_test_data):
        """Predict should return numpy array."""
        _, X_test, _, _ = train_test_data
        result = simple_ensemble.predict(X_test)
        assert isinstance(result, np.ndarray)

    def test_predict_correct_length(self, simple_ensemble, train_test_data):
        """Predict should return correct number of predictions."""
        _, X_test, _, _ = train_test_data
        result = simple_ensemble.predict(X_test)
        assert len(result) == len(X_test)

    def test_predict_mean_is_average(self, fitted_models, train_test_data):
        """Mean aggregation should average predictions."""
        _, X_test, _, _ = train_test_data

        ensemble = SimpleEnsemble(models=fitted_models, aggregation="mean")
        ensemble_pred = ensemble.predict(X_test)

        # Manual average
        pred1 = fitted_models[0].predict(X_test)
        pred2 = fitted_models[1].predict(X_test)
        expected = (pred1 + pred2) / 2

        np.testing.assert_array_almost_equal(ensemble_pred, expected)

    def test_predict_weighted_average(self, fitted_models, train_test_data):
        """Weighted averaging should use weights."""
        _, X_test, _, _ = train_test_data
        weights = [0.7, 0.3]

        ensemble = SimpleEnsemble(models=fitted_models, weights=weights)
        ensemble_pred = ensemble.predict(X_test)

        # Manual weighted average
        pred1 = fitted_models[0].predict(X_test)
        pred2 = fitted_models[1].predict(X_test)
        expected = 0.7 * pred1 + 0.3 * pred2

        np.testing.assert_array_almost_equal(ensemble_pred, expected)

    def test_predict_median_aggregation(self, fitted_models, train_test_data):
        """Median aggregation should use median."""
        _, X_test, _, _ = train_test_data

        ensemble = SimpleEnsemble(models=fitted_models, aggregation="median")
        ensemble_pred = ensemble.predict(X_test)

        # With 2 models, median = mean
        pred1 = fitted_models[0].predict(X_test)
        pred2 = fitted_models[1].predict(X_test)
        expected = np.median([pred1, pred2], axis=0)

        np.testing.assert_array_almost_equal(ensemble_pred, expected)

    def test_predict_unfitted_raises(self, train_test_data):
        """Should raise when not fitted."""
        _, X_test, _, _ = train_test_data
        ensemble = SimpleEnsemble()
        with pytest.raises(ValueError, match="not fitted"):
            ensemble.predict(X_test)


# =============================================================================
# Test SimpleEnsemble robustness
# =============================================================================


class TestSimpleEnsembleRobustness:
    """Tests for ensemble robustness when models fail."""

    def test_handles_model_with_nan_predictions(self, train_test_data):
        """Ensemble should handle models that return NaN."""
        X_train, X_test, y_train, _ = train_test_data

        # Create a model that returns NaN (we'll mock this behavior)
        class NaNModel(LinearRegressionModel):
            def predict(self, X):
                result = super().predict(X)
                result[0] = np.nan  # Introduce NaN
                return result

        good_model = LinearRegressionModel().fit(X_train, y_train)
        nan_model = NaNModel().fit(X_train, y_train)

        # Ensemble should exclude the NaN model and still work
        ensemble = SimpleEnsemble(models=[good_model, nan_model])

        # With one model returning NaN, ensemble uses only the good one
        result = ensemble.predict(X_test)
        assert np.isfinite(result).all()

    def test_ensemble_all_models_fail_raises(self, train_test_data):
        """Should raise if all models fail."""
        X_train, X_test, y_train, _ = train_test_data

        class FailingModel(LinearRegressionModel):
            def predict(self, X):
                raise RuntimeError("Model failed")

        model = FailingModel().fit(X_train, y_train)
        ensemble = SimpleEnsemble(models=[model])

        with pytest.raises(ValueError, match="All models failed"):
            ensemble.predict(X_test)


# =============================================================================
# Test SimpleEnsemble performance
# =============================================================================


class TestSimpleEnsemblePerformance:
    """Tests for ensemble predictive performance."""

    def test_ensemble_not_worse_than_worst_model(self, fitted_models, train_test_data):
        """Ensemble should not be worse than the worst base model."""
        _, X_test, _, y_test = train_test_data

        ensemble = SimpleEnsemble(models=fitted_models)
        ensemble_pred = ensemble.predict(X_test)

        # Calculate errors
        ensemble_rmse = np.sqrt(np.mean((y_test - ensemble_pred) ** 2))

        model_rmses = []
        for model in fitted_models:
            pred = model.predict(X_test)
            rmse = np.sqrt(np.mean((y_test - pred) ** 2))
            model_rmses.append(rmse)

        worst_rmse = max(model_rmses)

        # Ensemble should be at least as good as worst model (with some tolerance)
        assert ensemble_rmse <= worst_rmse * 1.1, (
            f"Ensemble RMSE {ensemble_rmse:.2f} is worse than "
            f"worst model RMSE {worst_rmse:.2f}"
        )

    def test_ensemble_variance_reduction(self, train_test_data):
        """Ensemble predictions should have reduced variance vs individuals."""
        X_train, X_test, y_train, _ = train_test_data

        # Create multiple diverse models with different random seeds
        models = []
        for i in range(5):
            model = LinearRegressionModel(normalize=bool(i % 2))
            model.fit(X_train, y_train)
            models.append(model)

        ensemble = SimpleEnsemble(models=models)

        # Get predictions
        individual_preds = [m.predict(X_test) for m in models]
        ensemble_pred = ensemble.predict(X_test)

        # Variance of ensemble should be <= mean variance of individuals
        individual_variances = [np.var(p) for p in individual_preds]
        mean_individual_var = np.mean(individual_variances)
        ensemble_var = np.var(ensemble_pred)

        # This is a soft check - ensemble variance is typically lower
        # but not guaranteed in all cases
        assert ensemble_var <= mean_individual_var * 1.5


# =============================================================================
# Test StackingEnsemble
# =============================================================================


class TestStackingEnsembleInit:
    """Tests for StackingEnsemble initialization."""

    def test_init_with_base_models(self):
        """Should initialize with base models."""
        models = [LinearRegressionModel(), LinearRegressionModel()]
        ensemble = StackingEnsemble(base_models=models)
        assert ensemble.is_fitted is False
        assert len(ensemble._base_models) == 2

    def test_init_default_meta_learner(self):
        """Should use LinearRegressionModel as default meta-learner."""
        ensemble = StackingEnsemble()
        assert isinstance(ensemble.meta_learner, LinearRegressionModel)

    def test_init_custom_meta_learner(self):
        """Should accept custom meta-learner."""
        meta = LinearRegressionModel(normalize=False)
        ensemble = StackingEnsemble(meta_learner=meta)
        assert ensemble.meta_learner is meta

    def test_init_cv_folds(self):
        """Should accept cv_folds parameter."""
        ensemble = StackingEnsemble(cv_folds=3)
        assert ensemble.cv_folds == 3

    def test_init_cv_folds_too_small_raises(self):
        """Should raise if cv_folds < 2."""
        with pytest.raises(ValueError, match="cv_folds must be >= 2"):
            StackingEnsemble(cv_folds=1)

    def test_repr(self):
        """Repr should show useful info."""
        ensemble = StackingEnsemble(
            base_models=[LinearRegressionModel()],
        )
        repr_str = repr(ensemble)
        assert "StackingEnsemble" in repr_str
        assert "not fitted" in repr_str


class TestStackingEnsembleFit:
    """Tests for StackingEnsemble fitting."""

    def test_fit_returns_self(self, train_test_data):
        """Fit should return self for method chaining."""
        X_train, _, y_train, _ = train_test_data
        models = [LinearRegressionModel(), LinearRegressionModel()]
        ensemble = StackingEnsemble(base_models=models, cv_folds=3)

        result = ensemble.fit(X_train, y_train)
        assert result is ensemble

    def test_fit_sets_is_fitted(self, train_test_data):
        """Fit should set is_fitted to True."""
        X_train, _, y_train, _ = train_test_data
        models = [LinearRegressionModel(), LinearRegressionModel()]
        ensemble = StackingEnsemble(base_models=models, cv_folds=3)

        ensemble.fit(X_train, y_train)
        assert ensemble.is_fitted is True

    def test_fit_trains_base_models(self, train_test_data):
        """Fit should train all base models."""
        X_train, _, y_train, _ = train_test_data
        models = [LinearRegressionModel(), LinearRegressionModel()]
        ensemble = StackingEnsemble(base_models=models, cv_folds=3)

        ensemble.fit(X_train, y_train)

        # All base models should be fitted
        for model in ensemble.base_models:
            assert model.is_fitted

    def test_fit_trains_meta_learner(self, train_test_data):
        """Fit should train the meta-learner."""
        X_train, _, y_train, _ = train_test_data
        models = [LinearRegressionModel(), LinearRegressionModel()]
        ensemble = StackingEnsemble(base_models=models, cv_folds=3)

        ensemble.fit(X_train, y_train)
        assert ensemble.meta_learner.is_fitted

    def test_fit_no_models_raises(self, train_test_data):
        """Fit should raise when no base models provided."""
        X_train, _, y_train, _ = train_test_data
        ensemble = StackingEnsemble()

        with pytest.raises(ValueError, match="No base models provided"):
            ensemble.fit(X_train, y_train)


class TestStackingEnsemblePredict:
    """Tests for StackingEnsemble prediction."""

    def test_predict_returns_array(self, train_test_data):
        """Predict should return numpy array."""
        X_train, X_test, y_train, _ = train_test_data
        models = [LinearRegressionModel(), LinearRegressionModel()]
        ensemble = StackingEnsemble(base_models=models, cv_folds=3)
        ensemble.fit(X_train, y_train)

        result = ensemble.predict(X_test)
        assert isinstance(result, np.ndarray)

    def test_predict_correct_length(self, train_test_data):
        """Predict should return correct number of predictions."""
        X_train, X_test, y_train, _ = train_test_data
        models = [LinearRegressionModel(), LinearRegressionModel()]
        ensemble = StackingEnsemble(base_models=models, cv_folds=3)
        ensemble.fit(X_train, y_train)

        result = ensemble.predict(X_test)
        assert len(result) == len(X_test)

    def test_predict_unfitted_raises(self, train_test_data):
        """Should raise when not fitted."""
        _, X_test, _, _ = train_test_data
        models = [LinearRegressionModel()]
        ensemble = StackingEnsemble(base_models=models)

        with pytest.raises(ValueError, match="not fitted"):
            ensemble.predict(X_test)

    def test_predict_with_use_features(self, train_test_data):
        """Should work with use_features=True."""
        X_train, X_test, y_train, _ = train_test_data
        models = [LinearRegressionModel(), LinearRegressionModel()]
        ensemble = StackingEnsemble(
            base_models=models,
            use_features=True,
            cv_folds=3,
        )
        ensemble.fit(X_train, y_train)

        result = ensemble.predict(X_test)
        assert len(result) == len(X_test)


class TestStackingEnsemblePerformance:
    """Tests for stacking ensemble performance."""

    def test_stacking_not_worse_than_worst_base(self, train_test_data):
        """Stacking should not be worse than worst base model."""
        X_train, X_test, y_train, y_test = train_test_data

        models = [
            LinearRegressionModel(normalize=True),
            LinearRegressionModel(normalize=False),
        ]
        ensemble = StackingEnsemble(base_models=models, cv_folds=3)
        ensemble.fit(X_train, y_train)

        ensemble_pred = ensemble.predict(X_test)
        ensemble_rmse = np.sqrt(np.mean((y_test - ensemble_pred) ** 2))

        # Calculate base model errors
        base_rmses = []
        for model in ensemble.base_models:
            pred = model.predict(X_test)
            rmse = np.sqrt(np.mean((y_test - pred) ** 2))
            base_rmses.append(rmse)

        worst_rmse = max(base_rmses)

        # Stacking should be at least as good as worst model
        assert ensemble_rmse <= worst_rmse * 1.1, (
            f"Stacking RMSE {ensemble_rmse:.2f} is worse than "
            f"worst base RMSE {worst_rmse:.2f}"
        )


# =============================================================================
# Test helper functions
# =============================================================================


class TestCreateEnsemble:
    """Tests for create_ensemble factory function."""

    def test_create_average_ensemble(self, fitted_models):
        """Should create averaging ensemble."""
        ensemble = create_ensemble(fitted_models, method="average")
        assert isinstance(ensemble, SimpleEnsemble)
        assert ensemble.aggregation == "mean"

    def test_create_weighted_ensemble(self, fitted_models):
        """Should create weighted ensemble."""
        weights = [0.6, 0.4]
        ensemble = create_ensemble(fitted_models, method="weighted", weights=weights)
        assert isinstance(ensemble, SimpleEnsemble)
        np.testing.assert_array_almost_equal(ensemble.weights, [0.6, 0.4])

    def test_create_weighted_without_weights_raises(self, fitted_models):
        """Should raise when weighted method without weights."""
        with pytest.raises(ValueError, match="weights must be provided"):
            create_ensemble(fitted_models, method="weighted")

    def test_create_stacking_ensemble(self):
        """Should create stacking ensemble."""
        models = [LinearRegressionModel(), LinearRegressionModel()]
        ensemble = create_ensemble(models, method="stacking")
        assert isinstance(ensemble, StackingEnsemble)

    def test_create_stacking_with_meta_learner(self):
        """Should accept custom meta-learner for stacking."""
        models = [LinearRegressionModel()]
        meta = LinearRegressionModel(normalize=False)
        ensemble = create_ensemble(models, method="stacking", meta_learner=meta)
        assert ensemble.meta_learner is meta

    def test_create_invalid_method_raises(self, fitted_models):
        """Should raise for invalid method."""
        with pytest.raises(ValueError, match="method must be one of"):
            create_ensemble(fitted_models, method="invalid")


class TestGetModelWeights:
    """Tests for get_model_weights function."""

    def test_get_weights_simple_ensemble(self, fitted_models):
        """Should return weights for SimpleEnsemble."""
        weights = [0.7, 0.3]
        ensemble = SimpleEnsemble(models=fitted_models, weights=weights)

        result = get_model_weights(ensemble)

        assert isinstance(result, pd.DataFrame)
        assert "model" in result.columns
        assert "weight" in result.columns
        assert len(result) == 2
        np.testing.assert_array_almost_equal(result["weight"].values, [0.7, 0.3])

    def test_get_weights_stacking_ensemble(self, train_test_data):
        """Should return learned weights for StackingEnsemble."""
        X_train, _, y_train, _ = train_test_data
        models = [LinearRegressionModel(), LinearRegressionModel()]
        ensemble = StackingEnsemble(base_models=models, cv_folds=3)
        ensemble.fit(X_train, y_train)

        result = get_model_weights(ensemble)

        assert isinstance(result, pd.DataFrame)
        assert "model" in result.columns
        assert "weight" in result.columns
        assert len(result) == 2

    def test_get_weights_unfitted_raises(self, fitted_models):
        """Should raise for unfitted ensemble."""
        ensemble = SimpleEnsemble()
        with pytest.raises(ValueError, match="not fitted"):
            get_model_weights(ensemble)

    def test_get_weights_invalid_type_raises(self):
        """Should raise for non-ensemble type."""
        model = LinearRegressionModel()
        with pytest.raises(ValueError, match="must be SimpleEnsemble or StackingEnsemble"):
            get_model_weights(model)


# =============================================================================
# Test save and load
# =============================================================================


class TestSimpleEnsembleSaveLoad:
    """Tests for SimpleEnsemble persistence."""

    def test_save_creates_file(self, simple_ensemble, tmp_path):
        """Save should create file."""
        model_path = tmp_path / "ensemble.pkl"
        simple_ensemble.save(model_path)
        assert model_path.exists()

    def test_save_unfitted_raises(self, tmp_path):
        """Save should raise on unfitted ensemble."""
        ensemble = SimpleEnsemble()
        model_path = tmp_path / "ensemble.pkl"
        with pytest.raises(ValueError, match="not fitted"):
            ensemble.save(model_path)

    def test_load_returns_ensemble(self, simple_ensemble, tmp_path):
        """Load should return ensemble instance."""
        model_path = tmp_path / "ensemble.pkl"
        simple_ensemble.save(model_path)

        loaded = SimpleEnsemble.load(model_path)
        assert isinstance(loaded, SimpleEnsemble)
        assert loaded.is_fitted

    def test_load_preserves_predictions(
        self, simple_ensemble, tmp_path, train_test_data
    ):
        """Loaded ensemble should give same predictions."""
        _, X_test, _, _ = train_test_data
        model_path = tmp_path / "ensemble.pkl"

        original_pred = simple_ensemble.predict(X_test)
        simple_ensemble.save(model_path)

        loaded = SimpleEnsemble.load(model_path)
        loaded_pred = loaded.predict(X_test)

        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_load_nonexistent_raises(self, tmp_path):
        """Load should raise for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            SimpleEnsemble.load(tmp_path / "nonexistent.pkl")


class TestStackingEnsembleSaveLoad:
    """Tests for StackingEnsemble persistence."""

    def test_save_creates_file(self, train_test_data, tmp_path):
        """Save should create file."""
        X_train, _, y_train, _ = train_test_data
        models = [LinearRegressionModel(), LinearRegressionModel()]
        ensemble = StackingEnsemble(base_models=models, cv_folds=3)
        ensemble.fit(X_train, y_train)

        model_path = tmp_path / "stacking.pkl"
        ensemble.save(model_path)
        assert model_path.exists()

    def test_load_preserves_predictions(self, train_test_data, tmp_path):
        """Loaded ensemble should give same predictions."""
        X_train, X_test, y_train, _ = train_test_data
        models = [LinearRegressionModel(), LinearRegressionModel()]
        ensemble = StackingEnsemble(base_models=models, cv_folds=3)
        ensemble.fit(X_train, y_train)

        model_path = tmp_path / "stacking.pkl"
        original_pred = ensemble.predict(X_test)
        ensemble.save(model_path)

        loaded = StackingEnsemble.load(model_path)
        loaded_pred = loaded.predict(X_test)

        np.testing.assert_array_almost_equal(original_pred, loaded_pred)


# =============================================================================
# Test feature importance
# =============================================================================


class TestFeatureImportance:
    """Tests for ensemble feature importance."""

    def test_simple_ensemble_importance(self, simple_ensemble):
        """SimpleEnsemble should return averaged importance."""
        result = simple_ensemble.get_feature_importance()

        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "importance" in result.columns

        # Should have all features
        assert len(result) == len(simple_ensemble.feature_names)

        # Importance should sum to ~1
        assert abs(result["importance"].sum() - 1.0) < 0.01

    def test_stacking_ensemble_importance(self, train_test_data):
        """StackingEnsemble should return meta-learner importance."""
        X_train, _, y_train, _ = train_test_data
        models = [LinearRegressionModel(), LinearRegressionModel()]
        ensemble = StackingEnsemble(base_models=models, cv_folds=3)
        ensemble.fit(X_train, y_train)

        result = ensemble.get_feature_importance()

        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "importance" in result.columns


# =============================================================================
# Test get_params and set_params
# =============================================================================


class TestParams:
    """Tests for parameter management."""

    def test_simple_ensemble_get_params(self, simple_ensemble):
        """get_params should return configuration."""
        params = simple_ensemble.get_params()

        assert "n_models" in params
        assert params["n_models"] == 2
        assert "aggregation" in params
        assert "weights" in params
        assert "model_types" in params

    def test_simple_ensemble_set_params(self, simple_ensemble):
        """set_params should update configuration."""
        simple_ensemble.set_params(aggregation="median")
        assert simple_ensemble.aggregation == "median"

    def test_simple_ensemble_set_weights(self, simple_ensemble):
        """set_params should update weights."""
        simple_ensemble.set_params(weights=[0.8, 0.2])
        np.testing.assert_array_almost_equal(simple_ensemble.weights, [0.8, 0.2])

    def test_stacking_ensemble_get_params(self, train_test_data):
        """get_params should return stacking configuration."""
        X_train, _, y_train, _ = train_test_data
        models = [LinearRegressionModel(), LinearRegressionModel()]
        ensemble = StackingEnsemble(base_models=models, cv_folds=3)
        ensemble.fit(X_train, y_train)

        params = ensemble.get_params()

        assert "n_base_models" in params
        assert params["n_base_models"] == 2
        assert "meta_learner_type" in params
        assert "use_features" in params
        assert "cv_folds" in params

    def test_stacking_set_params_fitted_raises(self, train_test_data):
        """set_params should raise on fitted StackingEnsemble."""
        X_train, _, y_train, _ = train_test_data
        models = [LinearRegressionModel()]
        ensemble = StackingEnsemble(base_models=models, cv_folds=3)
        ensemble.fit(X_train, y_train)

        with pytest.raises(ValueError, match="Cannot change parameters"):
            ensemble.set_params(cv_folds=5)


# =============================================================================
# Test edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_model_ensemble(self, train_test_data):
        """Ensemble should work with single model."""
        X_train, X_test, y_train, _ = train_test_data

        model = LinearRegressionModel().fit(X_train, y_train)
        ensemble = SimpleEnsemble(models=[model])

        result = ensemble.predict(X_test)
        expected = model.predict(X_test)

        np.testing.assert_array_almost_equal(result, expected)

    def test_many_models_ensemble(self, train_test_data):
        """Ensemble should work with many models."""
        X_train, X_test, y_train, _ = train_test_data

        models = []
        for _ in range(10):
            model = LinearRegressionModel().fit(X_train, y_train)
            models.append(model)

        ensemble = SimpleEnsemble(models=models)
        result = ensemble.predict(X_test)

        assert len(result) == len(X_test)
        assert np.isfinite(result).all()

    def test_set_models_after_init(self, fitted_models, train_test_data):
        """Should be able to set models after initialization."""
        _, X_test, _, _ = train_test_data

        ensemble = SimpleEnsemble()
        assert not ensemble.is_fitted

        ensemble.set_models(fitted_models)
        assert ensemble.is_fitted

        result = ensemble.predict(X_test)
        assert len(result) == len(X_test)

    def test_stacking_with_small_data(self):
        """Stacking should work with small dataset."""
        np.random.seed(42)
        n_samples = 30  # Small dataset

        X = pd.DataFrame({
            "feature1": np.random.randn(n_samples),
            "feature2": np.random.randn(n_samples),
        })
        y = pd.Series(np.random.randn(n_samples))

        models = [LinearRegressionModel()]
        ensemble = StackingEnsemble(base_models=models, cv_folds=3)
        ensemble.fit(X, y)

        result = ensemble.predict(X)
        assert len(result) == n_samples
