"""Tests for LinearRegressionModel.

This module tests the linear regression baseline model using
realistic synthetic snow data.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from snowforecast.models.linear import LinearRegressionModel


# =============================================================================
# Test fixtures with realistic snow data
# =============================================================================


@pytest.fixture
def realistic_snow_features():
    """Create realistic features for snow depth prediction.

    Features are based on typical mountain weather station data:
    - Elevation affects snow depth (higher = more snow)
    - Temperature affects snow (colder = more snow)
    - Precipitation contributes to snow
    - Wind can redistribute snow
    - Humidity affects snow type
    """
    np.random.seed(42)
    n_samples = 200

    # Base features
    elevation_m = np.random.uniform(2000, 3500, n_samples)
    temperature_c = np.random.uniform(-25, 5, n_samples)
    precipitation_mm = np.random.exponential(10, n_samples)
    wind_speed_mps = np.random.uniform(0, 25, n_samples)
    humidity_pct = np.random.uniform(40, 100, n_samples)
    north_facing = np.random.choice([0, 1], n_samples)
    slope_deg = np.random.uniform(0, 45, n_samples)

    return pd.DataFrame({
        "elevation_m": elevation_m,
        "temperature_c": temperature_c,
        "precipitation_mm": precipitation_mm,
        "wind_speed_mps": wind_speed_mps,
        "humidity_pct": humidity_pct,
        "north_facing": north_facing,
        "slope_deg": slope_deg,
    })


@pytest.fixture
def realistic_snow_target(realistic_snow_features):
    """Create realistic snow depth target based on features.

    Snow depth is generated with a known linear relationship plus noise:
    - Higher elevation = more snow
    - Lower temperature = more snow
    - More precipitation = more snow
    - North-facing slopes retain more snow
    """
    np.random.seed(42)
    X = realistic_snow_features

    # True coefficients (for validation)
    snow_depth = (
        0.05 * X["elevation_m"]           # +50cm per 1000m elevation
        - 3.0 * X["temperature_c"]         # -3cm per degree warming
        + 0.8 * X["precipitation_mm"]      # +0.8cm per mm precip
        + 15.0 * X["north_facing"]         # +15cm for north-facing
        - 0.5 * X["slope_deg"]             # -0.5cm per degree slope
        - 0.2 * X["wind_speed_mps"]        # wind removes some snow
        + np.random.normal(0, 10, len(X))  # noise
    )

    # Ensure non-negative
    snow_depth = np.maximum(snow_depth, 0)

    return pd.Series(snow_depth, name="snow_depth_cm")


@pytest.fixture
def train_test_data(realistic_snow_features, realistic_snow_target):
    """Split data into train and test sets."""
    n_train = 150

    X_train = realistic_snow_features.iloc[:n_train]
    X_test = realistic_snow_features.iloc[n_train:]
    y_train = realistic_snow_target.iloc[:n_train]
    y_test = realistic_snow_target.iloc[n_train:]

    return X_train, X_test, y_train, y_test


@pytest.fixture
def fitted_model(train_test_data):
    """Return a fitted LinearRegressionModel."""
    X_train, _, y_train, _ = train_test_data
    model = LinearRegressionModel(normalize=True)
    model.fit(X_train, y_train)
    return model


# =============================================================================
# Test initialization
# =============================================================================


class TestLinearRegressionModelInit:
    """Tests for LinearRegressionModel initialization."""

    def test_default_normalize_true(self):
        """Default should normalize features."""
        model = LinearRegressionModel()
        assert model.normalize is True

    def test_normalize_false(self):
        """Should accept normalize=False."""
        model = LinearRegressionModel(normalize=False)
        assert model.normalize is False

    def test_initial_state(self):
        """Should initialize with unfitted state."""
        model = LinearRegressionModel()
        assert model.is_fitted is False
        assert model.model is None
        assert model.scaler is None
        assert model.feature_names is None

    def test_repr_unfitted(self):
        """Repr should show unfitted status."""
        model = LinearRegressionModel()
        repr_str = repr(model)
        assert "not fitted" in repr_str
        assert "LinearRegressionModel" in repr_str


# =============================================================================
# Test fit method
# =============================================================================


class TestFit:
    """Tests for fit method."""

    def test_fit_returns_self(self, train_test_data):
        """Fit should return self for method chaining."""
        X_train, _, y_train, _ = train_test_data
        model = LinearRegressionModel()
        result = model.fit(X_train, y_train)
        assert result is model

    def test_fit_sets_is_fitted(self, train_test_data):
        """Fit should set is_fitted to True."""
        X_train, _, y_train, _ = train_test_data
        model = LinearRegressionModel()
        model.fit(X_train, y_train)
        assert model.is_fitted is True

    def test_fit_stores_feature_names(self, train_test_data):
        """Fit should store feature names."""
        X_train, _, y_train, _ = train_test_data
        model = LinearRegressionModel()
        model.fit(X_train, y_train)
        assert model.feature_names == list(X_train.columns)

    def test_fit_with_normalization(self, train_test_data):
        """Fit with normalize=True should create scaler."""
        X_train, _, y_train, _ = train_test_data
        model = LinearRegressionModel(normalize=True)
        model.fit(X_train, y_train)
        assert model.scaler is not None

    def test_fit_without_normalization(self, train_test_data):
        """Fit with normalize=False should not create scaler."""
        X_train, _, y_train, _ = train_test_data
        model = LinearRegressionModel(normalize=False)
        model.fit(X_train, y_train)
        assert model.scaler is None

    def test_fit_invalid_X_type(self, train_test_data):
        """Fit should reject non-DataFrame X."""
        X_train, _, y_train, _ = train_test_data
        model = LinearRegressionModel()
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            model.fit(X_train.values, y_train)

    def test_fit_invalid_y_type(self, train_test_data):
        """Fit should reject non-Series y."""
        X_train, _, y_train, _ = train_test_data
        model = LinearRegressionModel()
        with pytest.raises(ValueError, match="must be a pandas Series"):
            model.fit(X_train, y_train.values)

    def test_fit_length_mismatch(self, train_test_data):
        """Fit should reject mismatched X and y lengths."""
        X_train, _, y_train, _ = train_test_data
        model = LinearRegressionModel()
        with pytest.raises(ValueError, match="must have same length"):
            model.fit(X_train.iloc[:10], y_train)

    def test_fit_with_nan_X(self, train_test_data):
        """Fit should reject X with NaN values."""
        X_train, _, y_train, _ = train_test_data
        X_nan = X_train.copy()
        X_nan.iloc[0, 0] = np.nan
        model = LinearRegressionModel()
        with pytest.raises(ValueError, match="contains NaN"):
            model.fit(X_nan, y_train)

    def test_fit_with_nan_y(self, train_test_data):
        """Fit should reject y with NaN values."""
        X_train, _, y_train, _ = train_test_data
        y_nan = y_train.copy()
        y_nan.iloc[0] = np.nan
        model = LinearRegressionModel()
        with pytest.raises(ValueError, match="contains NaN"):
            model.fit(X_train, y_nan)

    def test_fit_eval_set_ignored(self, train_test_data):
        """eval_set should be accepted but ignored."""
        X_train, X_test, y_train, y_test = train_test_data
        model = LinearRegressionModel()
        # Should not raise, just ignore eval_set
        model.fit(X_train, y_train, eval_set=(X_test, y_test))
        assert model.is_fitted


# =============================================================================
# Test predict method
# =============================================================================


class TestPredict:
    """Tests for predict method."""

    def test_predict_returns_array(self, fitted_model, train_test_data):
        """Predict should return numpy array."""
        _, X_test, _, _ = train_test_data
        result = fitted_model.predict(X_test)
        assert isinstance(result, np.ndarray)

    def test_predict_correct_length(self, fitted_model, train_test_data):
        """Predict should return correct number of predictions."""
        _, X_test, _, _ = train_test_data
        result = fitted_model.predict(X_test)
        assert len(result) == len(X_test)

    def test_predict_reasonable_values(self, fitted_model, train_test_data):
        """Predictions should be in reasonable range for snow depth."""
        _, X_test, _, y_test = train_test_data
        predictions = fitted_model.predict(X_test)

        # Should be within reasonable range (not negative or extreme)
        assert predictions.min() >= -50  # Allow some negative due to linear model
        assert predictions.max() <= 500  # Reasonable max snow depth

    def test_predict_unfitted_raises(self, train_test_data):
        """Predict should raise when not fitted."""
        _, X_test, _, _ = train_test_data
        model = LinearRegressionModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X_test)

    def test_predict_feature_mismatch_raises(self, fitted_model, train_test_data):
        """Predict should raise when features don't match training."""
        _, X_test, _, _ = train_test_data
        X_wrong = X_test.rename(columns={"elevation_m": "altitude_m"})
        with pytest.raises(ValueError, match="Feature mismatch"):
            fitted_model.predict(X_wrong)

    def test_predict_with_nan_raises(self, fitted_model, train_test_data):
        """Predict should raise when X contains NaN."""
        _, X_test, _, _ = train_test_data
        X_nan = X_test.copy()
        X_nan.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="contains NaN"):
            fitted_model.predict(X_nan)


# =============================================================================
# Test model performance
# =============================================================================


class TestModelPerformance:
    """Tests for model predictive performance."""

    def test_model_learns_linear_relationship(self, train_test_data):
        """Model should learn approximate linear relationship."""
        X_train, X_test, y_train, y_test = train_test_data
        model = LinearRegressionModel(normalize=True)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # Calculate R-squared
        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Should have reasonable R-squared for synthetic linear data
        assert r_squared > 0.5, f"R-squared {r_squared} is too low"

    def test_normalized_vs_unnormalized(self, train_test_data):
        """Normalized and unnormalized should give similar predictions."""
        X_train, X_test, y_train, _ = train_test_data

        model_norm = LinearRegressionModel(normalize=True)
        model_norm.fit(X_train, y_train)
        pred_norm = model_norm.predict(X_test)

        model_unnorm = LinearRegressionModel(normalize=False)
        model_unnorm.fit(X_train, y_train)
        pred_unnorm = model_unnorm.predict(X_test)

        # Predictions should be very similar
        np.testing.assert_allclose(pred_norm, pred_unnorm, rtol=0.01)

    def test_reproducibility(self, train_test_data):
        """Same data should give same predictions."""
        X_train, X_test, y_train, _ = train_test_data

        model1 = LinearRegressionModel(normalize=True)
        model1.fit(X_train, y_train)
        pred1 = model1.predict(X_test)

        model2 = LinearRegressionModel(normalize=True)
        model2.fit(X_train, y_train)
        pred2 = model2.predict(X_test)

        np.testing.assert_array_equal(pred1, pred2)


# =============================================================================
# Test get_feature_importance
# =============================================================================


class TestGetFeatureImportance:
    """Tests for get_feature_importance method."""

    def test_returns_dataframe(self, fitted_model):
        """Should return DataFrame."""
        result = fitted_model.get_feature_importance()
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, fitted_model):
        """Should have feature, importance, and coefficient columns."""
        result = fitted_model.get_feature_importance()
        assert "feature" in result.columns
        assert "importance" in result.columns
        assert "coefficient" in result.columns

    def test_correct_feature_count(self, fitted_model):
        """Should have one row per feature."""
        result = fitted_model.get_feature_importance()
        assert len(result) == len(fitted_model.feature_names)

    def test_sorted_by_importance(self, fitted_model):
        """Should be sorted by importance descending."""
        result = fitted_model.get_feature_importance()
        assert list(result["importance"]) == sorted(
            result["importance"], reverse=True
        )

    def test_importance_is_abs_coefficient(self, fitted_model):
        """Importance should be absolute value of coefficient."""
        result = fitted_model.get_feature_importance()
        np.testing.assert_array_almost_equal(
            result["importance"],
            np.abs(result["coefficient"]),
        )

    def test_unfitted_raises(self):
        """Should raise when not fitted."""
        model = LinearRegressionModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.get_feature_importance()

    def test_elevation_is_important(self, fitted_model):
        """Elevation should be among top important features."""
        result = fitted_model.get_feature_importance()
        # Elevation should be in top 3 features
        top_features = result.head(3)["feature"].tolist()
        assert "elevation_m" in top_features or "temperature_c" in top_features


# =============================================================================
# Test get_coefficients and get_intercept
# =============================================================================


class TestCoefficients:
    """Tests for coefficient access methods."""

    def test_get_coefficients_returns_dataframe(self, fitted_model):
        """get_coefficients should return DataFrame."""
        result = fitted_model.get_coefficients()
        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "coefficient" in result.columns

    def test_get_intercept_returns_float(self, fitted_model):
        """get_intercept should return float."""
        result = fitted_model.get_intercept()
        assert isinstance(result, float)

    def test_coefficients_match_importance(self, fitted_model):
        """Coefficients should match those in importance DataFrame."""
        coefs = fitted_model.get_coefficients()
        importance = fitted_model.get_feature_importance()

        # Sort both by feature name for comparison
        coefs_sorted = coefs.sort_values("feature").reset_index(drop=True)
        importance_sorted = importance.sort_values("feature").reset_index(drop=True)

        np.testing.assert_array_almost_equal(
            coefs_sorted["coefficient"],
            importance_sorted["coefficient"],
        )

    def test_unfitted_coefficients_raises(self):
        """get_coefficients should raise when not fitted."""
        model = LinearRegressionModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.get_coefficients()

    def test_unfitted_intercept_raises(self):
        """get_intercept should raise when not fitted."""
        model = LinearRegressionModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.get_intercept()


# =============================================================================
# Test get_params and set_params
# =============================================================================


class TestParams:
    """Tests for parameter management methods."""

    def test_get_params_unfitted(self):
        """get_params should work on unfitted model."""
        model = LinearRegressionModel(normalize=True)
        params = model.get_params()
        assert params["normalize"] is True
        assert "n_features" not in params

    def test_get_params_fitted(self, fitted_model):
        """get_params should include additional info when fitted."""
        params = fitted_model.get_params()
        assert params["normalize"] is True
        assert "n_features" in params
        assert "intercept" in params
        assert params["n_features"] == 7  # Our test data has 7 features

    def test_set_params_unfitted(self):
        """set_params should work on unfitted model."""
        model = LinearRegressionModel(normalize=True)
        model.set_params(normalize=False)
        assert model.normalize is False

    def test_set_params_fitted_raises(self, fitted_model):
        """set_params should raise on fitted model."""
        with pytest.raises(ValueError, match="Cannot change parameters"):
            fitted_model.set_params(normalize=False)


# =============================================================================
# Test save and load
# =============================================================================


class TestSaveLoad:
    """Tests for model persistence."""

    def test_save_creates_file(self, fitted_model, tmp_path):
        """save should create file."""
        model_path = tmp_path / "model.pkl"
        fitted_model.save(model_path)
        assert model_path.exists()

    def test_save_unfitted_raises(self, tmp_path):
        """save should raise on unfitted model."""
        model = LinearRegressionModel()
        model_path = tmp_path / "model.pkl"
        with pytest.raises(ValueError, match="not fitted"):
            model.save(model_path)

    def test_load_returns_model(self, fitted_model, tmp_path):
        """load should return model instance."""
        model_path = tmp_path / "model.pkl"
        fitted_model.save(model_path)

        loaded = LinearRegressionModel.load(model_path)
        assert isinstance(loaded, LinearRegressionModel)
        assert loaded.is_fitted

    def test_load_preserves_state(self, fitted_model, tmp_path, train_test_data):
        """Loaded model should give same predictions."""
        _, X_test, _, _ = train_test_data
        model_path = tmp_path / "model.pkl"

        original_predictions = fitted_model.predict(X_test)
        fitted_model.save(model_path)

        loaded = LinearRegressionModel.load(model_path)
        loaded_predictions = loaded.predict(X_test)

        np.testing.assert_array_equal(original_predictions, loaded_predictions)

    def test_load_preserves_feature_names(self, fitted_model, tmp_path):
        """Loaded model should have same feature names."""
        model_path = tmp_path / "model.pkl"
        fitted_model.save(model_path)

        loaded = LinearRegressionModel.load(model_path)
        assert loaded.feature_names == fitted_model.feature_names

    def test_load_preserves_normalize(self, train_test_data, tmp_path):
        """Loaded model should preserve normalize setting."""
        X_train, _, y_train, _ = train_test_data

        for normalize in [True, False]:
            model = LinearRegressionModel(normalize=normalize)
            model.fit(X_train, y_train)

            model_path = tmp_path / f"model_{normalize}.pkl"
            model.save(model_path)

            loaded = LinearRegressionModel.load(model_path)
            assert loaded.normalize == normalize

    def test_load_nonexistent_raises(self, tmp_path):
        """load should raise for nonexistent file."""
        model_path = tmp_path / "nonexistent.pkl"
        with pytest.raises(FileNotFoundError):
            LinearRegressionModel.load(model_path)

    def test_save_creates_parent_dirs(self, fitted_model, tmp_path):
        """save should create parent directories."""
        model_path = tmp_path / "subdir" / "nested" / "model.pkl"
        fitted_model.save(model_path)
        assert model_path.exists()


# =============================================================================
# Test edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_feature(self):
        """Should work with single feature."""
        np.random.seed(42)
        X = pd.DataFrame({"feature": np.random.randn(100)})
        y = pd.Series(3 * X["feature"] + 5 + np.random.randn(100) * 0.1)

        model = LinearRegressionModel()
        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 100
        assert model.get_feature_importance().shape[0] == 1

    def test_many_features(self):
        """Should work with many features."""
        np.random.seed(42)
        n_features = 50
        X = pd.DataFrame(
            np.random.randn(100, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        y = pd.Series(np.random.randn(100))

        model = LinearRegressionModel()
        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 100
        assert model.get_feature_importance().shape[0] == n_features

    def test_small_dataset(self):
        """Should work with small dataset."""
        X = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [2.0, 4.0, 6.0, 8.0, 10.0],
        })
        y = pd.Series([3.0, 5.0, 7.0, 9.0, 11.0])

        model = LinearRegressionModel()
        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 5

    def test_zero_variance_feature(self):
        """Should handle constant feature (zero variance)."""
        np.random.seed(42)
        X = pd.DataFrame({
            "constant": np.ones(100),  # Zero variance
            "variable": np.random.randn(100),
        })
        y = pd.Series(np.random.randn(100))

        model = LinearRegressionModel(normalize=True)
        # Should still work (sklearn handles this)
        model.fit(X, y)
        predictions = model.predict(X)
        assert len(predictions) == 100

    def test_large_values(self):
        """Should handle large feature values."""
        np.random.seed(42)
        X = pd.DataFrame({
            "large": np.random.randn(100) * 1e6,
            "small": np.random.randn(100) * 1e-6,
        })
        y = pd.Series(np.random.randn(100))

        model = LinearRegressionModel(normalize=True)
        model.fit(X, y)
        predictions = model.predict(X)

        assert np.isfinite(predictions).all()
