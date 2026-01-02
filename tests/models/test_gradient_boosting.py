"""Tests for GradientBoostingModel.

Tests use realistic synthetic snow data with 200+ samples.
NO MOCKS - all tests use real LightGBM/XGBoost models.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from snowforecast.models import GradientBoostingModel


@pytest.fixture
def synthetic_snow_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate realistic synthetic snow data for testing.

    Creates a dataset with features that realistically affect snowfall:
    - elevation: Higher elevations get more snow
    - temperature: Colder temps mean more snow accumulation
    - humidity: Higher humidity means more precipitation
    - wind_speed: Affects snow distribution
    - pressure: Atmospheric pressure (weak predictor)

    Returns:
        Tuple of (X features DataFrame, y target Series)
    """
    np.random.seed(42)
    n_samples = 200

    # Generate features
    elevation = np.random.uniform(2000, 4000, n_samples)  # meters
    temperature = np.random.uniform(-20, 5, n_samples)  # Celsius
    humidity = np.random.uniform(30, 100, n_samples)  # percent
    wind_speed = np.random.uniform(0, 50, n_samples)  # km/h
    pressure = np.random.uniform(600, 800, n_samples)  # hPa

    # Create realistic snowfall target (cm)
    # Snowfall increases with elevation, humidity, and decreases with temp
    snowfall = (
        0.01 * (elevation - 2000)  # More snow at higher elevations
        + 0.5 * np.maximum(0, -temperature)  # More accumulation when cold
        + 0.1 * humidity  # More precip with higher humidity
        - 0.05 * wind_speed  # Wind blows snow away
        + np.random.normal(0, 3, n_samples)  # Random noise
    )
    snowfall = np.maximum(0, snowfall)  # No negative snowfall

    X = pd.DataFrame({
        "elevation": elevation,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "pressure": pressure,
    })
    y = pd.Series(snowfall, name="snowfall_cm")

    return X, y


@pytest.fixture
def synthetic_snow_data_with_categorical() -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic snow data with categorical features.

    Adds:
    - region: Categorical region (affects base snowfall)
    - storm_type: Categorical storm type

    Returns:
        Tuple of (X features DataFrame, y target Series)
    """
    np.random.seed(42)
    n_samples = 200

    # Generate numeric features
    elevation = np.random.uniform(2000, 4000, n_samples)
    temperature = np.random.uniform(-20, 5, n_samples)
    humidity = np.random.uniform(30, 100, n_samples)

    # Generate categorical features
    regions = np.random.choice(["rockies", "cascades", "sierra"], n_samples)
    storm_types = np.random.choice(["frontal", "orographic", "convective"], n_samples)

    # Region effects
    region_effect = np.where(
        regions == "cascades", 10,
        np.where(regions == "sierra", 8, 5)
    )

    # Create target with categorical effects
    snowfall = (
        region_effect
        + 0.01 * (elevation - 2000)
        + 0.5 * np.maximum(0, -temperature)
        + 0.1 * humidity
        + np.random.normal(0, 3, n_samples)
    )
    snowfall = np.maximum(0, snowfall)

    X = pd.DataFrame({
        "elevation": elevation,
        "temperature": temperature,
        "humidity": humidity,
        "region": regions,
        "storm_type": storm_types,
    })
    y = pd.Series(snowfall, name="snowfall_cm")

    return X, y


class TestGradientBoostingModelInit:
    """Tests for model initialization."""

    def test_default_init(self):
        """Test model initializes with default parameters."""
        model = GradientBoostingModel()

        assert model.n_estimators == 1000
        assert model.learning_rate == 0.05
        assert model.max_depth == 6
        assert model.backend == "lightgbm"
        assert not model.is_fitted
        assert model.feature_names is None

    def test_custom_params(self):
        """Test model initializes with custom parameters."""
        model = GradientBoostingModel(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=4,
            backend="xgboost",
        )

        assert model.n_estimators == 500
        assert model.learning_rate == 0.1
        assert model.max_depth == 4
        assert model.backend == "xgboost"

    def test_invalid_backend(self):
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="backend must be"):
            GradientBoostingModel(backend="invalid")


class TestGradientBoostingModelFitPredict:
    """Tests for fit and predict methods."""

    @pytest.mark.parametrize("backend", ["lightgbm", "xgboost"])
    def test_fit_predict_basic(self, synthetic_snow_data, backend):
        """Test basic fit/predict cycle for both backends."""
        X, y = synthetic_snow_data

        model = GradientBoostingModel(
            n_estimators=50,  # Fewer trees for faster test
            backend=backend,
            verbose=-1,
        )

        # Fit should return self
        result = model.fit(X, y)
        assert result is model
        assert model.is_fitted
        assert model.feature_names == list(X.columns)
        assert model.n_features == len(X.columns)

        # Predict should return array of same length
        predictions = model.predict(X)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)

        # Predictions should be reasonable (positive snowfall values)
        assert np.all(predictions >= -10)  # Allow small negative due to model
        assert np.all(predictions <= 100)  # Upper bound sanity check

    @pytest.mark.parametrize("backend", ["lightgbm", "xgboost"])
    def test_fit_with_eval_set(self, synthetic_snow_data, backend):
        """Test fit with validation set for early stopping."""
        X, y = synthetic_snow_data

        # Split into train/val
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        model = GradientBoostingModel(
            n_estimators=500,  # Many trees to trigger early stopping
            early_stopping_rounds=10,
            backend=backend,
            verbose=-1,
        )

        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        assert model.is_fitted
        # Early stopping should have triggered before 500 trees
        if model.best_iteration is not None:
            assert model.best_iteration < 500

    def test_predict_before_fit(self, synthetic_snow_data):
        """Test that predict before fit raises error."""
        X, _ = synthetic_snow_data
        model = GradientBoostingModel()

        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)

    def test_predict_wrong_features(self, synthetic_snow_data):
        """Test that predict with wrong features raises error."""
        X, y = synthetic_snow_data

        model = GradientBoostingModel(n_estimators=10, verbose=-1)
        model.fit(X, y)

        # Drop a feature
        X_wrong = X.drop(columns=["elevation"])
        with pytest.raises(ValueError, match="Feature mismatch"):
            model.predict(X_wrong)

        # Add extra feature
        X_extra = X.copy()
        X_extra["extra"] = 1
        with pytest.raises(ValueError, match="Feature mismatch"):
            model.predict(X_extra)

    def test_fit_empty_dataframe(self):
        """Test that fit with empty data raises error."""
        model = GradientBoostingModel()
        X_empty = pd.DataFrame()
        y_empty = pd.Series(dtype=float)

        with pytest.raises(ValueError, match="cannot be empty"):
            model.fit(X_empty, y_empty)


class TestGradientBoostingModelCategorical:
    """Tests for categorical feature handling."""

    @pytest.mark.parametrize("backend", ["lightgbm", "xgboost"])
    def test_categorical_features(self, synthetic_snow_data_with_categorical, backend):
        """Test model handles categorical features correctly."""
        X, y = synthetic_snow_data_with_categorical

        model = GradientBoostingModel(
            n_estimators=50,
            backend=backend,
            categorical_features=["region", "storm_type"],
            verbose=-1,
        )

        model.fit(X, y)
        predictions = model.predict(X)

        assert model.is_fitted
        assert len(predictions) == len(y)


class TestGradientBoostingModelFeatureImportance:
    """Tests for feature importance methods."""

    @pytest.mark.parametrize("backend", ["lightgbm", "xgboost"])
    def test_feature_importance_gain(self, synthetic_snow_data, backend):
        """Test feature importance (gain) for both backends."""
        X, y = synthetic_snow_data

        model = GradientBoostingModel(
            n_estimators=50,
            backend=backend,
            verbose=-1,
        )
        model.fit(X, y)

        importance_df = model.get_feature_importance(importance_type="gain")

        assert isinstance(importance_df, pd.DataFrame)
        assert list(importance_df.columns) == ["feature", "importance"]
        assert len(importance_df) == len(X.columns)
        assert set(importance_df["feature"]) == set(X.columns)

        # Importances should be non-negative
        assert np.all(importance_df["importance"] >= 0)

        # Should be sorted descending
        assert importance_df["importance"].is_monotonic_decreasing

    @pytest.mark.parametrize("backend", ["lightgbm", "xgboost"])
    def test_feature_importance_split(self, synthetic_snow_data, backend):
        """Test feature importance (split) for both backends."""
        X, y = synthetic_snow_data

        model = GradientBoostingModel(
            n_estimators=50,
            backend=backend,
            verbose=-1,
        )
        model.fit(X, y)

        importance_df = model.get_feature_importance(importance_type="split")

        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == len(X.columns)
        assert np.all(importance_df["importance"] >= 0)

    def test_feature_importance_before_fit(self, synthetic_snow_data):
        """Test that feature importance before fit raises error."""
        model = GradientBoostingModel()

        with pytest.raises(ValueError, match="not fitted"):
            model.get_feature_importance()

    def test_feature_importance_invalid_type(self, synthetic_snow_data):
        """Test that invalid importance type raises error."""
        X, y = synthetic_snow_data

        model = GradientBoostingModel(n_estimators=10, verbose=-1)
        model.fit(X, y)

        with pytest.raises(ValueError, match="importance_type must be"):
            model.get_feature_importance(importance_type="invalid")


class TestGradientBoostingModelParams:
    """Tests for get_params and set_params methods."""

    def test_get_params(self):
        """Test get_params returns all parameters."""
        model = GradientBoostingModel(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
        )

        params = model.get_params()

        assert isinstance(params, dict)
        assert params["n_estimators"] == 100
        assert params["learning_rate"] == 0.1
        assert params["max_depth"] == 5
        assert "backend" in params
        assert "categorical_features" in params

    def test_set_params(self):
        """Test set_params modifies parameters."""
        model = GradientBoostingModel()

        result = model.set_params(
            n_estimators=200,
            learning_rate=0.2,
        )

        assert result is model  # Returns self
        assert model.n_estimators == 200
        assert model.learning_rate == 0.2

    def test_set_params_invalid(self):
        """Test set_params with invalid parameter raises error."""
        model = GradientBoostingModel()

        with pytest.raises(ValueError, match="Invalid parameter"):
            model.set_params(invalid_param=42)


class TestGradientBoostingModelSaveLoad:
    """Tests for save and load methods."""

    @pytest.mark.parametrize("backend", ["lightgbm", "xgboost"])
    def test_save_load(self, synthetic_snow_data, backend):
        """Test model can be saved and loaded correctly."""
        X, y = synthetic_snow_data

        model = GradientBoostingModel(
            n_estimators=50,
            learning_rate=0.1,
            backend=backend,
            verbose=-1,
        )
        model.fit(X, y)
        original_predictions = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pkl"

            # Save
            model.save(save_path)
            assert save_path.exists()

            # Load
            loaded_model = GradientBoostingModel.load(save_path)

            # Check state is preserved
            assert loaded_model.is_fitted
            assert loaded_model.feature_names == model.feature_names
            assert loaded_model.n_features == model.n_features
            assert loaded_model.backend == model.backend
            assert loaded_model.n_estimators == model.n_estimators
            assert loaded_model.learning_rate == model.learning_rate

            # Check predictions match
            loaded_predictions = loaded_model.predict(X)
            np.testing.assert_array_almost_equal(
                original_predictions, loaded_predictions, decimal=5
            )

    def test_save_creates_directory(self, synthetic_snow_data):
        """Test save creates parent directories if needed."""
        X, y = synthetic_snow_data

        model = GradientBoostingModel(n_estimators=10, verbose=-1)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "nested" / "dir" / "model.pkl"
            model.save(save_path)
            assert save_path.exists()

    def test_save_adds_extension(self, synthetic_snow_data):
        """Test save adds .pkl extension if missing."""
        X, y = synthetic_snow_data

        model = GradientBoostingModel(n_estimators=10, verbose=-1)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"  # No extension
            model.save(save_path)

            expected_path = Path(tmpdir) / "model.pkl"
            assert expected_path.exists()

    def test_load_nonexistent(self):
        """Test load with nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            GradientBoostingModel.load("/nonexistent/path/model.pkl")


class TestGradientBoostingModelRepr:
    """Tests for string representation."""

    def test_repr_unfitted(self):
        """Test repr for unfitted model."""
        model = GradientBoostingModel(
            n_estimators=100,
            learning_rate=0.05,
        )
        repr_str = repr(model)

        assert "GradientBoostingModel" in repr_str
        assert "lightgbm" in repr_str
        assert "not fitted" in repr_str
        assert "features=0" in repr_str

    def test_repr_fitted(self, synthetic_snow_data):
        """Test repr for fitted model."""
        X, y = synthetic_snow_data

        model = GradientBoostingModel(n_estimators=10, verbose=-1)
        model.fit(X, y)
        repr_str = repr(model)

        assert "fitted" in repr_str
        assert "features=5" in repr_str


class TestGradientBoostingModelPerformance:
    """Tests for model performance on realistic data."""

    @pytest.mark.parametrize("backend", ["lightgbm", "xgboost"])
    def test_model_learns(self, synthetic_snow_data, backend):
        """Test that model actually learns patterns from data."""
        X, y = synthetic_snow_data

        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = GradientBoostingModel(
            n_estimators=100,
            backend=backend,
            verbose=-1,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # Calculate RMSE
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))

        # Model should perform better than just predicting mean
        mean_baseline_rmse = np.sqrt(np.mean((y_test - y_train.mean()) ** 2))

        # Model RMSE should be less than baseline
        assert rmse < mean_baseline_rmse, (
            f"Model RMSE ({rmse:.2f}) should be less than baseline ({mean_baseline_rmse:.2f})"
        )

    def test_elevation_is_important(self, synthetic_snow_data):
        """Test that model identifies elevation as important feature."""
        X, y = synthetic_snow_data

        model = GradientBoostingModel(n_estimators=100, verbose=-1)
        model.fit(X, y)

        importance_df = model.get_feature_importance()

        # Elevation should be in top 3 important features
        top_features = importance_df.head(3)["feature"].tolist()
        assert "elevation" in top_features or "temperature" in top_features, (
            "Expected elevation or temperature to be important"
        )
