"""Tests for BaseModel abstract class.

This module tests the BaseModel interface and validation methods
using a concrete test implementation.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from snowforecast.models.base import BaseModel


# =============================================================================
# Concrete test implementation
# =============================================================================


class ConcreteModel(BaseModel):
    """Minimal concrete implementation for testing BaseModel."""

    def fit(self, X, y, eval_set=None):
        self._validate_X(X)
        self._validate_y(y)
        self.feature_names = list(X.columns)
        self.model = {"mean": y.mean()}  # Simple mock
        self.is_fitted = True
        return self

    def predict(self, X):
        self._validate_fitted()
        self._validate_X(X)
        return np.full(len(X), self.model["mean"])

    def get_feature_importance(self):
        self._validate_fitted()
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": [1.0 / len(self.feature_names)] * len(self.feature_names),
        })

    def get_params(self):
        return {"fitted": self.is_fitted}

    def set_params(self, **params):
        return self

    def save(self, path):
        self._validate_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("mock")

    @classmethod
    def load(cls, path):
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        model = cls()
        model.is_fitted = True
        model.feature_names = ["mock_feature"]
        model.model = {"mean": 0.0}
        return model


# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture
def sample_X():
    """Sample feature DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        "elevation_m": np.random.uniform(1500, 3500, 100),
        "temperature_c": np.random.uniform(-20, 5, 100),
        "humidity_pct": np.random.uniform(40, 100, 100),
        "wind_speed_mps": np.random.uniform(0, 20, 100),
    })


@pytest.fixture
def sample_y():
    """Sample target Series (snow depth in cm)."""
    np.random.seed(42)
    return pd.Series(np.random.uniform(0, 200, 100), name="snow_depth_cm")


@pytest.fixture
def concrete_model():
    """Unfitted concrete model instance."""
    return ConcreteModel()


@pytest.fixture
def fitted_model(sample_X, sample_y):
    """Fitted concrete model instance."""
    model = ConcreteModel()
    model.fit(sample_X, sample_y)
    return model


# =============================================================================
# Test initialization
# =============================================================================


class TestBaseModelInit:
    """Tests for BaseModel initialization."""

    def test_init_attributes(self, concrete_model):
        """Should initialize with correct default attributes."""
        assert concrete_model.model is None
        assert concrete_model.feature_names is None
        assert concrete_model.is_fitted is False

    def test_repr_unfitted(self, concrete_model):
        """Should show unfitted status in repr."""
        repr_str = repr(concrete_model)
        assert "not fitted" in repr_str
        assert "features=0" in repr_str

    def test_repr_fitted(self, fitted_model):
        """Should show fitted status in repr."""
        repr_str = repr(fitted_model)
        assert "fitted" in repr_str
        assert "features=4" in repr_str


# =============================================================================
# Test validation methods
# =============================================================================


class TestValidateX:
    """Tests for _validate_X method."""

    def test_valid_dataframe(self, concrete_model, sample_X):
        """Should accept valid DataFrame."""
        # Should not raise
        concrete_model._validate_X(sample_X)

    def test_rejects_array(self, concrete_model, sample_X):
        """Should reject numpy array."""
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            concrete_model._validate_X(sample_X.values)

    def test_rejects_series(self, concrete_model, sample_y):
        """Should reject Series."""
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            concrete_model._validate_X(sample_y)

    def test_rejects_empty(self, concrete_model):
        """Should reject empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="cannot be empty"):
            concrete_model._validate_X(empty_df)


class TestValidateY:
    """Tests for _validate_y method."""

    def test_valid_series(self, concrete_model, sample_y):
        """Should accept valid Series."""
        # Should not raise
        concrete_model._validate_y(sample_y)

    def test_rejects_dataframe(self, concrete_model, sample_X):
        """Should reject DataFrame."""
        with pytest.raises(ValueError, match="must be a pandas Series"):
            concrete_model._validate_y(sample_X)

    def test_rejects_array(self, concrete_model, sample_y):
        """Should reject numpy array."""
        with pytest.raises(ValueError, match="must be a pandas Series"):
            concrete_model._validate_y(sample_y.values)

    def test_rejects_empty(self, concrete_model):
        """Should reject empty Series."""
        empty_series = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="cannot be empty"):
            concrete_model._validate_y(empty_series)


class TestValidateFitted:
    """Tests for _validate_fitted method."""

    def test_unfitted_raises(self, concrete_model):
        """Should raise when model not fitted."""
        with pytest.raises(ValueError, match="Model is not fitted"):
            concrete_model._validate_fitted()

    def test_fitted_passes(self, fitted_model):
        """Should pass when model is fitted."""
        # Should not raise
        fitted_model._validate_fitted()


# =============================================================================
# Test abstract method implementations
# =============================================================================


class TestFit:
    """Tests for fit method via concrete implementation."""

    def test_fit_returns_self(self, concrete_model, sample_X, sample_y):
        """Fit should return self for chaining."""
        result = concrete_model.fit(sample_X, sample_y)
        assert result is concrete_model

    def test_fit_sets_fitted(self, concrete_model, sample_X, sample_y):
        """Fit should set is_fitted to True."""
        concrete_model.fit(sample_X, sample_y)
        assert concrete_model.is_fitted is True

    def test_fit_stores_features(self, concrete_model, sample_X, sample_y):
        """Fit should store feature names."""
        concrete_model.fit(sample_X, sample_y)
        assert concrete_model.feature_names == list(sample_X.columns)


class TestPredict:
    """Tests for predict method via concrete implementation."""

    def test_predict_returns_array(self, fitted_model, sample_X):
        """Predict should return numpy array."""
        result = fitted_model.predict(sample_X)
        assert isinstance(result, np.ndarray)

    def test_predict_correct_length(self, fitted_model, sample_X):
        """Predict should return correct number of predictions."""
        result = fitted_model.predict(sample_X)
        assert len(result) == len(sample_X)

    def test_predict_unfitted_raises(self, concrete_model, sample_X):
        """Predict should raise when not fitted."""
        with pytest.raises(ValueError, match="not fitted"):
            concrete_model.predict(sample_X)


class TestGetFeatureImportance:
    """Tests for get_feature_importance method."""

    def test_returns_dataframe(self, fitted_model):
        """Should return DataFrame."""
        result = fitted_model.get_feature_importance()
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, fitted_model):
        """Should have feature and importance columns."""
        result = fitted_model.get_feature_importance()
        assert "feature" in result.columns
        assert "importance" in result.columns

    def test_correct_feature_count(self, fitted_model):
        """Should have one row per feature."""
        result = fitted_model.get_feature_importance()
        assert len(result) == len(fitted_model.feature_names)

    def test_unfitted_raises(self, concrete_model):
        """Should raise when not fitted."""
        with pytest.raises(ValueError, match="not fitted"):
            concrete_model.get_feature_importance()


# =============================================================================
# Test save/load
# =============================================================================


class TestSaveLoad:
    """Tests for save and load methods."""

    def test_save_creates_file(self, fitted_model, tmp_path):
        """Save should create model file."""
        model_path = tmp_path / "model.pkl"
        fitted_model.save(model_path)
        assert model_path.exists()

    def test_save_unfitted_raises(self, concrete_model, tmp_path):
        """Save should raise when not fitted."""
        model_path = tmp_path / "model.pkl"
        with pytest.raises(ValueError, match="not fitted"):
            concrete_model.save(model_path)

    def test_load_returns_model(self, fitted_model, tmp_path):
        """Load should return model instance."""
        model_path = tmp_path / "model.pkl"
        fitted_model.save(model_path)

        loaded = ConcreteModel.load(model_path)
        assert isinstance(loaded, ConcreteModel)
        assert loaded.is_fitted

    def test_load_nonexistent_raises(self, tmp_path):
        """Load should raise for nonexistent file."""
        model_path = tmp_path / "nonexistent.pkl"
        with pytest.raises(FileNotFoundError):
            ConcreteModel.load(model_path)

    def test_save_creates_parent_dirs(self, fitted_model, tmp_path):
        """Save should create parent directories."""
        model_path = tmp_path / "subdir" / "nested" / "model.pkl"
        fitted_model.save(model_path)
        assert model_path.exists()
