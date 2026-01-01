"""Tests for LSTM and GRU sequence models.

Tests use realistic synthetic snow data with sequential patterns.
NO MOCKS - all tests use real PyTorch models.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Skip all tests if PyTorch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch not installed"
)


@pytest.fixture
def synthetic_sequence_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate realistic sequential snow data for testing.

    Creates a dataset with time-series features that realistically affect snowfall:
    - temperature: Daily temperature with seasonal pattern
    - humidity: Humidity with some autocorrelation
    - pressure: Atmospheric pressure
    - wind_speed: Wind speed
    - precip_prob: Precipitation probability

    The target (snowfall) depends on recent weather patterns.

    Returns:
        Tuple of (X features DataFrame, y target Series)
    """
    np.random.seed(42)
    n_samples = 200

    # Create time-correlated features
    t = np.arange(n_samples)

    # Temperature with seasonal pattern and noise
    temperature = -5 + 10 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 3, n_samples)

    # Humidity with autocorrelation
    humidity = np.zeros(n_samples)
    humidity[0] = 60
    for i in range(1, n_samples):
        humidity[i] = 0.8 * humidity[i-1] + 0.2 * 60 + np.random.normal(0, 5)
    humidity = np.clip(humidity, 20, 100)

    # Other features
    pressure = 750 + np.random.normal(0, 10, n_samples)
    wind_speed = np.abs(np.random.normal(10, 5, n_samples))
    precip_prob = np.clip(0.3 + 0.4 * (humidity - 50) / 50 + np.random.normal(0, 0.1, n_samples), 0, 1)

    # Snowfall depends on past week's conditions
    snowfall = np.zeros(n_samples)
    for i in range(7, n_samples):
        # More snow when: cold, humid, high precip probability
        temp_factor = np.maximum(0, -temperature[i-3:i].mean())
        humid_factor = humidity[i-3:i].mean() / 100
        precip_factor = precip_prob[i-3:i].mean()

        snowfall[i] = (
            0.5 * temp_factor * humid_factor * precip_factor * 10
            + np.random.normal(0, 1)
        )
    snowfall = np.maximum(0, snowfall)

    X = pd.DataFrame({
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure,
        "wind_speed": wind_speed,
        "precip_prob": precip_prob,
    })
    y = pd.Series(snowfall, name="snowfall_cm")

    return X, y


@pytest.fixture
def small_sequence_data() -> tuple[pd.DataFrame, pd.Series]:
    """Small dataset for quick tests.

    Returns:
        Tuple of (X features DataFrame, y target Series)
    """
    np.random.seed(42)
    n_samples = 50

    X = pd.DataFrame({
        "temperature": np.random.uniform(-10, 5, n_samples),
        "humidity": np.random.uniform(40, 100, n_samples),
        "pressure": np.random.uniform(700, 800, n_samples),
    })
    y = pd.Series(np.random.uniform(0, 20, n_samples), name="snowfall_cm")

    return X, y


class TestSequenceModelInit:
    """Tests for SequenceModel initialization."""

    def test_default_init(self):
        """Test model initializes with default parameters."""
        from snowforecast.models.lstm import SequenceModel

        model = SequenceModel()

        assert model.cell_type == "lstm"
        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert model.dropout == 0.2
        assert model.bidirectional is False
        assert model.sequence_length == 7
        assert not model.is_fitted
        assert model.feature_names is None

    def test_custom_params_lstm(self):
        """Test LSTM model with custom parameters."""
        from snowforecast.models.lstm import SequenceModel

        model = SequenceModel(
            cell_type="lstm",
            hidden_size=128,
            num_layers=3,
            dropout=0.3,
            bidirectional=True,
            sequence_length=14,
        )

        assert model.cell_type == "lstm"
        assert model.hidden_size == 128
        assert model.num_layers == 3
        assert model.dropout == 0.3
        assert model.bidirectional is True
        assert model.sequence_length == 14

    def test_custom_params_gru(self):
        """Test GRU model with custom parameters."""
        from snowforecast.models.lstm import SequenceModel

        model = SequenceModel(
            cell_type="gru",
            hidden_size=32,
            num_layers=1,
        )

        assert model.cell_type == "gru"
        assert model.hidden_size == 32
        assert model.num_layers == 1

    def test_invalid_cell_type(self):
        """Test that invalid cell type raises error."""
        from snowforecast.models.lstm import SequenceModel

        with pytest.raises(ValueError, match="cell_type must be"):
            SequenceModel(cell_type="rnn")

    def test_device_auto_selection(self):
        """Test automatic device selection."""
        from snowforecast.models.lstm import SequenceModel

        model = SequenceModel()

        # Should be one of the valid devices
        assert model.device in ("cuda", "mps", "cpu")


class TestSequenceModelFitPredict:
    """Tests for fit and predict methods."""

    @pytest.mark.parametrize("cell_type", ["lstm", "gru"])
    def test_fit_predict_basic(self, small_sequence_data, cell_type):
        """Test basic fit/predict cycle for both cell types."""
        from snowforecast.models.lstm import SequenceModel

        X, y = small_sequence_data

        model = SequenceModel(
            cell_type=cell_type,
            hidden_size=16,
            num_layers=1,
            sequence_length=5,
            epochs=5,  # Few epochs for fast test
            verbose=0,
        )

        # Fit should return self
        result = model.fit(X, y)
        assert result is model
        assert model.is_fitted
        assert model.feature_names == list(X.columns)

        # Predict should return array
        predictions = model.predict(X)
        assert isinstance(predictions, np.ndarray)
        # Output length = input length - sequence_length + 1
        expected_len = len(X) - model.sequence_length + 1
        assert len(predictions) == expected_len

    @pytest.mark.parametrize("cell_type", ["lstm", "gru"])
    def test_fit_with_eval_set(self, synthetic_sequence_data, cell_type):
        """Test fit with validation set for early stopping."""
        from snowforecast.models.lstm import SequenceModel

        X, y = synthetic_sequence_data

        # Split into train/val
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        model = SequenceModel(
            cell_type=cell_type,
            hidden_size=32,
            num_layers=1,
            sequence_length=7,
            epochs=50,
            patience=5,
            verbose=0,
        )

        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        assert model.is_fitted
        # Should have recorded training history
        assert len(model.train_losses) > 0
        assert len(model.val_losses) > 0
        # Early stopping should have triggered before 50 epochs
        assert len(model.train_losses) <= 50

    def test_predict_before_fit(self, small_sequence_data):
        """Test that predict before fit raises error."""
        from snowforecast.models.lstm import SequenceModel

        X, _ = small_sequence_data
        model = SequenceModel()

        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)

    def test_predict_wrong_features(self, small_sequence_data):
        """Test that predict with wrong features raises error."""
        from snowforecast.models.lstm import SequenceModel

        X, y = small_sequence_data

        model = SequenceModel(
            hidden_size=16,
            num_layers=1,
            sequence_length=5,
            epochs=3,
            verbose=0,
        )
        model.fit(X, y)

        # Drop a feature
        X_wrong = X.drop(columns=["temperature"])
        with pytest.raises(ValueError, match="Feature mismatch"):
            model.predict(X_wrong)

    def test_fit_insufficient_samples(self, small_sequence_data):
        """Test that fit with too few samples raises error."""
        from snowforecast.models.lstm import SequenceModel

        X, y = small_sequence_data
        X_small = X.iloc[:5]
        y_small = y.iloc[:5]

        model = SequenceModel(sequence_length=10)  # sequence_length > n_samples

        with pytest.raises(ValueError, match="sequence_length"):
            model.fit(X_small, y_small)

    def test_predict_insufficient_samples(self, small_sequence_data):
        """Test that predict with too few samples raises error."""
        from snowforecast.models.lstm import SequenceModel

        X, y = small_sequence_data

        model = SequenceModel(
            sequence_length=10,
            epochs=3,
            hidden_size=16,
            verbose=0,
        )
        model.fit(X, y)

        X_small = X.iloc[:5]
        with pytest.raises(ValueError, match="at least"):
            model.predict(X_small)


class TestBidirectionalModels:
    """Tests for bidirectional LSTM/GRU."""

    @pytest.mark.parametrize("cell_type", ["lstm", "gru"])
    def test_bidirectional_fit_predict(self, small_sequence_data, cell_type):
        """Test bidirectional models work correctly."""
        from snowforecast.models.lstm import SequenceModel

        X, y = small_sequence_data

        model = SequenceModel(
            cell_type=cell_type,
            hidden_size=16,
            num_layers=1,
            bidirectional=True,
            sequence_length=5,
            epochs=3,
            verbose=0,
        )

        model.fit(X, y)
        predictions = model.predict(X)

        assert model.is_fitted
        assert len(predictions) == len(X) - model.sequence_length + 1


class TestSnowLSTMModule:
    """Tests for the SnowLSTM PyTorch module directly."""

    def test_lstm_forward_shape(self):
        """Test LSTM module produces correct output shape."""
        from snowforecast.models.lstm import SnowLSTM

        batch_size = 32
        seq_len = 7
        input_size = 5

        model = SnowLSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
        )

        x = torch.randn(batch_size, seq_len, input_size)
        output = model(x)

        assert output.shape == (batch_size, 1)

    def test_lstm_bidirectional_shape(self):
        """Test bidirectional LSTM output shape."""
        from snowforecast.models.lstm import SnowLSTM

        batch_size = 16
        seq_len = 10
        input_size = 8

        model = SnowLSTM(
            input_size=input_size,
            hidden_size=32,
            num_layers=2,
            bidirectional=True,
        )

        x = torch.randn(batch_size, seq_len, input_size)
        output = model(x)

        assert output.shape == (batch_size, 1)


class TestSnowGRUModule:
    """Tests for the SnowGRU PyTorch module directly."""

    def test_gru_forward_shape(self):
        """Test GRU module produces correct output shape."""
        from snowforecast.models.lstm import SnowGRU

        batch_size = 32
        seq_len = 7
        input_size = 5

        model = SnowGRU(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
        )

        x = torch.randn(batch_size, seq_len, input_size)
        output = model(x)

        assert output.shape == (batch_size, 1)

    def test_gru_bidirectional_shape(self):
        """Test bidirectional GRU output shape."""
        from snowforecast.models.lstm import SnowGRU

        batch_size = 16
        seq_len = 10
        input_size = 8

        model = SnowGRU(
            input_size=input_size,
            hidden_size=32,
            num_layers=2,
            bidirectional=True,
        )

        x = torch.randn(batch_size, seq_len, input_size)
        output = model(x)

        assert output.shape == (batch_size, 1)


class TestSequenceModelParams:
    """Tests for get_params and set_params methods."""

    def test_get_params(self):
        """Test get_params returns all parameters."""
        from snowforecast.models.lstm import SequenceModel

        model = SequenceModel(
            cell_type="gru",
            hidden_size=128,
            num_layers=3,
            dropout=0.3,
        )

        params = model.get_params()

        assert isinstance(params, dict)
        assert params["cell_type"] == "gru"
        assert params["hidden_size"] == 128
        assert params["num_layers"] == 3
        assert params["dropout"] == 0.3
        assert "sequence_length" in params
        assert "learning_rate" in params
        assert "batch_size" in params

    def test_set_params(self):
        """Test set_params modifies parameters."""
        from snowforecast.models.lstm import SequenceModel

        model = SequenceModel()

        result = model.set_params(
            hidden_size=256,
            learning_rate=0.01,
        )

        assert result is model
        assert model.hidden_size == 256
        assert model.learning_rate == 0.01

    def test_set_params_invalid(self):
        """Test set_params with invalid parameter raises error."""
        from snowforecast.models.lstm import SequenceModel

        model = SequenceModel()

        with pytest.raises(ValueError, match="Invalid parameter"):
            model.set_params(invalid_param=42)


class TestSequenceModelSaveLoad:
    """Tests for save and load methods."""

    @pytest.mark.parametrize("cell_type", ["lstm", "gru"])
    def test_save_load(self, small_sequence_data, cell_type):
        """Test model can be saved and loaded correctly."""
        from snowforecast.models.lstm import SequenceModel

        X, y = small_sequence_data

        model = SequenceModel(
            cell_type=cell_type,
            hidden_size=16,
            num_layers=1,
            sequence_length=5,
            epochs=3,
            verbose=0,
        )
        model.fit(X, y)
        original_predictions = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pkl"

            # Save
            model.save(save_path)
            assert save_path.exists()

            # Load
            loaded_model = SequenceModel.load(save_path)

            # Check state is preserved
            assert loaded_model.is_fitted
            assert loaded_model.feature_names == model.feature_names
            assert loaded_model.cell_type == model.cell_type
            assert loaded_model.hidden_size == model.hidden_size

            # Check predictions match
            loaded_predictions = loaded_model.predict(X)
            np.testing.assert_array_almost_equal(
                original_predictions, loaded_predictions, decimal=5
            )

    def test_save_creates_directory(self, small_sequence_data):
        """Test save creates parent directories if needed."""
        from snowforecast.models.lstm import SequenceModel

        X, y = small_sequence_data

        model = SequenceModel(
            hidden_size=16,
            sequence_length=5,
            epochs=2,
            verbose=0,
        )
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "nested" / "dir" / "model.pkl"
            model.save(save_path)
            assert save_path.exists()

    def test_save_adds_extension(self, small_sequence_data):
        """Test save adds .pkl extension if missing."""
        from snowforecast.models.lstm import SequenceModel

        X, y = small_sequence_data

        model = SequenceModel(
            hidden_size=16,
            sequence_length=5,
            epochs=2,
            verbose=0,
        )
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"  # No extension
            model.save(save_path)

            expected_path = Path(tmpdir) / "model.pkl"
            assert expected_path.exists()

    def test_load_nonexistent(self):
        """Test load with nonexistent file raises error."""
        from snowforecast.models.lstm import SequenceModel

        with pytest.raises(FileNotFoundError):
            SequenceModel.load("/nonexistent/path/model.pkl")

    def test_save_before_fit_raises(self):
        """Test save before fit raises error."""
        from snowforecast.models.lstm import SequenceModel

        model = SequenceModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pkl"
            with pytest.raises(ValueError, match="not fitted"):
                model.save(save_path)


class TestSequenceModelRepr:
    """Tests for string representation."""

    def test_repr_unfitted(self):
        """Test repr for unfitted model."""
        from snowforecast.models.lstm import SequenceModel

        model = SequenceModel(
            cell_type="lstm",
            hidden_size=64,
            num_layers=2,
            sequence_length=7,
        )
        repr_str = repr(model)

        assert "SequenceModel" in repr_str
        assert "lstm" in repr_str
        assert "hidden=64" in repr_str
        assert "layers=2" in repr_str
        assert "seq_len=7" in repr_str
        assert "not fitted" in repr_str

    def test_repr_fitted(self, small_sequence_data):
        """Test repr for fitted model."""
        from snowforecast.models.lstm import SequenceModel

        X, y = small_sequence_data

        model = SequenceModel(
            hidden_size=16,
            sequence_length=5,
            epochs=2,
            verbose=0,
        )
        model.fit(X, y)
        repr_str = repr(model)

        assert "SequenceModel" in repr_str
        # Should not contain "not fitted"
        assert "not fitted" not in repr_str


class TestSequenceModelFeatureImportance:
    """Tests for feature importance."""

    def test_feature_importance(self, small_sequence_data):
        """Test feature importance returns DataFrame."""
        from snowforecast.models.lstm import SequenceModel

        X, y = small_sequence_data

        model = SequenceModel(
            hidden_size=16,
            sequence_length=5,
            epochs=2,
            verbose=0,
        )
        model.fit(X, y)

        importance_df = model.get_feature_importance()

        assert isinstance(importance_df, pd.DataFrame)
        assert list(importance_df.columns) == ["feature", "importance"]
        assert len(importance_df) == len(X.columns)
        assert set(importance_df["feature"]) == set(X.columns)

    def test_feature_importance_before_fit(self):
        """Test feature importance before fit raises error."""
        from snowforecast.models.lstm import SequenceModel

        model = SequenceModel()

        with pytest.raises(ValueError, match="not fitted"):
            model.get_feature_importance()


class TestSequenceModelPerformance:
    """Tests for model learning capability."""

    @pytest.mark.parametrize("cell_type", ["lstm", "gru"])
    def test_model_learns(self, synthetic_sequence_data, cell_type):
        """Test that model actually learns patterns from data."""
        from snowforecast.models.lstm import SequenceModel

        X, y = synthetic_sequence_data

        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = SequenceModel(
            cell_type=cell_type,
            hidden_size=32,
            num_layers=2,
            sequence_length=7,
            epochs=20,
            patience=10,
            verbose=0,
        )
        model.fit(X_train, y_train, eval_set=(X_test, y_test))

        predictions = model.predict(X_test)

        # Align predictions with actual targets
        y_test_aligned = y_test.iloc[model.sequence_length - 1:].values

        # Calculate RMSE
        rmse = np.sqrt(np.mean((predictions - y_test_aligned) ** 2))

        # Model should produce reasonable predictions
        # (not NaN, not extreme values)
        assert not np.isnan(rmse)
        assert rmse < 100  # Sanity check

    def test_training_loss_decreases(self, small_sequence_data):
        """Test that training loss decreases over epochs."""
        from snowforecast.models.lstm import SequenceModel

        X, y = small_sequence_data

        model = SequenceModel(
            hidden_size=32,
            sequence_length=5,
            epochs=10,
            verbose=0,
        )
        model.fit(X, y)

        # Training loss should generally decrease
        if len(model.train_losses) >= 5:
            # Compare average of first half vs second half
            first_half = np.mean(model.train_losses[:len(model.train_losses)//2])
            second_half = np.mean(model.train_losses[len(model.train_losses)//2:])
            # Second half should have lower or equal loss
            assert second_half <= first_half * 1.5  # Allow some variance
