"""Tests for U-Net spatial prediction model.

Tests use real PyTorch models - NO MOCKS.
Uses small input sizes for fast testing while verifying architecture correctness.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from snowforecast.models.unet import (
    SnowUNet,
    SpatialModel,
    UNetBlock,
    UNetDecoder,
    UNetEncoder,
)

# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture
def small_input():
    """Small input tensor for fast testing (batch=2, channels=4, 32x32)."""
    return torch.randn(2, 4, 32, 32)


@pytest.fixture
def medium_input():
    """Medium input tensor (batch=4, channels=8, 64x64)."""
    return torch.randn(4, 8, 64, 64)


@pytest.fixture
def synthetic_spatial_data():
    """Generate synthetic spatial data for training tests.

    Creates a simple pattern where output is related to input channels.
    """
    np.random.seed(42)
    batch_size = 16
    n_channels = 4
    size = 32

    # Create input with spatial patterns
    X = np.random.randn(batch_size, n_channels, size, size).astype(np.float32)

    # Target is a weighted sum of input channels plus some noise
    # This gives the model something learnable
    weights = np.array([0.5, 0.3, -0.2, 0.1]).reshape(1, -1, 1, 1)
    y = np.sum(X * weights, axis=1, keepdims=True)
    y = y + np.random.randn(batch_size, 1, size, size).astype(np.float32) * 0.1

    return X, y


# =============================================================================
# UNetBlock tests
# =============================================================================


class TestUNetBlock:
    """Tests for UNetBlock component."""

    def test_output_shape(self, small_input):
        """Block should produce correct output shape."""
        block = UNetBlock(in_channels=4, out_channels=64)
        output = block(small_input)

        assert output.shape == (2, 64, 32, 32)

    def test_channel_change(self):
        """Block should handle channel increase/decrease."""
        x = torch.randn(2, 32, 16, 16)

        # Increase channels
        block_up = UNetBlock(32, 64)
        out_up = block_up(x)
        assert out_up.shape == (2, 64, 16, 16)

        # Decrease channels
        block_down = UNetBlock(32, 16)
        out_down = block_down(x)
        assert out_down.shape == (2, 16, 16, 16)

    def test_preserves_spatial_dims(self):
        """Block should preserve spatial dimensions."""
        for size in [16, 32, 64]:
            x = torch.randn(1, 8, size, size)
            block = UNetBlock(8, 16)
            output = block(x)
            assert output.shape[2:] == (size, size)

    def test_has_batchnorm(self):
        """Block should contain batch normalization layers."""
        block = UNetBlock(4, 8)
        has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in block.modules())
        assert has_bn, "Block should contain BatchNorm2d"

    def test_has_relu(self):
        """Block should contain ReLU activation."""
        block = UNetBlock(4, 8)
        has_relu = any(isinstance(m, torch.nn.ReLU) for m in block.modules())
        assert has_relu, "Block should contain ReLU"


# =============================================================================
# UNetEncoder tests
# =============================================================================


class TestUNetEncoder:
    """Tests for UNetEncoder component."""

    def test_output_types(self, small_input):
        """Encoder should return feature list and bottleneck."""
        encoder = UNetEncoder(in_channels=4, base_channels=16, n_levels=3)
        features, bottleneck = encoder(small_input)

        assert isinstance(features, list)
        assert isinstance(bottleneck, torch.Tensor)

    def test_feature_count(self, small_input):
        """Encoder should return correct number of feature maps."""
        for n_levels in [2, 3, 4]:
            encoder = UNetEncoder(in_channels=4, base_channels=16, n_levels=n_levels)
            features, _ = encoder(small_input)
            assert len(features) == n_levels

    def test_feature_shapes(self, medium_input):
        """Encoder features should have correct shapes at each level."""
        encoder = UNetEncoder(in_channels=8, base_channels=16, n_levels=4)
        features, bottleneck = encoder(medium_input)

        # Check feature shapes (spatial dims halve at each level)
        expected_shapes = [
            (4, 16, 64, 64),   # Level 0: base_channels, full size
            (4, 32, 32, 32),   # Level 1: 2x channels, half size
            (4, 64, 16, 16),   # Level 2: 4x channels, quarter size
            (4, 128, 8, 8),    # Level 3: 8x channels, 1/8 size
        ]

        for feat, expected in zip(features, expected_shapes):
            assert feat.shape == expected, f"Expected {expected}, got {feat.shape}"

        # Bottleneck should have max channels and smallest spatial dims
        assert bottleneck.shape == (4, 256, 4, 4)

    def test_channel_doubling(self, small_input):
        """Channels should double at each encoder level."""
        encoder = UNetEncoder(in_channels=4, base_channels=8, n_levels=3)
        features, bottleneck = encoder(small_input)

        expected_channels = [8, 16, 32]  # base * 2^i
        for feat, expected in zip(features, expected_channels):
            assert feat.shape[1] == expected

        # Bottleneck has base * 2^n_levels channels
        assert bottleneck.shape[1] == 64


# =============================================================================
# UNetDecoder tests
# =============================================================================


class TestUNetDecoder:
    """Tests for UNetDecoder component."""

    def test_output_shape(self):
        """Decoder should produce correct output shape."""
        decoder = UNetDecoder(base_channels=16, n_levels=3)

        # Simulate encoder outputs
        bottleneck = torch.randn(2, 128, 4, 4)  # base * 2^n_levels
        skip_features = [
            torch.randn(2, 64, 8, 8),   # Level 2 (reversed)
            torch.randn(2, 32, 16, 16), # Level 1
            torch.randn(2, 16, 32, 32), # Level 0
        ]

        output = decoder(bottleneck, skip_features)

        # Output should have base_channels and full spatial dims
        assert output.shape == (2, 16, 32, 32)

    def test_skip_connection_concatenation(self):
        """Decoder should use skip connections correctly."""
        decoder = UNetDecoder(base_channels=16, n_levels=2)

        bottleneck = torch.randn(2, 64, 8, 8)
        skip_features = [
            torch.randn(2, 32, 16, 16),
            torch.randn(2, 16, 32, 32),
        ]

        # Should not raise
        output = decoder(bottleneck, skip_features)
        assert output.shape[1] == 16  # base_channels

    def test_handles_size_mismatch(self):
        """Decoder should handle size mismatches from non-power-of-2 inputs."""
        decoder = UNetDecoder(base_channels=16, n_levels=2)

        # Bottleneck with odd dimensions
        bottleneck = torch.randn(2, 64, 7, 7)
        skip_features = [
            torch.randn(2, 32, 14, 14),
            torch.randn(2, 16, 28, 28),
        ]

        # Should not raise - uses interpolation
        output = decoder(bottleneck, skip_features)
        assert output.shape == (2, 16, 28, 28)


# =============================================================================
# SnowUNet tests
# =============================================================================


class TestSnowUNet:
    """Tests for complete SnowUNet model."""

    def test_forward_pass(self, small_input):
        """Model should produce correct output shape."""
        model = SnowUNet(in_channels=4, out_channels=1, base_channels=16, n_levels=3)
        output = model(small_input)

        assert output.shape == (2, 1, 32, 32)

    def test_default_output_channel(self, medium_input):
        """Default output should be single channel (snow depth)."""
        model = SnowUNet(in_channels=8, base_channels=16, n_levels=3)
        output = model(medium_input)

        assert output.shape[1] == 1  # Single output channel

    def test_multi_output_channels(self, small_input):
        """Model should support multiple output channels."""
        model = SnowUNet(in_channels=4, out_channels=3, base_channels=16, n_levels=3)
        output = model(small_input)

        assert output.shape == (2, 3, 32, 32)

    def test_configurable_base_channels(self, small_input):
        """Model should work with different base channel widths."""
        for base in [16, 32, 64]:
            model = SnowUNet(in_channels=4, base_channels=base, n_levels=2)
            output = model(small_input)
            assert output.shape == (2, 1, 32, 32)

    def test_configurable_levels(self, small_input):
        """Model should work with different number of levels."""
        for n_levels in [2, 3, 4]:
            model = SnowUNet(in_channels=4, base_channels=16, n_levels=n_levels)
            output = model(small_input)
            assert output.shape == (2, 1, 32, 32)

    def test_various_input_sizes(self):
        """Model should handle various power-of-2 input sizes."""
        model = SnowUNet(in_channels=4, base_channels=16, n_levels=3)

        for size in [16, 32, 64, 128]:
            x = torch.randn(1, 4, size, size)
            output = model(x)
            assert output.shape == (1, 1, size, size)

    def test_non_power_of_2_input(self):
        """Model should handle non-power-of-2 input sizes gracefully."""
        model = SnowUNet(in_channels=4, base_channels=16, n_levels=2)

        # Non-power-of-2 size
        x = torch.randn(1, 4, 30, 30)
        output = model(x)

        # Output should have same spatial dims as input
        assert output.shape == (1, 1, 30, 30)

    def test_gradient_flow(self, small_input):
        """Gradients should flow through entire model."""
        model = SnowUNet(in_channels=4, base_channels=16, n_levels=3)
        target = torch.randn(2, 1, 32, 32)

        output = model(small_input)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        # Check gradients exist for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_num_parameters(self):
        """Should report correct parameter count."""
        model = SnowUNet(in_channels=4, base_channels=16, n_levels=3)
        n_params = model.get_num_parameters()

        assert n_params > 0
        assert isinstance(n_params, int)

    def test_repr(self):
        """String representation should be informative."""
        model = SnowUNet(in_channels=8, out_channels=2, base_channels=32, n_levels=4)
        repr_str = repr(model)

        assert "SnowUNet" in repr_str
        assert "in=8" in repr_str
        assert "out=2" in repr_str
        assert "base=32" in repr_str
        assert "levels=4" in repr_str


# =============================================================================
# SpatialModel wrapper tests
# =============================================================================


class TestSpatialModel:
    """Tests for SpatialModel wrapper class."""

    def test_init_default(self):
        """Should initialize with default SnowUNet."""
        wrapper = SpatialModel(in_channels=4)

        assert wrapper.model is not None
        assert isinstance(wrapper.model, SnowUNet)
        assert not wrapper.is_fitted

    def test_init_with_model(self):
        """Should accept pre-built model."""
        model = SnowUNet(in_channels=8, base_channels=32)
        wrapper = SpatialModel(model=model)

        assert wrapper.model is model
        assert wrapper.model.base_channels == 32

    def test_fit_returns_self(self, synthetic_spatial_data):
        """Fit should return self for chaining."""
        X, y = synthetic_spatial_data
        wrapper = SpatialModel(in_channels=4)

        result = wrapper.fit(X, y, epochs=2, verbose=False)

        assert result is wrapper
        assert wrapper.is_fitted

    def test_fit_with_numpy(self, synthetic_spatial_data):
        """Fit should accept numpy arrays."""
        X, y = synthetic_spatial_data
        wrapper = SpatialModel(in_channels=4)

        wrapper.fit(X, y, epochs=2, verbose=False)

        assert wrapper.is_fitted

    def test_fit_with_tensors(self, synthetic_spatial_data):
        """Fit should accept torch tensors."""
        X, y = synthetic_spatial_data
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)

        wrapper = SpatialModel(in_channels=4)
        wrapper.fit(X_tensor, y_tensor, epochs=2, verbose=False)

        assert wrapper.is_fitted

    def test_fit_with_eval_set(self, synthetic_spatial_data):
        """Fit should accept validation set."""
        X, y = synthetic_spatial_data

        # Split into train/val
        n_train = len(X) - 4
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        wrapper = SpatialModel(in_channels=4)
        wrapper.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            epochs=5,
            verbose=False,
        )

        assert wrapper.is_fitted
        # Should have validation loss in history
        assert "val_loss" in wrapper.train_history[-1]

    def test_predict_returns_numpy(self, synthetic_spatial_data):
        """Predict should return numpy array."""
        X, y = synthetic_spatial_data
        wrapper = SpatialModel(in_channels=4)
        wrapper.fit(X, y, epochs=2, verbose=False)

        predictions = wrapper.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == y.shape

    def test_predict_before_fit_raises(self, synthetic_spatial_data):
        """Predict should raise if model not fitted."""
        X, _ = synthetic_spatial_data
        wrapper = SpatialModel(in_channels=4)

        with pytest.raises(ValueError, match="not fitted"):
            wrapper.predict(X)

    def test_training_history(self, synthetic_spatial_data):
        """Training should record history."""
        X, y = synthetic_spatial_data
        wrapper = SpatialModel(in_channels=4)
        wrapper.fit(X, y, epochs=5, verbose=False)

        assert len(wrapper.train_history) == 5
        assert all("epoch" in h and "train_loss" in h for h in wrapper.train_history)

    def test_early_stopping(self, synthetic_spatial_data):
        """Early stopping should work with validation set."""
        X, y = synthetic_spatial_data

        # Create validation set
        n_train = len(X) - 4
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        wrapper = SpatialModel(in_channels=4)
        wrapper.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            epochs=1000,  # Many epochs
            early_stopping_rounds=3,  # Early stopping
            verbose=False,
        )

        # Should have stopped early
        assert len(wrapper.train_history) < 1000

    def test_model_learns(self, synthetic_spatial_data):
        """Model should reduce loss during training."""
        X, y = synthetic_spatial_data
        wrapper = SpatialModel(in_channels=4)
        wrapper.fit(X, y, epochs=20, verbose=False)

        # Loss should decrease
        first_loss = wrapper.train_history[0]["train_loss"]
        last_loss = wrapper.train_history[-1]["train_loss"]

        assert last_loss < first_loss, "Model should reduce training loss"


class TestSpatialModelSaveLoad:
    """Tests for save and load methods."""

    def test_save_load(self, synthetic_spatial_data):
        """Model should save and load correctly."""
        X, y = synthetic_spatial_data

        wrapper = SpatialModel(in_channels=4)
        wrapper.fit(X, y, epochs=3, verbose=False)
        original_predictions = wrapper.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.pt"

            wrapper.save(save_path)
            assert save_path.exists()

            loaded_wrapper = SpatialModel.load(save_path)

            assert loaded_wrapper.is_fitted
            assert loaded_wrapper.model.in_channels == 4

            # Predictions should match
            loaded_predictions = loaded_wrapper.predict(X)
            np.testing.assert_array_almost_equal(
                original_predictions, loaded_predictions, decimal=5
            )

    def test_save_adds_extension(self, synthetic_spatial_data):
        """Save should add .pt extension if missing."""
        X, y = synthetic_spatial_data

        wrapper = SpatialModel(in_channels=4)
        wrapper.fit(X, y, epochs=2, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"  # No extension
            wrapper.save(save_path)

            expected_path = Path(tmpdir) / "model.pt"
            assert expected_path.exists()

    def test_save_creates_directory(self, synthetic_spatial_data):
        """Save should create parent directories."""
        X, y = synthetic_spatial_data

        wrapper = SpatialModel(in_channels=4)
        wrapper.fit(X, y, epochs=2, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "nested" / "dir" / "model.pt"
            wrapper.save(save_path)
            assert save_path.exists()

    def test_save_unfitted_raises(self):
        """Save should raise if model not fitted."""
        wrapper = SpatialModel(in_channels=4)

        with pytest.raises(ValueError, match="not fitted"):
            wrapper.save("/tmp/model.pt")

    def test_load_nonexistent_raises(self):
        """Load should raise for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            SpatialModel.load("/nonexistent/path/model.pt")


class TestSpatialModelParams:
    """Tests for parameter methods."""

    def test_get_params(self, synthetic_spatial_data):
        """get_params should return model parameters."""
        X, y = synthetic_spatial_data

        wrapper = SpatialModel(in_channels=4, learning_rate=0.001)
        wrapper.fit(X, y, epochs=2, verbose=False)

        params = wrapper.get_params()

        assert params["in_channels"] == 4
        assert params["learning_rate"] == 0.001
        assert params["is_fitted"] is True
        assert "num_parameters" in params

    def test_repr(self):
        """String representation should be informative."""
        wrapper = SpatialModel(in_channels=8)
        repr_str = repr(wrapper)

        assert "SpatialModel" in repr_str
        assert "not fitted" in repr_str


# =============================================================================
# Skip connection tests
# =============================================================================


class TestSkipConnections:
    """Specific tests for skip connection behavior."""

    def test_skip_features_used(self):
        """Skip connections should carry information from encoder to decoder."""
        model = SnowUNet(in_channels=4, base_channels=16, n_levels=3)

        # Create input with distinct pattern
        x = torch.randn(1, 4, 32, 32)

        # Get output with skip connections
        model(x)

        # Verify skip connections exist by checking encoder-decoder structure
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")

        # The encoder produces features at each level
        skip_features, bottleneck = model.encoder(x)
        assert len(skip_features) == 3

        # The decoder uses these features
        decoded = model.decoder(bottleneck, skip_features[::-1])
        assert decoded.shape == (1, 16, 32, 32)

    def test_skip_connection_gradient_flow(self):
        """Gradients should flow through skip connections."""
        model = SnowUNet(in_channels=4, base_channels=16, n_levels=3)

        x = torch.randn(1, 4, 32, 32)
        target = torch.randn(1, 1, 32, 32)

        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()

        # Check encoder block gradients (start of skip connections)
        for block in model.encoder.blocks:
            for param in block.parameters():
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()

        # Check decoder block gradients (use skip connections)
        for block in model.decoder.blocks:
            for param in block.parameters():
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()
