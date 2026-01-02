"""U-Net architecture for spatial snowfall prediction.

This module provides a U-Net implementation for predicting snow depth across
spatial grids using gridded weather data (ERA5, HRRR).

U-Net is a convolutional neural network originally designed for biomedical
image segmentation. For snow prediction:
- Input: Gridded weather features (batch, channels, height, width)
- Output: Snow depth prediction grid (batch, 1, height, width)

Architecture:
- 4-level encoder-decoder structure
- Skip connections between encoder and decoder at each level
- Configurable base channel width (default 64)

References:
- Ronneberger et al. (2015): U-Net: Convolutional Networks for Biomedical
  Image Segmentation. https://arxiv.org/abs/1505.04597
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class UNetBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU.

    Each block consists of two 3x3 convolutions, each followed by
    batch normalization and ReLU activation.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels

    Example:
        >>> block = UNetBlock(64, 128)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> out = block(x)
        >>> out.shape
        torch.Size([2, 128, 32, 32])
    """

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize UNetBlock.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the block.

        Args:
            x: Input tensor of shape (batch, in_channels, height, width)

        Returns:
            Output tensor of shape (batch, out_channels, height, width)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class UNetEncoder(nn.Module):
    """Encoder (downsampling) path of U-Net.

    The encoder progressively reduces spatial dimensions while increasing
    channel depth. Each level consists of a UNetBlock followed by max pooling.

    Args:
        in_channels: Number of input channels
        base_channels: Number of channels in first level (default 64)
        n_levels: Number of encoder levels (default 4)

    Example:
        >>> encoder = UNetEncoder(8, base_channels=64, n_levels=4)
        >>> x = torch.randn(2, 8, 64, 64)
        >>> features, bottleneck = encoder(x)
        >>> len(features)
        4
        >>> bottleneck.shape
        torch.Size([2, 512, 4, 4])
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        n_levels: int = 4,
    ):
        """Initialize UNetEncoder.

        Args:
            in_channels: Number of input channels (weather features)
            base_channels: Number of channels in first level
            n_levels: Number of encoder levels
        """
        super().__init__()

        self.n_levels = n_levels
        self.base_channels = base_channels

        # Create encoder blocks
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        current_channels = in_channels
        for i in range(n_levels):
            out_channels = base_channels * (2**i)
            self.blocks.append(UNetBlock(current_channels, out_channels))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_channels = out_channels

        # Bottleneck (bottom of the U)
        self.bottleneck = UNetBlock(
            current_channels, base_channels * (2**n_levels)
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Forward pass through encoder.

        Args:
            x: Input tensor of shape (batch, in_channels, height, width)

        Returns:
            Tuple of:
                - List of feature maps from each level (for skip connections)
                - Bottleneck features
        """
        features = []

        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            features.append(x)  # Save for skip connection
            x = pool(x)

        x = self.bottleneck(x)

        return features, x


class UNetDecoder(nn.Module):
    """Decoder (upsampling) path of U-Net with skip connections.

    The decoder progressively increases spatial dimensions while decreasing
    channel depth. Skip connections from the encoder are concatenated at
    each level before the convolutional block.

    Args:
        base_channels: Number of channels in last decoder level
        n_levels: Number of decoder levels (should match encoder)

    Example:
        >>> decoder = UNetDecoder(base_channels=64, n_levels=4)
        >>> # Bottleneck features
        >>> bottleneck = torch.randn(2, 512, 4, 4)
        >>> # Skip connection features from encoder
        >>> skips = [torch.randn(2, 64, 64, 64), torch.randn(2, 128, 32, 32),
        ...          torch.randn(2, 256, 16, 16), torch.randn(2, 512, 8, 8)]
        >>> out = decoder(bottleneck, skips[::-1])  # Reverse order
        >>> out.shape
        torch.Size([2, 64, 64, 64])
    """

    def __init__(
        self,
        base_channels: int = 64,
        n_levels: int = 4,
    ):
        """Initialize UNetDecoder.

        Args:
            base_channels: Number of channels in final decoder level
            n_levels: Number of decoder levels
        """
        super().__init__()

        self.n_levels = n_levels
        self.base_channels = base_channels

        # Create decoder blocks and upsampling layers
        self.upconvs = nn.ModuleList()
        self.blocks = nn.ModuleList()

        for i in range(n_levels - 1, -1, -1):
            # Channels coming from previous level (or bottleneck)
            if i == n_levels - 1:
                in_channels = base_channels * (2 ** (n_levels))
            else:
                in_channels = base_channels * (2 ** (i + 1))

            out_channels = base_channels * (2**i)

            # Transposed convolution for upsampling
            self.upconvs.append(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=2, stride=2
                )
            )
            # Block takes concatenated features (skip + upsampled)
            self.blocks.append(
                UNetBlock(out_channels * 2, out_channels)
            )

    def forward(
        self, x: torch.Tensor, skip_features: list[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass through decoder with skip connections.

        Args:
            x: Bottleneck features from encoder
            skip_features: List of feature maps from encoder (in reverse order,
                i.e., deepest level first)

        Returns:
            Output features of shape (batch, base_channels, height, width)
        """
        for upconv, block, skip in zip(
            self.upconvs, self.blocks, skip_features
        ):
            x = upconv(x)

            # Handle size mismatches from non-power-of-2 inputs
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )

            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        return x


class SnowUNet(nn.Module):
    """U-Net model for spatial snow depth prediction.

    A complete U-Net architecture that takes gridded weather data and
    produces snow depth predictions for each grid cell.

    Args:
        in_channels: Number of input channels (weather features)
        out_channels: Number of output channels (default 1 for snow depth)
        base_channels: Number of channels in first encoder level (default 64)
        n_levels: Number of encoder/decoder levels (default 4)

    Input shape:
        (batch, in_channels, height, width)

    Output shape:
        (batch, out_channels, height, width)

    Note:
        For best results, input height and width should be divisible by
        2^n_levels (e.g., 16, 32, 64, 128 for n_levels=4).

    Example:
        >>> model = SnowUNet(in_channels=8, out_channels=1)
        >>> x = torch.randn(4, 8, 64, 64)  # 8 weather features
        >>> predictions = model(x)
        >>> predictions.shape
        torch.Size([4, 1, 64, 64])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        base_channels: int = 64,
        n_levels: int = 4,
    ):
        """Initialize SnowUNet.

        Args:
            in_channels: Number of input feature channels
            out_channels: Number of output channels (1 for snow depth)
            base_channels: Number of channels in first encoder level
            n_levels: Number of encoder/decoder levels
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.n_levels = n_levels

        # Encoder
        self.encoder = UNetEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            n_levels=n_levels,
        )

        # Decoder
        self.decoder = UNetDecoder(
            base_channels=base_channels,
            n_levels=n_levels,
        )

        # Final 1x1 convolution to produce output channels
        self.final_conv = nn.Conv2d(
            base_channels, out_channels, kernel_size=1
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the U-Net.

        Args:
            x: Input tensor of shape (batch, in_channels, height, width)

        Returns:
            Output tensor of shape (batch, out_channels, height, width)
        """
        # Encode
        skip_features, bottleneck = self.encoder(x)

        # Decode with skip connections (reverse order)
        x = self.decoder(bottleneck, skip_features[::-1])

        # Final convolution
        x = self.final_conv(x)

        return x

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """String representation of the model."""
        n_params = self.get_num_parameters()
        return (
            f"SnowUNet(in={self.in_channels}, out={self.out_channels}, "
            f"base={self.base_channels}, levels={self.n_levels}, "
            f"params={n_params:,})"
        )


class SpatialModel:
    """Wrapper class for spatial prediction models with BaseModel-like interface.

    Provides a consistent interface for training and prediction that matches
    the pattern used by other models in the snowforecast package.

    Args:
        model: SnowUNet instance or similar spatial model
        device: Torch device for computation (default: auto-detect)
        learning_rate: Learning rate for optimizer (default: 1e-4)
        weight_decay: L2 regularization weight (default: 1e-5)

    Example:
        >>> unet = SnowUNet(in_channels=8)
        >>> wrapper = SpatialModel(unet)
        >>> wrapper.fit(X_train, y_train, epochs=100)
        >>> predictions = wrapper.predict(X_test)
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        in_channels: int = 8,
        device: Optional[torch.device] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        """Initialize SpatialModel wrapper.

        Args:
            model: Pre-built model, or None to create default SnowUNet
            in_channels: Number of input channels (used if model is None)
            device: Torch device (CPU/GPU)
            learning_rate: Optimizer learning rate
            weight_decay: L2 regularization
        """
        # Auto-detect device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Create model if not provided
        if model is None:
            model = SnowUNet(in_channels=in_channels)
        self.model = model.to(self.device)

        # Training parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # State
        self.is_fitted = False
        self.train_history: list[dict] = []

    def fit(
        self,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        epochs: int = 100,
        batch_size: int = 16,
        eval_set: Optional[tuple] = None,
        early_stopping_rounds: int = 10,
        verbose: bool = True,
    ) -> "SpatialModel":
        """Train the model.

        Args:
            X: Input data of shape (n_samples, channels, height, width)
            y: Target data of shape (n_samples, 1, height, width)
            epochs: Number of training epochs
            batch_size: Batch size for training
            eval_set: Optional (X_val, y_val) for validation
            early_stopping_rounds: Stop if no improvement for this many epochs
            verbose: Whether to print training progress

        Returns:
            self for method chaining
        """
        # Convert to tensors
        X = self._to_tensor(X)
        y = self._to_tensor(y)

        # Create dataset and loader
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Validation set
        X_val, y_val = None, None
        if eval_set is not None:
            X_val = self._to_tensor(eval_set[0])
            y_val = self._to_tensor(eval_set[1])

        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Training loop
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_X)

            train_loss /= len(dataset)

            # Validation phase
            val_loss = None
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_dev = X_val.to(self.device)
                    y_val_dev = y_val.to(self.device)
                    val_predictions = self.model(X_val_dev)
                    val_loss = criterion(val_predictions, y_val_dev).item()

                scheduler.step(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= early_stopping_rounds:
                    if verbose:
                        logger.info(
                            f"Early stopping at epoch {epoch + 1} "
                            f"(no improvement for {early_stopping_rounds} epochs)"
                        )
                    break

            # Record history
            history_entry = {"epoch": epoch + 1, "train_loss": train_loss}
            if val_loss is not None:
                history_entry["val_loss"] = val_loss
            self.train_history.append(history_entry)

            # Logging
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}"
                if val_loss is not None:
                    msg += f", val_loss={val_loss:.4f}"
                logger.info(msg)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Make predictions.

        Args:
            X: Input data of shape (n_samples, channels, height, width)

        Returns:
            Predictions of shape (n_samples, 1, height, width)
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

        X = self._to_tensor(X)

        self.model.eval()
        with torch.no_grad():
            X_dev = X.to(self.device)
            predictions = self.model(X_dev)

        return predictions.cpu().numpy()

    def _to_tensor(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convert data to torch tensor."""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        return data.float()

    def save(self, path: str | Path) -> None:
        """Save model to disk.

        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Cannot save unfitted model.")

        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".pt")

        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "in_channels": self.model.in_channels,
                "out_channels": self.model.out_channels,
                "base_channels": self.model.base_channels,
                "n_levels": self.model.n_levels,
            },
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "train_history": self.train_history,
            "is_fitted": self.is_fitted,
        }

        torch.save(state, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path, device: Optional[torch.device] = None) -> "SpatialModel":
        """Load model from disk.

        Args:
            path: Path to the saved model
            device: Device to load model onto

        Returns:
            Loaded SpatialModel instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state = torch.load(path, map_location=device)

        # Recreate model
        model = SnowUNet(**state["model_config"])
        model.load_state_dict(state["model_state_dict"])

        # Create wrapper
        wrapper = cls(
            model=model,
            device=device,
            learning_rate=state["learning_rate"],
            weight_decay=state["weight_decay"],
        )
        wrapper.train_history = state["train_history"]
        wrapper.is_fitted = state["is_fitted"]

        logger.info(f"Model loaded from {path}")
        return wrapper

    def get_params(self) -> dict:
        """Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        return {
            "in_channels": self.model.in_channels,
            "out_channels": self.model.out_channels,
            "base_channels": self.model.base_channels,
            "n_levels": self.model.n_levels,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "is_fitted": self.is_fitted,
            "num_parameters": self.model.get_num_parameters(),
        }

    def __repr__(self) -> str:
        """String representation."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"SpatialModel({self.model}, {status})"
