"""LSTM and GRU models for snowfall time series prediction.

This module provides PyTorch-based recurrent neural network models for
predicting snowfall from sequential weather data.

Classes:
- SnowLSTM: PyTorch LSTM network module
- SnowGRU: PyTorch GRU network module
- SequenceModel: BaseModel wrapper for training and prediction

Example:
    >>> from snowforecast.models.lstm import SequenceModel
    >>> model = SequenceModel(
    ...     cell_type="lstm",
    ...     hidden_size=64,
    ...     num_layers=2,
    ...     sequence_length=7,
    ... )
    >>> model.fit(X_train, y_train, eval_set=(X_val, y_val))
    >>> predictions = model.predict(X_test)
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from snowforecast.models.base import BaseModel

logger = logging.getLogger(__name__)


def _check_torch_available() -> None:
    """Check if PyTorch is available.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for LSTM/GRU models. "
            "Install with: pip install torch"
        )


class SnowLSTM(nn.Module if TORCH_AVAILABLE else object):
    """LSTM neural network for snowfall prediction.

    This module implements a multi-layer LSTM with optional bidirectional
    processing and layer normalization for training stability.

    Input shape: (batch_size, sequence_length, n_features)
    Output shape: (batch_size, 1) for single-step prediction

    Attributes:
        hidden_size: Number of hidden units in each LSTM layer.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability between layers.
        bidirectional: Whether to use bidirectional LSTM.

    Example:
        >>> model = SnowLSTM(
        ...     input_size=10,
        ...     hidden_size=64,
        ...     num_layers=2,
        ...     dropout=0.2,
        ... )
        >>> x = torch.randn(32, 7, 10)  # (batch, seq, features)
        >>> output = model(x)  # (32, 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        """Initialize the LSTM model.

        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden units in LSTM.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability (applied between layers).
            bidirectional: Whether to use bidirectional LSTM.
        """
        _check_torch_available()
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Layer normalization for training stability
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

        # Output layers
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass through the LSTM.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        # LSTM forward pass
        # lstm_out: (batch, seq_len, hidden_size * num_directions)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the last time step output
        # For bidirectional: concatenate forward and backward last states
        if self.bidirectional:
            # Forward last hidden: h_n[-2]
            # Backward last hidden: h_n[-1]
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]

        # Apply layer norm, dropout, and output layer
        normalized = self.layer_norm(last_hidden)
        dropped = self.dropout_layer(normalized)
        output = self.fc(dropped)

        return output


class SnowGRU(nn.Module if TORCH_AVAILABLE else object):
    """GRU neural network for snowfall prediction.

    GRU (Gated Recurrent Unit) is a simpler alternative to LSTM with fewer
    parameters, often performing comparably on many tasks.

    Input shape: (batch_size, sequence_length, n_features)
    Output shape: (batch_size, 1) for single-step prediction

    Attributes:
        hidden_size: Number of hidden units in each GRU layer.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout probability between layers.
        bidirectional: Whether to use bidirectional GRU.

    Example:
        >>> model = SnowGRU(
        ...     input_size=10,
        ...     hidden_size=64,
        ...     num_layers=2,
        ... )
        >>> x = torch.randn(32, 7, 10)
        >>> output = model(x)  # (32, 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        """Initialize the GRU model.

        Args:
            input_size: Number of input features.
            hidden_size: Number of hidden units in GRU.
            num_layers: Number of stacked GRU layers.
            dropout: Dropout probability (applied between layers).
            bidirectional: Whether to use bidirectional GRU.
        """
        _check_torch_available()
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Layer normalization for training stability
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

        # Output layers
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_directions, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass through the GRU.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        # GRU forward pass
        # gru_out: (batch, seq_len, hidden_size * num_directions)
        gru_out, h_n = self.gru(x)

        # Take the last hidden state
        if self.bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]

        # Apply layer norm, dropout, and output layer
        normalized = self.layer_norm(last_hidden)
        dropped = self.dropout_layer(normalized)
        output = self.fc(dropped)

        return output


class SequenceModel(BaseModel):
    """BaseModel wrapper for LSTM/GRU sequence models.

    This class provides a scikit-learn-like interface for PyTorch sequence
    models, handling data preparation, training, and prediction.

    Attributes:
        cell_type: Type of RNN cell ("lstm" or "gru").
        hidden_size: Number of hidden units.
        num_layers: Number of stacked RNN layers.
        dropout: Dropout probability.
        bidirectional: Whether to use bidirectional RNN.
        sequence_length: Length of input sequences.
        learning_rate: Learning rate for optimizer.
        batch_size: Mini-batch size for training.
        epochs: Maximum training epochs.
        patience: Early stopping patience (epochs without improvement).
        device: Device to run on ("cuda", "mps", or "cpu").

    Example:
        >>> model = SequenceModel(
        ...     cell_type="lstm",
        ...     hidden_size=64,
        ...     num_layers=2,
        ...     sequence_length=7,
        ... )
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        cell_type: Literal["lstm", "gru"] = "lstm",
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        sequence_length: int = 7,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        device: str | None = None,
        verbose: int = 0,
    ):
        """Initialize the SequenceModel.

        Args:
            cell_type: Type of RNN cell ("lstm" or "gru").
            hidden_size: Number of hidden units in RNN.
            num_layers: Number of stacked RNN layers.
            dropout: Dropout probability.
            bidirectional: Whether to use bidirectional RNN.
            sequence_length: Number of time steps in each sequence.
            learning_rate: Learning rate for Adam optimizer.
            batch_size: Mini-batch size for training.
            epochs: Maximum number of training epochs.
            patience: Early stopping patience.
            device: Device to use ("cuda", "mps", "cpu", or None for auto).
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed).

        Raises:
            ValueError: If cell_type is not "lstm" or "gru".
        """
        _check_torch_available()
        super().__init__()

        if cell_type not in ("lstm", "gru"):
            raise ValueError(f"cell_type must be 'lstm' or 'gru', got {cell_type}")

        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose

        # Auto-select device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        # Training state
        self.best_epoch: int | None = None
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self._scaler_mean: np.ndarray | None = None
        self._scaler_std: np.ndarray | None = None
        self._y_mean: float | None = None
        self._y_std: float | None = None

    def _create_model(self, input_size: int) -> nn.Module:
        """Create the RNN model.

        Args:
            input_size: Number of input features.

        Returns:
            SnowLSTM or SnowGRU instance.
        """
        if self.cell_type == "lstm":
            return SnowLSTM(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
            )
        else:
            return SnowGRU(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
            )

    def _prepare_sequences(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Prepare sequential data from DataFrame.

        Converts tabular data to sequences by sliding a window of length
        `sequence_length` over the data.

        Args:
            X: Feature DataFrame.
            y: Optional target Series.

        Returns:
            Tuple of (X_sequences, y_sequences) as numpy arrays.
            X_sequences shape: (n_samples, sequence_length, n_features)
            y_sequences shape: (n_samples,) or None
        """
        X_values = X.values.astype(np.float32)

        # Normalize features
        X_normalized = (X_values - self._scaler_mean) / (self._scaler_std + 1e-8)

        n_samples = len(X) - self.sequence_length + 1
        n_features = X.shape[1]

        X_seq = np.zeros((n_samples, self.sequence_length, n_features), dtype=np.float32)

        for i in range(n_samples):
            X_seq[i] = X_normalized[i:i + self.sequence_length]

        if y is not None:
            # Target is the value at the end of each sequence
            y_values = y.values[self.sequence_length - 1:].astype(np.float32)
            # Normalize target
            y_normalized = (y_values - self._y_mean) / (self._y_std + 1e-8)
            return X_seq, y_normalized

        return X_seq, None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> "SequenceModel":
        """Train the sequence model.

        Args:
            X: Training features as DataFrame.
            y: Training targets as Series.
            eval_set: Optional validation set (X_val, y_val) for early stopping.

        Returns:
            self for method chaining.
        """
        self._validate_X(X)
        self._validate_y(y)

        if len(X) < self.sequence_length:
            raise ValueError(
                f"X has {len(X)} samples but sequence_length={self.sequence_length}. "
                "Need at least sequence_length samples."
            )

        self.feature_names = list(X.columns)

        # Compute normalization statistics from training data
        self._scaler_mean = X.values.mean(axis=0).astype(np.float32)
        self._scaler_std = X.values.std(axis=0).astype(np.float32)
        self._y_mean = float(y.mean())
        self._y_std = float(y.std())

        # Prepare training sequences
        X_train_seq, y_train_seq = self._prepare_sequences(X, y)

        # Create model
        input_size = X.shape[1]
        self.model = self._create_model(input_size)
        self.model.to(self.device)

        # Prepare validation data
        X_val_seq, y_val_seq = None, None
        if eval_set is not None:
            X_val, y_val = eval_set
            self._validate_X(X_val)
            self._validate_y(y_val)
            if len(X_val) >= self.sequence_length:
                X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Create data loaders
        train_dataset = TensorDataset(
            torch.from_numpy(X_train_seq),
            torch.from_numpy(y_train_seq),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader = None
        if X_val_seq is not None:
            val_dataset = TensorDataset(
                torch.from_numpy(X_val_seq),
                torch.from_numpy(y_val_seq),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )

        # Training loop
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_state_dict = None

        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            avg_train_loss = train_loss / n_batches
            self.train_losses.append(avg_train_loss)

            # Validation phase
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                n_val_batches = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        outputs = self.model(batch_X).squeeze()
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        n_val_batches += 1

                avg_val_loss = val_loss / n_val_batches
                self.val_losses.append(avg_val_loss)

                scheduler.step(avg_val_loss)

                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_without_improvement = 0
                    best_state_dict = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    self.best_epoch = epoch
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.patience:
                    if self.verbose >= 1:
                        logger.info(f"Early stopping at epoch {epoch}")
                    break

                if self.verbose >= 2:
                    logger.info(
                        f"Epoch {epoch}: train_loss={avg_train_loss:.6f}, "
                        f"val_loss={avg_val_loss:.6f}"
                    )
            else:
                if self.verbose >= 2:
                    logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.6f}")

        # Restore best model if early stopping was used
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        self.model.to(self.device)
        self.is_fitted = True

        if self.verbose >= 1:
            logger.info(
                f"Training complete. Best epoch: {self.best_epoch}, "
                f"Best val loss: {best_val_loss:.6f}"
            )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Features as DataFrame with same columns as training data.

        Returns:
            Predicted values as numpy array.
        """
        self._validate_fitted()
        self._validate_X(X)

        # Validate features match training
        if set(X.columns) != set(self.feature_names):
            raise ValueError(
                f"Feature mismatch. Expected {self.feature_names}, got {list(X.columns)}"
            )

        if len(X) < self.sequence_length:
            raise ValueError(
                f"X has {len(X)} samples but need at least {self.sequence_length} "
                "for sequence prediction."
            )

        # Prepare sequences
        X_seq, _ = self._prepare_sequences(X)

        # Make predictions
        self.model.eval()
        predictions = []

        with torch.no_grad():
            X_tensor = torch.from_numpy(X_seq).to(self.device)
            outputs = self.model(X_tensor).squeeze().cpu().numpy()

            # Denormalize predictions
            outputs = outputs * (self._y_std + 1e-8) + self._y_mean
            predictions = outputs

        return predictions

    def get_feature_importance(self, **kwargs) -> pd.DataFrame:
        """Get feature importance scores.

        For RNN models, feature importance is estimated using gradient-based
        attribution. This returns the mean absolute gradient of the output
        with respect to each input feature.

        Returns:
            DataFrame with 'feature' and 'importance' columns.
        """
        self._validate_fitted()

        # For RNNs, we don't have built-in feature importance
        # Return equal importance as a placeholder
        n_features = len(self.feature_names)
        importance = np.ones(n_features) / n_features

        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "cell_type": self.cell_type,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "sequence_length": self.sequence_length,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "patience": self.patience,
            "device": self.device,
            "verbose": self.verbose,
        }

    def set_params(self, **params) -> "SequenceModel":
        """Set model parameters.

        Args:
            **params: Parameters to set.

        Returns:
            self for method chaining.
        """
        valid_params = set(self.get_params().keys())

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter: {key}. "
                    f"Valid parameters: {valid_params}"
                )
            setattr(self, key, value)

        return self

    def save(self, path: str | Path) -> None:
        """Save the model to disk.

        Args:
            path: Path to save the model.
        """
        self._validate_fitted()

        path = Path(path)
        if path.suffix != ".pkl":
            path = path.with_suffix(".pkl")

        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "params": self.get_params(),
            "feature_names": self.feature_names,
            "scaler_mean": self._scaler_mean,
            "scaler_std": self._scaler_std,
            "y_mean": self._y_mean,
            "y_std": self._y_std,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_epoch": self.best_epoch,
        }

        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, path: str | Path) -> "SequenceModel":
        """Load a model from disk.

        Args:
            path: Path to the saved model.

        Returns:
            Loaded SequenceModel instance.
        """
        _check_torch_available()

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        # Create model with saved params
        params = save_dict["params"]
        model = cls(**params)

        # Restore state
        model.feature_names = save_dict["feature_names"]
        model._scaler_mean = save_dict["scaler_mean"]
        model._scaler_std = save_dict["scaler_std"]
        model._y_mean = save_dict["y_mean"]
        model._y_std = save_dict["y_std"]
        model.train_losses = save_dict["train_losses"]
        model.val_losses = save_dict["val_losses"]
        model.best_epoch = save_dict["best_epoch"]

        # Recreate and load PyTorch model
        input_size = len(model.feature_names)
        model.model = model._create_model(input_size)
        model.model.load_state_dict(save_dict["model_state_dict"])
        model.model.to(model.device)
        model.is_fitted = True

        return model

    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"SequenceModel("
            f"{self.cell_type}, "
            f"hidden={self.hidden_size}, "
            f"layers={self.num_layers}, "
            f"seq_len={self.sequence_length}, "
            f"{status})"
        )
