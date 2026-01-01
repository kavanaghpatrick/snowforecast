"""Machine learning models for snowfall prediction.

This module provides:

Base classes:
- BaseModel: Abstract base class for all models

Model implementations:
- LinearRegressionModel: Linear regression baseline model
- GradientBoostingModel: LightGBM/XGBoost gradient boosting
- SequenceModel: LSTM/GRU recurrent neural networks for time series
- SnowLSTM: PyTorch LSTM module
- SnowGRU: PyTorch GRU module
- SnowUNet: U-Net for spatial snow prediction
- SpatialModel: Wrapper for spatial models with BaseModel-like interface
"""

from snowforecast.models.base import BaseModel
from snowforecast.models.linear import LinearRegressionModel

# Gradient boosting is optional, import if available
try:
    from snowforecast.models.gradient_boosting import GradientBoostingModel
    _HAS_GB = True
except ImportError:
    _HAS_GB = False

# LSTM/GRU models require PyTorch - import conditionally
try:
    from snowforecast.models.lstm import SequenceModel, SnowLSTM, SnowGRU
    _LSTM_AVAILABLE = True
except ImportError:
    _LSTM_AVAILABLE = False

# UNet models require PyTorch - import conditionally
try:
    from snowforecast.models.unet import SnowUNet, SpatialModel
    _UNET_AVAILABLE = True
except ImportError:
    _UNET_AVAILABLE = False

__all__ = [
    "BaseModel",
    "LinearRegressionModel",
]

if _HAS_GB:
    __all__.append("GradientBoostingModel")

# Add LSTM exports only if PyTorch is available
if _LSTM_AVAILABLE:
    __all__.extend(["SequenceModel", "SnowLSTM", "SnowGRU"])

# Add UNet exports only if PyTorch is available
if _UNET_AVAILABLE:
    __all__.extend(["SnowUNet", "SpatialModel"])
