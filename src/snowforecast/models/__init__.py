"""Machine learning models for snowfall prediction.

This module provides:

Base classes:
- BaseModel: Abstract base class for all models

Model implementations:
- GradientBoostingModel: LightGBM/XGBoost gradient boosting
- SequenceModel: LSTM/GRU recurrent neural networks for time series
- SnowLSTM: PyTorch LSTM module
- SnowGRU: PyTorch GRU module
"""

from snowforecast.models.base import BaseModel
from snowforecast.models.gradient_boosting import GradientBoostingModel

# LSTM/GRU models require PyTorch - import conditionally
try:
    from snowforecast.models.lstm import SequenceModel, SnowLSTM, SnowGRU
    _LSTM_AVAILABLE = True
except ImportError:
    _LSTM_AVAILABLE = False

__all__ = [
    "BaseModel",
    "GradientBoostingModel",
]

# Add LSTM exports only if PyTorch is available
if _LSTM_AVAILABLE:
    __all__.extend(["SequenceModel", "SnowLSTM", "SnowGRU"])
