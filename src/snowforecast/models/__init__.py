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

Ensemble methods:
- SimpleEnsemble: Averaging ensemble with optional weights
- StackingEnsemble: Two-level stacking with meta-learner
- create_ensemble: Factory function for creating ensembles
- get_model_weights: Extract learned weights from ensembles

Comparison utilities:
- ModelComparison: Compare multiple models on the same dataset
- ComparisonResults: Container for comparison results
- compare_models: Convenience function for model comparison
- rank_models: Rank models by specified metric
- create_comparison_report: Generate formatted comparison report
"""

from snowforecast.models.base import BaseModel
from snowforecast.models.comparison import (
    ComparisonResults,
    ModelComparison,
    compare_models,
    create_comparison_report,
    rank_models,
)
from snowforecast.models.ensemble import (
    SimpleEnsemble,
    StackingEnsemble,
    create_ensemble,
    get_model_weights,
)
from snowforecast.models.linear import LinearRegressionModel

# Gradient boosting is optional, import if available
try:
    from snowforecast.models.gradient_boosting import GradientBoostingModel
    _HAS_GB = True
except ImportError:
    _HAS_GB = False

# LSTM/GRU models require PyTorch - import conditionally
try:
    from snowforecast.models.lstm import SequenceModel, SnowGRU, SnowLSTM
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
    # Ensemble methods
    "SimpleEnsemble",
    "StackingEnsemble",
    "create_ensemble",
    "get_model_weights",
    # Comparison utilities
    "ModelComparison",
    "ComparisonResults",
    "compare_models",
    "rank_models",
    "create_comparison_report",
]

if _HAS_GB:
    __all__.append("GradientBoostingModel")

# Add LSTM exports only if PyTorch is available
if _LSTM_AVAILABLE:
    __all__.extend(["SequenceModel", "SnowLSTM", "SnowGRU"])

# Add UNet exports only if PyTorch is available
if _UNET_AVAILABLE:
    __all__.extend(["SnowUNet", "SpatialModel"])
