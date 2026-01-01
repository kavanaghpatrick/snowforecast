"""Machine learning models for snowfall prediction.

This module provides:

Base classes:
- BaseModel: Abstract base class for all models

Model implementations:
- LinearRegressionModel: Linear regression baseline model
- GradientBoostingModel: LightGBM/XGBoost gradient boosting
"""

from snowforecast.models.base import BaseModel
from snowforecast.models.linear import LinearRegressionModel

# Gradient boosting is optional, import if available
try:
    from snowforecast.models.gradient_boosting import GradientBoostingModel
    _HAS_GB = True
except ImportError:
    _HAS_GB = False

__all__ = [
    "BaseModel",
    "LinearRegressionModel",
]

if _HAS_GB:
    __all__.append("GradientBoostingModel")
