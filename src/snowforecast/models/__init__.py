"""Machine learning models for snowfall prediction.

This module provides:

Model implementations:
- SnowUNet: U-Net for spatial snow prediction
- SpatialModel: Wrapper for spatial models with training interface
"""

from snowforecast.models.unet import SnowUNet, SpatialModel

__all__ = [
    "SnowUNet",
    "SpatialModel",
]
