"""Feature engineering utilities for snowforecast.

This module provides utilities for transforming raw data into features
suitable for machine learning models.
"""

from snowforecast.features.temporal import (
    DEFAULT_AGGREGATIONS,
    TemporalAligner,
)

__all__ = [
    "DEFAULT_AGGREGATIONS",
    "TemporalAligner",
]
