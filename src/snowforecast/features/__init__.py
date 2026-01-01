"""Feature engineering and data orchestration for snowforecast.

This module provides:
- DataOrchestrator: Unified coordination of all data pipelines
- LocationPoint: Geographic location representation
- OrchestrationResult: Result container for orchestration operations
- SpatialAligner: Spatial alignment of data sources
- ExtractionResult: Result container for extraction operations
- TemporalAligner: Temporal alignment and resampling
"""

from .orchestration import (
    DataOrchestrator,
    LocationPoint,
    OrchestrationResult,
)
from .spatial import SpatialAligner, ExtractionResult
from .temporal import (
    DEFAULT_AGGREGATIONS,
    TemporalAligner,
)

__all__ = [
    "DataOrchestrator",
    "LocationPoint",
    "OrchestrationResult",
    "SpatialAligner",
    "ExtractionResult",
    "DEFAULT_AGGREGATIONS",
    "TemporalAligner",
]
