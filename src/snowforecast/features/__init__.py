"""Feature engineering and data orchestration for snowforecast.

This module provides:
- DataOrchestrator: Unified coordination of all data pipelines
- SpatialAligner: Spatial alignment of data sources
- TemporalAligner: Temporal alignment and resampling
- DataQualityController: Data quality control and flagging
- AtmosphericFeatures: Atmospheric feature engineering
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
from .quality import (
    DataQualityController,
    QualityFlag,
    QualityReport,
)
from .atmospheric import AtmosphericFeatures

__all__ = [
    "DataOrchestrator",
    "LocationPoint",
    "OrchestrationResult",
    "SpatialAligner",
    "ExtractionResult",
    "DEFAULT_AGGREGATIONS",
    "TemporalAligner",
    "DataQualityController",
    "QualityFlag",
    "QualityReport",
    "AtmosphericFeatures",
]
