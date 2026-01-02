"""Feature engineering and data orchestration for snowforecast.

This module provides:
- DataOrchestrator: Unified coordination of all data pipelines
- SpatialAligner: Spatial alignment of data sources
- TemporalAligner: Temporal alignment and resampling
- DataQualityController: Data quality control and flagging
- AtmosphericFeatures: Atmospheric feature engineering
- TerrainFeatureEngineer: Terrain feature engineering
- TemporalFeatures: Temporal and cyclical features
- LaggedFeatures: Lagged and rolling window features
"""

from .atmospheric import AtmosphericFeatures
from .lagged import LaggedFeatures
from .orchestration import (
    DataOrchestrator,
    LocationPoint,
    OrchestrationResult,
)
from .quality import (
    DataQualityController,
    QualityFlag,
    QualityReport,
)
from .spatial import ExtractionResult, SpatialAligner
from .temporal import (
    DEFAULT_AGGREGATIONS,
    TemporalAligner,
)
from .temporal_features import TemporalFeatures
from .terrain import TerrainFeatureEngineer

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
    "TerrainFeatureEngineer",
    "TemporalFeatures",
    "LaggedFeatures",
]
