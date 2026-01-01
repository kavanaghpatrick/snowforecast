"""Prediction API for snowforecast.

This module provides:

- create_app: Factory function to create FastAPI application
- PredictionRequest: Request schema for predictions
- PredictionResponse: Response schema with forecast and confidence
- ModelCache: In-memory cache for loaded models
"""

from snowforecast.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    LocationInfo,
    ForecastResult,
    ConfidenceInterval,
    HealthResponse,
    ErrorResponse,
)
from snowforecast.api.app import create_app, get_model_cache, ModelCache

__all__ = [
    "create_app",
    "get_model_cache",
    "ModelCache",
    "PredictionRequest",
    "PredictionResponse",
    "LocationInfo",
    "ForecastResult",
    "ConfidenceInterval",
    "HealthResponse",
    "ErrorResponse",
]
