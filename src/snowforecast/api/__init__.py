"""Prediction API for snowforecast.

This module provides:

- create_app: Factory function to create FastAPI application
- PredictionRequest: Request schema for predictions
- PredictionResponse: Response schema with forecast and confidence
- ModelCache: In-memory cache for loaded models

Note: FastAPI-dependent exports (create_app, ModelCache) are lazy-loaded
to allow importing schemas without FastAPI installed.
"""

# Schemas can be imported directly (only depend on pydantic)
from snowforecast.api.schemas import (
    ConfidenceInterval,
    ErrorResponse,
    ForecastResult,
    HealthResponse,
    LocationInfo,
    PredictionRequest,
    PredictionResponse,
)


# Lazy imports for FastAPI-dependent components
def __getattr__(name):
    """Lazy load FastAPI-dependent components."""
    if name in ("create_app", "get_model_cache", "ModelCache"):
        from snowforecast.api.app import ModelCache, create_app, get_model_cache
        if name == "create_app":
            return create_app
        elif name == "get_model_cache":
            return get_model_cache
        elif name == "ModelCache":
            return ModelCache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
