"""Pydantic schemas for API request/response validation.

Defines all data models used by the prediction API.
"""

from datetime import date as date_type
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

# Western US bounding box for validation
WESTERN_US_BOUNDS = {
    "lat_min": 31.0,
    "lat_max": 49.0,
    "lon_min": -125.0,
    "lon_max": -102.0,
}


class PredictionRequest(BaseModel):
    """Request schema for snow predictions.

    Attributes:
        latitude: Latitude in decimal degrees (31-49 for Western US)
        longitude: Longitude in decimal degrees (-125 to -102 for Western US)
        date: Target date for prediction (YYYY-MM-DD)
        forecast_hours: Hours ahead to forecast (default 24)
    """

    latitude: float = Field(
        ...,
        ge=WESTERN_US_BOUNDS["lat_min"],
        le=WESTERN_US_BOUNDS["lat_max"],
        description="Latitude in decimal degrees",
    )
    longitude: float = Field(
        ...,
        ge=WESTERN_US_BOUNDS["lon_min"],
        le=WESTERN_US_BOUNDS["lon_max"],
        description="Longitude in decimal degrees",
    )
    target_date: date_type = Field(
        ...,
        description="Target date for prediction",
        alias="date",
    )
    forecast_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Hours ahead to forecast (1-168)",
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "latitude": 39.5,
                    "longitude": -106.0,
                    "date": "2026-01-15",
                    "forecast_hours": 24,
                }
            ]
        }
    }


class LocationInfo(BaseModel):
    """Location information in response.

    Attributes:
        lat: Latitude
        lon: Longitude
        elevation: Elevation in meters (derived from DEM)
    """

    lat: float
    lon: float
    elevation: Optional[float] = Field(
        default=None,
        description="Elevation in meters",
    )


class ForecastResult(BaseModel):
    """Snow forecast values.

    Attributes:
        snow_depth_cm: Predicted snow depth in centimeters
        new_snow_cm: Predicted new snowfall in centimeters
        snowfall_probability: Probability of measurable snowfall (0-1)
    """

    snow_depth_cm: float = Field(
        ...,
        ge=0,
        description="Predicted snow depth in cm",
    )
    new_snow_cm: float = Field(
        ...,
        ge=0,
        description="Predicted new snowfall in cm",
    )
    snowfall_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of measurable snowfall",
    )


class ConfidenceInterval(BaseModel):
    """Confidence interval for predictions.

    Attributes:
        lower: Lower bound of 95% CI
        upper: Upper bound of 95% CI
    """

    lower: float = Field(
        ...,
        description="Lower bound of 95% confidence interval",
    )
    upper: float = Field(
        ...,
        description="Upper bound of 95% confidence interval",
    )


class PredictionResponse(BaseModel):
    """Full prediction response.

    Attributes:
        location: Location information
        forecast: Snow forecast values
        confidence_interval: 95% CI for new_snow_cm
        model_version: Version of the model used
        generated_at: Timestamp when prediction was generated
    """

    location: LocationInfo
    forecast: ForecastResult
    confidence_interval: ConfidenceInterval
    model_version: str = Field(
        default="v1.0",
        description="Model version used for prediction",
    )
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of prediction generation",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "location": {"lat": 39.5, "lon": -106.0, "elevation": 3200},
                    "forecast": {
                        "snow_depth_cm": 45.2,
                        "new_snow_cm": 12.5,
                        "snowfall_probability": 0.85,
                    },
                    "confidence_interval": {"lower": 8.0, "upper": 18.0},
                    "model_version": "v1.0",
                    "generated_at": "2026-01-15T12:00:00Z",
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status: Service status ('healthy' or 'unhealthy')
        model_loaded: Whether model is loaded and ready
        version: API version
    """

    status: str = Field(
        default="healthy",
        description="Service status",
    )
    model_loaded: bool = Field(
        default=False,
        description="Whether model is loaded",
    )
    version: str = Field(
        default="1.0.0",
        description="API version",
    )


class ErrorResponse(BaseModel):
    """Error response schema.

    Attributes:
        error: Error type/code
        message: Human-readable error message
        detail: Additional error details
    """

    error: str = Field(
        ...,
        description="Error type",
    )
    message: str = Field(
        ...,
        description="Error message",
    )
    detail: Optional[str] = Field(
        default=None,
        description="Additional details",
    )
