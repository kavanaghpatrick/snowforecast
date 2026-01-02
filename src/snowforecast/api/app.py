"""FastAPI application for snow predictions.

Provides REST API endpoints for:
- Snow depth and snowfall predictions
- Health checks
- Model information

Example:
    >>> from snowforecast.api import create_app
    >>> app = create_app()
    >>> # Run with: uvicorn snowforecast.api.app:app --reload
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from snowforecast.api.schemas import (
    WESTERN_US_BOUNDS,
    ConfidenceInterval,
    ErrorResponse,
    ForecastResult,
    HealthResponse,
    LocationInfo,
    PredictionRequest,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

# API version
API_VERSION = "1.0.0"
MODEL_VERSION = "v1.0"


class ModelCache:
    """In-memory cache for loaded models.

    Provides lazy loading and caching of prediction models
    to avoid reloading on every request.

    Attributes:
        model: Cached prediction model
        model_path: Path to model file
        loaded_at: Timestamp when model was loaded
        use_real_data: Whether to use real HRRR/DEM data
    """

    def __init__(self, model_path: Optional[Path] = None, use_real_data: bool = True):
        """Initialize cache.

        Args:
            model_path: Optional path to model file
            use_real_data: Whether to use real HRRR/DEM data (default True)
        """
        self.model = None
        self.model_path = model_path
        self.loaded_at: Optional[datetime] = None
        self.use_real_data = use_real_data
        self._real_predictor = None

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def load_model(self, path: Optional[Path] = None) -> None:
        """Load model from disk.

        Args:
            path: Path to model file (uses self.model_path if not provided)

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        load_path = path or self.model_path

        if self.use_real_data:
            try:
                from snowforecast.api.predictor import HybridPredictor
                self.model = HybridPredictor()
                self._real_predictor = self.model
                logger.info("Loaded HybridPredictor with real data support")
            except ImportError as e:
                logger.warning(f"Could not load HybridPredictor: {e}, falling back to mock")
                self.model = MockPredictor()
        elif load_path is None:
            logger.warning("No model path provided, using mock model")
            self.model = MockPredictor()
        elif not Path(load_path).exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        else:
            logger.info(f"Loading model from {load_path}")
            self.model = MockPredictor()

        self.loaded_at = datetime.utcnow()
        logger.info("Model loaded successfully")

    def predict(
        self,
        lat: float,
        lon: float,
        date: datetime,
        forecast_hours: int = 24,
    ) -> tuple[ForecastResult, ConfidenceInterval]:
        """Generate prediction for location.

        Args:
            lat: Latitude
            lon: Longitude
            date: Target date
            forecast_hours: Hours ahead to forecast

        Returns:
            Tuple of (ForecastResult, ConfidenceInterval)

        Raises:
            RuntimeError: If model not loaded
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        return self.model.predict(lat, lon, date, forecast_hours)

    def get_elevation(self, lat: float, lon: float) -> float:
        """Look up elevation for location using DEM.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Elevation in meters
        """
        if self._real_predictor is not None:
            try:
                return self._real_predictor.get_elevation(lat, lon)
            except Exception as e:
                logger.warning(f"DEM lookup failed: {e}")

        # Fallback estimate
        base_elev = 1500 + (lat - 35) * 100
        return max(base_elev, 500)


class MockPredictor:
    """Mock predictor for testing and development.

    Generates plausible predictions based on location and season.
    """

    def predict(
        self,
        lat: float,
        lon: float,
        date: datetime,
        forecast_hours: int = 24,
    ) -> tuple[ForecastResult, ConfidenceInterval]:
        """Generate mock prediction.

        Uses simple heuristics based on:
        - Latitude (higher = more snow)
        - Month (winter = more snow)
        - Elevation proxy
        """
        import random

        # Season factor (0-1, higher in winter)
        month = date.month
        if month in (12, 1, 2):
            season_factor = 1.0
        elif month in (3, 11):
            season_factor = 0.6
        elif month in (4, 10):
            season_factor = 0.3
        else:
            season_factor = 0.1

        # Latitude factor (higher lat = more snow)
        lat_factor = (lat - 35) / 15  # 0-1 scale

        # Base prediction with some randomness
        base_snow_depth = max(0, (50 + lat_factor * 100) * season_factor)
        new_snow = max(0, random.uniform(0, 20) * season_factor * max(0.1, lat_factor))

        # Snowfall probability
        snowfall_prob = min(0.95, season_factor * 0.8 + lat_factor * 0.2)
        if new_snow < 2.5:
            snowfall_prob *= 0.3

        # Confidence interval (wider for larger predictions)
        ci_width = max(2.0, new_snow * 0.4)

        forecast = ForecastResult(
            snow_depth_cm=round(base_snow_depth, 1),
            new_snow_cm=round(new_snow, 1),
            snowfall_probability=round(snowfall_prob, 2),
        )

        confidence = ConfidenceInterval(
            lower=round(max(0, new_snow - ci_width), 1),
            upper=round(new_snow + ci_width, 1),
        )

        return forecast, confidence


# Global model cache
_model_cache: Optional[ModelCache] = None


def get_model_cache() -> ModelCache:
    """Get or create global model cache."""
    global _model_cache
    if _model_cache is None:
        _model_cache = ModelCache()
    return _model_cache


def create_app(
    model_path: Optional[Path] = None,
    load_model: bool = True,
) -> FastAPI:
    """Create FastAPI application.

    Args:
        model_path: Optional path to model file
        load_model: Whether to load model on startup

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Snow Forecast API",
        description="Prediction API for snow depth and snowfall in Western US mountains",
        version=API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize model cache
    cache = get_model_cache()
    if model_path:
        cache.model_path = model_path

    @app.on_event("startup")
    async def startup_event():
        """Load model on startup."""
        if load_model:
            try:
                cache.load_model()
                logger.info("Model loaded on startup")
            except Exception as e:
                logger.error(f"Failed to load model on startup: {e}")

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with custom response."""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=f"HTTP_{exc.status_code}",
                message=str(exc.detail),
            ).model_dump(),
        )

    @app.get("/", tags=["info"])
    async def root():
        """Root endpoint with API info."""
        return {
            "name": "Snow Forecast API",
            "version": API_VERSION,
            "docs": "/docs",
        }

    @app.get("/health", response_model=HealthResponse, tags=["info"])
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model_loaded=cache.is_loaded(),
            version=API_VERSION,
        )

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        responses={
            400: {"model": ErrorResponse, "description": "Invalid request"},
            500: {"model": ErrorResponse, "description": "Server error"},
            503: {"model": ErrorResponse, "description": "Model not loaded"},
        },
        tags=["predictions"],
    )
    async def predict(request: PredictionRequest):
        """Generate snow prediction for a location.

        Takes latitude, longitude, date, and forecast horizon as input.
        Returns predicted snow depth, new snowfall, probability, and confidence interval.
        """
        if not cache.is_loaded():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please try again later.",
            )

        try:
            # Get elevation for location
            elevation = cache.get_elevation(request.latitude, request.longitude)

            # Generate prediction
            forecast, confidence = cache.predict(
                lat=request.latitude,
                lon=request.longitude,
                date=datetime.combine(request.target_date, datetime.min.time()),
                forecast_hours=request.forecast_hours,
            )

            return PredictionResponse(
                location=LocationInfo(
                    lat=request.latitude,
                    lon=request.longitude,
                    elevation=round(elevation, 0),
                ),
                forecast=forecast,
                confidence_interval=confidence,
                model_version=MODEL_VERSION,
            )

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}",
            )

    @app.get("/model/info", tags=["info"])
    async def model_info():
        """Get information about the loaded model."""
        return {
            "version": MODEL_VERSION,
            "loaded": cache.is_loaded(),
            "loaded_at": cache.loaded_at.isoformat() if cache.loaded_at else None,
            "bounds": WESTERN_US_BOUNDS,
        }

    return app


# Default app instance for uvicorn
app = create_app()
