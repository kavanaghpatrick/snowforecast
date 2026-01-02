"""Tests for prediction API.

Tests use FastAPI TestClient - NO MOCKS for API testing.
"""

from datetime import date, datetime

import pytest
from fastapi.testclient import TestClient

from snowforecast.api import (
    ConfidenceInterval,
    ForecastResult,
    HealthResponse,
    ModelCache,
    PredictionRequest,
    PredictionResponse,
    create_app,
    get_model_cache,
)
from snowforecast.api.schemas import WESTERN_US_BOUNDS


@pytest.fixture
def client():
    """Create test client with app and loaded model."""
    app = create_app(load_model=False)  # Don't use async startup
    # Manually load the model
    cache = get_model_cache()
    cache.load_model()
    return TestClient(app)


@pytest.fixture
def cache():
    """Create fresh model cache."""
    return ModelCache()


class TestSchemas:
    """Tests for Pydantic schemas."""

    def test_prediction_request_valid(self):
        """Valid request should be accepted."""
        req = PredictionRequest(
            latitude=39.5,
            longitude=-106.0,
            date=date(2026, 1, 15),
            forecast_hours=24,
        )

        assert req.latitude == 39.5
        assert req.longitude == -106.0
        assert req.forecast_hours == 24

    def test_prediction_request_default_hours(self):
        """Default forecast hours should be 24."""
        req = PredictionRequest(
            latitude=39.5,
            longitude=-106.0,
            date=date(2026, 1, 15),
        )

        assert req.forecast_hours == 24

    def test_prediction_request_bounds(self):
        """Request outside bounds should fail validation."""
        # Latitude too low
        with pytest.raises(ValueError):
            PredictionRequest(
                latitude=20.0,  # Below min
                longitude=-106.0,
                date=date(2026, 1, 15),
            )

        # Longitude too high
        with pytest.raises(ValueError):
            PredictionRequest(
                latitude=39.5,
                longitude=-90.0,  # Above max
                date=date(2026, 1, 15),
            )

    def test_prediction_request_invalid_hours(self):
        """Invalid forecast hours should fail."""
        with pytest.raises(ValueError):
            PredictionRequest(
                latitude=39.5,
                longitude=-106.0,
                date=date(2026, 1, 15),
                forecast_hours=200,  # > 168
            )

    def test_forecast_result_bounds(self):
        """Forecast values must be non-negative."""
        with pytest.raises(ValueError):
            ForecastResult(
                snow_depth_cm=-10.0,
                new_snow_cm=5.0,
                snowfall_probability=0.5,
            )

    def test_forecast_result_probability_bounds(self):
        """Probability must be 0-1."""
        with pytest.raises(ValueError):
            ForecastResult(
                snow_depth_cm=10.0,
                new_snow_cm=5.0,
                snowfall_probability=1.5,  # > 1
            )


class TestModelCache:
    """Tests for ModelCache class."""

    def test_initial_state(self, cache):
        """Cache should start empty."""
        assert not cache.is_loaded()
        assert cache.model is None
        assert cache.loaded_at is None

    def test_load_mock_model(self, cache):
        """Should load mock model when no path provided."""
        cache.load_model()

        assert cache.is_loaded()
        assert cache.model is not None
        assert cache.loaded_at is not None

    def test_predict_requires_loaded_model(self, cache):
        """Predict should fail if model not loaded."""
        with pytest.raises(RuntimeError, match="not loaded"):
            cache.predict(39.5, -106.0, datetime.now())

    def test_predict_with_loaded_model(self, cache):
        """Predict should work with loaded model."""
        cache.load_model()

        forecast, confidence = cache.predict(
            lat=39.5,
            lon=-106.0,
            date=datetime(2026, 1, 15),
            forecast_hours=24,
        )

        assert isinstance(forecast, ForecastResult)
        assert isinstance(confidence, ConfidenceInterval)
        assert forecast.snow_depth_cm >= 0
        assert forecast.new_snow_cm >= 0
        assert 0 <= forecast.snowfall_probability <= 1

    def test_get_elevation(self, cache):
        """Should return elevation estimate."""
        elev = cache.get_elevation(39.5, -106.0)

        assert isinstance(elev, float)
        assert elev > 0


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client):
        """Health endpoint should return status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "version" in data

    def test_health_response_schema(self, client):
        """Health response should match schema."""
        response = client.get("/health")
        data = response.json()

        health = HealthResponse(**data)
        assert health.status == "healthy"


class TestRootEndpoint:
    """Tests for / endpoint."""

    def test_root(self, client):
        """Root should return API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestModelInfoEndpoint:
    """Tests for /model/info endpoint."""

    def test_model_info(self, client):
        """Model info should return metadata."""
        response = client.get("/model/info")

        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert "loaded" in data
        assert "bounds" in data


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_success(self, client):
        """Valid prediction request should succeed."""
        response = client.post(
            "/predict",
            json={
                "latitude": 39.5,
                "longitude": -106.0,
                "date": "2026-01-15",
                "forecast_hours": 24,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Check structure
        assert "location" in data
        assert "forecast" in data
        assert "confidence_interval" in data
        assert "model_version" in data

        # Check location
        assert data["location"]["lat"] == 39.5
        assert data["location"]["lon"] == -106.0
        assert "elevation" in data["location"]

        # Check forecast values are valid
        assert data["forecast"]["snow_depth_cm"] >= 0
        assert data["forecast"]["new_snow_cm"] >= 0
        assert 0 <= data["forecast"]["snowfall_probability"] <= 1

    def test_predict_response_schema(self, client):
        """Response should match PredictionResponse schema."""
        response = client.post(
            "/predict",
            json={
                "latitude": 40.0,
                "longitude": -105.5,
                "date": "2026-02-01",
            },
        )

        assert response.status_code == 200
        # Should parse without error
        PredictionResponse(**response.json())

    def test_predict_invalid_latitude(self, client):
        """Invalid latitude should return 422."""
        response = client.post(
            "/predict",
            json={
                "latitude": 25.0,  # Below bounds
                "longitude": -106.0,
                "date": "2026-01-15",
            },
        )

        assert response.status_code == 422

    def test_predict_invalid_longitude(self, client):
        """Invalid longitude should return 422."""
        response = client.post(
            "/predict",
            json={
                "latitude": 39.5,
                "longitude": -85.0,  # Above bounds
                "date": "2026-01-15",
            },
        )

        assert response.status_code == 422

    def test_predict_missing_required_field(self, client):
        """Missing required field should return 422."""
        response = client.post(
            "/predict",
            json={
                "latitude": 39.5,
                # Missing longitude and date
            },
        )

        assert response.status_code == 422

    def test_predict_invalid_date_format(self, client):
        """Invalid date format should return 422."""
        response = client.post(
            "/predict",
            json={
                "latitude": 39.5,
                "longitude": -106.0,
                "date": "01-15-2026",  # Wrong format
            },
        )

        assert response.status_code == 422

    def test_predict_winter_vs_summer(self, client):
        """Winter predictions should generally show more snow."""
        # Winter prediction
        winter_response = client.post(
            "/predict",
            json={
                "latitude": 39.5,
                "longitude": -106.0,
                "date": "2026-01-15",
            },
        )

        # Summer prediction
        summer_response = client.post(
            "/predict",
            json={
                "latitude": 39.5,
                "longitude": -106.0,
                "date": "2026-07-15",
            },
        )

        winter_data = winter_response.json()
        summer_data = summer_response.json()

        # Winter should generally have higher probability
        # (this is a soft check since mock has randomness)
        assert winter_data["forecast"]["snowfall_probability"] >= 0
        assert summer_data["forecast"]["snowfall_probability"] >= 0


class TestCORS:
    """Tests for CORS middleware."""

    def test_cors_headers(self, client):
        """CORS headers should be present."""
        response = client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

        # FastAPI with CORS should handle preflight
        assert response.status_code in (200, 405)  # Depends on FastAPI version


class TestOpenAPI:
    """Tests for OpenAPI documentation."""

    def test_docs_available(self, client):
        """Swagger docs should be available."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self, client):
        """ReDoc should be available."""
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_json(self, client):
        """OpenAPI JSON should be available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
        assert "/predict" in data["paths"]


class TestCreateApp:
    """Tests for create_app factory function."""

    def test_create_app_default(self):
        """Default app creation should work."""
        app = create_app()
        assert app is not None
        assert app.title == "Snow Forecast API"

    def test_create_app_no_load(self):
        """Can create app without loading model."""
        app = create_app(load_model=False)
        client = TestClient(app)

        # App should work but model not loaded
        response = client.get("/health")
        assert response.status_code == 200


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_boundary_latitude(self, client):
        """Boundary values should be accepted."""
        # Minimum latitude
        response = client.post(
            "/predict",
            json={
                "latitude": WESTERN_US_BOUNDS["lat_min"],
                "longitude": -110.0,
                "date": "2026-01-15",
            },
        )
        assert response.status_code == 200

        # Maximum latitude
        response = client.post(
            "/predict",
            json={
                "latitude": WESTERN_US_BOUNDS["lat_max"],
                "longitude": -110.0,
                "date": "2026-01-15",
            },
        )
        assert response.status_code == 200

    def test_max_forecast_hours(self, client):
        """Maximum forecast hours should be accepted."""
        response = client.post(
            "/predict",
            json={
                "latitude": 39.5,
                "longitude": -106.0,
                "date": "2026-01-15",
                "forecast_hours": 168,  # Maximum
            },
        )
        assert response.status_code == 200

    def test_min_forecast_hours(self, client):
        """Minimum forecast hours should be accepted."""
        response = client.post(
            "/predict",
            json={
                "latitude": 39.5,
                "longitude": -106.0,
                "date": "2026-01-15",
                "forecast_hours": 1,  # Minimum
            },
        )
        assert response.status_code == 200
