"""Tests for DataOrchestrator and orchestration module."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pandas as pd
import pytest

from snowforecast.features import DataOrchestrator, LocationPoint, OrchestrationResult
from snowforecast.features.orchestration import ResortProvider, StationProvider

# --- Mock data classes to simulate pipeline outputs ---


@dataclass
class MockStationMetadata:
    """Mock station metadata (simulates SNOTEL/GHCN station data)."""
    station_id: str
    name: str
    lat: float
    lon: float
    elevation: float
    state: str


@dataclass
class MockSkiResort:
    """Mock ski resort (simulates OpenSkiMap resort data)."""
    name: str
    lat: float
    lon: float
    base_elevation: float | None
    summit_elevation: float | None
    vertical_drop: float | None
    country: str
    state: str
    nearest_snotel: str | None = None


# --- Fixtures ---


@pytest.fixture
def mock_snotel_stations():
    """Sample SNOTEL station data."""
    return [
        MockStationMetadata(
            station_id="1050:CO:SNTL",
            name="Berthoud Summit",
            lat=39.80,
            lon=-105.78,
            elevation=3450,
            state="CO",
        ),
        MockStationMetadata(
            station_id="1051:CO:SNTL",
            name="Loveland Basin",
            lat=39.68,
            lon=-105.90,
            elevation=3520,
            state="CO",
        ),
        MockStationMetadata(
            station_id="978:UT:SNTL",
            name="Brighton",
            lat=40.60,
            lon=-111.58,
            elevation=2670,
            state="UT",
        ),
    ]


@pytest.fixture
def mock_ghcn_stations():
    """Sample GHCN station data."""
    return [
        MockStationMetadata(
            station_id="USC00053005",
            name="Climax",
            lat=39.37,
            lon=-106.18,
            elevation=3460,
            state="CO",
        ),
        MockStationMetadata(
            station_id="USC00420738",
            name="Alta",
            lat=40.59,
            lon=-111.63,
            elevation=2652,
            state="UT",
        ),
    ]


@pytest.fixture
def mock_resorts():
    """Sample ski resort data."""
    return [
        MockSkiResort(
            name="Vail",
            lat=39.64,
            lon=-106.37,
            base_elevation=2475,
            summit_elevation=3527,
            vertical_drop=1052,
            country="US",
            state="CO",
        ),
        MockSkiResort(
            name="Park City",
            lat=40.65,
            lon=-111.51,
            base_elevation=2103,
            summit_elevation=3049,
            vertical_drop=946,
            country="US",
            state="UT",
        ),
        MockSkiResort(
            name="Mammoth Mountain",
            lat=37.63,
            lon=-119.04,
            base_elevation=2424,
            summit_elevation=3369,
            vertical_drop=945,
            country="US",
            state="CA",
        ),
    ]


@pytest.fixture
def mock_snotel_provider(mock_snotel_stations):
    """Mock SNOTEL pipeline."""
    provider = MagicMock(spec=StationProvider)

    def get_stations(state=None):
        if state is None:
            return mock_snotel_stations
        return [s for s in mock_snotel_stations if s.state == state]

    provider.get_station_metadata = MagicMock(side_effect=get_stations)
    return provider


@pytest.fixture
def mock_ghcn_provider(mock_ghcn_stations):
    """Mock GHCN pipeline."""
    provider = MagicMock(spec=StationProvider)

    def get_stations(state=None):
        if state is None:
            return mock_ghcn_stations
        return [s for s in mock_ghcn_stations if s.state == state]

    provider.get_station_metadata = MagicMock(side_effect=get_stations)
    return provider


@pytest.fixture
def mock_resort_provider(mock_resorts):
    """Mock OpenSkiMap pipeline."""
    provider = MagicMock(spec=ResortProvider)
    provider.get_western_us_resorts = MagicMock(return_value=mock_resorts)

    def export_df(resorts):
        if not resorts:
            return pd.DataFrame(columns=[
                "name", "lat", "lon", "base_elevation", "summit_elevation",
                "vertical_drop", "country", "state", "nearest_snotel"
            ])
        return pd.DataFrame([
            {
                "name": r.name,
                "lat": r.lat,
                "lon": r.lon,
                "base_elevation": r.base_elevation,
                "summit_elevation": r.summit_elevation,
                "vertical_drop": r.vertical_drop,
                "country": r.country,
                "state": r.state,
                "nearest_snotel": r.nearest_snotel,
            }
            for r in resorts
        ])

    provider.export_to_dataframe = MagicMock(side_effect=export_df)
    return provider


@pytest.fixture
def orchestrator():
    """Create a fresh DataOrchestrator instance."""
    return DataOrchestrator()


# --- Test: Basic Initialization ---


class TestDataOrchestratorInit:
    """Test DataOrchestrator initialization."""

    def test_init_empty_providers(self, orchestrator):
        """Orchestrator starts with no registered providers."""
        providers = orchestrator.get_provider_names()
        assert providers["station_providers"] == []
        assert providers["resort_providers"] == []

    def test_init_no_cached_data(self, orchestrator):
        """Orchestrator starts with no cached data."""
        assert orchestrator._collected_stations is None
        assert orchestrator._collected_resorts is None


# --- Test: Provider Registration ---


class TestProviderRegistration:
    """Test provider registration functionality."""

    def test_register_station_provider(self, orchestrator, mock_snotel_provider):
        """Can register a station provider."""
        orchestrator.register_station_provider("snotel", mock_snotel_provider)

        providers = orchestrator.get_provider_names()
        assert "snotel" in providers["station_providers"]

    def test_register_multiple_station_providers(
        self, orchestrator, mock_snotel_provider, mock_ghcn_provider
    ):
        """Can register multiple station providers."""
        orchestrator.register_station_provider("snotel", mock_snotel_provider)
        orchestrator.register_station_provider("ghcn", mock_ghcn_provider)

        providers = orchestrator.get_provider_names()
        assert len(providers["station_providers"]) == 2
        assert "snotel" in providers["station_providers"]
        assert "ghcn" in providers["station_providers"]

    def test_register_resort_provider(self, orchestrator, mock_resort_provider):
        """Can register a resort provider."""
        orchestrator.register_resort_provider("openskimap", mock_resort_provider)

        providers = orchestrator.get_provider_names()
        assert "openskimap" in providers["resort_providers"]


# --- Test: Station Collection ---


class TestStationCollection:
    """Test station data collection."""

    def test_collect_stations_no_providers_raises(self, orchestrator):
        """Raises ValueError if no providers registered."""
        with pytest.raises(ValueError, match="No station providers registered"):
            orchestrator.collect_stations()

    def test_collect_stations_single_provider(
        self, orchestrator, mock_snotel_provider, mock_snotel_stations
    ):
        """Collects stations from single provider."""
        orchestrator.register_station_provider("snotel", mock_snotel_provider)

        df = orchestrator.collect_stations()

        assert len(df) == len(mock_snotel_stations)
        assert "station_id" in df.columns
        assert "source" in df.columns
        assert all(df["source"] == "snotel")

    def test_collect_stations_multiple_providers(
        self, orchestrator, mock_snotel_provider, mock_ghcn_provider,
        mock_snotel_stations, mock_ghcn_stations
    ):
        """Collects stations from multiple providers."""
        orchestrator.register_station_provider("snotel", mock_snotel_provider)
        orchestrator.register_station_provider("ghcn", mock_ghcn_provider)

        df = orchestrator.collect_stations()

        expected_count = len(mock_snotel_stations) + len(mock_ghcn_stations)
        assert len(df) == expected_count
        assert set(df["source"].unique()) == {"snotel", "ghcn"}

    def test_collect_stations_filter_by_state(
        self, orchestrator, mock_snotel_provider
    ):
        """Filters stations by state."""
        orchestrator.register_station_provider("snotel", mock_snotel_provider)

        df = orchestrator.collect_stations(states=["CO"])

        # Should only get Colorado stations
        assert len(df) == 2
        assert all(df["state"] == "CO")

    def test_collect_stations_filter_by_elevation(
        self, orchestrator, mock_snotel_provider
    ):
        """Filters stations by minimum elevation."""
        orchestrator.register_station_provider("snotel", mock_snotel_provider)

        # Filter for stations above 3000m
        df = orchestrator.collect_stations(min_elevation=3000)

        # Only Berthoud (3450m) and Loveland (3520m) should pass
        assert len(df) == 2
        assert all(df["elevation"] >= 3000)

    def test_collect_stations_caches_result(
        self, orchestrator, mock_snotel_provider
    ):
        """Station collection result is cached."""
        orchestrator.register_station_provider("snotel", mock_snotel_provider)

        orchestrator.collect_stations()

        assert orchestrator._collected_stations is not None


# --- Test: Resort Collection ---


class TestResortCollection:
    """Test resort data collection."""

    def test_collect_resorts_no_providers_raises(self, orchestrator):
        """Raises ValueError if no providers registered."""
        with pytest.raises(ValueError, match="No resort providers registered"):
            orchestrator.collect_resorts()

    def test_collect_resorts_single_provider(
        self, orchestrator, mock_resort_provider, mock_resorts
    ):
        """Collects resorts from provider."""
        orchestrator.register_resort_provider("openskimap", mock_resort_provider)

        df = orchestrator.collect_resorts()

        assert len(df) == len(mock_resorts)
        assert "name" in df.columns
        assert "base_elevation" in df.columns
        assert "source" in df.columns
        assert all(df["source"] == "openskimap")

    def test_collect_resorts_caches_result(
        self, orchestrator, mock_resort_provider
    ):
        """Resort collection result is cached."""
        orchestrator.register_resort_provider("openskimap", mock_resort_provider)

        orchestrator.collect_resorts()

        assert orchestrator._collected_resorts is not None


# --- Test: Location Consolidation ---


class TestLocationConsolidation:
    """Test location consolidation functionality."""

    def test_consolidate_no_data_raises(self, orchestrator):
        """Raises ValueError if no data collected."""
        with pytest.raises(ValueError, match="No locations to consolidate"):
            orchestrator.consolidate_locations()

    def test_consolidate_stations_only(
        self, orchestrator, mock_snotel_provider, mock_snotel_stations
    ):
        """Consolidates stations when only stations collected."""
        orchestrator.register_station_provider("snotel", mock_snotel_provider)
        orchestrator.collect_stations()

        df = orchestrator.consolidate_locations(
            include_stations=True,
            include_resorts=False,
        )

        assert len(df) == len(mock_snotel_stations)
        assert all(df["location_type"] == "station")
        assert "location_id" in df.columns

    def test_consolidate_resorts_only(
        self, orchestrator, mock_resort_provider, mock_resorts
    ):
        """Consolidates resorts when only resorts collected."""
        orchestrator.register_resort_provider("openskimap", mock_resort_provider)
        orchestrator.collect_resorts()

        df = orchestrator.consolidate_locations(
            include_stations=False,
            include_resorts=True,
        )

        assert len(df) == len(mock_resorts)
        assert all(df["location_type"] == "resort")

    def test_consolidate_both_stations_and_resorts(
        self, orchestrator, mock_snotel_provider, mock_resort_provider,
        mock_snotel_stations, mock_resorts
    ):
        """Consolidates both stations and resorts."""
        orchestrator.register_station_provider("snotel", mock_snotel_provider)
        orchestrator.register_resort_provider("openskimap", mock_resort_provider)
        orchestrator.collect_stations()
        orchestrator.collect_resorts()

        df = orchestrator.consolidate_locations()

        expected_count = len(mock_snotel_stations) + len(mock_resorts)
        assert len(df) == expected_count
        assert set(df["location_type"].unique()) == {"station", "resort"}

    def test_consolidated_has_unique_ids(
        self, orchestrator, mock_snotel_provider, mock_resort_provider
    ):
        """Consolidated locations have unique IDs."""
        orchestrator.register_station_provider("snotel", mock_snotel_provider)
        orchestrator.register_resort_provider("openskimap", mock_resort_provider)
        orchestrator.collect_stations()
        orchestrator.collect_resorts()

        df = orchestrator.consolidate_locations()

        assert df["location_id"].nunique() == len(df)


# --- Test: Full Orchestration ---


class TestFullOrchestration:
    """Test full orchestration workflow."""

    def test_collect_locations_success(
        self, orchestrator, mock_snotel_provider, mock_resort_provider,
        mock_snotel_stations, mock_resorts
    ):
        """Full location collection succeeds."""
        orchestrator.register_station_provider("snotel", mock_snotel_provider)
        orchestrator.register_resort_provider("openskimap", mock_resort_provider)

        result = orchestrator.collect_locations()

        assert result.success is True
        assert result.stations_df is not None
        assert result.resorts_df is not None
        assert result.consolidated_locations is not None
        assert len(result.stations_df) == len(mock_snotel_stations)
        assert len(result.resorts_df) == len(mock_resorts)

    def test_collect_locations_with_filters(
        self, orchestrator, mock_snotel_provider, mock_resort_provider
    ):
        """Location collection with filters works."""
        orchestrator.register_station_provider("snotel", mock_snotel_provider)
        orchestrator.register_resort_provider("openskimap", mock_resort_provider)

        result = orchestrator.collect_locations(
            states=["CO"],
            min_elevation=3000,
        )

        assert result.success is True
        # Only 2 CO stations above 3000m
        assert len(result.stations_df) == 2

    def test_collect_locations_returns_validation_results(
        self, orchestrator, mock_snotel_provider, mock_resort_provider
    ):
        """Orchestration returns validation results."""
        orchestrator.register_station_provider("snotel", mock_snotel_provider)
        orchestrator.register_resort_provider("openskimap", mock_resort_provider)

        result = orchestrator.collect_locations()

        assert "stations" in result.validation_results
        assert "resorts" in result.validation_results
        assert result.validation_results["stations"].valid is True
        assert result.validation_results["resorts"].valid is True

    def test_collect_locations_returns_stats(
        self, orchestrator, mock_snotel_provider, mock_resort_provider,
        mock_snotel_stations, mock_resorts
    ):
        """Orchestration returns summary stats."""
        orchestrator.register_station_provider("snotel", mock_snotel_provider)
        orchestrator.register_resort_provider("openskimap", mock_resort_provider)

        result = orchestrator.collect_locations()

        assert result.stats["total_stations"] == len(mock_snotel_stations)
        assert result.stats["total_resorts"] == len(mock_resorts)
        assert "snotel" in result.stats["station_providers"]
        assert "openskimap" in result.stats["resort_providers"]


# --- Test: Extraction Points ---


class TestExtractionPoints:
    """Test extraction point generation."""

    def test_get_extraction_points_no_data_raises(self, orchestrator):
        """Raises ValueError if no data collected."""
        with pytest.raises(ValueError, match="No locations collected"):
            orchestrator.get_extraction_points()

    def test_get_extraction_points_returns_tuples(
        self, orchestrator, mock_snotel_provider, mock_snotel_stations
    ):
        """Returns list of (lat, lon) tuples."""
        orchestrator.register_station_provider("snotel", mock_snotel_provider)
        orchestrator.collect_stations()

        points = orchestrator.get_extraction_points()

        assert len(points) == len(mock_snotel_stations)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in points)
        # Check first point matches first station
        assert points[0] == (mock_snotel_stations[0].lat, mock_snotel_stations[0].lon)


# --- Test: Training Scaffold ---


class TestTrainingScaffold:
    """Test training data scaffold creation."""

    def test_create_scaffold_no_data_raises(self, orchestrator):
        """Raises ValueError if no data collected."""
        with pytest.raises(ValueError, match="No locations collected"):
            orchestrator.create_training_scaffold("2024-01-01", "2024-01-31")

    def test_create_scaffold_structure(
        self, orchestrator, mock_snotel_provider, mock_snotel_stations
    ):
        """Scaffold has correct structure."""
        orchestrator.register_station_provider("snotel", mock_snotel_provider)
        orchestrator.collect_stations()

        df = orchestrator.create_training_scaffold("2024-01-01", "2024-01-05")

        # Should have location_id and datetime in MultiIndex
        assert df.index.names == ["location_id", "datetime"]

        # 3 stations x 5 days = 15 rows
        assert len(df) == len(mock_snotel_stations) * 5

        # Check expected columns exist
        assert "snow_depth_cm" in df.columns
        assert "temp_2m" in df.columns
        assert "dem_elevation" in df.columns


# --- Test: Cache Management ---


class TestCacheManagement:
    """Test cache clearing functionality."""

    def test_clear_cache_resets_data(
        self, orchestrator, mock_snotel_provider, mock_resort_provider
    ):
        """Clear cache removes collected data."""
        orchestrator.register_station_provider("snotel", mock_snotel_provider)
        orchestrator.register_resort_provider("openskimap", mock_resort_provider)
        orchestrator.collect_stations()
        orchestrator.collect_resorts()

        # Verify data exists
        assert orchestrator._collected_stations is not None
        assert orchestrator._collected_resorts is not None

        # Clear and verify
        orchestrator.clear_cache()

        assert orchestrator._collected_stations is None
        assert orchestrator._collected_resorts is None


# --- Test: LocationPoint Dataclass ---


class TestLocationPoint:
    """Test LocationPoint dataclass."""

    def test_location_point_creation(self):
        """Can create LocationPoint with all fields."""
        point = LocationPoint(
            id="test_001",
            name="Test Station",
            lat=40.0,
            lon=-105.0,
            elevation=3000.0,
            source="snotel",
            metadata={"state": "CO"},
        )

        assert point.id == "test_001"
        assert point.name == "Test Station"
        assert point.lat == 40.0
        assert point.lon == -105.0
        assert point.elevation == 3000.0
        assert point.source == "snotel"
        assert point.metadata == {"state": "CO"}

    def test_location_point_defaults(self):
        """LocationPoint has sensible defaults."""
        point = LocationPoint(
            id="test_001",
            name="Test",
            lat=40.0,
            lon=-105.0,
        )

        assert point.elevation is None
        assert point.source == ""
        assert point.metadata == {}


# --- Test: OrchestrationResult Dataclass ---


class TestOrchestrationResult:
    """Test OrchestrationResult dataclass."""

    def test_result_success_message(self):
        """OrchestrationResult stores success info."""
        result = OrchestrationResult(
            success=True,
            message="All good",
            stations_df=pd.DataFrame({"a": [1, 2]}),
            resorts_df=pd.DataFrame({"b": [3, 4]}),
        )

        assert result.success is True
        assert result.message == "All good"
        assert len(result.stations_df) == 2
        assert len(result.resorts_df) == 2

    def test_result_defaults(self):
        """OrchestrationResult has sensible defaults."""
        result = OrchestrationResult(success=False, message="Failed")

        assert result.stations_df is None
        assert result.resorts_df is None
        assert result.consolidated_locations is None
        assert result.validation_results == {}
        assert result.stats == {}


# --- Test: Error Handling ---


class TestErrorHandling:
    """Test error handling in orchestration."""

    def test_provider_error_logged_not_raised(
        self, orchestrator, caplog
    ):
        """Provider errors are logged but don't crash orchestration."""
        # Create a provider that raises an exception
        bad_provider = MagicMock(spec=StationProvider)
        bad_provider.get_station_metadata = MagicMock(
            side_effect=RuntimeError("Network error")
        )

        orchestrator.register_station_provider("bad", bad_provider)

        # Should return empty DataFrame, not raise
        df = orchestrator.collect_stations()
        assert len(df) == 0

    def test_partial_failure_still_returns_results(
        self, orchestrator, mock_snotel_provider, mock_snotel_stations
    ):
        """If one provider fails, others still work."""
        # Good provider
        orchestrator.register_station_provider("snotel", mock_snotel_provider)

        # Bad provider
        bad_provider = MagicMock(spec=StationProvider)
        bad_provider.get_station_metadata = MagicMock(
            side_effect=RuntimeError("Boom")
        )
        orchestrator.register_station_provider("bad", bad_provider)

        df = orchestrator.collect_stations()

        # Should still have SNOTEL stations
        assert len(df) == len(mock_snotel_stations)
        assert all(df["source"] == "snotel")
