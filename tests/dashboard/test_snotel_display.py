"""Tests for SNOTEL display component."""

from datetime import datetime

import pytest

from snowforecast.dashboard.components.snotel_display import (
    MOCK_SNOTEL_STATIONS,
    SnotelStation,
    _hex_to_rgb,
    calculate_pct_of_normal,
    create_snotel_map_layer,
    get_nearby_snotel_stations,
    get_snowpack_status,
)


class TestCalculatePctOfNormal:
    """Tests for calculate_pct_of_normal function."""

    def test_normal_calculation(self):
        """Should correctly calculate percentage of normal."""
        assert calculate_pct_of_normal(450, 400) == 112.5

    def test_below_normal(self):
        """Should return percentage below 100 for below-normal snowpack."""
        assert calculate_pct_of_normal(300, 400) == 75.0

    def test_exactly_normal(self):
        """Should return 100 when current equals normal."""
        assert calculate_pct_of_normal(400, 400) == 100.0

    def test_zero_normal(self):
        """Should return 0 when normal is zero."""
        assert calculate_pct_of_normal(100, 0) == 0.0

    def test_negative_normal(self):
        """Should return 0 when normal is negative."""
        assert calculate_pct_of_normal(100, -50) == 0.0

    def test_zero_current(self):
        """Should return 0 when current is zero."""
        assert calculate_pct_of_normal(0, 400) == 0.0

    def test_very_high_percentage(self):
        """Should handle very high percentages."""
        result = calculate_pct_of_normal(1000, 400)
        assert result == 250.0

    def test_small_values(self):
        """Should handle small values correctly."""
        result = calculate_pct_of_normal(10, 8)
        assert result == 125.0


class TestGetSnowpackStatus:
    """Tests for get_snowpack_status function."""

    def test_above_normal(self):
        """Should return 'Above Normal' for >120%."""
        status, color = get_snowpack_status(125.0)
        assert status == "Above Normal"
        assert color == "#3B82F6"  # Blue

    def test_exactly_120(self):
        """Should return 'Normal' at exactly 120%."""
        status, color = get_snowpack_status(120.0)
        assert status == "Normal"
        assert color == "#22C55E"  # Green

    def test_normal_range_upper(self):
        """Should return 'Normal' for 90-120%."""
        status, color = get_snowpack_status(110.0)
        assert status == "Normal"
        assert color == "#22C55E"

    def test_normal_range_lower(self):
        """Should return 'Normal' at exactly 90%."""
        status, color = get_snowpack_status(90.0)
        assert status == "Normal"
        assert color == "#22C55E"

    def test_below_normal_range(self):
        """Should return 'Below Normal' for 70-90%."""
        status, color = get_snowpack_status(80.0)
        assert status == "Below Normal"
        assert color == "#EAB308"  # Yellow

    def test_exactly_70(self):
        """Should return 'Below Normal' at exactly 70%."""
        status, color = get_snowpack_status(70.0)
        assert status == "Below Normal"
        assert color == "#EAB308"

    def test_well_below_normal(self):
        """Should return 'Well Below Normal' for <70%."""
        status, color = get_snowpack_status(50.0)
        assert status == "Well Below Normal"
        assert color == "#EF4444"  # Red

    def test_zero_percent(self):
        """Should handle 0%."""
        status, color = get_snowpack_status(0.0)
        assert status == "Well Below Normal"
        assert color == "#EF4444"

    def test_very_high_percent(self):
        """Should handle very high percentages."""
        status, color = get_snowpack_status(200.0)
        assert status == "Above Normal"
        assert color == "#3B82F6"

    def test_negative_percent(self):
        """Should handle negative percentages (edge case)."""
        status, color = get_snowpack_status(-10.0)
        assert status == "Well Below Normal"
        assert color == "#EF4444"


class TestGetNearbySnotelStations:
    """Tests for get_nearby_snotel_stations function."""

    @pytest.fixture
    def sample_stations(self):
        """Create sample stations for testing."""
        return [
            SnotelStation(
                "1", "Station A", 40.0, -111.0, 2500,
                400, 100, 350, 114.3, datetime.now()
            ),
            SnotelStation(
                "2", "Station B", 40.1, -111.1, 2600,
                300, 80, 350, 85.7, datetime.now()
            ),
            SnotelStation(
                "3", "Station C", 41.0, -112.0, 2700,  # Far away
                500, 130, 450, 111.1, datetime.now()
            ),
        ]

    def test_finds_nearby_stations(self, sample_stations):
        """Should find stations within radius."""
        nearby = get_nearby_snotel_stations(
            40.05, -111.05, radius_km=50, stations=sample_stations
        )
        # Stations A and B should be within 50km of (40.05, -111.05)
        assert len(nearby) >= 2
        station_ids = [s.station_id for s in nearby]
        assert "1" in station_ids
        assert "2" in station_ids

    def test_excludes_distant_stations(self, sample_stations):
        """Should exclude stations outside radius."""
        nearby = get_nearby_snotel_stations(
            40.05, -111.05, radius_km=50, stations=sample_stations
        )
        station_ids = [s.station_id for s in nearby]
        # Station C is far away
        assert "3" not in station_ids

    def test_sorted_by_distance(self, sample_stations):
        """Should return stations sorted by distance."""
        nearby = get_nearby_snotel_stations(
            40.0, -111.0, radius_km=100, stations=sample_stations
        )
        if len(nearby) >= 2:
            # First station should be closest (Station A at exact location)
            assert nearby[0].station_id == "1"

    def test_empty_result_when_no_nearby(self, sample_stations):
        """Should return empty list when no stations nearby."""
        nearby = get_nearby_snotel_stations(
            35.0, -100.0, radius_km=10, stations=sample_stations
        )
        assert len(nearby) == 0

    def test_uses_mock_data_when_stations_none(self):
        """Should use mock data when stations parameter is None."""
        # Use a location near Alta, UT where mock stations exist
        nearby = get_nearby_snotel_stations(40.58, -111.63, radius_km=30)
        # Should find some mock stations
        assert len(nearby) > 0

    def test_radius_zero(self, sample_stations):
        """Should handle zero radius."""
        nearby = get_nearby_snotel_stations(
            40.0, -111.0, radius_km=0, stations=sample_stations
        )
        # Only exact matches (unlikely)
        assert len(nearby) <= 1

    def test_large_radius(self, sample_stations):
        """Should find all stations with very large radius."""
        nearby = get_nearby_snotel_stations(
            40.0, -111.0, radius_km=500, stations=sample_stations
        )
        assert len(nearby) == 3


class TestHexToRgb:
    """Tests for _hex_to_rgb helper function."""

    def test_blue(self):
        """Should convert blue hex to RGB."""
        assert _hex_to_rgb("#3B82F6") == [59, 130, 246]

    def test_green(self):
        """Should convert green hex to RGB."""
        assert _hex_to_rgb("#22C55E") == [34, 197, 94]

    def test_yellow(self):
        """Should convert yellow hex to RGB."""
        assert _hex_to_rgb("#EAB308") == [234, 179, 8]

    def test_red(self):
        """Should convert red hex to RGB."""
        assert _hex_to_rgb("#EF4444") == [239, 68, 68]

    def test_without_hash(self):
        """Should handle hex without leading #."""
        assert _hex_to_rgb("3B82F6") == [59, 130, 246]


class TestCreateSnotelMapLayer:
    """Tests for create_snotel_map_layer function."""

    @pytest.fixture
    def sample_stations(self):
        """Create sample stations for testing."""
        return [
            SnotelStation(
                "1", "Station A", 40.0, -111.0, 2500,
                400, 100, 350, 114.3, datetime.now()
            ),
            SnotelStation(
                "2", "Station B", 40.1, -111.1, 2600,
                300, 80, 350, 85.7, datetime.now()
            ),
        ]

    def test_returns_pydeck_layer(self, sample_stations):
        """Should return a PyDeck layer."""
        import pydeck as pdk
        layer = create_snotel_map_layer(sample_stations)
        assert isinstance(layer, pdk.Layer)

    def test_layer_type(self, sample_stations):
        """Should create a ScatterplotLayer."""
        layer = create_snotel_map_layer(sample_stations)
        assert layer.type == "ScatterplotLayer"

    def test_layer_is_pickable(self, sample_stations):
        """Layer should be pickable for tooltips."""
        layer = create_snotel_map_layer(sample_stations)
        assert layer.pickable is True

    def test_empty_stations(self):
        """Should handle empty station list."""
        layer = create_snotel_map_layer([])
        assert isinstance(layer.data, list)
        assert len(layer.data) == 0

    def test_data_has_required_fields(self, sample_stations):
        """Layer data should have all required fields."""
        layer = create_snotel_map_layer(sample_stations)
        for item in layer.data:
            assert "station_id" in item
            assert "name" in item
            assert "latitude" in item
            assert "longitude" in item
            assert "elevation_m" in item
            assert "current_swe_mm" in item
            assert "current_depth_cm" in item
            assert "pct_of_normal" in item
            assert "color" in item

    def test_colors_based_on_status(self, sample_stations):
        """Colors should be based on snowpack status."""
        layer = create_snotel_map_layer(sample_stations)

        # Station A has 114.3% - Normal (green)
        station_a = next(d for d in layer.data if d["station_id"] == "1")
        green_rgb = [34, 197, 94, 200]  # #22C55E with alpha
        assert station_a["color"] == green_rgb

        # Station B has 85.7% - Below Normal (yellow)
        station_b = next(d for d in layer.data if d["station_id"] == "2")
        yellow_rgb = [234, 179, 8, 200]  # #EAB308 with alpha
        assert station_b["color"] == yellow_rgb


class TestSnotelStation:
    """Tests for SnotelStation dataclass."""

    def test_create_station(self):
        """Should create station with all fields."""
        station = SnotelStation(
            station_id="322",
            name="Alta Guard Station",
            lat=40.58,
            lon=-111.63,
            elevation_m=2661,
            current_swe_mm=450,
            current_depth_cm=120,
            normal_swe_mm=400,
            pct_of_normal=112.5,
            last_updated=datetime.now(),
        )
        assert station.station_id == "322"
        assert station.name == "Alta Guard Station"
        assert station.lat == 40.58
        assert station.lon == -111.63
        assert station.elevation_m == 2661
        assert station.current_swe_mm == 450
        assert station.current_depth_cm == 120
        assert station.normal_swe_mm == 400
        assert station.pct_of_normal == 112.5

    def test_station_equality(self):
        """Stations with same values should be equal."""
        now = datetime.now()
        station1 = SnotelStation(
            "1", "Test", 40.0, -111.0, 2500,
            400, 100, 350, 114.3, now
        )
        station2 = SnotelStation(
            "1", "Test", 40.0, -111.0, 2500,
            400, 100, 350, 114.3, now
        )
        assert station1 == station2


class TestMockData:
    """Tests for mock SNOTEL data."""

    def test_mock_stations_not_empty(self):
        """Mock stations list should not be empty."""
        assert len(MOCK_SNOTEL_STATIONS) > 0

    def test_mock_stations_have_valid_data(self):
        """Mock stations should have valid field values."""
        for station in MOCK_SNOTEL_STATIONS:
            assert station.station_id
            assert station.name
            assert -90 <= station.lat <= 90
            assert -180 <= station.lon <= 180
            assert station.elevation_m > 0
            assert station.current_swe_mm >= 0
            assert station.current_depth_cm >= 0
            assert station.normal_swe_mm >= 0
            assert station.pct_of_normal >= 0

    def test_mock_stations_cover_multiple_regions(self):
        """Mock stations should cover multiple ski regions."""
        # Check that stations span different latitudes/longitudes
        lats = [s.lat for s in MOCK_SNOTEL_STATIONS]
        lons = [s.lon for s in MOCK_SNOTEL_STATIONS]

        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)

        # Should cover at least 3 degrees of latitude and longitude
        assert lat_range >= 3, "Mock stations should span at least 3 degrees latitude"
        assert lon_range >= 3, "Mock stations should span at least 3 degrees longitude"


class TestIntegration:
    """Integration tests for SNOTEL display components."""

    def test_full_workflow(self):
        """Test the full workflow from nearby stations to map layer."""
        # Get nearby stations for Alta, UT
        nearby = get_nearby_snotel_stations(40.58, -111.63, radius_km=30)

        # Should find some stations
        assert len(nearby) > 0

        # Get status for each
        for station in nearby:
            pct = calculate_pct_of_normal(
                station.current_swe_mm,
                station.normal_swe_mm
            )
            status, color = get_snowpack_status(pct)
            assert status in [
                "Above Normal",
                "Normal",
                "Below Normal",
                "Well Below Normal"
            ]
            assert color.startswith("#")

        # Create map layer
        layer = create_snotel_map_layer(nearby)
        assert len(layer.data) == len(nearby)
