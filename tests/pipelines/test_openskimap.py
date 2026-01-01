"""Tests for the OpenSkiMap ski resort extraction pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from snowforecast.pipelines.openskimap import (
    OpenSkiMapPipeline,
    SkiResort,
    SKI_AREAS_URL,
    LIFTS_URL,
)
from snowforecast.utils.geo import haversine


# Sample GeoJSON data for testing
SAMPLE_SKI_AREAS_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-105.9, 39.6],
                    [-105.8, 39.6],
                    [-105.8, 39.7],
                    [-105.9, 39.7],
                    [-105.9, 39.6],
                ]]
            },
            "properties": {
                "name": "Test Mountain Resort",
                "location": {
                    "localized": {
                        "en": {
                            "country": {"name": "United States"},
                            "region": {"name": "Colorado"},
                        }
                    }
                }
            }
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-111.6, 40.5],
                    [-111.5, 40.5],
                    [-111.5, 40.6],
                    [-111.6, 40.6],
                    [-111.6, 40.5],
                ]]
            },
            "properties": {
                "name": "Powder Valley Ski Area",
                "location": {
                    "localized": {
                        "en": {
                            "country": {"name": "United States"},
                            "region": {"name": "Utah"},
                        }
                    }
                }
            }
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-80.0, 35.0],
                    [-79.9, 35.0],
                    [-79.9, 35.1],
                    [-80.0, 35.1],
                    [-80.0, 35.0],
                ]]
            },
            "properties": {
                "name": "East Coast Ski Area",
                "location": {
                    "localized": {
                        "en": {
                            "country": {"name": "United States"},
                            "region": {"name": "North Carolina"},
                        }
                    }
                }
            }
        },
    ]
}

SAMPLE_LIFTS_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [-105.85, 39.65, 2800],
                    [-105.85, 39.68, 3500],
                ]
            },
            "properties": {
                "name": "Main Lift",
                "ski_areas": [
                    {
                        "type": "Feature",
                        "properties": {"name": "Test Mountain Resort"}
                    }
                ]
            }
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [-105.87, 39.66, 2900],
                    [-105.87, 39.69, 3400],
                ]
            },
            "properties": {
                "name": "Secondary Lift",
                "ski_areas": [
                    {
                        "type": "Feature",
                        "properties": {"name": "Test Mountain Resort"}
                    }
                ]
            }
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [-111.55, 40.55, 2400],
                    [-111.55, 40.58, 3200],
                ]
            },
            "properties": {
                "name": "Utah Lift",
                "ski_areas": [
                    {
                        "type": "Feature",
                        "properties": {"name": "Powder Valley Ski Area"}
                    }
                ]
            }
        },
    ]
}


@pytest.fixture
def pipeline_with_mock_paths(tmp_path):
    """Create a pipeline with mocked data paths."""
    pipeline = OpenSkiMapPipeline()
    pipeline.raw_path = tmp_path / "raw"
    pipeline.processed_path = tmp_path / "processed"
    pipeline.raw_path.mkdir(parents=True, exist_ok=True)
    pipeline.processed_path.mkdir(parents=True, exist_ok=True)
    return pipeline


@pytest.fixture
def sample_geojson_files(pipeline_with_mock_paths):
    """Create sample GeoJSON files in the pipeline's raw path."""
    pipeline = pipeline_with_mock_paths

    # Write ski areas GeoJSON
    with open(pipeline.raw_path / "ski_areas.geojson", "w") as f:
        json.dump(SAMPLE_SKI_AREAS_GEOJSON, f)

    # Write lifts GeoJSON
    with open(pipeline.raw_path / "lifts.geojson", "w") as f:
        json.dump(SAMPLE_LIFTS_GEOJSON, f)

    return pipeline


@pytest.fixture
def sample_snotel_stations():
    """Create sample SNOTEL station DataFrame."""
    return pd.DataFrame([
        {"station_id": "1050:CO:SNTL", "lat": 39.80, "lon": -105.78},
        {"station_id": "1051:CO:SNTL", "lat": 39.68, "lon": -105.90},
        {"station_id": "978:UT:SNTL", "lat": 40.60, "lon": -111.58},
    ])


class TestSkiResortDataclass:
    """Tests for the SkiResort dataclass."""

    def test_ski_resort_creation(self):
        """Should create a SkiResort with all fields."""
        resort = SkiResort(
            name="Test Resort",
            lat=39.5,
            lon=-105.8,
            base_elevation=2800.0,
            summit_elevation=3500.0,
            vertical_drop=700.0,
            country="US",
            state="CO",
            nearest_snotel="1050:CO:SNTL",
        )

        assert resort.name == "Test Resort"
        assert resort.lat == 39.5
        assert resort.lon == -105.8
        assert resort.base_elevation == 2800.0
        assert resort.summit_elevation == 3500.0
        assert resort.vertical_drop == 700.0
        assert resort.country == "US"
        assert resort.state == "CO"
        assert resort.nearest_snotel == "1050:CO:SNTL"

    def test_ski_resort_optional_fields(self):
        """Should allow None for optional elevation fields."""
        resort = SkiResort(
            name="Test Resort",
            lat=39.5,
            lon=-105.8,
            base_elevation=None,
            summit_elevation=None,
            vertical_drop=None,
            country="US",
            state="CO",
        )

        assert resort.base_elevation is None
        assert resort.summit_elevation is None
        assert resort.vertical_drop is None
        assert resort.nearest_snotel is None


class TestDownloadSkiAreas:
    """Tests for downloading ski areas GeoJSON."""

    @patch("snowforecast.pipelines.openskimap.requests.get")
    def test_download_ski_areas(self, mock_get, pipeline_with_mock_paths):
        """Should download ski areas GeoJSON."""
        mock_response = Mock()
        mock_response.text = json.dumps(SAMPLE_SKI_AREAS_GEOJSON)
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        path = pipeline_with_mock_paths.download_ski_areas()

        assert path.exists()
        assert path.name == "ski_areas.geojson"
        mock_get.assert_called_once()
        assert SKI_AREAS_URL in str(mock_get.call_args)

    @patch("snowforecast.pipelines.openskimap.requests.get")
    def test_download_lifts(self, mock_get, pipeline_with_mock_paths):
        """Should download lifts GeoJSON."""
        mock_response = Mock()
        mock_response.text = json.dumps(SAMPLE_LIFTS_GEOJSON)
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        path = pipeline_with_mock_paths.download_lifts()

        assert path.exists()
        assert path.name == "lifts.geojson"
        mock_get.assert_called_once()
        assert LIFTS_URL in str(mock_get.call_args)


class TestParseSkiAreas:
    """Tests for parsing ski area features."""

    def test_parse_ski_areas(self, sample_geojson_files):
        """Should parse ski area features correctly."""
        pipeline = sample_geojson_files
        ski_areas_path = pipeline.raw_path / "ski_areas.geojson"

        features = pipeline.parse_ski_areas(ski_areas_path)

        assert len(features) == 3

        # Check first feature
        test_mountain = next(f for f in features if f["name"] == "Test Mountain Resort")
        assert test_mountain["lat"] == pytest.approx(39.65, abs=0.1)
        assert test_mountain["lon"] == pytest.approx(-105.85, abs=0.1)

    def test_parse_ski_areas_filters_nameless(self, pipeline_with_mock_paths):
        """Should skip ski areas without names."""
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[-105.9, 39.6], [-105.8, 39.6], [-105.8, 39.7], [-105.9, 39.7], [-105.9, 39.6]]]
                    },
                    "properties": {}  # No name
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[-105.9, 39.6], [-105.8, 39.6], [-105.8, 39.7], [-105.9, 39.7], [-105.9, 39.6]]]
                    },
                    "properties": {"name": "Valid Resort"}
                },
            ]
        }

        path = pipeline_with_mock_paths.raw_path / "test.geojson"
        with open(path, "w") as f:
            json.dump(geojson_data, f)

        features = pipeline_with_mock_paths.parse_ski_areas(path)

        assert len(features) == 1
        assert features[0]["name"] == "Valid Resort"


class TestExtractElevations:
    """Tests for extracting elevations from lifts."""

    def test_extract_elevations_from_lifts(self, sample_geojson_files):
        """Should get base/summit from lifts."""
        pipeline = sample_geojson_files

        with open(pipeline.raw_path / "lifts.geojson") as f:
            lifts_data = json.load(f)

        base, summit = pipeline.extract_elevations_from_lifts(
            "Test Mountain Resort", lifts_data
        )

        # From SAMPLE_LIFTS_GEOJSON, Test Mountain Resort has lifts from 2800-3500 and 2900-3400
        assert base == 2800
        assert summit == 3500

    def test_extract_elevations_no_match(self, sample_geojson_files):
        """Should return None when no lifts match."""
        pipeline = sample_geojson_files

        with open(pipeline.raw_path / "lifts.geojson") as f:
            lifts_data = json.load(f)

        base, summit = pipeline.extract_elevations_from_lifts(
            "Nonexistent Resort", lifts_data
        )

        assert base is None
        assert summit is None

    def test_extract_elevations_for_utah_resort(self, sample_geojson_files):
        """Should correctly extract elevations for different resorts."""
        pipeline = sample_geojson_files

        with open(pipeline.raw_path / "lifts.geojson") as f:
            lifts_data = json.load(f)

        base, summit = pipeline.extract_elevations_from_lifts(
            "Powder Valley Ski Area", lifts_data
        )

        assert base == 2400
        assert summit == 3200


class TestFilterRegion:
    """Tests for filtering resorts by region."""

    def test_filter_western_us(self):
        """Should filter to Western US only."""
        resorts = [
            SkiResort("Western Resort", 39.5, -105.8, None, None, None, "US", "CO"),
            SkiResort("Eastern Resort", 35.0, -79.9, None, None, None, "US", "NC"),
            SkiResort("Canada Resort", 50.0, -115.0, None, None, None, "CA", "BC"),
        ]

        pipeline = OpenSkiMapPipeline()
        bbox = {"west": -125.0, "east": -102.0, "south": 31.0, "north": 49.0}

        filtered = pipeline.filter_region(resorts, bbox=bbox)

        assert len(filtered) == 1
        assert filtered[0].name == "Western Resort"

    def test_filter_by_country(self):
        """Should filter by country code."""
        resorts = [
            SkiResort("US Resort", 39.5, -105.8, None, None, None, "US", "CO"),
            SkiResort("Canada Resort", 50.0, -115.0, None, None, None, "CA", "BC"),
        ]

        pipeline = OpenSkiMapPipeline()

        filtered = pipeline.filter_region(resorts, countries=["US"])

        assert len(filtered) == 1
        assert filtered[0].name == "US Resort"

    def test_filter_combined(self):
        """Should filter by both bbox and country."""
        resorts = [
            SkiResort("Western US", 39.5, -105.8, None, None, None, "US", "CO"),
            SkiResort("Eastern US", 35.0, -79.9, None, None, None, "US", "NC"),
            SkiResort("Western CA", 50.0, -115.0, None, None, None, "CA", "BC"),
        ]

        pipeline = OpenSkiMapPipeline()
        bbox = {"west": -125.0, "east": -102.0, "south": 31.0, "north": 49.0}

        filtered = pipeline.filter_region(resorts, bbox=bbox, countries=["US"])

        assert len(filtered) == 1
        assert filtered[0].name == "Western US"


class TestFindNearestSnotel:
    """Tests for finding nearest SNOTEL station."""

    def test_find_nearest_snotel(self, sample_snotel_stations):
        """Should find closest SNOTEL station."""
        resort = SkiResort(
            "Test Resort", 39.70, -105.80, None, None, None, "US", "CO"
        )

        pipeline = OpenSkiMapPipeline()
        nearest = pipeline.find_nearest_snotel(resort, sample_snotel_stations)

        # Should find a nearby Colorado station
        assert nearest is not None
        assert "CO:SNTL" in nearest

    def test_find_nearest_snotel_max_distance(self, sample_snotel_stations):
        """Should return None when no station within max distance."""
        # Resort far from any station
        resort = SkiResort(
            "Remote Resort", 45.0, -120.0, None, None, None, "US", "WA"
        )

        pipeline = OpenSkiMapPipeline()
        nearest = pipeline.find_nearest_snotel(
            resort, sample_snotel_stations, max_distance_km=10
        )

        assert nearest is None

    def test_find_nearest_snotel_empty_df(self):
        """Should return None when no stations available."""
        resort = SkiResort(
            "Test Resort", 39.70, -105.80, None, None, None, "US", "CO"
        )

        pipeline = OpenSkiMapPipeline()
        nearest = pipeline.find_nearest_snotel(resort, pd.DataFrame())

        assert nearest is None


class TestHaversineDistance:
    """Tests for haversine distance calculation."""

    def test_haversine_distance(self):
        """Should calculate distance correctly."""
        # Denver to Boulder is approximately 40 km
        denver = (39.7392, -104.9903)
        boulder = (40.0150, -105.2705)

        distance = haversine(denver[0], denver[1], boulder[0], boulder[1])

        assert 35 < distance < 45  # Approximately 40 km

    def test_haversine_same_point(self):
        """Should return 0 for same point."""
        distance = haversine(39.7392, -104.9903, 39.7392, -104.9903)

        assert distance == pytest.approx(0.0, abs=0.001)

    def test_haversine_known_distance(self):
        """Test with known distance between cities."""
        # New York to Los Angeles is approximately 3940 km
        ny = (40.7128, -74.0060)
        la = (34.0522, -118.2437)

        distance = haversine(ny[0], ny[1], la[0], la[1])

        assert 3900 < distance < 4000


class TestExportDataframe:
    """Tests for exporting to DataFrame."""

    def test_export_dataframe(self):
        """Should create valid DataFrame."""
        resorts = [
            SkiResort("Resort A", 39.5, -105.8, 2800.0, 3500.0, 700.0, "US", "CO", "1050:CO:SNTL"),
            SkiResort("Resort B", 40.5, -111.5, 2400.0, 3200.0, 800.0, "US", "UT", "978:UT:SNTL"),
        ]

        pipeline = OpenSkiMapPipeline()
        df = pipeline.export_to_dataframe(resorts)

        assert len(df) == 2
        assert list(df.columns) == [
            "name", "lat", "lon", "base_elevation", "summit_elevation",
            "vertical_drop", "country", "state", "nearest_snotel"
        ]
        assert df.iloc[0]["name"] == "Resort A"
        assert df.iloc[1]["name"] == "Resort B"

    def test_export_empty_list(self):
        """Should handle empty resort list."""
        pipeline = OpenSkiMapPipeline()
        df = pipeline.export_to_dataframe([])

        assert len(df) == 0
        assert "name" in df.columns
        assert "lat" in df.columns


class TestValidation:
    """Tests for data validation."""

    def test_validate_valid_data(self):
        """Should validate correct data."""
        df = pd.DataFrame({
            "name": ["Resort A", "Resort B"],
            "lat": [39.5, 40.5],
            "lon": [-105.8, -111.5],
            "base_elevation": [2800.0, 2400.0],
            "summit_elevation": [3500.0, 3200.0],
            "vertical_drop": [700.0, 800.0],
            "country": ["US", "US"],
            "state": ["CO", "UT"],
            "nearest_snotel": [None, None],
        })

        pipeline = OpenSkiMapPipeline()
        result = pipeline.validate(df)

        assert result.valid is True
        assert result.total_rows == 2
        assert result.missing_pct < 1.0

    def test_validate_empty_dataframe(self):
        """Should reject empty DataFrame."""
        pipeline = OpenSkiMapPipeline()
        result = pipeline.validate(pd.DataFrame())

        assert result.valid is False
        assert "No resorts found" in result.issues

    def test_validate_invalid_coordinates(self):
        """Should detect invalid coordinates."""
        df = pd.DataFrame({
            "name": ["Resort A"],
            "lat": [100.0],  # Invalid latitude
            "lon": [-105.8],
            "base_elevation": [2800.0],
            "summit_elevation": [3500.0],
            "vertical_drop": [700.0],
            "country": ["US"],
            "state": ["CO"],
            "nearest_snotel": [None],
        })

        pipeline = OpenSkiMapPipeline()
        result = pipeline.validate(df)

        assert result.valid is False
        assert any("invalid latitude" in issue for issue in result.issues)

    def test_validate_non_dataframe(self):
        """Should reject non-DataFrame input."""
        pipeline = OpenSkiMapPipeline()
        result = pipeline.validate([1, 2, 3])

        assert result.valid is False
        assert "not a DataFrame" in result.issues[0]


class TestGetWesternUSResorts:
    """Tests for the main resort extraction method."""

    def test_get_western_us_resorts(self, sample_geojson_files):
        """Should return Western US resorts with elevations."""
        pipeline = sample_geojson_files

        resorts = pipeline.get_western_us_resorts()

        # Should only include Western US resorts (not East Coast)
        assert len(resorts) == 2

        resort_names = [r.name for r in resorts]
        assert "Test Mountain Resort" in resort_names
        assert "Powder Valley Ski Area" in resort_names
        assert "East Coast Ski Area" not in resort_names

    def test_resorts_have_elevations(self, sample_geojson_files):
        """Should extract elevation data from lifts."""
        pipeline = sample_geojson_files

        resorts = pipeline.get_western_us_resorts()

        test_mountain = next(r for r in resorts if r.name == "Test Mountain Resort")
        assert test_mountain.base_elevation == 2800
        assert test_mountain.summit_elevation == 3500
        assert test_mountain.vertical_drop == 700


class TestSaveToParquet:
    """Tests for saving to parquet format."""

    def test_save_to_parquet(self, pipeline_with_mock_paths):
        """Should save DataFrame to parquet file."""
        df = pd.DataFrame({
            "name": ["Resort A"],
            "lat": [39.5],
            "lon": [-105.8],
            "base_elevation": [2800.0],
            "summit_elevation": [3500.0],
            "vertical_drop": [700.0],
            "country": ["US"],
            "state": ["CO"],
            "nearest_snotel": [None],
        })

        path = pipeline_with_mock_paths.save_to_parquet(df)

        assert path.exists()
        assert path.suffix == ".parquet"

        # Verify we can read it back
        loaded = pd.read_parquet(path)
        assert len(loaded) == 1
        assert loaded.iloc[0]["name"] == "Resort A"
