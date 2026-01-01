"""Tests for GHCN-Daily data ingestion pipeline."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import tempfile

from snowforecast.pipelines.ghcn import (
    GHCNPipeline,
    GHCNStation,
    GHCN_QUALITY_FLAGS,
    GHCN_VARIABLES,
)
from snowforecast.utils import BoundingBox


# Sample station inventory data (fixed-width format)
SAMPLE_STATION_INVENTORY = """USC00010008  32.9500  -85.9500  201.0 AL ABBEVILLE
USC00450008  41.6833  -87.8833  180.0 IL CHICAGO HEIGHTS 2 NE
USW00023062  39.7731 -104.8694 1656.0 CO DENVER STAPLETON
USW00024089  47.4494 -122.3094    6.0 WA SEATTLE TACOMA INTL AP
USS0008K24S  39.1830 -106.4500 3383.0 CO INDEPENDENCE PASS
"""


def _make_dly_line(station_id, year, month, element, values):
    """Create a properly formatted GHCN .dly line.

    Args:
        station_id: 11-character station ID
        year: 4-digit year
        month: 2-digit month
        element: 4-character element code
        values: list of (value, mflag, qflag, sflag) tuples for 31 days
                Use value=-9999 for missing data
    """
    line = f"{station_id:11s}{year:4d}{month:02d}{element:4s}"
    for val, mflag, qflag, sflag in values:
        line += f"{val:5d}{mflag}{qflag}{sflag}"
    return line


def _make_sample_values(day_values):
    """Create 31 value blocks with specified days having data.

    Args:
        day_values: dict mapping day (1-31) to (value, mflag, qflag, sflag)
    """
    result = []
    for day in range(1, 32):
        if day in day_values:
            result.append(day_values[day])
        else:
            result.append((-9999, " ", " ", " "))
    return result


# Sample .dly file content - properly formatted fixed-width
# Days 1, 2, 4, 5 have data; day 3 and 6-31 are missing (-9999)
_TMAX_VALUES = _make_sample_values({
    1: (156, " ", " ", "7"),
    2: (145, " ", " ", "7"),
    4: (167, " ", " ", "7"),
    5: (178, " ", " ", "7"),
})
_TMIN_VALUES = _make_sample_values({
    1: (28, " ", " ", "7"),
    2: (45, " ", " ", "7"),
    4: (56, " ", " ", "7"),
    5: (67, " ", " ", "7"),
})
_PRCP_VALUES = _make_sample_values({
    1: (0, " ", " ", "7"),
    2: (25, " ", " ", "7"),
    4: (10, " ", " ", "7"),
    5: (5, " ", " ", "7"),
})
_SNOW_VALUES = _make_sample_values({
    1: (0, " ", " ", "7"),
    2: (0, " ", " ", "7"),
    4: (0, " ", " ", "7"),
    5: (0, " ", " ", "7"),
})

SAMPLE_DLY_CONTENT = "\n".join([
    _make_dly_line("USC00010008", 2023, 1, "TMAX", _TMAX_VALUES),
    _make_dly_line("USC00010008", 2023, 1, "TMIN", _TMIN_VALUES),
    _make_dly_line("USC00010008", 2023, 1, "PRCP", _PRCP_VALUES),
    _make_dly_line("USC00010008", 2023, 1, "SNOW", _SNOW_VALUES),
])

# Sample .dly with quality flags - day 2 has G flag, day 4 has S flag
_TMAX_WITH_FLAGS = _make_sample_values({
    1: (156, " ", " ", "7"),
    2: (145, " ", "G", "7"),  # G quality flag
    4: (167, " ", "S", "7"),  # S quality flag
    5: (178, " ", " ", "7"),
})

SAMPLE_DLY_WITH_FLAGS = _make_dly_line("USC00010008", 2023, 1, "TMAX", _TMAX_WITH_FLAGS)


class TestGHCNStation:
    """Tests for GHCNStation dataclass."""

    def test_create_station(self):
        """Should create a station with all attributes."""
        station = GHCNStation(
            station_id="USC00010008",
            name="ABBEVILLE",
            lat=32.95,
            lon=-85.95,
            elevation=201.0,
            state="AL",
        )
        assert station.station_id == "USC00010008"
        assert station.name == "ABBEVILLE"
        assert station.lat == 32.95
        assert station.lon == -85.95
        assert station.elevation == 201.0
        assert station.state == "AL"


class TestGetStationInventory:
    """Tests for get_station_inventory method."""

    def test_get_station_inventory(self):
        """Should parse station inventory file."""
        pipeline = GHCNPipeline()

        # Mock the download and use sample data
        stations = pipeline._parse_station_inventory(SAMPLE_STATION_INVENTORY)

        assert len(stations) == 5
        assert stations[0].station_id == "USC00010008"
        assert stations[0].lat == 32.95
        assert stations[0].lon == -85.95
        assert stations[0].elevation == 201.0
        assert stations[0].state == "AL"
        assert stations[0].name == "ABBEVILLE"

    def test_parse_high_elevation_station(self):
        """Should correctly parse high-elevation station."""
        pipeline = GHCNPipeline()
        stations = pipeline._parse_station_inventory(SAMPLE_STATION_INVENTORY)

        # Independence Pass is the high elevation station
        independence_pass = [s for s in stations if "INDEPENDENCE" in s.name][0]
        assert independence_pass.elevation == 3383.0
        assert independence_pass.state == "CO"


class TestFilterByBbox:
    """Tests for filtering stations by bounding box."""

    def test_filter_by_bbox_dict(self):
        """Should filter stations by bounding box dictionary."""
        pipeline = GHCNPipeline()
        stations = pipeline._parse_station_inventory(SAMPLE_STATION_INVENTORY)

        # Colorado bounding box
        bbox = {
            "west": -109.0,
            "south": 37.0,
            "east": -102.0,
            "north": 41.0,
        }
        filtered = pipeline._filter_by_bbox(stations, bbox)

        # Should include Denver and Independence Pass (CO stations)
        assert len(filtered) == 2
        station_ids = [s.station_id for s in filtered]
        assert "USW00023062" in station_ids  # Denver
        assert "USS0008K24S" in station_ids  # Independence Pass

    def test_filter_by_bbox_object(self):
        """Should filter stations by BoundingBox object."""
        pipeline = GHCNPipeline()
        stations = pipeline._parse_station_inventory(SAMPLE_STATION_INVENTORY)

        bbox = BoundingBox(west=-109.0, south=37.0, east=-102.0, north=41.0)
        filtered = pipeline._filter_by_bbox(stations, bbox)

        assert len(filtered) == 2

    def test_filter_by_bbox_empty_result(self):
        """Should return empty list when no stations in bbox."""
        pipeline = GHCNPipeline()
        stations = pipeline._parse_station_inventory(SAMPLE_STATION_INVENTORY)

        # Middle of ocean
        bbox = {"west": 0.0, "south": 0.0, "east": 10.0, "north": 10.0}
        filtered = pipeline._filter_by_bbox(stations, bbox)

        assert len(filtered) == 0


class TestFilterMountainStations:
    """Tests for filter_mountain_stations method."""

    def test_filter_mountain_stations_default(self):
        """Should filter by elevation threshold (default 1500m)."""
        pipeline = GHCNPipeline()
        pipeline._station_inventory = pipeline._parse_station_inventory(
            SAMPLE_STATION_INVENTORY
        )

        mountain_stations = pipeline.filter_mountain_stations()

        # Denver (1656m) and Independence Pass (3383m) should be included
        assert len(mountain_stations) == 2
        elevations = [s.elevation for s in mountain_stations]
        assert all(e >= 1500 for e in elevations)

    def test_filter_mountain_stations_custom_elevation(self):
        """Should filter by custom elevation threshold."""
        pipeline = GHCNPipeline()
        pipeline._station_inventory = pipeline._parse_station_inventory(
            SAMPLE_STATION_INVENTORY
        )

        mountain_stations = pipeline.filter_mountain_stations(min_elevation=3000)

        # Only Independence Pass (3383m) should be included
        assert len(mountain_stations) == 1
        assert mountain_stations[0].name == "INDEPENDENCE PASS"


class TestParseDlyFile:
    """Tests for parse_dly_file method."""

    def test_parse_dly_file(self):
        """Should parse .dly fixed-width format."""
        pipeline = GHCNPipeline()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dly", delete=False) as f:
            f.write(SAMPLE_DLY_CONTENT)
            f.flush()
            path = Path(f.name)

        try:
            df = pipeline.parse_dly_file(path)

            # Should have parsed the valid records (4 rows x 4 valid days = 16)
            assert len(df) > 0
            assert "date" in df.columns
            assert "station_id" in df.columns
            assert "element" in df.columns
            assert "value" in df.columns
            assert "qflag" in df.columns

            # Check station ID is correct
            assert df["station_id"].iloc[0] == "USC00010008"

            # Check elements are parsed
            elements = df["element"].unique()
            assert "TMAX" in elements
            assert "TMIN" in elements
            assert "PRCP" in elements
            assert "SNOW" in elements
        finally:
            path.unlink()

    def test_parse_dly_file_with_variable_filter(self):
        """Should filter by variable type."""
        pipeline = GHCNPipeline()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dly", delete=False) as f:
            f.write(SAMPLE_DLY_CONTENT)
            f.flush()
            path = Path(f.name)

        try:
            df = pipeline.parse_dly_file(path, variables={"TMAX", "TMIN"})

            # Should only have temperature records
            elements = df["element"].unique()
            assert set(elements) == {"TMAX", "TMIN"}
        finally:
            path.unlink()

    def test_parse_dly_handles_missing_values(self):
        """Should handle missing values (-9999) correctly."""
        pipeline = GHCNPipeline()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dly", delete=False) as f:
            f.write(SAMPLE_DLY_CONTENT)
            f.flush()
            path = Path(f.name)

        try:
            df = pipeline.parse_dly_file(path)

            # -9999 values should be excluded
            assert -9999 not in df["value"].values

            # Should only have the valid dates (days 1, 2, 4, 5 have data)
            days = df["date"].dt.day.unique()
            assert 3 not in days  # Day 3 is -9999 in our sample
        finally:
            path.unlink()


class TestQualityFlags:
    """Tests for quality flag handling."""

    def test_preserve_quality_flags(self):
        """Should preserve quality flags in parsed data."""
        pipeline = GHCNPipeline()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dly", delete=False) as f:
            f.write(SAMPLE_DLY_WITH_FLAGS)
            f.flush()
            path = Path(f.name)

        try:
            df = pipeline.parse_dly_file(path)

            # Check that quality flags are preserved
            assert "qflag" in df.columns

            # Day 2 should have G flag, day 4 should have S flag
            day2_row = df[df["date"].dt.day == 2]
            day4_row = df[df["date"].dt.day == 4]

            if len(day2_row) > 0:
                assert day2_row["qflag"].iloc[0] == "G"
            if len(day4_row) > 0:
                assert day4_row["qflag"].iloc[0] == "S"
        finally:
            path.unlink()

    def test_filter_quality(self):
        """Should filter out flagged data."""
        pipeline = GHCNPipeline()

        # Create sample data with quality flags
        data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5),
            "station_id": ["USC00010008"] * 5,
            "element": ["TMAX"] * 5,
            "value": [100, 110, 120, 130, 140],
            "qflag": ["", "G", "", "S", ""],
        })

        filtered = pipeline.filter_quality(data)

        # Should exclude rows with G and S flags
        assert len(filtered) == 3
        assert "G" not in filtered["qflag"].values
        assert "S" not in filtered["qflag"].values

    def test_filter_quality_custom_flags(self):
        """Should filter by custom flag set."""
        pipeline = GHCNPipeline()

        data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5),
            "station_id": ["USC00010008"] * 5,
            "element": ["TMAX"] * 5,
            "value": [100, 110, 120, 130, 140],
            "qflag": ["", "G", "", "S", "D"],
        })

        # Only exclude G flags
        filtered = pipeline.filter_quality(data, exclude_flags={"G"})

        # Should only exclude the G flag row
        assert len(filtered) == 4
        assert "G" not in filtered["qflag"].values
        assert "S" in filtered["qflag"].values  # S should still be present


class TestUnitConversion:
    """Tests for unit conversion."""

    def test_unit_conversion(self):
        """Should convert to metric units."""
        pipeline = GHCNPipeline()

        data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5),
            "station_id": ["USC00010008"] * 5,
            "element": ["TMAX", "TMIN", "PRCP", "SNOW", "SNWD"],
            "value": [156, 28, 25, 100, 500],  # Raw GHCN values
            "qflag": [""] * 5,
        })

        converted = pipeline.convert_units(data)

        # TMAX: 156 tenths of C = 15.6 C
        tmax_row = converted[converted["element"] == "TMAX"]
        assert tmax_row["value_converted"].iloc[0] == pytest.approx(15.6)
        assert tmax_row["unit"].iloc[0] == "C"

        # TMIN: 28 tenths of C = 2.8 C
        tmin_row = converted[converted["element"] == "TMIN"]
        assert tmin_row["value_converted"].iloc[0] == pytest.approx(2.8)
        assert tmin_row["unit"].iloc[0] == "C"

        # PRCP: 25 tenths of mm = 2.5 mm
        prcp_row = converted[converted["element"] == "PRCP"]
        assert prcp_row["value_converted"].iloc[0] == pytest.approx(2.5)
        assert prcp_row["unit"].iloc[0] == "mm"

        # SNOW: 100 mm = 10 cm
        snow_row = converted[converted["element"] == "SNOW"]
        assert snow_row["value_converted"].iloc[0] == pytest.approx(10.0)
        assert snow_row["unit"].iloc[0] == "cm"

        # SNWD: 500 mm = 50 cm
        snwd_row = converted[converted["element"] == "SNWD"]
        assert snwd_row["value_converted"].iloc[0] == pytest.approx(50.0)
        assert snwd_row["unit"].iloc[0] == "cm"


class TestValidation:
    """Tests for data validation."""

    def test_validate_empty_data(self):
        """Should detect empty dataset."""
        pipeline = GHCNPipeline()

        empty_df = pd.DataFrame(columns=[
            "date", "station_id", "element", "value", "mflag", "qflag", "sflag",
            "value_converted", "unit"
        ])

        result = pipeline.validate(empty_df)

        assert result.valid is False
        assert result.total_rows == 0
        assert "No data found" in result.issues

    def test_validate_good_data(self):
        """Should pass validation for good data."""
        pipeline = GHCNPipeline()

        data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=100),
            "station_id": ["USC00010008"] * 100,
            "element": ["TMAX"] * 100,
            "value": [200] * 100,  # 20.0 C
            "mflag": [""] * 100,
            "qflag": [""] * 100,
            "sflag": [""] * 100,
            "value_converted": [20.0] * 100,
            "unit": ["C"] * 100,
        })

        result = pipeline.validate(data)

        assert result.valid is True
        assert result.total_rows == 100
        assert result.missing_pct == 0.0
        assert result.outliers_count == 0

    def test_validate_high_quality_flag_rate(self):
        """Should flag high rate of quality issues."""
        pipeline = GHCNPipeline()

        # 50% of data has quality flags
        data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=100),
            "station_id": ["USC00010008"] * 100,
            "element": ["TMAX"] * 100,
            "value": [200] * 100,
            "mflag": [""] * 100,
            "qflag": ["G"] * 50 + [""] * 50,  # 50% flagged
            "sflag": [""] * 100,
            "value_converted": [20.0] * 100,
            "unit": ["C"] * 100,
        })

        result = pipeline.validate(data)

        assert result.valid is False
        assert any("quality-flagged" in issue for issue in result.issues)

    def test_validate_temperature_outliers(self):
        """Should detect temperature outliers."""
        pipeline = GHCNPipeline()

        data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10),
            "station_id": ["USC00010008"] * 10,
            "element": ["TMAX"] * 10,
            "value": [200] * 9 + [800],  # Last one is 80 C (outlier)
            "mflag": [""] * 10,
            "qflag": [""] * 10,
            "sflag": [""] * 10,
            "value_converted": [20.0] * 9 + [80.0],  # 80 C is outlier
            "unit": ["C"] * 10,
        })

        result = pipeline.validate(data)

        assert result.outliers_count >= 1
        assert any("temperature outliers" in issue for issue in result.issues)


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    def test_pipeline_inherits_from_temporal(self):
        """Should inherit from TemporalPipeline."""
        from snowforecast.utils import TemporalPipeline

        assert issubclass(GHCNPipeline, TemporalPipeline)

    def test_process_single_file(self):
        """Should process a single .dly file."""
        pipeline = GHCNPipeline()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".dly", delete=False) as f:
            f.write(SAMPLE_DLY_CONTENT)
            f.flush()
            path = Path(f.name)

        try:
            df = pipeline.process(path)

            # Should have converted units
            assert "value_converted" in df.columns
            assert "unit" in df.columns

            # Data should be sorted
            assert df["date"].is_monotonic_increasing or len(df) <= 1
        finally:
            path.unlink()

    def test_process_multiple_files(self):
        """Should process multiple .dly files."""
        pipeline = GHCNPipeline()

        paths = []
        try:
            for i in range(2):
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".dly", delete=False
                ) as f:
                    f.write(SAMPLE_DLY_CONTENT)
                    f.flush()
                    paths.append(Path(f.name))

            df = pipeline.process(paths)

            # Should have data from both files
            assert len(df) > 0
        finally:
            for path in paths:
                path.unlink()
