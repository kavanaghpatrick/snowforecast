"""Tests for geographic utilities."""

import pytest
from snowforecast.utils.geo import BoundingBox, Point, haversine, WESTERN_US_BBOX


class TestBoundingBox:
    """Tests for BoundingBox class."""

    def test_create_bbox(self):
        """Should create a bounding box with correct coordinates."""
        bbox = BoundingBox(west=-106.0, south=39.0, east=-105.0, north=40.0)
        assert bbox.west == -106.0
        assert bbox.south == 39.0
        assert bbox.east == -105.0
        assert bbox.north == 40.0

    def test_contains_point_inside(self):
        """Should return True for point inside bbox."""
        bbox = BoundingBox(west=-106.0, south=39.0, east=-105.0, north=40.0)
        assert bbox.contains(lat=39.5, lon=-105.5) is True

    def test_contains_point_outside(self):
        """Should return False for point outside bbox."""
        bbox = BoundingBox(west=-106.0, south=39.0, east=-105.0, north=40.0)
        assert bbox.contains(lat=41.0, lon=-105.5) is False
        assert bbox.contains(lat=39.5, lon=-107.0) is False

    def test_contains_point_on_edge(self):
        """Should return True for point on bbox edge."""
        bbox = BoundingBox(west=-106.0, south=39.0, east=-105.0, north=40.0)
        assert bbox.contains(lat=39.0, lon=-105.5) is True
        assert bbox.contains(lat=40.0, lon=-106.0) is True

    def test_to_tuple(self):
        """Should return (north, west, south, east) for ERA5 API."""
        bbox = BoundingBox(west=-106.0, south=39.0, east=-105.0, north=40.0)
        assert bbox.to_tuple() == (40.0, -106.0, 39.0, -105.0)

    def test_to_dict(self):
        """Should return dictionary representation."""
        bbox = BoundingBox(west=-106.0, south=39.0, east=-105.0, north=40.0)
        d = bbox.to_dict()
        assert d == {"west": -106.0, "south": 39.0, "east": -105.0, "north": 40.0}


class TestPoint:
    """Tests for Point class."""

    def test_create_point(self):
        """Should create a point with lat/lon."""
        point = Point(lat=39.5, lon=-105.5)
        assert point.lat == 39.5
        assert point.lon == -105.5
        assert point.elevation is None

    def test_create_point_with_elevation(self):
        """Should create a point with elevation."""
        point = Point(lat=39.5, lon=-105.5, elevation=3000.0)
        assert point.elevation == 3000.0

    def test_distance_to(self):
        """Should calculate distance between two points."""
        denver = Point(lat=39.7392, lon=-104.9903)
        boulder = Point(lat=40.0150, lon=-105.2705)
        distance = denver.distance_to(boulder)
        # Denver to Boulder is approximately 40km
        assert 35 < distance < 45


class TestHaversine:
    """Tests for haversine distance calculation."""

    def test_same_point(self):
        """Should return 0 for same point."""
        distance = haversine(39.5, -105.5, 39.5, -105.5)
        assert distance == 0.0

    def test_known_distance(self):
        """Should calculate known distance correctly."""
        # New York to Los Angeles: approximately 3940 km
        ny_lat, ny_lon = 40.7128, -74.0060
        la_lat, la_lon = 34.0522, -118.2437
        distance = haversine(ny_lat, ny_lon, la_lat, la_lon)
        assert 3900 < distance < 4000

    def test_short_distance(self):
        """Should handle short distances accurately."""
        # 1 degree latitude is approximately 111 km
        distance = haversine(39.0, -105.0, 40.0, -105.0)
        assert 110 < distance < 112


class TestWesternUSBbox:
    """Tests for WESTERN_US_BBOX constant."""

    def test_covers_colorado(self):
        """Should contain Colorado ski resorts."""
        # Vail, CO
        assert WESTERN_US_BBOX.contains(lat=39.6403, lon=-106.3742) is True
        # Breckenridge, CO
        assert WESTERN_US_BBOX.contains(lat=39.4817, lon=-106.0384) is True

    def test_covers_utah(self):
        """Should contain Utah ski resorts."""
        # Park City, UT
        assert WESTERN_US_BBOX.contains(lat=40.6461, lon=-111.4980) is True

    def test_covers_california(self):
        """Should contain California ski resorts."""
        # Lake Tahoe, CA
        assert WESTERN_US_BBOX.contains(lat=39.0968, lon=-120.0324) is True

    def test_excludes_east_coast(self):
        """Should not contain East Coast locations."""
        # New York City
        assert WESTERN_US_BBOX.contains(lat=40.7128, lon=-74.0060) is False
