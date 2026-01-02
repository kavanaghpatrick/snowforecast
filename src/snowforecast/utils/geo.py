"""Geographic utilities and constants."""

from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt


@dataclass
class BoundingBox:
    """Geographic bounding box."""

    west: float
    south: float
    east: float
    north: float

    def contains(self, lat: float, lon: float) -> bool:
        """Check if a point is within this bounding box."""
        return (
            self.south <= lat <= self.north
            and self.west <= lon <= self.east
        )

    def to_tuple(self) -> tuple[float, float, float, float]:
        """Return as (north, west, south, east) for ERA5 API."""
        return (self.north, self.west, self.south, self.east)

    def to_dict(self) -> dict:
        """Return as dictionary."""
        return {
            "west": self.west,
            "south": self.south,
            "east": self.east,
            "north": self.north,
        }


@dataclass
class Point:
    """Geographic point with optional elevation."""

    lat: float
    lon: float
    elevation: float | None = None

    def distance_to(self, other: "Point") -> float:
        """Calculate distance in km to another point using Haversine formula."""
        return haversine(self.lat, self.lon, other.lat, other.lon)


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance in km between two points.

    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)

    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    return R * c


# Western US bounding box (covers SNOTEL network)
WESTERN_US_BBOX = BoundingBox(
    west=-125.0,
    south=31.0,
    east=-102.0,
    north=49.0,
)
