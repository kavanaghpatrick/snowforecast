"""Copernicus DEM terrain data pipeline.

This pipeline downloads Copernicus GLO-30 DEM data from AWS Open Data
and calculates terrain features (elevation, slope, aspect, roughness, TPI)
for snow prediction.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

from snowforecast.utils import StaticPipeline, ValidationResult, get_data_path

# AWS S3 bucket for Copernicus DEM (public, no auth required)
COPERNICUS_DEM_BUCKET = "copernicus-dem-30m"
COPERNICUS_DEM_URL = f"https://{COPERNICUS_DEM_BUCKET}.s3.eu-central-1.amazonaws.com"

# DEM cell size in meters
DEM_CELL_SIZE = 30.0


@dataclass
class TerrainFeatures:
    """Terrain features derived from DEM data.

    Attributes:
        elevation: Elevation in meters
        slope: Slope angle in degrees
        aspect: Aspect (direction of slope) in degrees from north (0-360)
        aspect_sin: Sine of aspect for cyclical encoding
        aspect_cos: Cosine of aspect for cyclical encoding
        roughness: Terrain roughness index
        tpi: Topographic Position Index (positive=ridge, negative=valley)
    """

    elevation: float
    slope: float
    aspect: float
    aspect_sin: float
    aspect_cos: float
    roughness: float
    tpi: float


def _tile_name(lat: int, lon: int) -> str:
    """Generate Copernicus DEM tile name for a lat/lon.

    Args:
        lat: Latitude (integer, south edge of tile)
        lon: Longitude (integer, west edge of tile)

    Returns:
        Tile name string, e.g., "Copernicus_DSM_COG_10_N40_00_W112_00_DEM"
    """
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    lat_abs = abs(lat)
    lon_abs = abs(lon)
    return f"Copernicus_DSM_COG_10_{lat_dir}{lat_abs:02d}_00_{lon_dir}{lon_abs:03d}_00_DEM"


def _tile_url(lat: int, lon: int) -> str:
    """Get the full URL for a DEM tile.

    Args:
        lat: Latitude (integer)
        lon: Longitude (integer)

    Returns:
        Full S3 URL for the tile
    """
    name = _tile_name(lat, lon)
    return f"{COPERNICUS_DEM_URL}/{name}/{name}.tif"


class DEMPipeline(StaticPipeline):
    """Copernicus DEM terrain data pipeline.

    Downloads elevation data from Copernicus GLO-30 DEM on AWS and
    calculates terrain features for snow prediction models.
    """

    def __init__(self, cell_size: float = DEM_CELL_SIZE):
        """Initialize the DEM pipeline.

        Args:
            cell_size: DEM cell size in meters (default 30m)
        """
        self.cell_size = cell_size
        self.raw_path = get_data_path("dem", "raw")
        self.processed_path = get_data_path("dem", "processed")
        self._cached_datasets: dict[tuple[int, int], rasterio.DatasetReader] = {}

    def download_tile(self, lat: int, lon: int, force: bool = False) -> Path:
        """Download a single DEM tile.

        Args:
            lat: Latitude of tile (integer, south edge)
            lon: Longitude of tile (integer, west edge)
            force: If True, re-download even if file exists

        Returns:
            Path to downloaded tile

        Note:
            Uses rasterio's built-in HTTPS/COG support to stream data.
            Tiles are ~25-50MB each.
        """
        tile_name = _tile_name(lat, lon)
        local_path = self.raw_path / f"{tile_name}.tif"

        if local_path.exists() and not force:
            return local_path

        # Stream from S3 using rasterio's GDAL/HTTPS support
        url = _tile_url(lat, lon)
        try:
            with rasterio.open(url) as src:
                # Read entire tile and write locally
                data = src.read(1)
                profile = src.profile.copy()

                with rasterio.open(local_path, "w", **profile) as dst:
                    dst.write(data, 1)

            return local_path
        except Exception as e:
            raise RuntimeError(f"Failed to download DEM tile {tile_name}: {e}") from e

    def download_region(
        self, bbox: dict, force: bool = False
    ) -> list[Path]:
        """Download all DEM tiles covering a bounding box region.

        Args:
            bbox: Dictionary with west, south, east, north keys
            force: If True, re-download even if files exist

        Returns:
            List of paths to downloaded tiles
        """
        west, south, east, north = bbox["west"], bbox["south"], bbox["east"], bbox["north"]

        # Find all tiles needed (tiles are 1x1 degree)
        # Tile covers lat to lat+1 and lon to lon+1
        lat_min = int(np.floor(south))
        lat_max = int(np.floor(north))
        lon_min = int(np.floor(west))
        lon_max = int(np.floor(east))

        tiles = []
        for lat in range(lat_min, lat_max + 1):
            for lon in range(lon_min, lon_max + 1):
                try:
                    tile_path = self.download_tile(lat, lon, force=force)
                    tiles.append(tile_path)
                except RuntimeError:
                    # Some tiles may not exist (ocean, etc.)
                    continue

        return tiles

    def download(self, bbox: dict | None = None, **kwargs) -> Path | list[Path]:
        """Download DEM data for a region.

        Args:
            bbox: Bounding box dict with west, south, east, north
            **kwargs: Additional options (e.g., force=True)

        Returns:
            List of paths to downloaded tiles
        """
        if bbox is None:
            # Default to Western US
            from snowforecast.utils import WESTERN_US_BBOX
            bbox = WESTERN_US_BBOX.to_dict()

        return self.download_region(bbox, force=kwargs.get("force", False))

    def _open_tile(self, lat: int, lon: int) -> rasterio.DatasetReader | None:
        """Open a tile, downloading if necessary.

        Args:
            lat: Tile latitude
            lon: Tile longitude

        Returns:
            Rasterio dataset or None if tile doesn't exist
        """
        key = (lat, lon)
        if key not in self._cached_datasets:
            tile_name = _tile_name(lat, lon)
            local_path = self.raw_path / f"{tile_name}.tif"

            if not local_path.exists():
                try:
                    self.download_tile(lat, lon)
                except RuntimeError:
                    return None

            if local_path.exists():
                self._cached_datasets[key] = rasterio.open(local_path)

        return self._cached_datasets.get(key)

    def get_elevation(self, lat: float, lon: float) -> float:
        """Get elevation at a point using bilinear interpolation.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees

        Returns:
            Elevation in meters

        Raises:
            ValueError: If point is outside available data
        """
        # Find which tile this point is in
        tile_lat = int(np.floor(lat))
        tile_lon = int(np.floor(lon))

        ds = self._open_tile(tile_lat, tile_lon)
        if ds is None:
            raise ValueError(f"No DEM data available for ({lat}, {lon})")

        # Convert lat/lon to pixel coordinates
        row, col = ds.index(lon, lat)

        # Read a small window for interpolation
        try:
            data = ds.read(1, window=rasterio.windows.Window(col - 1, row - 1, 3, 3))
        except Exception:
            # Near tile edge, just read single pixel
            data = ds.read(1, window=rasterio.windows.Window(col, row, 1, 1))
            return float(data[0, 0])

        # Bilinear interpolation
        transform = ds.window_transform(rasterio.windows.Window(col - 1, row - 1, 3, 3))
        x_frac = (lon - transform.c) / transform.a - 1
        y_frac = (lat - transform.f) / transform.e - 1

        x_frac = max(0, min(1, x_frac))
        y_frac = max(0, min(1, y_frac))

        # Bilinear interpolation on center 2x2
        v00 = data[1, 1]
        v01 = data[1, 2] if data.shape[1] > 2 else v00
        v10 = data[2, 1] if data.shape[0] > 2 else v00
        v11 = data[2, 2] if data.shape[0] > 2 and data.shape[1] > 2 else v00

        elevation = (
            v00 * (1 - x_frac) * (1 - y_frac)
            + v01 * x_frac * (1 - y_frac)
            + v10 * (1 - x_frac) * y_frac
            + v11 * x_frac * y_frac
        )

        return float(elevation)

    def calculate_slope(self, dem: np.ndarray, resolution: float | None = None) -> np.ndarray:
        """Calculate slope in degrees from DEM.

        Args:
            dem: 2D array of elevation values
            resolution: Cell size in meters (default: self.cell_size)

        Returns:
            2D array of slope values in degrees
        """
        if resolution is None:
            resolution = self.cell_size

        # Calculate gradients
        dy, dx = np.gradient(dem, resolution)

        # Calculate slope
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        return np.degrees(slope_rad)

    def calculate_aspect(self, dem: np.ndarray, resolution: float | None = None) -> np.ndarray:
        """Calculate aspect (direction of slope) in degrees.

        Args:
            dem: 2D array of elevation values
            resolution: Cell size in meters (default: self.cell_size)

        Returns:
            2D array of aspect values (0=N, 90=E, 180=S, 270=W)
        """
        if resolution is None:
            resolution = self.cell_size

        # Calculate gradients
        dy, dx = np.gradient(dem, resolution)

        # Calculate aspect (note: -dx for correct orientation)
        aspect_rad = np.arctan2(-dx, dy)
        aspect_deg = np.degrees(aspect_rad)

        # Convert to 0-360 range
        aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360, aspect_deg)

        return aspect_deg

    def calculate_roughness(self, dem: np.ndarray, window: int = 3) -> np.ndarray:
        """Calculate terrain roughness index (TRI).

        TRI = mean of absolute differences from neighbors.

        Args:
            dem: 2D array of elevation values
            window: Size of neighborhood window (default 3)

        Returns:
            2D array of roughness values
        """
        # Calculate mean in neighborhood
        mean = ndimage.uniform_filter(dem.astype(np.float64), size=window)
        # TRI is mean absolute difference from neighborhood mean
        roughness = ndimage.uniform_filter(np.abs(dem - mean), size=window)

        return roughness

    def calculate_tpi(self, dem: np.ndarray, radius: int = 10) -> np.ndarray:
        """Calculate Topographic Position Index (TPI).

        TPI = elevation - mean elevation in neighborhood.
        Positive values indicate ridges, negative indicate valleys.

        Args:
            dem: 2D array of elevation values
            radius: Radius of neighborhood in pixels (default 10)

        Returns:
            2D array of TPI values
        """
        window_size = 2 * radius + 1
        mean = ndimage.uniform_filter(dem.astype(np.float64), size=window_size)
        return dem - mean

    def get_terrain_features(self, lat: float, lon: float, window_size: int = 11) -> TerrainFeatures:
        """Get all terrain features for a point.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            window_size: Size of window to read for calculations (must be odd)

        Returns:
            TerrainFeatures dataclass with all terrain metrics
        """
        # Find which tile this point is in
        tile_lat = int(np.floor(lat))
        tile_lon = int(np.floor(lon))

        ds = self._open_tile(tile_lat, tile_lon)
        if ds is None:
            raise ValueError(f"No DEM data available for ({lat}, {lon})")

        # Convert lat/lon to pixel coordinates
        row, col = ds.index(lon, lat)
        half_window = window_size // 2

        # Read window around point
        window = rasterio.windows.Window(
            col - half_window, row - half_window, window_size, window_size
        )
        try:
            dem = ds.read(1, window=window).astype(np.float64)
        except Exception:
            # If window is out of bounds, get single point
            elevation = self.get_elevation(lat, lon)
            return TerrainFeatures(
                elevation=elevation,
                slope=0.0,
                aspect=0.0,
                aspect_sin=0.0,
                aspect_cos=1.0,
                roughness=0.0,
                tpi=0.0,
            )

        # Calculate all terrain metrics
        slope = self.calculate_slope(dem)
        aspect = self.calculate_aspect(dem)
        roughness = self.calculate_roughness(dem)
        tpi = self.calculate_tpi(dem, radius=half_window)

        # Get center values
        center = window_size // 2
        center_elevation = float(dem[center, center])
        center_slope = float(slope[center, center])
        center_aspect = float(aspect[center, center])
        center_roughness = float(roughness[center, center])
        center_tpi = float(tpi[center, center])

        # Calculate cyclical encoding of aspect
        aspect_rad = np.radians(center_aspect)
        aspect_sin = float(np.sin(aspect_rad))
        aspect_cos = float(np.cos(aspect_rad))

        return TerrainFeatures(
            elevation=center_elevation,
            slope=center_slope,
            aspect=center_aspect,
            aspect_sin=aspect_sin,
            aspect_cos=aspect_cos,
            roughness=center_roughness,
            tpi=center_tpi,
        )

    def get_features_for_stations(
        self, stations: list[dict], window_size: int = 11
    ) -> pd.DataFrame:
        """Get terrain features for all station locations.

        Args:
            stations: List of dicts with at least 'lat', 'lon', 'station_id' keys
            window_size: Size of window for terrain calculations

        Returns:
            DataFrame with terrain features for each station
        """
        records = []
        for station in stations:
            station_id = station["station_id"]
            lat = station["lat"]
            lon = station["lon"]

            try:
                features = self.get_terrain_features(lat, lon, window_size)
                records.append({
                    "station_id": station_id,
                    "lat": lat,
                    "lon": lon,
                    "elevation": features.elevation,
                    "slope": features.slope,
                    "aspect": features.aspect,
                    "aspect_sin": features.aspect_sin,
                    "aspect_cos": features.aspect_cos,
                    "roughness": features.roughness,
                    "tpi": features.tpi,
                })
            except ValueError:
                # No data for this location
                records.append({
                    "station_id": station_id,
                    "lat": lat,
                    "lon": lon,
                    "elevation": np.nan,
                    "slope": np.nan,
                    "aspect": np.nan,
                    "aspect_sin": np.nan,
                    "aspect_cos": np.nan,
                    "roughness": np.nan,
                    "tpi": np.nan,
                })

        return pd.DataFrame(records)

    def process(self, raw_path: Path | list[Path]) -> pd.DataFrame:
        """Process raw DEM tiles into terrain features.

        For DEMPipeline, this is typically used with get_features_for_stations().
        If called with tile paths, it returns metadata about the tiles.

        Args:
            raw_path: Path(s) to raw DEM tiles

        Returns:
            DataFrame with tile metadata
        """
        if isinstance(raw_path, Path):
            paths = [raw_path]
        else:
            paths = raw_path

        records = []
        for path in paths:
            if not path.exists():
                continue
            with rasterio.open(path) as ds:
                bounds = ds.bounds
                records.append({
                    "tile_name": path.stem,
                    "path": str(path),
                    "west": bounds.left,
                    "south": bounds.bottom,
                    "east": bounds.right,
                    "north": bounds.top,
                    "crs": str(ds.crs),
                    "width": ds.width,
                    "height": ds.height,
                })

        return pd.DataFrame(records)

    def validate(self, data: Any) -> ValidationResult:
        """Validate terrain features data.

        Args:
            data: DataFrame with terrain features

        Returns:
            ValidationResult with quality metrics
        """
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(
                valid=False,
                total_rows=0,
                missing_pct=100.0,
                issues=["Data is not a DataFrame"],
            )

        if len(data) == 0:
            return ValidationResult(
                valid=False,
                total_rows=0,
                missing_pct=100.0,
                issues=["DataFrame is empty"],
            )

        issues = []
        total_rows = len(data)

        # Check for required columns in terrain features output
        terrain_cols = ["elevation", "slope", "aspect", "roughness", "tpi"]
        present_cols = [c for c in terrain_cols if c in data.columns]

        if not present_cols:
            # This might be tile metadata instead of terrain features
            if "tile_name" in data.columns:
                return ValidationResult(
                    valid=True,
                    total_rows=total_rows,
                    missing_pct=0.0,
                    stats={"type": "tile_metadata", "num_tiles": total_rows},
                )
            issues.append("No terrain feature columns found")

        # Calculate missing percentage across terrain columns
        missing_pct = 0.0
        if present_cols:
            total_values = total_rows * len(present_cols)
            missing_values = sum(data[col].isna().sum() for col in present_cols)
            missing_pct = (missing_values / total_values * 100) if total_values > 0 else 0

        # Check for outliers
        outliers_count = 0
        stats = {"type": "terrain_features", "num_stations": total_rows}

        if "elevation" in data.columns:
            valid_elev = data["elevation"].dropna()
            if len(valid_elev) > 0:
                stats["elevation_min"] = float(valid_elev.min())
                stats["elevation_max"] = float(valid_elev.max())
                stats["elevation_mean"] = float(valid_elev.mean())
                # Outliers: elevation < -500 or > 9000
                outliers_count += int(((valid_elev < -500) | (valid_elev > 9000)).sum())

        if "slope" in data.columns:
            valid_slope = data["slope"].dropna()
            if len(valid_slope) > 0:
                stats["slope_max"] = float(valid_slope.max())
                stats["slope_mean"] = float(valid_slope.mean())
                # Outliers: slope > 90 degrees
                outliers_count += int((valid_slope > 90).sum())

        if outliers_count > 0:
            issues.append(f"Found {outliers_count} outlier values")

        # Validation passes if we have data and not too many missing values
        valid = len(issues) == 0 and missing_pct < 50

        return ValidationResult(
            valid=valid,
            total_rows=total_rows,
            missing_pct=missing_pct,
            outliers_count=outliers_count,
            issues=issues,
            stats=stats,
        )

    def save_terrain_features(
        self, df: pd.DataFrame, filename: str = "terrain_features.parquet"
    ) -> Path:
        """Save terrain features to parquet file.

        Args:
            df: DataFrame with terrain features
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.processed_path / filename
        df.to_parquet(output_path, index=False)
        return output_path

    def load_terrain_features(
        self, filename: str = "terrain_features.parquet"
    ) -> pd.DataFrame:
        """Load terrain features from parquet file.

        Args:
            filename: Input filename

        Returns:
            DataFrame with terrain features
        """
        input_path = self.processed_path / filename
        return pd.read_parquet(input_path)

    def close(self):
        """Close any open dataset readers."""
        for ds in self._cached_datasets.values():
            ds.close()
        self._cached_datasets.clear()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
