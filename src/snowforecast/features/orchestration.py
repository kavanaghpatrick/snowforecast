"""Unified data pipeline orchestration for snowforecast.

The DataOrchestrator coordinates all data pipelines to collect station lists,
ski resort locations, and prepare the framework for training data consolidation.

This module does NOT run actual data downloads - it provides the structure
for orchestrating pipeline calls in a coordinated manner.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from snowforecast.utils import ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class LocationPoint:
    """A geographic location point with metadata.

    Used to represent stations, resorts, or any other location
    that needs weather data extraction.

    Attributes:
        id: Unique identifier for this location
        name: Human-readable name
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        elevation: Elevation in meters (optional)
        source: Data source (e.g., 'snotel', 'ghcn', 'openskimap')
        metadata: Additional source-specific metadata
    """

    id: str
    name: str
    lat: float
    lon: float
    elevation: float | None = None
    source: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    """Result of orchestration operations.

    Attributes:
        success: Whether the operation completed successfully
        message: Human-readable summary message
        stations_df: DataFrame of all station locations (SNOTEL + GHCN)
        resorts_df: DataFrame of ski resort locations
        consolidated_locations: DataFrame combining all locations for extraction
        validation_results: Dict of pipeline name to ValidationResult
        stats: Summary statistics
    """

    success: bool
    message: str
    stations_df: pd.DataFrame | None = None
    resorts_df: pd.DataFrame | None = None
    consolidated_locations: pd.DataFrame | None = None
    validation_results: dict[str, ValidationResult] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)


class StationProvider(Protocol):
    """Protocol for pipelines that provide station/location lists."""

    def get_station_metadata(self, state: str | None = None) -> list: ...


class ResortProvider(Protocol):
    """Protocol for pipelines that provide ski resort locations."""

    def get_western_us_resorts(self) -> list: ...
    def export_to_dataframe(self, resorts: list) -> pd.DataFrame: ...


class DataOrchestrator:
    """Unified data pipeline orchestration.

    Coordinates data collection from multiple pipelines to create
    a consolidated dataset for ML training.

    The orchestrator:
    1. Collects station lists from SNOTEL and GHCN pipelines
    2. Collects ski resort locations from OpenSkiMap
    3. Creates a consolidated location DataFrame for weather extraction
    4. Provides structure for temporal data alignment

    Example:
        >>> orchestrator = DataOrchestrator()
        >>> orchestrator.register_station_provider('snotel', snotel_pipeline)
        >>> orchestrator.register_station_provider('ghcn', ghcn_pipeline)
        >>> orchestrator.register_resort_provider('openskimap', openskimap_pipeline)
        >>> result = orchestrator.collect_locations()
        >>> print(f"Found {len(result.consolidated_locations)} total locations")
    """

    def __init__(self):
        """Initialize the orchestrator with empty provider registries."""
        self._station_providers: dict[str, StationProvider] = {}
        self._resort_providers: dict[str, ResortProvider] = {}
        self._collected_stations: pd.DataFrame | None = None
        self._collected_resorts: pd.DataFrame | None = None

    def register_station_provider(self, name: str, provider: StationProvider) -> None:
        """Register a station data provider (e.g., SNOTEL, GHCN).

        Args:
            name: Unique name for this provider (e.g., 'snotel', 'ghcn')
            provider: Object implementing StationProvider protocol
        """
        self._station_providers[name] = provider
        logger.info(f"Registered station provider: {name}")

    def register_resort_provider(self, name: str, provider: ResortProvider) -> None:
        """Register a resort location provider (e.g., OpenSkiMap).

        Args:
            name: Unique name for this provider (e.g., 'openskimap')
            provider: Object implementing ResortProvider protocol
        """
        self._resort_providers[name] = provider
        logger.info(f"Registered resort provider: {name}")

    def collect_stations(
        self,
        states: list[str] | None = None,
        min_elevation: float | None = None,
    ) -> pd.DataFrame:
        """Collect station metadata from all registered providers.

        Args:
            states: Optional list of state codes to filter (e.g., ['CO', 'UT'])
            min_elevation: Optional minimum elevation filter in meters

        Returns:
            DataFrame with columns:
                - station_id: Unique station identifier
                - name: Station name
                - lat: Latitude
                - lon: Longitude
                - elevation: Elevation in meters
                - state: State code
                - source: Provider name (e.g., 'snotel', 'ghcn')

        Raises:
            ValueError: If no station providers are registered
        """
        if not self._station_providers:
            raise ValueError("No station providers registered. Call register_station_provider first.")

        all_stations = []

        for provider_name, provider in self._station_providers.items():
            logger.info(f"Collecting stations from {provider_name}")

            try:
                # Collect stations, optionally filtering by state
                stations = []
                if states:
                    for state in states:
                        stations.extend(provider.get_station_metadata(state=state))
                else:
                    stations = provider.get_station_metadata(state=None)

                for station in stations:
                    # Handle both dataclass and dict-like objects
                    if hasattr(station, "__dict__"):
                        record = {
                            "station_id": getattr(station, "station_id", getattr(station, "id", "")),
                            "name": getattr(station, "name", ""),
                            "lat": getattr(station, "lat", 0.0),
                            "lon": getattr(station, "lon", 0.0),
                            "elevation": getattr(station, "elevation", None),
                            "state": getattr(station, "state", ""),
                            "source": provider_name,
                        }
                    else:
                        record = {
                            "station_id": station.get("station_id", station.get("id", "")),
                            "name": station.get("name", ""),
                            "lat": station.get("lat", 0.0),
                            "lon": station.get("lon", 0.0),
                            "elevation": station.get("elevation"),
                            "state": station.get("state", ""),
                            "source": provider_name,
                        }
                    all_stations.append(record)

                logger.info(f"Collected {len(stations)} stations from {provider_name}")

            except Exception as e:
                logger.error(f"Failed to collect stations from {provider_name}: {e}")
                continue

        if not all_stations:
            return pd.DataFrame(columns=[
                "station_id", "name", "lat", "lon", "elevation", "state", "source"
            ])

        df = pd.DataFrame(all_stations)

        # Apply elevation filter if specified
        if min_elevation is not None and "elevation" in df.columns:
            df = df[df["elevation"].notna() & (df["elevation"] >= min_elevation)]

        self._collected_stations = df
        logger.info(f"Total stations collected: {len(df)}")
        return df

    def collect_resorts(self) -> pd.DataFrame:
        """Collect ski resort locations from all registered providers.

        Returns:
            DataFrame with columns:
                - name: Resort name
                - lat: Latitude
                - lon: Longitude
                - base_elevation: Base elevation in meters
                - summit_elevation: Summit elevation in meters
                - vertical_drop: Vertical drop in meters
                - country: Country code
                - state: State/province code
                - source: Provider name

        Raises:
            ValueError: If no resort providers are registered
        """
        if not self._resort_providers:
            raise ValueError("No resort providers registered. Call register_resort_provider first.")

        all_resorts = []

        for provider_name, provider in self._resort_providers.items():
            logger.info(f"Collecting resorts from {provider_name}")

            try:
                resorts = provider.get_western_us_resorts()
                df = provider.export_to_dataframe(resorts)

                # Add source column
                df["source"] = provider_name

                all_resorts.append(df)
                logger.info(f"Collected {len(df)} resorts from {provider_name}")

            except Exception as e:
                logger.error(f"Failed to collect resorts from {provider_name}: {e}")
                continue

        if not all_resorts:
            return pd.DataFrame(columns=[
                "name", "lat", "lon", "base_elevation", "summit_elevation",
                "vertical_drop", "country", "state", "source"
            ])

        df = pd.concat(all_resorts, ignore_index=True)
        self._collected_resorts = df
        logger.info(f"Total resorts collected: {len(df)}")
        return df

    def consolidate_locations(
        self,
        include_stations: bool = True,
        include_resorts: bool = True,
    ) -> pd.DataFrame:
        """Create a consolidated DataFrame of all locations for weather extraction.

        Combines stations and resorts into a single DataFrame with unified schema
        for point-based weather data extraction from gridded sources (ERA5, HRRR).

        Args:
            include_stations: Whether to include SNOTEL/GHCN stations
            include_resorts: Whether to include ski resort locations

        Returns:
            DataFrame with columns:
                - location_id: Unique identifier
                - name: Location name
                - lat: Latitude
                - lon: Longitude
                - elevation: Elevation in meters (may be None)
                - location_type: 'station' or 'resort'
                - source: Original data source

        Raises:
            ValueError: If no locations have been collected
        """
        locations = []

        if include_stations and self._collected_stations is not None:
            for _, row in self._collected_stations.iterrows():
                locations.append({
                    "location_id": f"station_{row['source']}_{row['station_id']}",
                    "name": row["name"],
                    "lat": row["lat"],
                    "lon": row["lon"],
                    "elevation": row.get("elevation"),
                    "location_type": "station",
                    "source": row["source"],
                })

        if include_resorts and self._collected_resorts is not None:
            for idx, row in self._collected_resorts.iterrows():
                # Use base elevation for resort elevation
                locations.append({
                    "location_id": f"resort_{row['source']}_{idx}",
                    "name": row["name"],
                    "lat": row["lat"],
                    "lon": row["lon"],
                    "elevation": row.get("base_elevation"),
                    "location_type": "resort",
                    "source": row["source"],
                })

        if not locations:
            raise ValueError(
                "No locations to consolidate. Call collect_stations() or "
                "collect_resorts() first."
            )

        df = pd.DataFrame(locations)
        logger.info(f"Consolidated {len(df)} locations")
        return df

    def collect_locations(
        self,
        states: list[str] | None = None,
        min_elevation: float | None = None,
    ) -> OrchestrationResult:
        """Run full location collection from all registered providers.

        This is the main entry point for orchestration. It collects stations
        and resorts, consolidates them, and returns a comprehensive result.

        Args:
            states: Optional list of state codes to filter stations
            min_elevation: Optional minimum elevation filter for stations

        Returns:
            OrchestrationResult with all collected data and validation info
        """
        issues = []
        validation_results = {}

        # Collect stations
        stations_df = None
        if self._station_providers:
            try:
                stations_df = self.collect_stations(
                    states=states,
                    min_elevation=min_elevation,
                )
                validation_results["stations"] = ValidationResult(
                    valid=len(stations_df) > 0,
                    total_rows=len(stations_df),
                    missing_pct=0.0,
                    outliers_count=0,
                    issues=[],
                    stats={
                        "sources": stations_df["source"].value_counts().to_dict() if len(stations_df) > 0 else {},
                        "states": stations_df["state"].value_counts().to_dict() if len(stations_df) > 0 else {},
                    },
                )
            except Exception as e:
                issues.append(f"Station collection failed: {e}")
                logger.error(f"Station collection failed: {e}")

        # Collect resorts
        resorts_df = None
        if self._resort_providers:
            try:
                resorts_df = self.collect_resorts()
                validation_results["resorts"] = ValidationResult(
                    valid=len(resorts_df) > 0,
                    total_rows=len(resorts_df),
                    missing_pct=0.0,
                    outliers_count=0,
                    issues=[],
                    stats={
                        "sources": resorts_df["source"].value_counts().to_dict() if len(resorts_df) > 0 else {},
                    },
                )
            except Exception as e:
                issues.append(f"Resort collection failed: {e}")
                logger.error(f"Resort collection failed: {e}")

        # Consolidate if we have any data
        consolidated = None
        if stations_df is not None or resorts_df is not None:
            try:
                consolidated = self.consolidate_locations(
                    include_stations=stations_df is not None,
                    include_resorts=resorts_df is not None,
                )
            except Exception as e:
                issues.append(f"Location consolidation failed: {e}")
                logger.error(f"Location consolidation failed: {e}")

        # Build stats
        stats = {
            "total_stations": len(stations_df) if stations_df is not None else 0,
            "total_resorts": len(resorts_df) if resorts_df is not None else 0,
            "total_locations": len(consolidated) if consolidated is not None else 0,
            "station_providers": list(self._station_providers.keys()),
            "resort_providers": list(self._resort_providers.keys()),
        }

        success = len(issues) == 0 and (stations_df is not None or resorts_df is not None)

        if success:
            message = (
                f"Successfully collected {stats['total_stations']} stations and "
                f"{stats['total_resorts']} resorts ({stats['total_locations']} total locations)"
            )
        else:
            message = f"Collection completed with issues: {'; '.join(issues)}"

        return OrchestrationResult(
            success=success,
            message=message,
            stations_df=stations_df,
            resorts_df=resorts_df,
            consolidated_locations=consolidated,
            validation_results=validation_results,
            stats=stats,
        )

    def get_extraction_points(self) -> list[tuple[float, float]]:
        """Get list of (lat, lon) tuples for weather data extraction.

        Returns consolidated locations as a list of coordinate tuples,
        suitable for passing to GriddedPipeline.extract_at_points().

        Returns:
            List of (lat, lon) tuples

        Raises:
            ValueError: If no locations have been collected
        """
        if self._collected_stations is None and self._collected_resorts is None:
            raise ValueError("No locations collected. Call collect_locations() first.")

        consolidated = self.consolidate_locations()
        return list(zip(consolidated["lat"], consolidated["lon"]))

    def create_training_scaffold(
        self,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Create an empty DataFrame scaffold for training data.

        Creates a MultiIndex DataFrame with (location_id, datetime) as index,
        ready to be filled with weather and observation data.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Empty DataFrame with proper structure for training data

        Raises:
            ValueError: If no locations have been collected
        """
        if self._collected_stations is None and self._collected_resorts is None:
            raise ValueError("No locations collected. Call collect_locations() first.")

        consolidated = self.consolidate_locations()

        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        # Create MultiIndex
        location_ids = consolidated["location_id"].tolist()
        index = pd.MultiIndex.from_product(
            [location_ids, dates],
            names=["location_id", "datetime"]
        )

        # Create empty DataFrame with expected columns
        columns = [
            # Location metadata (will be filled from consolidated)
            "lat", "lon", "elevation", "location_type", "source",
            # Ground truth (from SNOTEL/GHCN)
            "snow_depth_cm", "swe_mm", "temp_avg_c", "snowfall_cm",
            # Weather features (from ERA5/HRRR)
            "temp_2m", "precip_total", "wind_speed", "humidity",
            "pressure", "cloud_cover",
            # Terrain features (from DEM)
            "dem_elevation", "slope", "aspect", "curvature",
        ]

        df = pd.DataFrame(index=index, columns=columns)

        logger.info(
            f"Created training scaffold: {len(location_ids)} locations x "
            f"{len(dates)} days = {len(df)} rows"
        )

        return df

    def get_provider_names(self) -> dict[str, list[str]]:
        """Get names of all registered providers.

        Returns:
            Dict with 'station_providers' and 'resort_providers' lists
        """
        return {
            "station_providers": list(self._station_providers.keys()),
            "resort_providers": list(self._resort_providers.keys()),
        }

    def clear_cache(self) -> None:
        """Clear cached station and resort data."""
        self._collected_stations = None
        self._collected_resorts = None
        logger.info("Cleared orchestrator cache")
