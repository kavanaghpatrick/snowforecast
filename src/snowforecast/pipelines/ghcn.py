"""GHCN-Daily data ingestion pipeline.

GHCN-Daily (Global Historical Climatology Network - Daily) provides daily
climate summaries from land surface stations worldwide. This pipeline
downloads and processes GHCN-Daily data for supplemental ground truth
on snowfall and temperature.

Data sources:
- Station inventory: https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt
- Daily data: https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/

Variables:
- TMAX: Maximum temperature (tenths of degrees Celsius)
- TMIN: Minimum temperature (tenths of degrees Celsius)
- PRCP: Precipitation (tenths of mm)
- SNOW: Snowfall (mm)
- SNWD: Snow depth (mm)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from snowforecast.utils import BoundingBox, TemporalPipeline, ValidationResult
from snowforecast.utils.io import get_data_path

logger = logging.getLogger(__name__)

# GHCN-Daily base URLs
GHCN_BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
GHCN_STATIONS_URL = f"{GHCN_BASE_URL}/ghcnd-stations.txt"
GHCN_ALL_URL = f"{GHCN_BASE_URL}/all"

# Quality flags that indicate problematic data
GHCN_QUALITY_FLAGS = {
    "D": "Failed duplicate check",
    "G": "Failed gap check",
    "I": "Failed internal consistency",
    "K": "Failed streak check",
    "L": "Failed temperature limits",
    "M": "Failed megaconsistency",
    "N": "Failed naught check",
    "O": "Failed climatological outlier",
    "R": "Failed lagged range check",
    "S": "Failed spatial consistency",
    "T": "Failed temporal consistency",
    "W": "Failed bounds check",
    "X": "Failed bounds check",
    "Z": "Flagged as multiday accumulation",
}

# Variables we care about
GHCN_VARIABLES = {"TMAX", "TMIN", "PRCP", "SNOW", "SNWD"}

# Fixed-width positions for station inventory file
# ID: 1-11, LAT: 13-20, LON: 22-30, ELEV: 32-37, STATE: 39-40, NAME: 42-71
STATION_WIDTHS = [11, 1, 8, 1, 9, 1, 6, 1, 2, 1, 30, 4]
STATION_COLUMNS = [
    "station_id", "_1", "lat", "_2", "lon", "_3", "elevation", "_4",
    "state", "_5", "name", "gsn_flag"
]


@dataclass
class GHCNStation:
    """GHCN station metadata.

    Attributes:
        station_id: Unique station identifier (e.g., 'USC00010008')
        name: Station name
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        elevation: Elevation in meters
        state: State/province code (may be empty for non-US stations)
    """
    station_id: str
    name: str
    lat: float
    lon: float
    elevation: float
    state: str


class GHCNPipeline(TemporalPipeline):
    """GHCN-Daily data ingestion pipeline.

    This pipeline downloads and processes GHCN-Daily data, which provides
    daily climate observations from weather stations worldwide.

    Example:
        >>> pipeline = GHCNPipeline()
        >>> stations = pipeline.filter_mountain_stations(min_elevation=2000)
        >>> path = pipeline.download_station(stations[0].station_id)
        >>> df = pipeline.parse_dly_file(path)
    """

    def __init__(self, timeout: int = 30):
        """Initialize the GHCN pipeline.

        Args:
            timeout: HTTP request timeout in seconds
        """
        self.timeout = timeout
        self._station_inventory: Optional[list[GHCNStation]] = None

    def get_station_inventory(
        self,
        bbox: Optional[dict | BoundingBox] = None,
        min_years: int = 10,
        force_refresh: bool = False,
    ) -> list[GHCNStation]:
        """Get station metadata filtered by region and data availability.

        Downloads and parses the GHCN station inventory file. Results are
        cached in memory after the first call.

        Args:
            bbox: Optional bounding box to filter stations. Can be a dict with
                keys 'west', 'south', 'east', 'north' or a BoundingBox object.
            min_years: Minimum years of data required (not enforced without
                additional metadata download)
            force_refresh: If True, re-download the inventory even if cached

        Returns:
            List of GHCNStation objects matching the filter criteria
        """
        if self._station_inventory is None or force_refresh:
            self._station_inventory = self._download_station_inventory()

        stations = self._station_inventory

        if bbox is not None:
            stations = self._filter_by_bbox(stations, bbox)

        return stations

    def _download_station_inventory(self) -> list[GHCNStation]:
        """Download and parse the GHCN station inventory file."""
        logger.info(f"Downloading station inventory from {GHCN_STATIONS_URL}")

        cache_path = get_data_path("ghcn", "cache") / "ghcnd-stations.txt"

        # Try to use cached version first
        if cache_path.exists():
            logger.info(f"Using cached station inventory: {cache_path}")
            with open(cache_path, "r") as f:
                content = f.read()
        else:
            response = requests.get(GHCN_STATIONS_URL, timeout=self.timeout)
            response.raise_for_status()
            content = response.text

            # Cache the file
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                f.write(content)

        return self._parse_station_inventory(content)

    def _parse_station_inventory(self, content: str) -> list[GHCNStation]:
        """Parse the fixed-width station inventory file."""
        stations = []

        for line in content.strip().split("\n"):
            if len(line) < 40:
                continue

            try:
                # Parse fixed-width fields
                station_id = line[0:11].strip()
                lat = float(line[12:20].strip())
                lon = float(line[21:30].strip())
                elevation_str = line[31:37].strip()
                state = line[38:40].strip()
                name = line[41:71].strip()

                # Handle missing elevation (-999.9 means missing)
                elevation = float(elevation_str) if elevation_str else -999.9
                if elevation == -999.9:
                    elevation = 0.0  # Default for missing

                stations.append(GHCNStation(
                    station_id=station_id,
                    name=name,
                    lat=lat,
                    lon=lon,
                    elevation=elevation,
                    state=state,
                ))
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse station line: {line[:50]}... Error: {e}")
                continue

        logger.info(f"Parsed {len(stations)} stations from inventory")
        return stations

    def _filter_by_bbox(
        self,
        stations: list[GHCNStation],
        bbox: dict | BoundingBox,
    ) -> list[GHCNStation]:
        """Filter stations by bounding box."""
        if isinstance(bbox, BoundingBox):
            return [
                s for s in stations
                if bbox.contains(s.lat, s.lon)
            ]
        else:
            return [
                s for s in stations
                if (bbox["south"] <= s.lat <= bbox["north"]
                    and bbox["west"] <= s.lon <= bbox["east"])
            ]

    def filter_mountain_stations(
        self,
        min_elevation: float = 1500,
        bbox: Optional[dict | BoundingBox] = None,
    ) -> list[GHCNStation]:
        """Get only stations above elevation threshold.

        Args:
            min_elevation: Minimum elevation in meters (default 1500m)
            bbox: Optional bounding box to further filter results

        Returns:
            List of GHCNStation objects above the elevation threshold
        """
        stations = self.get_station_inventory(bbox=bbox)
        return [s for s in stations if s.elevation >= min_elevation]

    def download_station(
        self,
        station_id: str,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> Path:
        """Download data for a single station.

        Downloads the .dly file for the specified station and caches it
        in the raw data directory.

        Args:
            station_id: GHCN station ID (e.g., 'USC00010008')
            start_year: Optional start year filter (applied during parse)
            end_year: Optional end year filter (applied during parse)

        Returns:
            Path to the downloaded .dly file
        """
        url = f"{GHCN_ALL_URL}/{station_id}.dly"
        raw_path = get_data_path("ghcn", "raw") / f"{station_id}.dly"

        # Use cached file if it exists
        if raw_path.exists():
            logger.info(f"Using cached data for station {station_id}: {raw_path}")
            return raw_path

        logger.info(f"Downloading station {station_id} from {url}")

        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()

        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "w") as f:
            f.write(response.text)

        return raw_path

    def download_stations(
        self,
        station_ids: list[str],
        parallel: bool = True,
    ) -> list[Path]:
        """Download data for multiple stations.

        Args:
            station_ids: List of GHCN station IDs
            parallel: If True, download stations in parallel (not implemented)

        Returns:
            List of paths to downloaded .dly files
        """
        # For simplicity, sequential download for now
        # TODO: Implement parallel download with concurrent.futures
        paths = []
        for station_id in station_ids:
            try:
                path = self.download_station(station_id)
                paths.append(path)
            except requests.RequestException as e:
                logger.error(f"Failed to download station {station_id}: {e}")

        return paths

    def parse_dly_file(
        self,
        path: Path,
        variables: Optional[set[str]] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """Parse GHCN .dly file format.

        The .dly format is fixed-width with one row per station-element-month.
        Each row contains 31 daily values (one per possible day of month).

        Format per row:
        - Columns 1-11: Station ID
        - Columns 12-15: Year
        - Columns 16-17: Month
        - Columns 18-21: Element (TMAX, TMIN, etc.)
        - Columns 22-269: 31 sets of (value, mflag, qflag, sflag)

        Args:
            path: Path to the .dly file
            variables: Set of variables to extract (default: TMAX, TMIN, PRCP, SNOW, SNWD)
            start_year: Optional start year filter
            end_year: Optional end year filter

        Returns:
            DataFrame with columns: date, station_id, element, value, mflag, qflag, sflag
        """
        if variables is None:
            variables = GHCN_VARIABLES

        records = []

        with open(path, "r") as f:
            for line in f:
                if len(line) < 269:
                    continue

                station_id = line[0:11].strip()
                year = int(line[11:15])
                month = int(line[15:17])
                element = line[17:21].strip()

                # Filter by variable
                if element not in variables:
                    continue

                # Filter by year
                if start_year is not None and year < start_year:
                    continue
                if end_year is not None and year > end_year:
                    continue

                # Parse 31 daily values
                for day in range(1, 32):
                    # Each value block is 8 characters: 5 for value, 3 for flags
                    offset = 21 + (day - 1) * 8
                    value_str = line[offset:offset + 5].strip()
                    mflag = line[offset + 5:offset + 6].strip()
                    qflag = line[offset + 6:offset + 7].strip()
                    sflag = line[offset + 7:offset + 8].strip()

                    # -9999 indicates missing data
                    if value_str == "-9999" or value_str == "":
                        continue

                    try:
                        value = int(value_str)
                    except ValueError:
                        continue

                    # Validate day for this month
                    try:
                        date = pd.Timestamp(year=year, month=month, day=day)
                    except ValueError:
                        # Invalid date (e.g., Feb 30)
                        continue

                    records.append({
                        "date": date,
                        "station_id": station_id,
                        "element": element,
                        "value": value,
                        "mflag": mflag,
                        "qflag": qflag,
                        "sflag": sflag,
                    })

        df = pd.DataFrame(records)
        if len(df) == 0:
            return pd.DataFrame(columns=[
                "date", "station_id", "element", "value", "mflag", "qflag", "sflag"
            ])

        return df.sort_values("date").reset_index(drop=True)

    def convert_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert GHCN units to standard metric units.

        Conversions:
        - TMAX/TMIN: tenths of C -> C
        - PRCP: tenths of mm -> mm
        - SNOW: mm -> cm
        - SNWD: mm -> cm

        Args:
            df: DataFrame with 'element' and 'value' columns

        Returns:
            DataFrame with new 'value_converted' and 'unit' columns
        """
        df = df.copy()

        # Create converted value column
        df["value_converted"] = df["value"].astype(float)
        df["unit"] = ""

        # Apply conversions
        mask_temp = df["element"].isin(["TMAX", "TMIN"])
        df.loc[mask_temp, "value_converted"] = df.loc[mask_temp, "value"] / 10.0
        df.loc[mask_temp, "unit"] = "C"

        mask_prcp = df["element"] == "PRCP"
        df.loc[mask_prcp, "value_converted"] = df.loc[mask_prcp, "value"] / 10.0
        df.loc[mask_prcp, "unit"] = "mm"

        mask_snow = df["element"].isin(["SNOW", "SNWD"])
        df.loc[mask_snow, "value_converted"] = df.loc[mask_snow, "value"] / 10.0
        df.loc[mask_snow, "unit"] = "cm"

        return df

    def filter_quality(
        self,
        df: pd.DataFrame,
        exclude_flags: Optional[set[str]] = None,
    ) -> pd.DataFrame:
        """Filter data by quality flags.

        By default, excludes data that failed any quality check.

        Args:
            df: DataFrame with 'qflag' column
            exclude_flags: Set of quality flags to exclude (default: all GHCN flags)

        Returns:
            DataFrame with flagged data removed
        """
        if exclude_flags is None:
            exclude_flags = set(GHCN_QUALITY_FLAGS.keys())

        mask = ~df["qflag"].isin(exclude_flags)
        return df[mask].copy()

    def download(
        self,
        start_date: str,
        end_date: str,
        station_ids: Optional[list[str]] = None,
        bbox: Optional[dict | BoundingBox] = None,
        **kwargs
    ) -> list[Path]:
        """Download raw data from GHCN for a date range.

        Implements the TemporalPipeline interface.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            station_ids: Optional list of specific station IDs to download
            bbox: Optional bounding box to filter stations
            **kwargs: Additional parameters (ignored)

        Returns:
            List of paths to downloaded .dly files
        """
        if station_ids is None:
            # Get stations in bbox or all stations
            stations = self.get_station_inventory(bbox=bbox)
            station_ids = [s.station_id for s in stations[:100]]  # Limit for safety
            logger.warning("No station_ids specified, using first 100 stations")

        return self.download_stations(station_ids)

    def process(
        self,
        raw_paths: Path | list[Path],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Process raw .dly files into standardized DataFrame format.

        Implements the TemporalPipeline interface.

        Args:
            raw_paths: Path or list of paths to .dly files
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)

        Returns:
            DataFrame with standardized columns and converted units
        """
        if isinstance(raw_paths, Path):
            raw_paths = [raw_paths]

        # Parse year from dates if provided
        start_year = int(start_date[:4]) if start_date else None
        end_year = int(end_date[:4]) if end_date else None

        dfs = []
        for path in raw_paths:
            try:
                df = self.parse_dly_file(path, start_year=start_year, end_year=end_year)
                if len(df) > 0:
                    df = self.convert_units(df)
                    dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")

        if not dfs:
            return pd.DataFrame(columns=[
                "date", "station_id", "element", "value", "mflag", "qflag", "sflag",
                "value_converted", "unit"
            ])

        result = pd.concat(dfs, ignore_index=True)

        # Filter by date range
        if start_date:
            result = result[result["date"] >= start_date]
        if end_date:
            result = result[result["date"] <= end_date]

        return result.sort_values(["station_id", "date", "element"]).reset_index(drop=True)

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """Validate processed GHCN data.

        Checks for:
        - Missing values
        - Outliers (extreme temperature/precipitation values)
        - Quality flag issues

        Args:
            data: DataFrame from process()

        Returns:
            ValidationResult with quality metrics
        """
        if len(data) == 0:
            return ValidationResult(
                valid=False,
                total_rows=0,
                missing_pct=100.0,
                issues=["No data found"],
            )

        issues = []
        total_rows = len(data)

        # Check for missing values in key columns
        missing_count = data["value"].isna().sum()
        missing_pct = (missing_count / total_rows) * 100 if total_rows > 0 else 0

        # Check for quality-flagged data
        flagged_count = data["qflag"].isin(GHCN_QUALITY_FLAGS.keys()).sum()
        flagged_pct = (flagged_count / total_rows) * 100 if total_rows > 0 else 0
        if flagged_pct > 10:
            issues.append(f"High percentage of quality-flagged data: {flagged_pct:.1f}%")

        # Check for temperature outliers (< -60C or > 60C)
        temp_data = data[data["element"].isin(["TMAX", "TMIN"])]
        if len(temp_data) > 0:
            temp_values = temp_data["value_converted"]
            temp_outliers = ((temp_values < -60) | (temp_values > 60)).sum()
            if temp_outliers > 0:
                issues.append(f"Found {temp_outliers} temperature outliers (outside -60C to 60C)")
        else:
            temp_outliers = 0

        # Check for precipitation outliers (> 2000mm)
        prcp_data = data[data["element"] == "PRCP"]
        if len(prcp_data) > 0:
            prcp_values = prcp_data["value_converted"]
            prcp_outliers = (prcp_values > 2000).sum()
            if prcp_outliers > 0:
                issues.append(f"Found {prcp_outliers} precipitation outliers (> 2000mm)")
        else:
            prcp_outliers = 0

        outliers_count = temp_outliers + prcp_outliers

        # Calculate stats
        stats = {
            "stations": data["station_id"].nunique(),
            "elements": data["element"].unique().tolist(),
            "date_range": (
                data["date"].min().isoformat() if len(data) > 0 else None,
                data["date"].max().isoformat() if len(data) > 0 else None,
            ),
            "flagged_pct": flagged_pct,
        }

        # Consider valid if < 20% missing and no critical issues
        valid = missing_pct < 20 and len(issues) == 0

        return ValidationResult(
            valid=valid,
            total_rows=total_rows,
            missing_pct=missing_pct,
            outliers_count=outliers_count,
            issues=issues,
            stats=stats,
        )
