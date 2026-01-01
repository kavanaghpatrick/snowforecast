"""Base classes and protocols for pipelines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar
import pandas as pd

# Type variable for flexible data types (DataFrame, xarray.Dataset, etc.)
DataT = TypeVar("DataT")


@dataclass
class ValidationResult:
    """Result of data validation.

    Attributes:
        valid: Whether the data passed all validation checks
        total_rows: Total number of rows/records in the dataset
        missing_pct: Percentage of missing values (0-100)
        outliers_count: Number of outlier values detected
        issues: List of validation issues found
        stats: Dictionary of summary statistics
    """

    valid: bool
    total_rows: int
    missing_pct: float
    outliers_count: int = 0
    issues: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        return (
            f"ValidationResult({status}, "
            f"rows={self.total_rows}, "
            f"missing={self.missing_pct:.1f}%, "
            f"outliers={self.outliers_count})"
        )

    @classmethod
    def from_dict(cls, d: dict) -> "ValidationResult":
        """Create ValidationResult from a dictionary (for backwards compat)."""
        return cls(
            valid=d.get("valid", True),
            total_rows=d.get("total_rows", 0),
            missing_pct=d.get("missing_pct", 0.0),
            outliers_count=d.get("outliers_count", 0),
            issues=d.get("issues", []),
            stats=d.get("stats", {}),
        )


class BasePipeline(ABC):
    """Abstract base class for all data ingestion pipelines.

    This is the minimal interface all pipelines must satisfy.
    Use TemporalPipeline for time-series data (SNOTEL, GHCN, ERA5, HRRR).
    Use StaticPipeline for static data (DEM, OpenSkiMap).
    """

    @abstractmethod
    def validate(self, data: Any) -> ValidationResult:
        """Validate data for quality and completeness.

        Args:
            data: Data to validate (DataFrame, Dataset, etc.)

        Returns:
            ValidationResult with quality metrics and issues
        """
        pass


class TemporalPipeline(BasePipeline):
    """Base class for time-series data pipelines (SNOTEL, GHCN, ERA5, HRRR).

    These pipelines download data for a date range and produce DataFrames.
    """

    @abstractmethod
    def download(self, start_date: str, end_date: str, **kwargs) -> Path | list[Path]:
        """Download raw data from the source for a date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            **kwargs: Additional source-specific parameters

        Returns:
            Path to downloaded data, or list of Paths for sharded data.
        """
        pass

    @abstractmethod
    def process(self, raw_path: Path | list[Path]) -> pd.DataFrame:
        """Process raw data into standardized DataFrame format.

        Args:
            raw_path: Path(s) to raw data from download().

        Returns:
            DataFrame with standardized columns and types
        """
        pass

    def run(
        self,
        start_date: str,
        end_date: str,
        raise_on_invalid: bool = True,
        **kwargs
    ) -> tuple[pd.DataFrame, ValidationResult]:
        """Run the full pipeline: download → process → validate.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            raise_on_invalid: If True, raise ValueError when validation fails
            **kwargs: Additional parameters passed to download()

        Returns:
            Tuple of (processed DataFrame, validation result)

        Raises:
            ValueError: If raise_on_invalid=True and validation fails
        """
        raw_path = self.download(start_date, end_date, **kwargs)
        df = self.process(raw_path)
        validation = self.validate(df)

        if raise_on_invalid and not validation.valid:
            raise ValueError(
                f"Data validation failed: {validation.issues}. "
                f"Missing: {validation.missing_pct:.1f}%, "
                f"Outliers: {validation.outliers_count}"
            )

        return df, validation


class StaticPipeline(BasePipeline):
    """Base class for static/spatial data pipelines (DEM, OpenSkiMap).

    These pipelines download data for a region (not time range) and may
    produce DataFrames, GeoDataFrames, or other formats.
    """

    @abstractmethod
    def download(self, **kwargs) -> Path | list[Path]:
        """Download static data from the source.

        Args:
            **kwargs: Source-specific parameters (bbox, region, etc.)

        Returns:
            Path to downloaded data, or list of Paths.
        """
        pass

    @abstractmethod
    def process(self, raw_path: Path | list[Path]) -> Any:
        """Process raw data into usable format.

        Args:
            raw_path: Path(s) to raw data from download().

        Returns:
            Processed data (DataFrame, GeoDataFrame, lookup table, etc.)
        """
        pass

    def run(
        self,
        raise_on_invalid: bool = True,
        **kwargs
    ) -> tuple[Any, ValidationResult]:
        """Run the full pipeline: download → process → validate.

        Args:
            raise_on_invalid: If True, raise ValueError when validation fails
            **kwargs: Parameters passed to download()

        Returns:
            Tuple of (processed data, validation result)

        Raises:
            ValueError: If raise_on_invalid=True and validation fails
        """
        raw_path = self.download(**kwargs)
        data = self.process(raw_path)
        validation = self.validate(data)

        if raise_on_invalid and not validation.valid:
            raise ValueError(
                f"Data validation failed: {validation.issues}. "
                f"Missing: {validation.missing_pct:.1f}%, "
                f"Outliers: {validation.outliers_count}"
            )

        return data, validation


class GriddedPipeline(TemporalPipeline):
    """Base class for gridded weather data pipelines (ERA5, HRRR).

    Extends TemporalPipeline with methods for extracting point data
    from gridded datasets. Can return xarray.Dataset in addition to DataFrame.
    """

    @abstractmethod
    def process_to_dataset(self, raw_path: Path | list[Path]) -> Any:
        """Process raw data to xarray.Dataset (preserving spatial structure).

        Args:
            raw_path: Path(s) to raw data from download().

        Returns:
            xarray.Dataset with standardized variables
        """
        pass

    @abstractmethod
    def extract_at_points(
        self,
        data: Any,
        points: list[tuple[float, float]]
    ) -> pd.DataFrame:
        """Extract time series at specific lat/lon points.

        Args:
            data: Dataset from process_to_dataset()
            points: List of (lat, lon) tuples

        Returns:
            DataFrame with time series for each point
        """
        pass

    def process(self, raw_path: Path | list[Path]) -> pd.DataFrame:
        """Default process: convert Dataset to DataFrame.

        Override if you need different behavior.
        """
        ds = self.process_to_dataset(raw_path)
        # Convert to DataFrame - specific implementation in subclasses
        return ds.to_dataframe().reset_index()
