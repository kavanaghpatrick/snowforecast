"""Base classes and protocols for pipelines."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pandas as pd


@dataclass
class ValidationResult:
    """Result of data validation.

    Attributes:
        valid: Whether the data passed all validation checks
        total_rows: Total number of rows in the dataset
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


class BasePipeline(ABC):
    """Base class for all data ingestion pipelines.

    All pipelines must implement:
    - download(): Fetch raw data from source
    - process(): Transform raw data to standardized format
    - validate(): Check data quality and completeness

    Example:
        class SnotelPipeline(BasePipeline):
            def download(self, start_date, end_date):
                # Download SNOTEL data
                ...

            def process(self, raw_path):
                # Process to DataFrame
                ...

            def validate(self, df):
                # Validate data
                ...
    """

    @abstractmethod
    def download(self, start_date: str, end_date: str, **kwargs) -> Path | list[Path]:
        """Download raw data from the source.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            **kwargs: Additional source-specific parameters

        Returns:
            Path to downloaded data, or list of Paths for sharded data.
            - Single file: Path to the file
            - Directory: Path to directory containing files
            - Sharded: List of Paths (e.g., one per month for ERA5/HRRR)
        """
        pass

    @abstractmethod
    def process(self, raw_path: Path | list[Path]) -> pd.DataFrame:
        """Process raw data into standardized format.

        Args:
            raw_path: Path(s) to raw data from download().
                      May be single Path or list of Paths for sharded data.

        Returns:
            DataFrame with standardized columns and types
        """
        pass

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """Validate processed data for quality and completeness.

        Args:
            df: Processed DataFrame from process()

        Returns:
            ValidationResult with quality metrics and issues
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
