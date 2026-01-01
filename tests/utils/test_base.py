"""Tests for base pipeline classes."""

import pytest
from pathlib import Path
import pandas as pd
from snowforecast.utils.base import (
    ValidationResult,
    BasePipeline,
    TemporalPipeline,
    StaticPipeline,
    GriddedPipeline,
)


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_create_valid_result(self):
        """Should create a valid result."""
        result = ValidationResult(
            valid=True,
            total_rows=1000,
            missing_pct=5.0,
            outliers_count=10,
        )
        assert result.valid is True
        assert result.total_rows == 1000
        assert result.missing_pct == 5.0
        assert result.outliers_count == 10
        assert result.issues == []
        assert result.stats == {}

    def test_create_invalid_result_with_issues(self):
        """Should create an invalid result with issues."""
        result = ValidationResult(
            valid=False,
            total_rows=100,
            missing_pct=50.0,
            issues=["Too many missing values", "Data gap detected"],
        )
        assert result.valid is False
        assert len(result.issues) == 2

    def test_str_valid(self):
        """Should format valid result as string."""
        result = ValidationResult(
            valid=True,
            total_rows=1000,
            missing_pct=5.0,
            outliers_count=10,
        )
        s = str(result)
        assert "VALID" in s
        assert "1000" in s
        assert "5.0%" in s

    def test_str_invalid(self):
        """Should format invalid result as string."""
        result = ValidationResult(
            valid=False,
            total_rows=100,
            missing_pct=50.0,
        )
        s = str(result)
        assert "INVALID" in s

    def test_from_dict(self):
        """Should create result from dictionary."""
        d = {
            "valid": True,
            "total_rows": 500,
            "missing_pct": 2.5,
            "outliers_count": 5,
            "issues": ["Warning: some issue"],
            "stats": {"mean": 10.5},
        }
        result = ValidationResult.from_dict(d)
        assert result.valid is True
        assert result.total_rows == 500
        assert result.missing_pct == 2.5
        assert result.outliers_count == 5
        assert len(result.issues) == 1
        assert result.stats["mean"] == 10.5

    def test_from_dict_defaults(self):
        """Should use defaults for missing keys."""
        result = ValidationResult.from_dict({})
        assert result.valid is True
        assert result.total_rows == 0
        assert result.missing_pct == 0.0
        assert result.outliers_count == 0


class ConcreteTemporalPipeline(TemporalPipeline):
    """Concrete implementation for testing."""

    def __init__(self, should_fail_validation: bool = False):
        self.should_fail_validation = should_fail_validation

    def download(self, start_date: str, end_date: str, **kwargs) -> Path:
        return Path("/tmp/test_data.parquet")

    def process(self, raw_path: Path) -> pd.DataFrame:
        return pd.DataFrame({"value": [1, 2, 3]})

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        if self.should_fail_validation:
            return ValidationResult(
                valid=False,
                total_rows=len(data),
                missing_pct=50.0,
                issues=["Validation failed"],
            )
        return ValidationResult(
            valid=True,
            total_rows=len(data),
            missing_pct=0.0,
        )


class ConcreteStaticPipeline(StaticPipeline):
    """Concrete implementation for testing."""

    def download(self, **kwargs) -> Path:
        return Path("/tmp/test_static.tif")

    def process(self, raw_path: Path) -> pd.DataFrame:
        return pd.DataFrame({"elevation": [1000, 2000, 3000]})

    def validate(self, data: pd.DataFrame) -> ValidationResult:
        return ValidationResult(
            valid=True,
            total_rows=len(data),
            missing_pct=0.0,
        )


class TestTemporalPipeline:
    """Tests for TemporalPipeline base class."""

    def test_run_success(self):
        """Should run full pipeline successfully."""
        pipeline = ConcreteTemporalPipeline()
        df, validation = pipeline.run("2023-01-01", "2023-01-31")
        assert len(df) == 3
        assert validation.valid is True

    def test_run_validation_failure_raises(self):
        """Should raise on validation failure when raise_on_invalid=True."""
        pipeline = ConcreteTemporalPipeline(should_fail_validation=True)
        with pytest.raises(ValueError, match="Data validation failed"):
            pipeline.run("2023-01-01", "2023-01-31")

    def test_run_validation_failure_no_raise(self):
        """Should return invalid result when raise_on_invalid=False."""
        pipeline = ConcreteTemporalPipeline(should_fail_validation=True)
        df, validation = pipeline.run(
            "2023-01-01", "2023-01-31", raise_on_invalid=False
        )
        assert len(df) == 3
        assert validation.valid is False


class TestStaticPipeline:
    """Tests for StaticPipeline base class."""

    def test_run_success(self):
        """Should run full pipeline successfully."""
        pipeline = ConcreteStaticPipeline()
        data, validation = pipeline.run()
        assert len(data) == 3
        assert validation.valid is True

    def test_download_no_dates(self):
        """Should download without date parameters."""
        pipeline = ConcreteStaticPipeline()
        path = pipeline.download()
        assert isinstance(path, Path)


class TestPipelineInheritance:
    """Tests for pipeline class hierarchy."""

    def test_temporal_is_base(self):
        """TemporalPipeline should inherit from BasePipeline."""
        assert issubclass(TemporalPipeline, BasePipeline)

    def test_static_is_base(self):
        """StaticPipeline should inherit from BasePipeline."""
        assert issubclass(StaticPipeline, BasePipeline)

    def test_gridded_is_temporal(self):
        """GriddedPipeline should inherit from TemporalPipeline."""
        assert issubclass(GriddedPipeline, TemporalPipeline)
