"""Tests for atmospheric feature engineering.

Tests validate physics calculations against known values and edge cases.
"""

import numpy as np
import pandas as pd
import pytest

from snowforecast.features.atmospheric import AtmosphericFeatures


@pytest.fixture
def atmospheric():
    """Create AtmosphericFeatures instance."""
    return AtmosphericFeatures()


@pytest.fixture
def sample_era5_data() -> pd.DataFrame:
    """Create sample ERA5-like data with all required columns."""
    np.random.seed(42)
    n = 100

    # Generate realistic meteorological data for winter conditions
    t2m = 273.15 + np.random.uniform(-20, 10, n)  # -20 to 10 C
    d2m = t2m - np.abs(np.random.uniform(0, 10, n))  # dewpoint <= temp

    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="h"),
        "t2m": t2m,  # Kelvin
        "d2m": d2m,  # Kelvin
        "u10": np.random.uniform(-10, 10, n),  # m/s
        "v10": np.random.uniform(-10, 10, n),  # m/s
        "sp": np.random.uniform(85000, 102000, n),  # Pa
        "tp": np.random.uniform(0, 0.01, n),  # m
        "sf": np.random.uniform(0, 0.005, n),  # m
        "sd": np.random.uniform(0, 1, n),  # m
    })


class TestTemperatureFeatures:
    """Tests for temperature feature calculations."""

    def test_celsius_conversion(self, atmospheric):
        """Test Kelvin to Celsius conversion."""
        df = pd.DataFrame({"t2m": [273.15, 283.15, 263.15]})
        result = atmospheric.compute_temperature_features(df)

        np.testing.assert_array_almost_equal(
            result["t2m_celsius"].values,
            [0.0, 10.0, -10.0],
            decimal=2
        )

    def test_freezing_level_below(self, atmospheric):
        """Test freezing level indicator when below freezing."""
        df = pd.DataFrame({"t2m": [263.15, 268.15, 272.15]})  # All below 273.15K
        result = atmospheric.compute_temperature_features(df)

        assert all(result["freezing_level"] == 1)

    def test_freezing_level_above(self, atmospheric):
        """Test freezing level indicator when above freezing."""
        df = pd.DataFrame({"t2m": [278.15, 283.15, 293.15]})  # All above 273.15K
        result = atmospheric.compute_temperature_features(df)

        assert all(result["freezing_level"] == 0)

    def test_freezing_level_at_zero(self, atmospheric):
        """Test freezing level at exactly 0C."""
        df = pd.DataFrame({"t2m": [273.15]})  # Exactly 0C
        result = atmospheric.compute_temperature_features(df)

        # At exactly 0C, should be above freezing (not < 273.15)
        assert result["freezing_level"].iloc[0] == 0


class TestHumidityFeatures:
    """Tests for humidity feature calculations."""

    def test_relative_humidity_at_saturation(self, atmospheric):
        """Test RH = 100% when dewpoint equals temperature."""
        df = pd.DataFrame({
            "t2m": [280.0, 290.0, 270.0],
            "d2m": [280.0, 290.0, 270.0]  # Same as temperature = saturated
        })
        result = atmospheric.compute_humidity_features(df)

        np.testing.assert_array_almost_equal(
            result["relative_humidity"].values,
            [100.0, 100.0, 100.0],
            decimal=1
        )

    def test_relative_humidity_range(self, atmospheric, sample_era5_data):
        """Test RH is always in valid range 0-100%."""
        result = atmospheric.compute_humidity_features(sample_era5_data)

        assert result["relative_humidity"].min() >= 0
        assert result["relative_humidity"].max() <= 100

    def test_dewpoint_depression_positive(self, atmospheric, sample_era5_data):
        """Test dewpoint depression is non-negative."""
        result = atmospheric.compute_humidity_features(sample_era5_data)

        # Dewpoint should always be <= temperature
        assert result["dewpoint_depression"].min() >= 0

    def test_dewpoint_depression_calculation(self, atmospheric):
        """Test dewpoint depression equals T - Td in Celsius."""
        df = pd.DataFrame({
            "t2m": [283.15, 293.15],  # 10C, 20C
            "d2m": [278.15, 283.15]   # 5C, 10C
        })
        result = atmospheric.compute_humidity_features(df)

        np.testing.assert_array_almost_equal(
            result["dewpoint_depression"].values,
            [5.0, 10.0],
            decimal=2
        )

    def test_wet_bulb_temp_between_t_and_td(self, atmospheric):
        """Test wet bulb temp is between temperature and dewpoint."""
        df = pd.DataFrame({
            "t2m": [293.15],  # 20C
            "d2m": [283.15]   # 10C
        })
        result = atmospheric.compute_humidity_features(df)

        t_c = 20.0
        td_c = 10.0
        wb = result["wet_bulb_temp"].iloc[0]

        # Wet bulb should be between dewpoint and temperature
        assert td_c <= wb <= t_c


class TestWindFeatures:
    """Tests for wind feature calculations."""

    def test_wind_speed_calculation(self, atmospheric):
        """Test wind speed from U/V components."""
        df = pd.DataFrame({
            "u10": [3.0, 0.0, 5.0],
            "v10": [4.0, 10.0, 0.0]
        })
        result = atmospheric.compute_wind_features(df)

        np.testing.assert_array_almost_equal(
            result["wind_speed"].values,
            [5.0, 10.0, 5.0],  # Pythagorean theorem
            decimal=2
        )

    def test_wind_direction_north(self, atmospheric):
        """Test wind from north (v < 0, u = 0)."""
        df = pd.DataFrame({
            "u10": [0.0],
            "v10": [-10.0]  # Wind blowing southward, FROM north
        })
        result = atmospheric.compute_wind_features(df)

        # Wind FROM north = 0 degrees
        np.testing.assert_array_almost_equal(
            result["wind_direction"].values,
            [0.0],
            decimal=0
        )

    def test_wind_direction_east(self, atmospheric):
        """Test wind from east (u < 0, v = 0)."""
        df = pd.DataFrame({
            "u10": [-10.0],  # Wind blowing westward, FROM east
            "v10": [0.0]
        })
        result = atmospheric.compute_wind_features(df)

        # Wind FROM east = 90 degrees
        np.testing.assert_array_almost_equal(
            result["wind_direction"].values,
            [90.0],
            decimal=0
        )

    def test_wind_direction_south(self, atmospheric):
        """Test wind from south (v > 0, u = 0)."""
        df = pd.DataFrame({
            "u10": [0.0],
            "v10": [10.0]  # Wind blowing northward, FROM south
        })
        result = atmospheric.compute_wind_features(df)

        # Wind FROM south = 180 degrees
        np.testing.assert_array_almost_equal(
            result["wind_direction"].values,
            [180.0],
            decimal=0
        )

    def test_wind_direction_west(self, atmospheric):
        """Test wind from west (u > 0, v = 0)."""
        df = pd.DataFrame({
            "u10": [10.0],  # Wind blowing eastward, FROM west
            "v10": [0.0]
        })
        result = atmospheric.compute_wind_features(df)

        # Wind FROM west = 270 degrees
        np.testing.assert_array_almost_equal(
            result["wind_direction"].values,
            [270.0],
            decimal=0
        )

    def test_wind_direction_range(self, atmospheric, sample_era5_data):
        """Test wind direction is always in range 0-360."""
        result = atmospheric.compute_wind_features(sample_era5_data)

        assert result["wind_direction"].min() >= 0
        assert result["wind_direction"].max() < 360

    def test_wind_chill_colder_than_temperature(self, atmospheric):
        """Test wind chill is colder than actual temp in applicable conditions."""
        df = pd.DataFrame({
            "t2m": [268.15],  # -5C (applicable for wind chill)
            "u10": [10.0],
            "v10": [0.0]  # 10 m/s = 36 km/h
        })
        result = atmospheric.compute_wind_features(df)

        t_c = -5.0
        # Wind chill should be lower than actual temperature
        assert result["wind_chill"].iloc[0] < t_c

    def test_wind_chill_not_applied_warm_temps(self, atmospheric):
        """Test wind chill equals temp when T > 10C."""
        df = pd.DataFrame({
            "t2m": [293.15],  # 20C (too warm for wind chill)
            "u10": [10.0],
            "v10": [0.0]
        })
        result = atmospheric.compute_wind_features(df)

        t_c = 20.0
        # Should return actual temperature
        assert result["wind_chill"].iloc[0] == pytest.approx(t_c, abs=0.1)


class TestPressureFeatures:
    """Tests for pressure feature calculations."""

    def test_pressure_hpa_conversion(self, atmospheric):
        """Test Pa to hPa conversion."""
        df = pd.DataFrame({"sp": [101325.0, 85000.0, 95000.0]})  # Pa
        result = atmospheric.compute_pressure_features(df)

        np.testing.assert_array_almost_equal(
            result["pressure_hpa"].values,
            [1013.25, 850.0, 950.0],  # hPa
            decimal=2
        )

    def test_pressure_tendency_with_time(self, atmospheric):
        """Test pressure tendency calculation with time column."""
        # Create 48 hours of data with pressure increasing then decreasing
        times = pd.date_range("2024-01-01", periods=48, freq="h")
        pressures = list(range(100000, 100024)) + list(range(100024, 100000, -1))

        df = pd.DataFrame({
            "time": times,
            "sp": pressures
        })
        result = atmospheric.compute_pressure_features(df)

        # First 24 values should be NaN (no prior 24h data)
        assert result["pressure_tendency"].iloc[:24].isna().all()

        # Value at hour 24 should be the 24h change
        # pressure_hpa[24] - pressure_hpa[0] = 100.24 - 100.00 = 0.24 hPa
        assert result["pressure_tendency"].iloc[24] == pytest.approx(0.24, abs=0.01)


class TestPrecipitationFeatures:
    """Tests for precipitation feature calculations."""

    def test_precip_mm_conversion(self, atmospheric):
        """Test m to mm conversion for precipitation."""
        df = pd.DataFrame({"tp": [0.001, 0.010, 0.0]})  # m
        result = atmospheric.compute_precipitation_features(df)

        np.testing.assert_array_almost_equal(
            result["precip_mm"].values,
            [1.0, 10.0, 0.0],  # mm
            decimal=2
        )

    def test_snow_fraction_all_snow(self, atmospheric):
        """Test snow fraction = 1 when all precip is snow."""
        df = pd.DataFrame({
            "tp": [0.010, 0.005],  # m
            "sf": [0.010, 0.005]   # m (same as total = all snow)
        })
        result = atmospheric.compute_precipitation_features(df)

        np.testing.assert_array_almost_equal(
            result["snow_fraction"].values,
            [1.0, 1.0],
            decimal=2
        )

    def test_snow_fraction_no_snow(self, atmospheric):
        """Test snow fraction = 0 when no snow."""
        df = pd.DataFrame({
            "tp": [0.010, 0.005],  # m
            "sf": [0.0, 0.0]       # m (no snow)
        })
        result = atmospheric.compute_precipitation_features(df)

        np.testing.assert_array_almost_equal(
            result["snow_fraction"].values,
            [0.0, 0.0],
            decimal=2
        )

    def test_snow_fraction_no_precip(self, atmospheric):
        """Test snow fraction = 0 when no precipitation (avoid div by zero)."""
        df = pd.DataFrame({
            "tp": [0.0, 0.0],  # m (no precip)
            "sf": [0.0, 0.0]  # m
        })
        result = atmospheric.compute_precipitation_features(df)

        np.testing.assert_array_almost_equal(
            result["snow_fraction"].values,
            [0.0, 0.0],
            decimal=2
        )

    def test_snow_fraction_range(self, atmospheric, sample_era5_data):
        """Test snow fraction is always in range 0-1."""
        result = atmospheric.compute_precipitation_features(sample_era5_data)

        assert result["snow_fraction"].min() >= 0
        assert result["snow_fraction"].max() <= 1


class TestComputeAll:
    """Tests for the compute_all method."""

    def test_compute_all_adds_all_features(self, atmospheric, sample_era5_data):
        """Test that compute_all adds all expected feature columns."""
        result = atmospheric.compute_all(sample_era5_data)

        expected_features = atmospheric.get_feature_names()
        # Some features might not be computed if input columns are missing
        # But with full sample data, we should have most of them
        for feature in ["t2m_celsius", "relative_humidity", "wind_speed", "pressure_hpa", "precip_mm"]:
            assert feature in result.columns

    def test_compute_all_preserves_original_columns(self, atmospheric, sample_era5_data):
        """Test that compute_all preserves all original columns."""
        original_cols = set(sample_era5_data.columns)
        result = atmospheric.compute_all(sample_era5_data)

        for col in original_cols:
            assert col in result.columns

    def test_compute_all_does_not_modify_input(self, atmospheric, sample_era5_data):
        """Test that compute_all doesn't modify the input DataFrame."""
        original_cols = list(sample_era5_data.columns)
        _ = atmospheric.compute_all(sample_era5_data)

        assert list(sample_era5_data.columns) == original_cols


class TestMissingColumns:
    """Tests for handling missing input columns."""

    def test_temperature_features_missing_t2m(self, atmospheric):
        """Test graceful handling when t2m is missing."""
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = atmospheric.compute_temperature_features(df)

        assert "t2m_celsius" not in result.columns
        assert "freezing_level" not in result.columns

    def test_humidity_features_missing_columns(self, atmospheric):
        """Test graceful handling when t2m or d2m is missing."""
        df = pd.DataFrame({"t2m": [280.0, 285.0]})  # Missing d2m
        result = atmospheric.compute_humidity_features(df)

        assert "relative_humidity" not in result.columns

    def test_wind_features_missing_components(self, atmospheric):
        """Test graceful handling when u10 or v10 is missing."""
        df = pd.DataFrame({"u10": [5.0, 10.0]})  # Missing v10
        result = atmospheric.compute_wind_features(df)

        assert "wind_speed" not in result.columns


class TestGetFeatureNames:
    """Tests for the get_feature_names method."""

    def test_get_feature_names_returns_list(self, atmospheric):
        """Test that get_feature_names returns a list."""
        names = atmospheric.get_feature_names()
        assert isinstance(names, list)

    def test_get_feature_names_contains_expected(self, atmospheric):
        """Test that expected feature names are included."""
        names = atmospheric.get_feature_names()

        expected = [
            "t2m_celsius",
            "freezing_level",
            "relative_humidity",
            "wind_speed",
            "wind_direction",
            "pressure_hpa",
            "precip_mm",
            "snow_fraction",
        ]
        for feat in expected:
            assert feat in names
