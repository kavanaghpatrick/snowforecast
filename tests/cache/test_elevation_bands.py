"""Tests for elevation band forecasting.

Tests lapse rate calculations, snow line computation, and elevation band forecasts.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from snowforecast.cache.elevation_bands import (
    LAPSE_RATE_C_PER_KM,
    RAIN_THRESHOLD_C,
    RESORT_VERTICAL_M,
    SNOW_THRESHOLD_C,
    ElevationBandForecast,
    ElevationBandResult,
    PrecipType,
    adjust_new_snow,
    apply_lapse_rate,
    compute_elevation_bands,
    get_precip_type,
    get_snow_line,
    get_summit_elevation,
)


class TestLapseRate:
    """Tests for lapse rate application."""

    def test_lapse_rate_constant(self):
        """Verify lapse rate is standard atmospheric value."""
        assert LAPSE_RATE_C_PER_KM == -6.5

    def test_apply_lapse_rate_increase_elevation(self):
        """Temperature decreases when going up."""
        base_temp = 0.0
        base_elev = 2000
        target_elev = 3000  # 1km higher

        result = apply_lapse_rate(base_temp, base_elev, target_elev)

        # 1km up = -6.5C adjustment
        assert result == pytest.approx(-6.5, rel=0.01)

    def test_apply_lapse_rate_decrease_elevation(self):
        """Temperature increases when going down."""
        base_temp = -5.0
        base_elev = 3000
        target_elev = 2000  # 1km lower

        result = apply_lapse_rate(base_temp, base_elev, target_elev)

        # 1km down = +6.5C adjustment
        assert result == pytest.approx(1.5, rel=0.01)

    def test_apply_lapse_rate_same_elevation(self):
        """No change at same elevation."""
        base_temp = -3.0
        base_elev = 2500
        target_elev = 2500

        result = apply_lapse_rate(base_temp, base_elev, target_elev)

        assert result == pytest.approx(base_temp, rel=0.01)

    def test_apply_lapse_rate_500m_change(self):
        """Half kilometer elevation change."""
        base_temp = 2.0
        base_elev = 2000
        target_elev = 2500  # 500m higher

        result = apply_lapse_rate(base_temp, base_elev, target_elev)

        # 0.5km up = -3.25C adjustment
        expected = 2.0 + (-6.5 * 0.5)
        assert result == pytest.approx(expected, rel=0.01)


class TestPrecipType:
    """Tests for precipitation type determination."""

    def test_snow_below_threshold(self):
        """Snow when temp below snow threshold."""
        result = get_precip_type(-5.0, has_precip=True)
        assert result == PrecipType.SNOW

    def test_rain_above_threshold(self):
        """Rain when temp above rain threshold."""
        result = get_precip_type(5.0, has_precip=True)
        assert result == PrecipType.RAIN

    def test_mixed_between_thresholds(self):
        """Mixed precip between thresholds."""
        result = get_precip_type(0.5, has_precip=True)
        assert result == PrecipType.MIXED

    def test_no_precip_returns_none(self):
        """No precip type when no precipitation."""
        result = get_precip_type(-10.0, has_precip=False)
        assert result == PrecipType.NONE

    def test_at_snow_threshold(self):
        """At exactly snow threshold should be snow."""
        result = get_precip_type(SNOW_THRESHOLD_C, has_precip=True)
        assert result == PrecipType.SNOW

    def test_at_rain_threshold(self):
        """At exactly rain threshold should be rain."""
        result = get_precip_type(RAIN_THRESHOLD_C, has_precip=True)
        assert result == PrecipType.RAIN


class TestAdjustNewSnow:
    """Tests for new snow amount adjustment."""

    def test_rain_gets_no_snow(self):
        """Rain precip type results in zero snow."""
        result = adjust_new_snow(10.0, 5.0, PrecipType.RAIN)
        assert result == 0.0

    def test_mixed_reduced_snow(self):
        """Mixed precip reduces snow amount."""
        result = adjust_new_snow(10.0, 0.5, PrecipType.MIXED)
        assert result == 5.0  # 50%

    def test_snow_at_moderate_cold(self):
        """Normal snow at moderate cold temps."""
        result = adjust_new_snow(10.0, -3.0, PrecipType.SNOW)
        assert result == 10.0

    def test_snow_at_very_cold(self):
        """Fluffy snow at very cold temps."""
        result = adjust_new_snow(10.0, -15.0, PrecipType.SNOW)
        assert result == 13.0  # 130%

    def test_snow_at_cold(self):
        """Slightly fluffy snow at cold temps."""
        result = adjust_new_snow(10.0, -7.0, PrecipType.SNOW)
        assert result == pytest.approx(11.0, rel=0.01)

    def test_no_precip_no_snow(self):
        """No precip type results in zero snow."""
        result = adjust_new_snow(10.0, -5.0, PrecipType.NONE)
        assert result == 0.0


class TestSnowLine:
    """Tests for snow line calculation."""

    def test_snow_line_warm_base(self):
        """Snow line above base when warm at base."""
        base_temp = 5.0
        base_elev = 2000
        summit_elev = 3500

        result = get_snow_line(base_temp, base_elev, summit_elev)

        # Need to go up ~923m to reach -1C from 5C
        # (5 - (-1)) / 6.5 = 0.923 km = 923m
        expected = 2000 + 923
        assert result == pytest.approx(expected, rel=0.05)

    def test_snow_line_at_base_when_cold(self):
        """Snow line at base when already cold."""
        base_temp = -5.0
        base_elev = 2000
        summit_elev = 3500

        result = get_snow_line(base_temp, base_elev, summit_elev)

        # Already snowing at base
        assert result == 2000

    def test_snow_line_clamped_to_summit(self):
        """Snow line capped at summit elevation."""
        base_temp = 15.0  # Very warm
        base_elev = 2000
        summit_elev = 2500  # Low summit

        result = get_snow_line(base_temp, base_elev, summit_elev)

        # Snow line would be above summit, clamp to summit
        assert result == summit_elev

    def test_snow_line_at_threshold(self):
        """Snow line at base when temp equals threshold."""
        base_temp = SNOW_THRESHOLD_C
        base_elev = 2000
        summit_elev = 3000

        result = get_snow_line(base_temp, base_elev, summit_elev)

        assert result == 2000


class TestElevationBandForecast:
    """Tests for ElevationBandForecast dataclass."""

    def test_precip_icon_snow(self):
        """Snow icon for snow precip type."""
        band = ElevationBandForecast(
            name="Summit",
            elevation_m=3000,
            temp_c=-10,
            new_snow_cm=15,
            snow_depth_cm=100,
            precip_type=PrecipType.SNOW,
            snowfall_probability=0.9,
        )
        assert band.precip_icon == "snow"

    def test_precip_icon_rain(self):
        """Rain icon for rain precip type."""
        band = ElevationBandForecast(
            name="Base",
            elevation_m=2000,
            temp_c=5,
            new_snow_cm=0,
            snow_depth_cm=50,
            precip_type=PrecipType.RAIN,
            snowfall_probability=0.1,
        )
        assert band.precip_icon == "rain"

    def test_temp_display(self):
        """Temperature display formatting."""
        band = ElevationBandForecast(
            name="Mid",
            elevation_m=2500,
            temp_c=-4.7,
            new_snow_cm=8,
            snow_depth_cm=80,
            precip_type=PrecipType.SNOW,
            snowfall_probability=0.7,
        )
        assert band.temp_display == "-5C"  # Rounds


class TestElevationBandResult:
    """Tests for ElevationBandResult dataclass."""

    def test_bands_order(self):
        """Bands returned in Summit, Mid, Base order."""
        base = ElevationBandForecast(
            name="Base", elevation_m=2000, temp_c=0, new_snow_cm=0,
            snow_depth_cm=50, precip_type=PrecipType.MIXED, snowfall_probability=0.3
        )
        mid = ElevationBandForecast(
            name="Mid", elevation_m=2500, temp_c=-3, new_snow_cm=5,
            snow_depth_cm=70, precip_type=PrecipType.SNOW, snowfall_probability=0.6
        )
        summit = ElevationBandForecast(
            name="Summit", elevation_m=3000, temp_c=-6, new_snow_cm=10,
            snow_depth_cm=100, precip_type=PrecipType.SNOW, snowfall_probability=0.9
        )

        result = ElevationBandResult(
            ski_area="Test Resort",
            forecast_time=datetime.now(),
            base=base,
            mid=mid,
            summit=summit,
            snow_line_m=2200,
        )

        bands = result.bands
        assert len(bands) == 3
        assert bands[0].name == "Summit"
        assert bands[1].name == "Mid"
        assert bands[2].name == "Base"


class TestComputeElevationBands:
    """Tests for compute_elevation_bands function."""

    @pytest.fixture
    def mock_predictor(self):
        """Create mock predictor."""
        predictor = MagicMock()

        # Mock forecast result
        from snowforecast.api.schemas import ConfidenceInterval, ForecastResult

        forecast = ForecastResult(
            snow_depth_cm=80.0,
            new_snow_cm=10.0,
            snowfall_probability=0.7,
        )
        confidence = ConfidenceInterval(lower=5.0, upper=15.0)
        predictor.predict.return_value = (forecast, confidence)

        # Mock HRRR data
        predictor.fetch_hrrr_forecast.return_value = {
            "temp_k": 268.15,  # -5C
            "precip_mm": 5.0,
            "snow_depth_m": 0.8,
            "categorical_snow": 1.0,
        }

        return predictor

    def test_compute_basic(self, mock_predictor):
        """Basic computation with mock predictor."""
        result = compute_elevation_bands(
            predictor=mock_predictor,
            lat=39.5,
            lon=-106.0,
            base_elev_m=2500,
            summit_elev_m=3500,
            ski_area_name="Test Resort",
            target_datetime=datetime(2025, 1, 15, 12, 0),
        )

        assert result.ski_area == "Test Resort"
        assert len(result.bands) == 3
        assert result.base.elevation_m == 2500
        assert result.summit.elevation_m == 3500
        assert result.mid.elevation_m == 3000  # Midpoint

    def test_temperature_decreases_with_elevation(self, mock_predictor):
        """Summit should be colder than base."""
        result = compute_elevation_bands(
            predictor=mock_predictor,
            lat=39.5,
            lon=-106.0,
            base_elev_m=2500,
            summit_elev_m=3500,
            ski_area_name="Test Resort",
        )

        assert result.summit.temp_c < result.mid.temp_c
        assert result.mid.temp_c < result.base.temp_c

    def test_more_snow_at_higher_elevation(self, mock_predictor):
        """Summit should have more snow depth."""
        result = compute_elevation_bands(
            predictor=mock_predictor,
            lat=39.5,
            lon=-106.0,
            base_elev_m=2500,
            summit_elev_m=3500,
            ski_area_name="Test Resort",
        )

        assert result.summit.snow_depth_cm > result.base.snow_depth_cm

    def test_snow_line_calculated(self, mock_predictor):
        """Snow line should be calculated."""
        result = compute_elevation_bands(
            predictor=mock_predictor,
            lat=39.5,
            lon=-106.0,
            base_elev_m=2500,
            summit_elev_m=3500,
            ski_area_name="Test Resort",
        )

        assert result.snow_line_m >= 2500
        assert result.snow_line_m <= 3500

    def test_fallback_when_no_hrrr(self, mock_predictor):
        """Should use fallback when HRRR unavailable."""
        mock_predictor.fetch_hrrr_forecast.return_value = None

        result = compute_elevation_bands(
            predictor=mock_predictor,
            lat=39.5,
            lon=-106.0,
            base_elev_m=2500,
            summit_elev_m=3500,
            ski_area_name="Test Resort",
            target_datetime=datetime(2025, 1, 15, 12, 0),  # Winter
        )

        # Should still produce valid result
        assert result is not None
        assert len(result.bands) == 3


class TestGetSummitElevation:
    """Tests for get_summit_elevation function."""

    def test_known_resort(self):
        """Summit elevation for known resort."""
        # Stevens Pass: base 1241m, vertical 823m
        result = get_summit_elevation("Stevens Pass", 1241)
        assert result == 1241 + 823

    def test_unknown_resort_uses_default(self):
        """Unknown resort uses default vertical."""
        result = get_summit_elevation("Unknown Resort", 2000)
        assert result == 2000 + 800  # Default vertical

    def test_all_resorts_have_vertical(self):
        """All resorts in list have reasonable vertical."""
        for name, vertical in RESORT_VERTICAL_M.items():
            assert vertical > 400, f"{name} vertical too small"
            assert vertical < 2000, f"{name} vertical too large"
