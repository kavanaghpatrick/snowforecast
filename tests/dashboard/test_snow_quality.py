"""Tests for snow quality indicators component.

Tests the snow quality calculation functions including:
- SLR (Snow-to-Liquid Ratio) calculation
- Temperature trend detection
- Quality classification logic
- Badge and display functions
"""


from unittest.mock import MagicMock, patch

from snowforecast.dashboard.components.snow_quality import (
    QualityMetrics,
    SnowQuality,
    calculate_slr,
    classify_snow_quality,
    create_quality_metrics,
    get_quality_badge,
    get_quality_explanation,
    get_slr_description,
    get_temp_trend,
)


class TestCalculateSLR:
    """Tests for Snow-to-Liquid Ratio calculation."""

    def test_typical_powder_slr(self):
        """15cm snow from 10mm precip gives 15:1 ratio."""
        result = calculate_slr(15.0, 10.0)
        assert result == 15.0

    def test_typical_average_slr(self):
        """10cm snow from 10mm precip gives 10:1 ratio."""
        result = calculate_slr(10.0, 10.0)
        assert result == 10.0

    def test_wet_snow_slr(self):
        """5cm snow from 10mm precip gives 5:1 ratio."""
        result = calculate_slr(5.0, 10.0)
        assert result == 5.0

    def test_zero_precip_returns_default(self):
        """Zero precipitation returns default 10:1 ratio."""
        result = calculate_slr(15.0, 0.0)
        assert result == 10.0

    def test_negative_precip_returns_default(self):
        """Negative precipitation returns default 10:1 ratio."""
        result = calculate_slr(15.0, -5.0)
        assert result == 10.0

    def test_zero_snow_zero_precip(self):
        """No snow and no precip returns default."""
        result = calculate_slr(0.0, 0.0)
        assert result == 10.0

    def test_zero_snow_with_precip(self):
        """No snow accumulation with precip gives 0 SLR."""
        result = calculate_slr(0.0, 10.0)
        assert result == 0.0

    def test_small_values(self):
        """Small values calculate correctly."""
        result = calculate_slr(1.0, 1.0)
        assert result == 10.0  # 1cm = 10mm, 10mm/1mm = 10

    def test_large_values(self):
        """Large values calculate correctly."""
        result = calculate_slr(100.0, 50.0)
        assert result == 20.0  # 100cm = 1000mm, 1000mm/50mm = 20

    def test_fractional_values(self):
        """Fractional values calculate correctly."""
        result = calculate_slr(7.5, 5.0)
        assert result == 15.0  # 7.5cm = 75mm, 75mm/5mm = 15

    def test_very_light_powder_slr(self):
        """Very light powder can have 20:1 ratio or higher."""
        result = calculate_slr(20.0, 10.0)
        assert result == 20.0


class TestGetTempTrend:
    """Tests for temperature trend detection."""

    def test_warming_trend(self):
        """Rising temperatures detected as warming."""
        temps = [-10, -8, -5, -2]
        result = get_temp_trend(temps)
        assert result == "warming"

    def test_cooling_trend(self):
        """Falling temperatures detected as cooling."""
        temps = [0, -2, -5, -8]
        result = get_temp_trend(temps)
        assert result == "cooling"

    def test_stable_trend_small_change(self):
        """Small temperature changes are stable."""
        temps = [-5, -4, -5, -6]
        result = get_temp_trend(temps)
        assert result == "stable"

    def test_stable_at_exactly_threshold(self):
        """Exactly 2 degree change is stable."""
        temps = [-5, -4, -3]
        result = get_temp_trend(temps)
        assert result == "stable"

    def test_warming_just_over_threshold(self):
        """Just over 2 degrees is warming."""
        temps = [-5, -3, -2]  # -5 to -2 = +3
        result = get_temp_trend(temps)
        assert result == "warming"

    def test_cooling_just_over_threshold(self):
        """Just over -2 degrees is cooling."""
        temps = [0, -1, -3]  # 0 to -3 = -3
        result = get_temp_trend(temps)
        assert result == "cooling"

    def test_single_temperature(self):
        """Single temperature returns stable."""
        temps = [-5]
        result = get_temp_trend(temps)
        assert result == "stable"

    def test_empty_list(self):
        """Empty list returns stable."""
        temps = []
        result = get_temp_trend(temps)
        assert result == "stable"

    def test_two_temperatures_warming(self):
        """Two temperatures can show warming."""
        temps = [-10, -5]  # +5 degrees
        result = get_temp_trend(temps)
        assert result == "warming"

    def test_two_temperatures_cooling(self):
        """Two temperatures can show cooling."""
        temps = [0, -5]  # -5 degrees
        result = get_temp_trend(temps)
        assert result == "cooling"

    def test_oscillating_temperatures_uses_endpoints(self):
        """Oscillating temps use only first and last values."""
        temps = [-5, 0, -8, -3]  # -5 to -3 = +2, stable
        result = get_temp_trend(temps)
        assert result == "stable"

    def test_positive_temperatures(self):
        """Works with positive temperatures."""
        temps = [2, 4, 8, 10]
        result = get_temp_trend(temps)
        assert result == "warming"


class TestClassifySnowQuality:
    """Tests for snow quality classification."""

    def test_powder_conditions(self):
        """High SLR and cold temp gives powder."""
        result = classify_snow_quality(15.0, -8.0, "stable")
        assert result == SnowQuality.POWDER

    def test_powder_at_boundary(self):
        """SLR just above 12 and temp just below -5 is powder."""
        result = classify_snow_quality(12.1, -5.1, "stable")
        assert result == SnowQuality.POWDER

    def test_good_conditions(self):
        """Moderate SLR and freezing temp gives good."""
        result = classify_snow_quality(10.0, -3.0, "stable")
        assert result == SnowQuality.GOOD

    def test_good_at_high_slr_warm_temp(self):
        """High SLR but not cold enough is good, not powder."""
        result = classify_snow_quality(15.0, -3.0, "stable")
        assert result == SnowQuality.GOOD

    def test_good_at_low_slr_cold_temp(self):
        """Low SLR but cold enough is good if SLR >= 8."""
        result = classify_snow_quality(9.0, -10.0, "stable")
        assert result == SnowQuality.GOOD

    def test_wet_heavy_low_slr(self):
        """Low SLR gives wet/heavy regardless of temp."""
        result = classify_snow_quality(6.0, -10.0, "stable")
        assert result == SnowQuality.WET_HEAVY

    def test_wet_heavy_above_freezing(self):
        """Above freezing gives wet/heavy regardless of SLR."""
        result = classify_snow_quality(15.0, 2.0, "stable")
        assert result == SnowQuality.WET_HEAVY

    def test_wet_heavy_both_conditions(self):
        """Both low SLR and warm temp is wet/heavy."""
        result = classify_snow_quality(5.0, 3.0, "warming")
        assert result == SnowQuality.WET_HEAVY

    def test_icy_warming_near_zero(self):
        """Warming trend near freezing gives icy."""
        result = classify_snow_quality(10.0, 0.0, "warming")
        assert result == SnowQuality.ICY

    def test_icy_warming_just_below_zero(self):
        """Warming at -1C gives icy."""
        result = classify_snow_quality(10.0, -1.0, "warming")
        assert result == SnowQuality.ICY

    def test_icy_warming_at_upper_bound(self):
        """Warming at 2C gives icy."""
        result = classify_snow_quality(10.0, 2.0, "warming")
        assert result == SnowQuality.ICY

    def test_icy_warming_at_lower_bound(self):
        """Warming at -2C gives icy."""
        result = classify_snow_quality(10.0, -2.0, "warming")
        assert result == SnowQuality.ICY

    def test_not_icy_when_cooling(self):
        """Cooling near zero is not icy, just good."""
        result = classify_snow_quality(10.0, 0.0, "cooling")
        assert result == SnowQuality.GOOD

    def test_not_icy_when_stable(self):
        """Stable near zero is good, not icy."""
        result = classify_snow_quality(10.0, -1.0, "stable")
        assert result == SnowQuality.GOOD

    def test_not_icy_when_too_warm(self):
        """Warming above 2C is wet, not icy."""
        result = classify_snow_quality(10.0, 3.0, "warming")
        assert result == SnowQuality.WET_HEAVY

    def test_not_icy_when_too_cold(self):
        """Warming below -2C is good, not icy."""
        result = classify_snow_quality(10.0, -5.0, "warming")
        assert result == SnowQuality.GOOD

    def test_boundary_slr_8_is_good(self):
        """SLR exactly 8 is good (not wet)."""
        result = classify_snow_quality(8.0, -3.0, "stable")
        assert result == SnowQuality.GOOD

    def test_boundary_slr_just_below_8_is_wet(self):
        """SLR just below 8 is wet."""
        result = classify_snow_quality(7.9, -3.0, "stable")
        assert result == SnowQuality.WET_HEAVY

    def test_boundary_temp_0_is_good(self):
        """Temp exactly 0 is good (not wet)."""
        result = classify_snow_quality(10.0, 0.0, "stable")
        assert result == SnowQuality.GOOD


class TestGetQualityBadge:
    """Tests for quality badge display properties."""

    def test_powder_badge(self):
        """Powder badge has correct properties."""
        emoji, label, color = get_quality_badge(SnowQuality.POWDER)
        assert label == "Fresh Powder"
        assert color == "#60a5fa"  # Blue

    def test_good_badge(self):
        """Good badge has correct properties."""
        emoji, label, color = get_quality_badge(SnowQuality.GOOD)
        assert label == "Good Snow"
        assert color == "#22c55e"  # Green

    def test_wet_heavy_badge(self):
        """Wet/Heavy badge has correct properties."""
        emoji, label, color = get_quality_badge(SnowQuality.WET_HEAVY)
        assert label == "Wet/Heavy"
        assert color == "#f59e0b"  # Yellow/Orange

    def test_icy_badge(self):
        """Icy badge has correct properties."""
        emoji, label, color = get_quality_badge(SnowQuality.ICY)
        assert label == "Icy"
        assert color == "#94a3b8"  # Gray

    def test_all_badges_return_three_elements(self):
        """All quality values return 3-tuple."""
        for quality in SnowQuality:
            result = get_quality_badge(quality)
            assert len(result) == 3
            assert isinstance(result[0], str)  # emoji
            assert isinstance(result[1], str)  # label
            assert isinstance(result[2], str)  # color


class TestGetSLRDescription:
    """Tests for SLR description strings."""

    def test_very_light_powder(self):
        """SLR >= 15 is very light powder."""
        assert "very light" in get_slr_description(15.0).lower()
        assert "very light" in get_slr_description(20.0).lower()

    def test_light_powder(self):
        """SLR 12-15 is light powder."""
        desc = get_slr_description(12.0)
        assert "light" in desc.lower()
        assert "very" not in desc.lower()

    def test_average_density(self):
        """SLR 10-12 is average density."""
        assert "average" in get_slr_description(10.0).lower()
        assert "average" in get_slr_description(11.9).lower()

    def test_slightly_dense(self):
        """SLR 8-10 is slightly dense."""
        assert "dense" in get_slr_description(8.0).lower()
        assert "dense" in get_slr_description(9.9).lower()

    def test_heavy_wet(self):
        """SLR < 8 is heavy/wet snow."""
        desc = get_slr_description(5.0)
        assert "heavy" in desc.lower() or "wet" in desc.lower()


class TestGetQualityExplanation:
    """Tests for quality explanation tooltips."""

    def test_all_qualities_have_explanations(self):
        """Every quality value has an explanation."""
        for quality in SnowQuality:
            explanation = get_quality_explanation(quality)
            assert len(explanation) > 20  # Meaningful explanation
            assert isinstance(explanation, str)

    def test_powder_explanation_mentions_light(self):
        """Powder explanation mentions light snow."""
        explanation = get_quality_explanation(SnowQuality.POWDER)
        assert "light" in explanation.lower() or "fluffy" in explanation.lower()

    def test_icy_explanation_mentions_freeze(self):
        """Icy explanation mentions freeze-thaw."""
        explanation = get_quality_explanation(SnowQuality.ICY)
        assert "freeze" in explanation.lower()

    def test_wet_explanation_mentions_water(self):
        """Wet/heavy explanation mentions water content."""
        explanation = get_quality_explanation(SnowQuality.WET_HEAVY)
        assert "water" in explanation.lower() or "wet" in explanation.lower()


class TestCreateQualityMetrics:
    """Tests for the metrics factory function."""

    def test_creates_metrics_with_all_fields(self):
        """create_quality_metrics returns complete QualityMetrics."""
        # Use temps with >2 degree change to get "warming" trend
        metrics = create_quality_metrics(15.0, 10.0, -8.0, [-12, -10, -8])
        assert isinstance(metrics, QualityMetrics)
        assert metrics.slr == 15.0
        assert metrics.temp_c == -8.0
        assert metrics.temp_trend == "warming"  # -12 to -8 = +4 degrees
        assert metrics.quality == SnowQuality.POWDER

    def test_without_hourly_temps(self):
        """Works without hourly temperatures."""
        metrics = create_quality_metrics(10.0, 10.0, -3.0)
        assert metrics.temp_trend == "stable"  # Default when no temps

    def test_empty_hourly_temps(self):
        """Works with empty hourly temps list."""
        metrics = create_quality_metrics(10.0, 10.0, -3.0, [])
        assert metrics.temp_trend == "stable"

    def test_powder_scenario(self):
        """Typical powder scenario."""
        metrics = create_quality_metrics(
            snow_depth_change=20.0,
            precip_mm=10.0,
            temp_c=-10.0,
            hourly_temps=[-12, -11, -10]
        )
        assert metrics.quality == SnowQuality.POWDER
        assert metrics.slr == 20.0

    def test_wet_snow_scenario(self):
        """Typical wet snow scenario."""
        metrics = create_quality_metrics(
            snow_depth_change=5.0,
            precip_mm=10.0,
            temp_c=2.0,
            hourly_temps=[0, 1, 2]
        )
        assert metrics.quality == SnowQuality.WET_HEAVY
        assert metrics.slr == 5.0

    def test_icy_scenario(self):
        """Typical icy conditions scenario."""
        metrics = create_quality_metrics(
            snow_depth_change=10.0,
            precip_mm=10.0,
            temp_c=0.0,
            hourly_temps=[-5, -3, -1, 0]
        )
        assert metrics.quality == SnowQuality.ICY


class TestQualityMetricsDataclass:
    """Tests for QualityMetrics dataclass."""

    def test_dataclass_creation(self):
        """Can create QualityMetrics directly."""
        metrics = QualityMetrics(
            slr=12.0,
            temp_c=-5.0,
            temp_trend="cooling",
            quality=SnowQuality.GOOD
        )
        assert metrics.slr == 12.0
        assert metrics.temp_c == -5.0
        assert metrics.temp_trend == "cooling"
        assert metrics.quality == SnowQuality.GOOD

    def test_dataclass_equality(self):
        """QualityMetrics instances compare by value."""
        m1 = QualityMetrics(10.0, -3.0, "stable", SnowQuality.GOOD)
        m2 = QualityMetrics(10.0, -3.0, "stable", SnowQuality.GOOD)
        assert m1 == m2


class TestSnowQualityEnum:
    """Tests for SnowQuality enum."""

    def test_all_values_exist(self):
        """All expected quality values exist."""
        assert SnowQuality.POWDER.value == "powder"
        assert SnowQuality.GOOD.value == "good"
        assert SnowQuality.WET_HEAVY.value == "wet"
        assert SnowQuality.ICY.value == "icy"

    def test_four_quality_levels(self):
        """There are exactly 4 quality levels."""
        assert len(SnowQuality) == 4


class TestIntegrationScenarios:
    """Integration tests for realistic skiing scenarios."""

    def test_powder_day_scenario(self):
        """Classic powder day: cold, light snow."""
        # Overnight storm dropped 30cm from 15mm of water
        slr = calculate_slr(30.0, 15.0)  # 20:1 ratio
        temp_trend = get_temp_trend([-15, -14, -12, -10])
        quality = classify_snow_quality(slr, -10.0, temp_trend)

        assert slr == 20.0
        assert temp_trend == "warming"  # Still warming but cold
        # At -10C with warming, not in icy range (-2 to 2)
        assert quality == SnowQuality.POWDER

    def test_spring_slush_scenario(self):
        """Spring afternoon: warm, wet snow."""
        slr = calculate_slr(5.0, 10.0)  # 5:1 ratio
        temp_trend = get_temp_trend([-2, 0, 3, 5])
        quality = classify_snow_quality(slr, 5.0, temp_trend)

        assert slr == 5.0
        assert temp_trend == "warming"
        assert quality == SnowQuality.WET_HEAVY

    def test_freeze_thaw_scenario(self):
        """Morning after warm day: icy conditions."""
        slr = calculate_slr(0.0, 0.0)  # No new snow
        temp_trend = get_temp_trend([-5, -3, -1, 0])
        quality = classify_snow_quality(slr, 0.0, temp_trend)

        assert temp_trend == "warming"
        assert quality == SnowQuality.ICY

    def test_good_groomer_scenario(self):
        """Typical groomed run: moderate conditions."""
        slr = calculate_slr(8.0, 8.0)  # 10:1 ratio
        temp_trend = get_temp_trend([-5, -5, -4, -5])
        quality = classify_snow_quality(slr, -5.0, temp_trend)

        assert temp_trend == "stable"
        assert quality == SnowQuality.GOOD

    def test_cold_but_heavy_snow_scenario(self):
        """Cold temps but dense snow (rain shadow effect)."""
        slr = calculate_slr(6.0, 10.0)  # 6:1 - dense
        temp_trend = get_temp_trend([-8, -9, -10, -11])
        quality = classify_snow_quality(slr, -11.0, temp_trend)

        assert temp_trend == "cooling"
        # Low SLR makes it wet/heavy despite cold
        assert quality == SnowQuality.WET_HEAVY


class TestMockedStreamlitRender:
    """Tests for Streamlit rendering functions using mocks."""

    def test_render_badge_calls_markdown(self):
        """render_snow_quality_badge calls st.markdown."""
        with patch.dict("sys.modules", {"streamlit": MagicMock()}):
            import sys
            mock_st = sys.modules["streamlit"]

            from snowforecast.dashboard.components.snow_quality import render_snow_quality_badge

            metrics = QualityMetrics(15.0, -8.0, "stable", SnowQuality.POWDER)
            render_snow_quality_badge(metrics)

            mock_st.markdown.assert_called_once()
            call_args = mock_st.markdown.call_args
            assert call_args[1]["unsafe_allow_html"] is True

    def test_render_badge_includes_color(self):
        """render_snow_quality_badge HTML includes quality color."""
        with patch.dict("sys.modules", {"streamlit": MagicMock()}):
            import sys
            mock_st = sys.modules["streamlit"]

            from snowforecast.dashboard.components.snow_quality import render_snow_quality_badge

            metrics = QualityMetrics(15.0, -8.0, "stable", SnowQuality.POWDER)
            render_snow_quality_badge(metrics)

            html = mock_st.markdown.call_args[0][0]
            assert "#60a5fa" in html  # Powder blue color

    def test_render_badge_uses_container(self):
        """render_snow_quality_badge uses provided container."""
        from snowforecast.dashboard.components.snow_quality import render_snow_quality_badge

        mock_container = MagicMock()
        metrics = QualityMetrics(10.0, -3.0, "stable", SnowQuality.GOOD)

        render_snow_quality_badge(metrics, container=mock_container)

        mock_container.markdown.assert_called_once()

    def test_render_details_shows_slr(self):
        """render_snow_quality_details shows SLR value."""
        from snowforecast.dashboard.components.snow_quality import render_snow_quality_details

        # Create mock container to avoid streamlit import issues
        mock_container = MagicMock()
        mock_expander = MagicMock()
        mock_container.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_container.expander.return_value.__exit__ = MagicMock(return_value=False)

        metrics = QualityMetrics(15.0, -8.0, "cooling", SnowQuality.POWDER)
        render_snow_quality_details(metrics, container=mock_container)

        # Check markdown calls contain SLR
        markdown_calls = [call[0][0] for call in mock_container.markdown.call_args_list]
        slr_found = any("15.0:1" in str(call) for call in markdown_calls)
        assert slr_found

    def test_render_compact_single_line(self):
        """render_snow_quality_compact produces single markdown call."""
        from snowforecast.dashboard.components.snow_quality import render_snow_quality_compact

        mock_container = MagicMock()
        metrics = QualityMetrics(12.0, -5.0, "stable", SnowQuality.GOOD)
        render_snow_quality_compact(metrics, container=mock_container)

        # Should be exactly one markdown call
        mock_container.markdown.assert_called_once()
