"""Tests for color scale functions."""

from snowforecast.visualization.colors import (
    ELEVATION_SCALE,
    SNOW_DEPTH_SCALE,
    elevation_category,
    elevation_to_rgb,
    hex_to_rgb,
    rgb_to_hex,
    snow_depth_category,
    snow_depth_to_hex,
    snow_depth_to_rgb,
)


class TestSnowDepthColors:
    """Tests for snow depth color functions."""

    def test_snow_depth_to_hex_trace(self):
        """Trace snow (<10cm) should be very light blue."""
        assert snow_depth_to_hex(0) == "#E6F3FF"
        assert snow_depth_to_hex(5) == "#E6F3FF"
        assert snow_depth_to_hex(9.9) == "#E6F3FF"

    def test_snow_depth_to_hex_light(self):
        """Light snow (10-30cm) should be light blue."""
        assert snow_depth_to_hex(10) == "#ADD8E6"
        assert snow_depth_to_hex(20) == "#ADD8E6"
        assert snow_depth_to_hex(29.9) == "#ADD8E6"

    def test_snow_depth_to_hex_moderate(self):
        """Moderate snow (30-60cm) should be cornflower."""
        assert snow_depth_to_hex(30) == "#6495ED"
        assert snow_depth_to_hex(45) == "#6495ED"

    def test_snow_depth_to_hex_heavy(self):
        """Heavy snow (60-100cm) should be royal blue."""
        assert snow_depth_to_hex(60) == "#4169E1"
        assert snow_depth_to_hex(80) == "#4169E1"

    def test_snow_depth_to_hex_very_heavy(self):
        """Very heavy snow (100-150cm) should be medium blue."""
        assert snow_depth_to_hex(100) == "#0000CD"
        assert snow_depth_to_hex(125) == "#0000CD"

    def test_snow_depth_to_hex_extreme(self):
        """Extreme snow (>150cm) should be purple."""
        assert snow_depth_to_hex(150) == "#8A2BE2"
        assert snow_depth_to_hex(200) == "#8A2BE2"
        assert snow_depth_to_hex(500) == "#8A2BE2"

    def test_snow_depth_to_hex_negative(self):
        """Negative depth should be treated as zero."""
        assert snow_depth_to_hex(-10) == "#E6F3FF"

    def test_snow_depth_to_rgb_format(self):
        """RGB output should be [R, G, B, A] list."""
        result = snow_depth_to_rgb(50)
        assert isinstance(result, list)
        assert len(result) == 4
        assert all(isinstance(v, int) for v in result)
        assert all(0 <= v <= 255 for v in result)

    def test_snow_depth_to_rgb_alpha(self):
        """Alpha should be customizable."""
        result_default = snow_depth_to_rgb(50)
        assert result_default[3] == 200

        result_custom = snow_depth_to_rgb(50, alpha=128)
        assert result_custom[3] == 128

    def test_snow_depth_to_rgb_values(self):
        """RGB values should match hex conversion."""
        # Cornflower blue #6495ED = RGB(100, 149, 237)
        result = snow_depth_to_rgb(50)
        assert result[0] == 100  # R
        assert result[1] == 149  # G
        assert result[2] == 237  # B

    def test_snow_depth_category(self):
        """Category names should match thresholds."""
        assert snow_depth_category(5) == "Trace"
        assert snow_depth_category(15) == "Light"
        assert snow_depth_category(45) == "Moderate"
        assert snow_depth_category(80) == "Heavy"
        assert snow_depth_category(120) == "Very Heavy"
        assert snow_depth_category(200) == "Extreme"


class TestElevationColors:
    """Tests for elevation color functions."""

    def test_elevation_to_rgb_valley(self):
        """Low elevation should be forest green."""
        result = elevation_to_rgb(1000)
        assert result[:3] == [34, 139, 34]

    def test_elevation_to_rgb_foothills(self):
        """Foothills should be sage green."""
        result = elevation_to_rgb(1750)
        assert result[:3] == [143, 188, 143]

    def test_elevation_to_rgb_mid_mountain(self):
        """Mid-mountain should be tan."""
        result = elevation_to_rgb(2250)
        assert result[:3] == [210, 180, 140]

    def test_elevation_to_rgb_high_mountain(self):
        """High mountain should be brown."""
        result = elevation_to_rgb(2750)
        assert result[:3] == [139, 119, 101]

    def test_elevation_to_rgb_alpine(self):
        """Alpine should be gray."""
        result = elevation_to_rgb(3250)
        assert result[:3] == [169, 169, 169]

    def test_elevation_to_rgb_peak(self):
        """Peak should be white."""
        result = elevation_to_rgb(4000)
        assert result[:3] == [255, 255, 255]

    def test_elevation_to_rgb_alpha(self):
        """Alpha should be customizable."""
        result = elevation_to_rgb(2000, alpha=100)
        assert result[3] == 100

    def test_elevation_to_rgb_negative(self):
        """Negative elevation should be treated as zero."""
        result = elevation_to_rgb(-100)
        assert result[:3] == [34, 139, 34]

    def test_elevation_category(self):
        """Category names should match thresholds."""
        assert elevation_category(1000) == "Valley"
        assert elevation_category(1750) == "Foothills"
        assert elevation_category(2250) == "Mid-Mountain"
        assert elevation_category(2750) == "High Mountain"
        assert elevation_category(3250) == "Alpine"
        assert elevation_category(4000) == "Peak"


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_hex_to_rgb(self):
        """Hex to RGB conversion should work."""
        assert hex_to_rgb("#FF0000") == (255, 0, 0)
        assert hex_to_rgb("#00FF00") == (0, 255, 0)
        assert hex_to_rgb("#0000FF") == (0, 0, 255)
        assert hex_to_rgb("#6495ED") == (100, 149, 237)

    def test_hex_to_rgb_no_hash(self):
        """Should work with or without hash prefix."""
        assert hex_to_rgb("FF0000") == (255, 0, 0)

    def test_rgb_to_hex(self):
        """RGB to hex conversion should work."""
        assert rgb_to_hex(255, 0, 0) == "#ff0000"
        assert rgb_to_hex(0, 255, 0) == "#00ff00"
        assert rgb_to_hex(100, 149, 237) == "#6495ed"

    def test_roundtrip_conversion(self):
        """Hex -> RGB -> Hex should be idempotent."""
        original = "#6495ed"
        rgb = hex_to_rgb(original)
        result = rgb_to_hex(*rgb)
        assert result.lower() == original.lower()


class TestScaleConsistency:
    """Tests for scale consistency."""

    def test_snow_scale_ordered(self):
        """Snow depth thresholds should be in ascending order."""
        thresholds = [t for t, _, _ in SNOW_DEPTH_SCALE]
        assert thresholds == sorted(thresholds)

    def test_elevation_scale_ordered(self):
        """Elevation thresholds should be in ascending order."""
        thresholds = [t for t, _, _ in ELEVATION_SCALE]
        assert thresholds == sorted(thresholds)

    def test_snow_scale_complete(self):
        """Snow scale should have 6 categories."""
        assert len(SNOW_DEPTH_SCALE) == 6

    def test_elevation_scale_complete(self):
        """Elevation scale should have 6 categories."""
        assert len(ELEVATION_SCALE) == 6
