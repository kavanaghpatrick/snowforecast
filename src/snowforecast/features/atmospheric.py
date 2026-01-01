"""Atmospheric feature engineering for snow prediction.

This module computes derived atmospheric features from ERA5-Land and HRRR data.
Features are designed to be physically meaningful for snow prediction.
"""

import numpy as np
import pandas as pd


class AtmosphericFeatures:
    """Engineer atmospheric features for snow prediction.

    Takes raw meteorological variables (temperature, humidity, wind, pressure,
    precipitation) and computes derived features useful for snow prediction.

    Expected input columns (from ERA5-Land):
        - t2m: 2m temperature (Kelvin)
        - d2m: 2m dewpoint temperature (Kelvin)
        - u10: 10m U wind component (m/s)
        - v10: 10m V wind component (m/s)
        - sp: Surface pressure (Pa)
        - tp: Total precipitation (m)
        - sd: Snow depth (m of water equivalent)
        - sf: Snowfall (m of water equivalent)
    """

    # Physical constants
    KELVIN_OFFSET = 273.15  # K to C conversion
    PA_TO_HPA = 100.0       # Pa to hPa conversion
    M_TO_MM = 1000.0        # m to mm conversion

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all atmospheric features.

        Args:
            df: DataFrame with raw meteorological variables

        Returns:
            DataFrame with original columns plus all computed features
        """
        result = df.copy()

        # Compute each feature set
        result = self.compute_temperature_features(result)
        result = self.compute_humidity_features(result)
        result = self.compute_wind_features(result)
        result = self.compute_pressure_features(result)
        result = self.compute_precipitation_features(result)

        return result

    def compute_temperature_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute temperature-derived features.

        Features:
            - t2m_celsius: Temperature in Celsius (from Kelvin)
            - freezing_level: 1 if t2m < 273.15K (below freezing), else 0

        Args:
            df: DataFrame with t2m column (2m temperature in Kelvin)

        Returns:
            DataFrame with temperature features added
        """
        result = df.copy()

        if "t2m" not in result.columns:
            return result

        # Convert temperature to Celsius
        result["t2m_celsius"] = result["t2m"] - self.KELVIN_OFFSET

        # Freezing level indicator (1 = below freezing, 0 = above)
        result["freezing_level"] = (result["t2m"] < self.KELVIN_OFFSET).astype(int)

        return result

    def compute_humidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute humidity-derived features.

        Features:
            - relative_humidity: RH computed from t2m and d2m using Magnus formula
            - dewpoint_depression: Difference between t2m and d2m (Celsius)
            - wet_bulb_temp: Approximate wet bulb temperature (Celsius)

        Args:
            df: DataFrame with t2m and d2m columns (in Kelvin)

        Returns:
            DataFrame with humidity features added
        """
        result = df.copy()

        if "t2m" not in result.columns or "d2m" not in result.columns:
            return result

        # Convert to Celsius for calculations
        t_c = result["t2m"] - self.KELVIN_OFFSET
        td_c = result["d2m"] - self.KELVIN_OFFSET

        # Relative humidity using Magnus formula approximation
        # RH = 100 * exp((17.625 * Td) / (243.04 + Td)) / exp((17.625 * T) / (243.04 + T))
        result["relative_humidity"] = self._calc_relative_humidity(t_c, td_c)

        # Dewpoint depression (how far T is above dewpoint)
        result["dewpoint_depression"] = t_c - td_c

        # Wet bulb temperature (Stull 2011 approximation)
        result["wet_bulb_temp"] = self._calc_wet_bulb_temp(t_c, result["relative_humidity"])

        return result

    def _calc_relative_humidity(self, t_celsius: pd.Series, td_celsius: pd.Series) -> pd.Series:
        """Calculate relative humidity using Magnus formula.

        Args:
            t_celsius: Temperature in Celsius
            td_celsius: Dewpoint temperature in Celsius

        Returns:
            Relative humidity (0-100%)
        """
        # Magnus formula constants
        a = 17.625
        b = 243.04

        # Saturation vapor pressure ratio
        numerator = np.exp((a * td_celsius) / (b + td_celsius))
        denominator = np.exp((a * t_celsius) / (b + t_celsius))

        rh = 100.0 * numerator / denominator

        # Clip to valid range (can exceed due to measurement errors)
        return rh.clip(0, 100)

    def _calc_wet_bulb_temp(self, t_celsius: pd.Series, rh: pd.Series) -> pd.Series:
        """Calculate wet bulb temperature using Stull (2011) formula.

        Accurate to within 0.3C for typical atmospheric conditions.

        Args:
            t_celsius: Temperature in Celsius
            rh: Relative humidity (0-100%)

        Returns:
            Wet bulb temperature in Celsius
        """
        # Stull (2011) empirical formula
        # Tw = T * atan(0.151977 * sqrt(RH + 8.313659))
        #    + atan(T + RH) - atan(RH - 1.676331)
        #    + 0.00391838 * RH^1.5 * atan(0.023101 * RH) - 4.686035

        term1 = t_celsius * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        term2 = np.arctan(t_celsius + rh)
        term3 = -np.arctan(rh - 1.676331)
        term4 = 0.00391838 * np.power(rh, 1.5) * np.arctan(0.023101 * rh)
        term5 = -4.686035

        return term1 + term2 + term3 + term4 + term5

    def compute_wind_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute wind-derived features.

        Features:
            - wind_speed: Wind speed from U/V components (m/s)
            - wind_direction: Meteorological wind direction (0=N, 90=E, 180=S, 270=W)
            - wind_chill: Wind chill temperature (Celsius)

        Args:
            df: DataFrame with u10 and v10 columns (wind components in m/s)

        Returns:
            DataFrame with wind features added
        """
        result = df.copy()

        if "u10" not in result.columns or "v10" not in result.columns:
            return result

        # Wind speed: sqrt(u^2 + v^2)
        result["wind_speed"] = np.sqrt(result["u10"]**2 + result["v10"]**2)

        # Wind direction (meteorological convention: direction wind is FROM)
        # 0 = North, 90 = East, 180 = South, 270 = West
        result["wind_direction"] = self._calc_wind_direction(result["u10"], result["v10"])

        # Wind chill (requires temperature)
        if "t2m" in result.columns:
            t_c = result["t2m"] - self.KELVIN_OFFSET
            result["wind_chill"] = self._calc_wind_chill(t_c, result["wind_speed"])

        return result

    def _calc_wind_direction(self, u: pd.Series, v: pd.Series) -> pd.Series:
        """Calculate meteorological wind direction from U/V components.

        Wind direction is the direction the wind is coming FROM.
        0/360 = North, 90 = East, 180 = South, 270 = West.

        Args:
            u: U (east-west) wind component (positive = eastward)
            v: V (north-south) wind component (positive = northward)

        Returns:
            Wind direction in degrees (0-360)
        """
        # Mathematical angle (counterclockwise from east)
        # Then convert to meteorological (clockwise from north, direction FROM)
        wind_dir = (270 - np.degrees(np.arctan2(v, u))) % 360

        # Handle calm winds (set direction to 0)
        wind_speed = np.sqrt(u**2 + v**2)
        wind_dir = wind_dir.where(wind_speed > 0.1, 0)

        return wind_dir

    def _calc_wind_chill(self, t_celsius: pd.Series, wind_speed_ms: pd.Series) -> pd.Series:
        """Calculate wind chill temperature using North American formula.

        Valid for T <= 10C and wind speed >= 1.34 m/s (4.8 km/h).
        Returns actual temperature when conditions don't apply.

        Args:
            t_celsius: Temperature in Celsius
            wind_speed_ms: Wind speed in m/s

        Returns:
            Wind chill temperature in Celsius
        """
        # Convert wind speed to km/h for the formula
        wind_kmh = wind_speed_ms * 3.6

        # North American wind chill formula
        # Tc = 13.12 + 0.6215*T - 11.37*V^0.16 + 0.3965*T*V^0.16
        wind_chill = (
            13.12 +
            0.6215 * t_celsius -
            11.37 * np.power(wind_kmh, 0.16) +
            0.3965 * t_celsius * np.power(wind_kmh, 0.16)
        )

        # Formula only valid for T <= 10C and V >= 4.8 km/h
        # Return actual temperature otherwise
        applies = (t_celsius <= 10) & (wind_kmh >= 4.8)

        return wind_chill.where(applies, t_celsius)

    def compute_pressure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute pressure-derived features.

        Features:
            - pressure_hpa: Surface pressure in hPa (from Pa)
            - pressure_tendency: 24h pressure change (hPa, if time series)

        Args:
            df: DataFrame with sp column (surface pressure in Pa)

        Returns:
            DataFrame with pressure features added
        """
        result = df.copy()

        if "sp" not in result.columns:
            return result

        # Convert pressure to hPa
        result["pressure_hpa"] = result["sp"] / self.PA_TO_HPA

        # Pressure tendency (24h change) if we have time-ordered data
        # This requires knowing the time resolution and having sorted data
        # For now, compute simple difference which can be adjusted based on actual time steps
        if "time" in result.columns or result.index.name == "time":
            # Shift by 24 rows for hourly data to get 24h tendency
            # Adjust shift based on actual time resolution if different
            result["pressure_tendency"] = result["pressure_hpa"].diff(periods=24)

        return result

    def compute_precipitation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute precipitation-derived features.

        Features:
            - precip_mm: Total precipitation in mm (from m)
            - snow_water_equiv_mm: Snow water equivalent in mm (from m)
            - snow_fraction: Fraction of precipitation that fell as snow (0-1)

        Args:
            df: DataFrame with tp (total precip), sf (snowfall), sd (snow depth)
                all in meters of water equivalent

        Returns:
            DataFrame with precipitation features added
        """
        result = df.copy()

        # Total precipitation in mm
        if "tp" in result.columns:
            result["precip_mm"] = result["tp"] * self.M_TO_MM

        # Snowfall in mm water equivalent
        if "sf" in result.columns:
            result["snow_water_equiv_mm"] = result["sf"] * self.M_TO_MM

        # Snow fraction (fraction that fell as snow vs rain)
        if "tp" in result.columns and "sf" in result.columns:
            # Avoid division by zero
            result["snow_fraction"] = np.where(
                result["tp"] > 0,
                result["sf"] / result["tp"],
                0.0
            )
            # Clip to valid range (can exceed 1 due to accumulation differences)
            result["snow_fraction"] = result["snow_fraction"].clip(0, 1)

        # Snow depth in mm (for reference, not SWE)
        if "sd" in result.columns:
            result["snow_depth_mm"] = result["sd"] * self.M_TO_MM

        return result

    def get_feature_names(self) -> list[str]:
        """Get list of all feature names this class can compute.

        Returns:
            List of feature column names
        """
        return [
            # Temperature features
            "t2m_celsius",
            "freezing_level",
            # Humidity features
            "relative_humidity",
            "dewpoint_depression",
            "wet_bulb_temp",
            # Wind features
            "wind_speed",
            "wind_direction",
            "wind_chill",
            # Pressure features
            "pressure_hpa",
            "pressure_tendency",
            # Precipitation features
            "precip_mm",
            "snow_water_equiv_mm",
            "snow_fraction",
            "snow_depth_mm",
        ]
