"""Data ingestion pipelines for snowforecast.

Each pipeline is responsible for:
1. Downloading raw data from its source
2. Processing data into standardized format
3. Validating data quality

Pipelines:
- snotel: SNOTEL snow observation network (primary ground truth)
- ghcn: GHCN-Daily climate observations (supplemental ground truth)
- era5: ERA5-Land reanalysis data (historical weather)
- hrrr: HRRR high-resolution weather model (fine-grained weather)
- dem: Copernicus DEM terrain data (elevation, slope, aspect)
- openskimap: Ski resort locations (forecast targets)
"""

# Pipelines will be imported here after implementation
# from .snotel import SnotelPipeline
# from .ghcn import GHCNPipeline
# from .era5 import ERA5Pipeline
# from .hrrr import HRRRPipeline
# from .dem import DEMPipeline
# from .openskimap import OpenSkiMapPipeline
