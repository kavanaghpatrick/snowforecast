# Snowforecast

ML-powered snowfall predictions for Western US ski resorts.

[![CI](https://github.com/kavanaghpatrick/snowforecast/actions/workflows/ci.yml/badge.svg)](https://github.com/kavanaghpatrick/snowforecast/actions/workflows/ci.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://snowforecast.streamlit.app)

## Features

- 7-day snow forecasts for 100+ Western US ski resorts
- Interactive map with resort markers and forecast overlay
- 3D terrain visualization with elevation bands
- SNOTEL snowpack integration
- Snow quality indicators (powder vs wet snow)
- Favorites with local storage
- Confidence visualization with uncertainty bands

## Quick Start

```bash
# Install
pip install -e ".[all]"

# Run dashboard locally
streamlit run src/snowforecast/dashboard/app.py

# Run API locally
uvicorn src.snowforecast.api.app:app --reload
```

## Project Structure

```
snowforecast/
├── src/snowforecast/
│   ├── pipelines/     # Data ingestion (SNOTEL, GHCN, ERA5, HRRR, DEM, OpenSkiMap)
│   ├── features/      # Feature engineering (spatial, temporal, terrain, atmospheric)
│   ├── models/        # ML models (Linear, GradientBoosting, LSTM, U-Net, Ensemble)
│   ├── cache/         # DuckDB cache with background refresh
│   ├── api/           # FastAPI backend
│   └── dashboard/     # Streamlit dashboard with 14 components
└── tests/             # 1,626 tests
```

## Data Sources

| Source | Type | Resolution | Purpose |
|--------|------|------------|---------|
| SNOTEL | Point | Hourly | Ground truth snow depth |
| GHCN | Point | Daily | Historical temperature/precip |
| ERA5-Land | Grid | 9km | Long-term weather history |
| HRRR | Grid | 3km | High-resolution forecasts |
| Copernicus DEM | Grid | 30m | Terrain features |
| OpenSkiMap | Point | - | Resort locations |

## Development

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/snowforecast --cov-report=html

# Lint
ruff check src/ tests/
```

## Deployment

### Streamlit Cloud (Dashboard)

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub and select this repo
4. Set main file path: `src/snowforecast/dashboard/app.py`
5. Deploy!

### GitHub Actions (CI)

CI runs automatically on push/PR:
- Linting with ruff
- Parallel test groups (pipelines, features, models, cache, dashboard, api)
- Coverage reporting on main

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  Streamlit      │────▶│  CachedPredictor│
│  Dashboard      │     │  (DuckDB)       │
└─────────────────┘     └────────┬────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
           ┌────────────────┐       ┌────────────────┐
           │  HRRR/NOAA     │       │  DEM/Terrain   │
           │  Forecasts     │       │  Features      │
           └────────────────┘       └────────────────┘
```

## License

MIT
