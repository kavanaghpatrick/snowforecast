# Model Card: Snow Forecast Model

## Model Details

### Description
Machine learning model for predicting 24-hour snowfall and snow depth at mountain locations in the Western United States.

### Version
- Current: v1.0
- Release Date: 2026-01-01

### Model Type
Ensemble of:
- Gradient Boosting (LightGBM/XGBoost)
- LSTM/GRU for temporal patterns
- U-Net for spatial downscaling

### Intended Use
- Mountain weather forecasting
- Ski resort snow predictions
- Backcountry avalanche assessment support
- Research and educational purposes

### Out-of-Scope Use
- Not intended for aviation weather decisions
- Not a replacement for official NWS forecasts
- Should not be used for life-safety decisions without professional verification

## Training Data

### Sources
- **SNOTEL**: USDA Natural Resources Conservation Service snow telemetry network (~800 stations)
- **GHCN**: NOAA Global Historical Climatology Network daily summaries
- **ERA5-Land**: ECMWF reanalysis at 9km resolution
- **HRRR**: NOAA High-Resolution Rapid Refresh model at 3km resolution
- **USGS DEM**: 30-meter digital elevation model

### Geographic Coverage
Western United States:
- Latitude: 31N to 49N
- Longitude: 125W to 102W
- States: WA, OR, CA, NV, ID, MT, WY, UT, CO, AZ, NM

### Temporal Coverage
- Training: 2010-2022 (12 years)
- Validation: 2022-2024 (2 years)
- Holdout: 2024-2026 (2 years)

### Data Volume
- ~500 station locations
- ~4,000 daily observations per station
- ~2M total training samples

## Features

### Atmospheric Features
- Temperature (2m)
- Precipitation (total, frozen)
- Wind speed and direction
- Humidity
- Pressure
- Cloud cover

### Terrain Features
- Elevation
- Slope
- Aspect
- Curvature
- Distance to peaks/valleys

### Temporal Features
- Day of year (cyclical)
- Hour of day (cyclical)
- Lagged observations (1-7 days)
- Rolling statistics (3, 7, 14 day windows)

## Performance

### Success Metrics (PRD Targets)
| Metric | Target | Achieved |
|--------|--------|----------|
| Snow Depth RMSE | <15 cm | TBD |
| Snowfall F1-score | >85% | TBD |
| Improvement over baseline | >20% | TBD |
| Bias | <5% | TBD |

### Performance by Elevation
| Elevation Band | RMSE (cm) | F1-score |
|----------------|-----------|----------|
| 0-2000m | TBD | TBD |
| 2000-2500m | TBD | TBD |
| 2500-3000m | TBD | TBD |
| 3000-3500m | TBD | TBD |
| 3500m+ | TBD | TBD |

### Seasonal Performance
| Season | RMSE (cm) | Bias (cm) |
|--------|-----------|-----------|
| Winter (DJF) | TBD | TBD |
| Spring (MAM) | TBD | TBD |
| Fall (SON) | TBD | TBD |

## Limitations

### Known Limitations
1. **Sparse station coverage**: Some mountain regions have limited ground truth data
2. **Extreme events**: Model may underpredict rare high-snowfall events
3. **Rain-snow line**: Predictions less reliable near 0C boundary
4. **Lake effect**: Not specifically trained for lake-effect snow regions

### Failure Modes
- Rapid temperature changes near freezing
- Unusual atmospheric patterns (atmospheric rivers)
- Very high elevation (>4000m) with limited training data

## Ethical Considerations

### Potential Biases
- **Geographic**: Better performance in Colorado/Utah where station density is higher
- **Seasonal**: Trained on winter data; spring/fall transitions may be less accurate
- **Elevation**: Most training data from 2000-3500m range

### Fairness
- Model treats all geographic locations equally within bounds
- No demographic or socioeconomic data used

### Environmental Impact
- Training: Estimated 10 GPU-hours on consumer hardware
- Inference: <100ms per prediction on CPU

## Reproducibility

### Requirements
See `pyproject.toml` for full dependency list:
- Python 3.11+
- PyTorch 2.0+
- LightGBM 4.0+
- FastAPI 0.100+

### Random Seeds
```python
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
```

### Training Command
```bash
python -m snowforecast.train --config configs/default.yaml
```

## Contact

- **Repository**: https://github.com/kavanaghpatrick/snowforecast
- **Issues**: https://github.com/kavanaghpatrick/snowforecast/issues
- **Author**: Patrick Kavanagh

## Citation

```bibtex
@software{snowforecast2026,
  title = {Snow Forecast: ML Model for Mountain Snowfall Prediction},
  author = {Kavanagh, Patrick},
  year = {2026},
  url = {https://github.com/kavanaghpatrick/snowforecast}
}
```
