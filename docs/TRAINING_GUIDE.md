# Training Guide

Step-by-step guide to reproduce the Snow Forecast model.

## Prerequisites

### System Requirements
- Python 3.11+
- 16GB RAM minimum (32GB recommended)
- 50GB disk space for data
- NVIDIA GPU with 8GB+ VRAM (optional, for deep learning)

### Environment Setup

```bash
# Clone repository
git clone https://github.com/kavanaghpatrick/snowforecast.git
cd snowforecast

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -e ".[all]"
```

## Step 1: Data Acquisition

### 1.1 SNOTEL Data

```python
from snowforecast.pipelines import SnotelPipeline

pipeline = SnotelPipeline()
stations = pipeline.get_station_metadata()
data = pipeline.download("2010-01-01", "2024-12-31")
processed = pipeline.process(data)
```

### 1.2 GHCN Data

```python
from snowforecast.pipelines import GHCNPipeline

pipeline = GHCNPipeline()
stations = pipeline.get_western_us_stations()
data = pipeline.download(stations, "2010-01-01", "2024-12-31")
```

### 1.3 ERA5 Data

```python
from snowforecast.pipelines import ERA5Pipeline

pipeline = ERA5Pipeline()
data = pipeline.download("2010-01-01", "2024-12-31")
```

### 1.4 DEM Data

```python
from snowforecast.pipelines import DEMPipeline

pipeline = DEMPipeline()
dem = pipeline.download()
```

## Step 2: Feature Engineering

### 2.1 Temporal Features

```python
from snowforecast.features import add_temporal_features

df = add_temporal_features(df, date_column="date")
```

### 2.2 Lagged Features

```python
from snowforecast.features import add_lagged_features

df = add_lagged_features(df, target_column="snow_depth_cm", lags=[1, 2, 3, 7])
```

### 2.3 Terrain Features

```python
from snowforecast.features import add_terrain_features

df = add_terrain_features(df, dem_path="data/processed/dem.tif")
```

### 2.4 Atmospheric Features

```python
from snowforecast.features import add_atmospheric_features

df = add_atmospheric_features(df, era5_path="data/processed/era5.nc")
```

## Step 3: Train-Test Split

Use temporal holdout to prevent data leakage:

```python
from snowforecast.models.cross_validation import TemporalSplit

splitter = TemporalSplit(test_years=2)
train_idx, test_idx = splitter.split(df)

X_train, X_test = df.loc[train_idx], df.loc[test_idx]
y_train, y_test = df.loc[train_idx, "target"], df.loc[test_idx, "target"]
```

## Step 4: Model Training

### 4.1 Gradient Boosting

```python
from snowforecast.models import GradientBoostingModel

model = GradientBoostingModel(
    framework="lightgbm",
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    random_state=42,
)
model.fit(X_train, y_train)
```

### 4.2 LSTM/GRU

```python
from snowforecast.models import SequenceModel

model = SequenceModel(
    model_type="lstm",
    hidden_size=128,
    num_layers=2,
    dropout=0.2,
)
model.fit(X_train, y_train, epochs=50)
```

### 4.3 U-Net (Spatial)

```python
from snowforecast.models import SpatialModel

model = SpatialModel(
    in_channels=10,
    out_channels=1,
    base_features=32,
)
model.fit(X_train, y_train)
```

## Step 5: Hyperparameter Tuning

```python
from snowforecast.models.tuning import HyperparameterTuner, TuningConfig

config = TuningConfig(
    n_trials=100,
    metric="rmse",
    direction="minimize",
)

tuner = HyperparameterTuner(config)
best_params = tuner.tune(model, X_train, y_train)
```

## Step 6: Ensemble

```python
from snowforecast.models import SimpleEnsemble

ensemble = SimpleEnsemble(
    models=[gb_model, lstm_model, unet_model],
    weights=[0.5, 0.3, 0.2],
)
ensemble.fit(X_train, y_train)
```

## Step 7: Evaluation

```python
from snowforecast.evaluation import HoldoutEvaluator, PRDMetrics

evaluator = HoldoutEvaluator(ensemble)
metrics = evaluator.evaluate(X_test, y_test)
print(metrics.summary())
```

## Step 8: Save Model

```python
ensemble.save("models/ensemble_v1.0.pkl")
```

## Configuration Files

### configs/default.yaml
```yaml
data:
  start_date: "2010-01-01"
  end_date: "2024-12-31"
  holdout_years: 2

features:
  temporal: true
  lagged: [1, 2, 3, 7]
  terrain: true
  atmospheric: true

model:
  type: ensemble
  components:
    - gradient_boosting
    - lstm
    - unet

training:
  random_seed: 42
  cross_validation:
    n_folds: 5
    method: station_kfold
```

## Troubleshooting

### Out of Memory
- Reduce batch size for deep learning models
- Use `n_jobs=1` for gradient boosting
- Process data in chunks

### Slow Training
- Enable GPU for PyTorch models: `device='cuda'`
- Use early stopping to prevent overfitting
- Reduce n_estimators during development

### Data Download Issues
- Check CDS API key for ERA5
- Verify network connectivity
- Use cached data when available

## References

- [SNOTEL Documentation](https://www.nrcs.usda.gov/wps/portal/wcc/home/snowClimateMonitoring/snowpack)
- [ERA5-Land Documentation](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
