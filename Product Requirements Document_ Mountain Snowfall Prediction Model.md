# Product Requirements Document: Mountain Snowfall Prediction Model

**Date**: January 01, 2026
**Author**: Manus AI
**Version**: 1.0

## 1. Introduction

Accurate snowfall prediction in mountainous regions is a significant challenge for standard numerical weather prediction (NWP) models. The complex terrain introduces microclimates and weather effects that coarse-resolution models fail to capture. Services like OpenSnow have demonstrated that a significant competitive advantage can be gained by using machine learning (ML) to post-process and downscale raw NWP output, resulting in more accurate mountain-specific forecasts. 

This document outlines the requirements for a data science project to develop a machine learning model that predicts snowfall and snow depth in mountain locations. The primary goal is to leverage the wealth of publicly available historical weather data, reanalysis datasets, and ground-truth observations to create a model that demonstrably improves upon the accuracy of standard weather forecasts.

## 2. Goals and Objectives

The primary goal of this project is to **develop a machine learning model that accurately predicts 24-hour snowfall and total snow depth for specific mountain locations.**

### Key Objectives:

1.  **Data Ingestion**: Build a robust data pipeline to acquire and process historical data from multiple sources, including weather reanalysis models, high-resolution forecast archives, and ground-truth observation networks.
2.  **Feature Engineering**: Develop a comprehensive feature set that includes atmospheric variables, terrain characteristics (elevation, slope, aspect), and temporal features.
3.  **Model Development**: Train, evaluate, and compare multiple machine learning architectures (e.g., Gradient Boosting, Deep Learning) to identify the most effective approach for snowfall prediction.
4.  **Performance Evaluation**: Quantify the model's performance against baseline NWP models (e.g., GFS, HRRR) and establish a clear measure of its predictive lift.
5.  **Prototype Deliverable**: Produce a trained model and a prototype API that can generate a snow forecast for a given latitude, longitude, and date.

## 3. Success Metrics

The success of this project will be measured by the following key performance indicators (KPIs):

| Metric | Target | Description |
|---|---|---|
| **Snow Depth RMSE** | < 15 cm | Root Mean Square Error for 24-hour snow depth prediction compared to SNOTEL observations. |
| **Snowfall Prediction Accuracy** | > 85% | F1-score for predicting snowfall events (>1 inch) over a 24-hour period. |
| **Improvement over Baseline** | > 20% | Percentage reduction in RMSE compared to the raw ERA5-Land or HRRR model output for the same locations. |
| **Bias** | < 5% | The model should not have a systematic tendency to over- or under-predict snowfall amounts. |

## 4. Requirements

This project is divided into several key requirement areas, from data acquisition to model deployment.

### 4.1. Data Acquisition

The model will be trained on a combination of ground-truth observations, historical weather model data, and static geographic data. The following table details the primary data sources to be used.

| Data Type | Primary Source | Resolution | Historical Period | Purpose |
|---|---|---|---|---|
| **Ground Truth (Snow)** | **SNOTEL** [1] | Point-based | 40+ years | The primary target variable for model training (snow depth, SWE). |
| **Ground Truth (Global)** | **GHCN-Daily** [2] | Point-based | 100+ years | Supplemental ground truth for snowfall and temperature. |
| **Long-Term Weather** | **ERA5-Land** [3] | ~9 km | 1950-Present | Provides a long, consistent historical record of atmospheric variables for initial model training. |
| **High-Resolution Weather** | **HRRR Archive** [4] | 3 km | 2014-Present | Provides high-resolution data for fine-tuning the model and capturing convective events in the US. |
| **Static Terrain Data** | **Copernicus DEM (GLO-30)** [5] | 30 m | N/A | Provides elevation data for deriving terrain features (slope, aspect). |
| **Ski Resort Locations** | **OpenSkiMap** [6] | Point/Polygon | N/A | Provides the specific locations (lat/lon) for which we want to generate forecasts. |

### 4.2. Data Processing and Engineering

1.  **Data Ingestion Pipeline**: Develop Python scripts using libraries like `metloom`, `herbie`, and `cdsapi` to automate the download and formatting of data from the sources listed above.
2.  **Spatial Alignment**: All data must be aligned to a common grid or referenced to specific point locations. SNOTEL and GHCN station data will be the ground truth points. Weather model data (ERA5-Land, HRRR) will be sampled at these station locations using nearest-neighbor or bilinear interpolation.
3.  **Temporal Alignment**: All time series data must be aligned to a consistent hourly or daily timestep (UTC).
4.  **Data Cleaning**: Handle missing values, flag outliers, and correct for any data quality issues identified in the source datasets.

### 4.3. Feature Engineering

The following features will be engineered to provide the model with the necessary predictive signals:

*   **Atmospheric Variables**: Temperature (at multiple pressure levels), dew point, relative humidity, wind speed and direction (U/V components), surface pressure, geopotential height.
*   **Precipitation Variables**: Total precipitation, precipitation type (rain vs. snow), snowfall rate.
*   **Terrain Features**: Elevation, slope, aspect, terrain roughness, distance to the nearest coast.
*   **Temporal Features**: Time of year (sine/cosine transformation), day of the week, hour of the day.
*   **Lagged Variables**: Include atmospheric variables from previous timesteps (e.g., T-6h, T-12h, T-24h) to capture trends.

### 4.4. Model Development

An iterative approach to model development will be taken, starting with simpler models and progressing to more complex architectures.

1.  **Baseline Model**: Establish a baseline performance using a simple linear regression or the raw output from the ERA5-Land/HRRR models.
2.  **Tree-Based Models**: Implement a Gradient Boosting model (e.g., LightGBM, XGBoost) as a strong, interpretable baseline. This model will use a flattened feature set for each location and timestep.
3.  **Deep Learning Models**: Explore deep learning architectures for capturing complex spatial and temporal patterns:
    *   **LSTM/GRU**: To model the time-series nature of the data at individual SNOTEL stations.
    *   **U-Net Architecture**: For statistical downscaling of coarse reanalysis data (ERA5-Land) to a higher resolution, using terrain features as a guide. This approach is inspired by recent advances in climate modeling [7].

### 4.5. Model Evaluation

1.  **Cross-Validation**: Use station-based, k-fold cross-validation to ensure the model generalizes well to unseen locations. Folds will be created by splitting the SNOTEL stations into distinct training and validation sets.
2.  **Temporal Holdout Set**: A final holdout set comprising the most recent two years of data will be reserved for final model evaluation and to test for performance degradation over time.
3.  **Evaluation Metrics**: The primary metrics will be RMSE for snow depth and F1-score for snowfall events, as defined in Section 3. Bias and Mean Absolute Error (MAE) will also be tracked.

## 5. Out of Scope

The following items are explicitly out of scope for this initial project:

*   Development of a user-facing web or mobile application.
*   Real-time, operational deployment of the model.
*   Building a custom data storage solution (e.g., a database). Data will be handled as files.
*   Forecasting for locations outside of the SNOTEL network coverage area (initially focused on the Western US).

## 6. High-Level Timeline

This project will be executed in four phases over an estimated 12-16 week period.

| Phase | Duration | Key Activities |
|---|---|---|
| **Phase 1: Data Acquisition & Pipeline** | 3-4 weeks | Develop and test data ingestion scripts for all primary data sources. |
| **Phase 2: Feature Engineering & Baseline Model** | 3-4 weeks | Process raw data, create feature set, train and evaluate baseline Gradient Boosting model. |
| **Phase 3: Deep Learning Model Development** | 4-6 weeks | Implement and train deep learning models (LSTM, U-Net). Hyperparameter tuning. |
| **Phase 4: Final Evaluation & Reporting** | 2 weeks | Evaluate final model on holdout set, compare all models, and document results and prototype API. |

## 7. References

[1] USDA Natural Resources Conservation Service. "SNOTEL (Snow Telemetry) Network." [https://www.nrcs.usda.gov/wps/portal/wcc/home/snowClimateMonitoring/snowpack/snotelDataInformation/](https://www.nrcs.usda.gov/wps/portal/wcc/home/snowClimateMonitoring/snowpack/snotelDataInformation/)

[2] NOAA National Centers for Environmental Information. "Global Historical Climatology Network daily (GHCNd)." [https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily)

[3] Copernicus Climate Change Service. "ERA5-Land hourly data from 1950 to present." [https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land)

[4] NOAA. "NOAA High-Resolution Rapid Refresh (HRRR) on AWS." *Registry of Open Data on AWS*. [https://registry.opendata.aws/noaa-hrrr-pds/](https://registry.opendata.aws/noaa-hrrr-pds/)

[5] Copernicus. "Copernicus DEM." *Registry of Open Data on AWS*. [https://registry.opendata.aws/copernicus-dem/](https://registry.opendata.aws/copernicus-dem/)

[6] OpenSkiMap.org. "OpenSkiMap - About." [https://openskimap.org/?about](https://openskimap.org/?about)

[7] ECMWF. "Let it snow! How machine learning will forecast snow in the AIFS." *AIFS Blog*. [https://www.ecmwf.int/en/about/media-centre/aifs-blog/2025/let-it-snow-machine-learning-snow-forecast](https://www.ecmwf.int/en/about/media-centre/aifs-blog/2025/let-it-snow-machine-learning-snow-forecast)
