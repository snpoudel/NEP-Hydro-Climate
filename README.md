# NEP-Hydro-Climate

This repository contains scripts for hydrological modeling and climate change impact assessment on design flood statistics for small hydropower projects in Nepal. A process-based hydrological model (HBV) enhanced with a Light Gradient Boosting Machine (LGBM) post-processor is used to simulate streamflow under historical and future climate scenarios, enabling assessment of changes in key design flood statistics under future climate conditions.

---

## Overview of Scripts

The repository is organized into four categories: data preprocessing, model calibration and simulation, figure generation, and core model implementation.

---

### 📦 Preprocessing Scripts

| Script                               | Description |
|--------------------------------------|-------------|
| `1.1.select_hpp.py`                  | Selects small hydropower projects in Nepal from the DoED database. |
| `1.2preprocess_obs_streamflow.py`    | Preprocesses observed streamflow data for input into the HBV model. |
| `2.1download_era5_data.py`           | Downloads ERA5 reanalysis climate data for the selected region and period. |
| `2.2preprocess_era5.py`              | Processes ERA5 data to make it suitable for HBV model input. |
| `2.3combine_era5.py`                 | Combines ERA5 climate data with observed streamflow for model calibration. |
| `3.1download_hist_climate_data.py`   | Downloads historical climate model outputs from NASA's Center for Climate Simulation. |
| `3.2preprocess_hist_climate_data.py` | Preprocesses historical climate data for hydrological modeling. |
| `3.3combine_hist_climate.py`         | Combines processed historical climate variables into HBV-compatible format. |
| `4.1download_future_climate_data.py` | Downloads future climate projections from NASA's Center for Climate Simulation. |
| `4.2preprocess_future_climate.py`    | Processes future climate data for hydrological simulations. |
| `4.3combine_future_climate.py`       | Combines future climate variables into a format suitable for HBV simulations. |

---

### ⚙️ Model Calibration and Simulation Scripts

| Script                               | Description |
|--------------------------------------|-------------|
| `5.1hbv_era5calibrate.py`            | Calibrates the HBV model using ERA5 data and observed streamflow. |
| `5.2postprocessor_hbv.py`            | Develops LGBM post-processing models to correct biases in HBV simulations. |
| `5.3hbv_era5simulate.py`             | Simulates streamflow using ERA5 data and calibrated HBV parameters. |
| `5.4hbv_ncss_simulate.py`            | Simulates streamflow using historical climate model data. |
| `5.5hbv_ncss_simulate_future.py`     | Simulates future streamflow using projected climate scenarios. |

---

### 📊 Figure Generation Scripts

| Script                               | Description |
|--------------------------------------|-------------|
| `6.1hbv_era5evaluate_plot.py`        | Evaluates model performance using metrics like NSE during calibration and validation. |
| `6.2timeseries_era5plot.py`          | Generates time series plots of simulated streamflow and precipitation during validation. |
| `6.3return_period_plot.py`           | Computes and plots return period statistics (e.g., 25-, 50-, and 100-year floods). |
| `6.4relative_change_plot.py`         | Computes and plots relative changes in flood magnitudes under future climate scenarios. |

---

### 🤖 Hydrological Model

| Script         | Description |
|----------------|-------------|
| `hbv_model.py` | Core implementation of the HBV hydrological model used in all simulations. |

---

