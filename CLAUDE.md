# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **2-day time series lecture/workshop** repository consisting of Jupyter notebooks and supporting datasets. The content is bilingual (Korean/English) and progresses from foundational time series concepts to deep learning-based forecasting and anomaly detection.

## Notebook Curriculum (numbered by topic progression)

- **010-020**: Foundations — stationarity, ADF test, differencing, moving averages, simple trade strategies
- **030-035**: Persistence forecasting, series-to-supervised data transformation
- **041-045**: RNN input/output shapes, windowed data generation, learning rate & Huber loss
- **080**: Conv1D with learning rate scheduling
- **100-120**: LSTM input/output shapes, LSTM on shampoo sales dataset
- **130**: Multivariate multi-step forecasting (household energy, LSTM)
- **170-171**: Anomaly detection with Conv1D and LSTM autoencoders

## Key Libraries

- **TensorFlow/Keras** for all deep learning models (RNN, LSTM, Conv1D, autoencoders)
- **pandas, numpy, matplotlib, seaborn** for data manipulation and visualization
- **scikit-learn** for preprocessing (StandardScaler) and metrics (MSE, MAE)
- **statsmodels** for statistical tests (ADF test)

## Datasets

Local datasets are in `datasets/`. Some notebooks also fetch data from remote URLs (e.g., TensorFlow storage for sunspots data).

- `sunspots.csv`, `shampoo-sales.csv`, `international-airline-passengers.csv` — univariate time series
- `household_power_consumption_days.csv` — multivariate (8 features, daily aggregated)
- `artificialNoAnomaly_*.csv`, `artificialWithAnomaly_*.csv` — anomaly detection datasets
- `annual_csv.csv`, `Sample - Superstore.xls`, `winequality-red.csv` — supplementary datasets

## Korean Font Support

[korean.py](korean.py) configures matplotlib for Korean text rendering (AppleGothic on Mac, Gulim on Windows). Import it in notebooks that use Korean labels on plots.

## Common Patterns

- **Windowed dataset creation**: `tf.data.Dataset.from_tensor_slices` → `.window()` → `.flat_map()` → `.shuffle()` → `.map()` → `.batch().prefetch()` — used across multiple notebooks (035, 043, 130)
- **Train/test split**: Typically chronological split (no shuffle), often the last N records as test
- **Scaling**: `StandardScaler` fit on train, transform on both train and test; inverse transform for plotting predictions
