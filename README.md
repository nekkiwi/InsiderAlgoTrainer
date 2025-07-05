# InsiderAlgo

A research pipeline that scrapes insider trading disclosures, enriches them with stock and fundamental data, and trains machine‑learning models to predict future performance.  The workflow is organised into several stages that can be run individually or from `main.py`.

## Features

- **Feature scraping** – downloads insider trade filings from `openinsider.com`, cleans them and augments with technical indicators and fundamental ratios.
- **Stock scraping** – collects daily prices for each ticker and computes raw returns and alpha versus the S&P 500.
- **Target generation** – produces classification and regression targets using configurable limit/stop thresholds.
- **Data exploration** – normalises features, analyses correlations and selects informative predictors.
- **Model training** – supports RandomForest and simple neural‑network models with cross‑validation and optional over/undersampling.
- **Backtesting** – evaluates predictions against historical prices to measure hypothetical performance.

All intermediate artefacts are written under the `data/` directory so that later stages can be rerun without repeating the entire pipeline.

## Installation

The project uses Python 3.9 and relies on packages listed in `insideralgo.yml`.  A Conda environment can be created as follows:

```bash
conda env create -f insideralgo.yml
conda activate insideralgo
```

Some steps fetch large amounts of data from the internet (OpenInsider and Yahoo Finance).  Network access may be required when running the scrapers.

## Usage

To execute the full pipeline with default settings:

```bash
python main.py
```

This will sequentially scrape features and stock prices, compute targets, select features, train models and finally run a basic backtest.  Each component can also be used independently; see the corresponding modules under `src/` for details.

Results and intermediate files are stored in the `data/` folder:

- `interim/` – cleaned feature data at various stages
- `final/` – final feature set, stock data and targets
- `output/` – analysis plots and feature distributions
- `training/` – saved statistics, predictions and models
- `backtest/` – aggregated backtesting results

## Repository Structure

```
src/
  scraper/           scraping utilities and feature/target generation
  data_exploration/  feature normalisation and analysis
  training/          model training helpers
  backtesting/       prediction backtesting utilities
  analysis/          additional evaluation scripts
main.py              example pipeline that ties everything together
```

## Notes

- The project currently saves models using `joblib`, but the call is commented out in `train_helpers.py`.
- HTTP is used when contacting OpenInsider (`http://openinsider.com`).  Consider switching to HTTPS if possible.

