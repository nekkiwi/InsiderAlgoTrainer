# InsiderAlgo

InsiderAlgo provides a full pipeline for collecting insider trading data, training predictive models and placing demo trades via the Alpaca API. The code is split into scraping, training, inference and trading modules under `src/`.

## Requirements

Create a Python 3.9+ environment and install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

You will also need an Alpaca account. Place your API credentials in `src/alpaca/api_keys.json` as:

```json
{ "api_key": "YOUR_KEY", "api_secret_key": "YOUR_SECRET" }
```

## Usage

### 1. Scrape data
Run `scrape_data.py` to download insider trades, technical indicators and financial ratios. The script saves cleaned features, price returns and target labels under `data/`.

```bash
python scrape_data.py
```

### 2. Train models
After scraping, train ML models with `train_model.py`. Feature selection and cross‑validation statistics are written to `data/training/`.

```bash
python train_model.py
```

### 3. Choose model and threshold
Use the notebook `find_best_model_and_pred_threshold.ipynb` to load the training results, visualise performance and decide on the best model/threshold combination.

### 4. Run inference
Generate predictions for the latest scraped data via `run_inference.py`. The output Excel file is stored in `data/inference/`.

```bash
python run_inference.py
```

### 5. Execute trades
Configure and execute paper trades with `excecute_trade.py`. You can trade purely on the chosen model/threshold or further screen tickers using valuation ratios (see `filter_eligible_stocks.ipynb`).

```bash
python excecute_trade.py
```

## Project layout

- **scrape_data.py** – orchestrates feature scraping and target generation.
- **train_model.py** – runs feature selection and cross‑validated model training.
- **run_inference.py** – loads trained models to score new insider trades.
- **excecute_trade.py** – places demo trades on Alpaca.
- **src/**
  - `scraper/` – scraping & preprocessing utilities.
  - `training/` – model training helpers.
  - `inference/` – inference pipeline.
  - `alpaca/` – Alpaca API integration.

All output files are written below `data/` (ignored from version control).

## Example workflow
1. `python scrape_data.py`
2. `python train_model.py`
3. Analyse `find_best_model_and_pred_threshold.ipynb` and choose the best score threshold.
4. `python run_inference.py`
5. Optionally run `filter_eligible_stocks.ipynb` to screen by fundamentals.
6. `python excecute_trade.py`

This repository is intended for research and educational use. Use at your own risk when trading real funds.
