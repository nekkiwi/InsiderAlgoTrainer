# Insider Algo Trainer

This repository collects insider trading information from [OpenInsider](http://openinsider.com) and historical market data to build machine-learning models that predict future stock performance. It provides utilities for scraping, feature engineering, target generation, walk‑forward validation, and final model training.

## Features

- Scrape periodic insider transactions and aggregate them by ticker.
- Enrich datasets with technical indicators and financial ratios.
- Download historical prices and compute return/alpha targets for multiple horizons.
- Perform walk‑forward validation using Random Forest or LightGBM models.
- Train final deployment models and save scalers, feature lists, and weights.

## Installation

Requires **Python 3.8+**. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. **Build the dataset**

   ```bash
   python scrape_data.py
   ```

   Intermediate and final data are written to `data/`.

2. **Run walk‑forward training**

   ```bash
   python train_walk_forward.py
   ```

   Configure time horizons, thresholds, and model type in the script or via arguments.

3. **Train final models**

   ```bash
   python train_final_model.py
   ```

   Saved models appear under `data/models/final_deployment/`.

## Project Structure

```
├── data/                 # Raw data, targets, and trained model artifacts
├── src/
│   ├── scraper/          # Data scraping and feature engineering
│   └── training/         # Walk‑forward and final model training
├── scrape_data.py        # Orchestrate data collection and target generation
├── train_walk_forward.py # Perform walk‑forward validation
├── train_final_model.py  # Train final models on full dataset
└── requirements.txt      # Python dependencies
```

## Contributing

Pull requests and issues are welcome. Please add tests for new features and follow PEP8 style guidelines.

## License

Provided for research and educational purposes. No warranty is offered; use at your own risk.

