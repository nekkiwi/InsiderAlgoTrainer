# OpenInsider Project

## Overview

This project is designed to scrape and analyze insider trading data, integrate stock information, and calculate financial ratios and technical indicators. It also includes functionalities for data preprocessing and potentially applying machine learning models for predictive analysis.

## Project Structure

- `src/`: Contains the main codebase.
  - `scraper/`: 
    - `feature_scraper.py`: Main script for scraping and processing insider trading data.
  - `utils/`: Utility functions and helpers.
    - `feature_scraper_helpers.py`: Functions to clean and process scraped data.
    - `technical_indicators_helpers.py`: Functions to calculate and normalize technical indicators.
    - `financial_ratios_helpers.py`: Functions to fetch financial data and calculate financial ratios.
  - `analysis/`: (Future development) Modules for feature analysis.
  - `models/`: (Future development) Modules for model training and evaluation.
- `tests/`: Unit tests for the project.
- `docs/`: Project documentation.

## Setup

### 1. Clone the Repository

Clone this repository to your local machine using:

```bash
git clone https://github.com/nekkiwi/InsiderAlgo.git
