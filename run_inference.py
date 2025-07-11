# In run_inference.py (new file in the project root directory)

import pandas as pd
import re

# Import the necessary classes from your project structure
from src.scraper.feature_scraper import FeatureScraper
from src.scraper.feature_preprocess import FeaturePreprocessor
from src.inference.model_inference import ModelInference

def main():
    """
    Main function to run the complete inference pipeline.
    1. Defines the winning strategy from backtesting.
    2. Scrapes and preprocesses new data.
    3. Runs inference using the final, deployed models.
    """
    ###############################################################
    # Part 1: Define the Winning Strategy from Walk-Forward Backtest
    ###############################################################
    # These parameters MUST match the strategy you identified as the best.
    # They determine which models and settings are loaded.
    
    WINNING_MODEL_TYPE      = "RandomForest"
    WINNING_CATEGORY        = "alpha"
    WINNING_OPTIMIZE_FOR    = "adjusted_sharpe"
    WINNING_TIMEPOINT       = "1w"
    WINNING_THRESHOLD_PCT   = 2
    WINNING_TOP_N           = 10

    ##################################################
    # Part 2: Scrape and Preprocess New Data
    ##################################################
    # This section would be run periodically (e.g., daily) to get the
    # latest insider transactions to run the model on.

    print("--- Scraping and Preprocessing New Data ---")
    feature_scraper = FeatureScraper()
    feature_preprocessor = FeaturePreprocessor()
    
    new_insider_buys_df = feature_scraper.run(num_weeks=1, train=False)
    
    if new_insider_buys_df is None or new_insider_buys_df.empty:
        print("No new data scraped. Exiting.")
        return
        
    preprocessed_data_df = feature_preprocessor.run(new_insider_buys_df, train=False)

    if preprocessed_data_df is None or preprocessed_data_df.empty:
        print("No data available after preprocessing. Exiting.")
        return

    ##############################################
    # Part 3: Run Inference on the New Data
    ##############################################
    
    print("\n--- Running Inference on New Data ---")
    
    # Initialize the inference class with the parameters of your winning strategy
    model_inference = ModelInference(
        model_type=WINNING_MODEL_TYPE,
        category=WINNING_CATEGORY,
        timepoint=WINNING_TIMEPOINT,
        threshold_pct=WINNING_THRESHOLD_PCT,
        top_n=WINNING_TOP_N,
        optimize_for=WINNING_OPTIMIZE_FOR
    )
    
    # Pass the preprocessed dataframe directly to the run method
    model_inference.run(inference_df=preprocessed_data_df)
    
if __name__ == "__main__":
    main()

