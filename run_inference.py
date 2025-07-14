# In run_inference.py (in the project root directory)

import pandas as pd
import argparse # Import the argparse library

# Import the necessary classes from your project structure
from src.scraper.feature_scraper import FeatureScraper
from src.scraper.feature_preprocess import FeaturePreprocessor
from src.inference.model_inference import ModelInference

def main(args):
    """
    Main function to run the complete inference pipeline.
    1. Defines the winning strategy from backtesting.
    2. Scrapes and preprocesses new data.
    3. Runs inference using the final, deployed models.
    """
    ###############################################################
    # Part 1: Define the Winning Strategy from Walk-Forward Backtest
    ###############################################################
    # These parameters are kept constant as they define the core strategy.
    # Timepoint and Threshold are now passed as command-line arguments.
    WINNING_MODEL_TYPE = "RandomForest"
    WINNING_CATEGORY = "alpha"
    WINNING_OPTIMIZE_FOR = "adjusted_sharpe"
    WINNING_TOP_N = 10

    ##################################################
    # Part 2: Scrape and Preprocess New Data
    ##################################################
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
    
    # Initialize the inference class with a mix of constant parameters
    # and arguments passed from the command line.
    model_inference = ModelInference(
        model_type=WINNING_MODEL_TYPE,
        category=WINNING_CATEGORY,
        timepoint=args.timepoint, # Use the timepoint from CLI args
        threshold_pct=args.threshold_pct, # Use the threshold from CLI args
        top_n=WINNING_TOP_N,
        optimize_for=WINNING_OPTIMIZE_FOR
    )

    # Pass the preprocessed dataframe directly to the run method
    model_inference.run(inference_df=preprocessed_data_df)

if __name__ == "__main__":
    # --- Define and parse command-line arguments ---
    parser = argparse.ArgumentParser(description="Run the model inference pipeline with specific parameters.")
    
    # Add an argument for 'timepoint', making it required.
    parser.add_argument(
        "--timepoint", 
        type=str, 
        required=True, 
        help="The prediction timepoint to use (e.g., '1w', '3m')."
    )
    
    # Add an argument for 'threshold_pct', making it required.
    parser.add_argument(
        "--threshold_pct", 
        type=int, 
        required=True, 
        help="The threshold percentage for defining the binary target (e.g., 2 for 2%%)."
    )
    
    args = parser.parse_args()
    
    # Pass the parsed arguments to the main function
    main(args)
