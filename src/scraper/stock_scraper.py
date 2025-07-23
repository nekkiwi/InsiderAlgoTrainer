import os
import time
from datetime import timedelta
import pandas as pd

# Import the new, optimized helpers
from .utils.stock_scraper_helpers import (
    convert_timepoints_to_bdays,
    batch_fetch_stock_targets,
    save_formatted_sheets,
    save_final_features,
    generate_and_save_final_targets_ohlc,
    batch_fetch_stock_targets_ohlc,
    save_formatted_ohlc_sheets
)

class StockDataScraper:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.features_file = os.path.join(data_dir, 'interim/train/3_features_preprocessed.xlsx')
        self.features_out_file = os.path.join(data_dir, 'final/features_final.xlsx')
        self.stock_output_file = os.path.join(data_dir, 'final/stock_data_final.xlsx')
        self.targets_output_file = os.path.join(data_dir, 'final/targets_final.xlsx')
        self.targets_distribution_file = os.path.join(data_dir, 'analysis/targets_distribution.xlsx')

    def run(self, features_df, timepoints: list):
        start_time = time.time()
        print("\n### START ### Stock Scraper")
        print(f"[DEBUG] Input features_df is {'provided' if features_df is not None else 'None'}")

        if features_df is None:
            print(f"[DEBUG] Attempting to load features from {self.features_file}")
            if not os.path.exists(self.features_file):
                raise FileNotFoundError(f"Input features file not found: {self.features_file}")
            self.features_df = pd.read_excel(self.features_file)
            print(f"[DEBUG] Loaded features: {self.features_df.shape}")
        else:
            self.features_df = features_df
            print(f"[DEBUG] Using provided features: {self.features_df.shape}")

        timepoints_bdays = convert_timepoints_to_bdays(timepoints)
        print(f"- Converted timepoints to business days: {timepoints_bdays}")

        print(f"[DEBUG] Features date range: {self.features_df['Filing Date'].min()} to {self.features_df['Filing Date'].max()}")

        print("- Fetching OHLC-based targets...")
        final_features_df, ohlc_targets_dict = batch_fetch_stock_targets_ohlc(self.features_df, timepoints_bdays)

        if final_features_df.empty or all(len(df) == 0 for df in ohlc_targets_dict.values()):
            print("[ERROR] Failed to generate any OHLC target data. Aborting save operations.")
            print(f"[DEBUG] final_features_df shape: {final_features_df.shape}")

        print(f"- Saving {len(final_features_df)} aligned feature rows...")
        save_final_features(final_features_df, self.features_out_file)

        print(f"- Saving daily OHLC stock/alpha data...")
        save_formatted_ohlc_sheets(ohlc_targets_dict, self.stock_output_file)

        print("- Generating and saving final aggregated targets...")
        generate_and_save_final_targets_ohlc(stock_data_file=self.stock_output_file, targets_output_file=self.targets_output_file, timepoints=timepoints_bdays)

        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"### END ### Stock Scraper - time elapsed: {elapsed_time}")

