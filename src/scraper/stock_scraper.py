import os
import time
from datetime import timedelta
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
# --- UPDATED: Import the new helper function ---
from .utils.stock_scraper_helpers import *


class StockDataScraper:
    """
    A class to scrape stock price data, calculate returns and alphas,
    and generate final target variables.
    """
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.features_file = os.path.join(data_dir, 'interim/train/5_features_full_cleaned.xlsx')
        self.features_out_file = os.path.join(data_dir, 'final/features_final.xlsx')
        self.stock_output_file = os.path.join(data_dir, 'final/stock_data_final.xlsx')
        self.targets_output_file = os.path.join(data_dir, 'final/targets_final.xlsx')
        self.targets_distribution_file = os.path.join(data_dir, 'analysis/targets_distribution.xlsx')
        
        self.ticker_filing_dates = None
        self.all_targets_df = None
        self.features_df = None

    def prepare_data(self, timepoints: dict):
        """
        Prepares the features DataFrame by converting dates, filtering for
        complete historical data availability, and extracting key columns.
        """
        self.features_df['Filing Date'] = pd.to_datetime(self.features_df['Filing Date'], dayfirst=True)

        max_days_needed = max(timepoints.values())
        
        cutoff_date = pd.Timestamp.now().normalize() - BDay(max_days_needed)
        
        original_rows = len(self.features_df)
        self.features_df = self.features_df[self.features_df['Filing Date'] <= cutoff_date].copy()
        
        new_rows = len(self.features_df)
        dropped_rows = original_rows - new_rows
        
        if dropped_rows > 0:
            print(f"\n- Filtering for data availability: Dropped {dropped_rows} recent entries.")
            print(f"- Processing {new_rows} entries with filing dates on or before {cutoff_date.date()} to ensure a full {max_days_needed}-day history is available.")
        
        self.ticker_filing_dates = self.features_df[['Ticker', 'Filing Date']]

    def create_stock_data_sheet(self, timepoints: dict):
        """
        Downloads stock data, calculates daily returns and alphas up to a max
        horizon, and creates a single DataFrame with all target values.
        """
        # --- FIXED: Correct calculation of max_days_needed (removed +1) ---
        # We need to fetch exactly the maximum number of business days required.
        max_days_needed = max(timepoints.values()) if timepoints else 0
        if max_days_needed == 0:
            raise ValueError("Timepoints dictionary cannot be empty.")

        ticker_info_list = [
            (row['Ticker'], row['Filing Date'], max_days_needed)
            for _, row in self.ticker_filing_dates.iterrows()
        ]

        with Pool(cpu_count()) as pool:
            results = list(tqdm(
                pool.imap(download_data_wrapper, ticker_info_list),
                total=len(ticker_info_list),
                desc="- Downloading stock data"
            ))

        rows = []
        for ticker, filing_date, max_days, stock_df, spy_df in results:
            if (stock_df is None or spy_df is None or stock_df.empty or spy_df.empty or
                'Close' not in stock_df.columns or 'Close' not in spy_df.columns):
                continue
            
            sp_raw = stock_df['Close']
            sp = sp_raw.iloc[:, 0] if isinstance(sp_raw, pd.DataFrame) else sp_raw
            bp_raw = spy_df['Close']
            bp = bp_raw.iloc[:, 0] if isinstance(bp_raw, pd.DataFrame) else bp_raw
            
            if len(sp) == 0 or len(bp) == 0 or sp.iloc[0] == 0 or bp.iloc[0] == 0:
                continue

            sr = (sp / sp.iloc[0]) - 1
            br = (bp / bp.iloc[0]) - 1
            alpha = sr - br

            row = {'Ticker': ticker, 'Filing Date': filing_date}
            # The loop now correctly iterates up to the number of days fetched
            for i in range(max_days_needed):
                row[f'return_day_{i+1}'] = sr.iloc[i] if i < len(sr) else None
                row[f'alpha_day_{i+1}'] = alpha.iloc[i] if i < len(alpha) else None
            rows.append(row)

        if not rows:
            raise RuntimeError("No valid stock data was produced after processing all downloads.")
        self.all_targets_df = pd.DataFrame(rows)

    def save_formatted_targets(self, timepoints: dict):
        """Saves the calculated return and alpha data in the specified format."""
        if self.all_targets_df is None:
            print("[WARN] No target data to save.")
            return
        
        max_days = max(timepoints.values())
        save_formatted_sheets(self.all_targets_df, max_days, self.stock_output_file)

    def save_final_features(self):
        """
        Cleans the features DataFrame by removing rows that do not have complete
        target data, ensuring perfect alignment between features and targets.
        """
        self.features_df['Filing Date'] = pd.to_datetime(self.features_df['Filing Date'], dayfirst=True)
        self.all_targets_df['Filing Date'] = pd.to_datetime(self.all_targets_df['Filing Date'], dayfirst=True)
        
        self.features_df['DateOnly'] = self.features_df['Filing Date'].dt.date
        self.all_targets_df['DateOnly'] = self.all_targets_df['Filing Date'].dt.date
        
        clean_targets = self.all_targets_df.dropna(subset=['Ticker', 'DateOnly'])
        unique_targets = clean_targets[['Ticker', 'DateOnly']].drop_duplicates()

        final_features = pd.merge(
            self.features_df, 
            unique_targets, 
            on=['Ticker', 'DateOnly'], 
            how='inner'
        )

        dropped = len(self.features_df) - len(final_features)
        print(f"- Dropped {dropped} rows from features due to missing target dates.")

        final_features = final_features.drop(columns=['DateOnly'])
        save_final_features(final_features, self.features_out_file)

    def generate_final_targets(self, timepoints):
        """
        Loads the generated stock data and processes it to create the final
        target variables for modeling.
        """
        print("\n--- Generating Final Target Variables ---")
        generate_and_save_final_targets(
            stock_data_file=self.stock_output_file,
            targets_output_file=self.targets_output_file,
            distribution_output_file=self.targets_distribution_file,
            timepoints=timepoints
        )

    # --- UPDATED: The run method now accepts a list of strings for timepoints ---
    def run(self, timepoints: list, features_df=None):
        """
        Executes the full scraping and processing pipeline.
        
        Args:
            timepoints (list): A list of time horizons (e.g., ['1w', '1m', '6m']).
        """
        start_time = time.time()
        print("\n### START ### Stock Scraper")
        
        if features_df is None:
            if not os.path.exists(self.features_file):
                raise FileNotFoundError(f"Input features file not found: {self.features_file}")
            self.features_df = pd.read_excel(self.features_file)
        else:
            self.features_df = features_df
        
        # --- NEW: Automatically convert timepoint strings to business days ---
        timepoints_bdays = convert_timepoints_to_bdays(timepoints)
        print(f"- Converted timepoints to business days: {timepoints_bdays}")
            
        self.prepare_data(timepoints_bdays)
        self.create_stock_data_sheet(timepoints_bdays)
        self.save_formatted_targets(timepoints_bdays)
        self.save_final_features()
        self.generate_final_targets(timepoints_bdays)
        
        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"### END ### Stock Scraper - time elapsed: {elapsed_time}")
        
        return self.features_df, self.all_targets_df
