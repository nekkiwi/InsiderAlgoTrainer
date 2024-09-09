import os
import time
from datetime import timedelta
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from .utils.stock_scraper_helpers import *

class StockDataScraper:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.features_file = os.path.join(data_dir, 'interim/5_features_full_cleaned.xlsx')
        self.features_out_file = os.path.join(data_dir, 'final/features_final.xlsx')
        self.output_file = os.path.join(data_dir, 'final/stock_data_final.xlsx')
        self.ticker_filing_dates = None
        self.return_df = None
        self.alpha_df = None
        self.max_days = 20

    def prepare_data(self):
        self.features_df['Filing Date'] = pd.to_datetime(self.features_df['Filing Date'], dayfirst=True)
        self.ticker_filing_dates = self.features_df[['Ticker', 'Filing Date']]

    def create_stock_data_sheet(self):
        """Create return and alpha data sheets."""
        ticker_info_list = [(row['Ticker'], row['Filing Date'], self.max_days) for _, row in self.ticker_filing_dates.iterrows()]

        # Download stock and SPY data in parallel
        with Pool(cpu_count()) as pool:
            results = list(tqdm(pool.imap(download_data_wrapper, ticker_info_list), total=len(ticker_info_list), desc="- Downloading stock data"))

        # Create stock data, return, and alpha DataFrames
        stock_data_dict = create_stock_data_dict(results)
        stock_data_df = pd.DataFrame.from_dict(stock_data_dict, orient='index').reset_index()

        # Rename 'level_0' and 'level_1' to 'Ticker' and 'Filing Date'
        stock_data_df.rename(columns={'level_0': 'Ticker', 'level_1': 'Filing Date'}, inplace=True)

        # Separate stock and alpha columns
        stock_columns = [col for col in stock_data_df.columns if 'Stock' in col]
        self.return_df = calculate_returns(stock_data_df, stock_columns)
        self.alpha_df = stock_data_df[['Ticker', 'Filing Date'] + [col for col in stock_data_df.columns if 'Alpha' in col]].copy()

    def save_to_excel(self):
        """Save the returns and alpha sheets to an Excel file."""
        save_to_excel(self.return_df, self.alpha_df, self.output_file)
        
    def save_final_features(self):
        save_final_features(self.features_df, self.features_out_file)

    def run(self, features_df):        
        start_time = time.time()
        print("\n### START ### Stock Scraper")
        
        if features_df is None:
            self.features_df = pd.read_excel(self.features_file)
        else:
            self.features_df = features_df
        self.prepare_data()
        self.create_stock_data_sheet()
        self.save_to_excel()
        self.save_final_features()
        
        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"### END ### Stock Scraper - time elapsed: {elapsed_time}")
        
        return self.features_df, self.return_df, self.alpha_df
