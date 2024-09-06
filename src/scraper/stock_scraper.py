import os
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
        self.stock_data_df = None
        self.return_df = None
        self.alpha_df = None
        self.max_days = 20

    def load_features(self):
        """Load feature data and set up ticker and date information."""
        self.ticker_filing_dates, _ = load_features(self.features_file)

    def create_stock_data_sheet(self):
        """Create a stock data sheet with tickers and filing dates as rows and daily closing prices as columns."""
        ticker_info_list = [(row['Ticker'], row['Filing Date'], self.max_days) for _, row in self.ticker_filing_dates.iterrows()]

        # Download stock and SPY data in parallel
        with Pool(cpu_count()) as pool:
            results = list(tqdm(pool.imap(download_data_wrapper, ticker_info_list), total=len(ticker_info_list), desc="Downloading stock data"))

        # Create stock data, return, and alpha DataFrames
        stock_data_dict = create_stock_data_dict(results)
        self.stock_data_df = pd.DataFrame.from_dict(stock_data_dict, orient='index').reset_index()

        # Rename 'level_0' and 'level_1' to 'Ticker' and 'Filing Date'
        self.stock_data_df.rename(columns={'level_0': 'Ticker', 'level_1': 'Filing Date'}, inplace=True)

        # Separate stock and alpha columns
        stock_columns = [col for col in self.stock_data_df.columns if 'Stock' in col]
        self.return_df = calculate_returns(self.stock_data_df, stock_columns)
        self.alpha_df = self.stock_data_df[['Ticker', 'Filing Date'] + [col for col in self.stock_data_df.columns if 'Alpha' in col]].copy()

        # Retain only stock columns for the stock data DataFrame
        self.stock_data_df = self.stock_data_df[['Ticker', 'Filing Date'] + stock_columns]

    def save_to_excel(self):
        """Save the stock data, returns, and alpha sheets to an Excel file."""
        save_to_excel(self.stock_data_df, self.return_df, self.alpha_df, self.output_file)

    def run(self):
        """Run the full process to create the stock data sheet, calculate returns, and calculate alpha."""
        self.load_features()
        self.create_stock_data_sheet()
        self.save_to_excel()

if __name__ == "__main__":
    scraper = StockDataScraper()
    scraper.run()
