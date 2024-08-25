import pandas as pd
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys
import os

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.stock_scraper_helpers import *

class StockDataScraper:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.features_file = os.path.join(data_dir, 'final/features_final.xlsx')
        self.output_file = os.path.join(data_dir, 'final/stock_data_final.xlsx')
        self.ticker_filing_dates = None
        self.stock_data_df = None
        self.max_days = 20
        self.return_df = None
        self.alpha_df = None
        self.earliest_date = None
        self.spy_data = None

    def load_features(self):
        """Load feature data and set up ticker and date information."""
        self.ticker_filing_dates, self.earliest_date = load_features(self.features_file)
        
    def download_spy_data(self):
        """Download SPY data for the period covering the earliest to the latest filing dates plus 20 business days."""
        latest_date = self.ticker_filing_dates['Filing Date'].max()
        self.spy_data = yf.download('SPY', start=self.earliest_date, end=latest_date, interval='1d', progress=False)
        if self.spy_data is not None:
            self.spy_data = self.spy_data['Close'].reset_index(drop=True)
        else:
            print("Failed to download SPY data.")

    def calculate_returns_and_alpha(self):
        """Calculate returns relative to filing date and alpha relative to SPY."""
        self.return_df = self.stock_data_df.copy()
        self.alpha_df = self.stock_data_df.copy()
        
        for col in self.stock_data_df.columns[2:]:  # Skip Ticker and Filing Date columns
            # Calculate returns
            self.return_df[col] = (self.stock_data_df[col] / self.stock_data_df['Day 1']) - 1
            
            # Calculate alpha relative to SPY
            if self.spy_data is not None:
                spy_return = (self.spy_data.loc[:len(self.stock_data_df[col])-1].reset_index(drop=True) / self.spy_data[0]) - 1
                self.alpha_df[col] = self.return_df[col] - spy_return

    def create_stock_data_sheet(self):
        """Create a stock data sheet with tickers as rows and daily closing prices as columns."""
        stock_data_dict = {}

        # Prepare the list of ticker and filing date pairs for parallel processing
        ticker_info_list = [(row['Ticker'], row['Filing Date'], self.max_days) for _, row in self.ticker_filing_dates.iterrows()]

        # Use multiprocessing to download ticker data in parallel
        print("Downloading ticker data in parallel...")
        with Pool(cpu_count()) as pool:
            results = list(tqdm(pool.imap(download_data_wrapper, ticker_info_list), total=len(ticker_info_list)))

        stock_data_dict = create_stock_data_dict(results)

        # Convert the dictionary to a DataFrame
        self.stock_data_df = pd.DataFrame.from_dict(stock_data_dict, orient='index').reset_index()
        self.stock_data_df.columns = ['Ticker', 'Filing Date'] + [f'Day {i+1}' for i in range(self.max_days)]

    def save_to_excel(self):
        """Save the stock data, returns, and alpha sheets to an Excel file."""
        save_to_excel(self.stock_data_df, self.return_df, self.alpha_df, self.output_file)

    def run(self):
        """Run the full process to create the stock data sheet, calculate returns, and calculate alpha."""
        self.load_features()
        self.download_spy_data()
        self.create_stock_data_sheet()
        self.calculate_returns_and_alpha()
        self.save_to_excel()
        print("Stock data sheet head:")
        print(self.stock_data_df.head())
        print("Return data sheet head:")
        print(self.return_df.head())
        print("Alpha data sheet head:")
        print(self.alpha_df.head())

if __name__ == "__main__":
    scraper = StockDataScraper()
    scraper.run()
