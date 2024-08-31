import os
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.stock_scraper_helpers import load_features, download_data_wrapper, create_stock_data_dict, save_to_excel

class StockDataScraper:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.features_file = os.path.join(data_dir, 'interim/5_features_full_cleaned.xlsx')
        self.features_out_file = os.path.join(data_dir, 'final/features_final.xlsx')
        self.output_file = os.path.join(data_dir, 'final/stock_data_final.xlsx')
        self.ticker_filing_dates = None
        self.stock_data_df = None
        self.max_days = 20
        self.return_df = None

    def load_features(self):
        """Load feature data and set up ticker and date information."""
        self.ticker_filing_dates, _ = load_features(self.features_file)

    def calculate_returns(self):
        """Calculate returns relative to filing date."""
        self.return_df = self.stock_data_df.copy()

        # Define the column names to operate on (skip 'Ticker' and 'Filing Date')
        data_columns = self.stock_data_df.columns[2:]

        for index, row in self.stock_data_df.iterrows():
            # Calculate returns
            self.return_df.loc[index, data_columns] = (self.stock_data_df.loc[index, data_columns] / self.stock_data_df.loc[index, 'Day 1']) - 1

    def create_stock_data_sheet(self):
        """Create a stock data sheet with tickers and filing dates as rows and daily closing prices as columns."""
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
        """Save the stock data and returns sheets to an Excel file after filtering out tickers not in stock data."""
        # Load the original features file
        features_df = pd.read_excel(self.features_file)

        # Ensure both DataFrames have matching indices based on Ticker and Filing Date
        self.stock_data_df['Filing Date'] = pd.to_datetime(self.stock_data_df['Filing Date'])
        features_df['Filing Date'] = pd.to_datetime(features_df['Filing Date'])

        # Set the index for both DataFrames to Ticker and Filing Date for proper alignment
        stock_indexed_df = self.stock_data_df.set_index(['Ticker', 'Filing Date'])
        features_indexed_df = features_df.set_index(['Ticker', 'Filing Date'])

        # Filter the features DataFrame to keep only the tickers and filing dates that have stock data
        filtered_features_df = features_indexed_df.loc[features_indexed_df.index.intersection(stock_indexed_df.index)]

        # Reset index to save to Excel
        filtered_features_df.reset_index(inplace=True)
        
        # Save the filtered features to the final features file
        filtered_features_df.to_excel(self.features_out_file, index=False)
        
        # Save the stock data and returns to the final output file
        stock_data_dir = os.path.dirname(self.output_file)
        os.makedirs(stock_data_dir, exist_ok=True)
        
        self.return_df['Filing Date'] = self.return_df['Filing Date'].dt.strftime('%d/%m/%Y %H:%M')
        
        with pd.ExcelWriter(self.output_file) as writer:
            self.stock_data_df.to_excel(writer, sheet_name='Stock Data', index=False)
            self.return_df.to_excel(writer, sheet_name='Returns', index=False)
        
        print(f"Filtered features saved to {self.features_out_file}.")
        print(f"Stock data and returns saved to {self.output_file}.")

    def run(self):
        """Run the full process to create the stock data sheet and calculate returns."""
        self.load_features()
        self.create_stock_data_sheet()
        self.calculate_returns()
        self.save_to_excel()
        print("Stock data sheet head:")
        print(self.stock_data_df.head())
        print("Return data sheet head:")
        print(self.return_df.head())

if __name__ == "__main__":
    scraper = StockDataScraper()
    scraper.run()
