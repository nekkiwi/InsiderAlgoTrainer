import pandas as pd
import os
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.target_scraper_helpers import download_daily_stock_data, process_targets, calculate_target_distribution

class TargetScraper:
    def __init__(self, limit_array=None, stop_array=None, timeout=30):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.features_file = os.path.join(data_dir, 'processed/features_complete.xlsx')
        self.output_file = os.path.join(data_dir, 'processed/targets.xlsx')
        self.distribution_output_file = os.path.join(data_dir, 'output/targets_distribution.xlsx')
        self.features_df = None
        self.ticker_filing_dates = None
        self.results = {}
        self.limit_array = limit_array if limit_array is not None else [0.1]  # Default to 10% limit
        self.stop_array = stop_array if stop_array is not None else [-0.05]  # Default to -5% stop
        self.timeout = timeout  # Default to 30 days

    def load_features(self):
        """Load the full features DataFrame and extract Ticker and Filing Date columns."""
        self.features_df = pd.read_excel(self.features_file)
        self.features_df['Filing Date'] = pd.to_datetime(self.features_df['Filing Date'], dayfirst=True)
        self.ticker_filing_dates = self.features_df[['Ticker', 'Filing Date']]

    def create_target_data(self):
        """Create target data for each ticker-filing datetime combination."""
        ticker_info_list = self.ticker_filing_dates.values.tolist()

        print("Downloading ticker data and processing targets in parallel...")
        with Pool(cpu_count()) as pool:
            results = list(tqdm(pool.imap(self.process_ticker_targets, ticker_info_list), total=len(ticker_info_list)))

        for ticker_filing_date, target_data in results:
            if target_data is not None:
                self.results[ticker_filing_date] = target_data

    def process_ticker_targets(self, ticker_info):
        """Helper function to download data and process targets in parallel."""
        ticker, filing_date = ticker_info
        end_date = min(filing_date + timedelta(days=self.timeout), datetime.now())
        stock_data, spy_data = download_daily_stock_data(ticker, filing_date, end_date)
        if stock_data is not None and spy_data is not None:
            formatted_filing_date = filing_date.strftime('%d/%m/%Y')
            return (ticker, formatted_filing_date), process_targets(ticker, stock_data, spy_data, self.limit_array, self.stop_array)
        else:
            return (ticker, filing_date.strftime('%d/%m/%Y')), None

    def save_to_excel(self):
        """Save the targets to an Excel file."""
        try:
            with pd.ExcelWriter(self.output_file) as writer:
                for target_name in self.results[next(iter(self.results))].keys():
                    try:
                        target_df = pd.DataFrame({
                            'Ticker': [ticker_filing_date[0] for ticker_filing_date in self.results.keys()],
                            'Filing Date': [ticker_filing_date[1] for ticker_filing_date in self.results.keys()],
                            **{ticker_filing_date: data[target_name] for ticker_filing_date, data in self.results.items() if data is not None}
                        })
                        target_df.to_excel(writer, sheet_name=target_name, index=False)
                    except ValueError as ve:
                        print(f"Skipping target '{target_name}' due to ValueError: {ve}")
                    except Exception as e:
                        print(f"An error occurred while saving target '{target_name}': {e}")

            print(f"Target data successfully saved to {self.output_file}.")
        except Exception as e:
            print(f"Failed to save target data to Excel: {e}")


    def save_target_distribution(self):
        """Calculate and save the target distribution."""
        distribution_df = calculate_target_distribution(self.results)
        distribution_df.to_excel(self.distribution_output_file, index=False)
        print(f"Target distribution successfully saved to {self.distribution_output_file}.")

    def run(self):
        """Run the full process to calculate targets and save the results."""
        self.load_features()
        self.create_target_data()
        self.save_to_excel()
        self.save_target_distribution()

if __name__ == "__main__":
    scraper = TargetScraper(limit_array=[0.1, 0.2], stop_array=[-0.05, -0.1])
    scraper.run()
