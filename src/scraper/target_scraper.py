import os
import time
from datetime import timedelta
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
from .utils.target_scraper_helpers import process_ticker_targets, calculate_target_distribution, save_targets_to_excel

class TargetScraper:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.stock_data_file = os.path.join(data_dir, 'final/stock_data_final.xlsx')
        self.output_file = os.path.join(data_dir, 'final/targets_final.xlsx')
        self.distribution_output_file = os.path.join(data_dir, 'analysis/targets_distribution.xlsx')
        self.return_df = None
        self.alpha_df = None
        self.results = {}

    def load_stock_data(self, return_sheet_name='Returns', alpha_sheet_name='Alpha'):
        """Load return and alpha data from the stock data file."""
        self.return_df = pd.read_excel(self.stock_data_file, sheet_name=return_sheet_name)
        self.alpha_df = pd.read_excel(self.stock_data_file, sheet_name=alpha_sheet_name)

    def create_target_data(self):
        """Process target data for all ticker-filing date combinations."""
        ticker_info_list = self.return_df[['Ticker', 'Filing Date']].values.tolist()
        with Pool(cpu_count()) as pool:
            results = list(tqdm(pool.imap(self._process_ticker_targets_wrapper, ticker_info_list), total=len(ticker_info_list), desc="- Processing targets in parallel..."))

        for ticker_filing_date, target_data in results:
            if target_data is not None:
                self.results[ticker_filing_date] = target_data

    def _process_ticker_targets_wrapper(self, ticker_info):
        """Wrapper to pass instance attributes to helper's process_ticker_targets function."""
        return process_ticker_targets(ticker_info, self.return_df, self.alpha_df, self.limit_array, self.stop_array)

    def run(self, return_df, alpha_df, limit_array, stop_array):
        """Run the entire process to calculate targets, save them, and return the final DataFrame."""
        self.load_stock_data()
        start_time = time.time()
        print("\n### START ### Target Scraper")

        # Assign input values
        self.return_df = return_df if return_df is not None else self.return_df
        self.alpha_df = alpha_df if alpha_df is not None else self.alpha_df
        self.limit_array = limit_array
        self.stop_array = stop_array

        # Process target data
        self.create_target_data()

        # Save results to Excel and get the final DataFrame
        final_df = save_targets_to_excel(self.results, self.limit_array, self.stop_array, self.output_file)

        # Save target distribution
        calculate_target_distribution(self.results, self.distribution_output_file)

        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"### END ### Target Scraper - time elapsed: {elapsed_time}")
        return final_df

if __name__ == "__main__":
    scraper = TargetScraper()
    limit_array = [0.06, 0.08, 0.1, 0.12]
    stop_array = [-0.06, -0.08, -0.1, -0.12]
    scraper.run(None, None, limit_array, stop_array)
