import pandas as pd
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.target_scraper_helpers import process_targets, calculate_target_distribution

class TargetScraper:
    def __init__(self, limit_array=None, stop_array=None):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.stock_data_file = os.path.join(data_dir, 'final/stock_data_final.xlsx')
        self.output_file = os.path.join(data_dir, 'final/targets_final.xlsx')
        self.distribution_output_file = os.path.join(data_dir, 'output/targets_distribution.xlsx')
        self.stock_data_df = None
        self.results = {}
        self.limit_array = limit_array if limit_array is not None else [0.1]  # Default to 10% limit
        self.stop_array = stop_array if stop_array is not None else [-0.05]  # Default to -5% stop

    def load_stock_data(self):
        """Load stock data from the pre-downloaded Excel file."""
        self.stock_data_df = pd.read_excel(self.stock_data_file, sheet_name='Stock Data')

    def create_target_data(self):
        """Create target data for each ticker-filing datetime combination."""
        ticker_info_list = self.stock_data_df[['Ticker', 'Filing Date']].values.tolist()

        print("Processing targets in parallel...")
        with Pool(cpu_count()) as pool:
            results = list(tqdm(pool.imap(self.process_ticker_targets, ticker_info_list), total=len(ticker_info_list)))

        for ticker_filing_date, target_data in results:
            if target_data is not None:
                self.results[ticker_filing_date] = target_data

    def process_ticker_targets(self, ticker_info):
        """Helper function to process targets in parallel."""
        ticker, filing_date = ticker_info
        ticker_filing_date = (ticker, filing_date)
        
        stock_data = self.stock_data_df[
            (self.stock_data_df['Ticker'] == ticker) & 
            (self.stock_data_df['Filing Date'] == filing_date)
        ].iloc[:, 2:].squeeze()

        if not stock_data.empty:
            return ticker_filing_date, process_targets(stock_data, self.limit_array, self.stop_array)
        else:
            return ticker_filing_date, None

    def save_to_excel(self):
        """Save the targets to an Excel file."""
        try:
            with pd.ExcelWriter(self.output_file) as writer:
                for target_key in self.results[next(iter(self.results))].keys():
                    # Extract limit and stop values to create a sheet name
                    limit, stop = target_key
                    sheet_name = f'lim {limit} stop {stop}'

                    # Prepare the DataFrame for the current limit-stop combination
                    target_df = pd.DataFrame({
                        'Ticker': [ticker_filing_date[0] for ticker_filing_date in self.results.keys()],
                        'Filing Date': [ticker_filing_date[1] for ticker_filing_date in self.results.keys()],
                        **{metric: [data[target_key].get(metric, None) for data in self.results.values()]
                           for metric in self.results[next(iter(self.results))][target_key].keys()}
                    })

                    # Save to the appropriate sheet
                    target_df.to_excel(writer, sheet_name=sheet_name, index=False)

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
        self.load_stock_data()
        self.create_target_data()
        self.save_to_excel()
        self.save_target_distribution()

if __name__ == "__main__":
    scraper = TargetScraper(limit_array=[0.1, 0.2], stop_array=[-0.05, -0.1])
    scraper.run()
