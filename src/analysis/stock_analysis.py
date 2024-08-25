import pandas as pd
from datetime import datetime, timedelta
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys
import os

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.stock_analysis_helpers import (
    download_daily_stock_data,
    calculate_alpha,
    plot_stock_histories,
    plot_median_mean_alpha,
    plot_median_mean_return
)

class StockHistoryAnalyzer:
    def __init__(self, timeout=30, limit_array=None, stop_array=None):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.features_file = os.path.join(data_dir, 'final/features_final.xlsx')
        self.output_dir = os.path.join(data_dir, 'output/stock_analysis')
        self.timeout = timeout
        self.limit_array = limit_array if limit_array is not None else [0.1]  # Default to 10% limit
        self.stop_array = stop_array if stop_array is not None else [-0.05]  # Default to -5% stop
        self.features_df = None
        self.ticker_filing_dates = None
        self.stock_data_dict = {}
        self.spy_data = None

    def load_features(self):
        """Load the full features DataFrame and extract Ticker and Filing Date columns."""
        self.features_df = pd.read_excel(self.features_file)
        self.features_df['Filing Date'] = pd.to_datetime(self.features_df['Filing Date'], dayfirst=True)
        self.ticker_filing_dates = self.features_df[['Ticker', 'Filing Date']]

    def download_all_stock_data(self):
        """Download stock data for all tickers from their filing date until the timeout period."""
        ticker_info_list = self.ticker_filing_dates.values.tolist()
        end_date = datetime.now()
        self.spy_data = download_daily_stock_data('SPY', self.ticker_filing_dates['Filing Date'].min(), end_date)

        with Pool(cpu_count()) as pool:
            results = list(tqdm(pool.imap(self.download_ticker_data, ticker_info_list), total=len(ticker_info_list)))

        # Filter and store the results
        for ticker, data in results:
            if data is not None:
                self.stock_data_dict[ticker] = data

    def download_ticker_data(self, ticker_info):
        """Helper function to download stock data for a single ticker."""
        ticker, filing_date = ticker_info
        end_date = min(filing_date + timedelta(days=self.timeout), datetime.now())
        return ticker, download_daily_stock_data(ticker, filing_date, end_date)

    def analyze_stock_histories(self):
        """Analyze stock histories and plot the required metrics."""
        alpha_data = {}
        return_data = {}

        for ticker, stock_data in self.stock_data_dict.items():
            stock_returns = stock_data['Close'].pct_change().cumsum()
            spy_returns = self.spy_data['Close'].pct_change().cumsum()
            alpha = calculate_alpha(stock_returns, spy_returns)

            alpha_data[ticker] = alpha
            return_data[ticker] = stock_returns

        # Plot all stock histories
        plot_stock_histories(self.stock_data_dict, self.output_dir)

        # Plot median and mean alpha over time
        plot_median_mean_alpha(alpha_data, self.output_dir)

        # Plot median and mean return over time
        plot_median_mean_return(return_data, self.output_dir)

    def save_final_metrics(self):
        """Save final metrics for each strategy in an Excel sheet."""
        final_metrics = {}

        for limit in self.limit_array:
            for stop in self.stop_array:
                strategy_name = f'lim_{limit}_stop_{stop}'
                final_metrics[strategy_name] = self.calculate_final_strategy_metrics(limit, stop)

        final_metrics_df = pd.DataFrame(final_metrics)
        final_metrics_df.to_excel(os.path.join(self.output_dir, 'final_strategy_metrics.xlsx'))

    def calculate_final_strategy_metrics(self, limit, stop):
        """Calculate final metrics for a given strategy (Limit, Stop, Timeout)."""
        # Implement logic to calculate final metrics like Median Return, Quartiles Return, etc.
        # For each strategy and populate the metrics dictionary accordingly
        metrics = {}
        return metrics

    def run(self):
        """Run the full stock history analysis."""
        self.load_features()
        self.download_all_stock_data()
        self.analyze_stock_histories()
        self.save_final_metrics()
        print("Stock history analysis completed. Check output directory for plots and metrics.")

if __name__ == "__main__":
    analyzer = StockHistoryAnalyzer(timeout=30, limit_array=[0.1, 0.2], stop_array=[-0.05, -0.1])
    analyzer.run()
