import os
import time
from datetime import timedelta
from .utils.stock_analysis_helpers import load_stock_data, filter_jumps, save_summary_statistics, plot_combined

class StockAnalysis:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.stock_returns_file = os.path.join(data_dir, 'final/stock_data_final.xlsx')
        self.output_dir = os.path.join(data_dir, 'analysis/stock_analysis/all')
        self.return_df = None
        self.alpha_df = None

    def run(self, return_df=None, alpha_df=None):
        start_time = time.time()
        print("\n### START ### Stock Analysis")

        # Load data if not provided
        if return_df is None:
            self.return_df = load_stock_data(self.stock_returns_file, sheet_name='Returns')
        else:
            self.return_df = return_df
        
        if alpha_df is None:
            self.alpha_df = load_stock_data(self.stock_returns_file, sheet_name='Alpha')
        else:
            self.alpha_df = alpha_df

        # Process return_df
        self.return_df = filter_jumps(self.return_df)
        save_summary_statistics(self.return_df, self.output_dir, 'stock_returns_summary_stats_return.xlsx')
        plot_combined(self.return_df, self.output_dir, '_return')

        # Process alpha_df
        self.alpha_df = filter_jumps(self.alpha_df)
        save_summary_statistics(self.alpha_df, self.output_dir, 'stock_returns_summary_stats_alpha.xlsx')
        plot_combined(self.alpha_df, self.output_dir, '_alpha')

        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"### END ### Stock Analysis - time elapsed: {elapsed_time}")


if __name__ == "__main__":
    analysis = StockAnalysis()
    analysis.run()
