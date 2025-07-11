import os
import time
from datetime import timedelta
# --- FIX: Use the new, more robust helper function ---
from .utils.stock_analysis_helpers import (
    load_stock_data, 
    filter_and_align_data, 
    save_summary_statistics, 
    plot_combined
)

class StockAnalysis:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.stock_returns_file = os.path.join(data_dir, 'final/stock_data_final.xlsx')
        self.output_dir = os.path.join(data_dir, 'analysis/stock_analysis/all')
        self.return_df = None
        self.alpha_df = None

    def run(self):
        start_time = time.time()
        print("\n### START ### Stock Analysis")

        # Load data
        self.return_df = load_stock_data(self.stock_returns_file, sheet_name='Returns')
        self.alpha_df = load_stock_data(self.stock_returns_file, sheet_name='Alpha')

        # --- FIX: Filter both dataframes together to keep them aligned ---
        # The filter condition is calculated based on the return_df, and then
        # the same rows are kept for both dataframes, ensuring they stay in sync.
        # self.return_df, self.alpha_df = filter_and_align_data(
        #     self.return_df, [self.alpha_df], max_jump=0.1
        # )

        # Process the aligned return_df
        print("\n--- Processing Returns ---")
        save_summary_statistics(self.return_df, self.output_dir, 'stock_returns_summary_stats_return.xlsx')
        plot_combined(self.return_df, self.output_dir, '_return')

        # Process the aligned alpha_df
        print("\n--- Processing Alpha ---")
        save_summary_statistics(self.alpha_df, self.output_dir, 'stock_returns_summary_stats_alpha.xlsx')
        plot_combined(self.alpha_df, self.output_dir, '_alpha')

        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"\n### END ### Stock Analysis - time elapsed: {elapsed_time}")


if __name__ == "__main__":
    analysis = StockAnalysis()
    analysis.run()
