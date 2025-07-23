import os
import time
from datetime import timedelta
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
        self.sheets = [  # Update this if you add more OHLC types.
            "Return_Close", "Return_Open", "Return_High", "Return_Low",
            "Alpha_Close", "Alpha_Open", "Alpha_High", "Alpha_Low"
        ]

    def run(self):
        start_time = time.time()
        print("\n### START ### Stock Analysis")

        for sheet in self.sheets:
            try:
                df = load_stock_data(self.stock_returns_file, sheet_name=sheet)
                print(f"\n--- Processing {sheet} ---")
                save_summary_statistics(
                    df, self.output_dir, f'{sheet.lower()}_summary_stats.xlsx'
                )
                plot_combined(df, self.output_dir, f'_{sheet.lower()}')
            except Exception as e:
                print(f"[WARN] Could not process {sheet}: {e}")

        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"\n### END ### Stock Analysis - time elapsed: {elapsed_time}")