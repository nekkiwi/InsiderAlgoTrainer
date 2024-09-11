from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
from datetime import timedelta
import os
from .utils.backtest_helpers import load_stock_data, gather_prediction_gt_pairs, process_prediction_pair, save_results

class Backtester:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.predictions_dir = os.path.join(data_dir, "training/predictions")
        self.stock_data_file = os.path.join(data_dir, "final/stock_data_final.xlsx")  # Stock data with returns and alpha
        self.output_file = os.path.join(data_dir, "backtest/backtesting_results.xlsx")
        self.limit_array = []
        self.stop_array = []
        self.stock_data = None

    def process_predictions_parallel(self, prediction_gt_pairs):
        """Process predictions and GT in parallel using multiprocessing."""
        # Prepare arguments for multiprocessing
        args_list = [
            (df, model_name, target_name, pair, self.limit_array, self.stop_array, self.stock_data)
            for df, model_name, target_name, pair in prediction_gt_pairs]

        all_results = []
        # Using multiprocessing Pool
        with Pool(cpu_count()) as pool:
            for result in tqdm(pool.imap_unordered(process_prediction_pair, args_list), total=len(args_list), desc="- Processing backtest results"):
                all_results.extend(result)
        
        return all_results

    def run(self, limit_array, stop_array):
        """Run the full backtesting process."""
        start_time = time.time()
        print(f"\n### START ### Backtesting")
        # 1. Load stock data and prediction pairs
        print("- Loading stock data")
        self.stock_data = load_stock_data(self.stock_data_file)
        self.limit_array = limit_array
        self.stop_array = stop_array
        prediction_gt_pairs = gather_prediction_gt_pairs(self.predictions_dir)

        # 2. Process each prediction-gt pair in parallel
        all_results = self.process_predictions_parallel(prediction_gt_pairs)

        # 3. Save the final results
        save_results(all_results, self.output_file)
        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"### END ### Backtesting - time elapsed: {elapsed_time}")
