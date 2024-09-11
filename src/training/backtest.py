from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from .utils.backtest_helpers import load_stock_data, gather_prediction_gt_pairs, process_prediction_pair, save_results

class Backtester:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.predictions_dir = os.path.join(data_dir, "training/predictions")
        self.stock_data_file = os.path.join(data_dir, "final/stock_data.xlsx")  # Stock data with returns and alpha
        self.output_file = os.path.join(data_dir, "backtest/backtesting_results.xlsx")
        self.limit_array = []
        self.stop_array = []
        self.stock_data = None

    def process_predictions_parallel(self, prediction_gt_pairs, limit_array, stop_array):
        """Process predictions and GT in parallel using multiprocessing."""
        all_results = []
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(process_prediction_pair, (df, model_name, target_name, pair, self.limit_array, self.stop_array, self.stock_data))
                for df, model_name, target_name, pair in prediction_gt_pairs
            ]
            for future in tqdm(futures, desc="Processing backtest results"):
                all_results.extend(future.result())
        return all_results

    def run(self, limit_array, stop_array):
        """Run the full backtesting process."""
        # 1. Load stock data and prediction pairs
        self.stock_data = load_stock_data(self.stock_data_file)
        self.limit_array = limit_array
        self.stop_array = stop_array
        prediction_gt_pairs = gather_prediction_gt_pairs(self.predictions_dir)

        # 2. Process each prediction-gt pair in parallel
        all_results = self.process_predictions_parallel(prediction_gt_pairs)

        # 3. Save the final results
        save_results(all_results, self.output_file)
