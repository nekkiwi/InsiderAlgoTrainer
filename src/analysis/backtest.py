import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from .utils.backtest_helpers import load_stock_data, calculate_all_stock_stats, process_simulation, extract_limit_stop_from_filename

class Backtest:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.stock_returns_file = os.path.join(data_dir, 'final/stock_data_final.xlsx')
        self.simulations_dir = os.path.join(data_dir, 'training/simulation')
        self.backtest_dir = os.path.join(data_dir, 'backtest')
        self.return_df = None
        self.model_name = None
        self.all_summary_stats = None

    def run_all_simulations(self, model_name, criterion):
        """Run stock return analysis for all simulations of a specific model and generate a backtest report."""
        self.model_name = model_name.replace(' ', '-').lower()
        self.return_df = load_stock_data(self.stock_returns_file)

        model_dir = os.path.join(self.simulations_dir, self.model_name)
        if not os.path.exists(model_dir):
            print(f"No simulations found for model: {self.model_name}")
            return

        self.all_summary_stats = pd.read_excel(os.path.join(self.simulations_dir, 'all', 'stock_returns_summary_stats.xlsx'), index_col=0)
        all_stock_stats = calculate_all_stock_stats(self.all_summary_stats)

        simulation_files = [file for file in os.listdir(model_dir) if file.endswith('.xlsx')]
        pool_args = [(os.path.join(model_dir, file), *extract_limit_stop_from_filename(file), all_stock_stats) for file in simulation_files]

        with Pool(cpu_count()) as pool:
            backtest_results = list(tqdm(pool.imap(process_simulation, pool_args), total=len(simulation_files), desc="Processing simulations"))

        backtest_df = pd.DataFrame(backtest_results)
        os.makedirs(self.backtest_dir, exist_ok=True)
        backtest_file = os.path.join(self.backtest_dir, f'{self.model_name}_{criterion}_backtest.xlsx')
        backtest_df.to_excel(backtest_file, index=False)
        print(f"Backtest results saved to {backtest_file}")

if __name__ == "__main__":
    backtest = Backtest()
    backtest.run_all_simulations("randomforest", "limit-stop")
