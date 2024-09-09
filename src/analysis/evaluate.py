import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from .utils.evaluate_helpers import *

class StockEvaluator:
    def __init__(self, model_name, criterion):
        self.model_name = model_name.replace(' ', '-').lower()
        
        if self.model_name == 'randomforest': self.model_short = 'rf'
        elif self.model_name == 'naivesbayes': self.model_short = 'nb'
        elif self.model_name == 'rbf-svm': self.model_short = 'svm'
        elif self.model_name == 'gaussian-process': self.model_short = 'gp'
        elif self.model_name == 'neural-net': self.model_short = 'nn'
        
        self.criterion = criterion
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.predictions_dir = os.path.join(data_dir, 'training/predictions', self.model_name)
        self.stock_data_file = os.path.join(data_dir, 'final/stock_data_final.xlsx')
        self.simulation_output_dir = os.path.join(data_dir, 'training/simulation')
        
        self.predictions = load_predictions(self.predictions_dir, self.model_short)
        self.stock_data = load_stock_data(self.stock_data_file)

    def run_simulation(self, limit_stop_pair):
        """Run the simulation for a single limit-stop pair."""
        limit_value, stop_value, df = limit_stop_pair
        df.dropna(inplace=True)
        if not check_required_targets(df, self.criterion):
            print(f"Skipping l{limit_value}_s{stop_value} due to missing required targets.")
            return

        df['pred-criterion-pass'] = determine_criterion_pass(df, self.criterion, "Pred")
        df['gt-criterion-pass'] = determine_criterion_pass(df, self.criterion, "GT")
        simulation_df, limit_value, stop_value = simulate_buying((limit_value, stop_value, df), self.stock_data, self.criterion)
        save_simulation(simulation_df, limit_value, stop_value, self.model_name, self.model_short, self.criterion, self.simulation_output_dir)

    def run_evaluation(self):
        """Run the full evaluation and simulation process in parallel."""
        limit_stop_pairs = [(limit_value, stop_value, df) for (limit_value, stop_value), df in self.predictions.items()]
        
        with Pool(cpu_count()) as pool:
            list(tqdm(pool.imap(self.run_simulation, limit_stop_pairs), total=len(limit_stop_pairs), desc="Running Simulations"))
        print(f"Simulation results saved to {self.simulation_output_dir}")

if __name__ == "__main__":
    evaluator = StockEvaluator(model_name="randomforest", criterion="limit-stop")
    evaluator.run_evaluation()
