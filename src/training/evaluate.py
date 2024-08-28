import os
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # Import tqdm for the progress bar

class StockEvaluator:
    def __init__(self, model_name, criterion):
        self.model_name = model_name.replace(' ','-').lower()
        
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
        self.load_data()

    def load_data(self):
        """Load the predictions data and stock data."""
        self.predictions = {}
        
        # Iterate over all files in the predictions directory
        for file_name in os.listdir(self.predictions_dir):
            if file_name.endswith('.xlsx'):
                # Construct the full file path
                file_path = os.path.join(self.predictions_dir, file_name)
                
                # Load the file and store it in the predictions dictionary
                df = pd.read_excel(file_path)
                
                # Extract the limit and stop values from the filename
                limit_stop = file_name.replace(f'pred_{self.model_short}_l', '').replace('.xlsx', '')
                limit_value, stop_value = map(float, limit_stop.split('_s'))
                
                # Store the DataFrame in the dictionary with the limit and stop as the key
                self.predictions[(limit_value, stop_value)] = df

        # Load stock data
        self.stock_data = pd.read_excel(self.stock_data_file, sheet_name='Returns')

    def check_required_targets(self, df):
        """Check if required targets are in the columns of the given DataFrame."""
        required_targets = []

        if self.criterion in ['limit', 'limit-stop']:
            required_targets.append('GT_limit-occurred-first')
        if self.criterion in ['stop', 'limit-stop']:
            required_targets.append('GT_stop-occurred-first')
        if self.criterion in ['spike_up', 'spike_up-down']:
            required_targets.append('GT_spike-up')
        if self.criterion in ['spike_up-down']:
            required_targets.append('GT_spike-down')
        if self.criterion in ['pos_return', 'high_return']:
            required_targets.append('GT_return')

        return all(target in df.columns for target in required_targets)

    def apply_criterion(self, df):
        """Apply the specified criterion to filter the DataFrame."""
        if self.criterion == 'limit':
            return df[df['Pred_limit-occurred-first'] == 1]
        elif self.criterion == 'stop':
            return df[df['Pred_stop-occurred-first'] == 0]
        elif self.criterion == 'limit-stop':
            return df[(df['Pred_limit-occurred-first'] == 1) & (df['Pred_stop-occurred-first'] == 0)]
        elif self.criterion == 'spike_up':
            return df[df['Pred_spike-up'] == 1]
        elif self.criterion == 'spike_up-down':
            return df[(df['Pred_spike-up'] == 1) & (df['Pred_spike-down'] == 0)]
        elif self.criterion == 'pos_return':
            return df[df['Pred_return'] > 0]
        elif self.criterion == 'high_return':
            return df[df['Pred_return'] > 0.04]
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def simulate_buying(self, args):
        """Simulate buying stocks based on filtered DataFrame."""
        limit_value, stop_value, filtered_df = args
        tickers = filtered_df['Ticker'].unique()
        simulation_results = []

        for ticker in tickers:
            ticker_data = self.stock_data[self.stock_data['Ticker'] == ticker].iloc[:, 2:22]
            for day in range(ticker_data.shape[1]):
                if ticker_data.iloc[0, day] >= limit_value:
                    selling_day = day + 1
                    break
                elif ticker_data.iloc[0, day] <= stop_value:
                    selling_day = day + 1
                    break
            else:
                selling_day = 20  # No limit or stop hit, sell on the last day

            filing_date = filtered_df[filtered_df['Ticker'] == ticker]['Filing Date'].values[0]
            simulation_results.append({'Ticker': ticker, 'Filing Date': filing_date, 'Selling Day': selling_day})

        return pd.DataFrame(simulation_results), limit_value, stop_value

    def save_simulation(self, simulation_df, limit_value, stop_value):
        """Save the simulation results to an Excel file."""
        # Creating subdirectory for the model type
        model_output_dir = os.path.join(self.simulation_output_dir, self.model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Determine the sheet name based on the criterion and model
        
        if self.criterion == 'limit': criterion_short = 'l'
        elif self.criterion == 'limit-stop': criterion_short = 'l-s'
        elif self.criterion == 'stop': criterion_short = 's'
        elif self.criterion == 'spike_up': criterion_short = 'su'
        elif self.criterion == 'spike_down': criterion_short = 'sd'
        elif self.criterion == 'spike_up-down': criterion_short = 'sud'
        elif self.criterion == 'pos_return': criterion_short = 'pr'
        elif self.criterion == 'high_return': criterion_short = 'hr'
        
        sheet_name = f'l_{limit_value}_s_{stop_value}_{self.model_short}_{criterion_short}'

        # Naming the output file based on the limit and stop values
        output_file = os.path.join(model_output_dir, f'sim_{self.model_short}_l{limit_value}_s{stop_value}.xlsx')
        
        # Save the simulation DataFrame to an Excel file
        simulation_df.to_excel(output_file, sheet_name=sheet_name, index=False)
        # print(f"Simulation results saved to {output_file}")

    def run_simulation(self, limit_stop_pair):
        """Run the simulation for a single limit-stop pair."""
        limit_value, stop_value, df = limit_stop_pair
        if not self.check_required_targets(df):
            print(f"Skipping l{limit_value}_s{stop_value} due to missing required targets.")
            return

        filtered_df = self.apply_criterion(df)
        if not filtered_df.empty:
            simulation_df, limit_value, stop_value = self.simulate_buying((limit_value, stop_value, filtered_df))
            self.save_simulation(simulation_df, limit_value, stop_value)

    def run_evaluation(self):
        """Run the full evaluation and simulation process in parallel."""
        limit_stop_pairs = [(limit_value, stop_value, df) for (limit_value, stop_value), df in self.predictions.items()]
        
        # Use multiprocessing Pool to parallelize the simulation process
        with Pool(cpu_count()) as pool:
            # Add tqdm to the pool's map function for a progress bar
            list(tqdm(pool.imap(self.run_simulation, limit_stop_pairs), total=len(limit_stop_pairs), desc="Running Simulations"))
        print(f"Simulation results saved to {self.simulation_output_dir}")

if __name__ == "__main__":
    evaluator = StockEvaluator(model_name="randomforest", criterion="limit-stop")
    evaluator.run_evaluation()
