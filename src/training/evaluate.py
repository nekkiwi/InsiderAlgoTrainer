import os
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

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
        
        for file_name in os.listdir(self.predictions_dir):
            if file_name.endswith('.xlsx'):
                file_path = os.path.join(self.predictions_dir, file_name)
                df = pd.read_excel(file_path)
                limit_stop = file_name.replace(f'pred_{self.model_short}_l', '').replace('.xlsx', '')
                limit_value, stop_value = map(float, limit_stop.split('_s'))
                self.predictions[(limit_value, stop_value)] = df

        self.stock_data = pd.read_excel(self.stock_data_file, sheet_name='Returns')

    def check_required_targets(self, df):
        """Check if required targets are in the columns of the given DataFrame."""
        required_targets = []

        if self.criterion in ['limit', 'limit-stop']:
            required_targets.append('GT_limit-occurred-first')
        if self.criterion in ['stop', 'limit-stop']:
            required_targets.append('GT_stop-occurred-first')
        if self.criterion in ['spike-up', 'spike-up-down']:
            required_targets.append('GT_spike-up')
        if self.criterion in ['spike-up-down']:
            required_targets.append('GT_spike-down')
        if self.criterion in ['pos-return']:
            required_targets.append('GT_pos-return')
        if self.criterion in ['high-return']:
            required_targets.append('GT_high-return')

        return all(target in df.columns for target in required_targets)

    def determine_criterion_pass(self, df, type="Pred"):
        """Determine whether each ticker passes the criterion."""
        if self.criterion == 'limit':
            return (df[type+'_limit-occurred-first'] == 1).astype(int)
        elif self.criterion == 'stop':
            return (df[type+'_stop-occurred-first'] == 0).astype(int)
        elif self.criterion == 'limit-stop':
            return ((df[type+'_limit-occurred-first'] == 1) & (df[type+'_stop-occurred-first'] == 0)).astype(int)
        elif self.criterion == 'spike-up':
            return (df[type+'_spike-up'] == 1).astype(int)
        elif self.criterion == 'spike-up-down':
            return ((df[type+'_spike-up'] == 1) & (df[type+'_spike-down'] == 0)).astype(int)
        elif self.criterion == 'pos-return':
            return (df[type+'_pos-return'] > 0).astype(int)
        elif self.criterion == 'high-return':
            return (df[type+'_high-return'] > 0.04).astype(int)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def simulate_buying(self, args):
        """Simulate buying stocks based on all ticker and filing date combinations and include criterion pass."""
        limit_value, stop_value, df = args
        simulation_results = []

        for _, row in df.iterrows():
            ticker = row['Ticker']
            filing_date = row['Filing Date']

            # Extract data for this specific ticker and filing date
            ticker_data = self.stock_data[(self.stock_data['Ticker'] == ticker) & (self.stock_data['Filing Date'] == filing_date)]
            
            if ticker_data.empty:
                print(f"No stock data available for ticker {ticker} on filing date {filing_date}. Skipping.")
                continue
            
            ticker_data = ticker_data.iloc[:, 2:22]  # Assuming the relevant data is in columns 2 to 22
            limit_day, stop_day = None, None
            limit_price, stop_price = None, None

            for day in range(ticker_data.shape[1]):
                price = ticker_data.iloc[0, day]
                if limit_day is None and price >= limit_value:
                    limit_day = day + 1
                    limit_price = price
                if stop_day is None and price <= stop_value:
                    stop_day = day + 1
                    stop_price = price

                if limit_day is not None or stop_day is not None:
                    break

            selling_day = min(limit_day or 21, stop_day or 21) if limit_day or stop_day else 20
            pred_criterion_pass = row['pred-criterion-pass']
            gt_criterion_pass = row['gt-criterion-pass']

            simulation_results.append({
                'Ticker': ticker,
                'Filing Date': filing_date,
                'Selling Day': selling_day,
                'Limit Day': limit_day,
                'Stop Day': stop_day,
                'Limit Price': limit_price,
                'Stop Price': stop_price,
                'Pred Criterion Pass': pred_criterion_pass,
                'GT Criterion Pass': gt_criterion_pass
            })

        if not simulation_results:
            print(f"No valid simulation results for limit {limit_value} and stop {stop_value}.")
            return pd.DataFrame(), limit_value, stop_value

        return pd.DataFrame(simulation_results), limit_value, stop_value


    def save_simulation(self, simulation_df, limit_value, stop_value):
        """Save the simulation results to an Excel file."""
        model_output_dir = os.path.join(self.simulation_output_dir, self.model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        if self.criterion == 'limit': criterion_short = 'l'
        elif self.criterion == 'limit-stop': criterion_short = 'l-s'
        elif self.criterion == 'stop': criterion_short = 's'
        elif self.criterion == 'spike-up': criterion_short = 'su'
        elif self.criterion == 'spike-down': criterion_short = 'sd'
        elif self.criterion == 'spike-up-down': criterion_short = 'sud'
        elif self.criterion == 'pos-return': criterion_short = 'pr'
        elif self.criterion == 'high-return': criterion_short = 'hr'
        
        sheet_name = f'l_{limit_value}_s_{stop_value}_{self.model_short}_{criterion_short}'
        output_file = os.path.join(model_output_dir, f'sim_{self.model_short}_l{limit_value}_s{stop_value}.xlsx')
        simulation_df.to_excel(output_file, sheet_name=sheet_name, index=False)
        # print(f"Simulation results saved to {output_file}")

    def run_simulation(self, limit_stop_pair):
        """Run the simulation for a single limit-stop pair."""
        limit_value, stop_value, df = limit_stop_pair
        df.dropna(inplace=True)
        if not self.check_required_targets(df):
            print(f"Skipping l{limit_value}_s{stop_value} due to missing required targets.")
            return

        df['pred-criterion-pass'] = self.determine_criterion_pass(df, "Pred")
        df['gt-criterion-pass'] = self.determine_criterion_pass(df, "GT")
        simulation_df, limit_value, stop_value = self.simulate_buying((limit_value, stop_value, df))
        self.save_simulation(simulation_df, limit_value, stop_value)

    def run_evaluation(self):
        """Run the full evaluation and simulation process in parallel."""
        limit_stop_pairs = [(limit_value, stop_value, df) for (limit_value, stop_value), df in self.predictions.items()]
        
        with Pool(cpu_count()) as pool:
            list(tqdm(pool.imap(self.run_simulation, limit_stop_pairs), total=len(limit_stop_pairs), desc="Running Simulations"))
        print(f"Simulation results saved to {self.simulation_output_dir}")

if __name__ == "__main__":
    evaluator = StockEvaluator(model_name="randomforest", criterion="limit-stop")
    evaluator.run_evaluation()
