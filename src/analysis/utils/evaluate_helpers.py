import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def load_predictions(predictions_dir, model_short):
    """Load the predictions data."""
    predictions = {}
    for file_name in os.listdir(predictions_dir):
        if file_name.endswith('.xlsx'):
            file_path = os.path.join(predictions_dir, file_name)
            df = pd.read_excel(file_path)
            limit_stop = file_name.replace(f'pred_{model_short}_l', '').replace('.xlsx', '')
            limit_value, stop_value = map(float, limit_stop.split('_s'))
            predictions[(limit_value, stop_value)] = df
    return predictions

def load_stock_data(stock_data_file):
    """Load stock data from Excel file."""
    return pd.read_excel(stock_data_file, sheet_name='Returns')

def check_required_targets(df, criterion):
    """Check if required targets are in the columns of the given DataFrame."""
    required_targets = []

    if criterion in ['limit', 'limit-stop']:
        required_targets.append('GT_limit-occurred-first')
    if criterion in ['stop', 'limit-stop']:
        required_targets.append('GT_stop-occurred-first')
    if criterion in ['spike-up', 'spike-up-down']:
        required_targets.append('GT_spike-up')
    if criterion in ['spike-up-down']:
        required_targets.append('GT_spike-down')
    if criterion in ['pos-return']:
        required_targets.append('GT_pos-return')
    if criterion in ['high-return']:
        required_targets.append('GT_high-return')

    return all(target in df.columns for target in required_targets)

def determine_criterion_pass(df, criterion, type="Pred"):
    """Determine whether each ticker passes the criterion."""
    if criterion == 'limit':
        return (df[type+'_limit-occurred-first'] == 1).astype(int)
    elif criterion == 'stop':
        return (df[type+'_stop-occurred-first'] == 0).astype(int)
    elif criterion == 'limit-stop':
        return ((df[type+'_limit-occurred-first'] == 1) & (df[type+'_stop-occurred-first'] == 0)).astype(int)
    elif criterion == 'spike-up':
        return (df[type+'_spike-up'] == 1).astype(int)
    elif criterion == 'spike-up-down':
        return ((df[type+'_spike-up'] == 1) & (df[type+'_spike-down'] == 0)).astype(int)
    elif criterion == 'pos-return':
        return (df[type+'_pos-return'] > 0).astype(int)
    elif criterion == 'high-return':
        return (df[type+'_high-return'] > 0.04).astype(int)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

def simulate_buying(args, stock_data, criterion):
    """Simulate buying stocks based on all ticker and filing date combinations and include criterion pass."""
    limit_value, stop_value, df = args
    simulation_results = []

    for _, row in df.iterrows():
        ticker = row['Ticker']
        filing_date = row['Filing Date']

        # Extract data for this specific ticker and filing date
        ticker_data = stock_data[(stock_data['Ticker'] == ticker) & (stock_data['Filing Date'] == filing_date)]
        
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

def save_simulation(simulation_df, limit_value, stop_value, model_name, model_short, criterion, simulation_output_dir):
    """Save the simulation results to an Excel file."""
    model_output_dir = os.path.join(simulation_output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    if criterion == 'limit': criterion_short = 'l'
    elif criterion == 'limit-stop': criterion_short = 'l-s'
    elif criterion == 'stop': criterion_short = 's'
    elif criterion == 'spike-up': criterion_short = 'su'
    elif criterion == 'spike-down': criterion_short = 'sd'
    elif criterion == 'spike-up-down': criterion_short = 'sud'
    elif criterion == 'pos-return': criterion_short = 'pr'
    elif criterion == 'high-return': criterion_short = 'hr'
    
    sheet_name = f'l_{limit_value}_s_{stop_value}_{model_short}_{criterion_short}'
    output_file = os.path.join(model_output_dir, f'sim_{model_short}_l{limit_value}_s{stop_value}.xlsx')
    simulation_df.to_excel(output_file, sheet_name=sheet_name, index=False)
