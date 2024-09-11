# backtest_helpers.py
import numpy as np
import os
import pandas as pd

def gather_prediction_gt_pairs(predictions_dir):
    """Gather all prediction-gt column pairs from prediction sheets."""
    prediction_files = []
    for root, _, files in os.walk(predictions_dir):
        for file in files:
            if file.endswith('.xlsx'):
                prediction_files.append(os.path.join(root, file))

    prediction_gt_pairs = []
    for prediction_file in prediction_files:
        model_name = os.path.basename(os.path.dirname(prediction_file))
        df = pd.read_excel(prediction_file)
        columns = df.columns
        pred_gt_pairs = [(pred_col, gt_col) for pred_col in columns if 'Pred_' in pred_col for gt_col in columns if f'GT_{pred_col.split("Pred_")[1]}' == gt_col]
        for pair in pred_gt_pairs:
            target_name = pair[0].split('_')[1]
            prediction_gt_pairs.append((df, model_name, target_name, pair))

    return prediction_gt_pairs

def load_stock_data(stock_data_file):
    """Load the stock data from the 'Returns' and 'Alpha' sheets."""
    stock_data = {}
    stock_data['Returns'] = pd.read_excel(stock_data_file, sheet_name="Returns")
    stock_data['Alpha'] = pd.read_excel(stock_data_file, sheet_name="Alpha")
    return stock_data

def is_binary(series):
    """Check if a column is already binary."""
    unique_values = series.dropna().unique()
    return set(unique_values).issubset({0, 1})

def threshold_binary(data, threshold=0):
    """Convert floats to binary based on a threshold."""
    return (data >= threshold).astype(int)

def calculate_limstop_value(data, limit, stop):
    """
    Simulate the limit/stop behavior for a single row.
    Return the first value where the limit or stop was triggered.
    If neither is triggered, return the Day 20 value.
    """
    for i, value in enumerate(data):
        if value >= limit:
            return value  # Limit hit
        elif value <= stop:
            return value  # Stop hit
    # If neither limit/stop is triggered, return Day 20 value
    return data.iloc[-1]

def extract_mean_return_alpha(tickers, stock_data, day, limit=None, stop=None):
    """
    Extract the raw mean and limstop mean return/alpha for all tickers.
    If limit/stop are provided, compute the limstop mean, otherwise raw mean.
    """
    returns_df = stock_data['Returns']
    alpha_df = stock_data['Alpha']
    
    mean_return_values = []
    mean_alpha_values = []

    for ticker in tickers:
        ticker_returns = returns_df[returns_df['Ticker'] == ticker]
        ticker_alpha = alpha_df[alpha_df['Ticker'] == ticker]

        if ticker_returns.empty or ticker_alpha.empty:
            continue

        # Raw mean for Day {day} (e.g., Day 20)
        raw_return = ticker_returns[f'Day {day} Stock'].values[0]
        raw_alpha = ticker_alpha[f'Day {day} Alpha'].values[0]

        mean_return_values.append(raw_return)
        mean_alpha_values.append(raw_alpha)

        # If we're computing limstop, simulate the limit/stop behavior
        if limit is not None and stop is not None:
            # Simulate limit/stop behavior across all days up to Day 20
            simulated_return = calculate_limstop_value(ticker_returns.iloc[1:day+1, 1:], limit, stop)
            simulated_alpha = calculate_limstop_value(ticker_alpha.iloc[1:day+1, 1:], limit, stop)

            mean_return_values[-1] = simulated_return
            mean_alpha_values[-1] = simulated_alpha

    # Compute the mean of all the return/alpha values
    return np.mean(mean_return_values), np.mean(mean_alpha_values)

def process_prediction_pair(args):
    """Process a single prediction-gt pair in parallel."""
    df, model_name, target_name, column_pair, limit_array, stop_array, stock_data = args
    pred_col, gt_col = column_pair
    results = []

    # Ensure both prediction and GT columns are binary
    if not is_binary(df[pred_col]):
        df[pred_col] = threshold_binary(df[pred_col])
    if not is_binary(df[gt_col]):
        df[gt_col] = threshold_binary(df[gt_col])

    tickers = df['Ticker'].unique()

    for limit_val, stop_val in zip(limit_array, stop_array):
        # Filter the dataframe for Pred = 1 and GT = 1 for respective calculations
        pred_filtered_df = df[df[pred_col] == 1]
        gt_filtered_df = df[df[gt_col] == 1]

        # Compute raw means (Day 20 values) for prediction (filtered by Pred = 1)
        raw_mean_return_1m, raw_mean_alpha_1m               = extract_mean_return_alpha(pred_filtered_df['Ticker'], stock_data, day=20)
        
        # Compute limstop means (using limit and stop, filtered by Pred = 1)
        limstop_mean_return_1m, limstop_mean_alpha_1m       = extract_mean_return_alpha(pred_filtered_df['Ticker'], stock_data, day=20, limit=limit_val, stop=stop_val)

        # Compute raw means for GT (filtered by GT = 1)
        raw_gt_mean_return_1m, raw_gt_mean_alpha_1m         = extract_mean_return_alpha(gt_filtered_df['Ticker'], stock_data, day=20)

        # Compute limstop means for GT (using limit and stop, filtered by GT = 1)
        limstop_gt_mean_return_1m, limstop_gt_mean_alpha_1m = extract_mean_return_alpha(gt_filtered_df['Ticker'], stock_data, day=20, limit=limit_val, stop=stop_val)

        result = {
            'Model': model_name,
            'Prediction': target_name,
            'Limit': limit_val,
            'Stop': stop_val,
            'raw pred mean return 1m': raw_mean_return_1m,
            'raw pred mean alpha 1m': raw_mean_alpha_1m,
            'limstop pred mean return 1m': limstop_mean_return_1m,
            'limstop pred mean alpha 1m': limstop_mean_alpha_1m,
            'raw GT mean return 1m': raw_gt_mean_return_1m,
            'raw GT mean alpha 1m': raw_gt_mean_alpha_1m,
            'limstop GT mean return 1m': limstop_gt_mean_return_1m,
            'limstop GT mean alpha 1m': limstop_gt_mean_alpha_1m
        }
        
        results.append(result)

    return results

def save_results(all_results, output_file):
    """Save all the backtest results into a single Excel file."""
    final_df = pd.DataFrame(all_results)
    final_df.to_excel(output_file, index=False)
    print(f"- Backtest results saved to {output_file}")
