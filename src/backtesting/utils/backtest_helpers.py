import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_gt_pair(prediction_file):
    model_name = os.path.basename(os.path.dirname(prediction_file))
    df = pd.read_excel(prediction_file)
    columns = df.columns

    # Gather all pairs of prediction and ground truth columns
    pred_gt_pairs = [(pred_col, gt_col) for pred_col in columns if 'Pred_' in pred_col for gt_col in columns if f'GT_{pred_col.split("Pred_")[1]}' == gt_col]

    results = []
    for pair in pred_gt_pairs:
        target_name = pair[0].split('_')[1:]
        results.append((df, model_name, target_name, pair))

    return results

def is_binary(series):
    """Check if a column is already binary."""
    unique_values = series.dropna().unique()
    return set(unique_values).issubset({0, 1})

def threshold_binary(data, threshold=0):
    """Convert floats to binary based on a threshold."""
    return (data >= threshold).astype(int)


def gather_prediction_gt_pairs(predictions_dir):
    """Gather all prediction-gt column pairs from prediction sheets."""
    prediction_files = []
    for root, _, files in os.walk(predictions_dir):
        for file in files:
            if file.endswith('.xlsx'):
                prediction_files.append(os.path.join(root, file))

    prediction_gt_pairs = []
    
    # Using Pool for multiprocessing
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap(process_gt_pair, prediction_files), total=len(prediction_files), desc="- Gathering Pred-GT pairs"):
            prediction_gt_pairs.extend(result)

    return prediction_gt_pairs


def load_stock_data(stock_data_file):
    """
    Load the stock data from the 'Returns' and 'Alpha' sheets, with specific structure:
    1. First row as column titles.
    2. First column as tickers, second column as filing dates (both as indices).
    3. All other content converted to float.
    """
    # Load the Returns sheet
    returns_df = pd.read_excel(stock_data_file, sheet_name="Returns", index_col=[0, 1])
    
    # Convert all other columns (Day 1, Day 2, ..., Day 20) to float
    returns_df = returns_df.apply(pd.to_numeric, errors='coerce')
    
    # Load the Alpha sheet with the same structure
    alpha_df = pd.read_excel(stock_data_file, sheet_name="Alpha", index_col=[0, 1])
    
    # Convert all other columns to float
    alpha_df = alpha_df.apply(pd.to_numeric, errors='coerce')

    # Return as a dictionary for access in further operations
    stock_data = {
        'Returns': returns_df,
        'Alpha': alpha_df
    }
    
    return stock_data

def calculate_limstop_value(data, limit, stop, category):
    """
    Vectorized version to simulate limit/stop behavior.
    For each row, return the first value where limit or stop is triggered, else return the Day 20 value.
    """
    # Select numeric columns (assuming they are labeled 'Day X Stock')
    day_columns = [col for col in data.columns if col.startswith('Day')]
    numeric_data = data[day_columns].apply(pd.to_numeric, errors='coerce')  # Convert to numeric, coercing errors

    # Create boolean masks for where limit/stop is triggered
    limit_hit = (numeric_data >= limit).idxmax(axis=1)  # First occurrence where limit is hit
    stop_hit = (numeric_data <= stop).idxmax(axis=1)    # First occurrence where stop is hit

    # Ensure 'limit_hit' and 'stop_hit' are numeric
    limit_hit = pd.to_numeric(limit_hit, errors='coerce')
    stop_hit = pd.to_numeric(stop_hit, errors='coerce')

    # Create a dataframe to store when limit or stop occurs
    result = pd.DataFrame({'limit_hit': limit_hit, 'stop_hit': stop_hit})

    # Determine which comes first, limit or stop
    result['first_hit'] = result[['limit_hit', 'stop_hit']].min(axis=1, skipna=True)

    # If neither limit nor stop was hit before Day 20, use Day 20 value
    day_20_value = numeric_data[f'Day 20 {category}']

    # Handle the selection: if first hit occurs before Day 20, use that; otherwise, use Day 20
    condition_first_hit = result['first_hit'].notna() & (result['first_hit'] > 0) & (result['first_hit'] <= 20)  # A hit occurred before Day 20

    # Replacing the deprecated 'lookup' with apply and lambda to fetch the correct values
    return_data = pd.Series(
        np.where(
            condition_first_hit,
            result.apply(lambda row: numeric_data.loc[row.name, f'Day {int(row.first_hit)} {category}'] if pd.notna(row.first_hit) else day_20_value[row.name], axis=1),
            day_20_value
        )
    )

    return return_data




def extract_mean_return_alpha(df, stock_data, day=20, limit=None, stop=None):
    """
    Optimized version to extract raw mean and limstop mean return/alpha.
    """
    # Filter stock data based on tickers present in the input dataframe
    returns_df = stock_data['Returns']
    alpha_df = stock_data['Alpha']
    
    # Get raw values for Day 20
    raw_returns = returns_df[f'Day {day} Stock']
    raw_alpha = alpha_df[f'Day {day} Alpha']
    
    # If no limit or stop is provided, just return the raw means
    if limit is None or stop is None:
        return raw_returns.mean(), raw_alpha.mean()

    # Limit/Stop Simulation: Use the vectorized limit/stop calculation
    limstop_returns = calculate_limstop_value(returns_df, limit, stop, 'Stock')
    limstop_alpha = calculate_limstop_value(alpha_df, limit, stop, 'Alpha')

    return limstop_returns.mean(), limstop_alpha.mean()


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

    # For each limit/stop combination
    for limit_val, stop_val in zip(limit_array, stop_array):
        # Compute raw means for prediction (filtered by Pred = 1)
        raw_mean_return_1m, raw_mean_alpha_1m = extract_mean_return_alpha(df[df[pred_col] == 1], stock_data, day=20)
        
        # Compute limstop means for prediction (filtered by Pred = 1)
        limstop_mean_return_1m, limstop_mean_alpha_1m = extract_mean_return_alpha(df[df[pred_col] == 1], stock_data, day=20, limit=limit_val, stop=stop_val)

        # Compute raw means for GT (filtered by GT = 1)
        raw_gt_mean_return_1m, raw_gt_mean_alpha_1m = extract_mean_return_alpha(df[df[gt_col] == 1], stock_data, day=20)

        # Compute limstop means for GT (filtered by GT = 1)
        limstop_gt_mean_return_1m, limstop_gt_mean_alpha_1m = extract_mean_return_alpha(df[df[gt_col] == 1], stock_data, day=20, limit=limit_val, stop=stop_val)

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
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_excel(output_file, index=False)
    print(f"- Backtest results saved to {output_file}")
