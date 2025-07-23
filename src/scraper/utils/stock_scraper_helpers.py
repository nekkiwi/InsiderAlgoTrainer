import os
import pandas as pd
import yfinance as yf
import contextlib
from datetime import timedelta
from pandas.tseries.offsets import BDay
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import re

def convert_timepoints_to_bdays(timepoints: list) -> dict:
    """
    Converts a list of timepoint strings (e.g., '1w', '2m') into a
    dictionary mapping the timepoint to its equivalent number of business days.
    - 1 week ('w') = 5 business days
    - 1 month ('m') = 20 business days
    """
    converted = {}
    for tp in timepoints:
        match = re.match(r"(\d+)([dwmy])", tp.lower())
        if not match:
            raise ValueError(f"Invalid timepoint format: '{tp}'. Use formats like '5d', '1w', '3m'.")
        
        num, unit = int(match.group(1)), match.group(2)
        
        if unit == 'd':
            converted[tp] = num
        elif unit == 'w':
            converted[tp] = num * 5
        elif unit == 'm':
            converted[tp] = num * 20  # Approximate business days in a month
        elif unit == 'y':
            converted[tp] = num * 240 # Approximate business days in a year (12 * 20)
        else:
            raise ValueError(f"Unknown time unit: '{unit}' in timepoint '{tp}'")
            
    return converted

def download_daily_stock_data(ticker: str, filing_date, n_days: int):
    """
    Robustly downloads exactly 'n_days' of daily stock data starting from a filing date.
    It fetches a wider date range to ensure enough trading days are captured,
    then trims to the exact required number.
    """
    start_date = pd.to_datetime(filing_date)
    
    # Estimate a calendar date range that is wide enough to capture n_days of trading.
    # A 1.7x multiplier is a safe buffer for weekends and holidays.
    end_date_estimate = start_date + timedelta(days=int(n_days * 1.7) + 5)
    
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date_estimate, progress=False)
            
            if stock_data.empty:
                return None
                
            # The crucial step: return exactly the first 'n_days' of data.
            # If the stock delisted and has fewer than n_days, this will return all available data.
            return stock_data.head(n_days)
            
        except Exception:
            return None

def download_data_wrapper(info: tuple):
    ticker, filing_date, max_days = info
    stock_data = download_daily_stock_data(ticker, filing_date, max_days)
    spy_data = download_daily_stock_data('SPY', filing_date, max_days)
    return (ticker, filing_date, max_days, stock_data, spy_data)

def save_formatted_sheets(all_targets_df: pd.DataFrame, max_days: int, output_file: str):
    if all_targets_df.empty: return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    all_targets_df['Filing Date'] = pd.to_datetime(all_targets_df['Filing Date']).dt.strftime('%d/%m/%Y %H:%M')
    alpha_cols, alpha_rename = ['Ticker', 'Filing Date'], {}
    for i in range(1, max_days + 1):
        c = f'alpha_day_{i}'; alpha_cols.append(c); alpha_rename[c] = f'Day {i} Alpha'
    alpha_df = all_targets_df[alpha_cols].rename(columns=alpha_rename)
    return_cols, return_rename = ['Ticker', 'Filing Date'], {}
    for i in range(1, max_days + 1):
        c = f'return_day_{i}'; return_cols.append(c); return_rename[c] = f'Day {i} Stock'
    returns_df = all_targets_df[return_cols].rename(columns=return_rename)
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        returns_df.to_excel(writer, sheet_name='Returns', index=False)
        alpha_df.to_excel(writer, sheet_name='Alpha', index=False)
    print(f"- Formatted stock data successfully saved to {output_file}")

def save_final_features(features_df: pd.DataFrame, features_out_file: str):
    os.makedirs(os.path.dirname(features_out_file), exist_ok=True)
    features_df['Filing Date'] = pd.to_datetime(features_df['Filing Date']).dt.strftime('%d-%m-%Y %H:%M:%S')
    features_df.to_excel(features_out_file, index=False)
    print(f"- Final filtered features saved to {features_out_file}.")

# --- ADDED: Functions migrated and adapted from TargetScraper ---

def _calculate_all_final_targets(stock_return_data, stock_alpha_data, timepoints):
    """Calculates the final raw return and alpha for each specified timepoint."""
    targets = {}
    num_days_available = len(stock_return_data)
    
    # `num_bdays` is the target day (e.g., 20 for the 20th business day)
    for name, num_bdays in timepoints.items():
        # To get the data for the Nth day from a 0-indexed series, we need index N-1
        target_index = num_bdays - 1
        
        # Check if the calculated index is valid and within the bounds of our data
        if target_index >= 0 and target_index < num_days_available:
            targets[f'return_{name}_raw'] = stock_return_data.iloc[target_index]
            targets[f'alpha_{name}_raw'] = stock_alpha_data.iloc[target_index]
        else:
            # If data is not available (e.g., not enough trading days), fill with None
            targets[f'return_{name}_raw'], targets[f'alpha_{name}_raw'] = None, None
            
    return targets

def _process_final_ticker_targets(args):
    ticker, filing_date, return_df, alpha_df, timepoints = args
    return_row = return_df.loc[(return_df['Ticker'] == ticker) & (return_df['Filing Date'] == filing_date)]
    alpha_row = alpha_df.loc[(alpha_df['Ticker'] == ticker) & (alpha_df['Filing Date'] == filing_date)]
    if return_row.empty or alpha_row.empty: return (ticker, filing_date), None
    stock_return_data, stock_alpha_data = return_row.iloc[0, 2:].dropna(), alpha_row.iloc[0, 2:].dropna()
    if not stock_return_data.empty and not stock_alpha_data.empty:
        processed_data = _calculate_all_final_targets(stock_return_data, stock_alpha_data, timepoints)
        return (ticker, filing_date), processed_data
    return (ticker, filing_date), None

def _save_final_targets_distribution(results, dist_out_file):
    if not results: return
    df = pd.DataFrame([{'Ticker': k[0], 'Filing Date': k[1], **v} for k, v in results.items()])
    distribution_data = []
    for col in [c for c in df.columns if c not in ['Ticker', 'Filing Date']]:
        series = df[col].dropna()
        if series.empty: continue
        desc = series.describe(percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99])
        row_data = {'Target': col, 'mean': desc['mean']}
        if 'binary' not in col:
            row_data.update({k: f"{v:.4f}" for k, v in desc.items() if k != 'count'})
        distribution_data.append(row_data)
    pd.DataFrame(distribution_data).to_excel(dist_out_file, index=False)
    print(f"- Final targets distribution successfully saved to {dist_out_file}.")

def _save_final_targets_sheets(results, output_file):
    if not results: return
    full_df = pd.DataFrame([{'Ticker': k[0], 'Filing Date': k[1], **v} for k, v in results.items()])
    with pd.ExcelWriter(output_file) as writer:
        for target_name in sorted([c for c in full_df.columns if c not in ['Ticker', 'Filing Date']]):
            target_df = full_df[['Ticker', 'Filing Date', target_name]].dropna()
            safe_sheet_name = target_name[:31]
            target_df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
    print(f"- Final targets data successfully saved to {output_file}.")

def generate_and_save_final_targets(stock_data_file, targets_output_file, distribution_output_file, timepoints):
    """
    Encapsulates the logic of the original TargetScraper to generate and save final targets.
    """
    print("- Loading formatted stock data for final target generation...")
    try:
        return_df = pd.read_excel(stock_data_file, sheet_name='Returns')
        alpha_df = pd.read_excel(stock_data_file, sheet_name='Alpha')
    except FileNotFoundError:
        print(f"[ERROR] Could not find input file: {stock_data_file}. Skipping final target generation.")
        return
        
    processing_args = [(t, fd, return_df, alpha_df, timepoints) for t, fd in return_df[['Ticker', 'Filing Date']].values]
    with Pool(cpu_count()) as pool:
        results_list = list(tqdm(pool.imap(_process_final_ticker_targets, processing_args), total=len(processing_args), desc="- Processing final targets"))
    results = {k: v for k, v in results_list if v is not None}
    if not results:
        print("[INFO] No final targets were generated. Aborting save operations.")
        return
    _save_final_targets_sheets(results, targets_output_file)
    _save_final_targets_distribution(results, distribution_output_file)
