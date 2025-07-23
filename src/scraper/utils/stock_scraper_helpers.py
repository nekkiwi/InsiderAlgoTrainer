import os
import pandas as pd
import yfinance as yf
import contextlib
from datetime import timedelta
from pandas.tseries.offsets import BDay
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import re

import os
import pandas as pd
import yfinance as yf
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import re

def batch_fetch_stock_targets_ohlc(df, timepoints_bdays):
    print("[DEBUG] Entered batch_fetch_stock_targets_ohlc")
    max_days_needed = max(timepoints_bdays.values()) if timepoints_bdays else 0
    print(f"[DEBUG] max_days_needed: {max_days_needed}")
    if max_days_needed == 0:
        raise ValueError("Timepoints dictionary cannot be empty.")

    print("[DEBUG] Converting Filing Date to datetime...")
    df['Filing Date'] = pd.to_datetime(df['Filing Date'], dayfirst=True)
    unique_tickers = df['Ticker'].unique().tolist()
    tickers_plus_bench = unique_tickers + ['SPY']
    start_date = min(df['Filing Date'].min(), pd.Timestamp.today()) - pd.Timedelta(days=1)
    end_date = pd.to_datetime('today', dayfirst=True)
    print(f"[DEBUG] Downloading data from {start_date} to {end_date}...")

    all_data = yf.download(
        tickers_plus_bench, start=start_date, end=end_date,
        group_by='ticker', progress=True, threads=True, auto_adjust=True
    )
    # Repackage as dict
    data_dict = {}
    for t in unique_tickers:
        if t in all_data.columns.levels[0]:
            data_dict[t] = all_data[t][['Open','High','Low','Close']].copy()
        else:
            print(f"[WARN] Data for {t} not found in all_data.")
    if 'SPY' in all_data.columns.levels[0]:
        data_dict['SPY'] = all_data['SPY'][['Open','High','Low','Close']].copy()
    else:
        print("[WARN] Data for SPY not found in all_data.")

    args = [(row, data_dict, max_days_needed) for _, row in df.iterrows()]
    with Pool(cpu_count()//2) as pool:
        results = list(tqdm(pool.imap(process_single_target_ohlc, args), total=len(args), desc="- Calculating OHLC/Alpha"))

    print(f"[DEBUG] Number of process_single_target_ohlc non-None results: {sum(r is not None for r in results)}")
    results_flat = {
        k: [r[k] for r in results if r is not None and k in r]
        for k in ['Return_Close','Return_Open','Return_High','Return_Low','Alpha_Close','Alpha_Open','Alpha_High','Alpha_Low']
    }
    ohlc_targets_dict = {k: pd.DataFrame(v) for k, v in results_flat.items()}

    df_rc = ohlc_targets_dict['Return_Close']
    if df_rc.empty or not set(['Ticker','Filing Date']).issubset(df_rc.columns):
        print("[ERROR] No valid OHLC targets - all downloads failed or malformed.")
        return pd.DataFrame(), {k: pd.DataFrame() for k in ohlc_targets_dict}
    valid_keys = df_rc[['Ticker','Filing Date']]
    final_features_df = pd.merge(df, valid_keys, on=['Ticker','Filing Date'], how='inner')
    dropped = len(df) - len(final_features_df)
    print(f"[DEBUG] After merge, final_features_df: {final_features_df.shape}. Dropped {dropped} rows.")
    return final_features_df, ohlc_targets_dict

def process_single_target_ohlc(args):
    row, data_dict, max_days_needed = args
    ticker, filing_date = row['Ticker'], row['Filing Date']
    stock_df = data_dict.get(ticker)
    spy_df = data_dict.get('SPY')
    if stock_df is None or spy_df is None: 
        print(f"[WARN] {ticker}: Missing stock or SPY data!")
        return None
    stock_fw = stock_df.loc[stock_df.index >= filing_date].head(max_days_needed)
    spy_fw = spy_df.loc[spy_df.index >= filing_date].head(max_days_needed)
    if len(stock_fw) < 1 or len(spy_fw) < 1:
        print(f"[WARN] {ticker}: No forward data after {filing_date}.")
        return None
    entry_open = stock_fw['Open'].iloc[0]
    entry_close = stock_fw['Close'].iloc[0]
    # Next-day open (for Open returns)
    next_open = stock_fw['Open'].shift(-1)
    # RETURNS: as in analytics best practice
    ret_close = (stock_fw['Close'] / entry_close).values - 1
    ret_open  = (next_open / entry_open).values - 1
    ret_high  = (stock_fw['High'] / entry_open).values - 1
    ret_low   = (stock_fw['Low'] / entry_open).values - 1
    # Benchmark
    spy_entry_close = spy_fw['Close'].iloc[0]
    spy_entry_open = spy_fw['Open'].iloc[0]
    spy_next_open = spy_fw['Open'].shift(-1)
    alpha_close = ret_close - ((spy_fw['Close'] / spy_entry_close).values - 1)
    alpha_open = ret_open - ((spy_next_open / spy_entry_open).values - 1)
    alpha_high = ret_high - ((spy_fw['High'] / spy_entry_open).values - 1)
    alpha_low  = ret_low - ((spy_fw['Low'] / spy_entry_open).values - 1)
    # Package rows
    def to_row(vals, label):
        d = {'Ticker': ticker, 'Filing Date': filing_date}
        d.update({f'Day_{i}': v for i, v in enumerate(vals)})
        return d
    return {
        'Return_Close': to_row(ret_close, 'Return_Close'),
        'Return_Open':  to_row(ret_open,  'Return_Open'),
        'Return_High':  to_row(ret_high,  'Return_High'),
        'Return_Low':   to_row(ret_low,   'Return_Low'),
        'Alpha_Close':  to_row(alpha_close,'Alpha_Close'),
        'Alpha_Open':   to_row(alpha_open, 'Alpha_Open'),
        'Alpha_High':   to_row(alpha_high, 'Alpha_High'),
        'Alpha_Low':    to_row(alpha_low,  'Alpha_Low')
    }

def save_formatted_ohlc_sheets(ohlc_targets_dict, output_file):
    "Write each OHLC/Alpha DataFrame to a separate sheet"
    if all(len(df) == 0 for df in ohlc_targets_dict.values()): return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    for df in ohlc_targets_dict.values():
        df['Filing Date'] = pd.to_datetime(df['Filing Date']).dt.strftime('%d/%m/%Y %H:%M')
    with pd.ExcelWriter(output_file) as writer:
        for sheet, df in ohlc_targets_dict.items():
            # Excel sheet name safety
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
    print(f"- OHLC stock data written to {output_file}")


def download_daily_stock_data(ticker: str, filing_date, n_days: int):
    """
    Robustly downloads exactly 'n_days' of daily stock data starting from a filing date.
    It fetches a wider date range to ensure enough trading days are captured,
    then trims to the exact required number.
    """
    start_date = pd.to_datetime(filing_date, dayfirst=True)
    
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














import pandas as pd
import yfinance as yf
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import re

# This function is unchanged and correctly converts timepoints
def convert_timepoints_to_bdays(timepoints: list) -> dict:
    converted = {}
    for tp in timepoints:
        match = re.match(r"(\d+)([dwmy])", tp.lower())
        if not match: raise ValueError(f"Invalid timepoint format: '{tp}'")
        num, unit = int(match.group(1)), match.group(2)
        if unit == 'd': converted[tp] = num
        elif unit == 'w': converted[tp] = num * 5
        elif unit == 'm': converted[tp] = num * 20
        elif unit == 'y': converted[tp] = num * 240
        else: raise ValueError(f"Unknown time unit: '{unit}'")
    return converted

# --- NEW, OPTIMIZED WORKER FUNCTION ---
def _process_single_target(args):
    """
    Worker function that calculates returns and alphas for a single ticker-date
    using pre-fetched market data. IT PERFORMS NO NETWORK CALLS.
    """
    row, market_data, max_days_needed = args
    ticker, filing_date = row['Ticker'], row['Filing Date']

    # Lookup pre-fetched data
    stock_hist = market_data.get(ticker)
    spy_hist = market_data.get('SPY')

    if stock_hist is None or spy_hist is None:
        return None

    # Slice the data to get the forward-looking prices from the filing date
    stock_forward = stock_hist.loc[stock_hist.index >= filing_date].head(max_days_needed)
    spy_forward = spy_hist.loc[spy_hist.index >= filing_date].head(max_days_needed)

    if stock_forward.empty or spy_forward.empty or len(stock_forward) < 1 or len(spy_forward) < 1:
        return None

    # Calculate cumulative returns and alpha from the filing date
    stock_returns = (stock_forward['Close'] / stock_forward['Close'].iloc[0]) - 1
    spy_returns = (spy_forward['Close'] / spy_forward['Close'].iloc[0]) - 1
    alpha = stock_returns - spy_returns

    # Create the result row
    result_row = {'Ticker': ticker, 'Filing Date': filing_date}
    for i in range(max_days_needed):
        result_row[f'return_day_{i+1}'] = stock_returns.iloc[i] if i < len(stock_returns) else None
        result_row[f'alpha_day_{i+1}'] = alpha.iloc[i] if i < len(alpha) else None

    return result_row

# --- NEW, OPTIMIZED ORCHESTRATOR ---
def batch_fetch_stock_targets(df, timepoints_bdays):
    """
    The main orchestrator that performs a single bulk download and then
    processes all targets in parallel.
    """
    max_days_needed = max(timepoints_bdays.values()) if timepoints_bdays else 0
    if max_days_needed == 0:
        raise ValueError("Timepoints dictionary cannot be empty.")
    
    df['Filing Date'] = pd.to_datetime(df['Filing Date'], dayfirst=True)


    # --- 1. BULK DATA DOWNLOAD (The Core Optimization) ---
    print(f"- Performing a single bulk download for all required market data...")
    unique_tickers = df['Ticker'].unique().tolist()
    all_tickers_to_fetch = unique_tickers + ['SPY']
    
    # Define a wide enough date range to cover all scenarios
    start_date = df['Filing Date'].min() - pd.Timedelta(days=1)
    end_date = pd.to_datetime('today', dayfirst=True) + pd.Timedelta(days=1)
    
    market_data_raw = yf.download(
        all_tickers_to_fetch,
        start=start_date,
        end=end_date,
        group_by='ticker',
        progress=True,
        threads=True,
        auto_adjust=True # Let yfinance handle adjustments
    )
    
    # Reformat into a more accessible dictionary
    market_data = {t: market_data_raw[t].copy() for t in unique_tickers if t in market_data_raw.columns.levels[0]}
    market_data['SPY'] = market_data_raw['SPY']

    # --- 2. PARALLEL PROCESSING (Now extremely fast) ---
    print(f"- Processing {len(df)} entries in parallel...")
    processing_args = [(row, market_data, max_days_needed) for _, row in df.iterrows()]
    
    with Pool(cpu_count()//2) as pool:
        results = list(tqdm(
            pool.imap(_process_single_target, processing_args),
            total=len(processing_args),
            desc="- Calculating returns and alphas"
        ))

    # --- 3. AGGREGATE RESULTS ---
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("[WARN] No valid target data could be generated.")
        return pd.DataFrame(), pd.DataFrame()
        
    all_targets_df = pd.DataFrame(valid_results)
    
    # Filter the original features DataFrame to ensure perfect alignment with targets
    # This is crucial for maintaining data integrity.
    successful_keys = all_targets_df[['Ticker', 'Filing Date']].copy()
    final_features_df = pd.merge(df, successful_keys, on=['Ticker', 'Filing Date'], how='inner')
    
    dropped_rows = len(df) - len(final_features_df)
    if dropped_rows > 0:
        print(f"- Dropped {dropped_rows} feature rows due to missing or incomplete target data.")

    return final_features_df, all_targets_df

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
    with pd.ExcelWriter(output_file) as writer:
        returns_df.to_excel(writer, sheet_name='Returns', index=False)
        alpha_df.to_excel(writer, sheet_name='Alpha', index=False)
    print(f"- Formatted stock data successfully saved to {output_file}")

def save_final_features(features_df: pd.DataFrame, features_out_file: str):
    os.makedirs(os.path.dirname(features_out_file), exist_ok=True)
    features_df['Filing Date'] = pd.to_datetime(features_df['Filing Date']).dt.strftime('%d-%m-%Y %H:%M:%S')
    features_df.to_excel(features_out_file, index=False)
    print(f"- Final filtered features saved to {features_out_file}.")

def generate_and_save_final_targets_ohlc(stock_data_file, targets_output_file, timepoints):
    """
    Reads 'Return_Open', ... 'Alpha_Close' sheets from stock_data_file,
    aggregates for each timepoint, and saves wide sheets (columns: Ticker, Filing Date, all return_<tp> or alpha_<tp>).
    """
    print("- Loading OHLC stock/alpha data for explicit sheet-by-type/wide targets...")
    try:
        sheets = pd.read_excel(stock_data_file, sheet_name=None)
    except FileNotFoundError:
        print(f"[ERROR] Could not find input file: {stock_data_file}. Skipping target generation.")
        return

    sheet_types = [
        "Return_Open", "Return_High", "Return_Low", "Return_Close",
        "Alpha_Open", "Alpha_High", "Alpha_Low", "Alpha_Close",
    ]
    # timepoints: dict like {'1w': 5, '1m': 20, ...}
    timepoint_keys = list(timepoints.items())

    wide_sheets = {}
    for sheet_type in sheet_types:
        if sheet_type not in sheets:
            print(f"[WARN] {sheet_type} missing; will be blank.")
            wide_sheets[sheet_type] = pd.DataFrame()
            continue

        df = sheets[sheet_type]
        output_rows = []
        all_target_cols = [
            (f'return_{tpstr}' if sheet_type.startswith('Return') else f'alpha_{tpstr}')
            for tpstr, bdays in timepoint_keys
        ]

        for idx, row in df.iterrows():
            out = {
                'Ticker': row['Ticker'],
                'Filing Date': row['Filing Date'],
            }
            for tpstr, bdays in timepoint_keys:
                col = f'Day_{bdays-1}'
                colname = f'return_{tpstr}' if sheet_type.startswith('Return') else f'alpha_{tpstr}'
                out[colname] = row[col] if col in row else None
            output_rows.append(out)
        wide_sheets[sheet_type] = pd.DataFrame(output_rows)

    with pd.ExcelWriter(targets_output_file) as writer:
        for sheet_type, df in wide_sheets.items():
            df.to_excel(writer, sheet_name=sheet_type, index=False)
    print(f"- Final per-type wide OHLC targets (w/timepoint labels) saved to {targets_output_file}")

