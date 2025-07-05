import os
import pandas as pd
import yfinance as yf
import contextlib
from pandas.tseries.offsets import BDay

def download_daily_stock_data(ticker, filing_date, max_days):
    """Download daily stock data for a given ticker between filing_date and 20 business days after."""
    end_date = pd.to_datetime(filing_date, dayfirst=True) + BDay(max_days + 5)
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            try:
                stock_data = yf.download(ticker, start=filing_date, end=end_date, interval='1d', progress=False)
                if stock_data.empty:
                    print(f"- No data found for {ticker} between {filing_date} and {end_date}.")
                    return None
                return stock_data['Close'][:max_days].reset_index(drop=True)  # Crop to max_days
            except Exception as e:
                print(f"- Failed to download data for {ticker}: {e}")
                return None

def create_stock_data_dict(results):
    """Create a dictionary of stock data aligned to filing dates and calculate alpha."""
    stock_data_dict = {}

    for ticker, filing_date, max_days, ticker_data, spy_data in results:
        # skip if missing
        if ticker_data is None or spy_data is None:
            continue

        # pick stock_prices
        if ticker_data.shape[1] == 1:
            stock_prices = ticker_data.iloc[:, 0]
        elif 'Close' in ticker_data.columns:
            stock_prices = ticker_data['Close']
        elif 'Adj Close' in ticker_data.columns:
            stock_prices = ticker_data['Adj Close']
        else:
            continue

        # pick spy_prices
        if spy_data.shape[1] == 1:
            spy_prices = spy_data.iloc[:, 0]
        elif 'Close' in spy_data.columns:
            spy_prices = spy_data['Close']
        elif 'Adj Close' in spy_data.columns:
            spy_prices = spy_data['Adj Close']
        else:
            continue

        # ensure sorted by date
        stock_prices = stock_prices.sort_index()
        spy_prices   = spy_prices.sort_index()

        # need at least one price
        if stock_prices.empty or spy_prices.empty:
            continue

        # avoid zero‐division
        first_sp = stock_prices.iloc[0]
        first_spy = spy_prices.iloc[0]
        if first_sp == 0 or first_spy == 0:
            continue

        # compute scalar Series of returns
        stock_returns = (stock_prices / first_sp) - 1
        spy_returns   = (spy_prices   / first_spy) - 1
        alpha         = stock_returns - spy_returns

        key = (ticker, filing_date.strftime('%d/%m/%Y %H:%M'))
        stock_data_dict[key] = {}

        # now each .iloc[i] is a scalar float64, so float(...) will work
        for i in range(max_days):
            if i < len(stock_returns):
                stock_data_dict[key][f'Day {i+1} Stock'] = float(stock_returns.iloc[i])
                stock_data_dict[key][f'Day {i+1} Alpha'] = float(alpha.iloc[i])
            else:
                stock_data_dict[key][f'Day {i+1} Stock'] = None
                stock_data_dict[key][f'Day {i+1} Alpha'] = None

    return stock_data_dict



def save_to_excel(return_df, alpha_df, output_file):
    """
    Save the returns and alpha sheets to an Excel file.
    
    Args:
        return_df (pd.DataFrame): DataFrame containing return data.
        alpha_df (pd.DataFrame): DataFrame containing alpha data.
        output_file (str): The path to the Excel file to save the data to.
    """
    stock_data_dir = os.path.dirname(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write the return and alpha data to separate sheets
    with pd.ExcelWriter(output_file) as writer:
        return_df.to_excel(writer, sheet_name='Returns', index=False)
        alpha_df.to_excel(writer, sheet_name='Alpha', index=False)
    
    print(f"- Data successfully saved to {output_file}.")

def download_data_wrapper(info):
    """Wrapper function to pass to multiprocessing."""
    ticker, filing_date, max_days = info
    stock_data  = download_daily_stock_data(ticker, filing_date, max_days)
    spy_data    = download_daily_stock_data('SPY', filing_date, max_days)
    return (ticker, filing_date, max_days, stock_data, spy_data)

def drop_empty_rows(features_df, return_df, alpha_df):
    """
    Drop rows with empty cells in return_df and alpha_df, and also drop corresponding rows in features_df.
    
    Args:
        features_df (pd.DataFrame): The original features DataFrame.
        return_df (pd.DataFrame): The returns DataFrame.
        alpha_df (pd.DataFrame): The alpha DataFrame.
    
    Returns:
        pd.DataFrame: The filtered features_df, return_df, and alpha_df.
    """
    initial_length = len(features_df)

    # Drop rows where any cell is NaN in return_df or alpha_df
    combined_df = pd.concat([return_df, alpha_df], axis=1)
    mask = combined_df.isnull().any(axis=1)

    # Filter the DataFrames
    filtered_features_df    = features_df[~mask].copy()
    filtered_return_df      = return_df[~mask].copy()
    filtered_alpha_df       = alpha_df[~mask].copy()

    # Print the number of dropped rows
    dropped_rows = initial_length - len(filtered_features_df)
    print(f"- Dropped {dropped_rows} rows with missing values.")

    return filtered_features_df, filtered_return_df, filtered_alpha_df

def save_final_features(features_df, features_out_file):
    """
    Save the final filtered features DataFrame after dropping rows with empty entries.
    
    Args:
        features_df (pd.DataFrame): DataFrame containing the final filtered features.
        features_out_file (str): The path to the Excel file to save the data to.
    """
    features_out_dir = os.path.dirname(features_out_file)
    os.makedirs(os.path.dirname(features_out_file), exist_ok=True)
    
    features_df['Filing Date'] = features_df['Filing Date'].dt.strftime('%d-%m-%Y %H:%M')
    
    # Save the features DataFrame
    features_df.to_excel(features_out_file, index=False)
    print(f"- Final filtered features saved to {features_out_file}.")
