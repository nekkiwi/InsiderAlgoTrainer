import os
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay

def download_daily_stock_data(ticker, filing_date, max_days):
    """Download daily stock data for a given ticker between filing_date and 20 business days after."""
    end_date = pd.to_datetime(filing_date, dayfirst=True) + BDay(max_days + 5)
    try:
        stock_data = yf.download(ticker, start=filing_date, end=end_date, interval='1d', progress=False)
        if stock_data.empty:
            print(f"- No data found for {ticker} between {filing_date} and {end_date}.")
            return None
        return stock_data['Close'][:max_days].reset_index(drop=True)  # Crop to max_days
    except Exception as e:
        print(f"- Failed to download data for {ticker}: {e}")
        return None

def download_spy_data_for_period(filing_date, max_days):
    """Download SPY data for the same period as the stock data."""
    end_date = pd.to_datetime(filing_date, dayfirst=True) + BDay(max_days + 5)
    try:
        spy_data = yf.download('SPY', start=filing_date, end=end_date, interval='1d', progress=False)
        if spy_data.empty:
            print(f"- No SPY data found between {filing_date} and {end_date}.")
            return None
        return spy_data['Close'][:max_days].reset_index(drop=True)  # Crop to max_days
    except Exception as e:
        print(f"- Failed to download SPY data: {e}")
        return None

def create_stock_data_dict(results):
    """Create a dictionary of stock data aligned to filing dates and calculate alpha."""
    stock_data_dict = {}

    for ticker, filing_date, max_days, ticker_data, spy_data in results:
        if ticker_data is not None and spy_data is not None:
            # Normalize stock returns
            stock_returns = (ticker_data / ticker_data[0]) - 1
            spy_returns = (spy_data / spy_data[0]) - 1

            # Calculate alpha: stock return - SPY return
            alpha = stock_returns - spy_returns

            # For each day, store the stock price and the alpha value as separate columns
            stock_data_dict[(ticker, filing_date.strftime('%d/%m/%Y %H:%M'))] = {}

            for i in range(max_days):
                stock_data_dict[(ticker, filing_date.strftime('%d/%m/%Y %H:%M'))][f'Day {i+1} Stock'] = ticker_data[i]
                stock_data_dict[(ticker, filing_date.strftime('%d/%m/%Y %H:%M'))][f'Day {i+1} Alpha'] = alpha[i]

    return stock_data_dict

def calculate_returns(stock_data_df, stock_columns):
    """Calculate the returns relative to Day 1."""
    return_df = stock_data_df[['Ticker', 'Filing Date'] + stock_columns].copy()

    for index, row in return_df.iterrows():
        day_1_price = row['Day 1 Stock']
        if pd.notnull(day_1_price):
            for col in stock_columns:
                return_df.at[index, col] = (row[col] / day_1_price) - 1  # Relative change from Day 1

    return return_df

def save_to_excel(return_df, alpha_df, output_file):
    """
    Save the returns and alpha sheets to an Excel file.
    
    Args:
        return_df (pd.DataFrame): DataFrame containing return data.
        alpha_df (pd.DataFrame): DataFrame containing alpha data.
        output_file (str): The path to the Excel file to save the data to.
    """
    stock_data_dir = os.path.dirname(output_file)
    os.makedirs(stock_data_dir, exist_ok=True)
    
    # Write the return and alpha data to separate sheets
    with pd.ExcelWriter(output_file) as writer:
        return_df.to_excel(writer, sheet_name='Returns', index=False)
        alpha_df.to_excel(writer, sheet_name='Alpha', index=False)
    
    print(f"- Data successfully saved to {output_file}.")

def download_data_wrapper(info):
    """Wrapper function to pass to multiprocessing."""
    ticker, filing_date, max_days = info
    stock_data = download_daily_stock_data(ticker, filing_date, max_days)
    spy_data = download_spy_data_for_period(filing_date, max_days)
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
    filtered_features_df = features_df[~mask].copy()
    filtered_return_df = return_df[~mask].copy()
    filtered_alpha_df = alpha_df[~mask].copy()

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
    os.makedirs(features_out_dir, exist_ok=True)
    
    features_df['Filing Date'] = features_df['Filing Date'].dt.strftime('%d-%m-%Y %H:%M')
    
    # Save the features DataFrame
    features_df.to_excel(features_out_file, index=False)
    print(f"- Final filtered features saved to {features_out_file}.")
