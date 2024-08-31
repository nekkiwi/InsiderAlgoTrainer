import os
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay

def load_features(features_file):
    """Load the full features DataFrame and extract Ticker and Filing Date columns."""
    features_df = pd.read_excel(features_file)
    features_df['Filing Date'] = pd.to_datetime(features_df['Filing Date'], dayfirst=True)
    ticker_filing_dates = features_df[['Ticker', 'Filing Date']]
    earliest_date = ticker_filing_dates['Filing Date'].min()
    return ticker_filing_dates, earliest_date

def download_daily_stock_data(ticker, filing_date, max_days):
    """Download daily stock data for a given ticker between filing_date and 20 business days after."""
    end_date = pd.to_datetime(filing_date, dayfirst=True) + BDay(max_days+5)
    try:
        stock_data = yf.download(ticker, start=filing_date, end=end_date, interval='1d', progress=False)
        if stock_data.empty:
            print(f"No data found for {ticker} between {filing_date} and {end_date}.")
            return None
        return stock_data
    except Exception as e:
        print(f"Failed to download data for {ticker}: {e}")
        return None

def download_spy_data_for_period(filing_date, max_days):
    """Download SPY data for the same period as the stock data."""
    end_date = pd.to_datetime(filing_date, dayfirst=True) + BDay(max_days+5)
    try:
        spy_data = yf.download('SPY', start=filing_date, end=end_date, interval='1d', progress=False)
        if spy_data.empty:
            print(f"No SPY data found between {filing_date} and {end_date}.")
            return None
        return spy_data['Close'].reset_index(drop=True)  # Return as a Series
    except Exception as e:
        print(f"Failed to download SPY data: {e}")
        return None

def create_stock_data_dict(results):
    """Create a dictionary of stock data aligned to filing dates."""
    stock_data_dict = {}

    for ticker, filing_date, max_days, ticker_data, spy_data in results:
        if ticker_data is not None:
            closing_data = ticker_data['Close'][:max_days].reset_index(drop=True)
            stock_data_dict[(ticker, filing_date.strftime('%d/%m/%Y %H:%M'))] = closing_data.T
            
    return stock_data_dict

def save_to_excel(stock_data_df, return_df, alpha_df, output_file):
    """Save the stock data, returns, and alpha sheets to an Excel file."""
    stock_data_dir = os.path.dirname(output_file)
    os.makedirs(stock_data_dir, exist_ok=True)
    
    with pd.ExcelWriter(output_file) as writer:
        stock_data_df.to_excel(writer, sheet_name='Stock Data', index=False)
        return_df.to_excel(writer, sheet_name='Returns', index=False)
        alpha_df.to_excel(writer, sheet_name='Alpha', index=False)
    
    print(f"Data successfully saved to {output_file}.")

def download_data_wrapper(info):
    """Wrapper function to pass to multiprocessing."""
    ticker, filing_date, max_days = info
    stock_data = download_daily_stock_data(ticker, filing_date, max_days)
    spy_data = download_spy_data_for_period(filing_date, max_days)
    return (ticker, filing_date, max_days, stock_data, spy_data)
