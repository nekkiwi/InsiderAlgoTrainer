import yfinance as yf
import pandas as pd

def download_stock_data(ticker, period='ytd', interval='1d'):
    """Download stock data for a given ticker."""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            print(f"No data found for {ticker}. Skipping this ticker.")
            return None
        return data
    except Exception as e:
        print(f"Failed to download data for {ticker}: {e}")
        return None