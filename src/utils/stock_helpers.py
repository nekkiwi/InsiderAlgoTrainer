import yfinance as yf
import pandas as pd
import contextlib
import os
import sys

def download_stock_data(ticker, filing_date, max_period=50, interval='1d', benchmark_ticker='SPY'):
    """Download stock data for a given ticker over a specific period."""
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            try:
                end_date = pd.to_datetime(filing_date, dayfirst=True) - pd.tseries.offsets.BDay(1)
                start_date = end_date - pd.tseries.offsets.BDay(max_period+10)
                
                stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
                benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date, interval=interval, progress=False)
                if stock_data.empty or benchmark_data.empty:
                    return None, None
                return stock_data, benchmark_data
            except Exception as e:
                # print(f"Failed to download data for {ticker}: {e}")
                return None, None