import ta
import pandas as pd
import yfinance as yf
import contextlib
import os
import numpy as np

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

def normalize_indicators(indicators, stock_data):
    """Normalize indicators where appropriate."""
    closing_price = stock_data['Close'].iloc[-1]
    total_volume = stock_data['Volume'].iloc[-1]
    normalized_indicators = {}

    for key, value in indicators.items():
        if isinstance(value, pd.Series):
            if key in ['SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'Bollinger_Upper', 'Bollinger_Lower', 'ATR_14', 'SAR']:
                # Normalize by the closing price
                normalized_indicators[key] = value / closing_price if not value.empty else None
            
            elif key in ['OBV']:
                # Normalize by the total volume
                normalized_indicators[key] = value / total_volume if not value.empty else None
            
            else:
                # Leave as is
                normalized_indicators[key] = value if not value.empty else None
        else:
            # Handle cases where value is not a Series (e.g., alpha indicators)
            if key in ['Cumulative_Alpha', 'Rolling_Alpha_30', 'Jensen_Alpha']:
                # Normalize alpha indicators by the closing price
                normalized_indicators[key] = value / closing_price if value is not None else None
            else:
                normalized_indicators[key] = value

    return normalized_indicators

def calculate_technical_indicators(row, stock_data):
    """
    Calculate technical indicators using the `ta` Python library.
    Assigns the most recent values to the row.
    """

    # 1) Drop NA rows early
    stock_data = stock_data.dropna()
    if len(stock_data) < 50:
        return row  # not enough data

    # 2) Utility to coerce any column into a 1-D Series
    def ensure_1d_series(col):
        # col may be a DataFrame (n×1) or Series (n,)
        if isinstance(col, pd.DataFrame):
            # take first column
            s = col.iloc[:, 0]
        else:
            s = col
        # flatten any extra dims
        values = s.values
        if values.ndim > 1:
            values = values.flatten()
        return pd.Series(values, index=stock_data.index)

    # 3) Extract each column as a proper Series
    close  = ensure_1d_series(stock_data['Close'])
    high   = ensure_1d_series(stock_data['High'])
    low    = ensure_1d_series(stock_data['Low'])
    volume = ensure_1d_series(stock_data['Volume'])
    open_  = ensure_1d_series(stock_data['Open'])

    indicators = {}

    # --- Moving Averages ---
    indicators['SMA_10'] = close.rolling(10).mean().iloc[-1]
    indicators['SMA_50'] = close.rolling(50).mean().iloc[-1]
    indicators['EMA_10'] = close.ewm(span=10).mean().iloc[-1]
    indicators['EMA_50'] = close.ewm(span=50).mean().iloc[-1]

    # --- Momentum Indicators ---
    rsi = ta.momentum.RSIIndicator(close=close, window=14)
    indicators['RSI_14'] = rsi.rsi().iloc[-1]

    macd = ta.trend.MACD(close=close)
    indicators['MACD']        = macd.macd().iloc[-1]
    indicators['MACD_Signal'] = macd.macd_signal().iloc[-1]
    indicators['MACD_Hist']   = macd.macd_diff().iloc[-1]

    adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
    indicators['ADX_14'] = adx.adx().iloc[-1]

    cci = ta.trend.CCIIndicator(high=high, low=low, close=close, window=14)
    indicators['CCI_14'] = cci.cci().iloc[-1]

    roc = ta.momentum.ROCIndicator(close=close, window=10)
    indicators['ROC'] = roc.roc().iloc[-1]

    mfi = ta.volume.MFIIndicator(high=high, low=low, close=close, volume=volume, window=14)
    indicators['MFI_14'] = mfi.money_flow_index().iloc[-1]

    willr = ta.momentum.WilliamsRIndicator(high=high, low=low, close=close, lbp=14)
    indicators['WILLR_14'] = willr.williams_r().iloc[-1]

    stoch = ta.momentum.StochasticOscillator(high=high, low=low, close=close)
    indicators['STOCH_K'] = stoch.stoch().iloc[-1]
    indicators['STOCH_D'] = stoch.stoch_signal().iloc[-1]

    # --- Volatility Indicators ---
    atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
    indicators['ATR_14'] = atr.average_true_range().iloc[-1]

    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    indicators['Bollinger_Upper'] = bb.bollinger_hband().iloc[-1]
    indicators['Bollinger_Lower'] = bb.bollinger_lband().iloc[-1]

    # --- Volume Indicators ---
    obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume)
    indicators['OBV'] = obv.on_balance_volume().iloc[-1]

    # 4) Write back into the row dict
    for key, val in indicators.items():
        row[key] = val

    return row

def calculate_alpha_indicators(stock_data, benchmark_data):
    """
    Calculate alpha-related indicators comparing the stock to the benchmark.
    Returns a dict with:
      Cumulative_Alpha, Rolling_Alpha_30, Beta,
      Jensen_Alpha, Tracking_Error, Information_Ratio
    """
    # Initialize all to None
    indicators = {k: None for k in [
        'Cumulative_Alpha','Rolling_Alpha_30','Beta',
        'Jensen_Alpha','Tracking_Error','Information_Ratio'
    ]}

    # Need at least 30 days to compute anything meaningful
    if len(stock_data) < 30 or len(benchmark_data) < 30:
        return indicators

    # 1) compute daily returns as pandas Series
    sr = (stock_data['Close']
          .astype('float64')
          .pct_change()
          .dropna())
    br = (benchmark_data['Close']
          .astype('float64')
          .pct_change()
          .dropna())

    # 2) align lengths
    n = min(len(sr), len(br))
    sr = sr.iloc[-n:]
    br = br.iloc[-n:]

    # 3) pull out 1-D numpy arrays
    sr_arr = sr.to_numpy().flatten()
    br_arr = br.to_numpy().flatten()

    # 4) compute excess returns (still a 1-D numpy)
    excess = sr_arr - br_arr

    # 5) cumulative alpha
    indicators['Cumulative_Alpha'] = np.nansum(excess)

    # 6) rolling 30-day alpha via pandas on a flat 1-D array
    excess_series = pd.Series(excess)
    indicators['Rolling_Alpha_30'] = excess_series.rolling(window=30).mean().iloc[-1]

    # 7) beta via sample covariance / variance
    if n > 1:
        cov = np.cov(sr_arr, br_arr, ddof=1)[0,1]
        var = np.var(br_arr, ddof=1)
        indicators['Beta'] = cov / var if var > 0 else None

    # 8) Jensen’s Alpha (annualized)
    beta = indicators['Beta']
    if beta is not None:
        rf = 0.01 / 252  # daily risk-free
        exp_ret = rf + beta * (np.nanmean(br_arr) - rf)
        indicators['Jensen_Alpha'] = (np.nanmean(sr_arr) - exp_ret) * 252

    # 9) Tracking Error (annualized stdev of excess)
    if excess.size > 1:
        indicators['Tracking_Error'] = np.nanstd(excess, ddof=1) * np.sqrt(252)

    # 10) Information Ratio
    te = indicators['Tracking_Error']
    if te:
        indicators['Information_Ratio'] = indicators['Cumulative_Alpha'] / te

    return indicators

def process_ticker_technical_indicators(row):
    """Process each ticker by downloading the stock data, benchmark data, and calculating indicators."""
    ticker = row['Ticker']
    filing_date = row['Filing Date']
    
    # Download stock and benchmark data
    stock_data, benchmark_data = download_stock_data(ticker, filing_date, max_period=50, interval='1d', benchmark_ticker='SPY')
    
    if stock_data is None or benchmark_data is None:
        return None
    
    # Calculate technical indicators
    indicators = calculate_technical_indicators(row, stock_data)
    # Calculate alpha-related indicators
    alpha_indicators = calculate_alpha_indicators(stock_data, benchmark_data)
    indicators.update(alpha_indicators)
    # Update the row with all indicators
    for key, value in indicators.items():
        row[key] = value

    return row