import talib
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
    Calculate a suite of technical indicators for a given stock data
    and assign the most recent values to the row.
    """
    # Convert relevant columns to 1-D numpy arrays of type double
    close  = stock_data['Close'].to_numpy(dtype='float64').flatten()
    high   = stock_data['High'].to_numpy(dtype='float64').flatten()
    low    = stock_data['Low'].to_numpy(dtype='float64').flatten()
    volume = stock_data['Volume'].to_numpy(dtype='float64').flatten()
    open_  = stock_data['Open'].to_numpy(dtype='float64').flatten()

    indicators = {}

    # Moving Averages
    sma10 = talib.SMA(close, timeperiod=10)
    sma50 = talib.SMA(close, timeperiod=50)
    ema10 = talib.EMA(close, timeperiod=10)
    ema50 = talib.EMA(close, timeperiod=50)
    indicators['SMA_10'] = sma10[-1]    if sma10.size    else None
    indicators['SMA_50'] = sma50[-1]    if sma50.size    else None
    indicators['EMA_10'] = ema10[-1]    if ema10.size    else None
    indicators['EMA_50'] = ema50[-1]    if ema50.size    else None

    # Momentum Indicators
    rsi14      = talib.RSI(close, timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(close)
    adx14      = talib.ADX(high, low, close, timeperiod=14)
    cci14      = talib.CCI(high, low, close, timeperiod=14)
    roc        = talib.ROC(close, timeperiod=10)
    mfi14      = talib.MFI(high, low, close, volume, timeperiod=14)
    willr14    = talib.WILLR(high, low, close, timeperiod=14)
    stoch_k, stoch_d = talib.STOCH(high, low, close)
    indicators['RSI_14']      = rsi14[-1]      if rsi14.size      else None
    indicators['MACD']        = macd[-1]       if macd.size       else None
    indicators['MACD_Signal'] = macd_signal[-1]if macd_signal.size else None
    indicators['MACD_Hist']   = macd_hist[-1]  if macd_hist.size  else None
    indicators['ADX_14']      = adx14[-1]      if adx14.size      else None
    indicators['CCI_14']      = cci14[-1]      if cci14.size      else None
    indicators['ROC']         = roc[-1]        if roc.size        else None
    indicators['MFI_14']      = mfi14[-1]      if mfi14.size      else None
    indicators['WILLR_14']    = willr14[-1]    if willr14.size    else None
    indicators['STOCH_K']     = stoch_k[-1]    if stoch_k.size    else None
    indicators['STOCH_D']     = stoch_d[-1]    if stoch_d.size    else None

    # Volatility Indicators
    atr14 = talib.ATR(high, low, close, timeperiod=14)
    upper, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    indicators['ATR_14']          = atr14[-1] if atr14.size else None
    indicators['Bollinger_Upper'] = upper[-1] if upper.size else None
    indicators['Bollinger_Lower'] = lower[-1] if lower.size else None

    # Volume Indicators
    obv = talib.OBV(close, volume)
    indicators['OBV'] = obv[-1] if obv.size else None

    # Pattern Recognition
    doji      = talib.CDLDOJI(open_, high, low, close)
    hammer    = talib.CDLHAMMER(open_, high, low, close)
    engulfing = talib.CDLENGULFING(open_, high, low, close)
    indicators['CDL_DOJI']      = int(doji[-1]      / 100) if doji.size      else None
    indicators['CDL_HAMMER']    = int(hammer[-1]    / 100) if hammer.size    else None
    indicators['CDL_ENGULFING'] = int(engulfing[-1] / 100) if engulfing.size else None

    # Aroon
    aroon_up, aroon_down = talib.AROON(high, low, timeperiod=14)
    indicators['AROON_Up']   = aroon_up[-1]   if aroon_up.size   else None
    indicators['AROON_Down'] = aroon_down[-1] if aroon_down.size else None

    # Parabolic SAR
    sar = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
    indicators['SAR'] = sar[-1] if sar.size else None

    # Assign all indicators back to the row
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