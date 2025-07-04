import talib
import pandas as pd
import yfinance as yf
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
    """Calculate technical indicators for a given stock data."""
    indicators = {}

    # Moving Averages
    indicators['SMA_10'] = talib.SMA(stock_data['Close'], timeperiod=10)
    indicators['SMA_50'] = talib.SMA(stock_data['Close'], timeperiod=50)
    indicators['EMA_10'] = talib.EMA(stock_data['Close'], timeperiod=10)
    indicators['EMA_50'] = talib.EMA(stock_data['Close'], timeperiod=50)

    # Momentum Indicators
    indicators['RSI_14'] = talib.RSI(stock_data['Close'], timeperiod=14)
    indicators['MACD'], indicators['MACD_Signal'], indicators['MACD_Hist'] = talib.MACD(stock_data['Close'])
    indicators['ADX_14'] = talib.ADX(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
    indicators['CCI_14'] = talib.CCI(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
    indicators['ROC'] = talib.ROC(stock_data['Close'], timeperiod=10)
    indicators['MFI_14'] = talib.MFI(stock_data['High'], stock_data['Low'], stock_data['Close'], stock_data['Volume'], timeperiod=14)
    indicators['WILLR_14'] = talib.WILLR(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
    indicators['STOCH_K'], indicators['STOCH_D'] = talib.STOCH(stock_data['High'], stock_data['Low'], stock_data['Close'])

    # Volatility Indicators
    indicators['ATR_14'] = talib.ATR(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
    indicators['Bollinger_Upper'], _, indicators['Bollinger_Lower'] = talib.BBANDS(stock_data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    # Volume Indicators
    indicators['OBV'] = talib.OBV(stock_data['Close'], stock_data['Volume'])
    
    # Pattern Recognition
    indicators['CDL_DOJI'] = talib.CDLDOJI(stock_data['Open'], stock_data['High'], stock_data['Low'], stock_data['Close'])
    indicators['CDL_HAMMER'] = talib.CDLHAMMER(stock_data['Open'], stock_data['High'], stock_data['Low'], stock_data['Close'])
    indicators['CDL_ENGULFING'] = talib.CDLENGULFING(stock_data['Open'], stock_data['High'], stock_data['Low'], stock_data['Close'])
    
    indicators['CDL_DOJI'] = indicators['CDL_DOJI'].apply(lambda x: int(x / 100))
    indicators['CDL_HAMMER'] = indicators['CDL_HAMMER'].apply(lambda x: int(x / 100))
    indicators['CDL_ENGULFING'] = indicators['CDL_ENGULFING'].apply(lambda x: int(x / 100))

    # Other Indicators (A selection of additional useful indicators)
    indicators['AROON_Up'], indicators['AROON_Down'] = talib.AROON(stock_data['High'], stock_data['Low'], timeperiod=14)
    indicators['SAR'] = talib.SAR(stock_data['High'], stock_data['Low'], acceleration=0.02, maximum=0.2)

    # Normalize indicators relative to the closing price
    normalized_indicators = normalize_indicators(indicators, stock_data)

    # Assign the most recent values of each indicator to the row
    for key, value in normalized_indicators.items():
        row[key] = value.iloc[-1] if not value.empty else None  # Get the most recent value of each indicator

    return row

def calculate_alpha_indicators(stock_data, benchmark_data):
    """Calculate alpha-related indicators comparing the stock to the benchmark."""
    indicators = {}

    # Check if there's enough data to proceed
    if len(stock_data) < 30 or len(benchmark_data) < 30:
        return {key: None for key in ['Cumulative_Alpha', 'Rolling_Alpha_30', 'Beta', 'Jensen_Alpha', 'Tracking_Error', 'Information_Ratio']}
    
    # Calculate daily returns
    stock_returns = stock_data['Close'].pct_change().dropna()
    benchmark_returns = benchmark_data['Close'].pct_change().dropna()

    # Ensure returns align in length
    min_length = min(len(stock_returns), len(benchmark_returns))
    stock_returns = stock_returns[-min_length:]
    benchmark_returns = benchmark_returns[-min_length:]

    # Calculate excess returns
    excess_returns = stock_returns - benchmark_returns

    # Cumulative Alpha
    indicators['Cumulative_Alpha'] = excess_returns.cumsum().iloc[-1]

    # Rolling Alpha (30 days)
    indicators['Rolling_Alpha_30'] = excess_returns.rolling(window=30).mean().iloc[-1]

    # Beta
    covariance = stock_returns.cov(benchmark_returns)
    variance = benchmark_returns.var()
    indicators['Beta'] = covariance / variance if variance > 0 else None

    # Jensen's Alpha (CAPM)
    risk_free_rate = 0.01 / 252  # Assuming a 1% annual risk-free rate, daily return
    expected_returns = risk_free_rate + indicators['Beta'] * (benchmark_returns.mean() - risk_free_rate) if indicators['Beta'] is not None else None
    indicators['Jensen_Alpha'] = (stock_returns.mean() - expected_returns) * 252 if expected_returns is not None else None  # Annualized

    # Tracking Error
    indicators['Tracking_Error'] = excess_returns.std() * (252 ** 0.5) if not excess_returns.empty else None  # Annualized

    # Information Ratio
    indicators['Information_Ratio'] = indicators['Cumulative_Alpha'] / indicators['Tracking_Error'] if indicators['Tracking_Error'] else None

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
    # Merge all indicators
    indicators.update(alpha_indicators)
    # Normalize indicators as needed
    normalized_indicators = normalize_indicators(indicators, stock_data)
    # Update the row with all indicators
    for key, value in normalized_indicators.items():
        row[key] = value

    return row