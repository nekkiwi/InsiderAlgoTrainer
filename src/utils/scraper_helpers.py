import yfinance as yf
import talib
import pandas as pd
from utils.date_helpers import get_next_market_open
from utils.stock_helpers import download_stock_data

def process_dates(df):
    # Convert date strings to datetime objects
    df['Filing Date'] = pd.to_datetime(df['Filing Date']).apply(get_next_market_open)
    df['Trade Date'] = pd.to_datetime(df['Trade Date'])
    
    # Calculate "Days Since Trade"
    df['Days Since Trade'] = (df['Filing Date'] - df['Trade Date']).dt.days
    return df

def clean_numeric_columns(df):
    df['Price'] = df['Price'].replace({r'\$': '', r',': ''}, regex=True).astype(float)
    df['Qty'] = df['Qty'].replace({r',': ''}, regex=True).astype(int)
    df['Owned'] = df['Owned'].replace({r',': ''}, regex=True).astype(int)
    df['Value'] = df['Value'].replace({r'\$': '', r',': '', r'\+': ''}, regex=True).astype(float)
    df['ΔOwn'] = df['ΔOwn'].replace({r'%': '', r'\+': '', r'New': '999', r'>': ''}, regex=True).astype(float)
    return df

def parse_titles(df):
    df['CEO'] = df['Title'].apply(lambda title: int('CEO' in title))
    df['CFO'] = df['Title'].apply(lambda title: int('CFO' in title))
    df['COO'] = df['Title'].apply(lambda title: int('COO' in title))
    df['Dir'] = df['Title'].apply(lambda title: int('Dir' in title))
    df['Pres'] = df['Title'].apply(lambda title: int('Pres' in title))
    df['VP'] = df['Title'].apply(lambda title: int('VP' in title))
    df['10%'] = df['Title'].apply(lambda title: int('10%' in title))
    return df

def normalize_indicators(indicators, stock_data):
    """Normalize indicators where appropriate."""
    closing_price = stock_data['Close'].iloc[-1]
    total_volume = stock_data['Volume'].iloc[-1]
    normalized_indicators = {}

    for key, value in indicators.items():
        if isinstance(value, pd.Series):
            if key in ['SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'Bollinger_Upper', 'Bollinger_Lower', 'ATR_14']:
                # Normalize by the closing price
                normalized_indicators[key] = value / closing_price if not value.empty else None
            
            elif key in ['OBV']:
                # Normalize by the total volume
                normalized_indicators[key] = value / total_volume if not value.empty else None
            
            else:
                # Leave as is
                normalized_indicators[key] = value if not value.empty else None
        else:
            # Handle cases where value is not a Series (e.g., string or scalar)
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
    indicators['Bollinger_Upper'], indicators['Bollinger_Middle'], indicators['Bollinger_Lower'] = talib.BBANDS(stock_data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    # Volume Indicators
    indicators['OBV'] = talib.OBV(stock_data['Close'], stock_data['Volume'])

    # Cycle Indicators
    indicators['HT_TRENDMODE'] = talib.HT_TRENDMODE(stock_data['Close'])
    
    # Pattern Recognition
    indicators['CDL_DOJI'] = talib.CDLDOJI(stock_data['Open'], stock_data['High'], stock_data['Low'], stock_data['Close'])
    indicators['CDL_HAMMER'] = talib.CDLHAMMER(stock_data['Open'], stock_data['High'], stock_data['Low'], stock_data['Close'])
    indicators['CDL_ENGULFING'] = talib.CDLENGULFING(stock_data['Open'], stock_data['High'], stock_data['Low'], stock_data['Close'])

    # Other Indicators (A selection of additional useful indicators)
    indicators['AROON_Up'], indicators['AROON_Down'] = talib.AROON(stock_data['High'], stock_data['Low'], timeperiod=14)
    indicators['SAR'] = talib.SAR(stock_data['High'], stock_data['Low'], acceleration=0.02, maximum=0.2)

    # Normalize indicators relative to the closing price
    normalized_indicators = normalize_indicators(indicators, stock_data)

    # Assign the most recent values of each indicator to the row
    for key, value in normalized_indicators.items():
        row[key] = value.iloc[-1] if not value.empty else None  # Get the most recent value of each indicator

    return row

def process_ticker_technical_indicators(row):
    """Process each ticker by downloading the stock data, benchmark data, and calculating indicators."""
    ticker = row['Ticker']
    filing_date = row['Filing Date']

    # Download stock and benchmark data
    stock_data, benchmark_data = download_stock_data(ticker, filing_date, max_period=50, interval='1d', benchmark_ticker='SPY')
    
    if stock_data is not None and benchmark_data is not None:
        # Calculate technical indicators
        indicators = calculate_technical_indicators(row, stock_data)
        
        # Calculate alpha-related indicators
        alpha_indicators = calculate_alpha_indicators(stock_data, benchmark_data)
        
        # Merge the indicators
        indicators.update(alpha_indicators)
        
        # Normalize indicators as needed
        normalized_indicators = normalize_indicators(indicators, stock_data)
        
        # Update the row with all indicators
        for key, value in normalized_indicators.items():
            row[key] = value

        return row
    return None

def calculate_alpha_indicators(stock_data, benchmark_data):
    """Calculate alpha-related indicators comparing the stock to the benchmark."""
    indicators = {}

    # Calculate daily returns
    stock_returns = stock_data['Close'].pct_change().dropna()
    benchmark_returns = benchmark_data['Close'].pct_change().dropna()

    # Calculate excess returns
    excess_returns = stock_returns - benchmark_returns

    # Cumulative Alpha
    indicators['Cumulative_Alpha'] = excess_returns.cumsum().iloc[-1]

    # Rolling Alpha (30 days)
    indicators['Rolling_Alpha_30'] = excess_returns.rolling(window=30).mean().iloc[-1]
    
    # Beta
    covariance = stock_returns.cov(benchmark_returns)
    variance = benchmark_returns.var()
    indicators['Beta'] = covariance / variance

    # Jensen's Alpha (CAPM)
    risk_free_rate = 0.01 / 252  # Assuming a 1% annual risk-free rate, daily return
    expected_returns = risk_free_rate + indicators['Beta'] * (benchmark_returns.mean() - risk_free_rate)
    indicators['Jensen_Alpha'] = (stock_returns.mean() - expected_returns) * 252  # Annualized

    # Tracking Error
    indicators['Tracking_Error'] = excess_returns.std() * (252 ** 0.5)  # Annualized

    # Information Ratio
    indicators['Information_Ratio'] = indicators['Cumulative_Alpha'] / indicators['Tracking_Error']

    return indicators

def aggregate_group(df):
    # Group by Ticker and Filing Date, then aggregate
    df = df.groupby(['Ticker', 'Filing Date']).agg(
        Number_of_Purchases=('Ticker', 'size'),
        Price=('Price', 'mean'),
        Qty=('Qty', 'sum'),
        Owned=('Owned', 'mean'),
        ΔOwn=('ΔOwn', 'mean'),
        Value=('Value', 'sum'),
        CEO=('CEO', 'max'),
        CFO=('CFO', 'max'),
        COO=('COO', 'max'),
        Dir=('Dir', 'max'),
        Pres=('Pres', 'max'),
        VP=('VP', 'max'),
        TenPercent=('10%', 'max')).sort_values(by='Filing Date', ascending=False).reset_index()
    return df