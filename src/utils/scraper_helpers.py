import yfinance as yf
import talib

def clean_numeric_columns(df):
    df['Price'] = df['Price'].replace({r'\$': '', r',': ''}, regex=True).astype(float)
    df['Qty'] = df['Qty'].replace({r',': ''}, regex=True).astype(int)
    df['Owned'] = df['Owned'].replace({r',': ''}, regex=True).astype(int)
    df['Value'] = df['Value'].replace({r'\$': '', r',': '', r'\+': ''}, regex=True).astype(float)
    df['ΔOwn'] = df['ΔOwn'].replace({
        r'%': '', r'\+': '', r'New': '999', r'>': ''
    }, regex=True).astype(float)
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

def calculate_technical_indicators(stock_data):
    df = {}
    # Example of calculating various indicators
    df['SMA_50'] = talib.SMA(stock_data['Close'], timeperiod=50)
    df['EMA_50'] = talib.EMA(stock_data['Close'], timeperiod=50)
    df['RSI_14'] = talib.RSI(stock_data['Close'], timeperiod=14)
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(stock_data['Close'])
    
    # Add more indicators as needed
    return df