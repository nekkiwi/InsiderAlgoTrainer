import yfinance as yf
import talib
import pandas as pd
from utils.date_helpers import get_next_market_open

def process_dates(self):
    # Convert date strings to datetime objects
    self.data['Filing Date'] = pd.to_datetime(self.data['Filing Date']).apply(get_next_market_open)
    self.data['Trade Date'] = pd.to_datetime(self.data['Trade Date'])
    
    # Calculate "Days Since Trade"
    self.data['Days Since Trade'] = (self.data['Filing Date'] - self.data['Trade Date']).dt.days

def clean_numeric_columns(self):
    self.data['Price'] = self.data['Price'].replace({r'\$': '', r',': ''}, regex=True).astype(float)
    self.data['Qty'] = self.data['Qty'].replace({r',': ''}, regex=True).astype(int)
    self.data['Owned'] = self.data['Owned'].replace({r',': ''}, regex=True).astype(int)
    self.data['Value'] = self.data['Value'].replace({r'\$': '', r',': '', r'\+': ''}, regex=True).astype(float)
    self.data['ΔOwn'] = self.data['ΔOwn'].replace({r'%': '', r'\+': '', r'New': '999', r'>': ''}, regex=True).astype(float)

def parse_titles(self):
    self.data['CEO'] = self.data['Title'].apply(lambda title: int('CEO' in title))
    self.data['CFO'] = self.data['Title'].apply(lambda title: int('CFO' in title))
    self.data['COO'] = self.data['Title'].apply(lambda title: int('COO' in title))
    self.data['Dir'] = self.data['Title'].apply(lambda title: int('Dir' in title))
    self.data['Pres'] = self.data['Title'].apply(lambda title: int('Pres' in title))
    self.data['VP'] = self.data['Title'].apply(lambda title: int('VP' in title))
    self.data['10%'] = self.data['Title'].apply(lambda title: int('10%' in title))

def calculate_technical_indicators(row, stock_data):
    indicators = {}
    # Example of calculating various indicators
    indicators['SMA_50'] = talib.SMA(stock_data['Close'], timeperiod=50)
    indicators['EMA_50'] = talib.EMA(stock_data['Close'], timeperiod=50)
    indicators['RSI_14'] = talib.RSI(stock_data['Close'], timeperiod=14)
    indicators['MACD'], indicators['MACD_Signal'], _ = talib.MACD(stock_data['Close'])
    
    for key, value in indicators.items():
        row[key] = value.iloc[-1]  # Get the most recent value of each indicator
        
    return row

def aggregate_group(self):
    # Group by Ticker and Filing Date, then aggregate
    self.data = self.data.groupby(['Ticker', 'Filing Date']).agg(
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