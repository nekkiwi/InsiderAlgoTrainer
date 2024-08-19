import pandas as pd
from utils.date_helpers import get_next_market_open
from datetime import datetime
import requests
from bs4 import BeautifulSoup

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

def get_recent_trades(ticker):
        url = f"http://openinsider.com/{ticker}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'class': 'tinytable'})
        if not table:
            return None
        rows = table.find_all('tr')[1:]  # Skip header row

        num_purchases_month = 0
        num_sales_month = 0
        total_value_purchases_month = 0
        total_value_sales_month = 0

        num_purchases_quarter = 0
        num_sales_quarter = 0
        total_value_purchases_quarter = 0
        total_value_sales_quarter = 0

        for row in rows:
            cells = row.find_all('td')
            trade_type = cells[6].text.strip()
            trade_date = pd.to_datetime(cells[1].text.strip())
            value = float(cells[11].text.strip().replace('$', '').replace(',', ''))

            days_since_trade = (datetime.now() - trade_date).days

            if days_since_trade <= 30:
                if trade_type == 'P - Purchase':
                    num_purchases_month += 1
                    total_value_purchases_month += value
                elif trade_type == 'S - Sale':
                    num_sales_month += 1
                    total_value_sales_month += value

            if days_since_trade <= 90:
                if trade_type == 'P - Purchase':
                    num_purchases_quarter += 1
                    total_value_purchases_quarter += value
                elif trade_type == 'S - Sale':
                    num_sales_quarter += 1
                    total_value_sales_quarter += value

        total_value_month = total_value_purchases_month + total_value_sales_month
        total_value_quarter = total_value_purchases_quarter + total_value_sales_quarter

        return {
            'num_purchases_month': num_purchases_month,
            'num_sales_month': num_sales_month,
            'total_value_month': total_value_month,
            'num_purchases_quarter': num_purchases_quarter,
            'num_sales_quarter': num_sales_quarter,
            'total_value_quarter': total_value_quarter,
        }