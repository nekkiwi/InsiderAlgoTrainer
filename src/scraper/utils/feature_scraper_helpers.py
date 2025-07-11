import pandas as pd
import datetime
import requests
from bs4 import BeautifulSoup
from io import StringIO

def clean_data(df, threshold=0.05):
    """
    Clean the DataFrame by first dropping columns with more than the specified percentage
    of missing values, then dropping rows with any missing values.
    
    Args:
        df (pd.DataFrame): The DataFrame to clean.
        threshold (float): The percentage of missing values allowed in a column before it is dropped.
                           Default is 5% (i.e., 0.05).
    
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Step 1: Drop columns where more than `threshold` % of the entries are missing
    missing_percentage = df.isnull().mean()
    columns_to_drop = missing_percentage[missing_percentage > threshold].index
    df_cleaned = df.drop(columns=columns_to_drop)

    if not columns_to_drop.empty:
        print(f"- Dropped columns: {list(columns_to_drop)}, missing in more than {int(threshold*100)}% of entries")

    # Step 2: Drop any remaining rows with missing values
    df_cleaned.dropna(inplace=True)

    print(f"- Remaining rows after dropping missing values: {len(df_cleaned)}")
    
    return df_cleaned

def get_next_market_open(dt):
    # Assume market hours are 9:00 AM to 5:00 PM
    market_open = datetime.time(9, 0)
    market_close = datetime.time(17, 0)
    
    # If it's before 9:00 AM, move to 9:00 AM today
    if dt.time() < market_open:
        return dt.replace(hour=9, minute=0, second=0, microsecond=0)
    
    # If it's after 5:00 PM, move to 9:00 AM the next business day
    if dt.time() >= market_close:
        dt = dt + pd.tseries.offsets.BDay()
        return dt.replace(hour=9, minute=0, second=0, microsecond=0)
    
    # Round up to the next full or half hour
    minutes = dt.minute
    if minutes > 0 and minutes <= 30:
        return dt.replace(minute=30, second=0, microsecond=0)
    else:
        dt += pd.tseries.offsets.Hour()
        return dt.replace(minute=0, second=0, microsecond=0)

def get_html(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def parse_table(html):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"class": "tinytable"})
    if not table:
        return None
    df = pd.read_html(StringIO(str(table)))[0]
    # Clean up column names by replacing \xa0 with a regular space
    df.columns = df.columns.str.replace('\xa0', ' ', regex=False)
    return df

def fetch_and_parse(url):
    html = get_html(url)
    return parse_table(html)

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
    df['Title'] = df['Title'].fillna('')
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

import requests
import time
import pandas as pd
from bs4 import BeautifulSoup
from requests.exceptions import RequestException

def _robust_get(url, max_retries=3, backoff_factor=0.5, timeout=10):
    """
    Performs a GET request with a timeout and exponential backoff for retries.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            # Raise an error for bad status codes (4xx or 5xx)
            response.raise_for_status()
            return response
        except RequestException as e:
            if attempt < max_retries - 1:
                # Calculate wait time and retry
                wait = backoff_factor * (2 ** attempt)
                time.sleep(wait)
            else:
                # Log the final failure and return None
                print(f"Failed to fetch {url} after {max_retries} attempts: {e}")
                return None

def get_recent_trades(ticker: str, filing_date: pd.Timestamp):
    """
    Scrapes recent insider trades for a given ticker relative to a specific
    historical filing date.
    """
    url = f"http://openinsider.com/{ticker}"
    try:
        response = _robust_get(url)
        if response is None: return None

        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'class': 'tinytable'})
        if not table: return None
            
        rows = table.find_all('tr')[1:]

        # Counters remain the same
        num_purchases_month, num_sales_month = 0, 0
        total_value_purchases_month, total_value_sales_month = 0, 0
        num_purchases_quarter, num_sales_quarter = 0, 0
        total_value_purchases_quarter, total_value_sales_quarter = 0, 0

        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 12: continue
            
            trade_type = cells[6].text.strip()
            trade_date = pd.to_datetime(cells[1].text.strip())
            
            # --- THE CRITICAL FIX IS HERE ---
            # We only consider trades that happened ON OR BEFORE the historical filing_date
            if trade_date > filing_date:
                continue

            value_str = cells[11].text.strip().replace('$', '').replace(',', '')
            if not value_str: continue
            value = float(value_str)
            
            # Replace pd.Timestamp.now() with the passed-in filing_date
            days_since_trade = (filing_date - trade_date).days

            # Aggregate data for the last month (30 days prior to the filing_date)
            if days_since_trade <= 30:
                if 'Purchase' in trade_type:
                    num_purchases_month += 1
                    total_value_purchases_month += value
                elif 'Sale' in trade_type:
                    num_sales_month += 1
                    total_value_sales_month += value
            
            # Aggregate data for the last quarter (90 days prior to the filing_date)
            if days_since_trade <= 90:
                if 'Purchase' in trade_type:
                    num_purchases_quarter += 1
                    total_value_purchases_quarter += value
                elif 'Sale' in trade_type:
                    num_sales_quarter += 1
                    total_value_sales_quarter += value
        
        # The return dictionary remains the same
        return {
            'num_purchases_month': num_purchases_month,
            'num_sales_month': num_sales_month,
            'total_value_month': total_value_purchases_month + total_value_sales_month,
            'num_purchases_quarter': num_purchases_quarter,
            'num_sales_quarter': num_sales_quarter,
            'total_value_quarter': total_value_purchases_quarter + total_value_sales_quarter,
        }
    except Exception as e:
        # Catch any other unexpected errors and return None to prevent crashing the pool
        print(f"An unexpected error occurred for ticker {ticker}: {e}")
        return None
