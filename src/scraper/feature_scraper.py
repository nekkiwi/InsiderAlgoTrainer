import requests
from bs4 import BeautifulSoup
import pandas as pd
import sys
import os
from io import StringIO
import yfinance as yf

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.date_helpers import get_next_market_open
from utils.scraper_helpers import parse_titles, clean_numeric_columns, calculate_technical_indicators
from utils.stock_helpers import download_stock_data

class FeatureScraper:
    def __init__(self):
        self.base_url = "http://openinsider.com/screener?s=&o=&pl=1&ph=&ll=&lh=&fd=0&fdr=&td=0&tdr=&fdlyl=&fdlyh=&daysago=&xp=1&vl=10&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=1000&page=1"
        self.data = pd.DataFrame()

    def get_html(self, url):
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def parse_table(self, html):
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", {"class": "tinytable"})
        self.data = pd.read_html(StringIO(str(table)))[0]
        # Clean up column names by replacing \xa0 with a regular space
        self.data.columns = self.data.columns.str.replace('\xa0', ' ', regex=False)
        # List of columns to extract
        columns_of_interest = ["Filing Date", "Trade Date", "Ticker", "Title", "Price", "Qty", "Owned", "ΔOwn", "Value"]
        # Filter the DataFrame to include only these columns
        self.data = self.data[columns_of_interest]
    
    def clean_table(self):
        # Convert date strings to datetime objects
        self.data['Filing Date'] = pd.to_datetime(self.data['Filing Date']).apply(get_next_market_open)
        self.data['Trade Date'] = pd.to_datetime(self.data['Trade Date'])
        
        # Calculate "Days Since Trade"
        self.data['Days Since Trade'] = (self.data['Filing Date'] - self.data['Trade Date']).dt.days
        
        self.data = clean_numeric_columns(self.data)
        self.data = parse_titles(self.data)
        
        self.data.drop(columns=['Title', 'Trade Date'], inplace=True)
        
    def add_technical_indicators(self):
        """Add technical indicators to the DataFrame."""
        def process_ticker(row):
            ticker = row['Ticker']
            stock_data = download_stock_data(ticker)
            if stock_data is None or stock_data.empty:
                print(f"Skipping ticker {ticker} due to missing data.")
                return row  # Return the row unchanged
            indicators = calculate_technical_indicators(stock_data)
            for key, value in indicators.items():
                if not value.empty:
                    row[key] = value.iloc[-1]  # Get the most recent value of each indicator
            return row

        # Apply the processing function to each row in the DataFrame
        self.data = self.data.apply(process_ticker, axis=1)
        return self.data
        
    def process_features(self):
        html = self.get_html(scraper.base_url)
        self.parse_table(html)
        self.clean_table()
        # add technical indicators
        # self.add_technical_indicators()
        # add financial ratios
        # add company purchases/sales
        
if __name__ == "__main__":
    scraper = FeatureScraper()
    scraper.process_features()
    print(scraper.data.head())
