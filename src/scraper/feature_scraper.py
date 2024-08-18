import requests
from bs4 import BeautifulSoup
import pandas as pd
import sys
import os
from io import StringIO
import openpyxl
import yfinance as yf
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.scraper_helpers import parse_titles, clean_numeric_columns, calculate_technical_indicators, process_dates, aggregate_group, process_ticker_technical_indicators

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
        print(f"{len(self.data)} entries extracted!")
    
    def clean_table(self):
        columns_of_interest = ["Filing Date", "Trade Date", "Ticker", "Title", "Price", "Qty", "Owned", "ΔOwn", "Value"]
        self.data = self.data[columns_of_interest]
        self.data = process_dates(self.data)
        self.data = clean_numeric_columns(self.data)
        self.data = parse_titles(self.data)
        self.data.drop(columns=['Title', 'Trade Date'], inplace=True)
        # Group by Ticker and Filing Date, then aggregate
        self.data = aggregate_group(self.data)
        self.data['Filing Date'] = self.data['Filing Date'].dt.strftime('%d-%m-%Y %H:%M')
        # Drop rows with any NaN in the specified columns
        self.data.dropna(inplace=True)
        print(f"{len(self.data)} entries remained after cleaning!")
    
    def add_technical_indicators(self):
        """Add technical indicators to the DataFrame with parallel processing and a progress bar."""
        # Convert DataFrame to a list of dictionaries for easier processing with multiprocessing
        rows = self.data.to_dict('records')
        
        # Initialize a pool with a number of processes equal to the CPU count
        with Pool(cpu_count()) as pool:
            # Use tqdm to show a progress bar while processing
            processed_rows = list(tqdm(pool.imap(process_ticker_technical_indicators, rows), total=len(rows)))
        
        # Filter out None values (rows where stock data was not found)
        self.data = pd.DataFrame(filter(None, processed_rows))
        self.data.dropna(inplace=True)
        print(f"{len(self.data)} entries remained after reading stock data!")
    
    def save_to_excel(self, file_path='output.xlsx'):
        """Save the self.data DataFrame to an Excel file."""
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        os.makedirs(data_dir, exist_ok=True)  # Create the data directory if it doesn't exist
        
        file_path = os.path.join(data_dir, file_path)
        if not self.data.empty:
            try:
                self.data.to_excel(file_path, index=False)
                print(f"Data successfully saved to {file_path}.")
            except Exception as e:
                print(f"Failed to save data to Excel: {e}")
        else:
            print("No data to save.")
        
    def extract_features(self):
        html = self.get_html(scraper.base_url)
        self.parse_table(html)
        
    def process_features(self):
        self.clean_table()
        self.add_technical_indicators()
        # add financial ratios
        # add company purchases/sales
        
if __name__ == "__main__":
    scraper = FeatureScraper()
    
    scraper.extract_features()
    scraper.save_to_excel('raw/features.xlsx')
    
    scraper.process_features()
    scraper.save_to_excel('interim/features.xlsx')
