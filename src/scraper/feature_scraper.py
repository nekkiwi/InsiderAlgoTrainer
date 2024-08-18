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

from utils.feature_scraper_helpers import parse_titles, clean_numeric_columns, process_dates, aggregate_group
from utils.technical_indicators_helpers import process_ticker_technical_indicators
from utils.financial_ratios_helpers import process_ticker_financial_ratios

class FeatureScraper:
    def __init__(self, base_url=None):
        self.base_url = base_url or "http://openinsider.com/screener?s=&o=&pl=1&ph=&ll=&lh=&fd=0&fdr=&td=0&tdr=&fdlyl=&fdlyh=&daysago=&xp=1&vl=10&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=1000&page="
        self.data = pd.DataFrame()

    def get_html(self, page_num):
        url = self.base_url + str(page_num)
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def parse_table(self, html):
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", {"class": "tinytable"})
        df = pd.read_html(StringIO(str(table)))[0]
        # Clean up column names by replacing \xa0 with a regular space
        df.columns = df.columns.str.replace('\xa0', ' ', regex=False)
        return df
    
    def fetch_data_from_pages(self, num_pages=1):
        """Fetch and parse data from multiple pages in parallel."""
        with Pool(cpu_count()) as pool:
            # Fetch HTML from each page in parallel
            html_pages = list(tqdm(pool.imap(self.get_html, range(1, num_pages + 1)), total=num_pages))
        
        # Parse each HTML page and combine the DataFrames
        data_frames = [self.parse_table(html) for html in html_pages]
        self.data = pd.concat(data_frames, ignore_index=True)
        print(f"{len(self.data)} entries extracted from {num_pages} pages!")
    
    def clean_table(self):
        columns_of_interest = ["Filing Date", "Trade Date", "Ticker", "Title", "Price", "Qty", "Owned", "ΔOwn", "Value"]
        self.data = self.data[columns_of_interest]
        self.data = process_dates(self.data)
        # Filter out entries where Filing Date is less than 30 business days in the past
        cutoff_date = pd.to_datetime('today') - pd.tseries.offsets.BDay(20)
        self.data = self.data[self.data['Filing Date'] < cutoff_date]
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
        rows = self.data.to_dict('records')
        with Pool(cpu_count()) as pool:
            processed_rows = list(tqdm(pool.imap(process_ticker_technical_indicators, rows), total=len(rows)))
        self.data = pd.DataFrame(filter(None, processed_rows))
        self.data.dropna(inplace=True)
        print(f"{len(self.data)} entries remained after adding technical indicators!")
        
    def add_financial_ratios(self):
        rows = self.data.to_dict('records')
        with Pool(cpu_count()) as pool:
            processed_rows = list(tqdm(pool.imap(process_ticker_financial_ratios, rows), total=len(rows)))
        self.data = pd.DataFrame(filter(None, processed_rows))
        self.data.dropna(inplace=True)
        print(f"{len(self.data)} entries remained after adding financial ratios!")
    
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
        
    def run(self, num_pages=1):
        self.fetch_data_from_pages(num_pages)
        self.clean_table()
        self.add_technical_indicators()
        self.add_financial_ratios()
        self.save_to_excel('interim/features.xlsx')
    
        # TODO add company purchases/sales
        # TODO add days since IPO
        # TODO give all files to chatGPT and tell it to make it production ready
        
if __name__ == "__main__":
    feature_scraper = FeatureScraper()
    feature_scraper.run(num_pages=5)
    
