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
import numpy as np

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.feature_scraper_helpers import parse_titles, clean_numeric_columns, process_dates, aggregate_group, get_recent_trades
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
    
    def fetch_and_parse(self, page_num):
        html = self.get_html(page_num)
        return self.parse_table(html)

    def fetch_data_from_pages(self, num_pages=1):
        with Pool(cpu_count()) as pool:
            data_frames = list(tqdm(pool.imap(self.fetch_and_parse, range(1, num_pages + 1)), total=num_pages))
        self.data = pd.concat(data_frames, ignore_index=True)
        print(f"{len(self.data)} entries extracted from {num_pages} pages!")
    
    def clean_table(self):
        columns_of_interest = ["Filing Date", "Trade Date", "Ticker", "Title", "Price", "Qty", "Owned", "ΔOwn", "Value"]
        self.data = self.data[columns_of_interest]
        self.data = process_dates(self.data)
        # Filter out entries where Filing Date is less than 30 business days in the past
        cutoff_date = pd.to_datetime('today') - pd.tseries.offsets.BDay(20)
        self.data = self.data[self.data['Filing Date'] < cutoff_date]
        print(f"{len(self.data)} entries remained after filtering for cutoff date > 20 business days ago!")
        self.data = clean_numeric_columns(self.data)
        self.data = parse_titles(self.data)
        self.data.drop(columns=['Title', 'Trade Date'], inplace=True)
        # Group by Ticker and Filing Date, then aggregate
        self.data = aggregate_group(self.data)
        self.data['Filing Date'] = self.data['Filing Date'].dt.strftime('%d-%m-%Y %H:%M')
        # Drop rows with any NaN in the specified columns
        self.data.dropna(inplace=True)
        print(f"{len(self.data)} entries remained after formatting and aggregating!")
    
    def add_technical_indicators(self):
        rows = self.data.to_dict('records')
        with Pool(cpu_count()) as pool:
            processed_rows = list(tqdm(pool.imap(process_ticker_technical_indicators, rows), total=len(rows)))
        self.data = pd.DataFrame(filter(None, processed_rows))
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data.dropna(inplace=True)
        print(f"{len(self.data)} entries remained after adding technical indicators!")

    def add_financial_ratios(self):
        rows = self.data.to_dict('records')
        with Pool(cpu_count()) as pool:
            processed_rows = list(tqdm(pool.imap(process_ticker_financial_ratios, rows), total=len(rows)))
        self.data = pd.DataFrame(filter(None, processed_rows))
        self.data.dropna(inplace=True)
        print(f"{len(self.data)} entries remained after adding financial ratios!")

    def add_insider_transactions(self):
        rows = self.data.to_dict('records')
        with Pool(cpu_count()) as pool:
            processed_rows = list(tqdm(pool.imap(get_recent_trades, [row['Ticker'] for row in rows]), total=len(rows)))
        for row, trade_data in zip(rows, processed_rows):
            if trade_data:
                row.update(trade_data)
        self.data = pd.DataFrame(rows)
        self.data.dropna(inplace=True)
        print(f"{len(self.data)} entries remained after adding insider transactions!")
        
    def save_feature_distribution(self, output_file='feature_distribution.xlsx'):
        """
        Summarizes the distribution of each column in the DataFrame and saves it as an Excel sheet.

        Args:
        - data (pd.DataFrame): The DataFrame containing the features to be summarized.
        - output_file (str): The path to the output Excel file.
        """
        # Define the quantiles and statistics to be calculated
        quantiles = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
        summary_df = pd.DataFrame()

        for column in self.data.columns:
            if pd.api.types.is_numeric_dtype(self.data[column]):
                stats = self.data[column].quantile(quantiles).to_dict()
                stats['mean'] = self.data[column].mean()
                summary_df[column] = pd.Series(stats)

        # Transpose the DataFrame so that each row is a feature
        summary_df = summary_df.T
        summary_df.columns = ['min', '1%', '5%', '10%', '25%', '50%', '75%', '90%', '95%', '99%', 'max', 'mean']

        # Save the summary to an Excel file
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        output_file = os.path.join(data_dir, output_file)
        
        summary_df.to_excel(output_file, sheet_name='Feature Distribution')
        print(f"Feature distribution summary saved to {output_file}.")
        
    def clip_non_categorical_features(self):
        """Clip non-categorical features at the 1st and 99th percentiles."""
        non_categorical_columns = self.data.select_dtypes(include=[np.number]).columns

        for column in non_categorical_columns:
            lower_bound = self.data[column].quantile(0.01)
            upper_bound = self.data[column].quantile(0.99)
            self.data[column] = self.data[column].clip(lower=lower_bound, upper=upper_bound)
            
        print("Clipped non-categorical features at the 1st and 99th percentiles.")

    def normalize_non_categorical_features(self):
        """Apply Min-Max Normalization to non-categorical features."""
        non_categorical_columns = self.data.select_dtypes(include=[np.number]).columns

        for column in non_categorical_columns:
            min_value = self.data[column].min()
            max_value = self.data[column].max()
            self.data[column] = (self.data[column] - min_value) / (max_value - min_value)
            
        print("Applied Min-Max Normalization to non-categorical features.")
    
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
            
    def load_sheet(self, file_path='output.xlsx'):
        """
        Load a specified sheet from an Excel file.

        Args:
        - file_path (str): The path to the Excel file.
        - sheet_name (str): The name of the sheet to load.

        Returns:
        - pd.DataFrame: The DataFrame containing the loaded sheet data.
        """
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        file_path = os.path.join(data_dir, file_path)

        if os.path.exists(file_path):
            try:
                self.data = pd.read_excel(file_path)
                print(f"Sheet successfully loaded from {file_path}.")
            except Exception as e:
                print(f"Failed to load sheet from {file_path}: {e}")
        else:
            print(f"File '{file_path}' does not exist.")
        
    def run(self, num_pages):
        self.fetch_data_from_pages(num_pages)
        self.save_to_excel('raw/features_raw.xlsx')
        self.clean_table()
        self.save_to_excel('interim/features_formatted.xlsx')
        self.add_technical_indicators()
        self.save_to_excel('interim/features_TI.xlsx')
        self.add_financial_ratios()
        self.save_to_excel('interim/features_TI_FR.xlsx')
        self.add_insider_transactions()
        self.save_to_excel('interim/features_TI_FR_IT.xlsx')
        self.save_feature_distribution('output/feature_distribution.xlsx')
        self.clip_non_categorical_features()
        self.save_to_excel('interim/features_TI_FR_IT_clip.xlsx')
        self.normalize_non_categorical_features()
        self.save_to_excel('processed/features_processed.xlsx')
        
if __name__ == "__main__":
    feature_scraper = FeatureScraper()
    feature_scraper.run(num_pages=5)
    
