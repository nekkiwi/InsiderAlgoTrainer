import requests
from bs4 import BeautifulSoup
import pandas as pd
import sys
import os
import openpyxl
import yfinance as yf
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay

from .utils.feature_scraper_helpers import *
from .utils.technical_indicators_helpers import *
from .utils.financial_ratios_helpers import *

class FeatureScraper:
    def __init__(self):
        self.base_url = "http://openinsider.com/screener?"
        self.data = pd.DataFrame()
        
    def process_web_page(self, date_range):
        start_date, end_date = date_range
        url = f"{self.base_url}pl=1&ph=&ll=&lh=&fd=-1&fdr={start_date.month}%2F{start_date.day}%2F{start_date.year}+-+{end_date.month}%2F{end_date.day}%2F{end_date.year}&td=0&tdr=&fdlyl=&fdlyh=&daysago=&xp=1&vl=10&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=1000&page=1"
        return fetch_and_parse(url)

    def fetch_data_from_pages(self, num_months):
        end_date = datetime.now() - timedelta(days=30)  # Start 1 month ago
        date_ranges = []

        # Prepare the date ranges
        for _ in range(num_months):
            start_date = end_date - timedelta(days=30)  # Each range is 1 month
            date_ranges.append((start_date, end_date))
            end_date = start_date  # Move back another month

        # Use multiprocessing to fetch and parse data in parallel
        with Pool(cpu_count()) as pool:
            data_frames = list(tqdm(pool.imap(self.process_web_page, date_ranges), total=len(date_ranges)))

        # Filter out None values (pages where no valid table was found)
        data_frames = [df for df in data_frames if df is not None]

        if data_frames:
            self.data = pd.concat(data_frames, ignore_index=True)
            print(f"{len(self.data)} total entries extracted from pages!")
        else:
            print("No data could be extracted.")
    
    def clean_table(self):
        columns_of_interest = ["Filing Date", "Trade Date", "Ticker", "Title", "Price", "Qty", "Owned", "ΔOwn", "Value"]
        self.data = self.data[columns_of_interest]
        self.data = process_dates(self.data)
        
        # Filter out entries where Filing Date is less than 20 business days in the past
        cutoff_date = pd.to_datetime('today') - pd.tseries.offsets.BDay(25)
        self.data = self.data[self.data['Filing Date'] < cutoff_date]
        
        # Clean numeric columns
        self.data = clean_numeric_columns(self.data)
        
        # Drop rows where ΔOwn is negative
        self.data = self.data[self.data['ΔOwn'] >= 0]
        
        # Parse titles
        self.data = parse_titles(self.data)
        self.data.drop(columns=['Title', 'Trade Date'], inplace=True)
        
        # Show the number of unique Ticker - Filing Date combinations
        unique_combinations = self.data[['Ticker', 'Filing Date']].drop_duplicates().shape[0]
        print(f"\nNumber of unique Ticker - Filing Date combinations before aggregation: {unique_combinations}")
        
        # Group by Ticker and Filing Date, then aggregate
        self.data = aggregate_group(self.data)
        
        # Format the date column and drop any remaining rows with missing values
        self.data['Filing Date'] = self.data['Filing Date'].dt.strftime('%d-%m-%Y %H:%M')
        self.data.dropna(inplace=True)
        
        print(f"{len(self.data)} entries remained after cleaning and aggregating!")

        
    def add_technical_indicators(self):
        rows = self.data.to_dict('records')
        
        # Apply technical indicators
        with Pool(cpu_count()) as pool:
            processed_rows = list(tqdm(pool.imap(process_ticker_technical_indicators, rows), total=len(rows)))
        
        self.data = pd.DataFrame(filter(None, processed_rows))
        
        # Replace infinite values and drop rows with missing values
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Log missing values before dropping NaNs
        log_missing_values(self.data, "before dropping rows after adding technical indicators")
        
        # Drop rows with missing values
        self.data.dropna(inplace=True)
        
        log_missing_values(self.data, "after dropping rows due to missing technical indicators")
        
        print(f"{len(self.data)} entries remained after adding technical indicators!")


    def add_financial_ratios(self):
        rows = self.data.to_dict('records')
        
        # Apply financial ratios
        with Pool(cpu_count()) as pool:
            processed_rows = list(tqdm(pool.imap(process_ticker_financial_ratios, rows), total=len(rows)))
        
        self.data = pd.DataFrame(filter(None, processed_rows))
        
        # Add sector dummies and drop the Sector column
        sector_dummies = pd.get_dummies(self.data['Sector'], prefix='Sector', dtype=int)
        self.data = pd.concat([self.data, sector_dummies], axis=1)
        self.data.drop(columns=['Sector'], inplace=True)
        
        # Log missing values before dropping NaNs
        log_missing_values(self.data, "before dropping rows after adding financial ratios")
        
        # Drop rows with missing values
        self.data.dropna(inplace=True)
        
        log_missing_values(self.data, "after dropping rows due to missing financial ratios")
        
        print(f"{len(self.data)} entries remained after adding financial ratios!")


    def add_insider_transactions(self):
        rows = self.data.to_dict('records')
        
        # Fetch insider transactions
        with Pool(cpu_count()) as pool:
            processed_rows = list(tqdm(pool.imap(get_recent_trades, [row['Ticker'] for row in rows]), total=len(rows)))
        
        for row, trade_data in zip(rows, processed_rows):
            if trade_data:
                row.update(trade_data)
        
        self.data = pd.DataFrame(rows)
        
        # Log missing values before dropping NaNs
        log_missing_values(self.data, "before dropping rows after adding insider transactions")
        
        # Drop rows with missing values
        self.data.dropna(inplace=True)
        
        log_missing_values(self.data, "after dropping rows due to missing insider transactions")
        
        print(f"{len(self.data)} entries remained after adding insider transactions!")

        
    def save_feature_distribution(self, output_file='feature_distribution.xlsx'):
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
        
    def run(self, num_months):
        self.fetch_data_from_pages(num_months)
        self.save_to_excel('interim/0_features_raw.xlsx')
        self.clean_table()
        self.save_to_excel('interim/1_features_formatted.xlsx')
        self.add_technical_indicators()
        self.save_to_excel('interim/2_features_TI.xlsx')
        self.add_financial_ratios()
        self.save_to_excel('interim/3_features_TI_FR.xlsx')
        self.add_insider_transactions()
        self.save_to_excel('interim/4_features_TI_FR_IT.xlsx')
        self.save_feature_distribution('output/feature_distribution.xlsx')
        
if __name__ == "__main__":
    feature_scraper = FeatureScraper()
    feature_scraper.run(num_months=12)
    
