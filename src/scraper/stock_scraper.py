import os
import time
from datetime import timedelta
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from .utils.stock_scraper_helpers import *

class StockDataScraper:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.features_file      = os.path.join(data_dir, 'interim/5_features_full_cleaned.xlsx')
        self.features_out_file  = os.path.join(data_dir, 'final/features_final.xlsx')
        self.output_file        = os.path.join(data_dir, 'final/stock_data_final.xlsx')
        self.ticker_filing_dates    = None
        self.return_df              = None
        self.alpha_df               = None
        self.max_days = 20

    def prepare_data(self):
        self.features_df['Filing Date'] = pd.to_datetime(self.features_df['Filing Date'], dayfirst=True)
        self.ticker_filing_dates = self.features_df[['Ticker', 'Filing Date']]

    def create_stock_data_sheet(self):
        """Create return and alpha data sheets."""
        # 1) Build list of args for download
        ticker_info_list = [
            (row['Ticker'], row['Filing Date'], self.max_days)
            for _, row in self.ticker_filing_dates.iterrows()
        ]

        # 2) Download in parallel
        with Pool(cpu_count()) as pool:
            results = list(tqdm(
                pool.imap(download_data_wrapper, ticker_info_list),
                total=len(ticker_info_list),
                desc="- Downloading stock data"
            ))

        # 3) Process each result into a flat row‐dict
        rows = []
        for ticker, filing_date, max_days, stock_df, spy_df in results:
            # skip missing
            if stock_df is None or spy_df is None:
                continue

            # extract stock price series
            if stock_df.shape[1] == 1:
                sp = stock_df.iloc[:, 0]
            elif 'Close' in stock_df.columns:
                sp = stock_df['Close']
            elif 'Adj Close' in stock_df.columns:
                sp = stock_df['Adj Close']
            else:
                continue

            # extract SPY price series
            if spy_df.shape[1] == 1:
                bp = spy_df.iloc[:, 0]
            elif 'Close' in spy_df.columns:
                bp = spy_df['Close']
            elif 'Adj Close' in spy_df.columns:
                bp = spy_df['Adj Close']
            else:
                continue

            # sort and guard
            sp = sp.sort_index()
            bp = bp.sort_index()
            if sp.empty or bp.empty or sp.iloc[0] == 0 or bp.iloc[0] == 0:
                continue

            # compute returns & alpha
            sr = (sp / sp.iloc[0]) - 1
            br = (bp / bp.iloc[0]) - 1
            alpha = sr - br

            # build output dict
            row = {
                'Ticker':      ticker,
                'Filing Date': filing_date
            }
            for i in range(max_days):
                row[f'Day {i+1} Stock'] = float(sr.iloc[i]) if i < len(sr) else None
                row[f'Day {i+1} Alpha'] = float(alpha.iloc[i]) if i < len(alpha) else None

            rows.append(row)

        # 4) Ensure we got something
        if not rows:
            raise RuntimeError("No valid stock rows were produced — check your downloads")

        # 5) Build the DataFrame
        stock_data_df = pd.DataFrame(rows)

        # 6) Split into returns vs. alpha
        stock_cols = [c for c in stock_data_df.columns if 'Stock' in c]
        alpha_cols = [c for c in stock_data_df.columns if 'Alpha' in c]

        self.return_df = stock_data_df[['Ticker', 'Filing Date'] + stock_cols].copy()
        self.alpha_df  = stock_data_df[['Ticker', 'Filing Date'] + alpha_cols].copy()

    def save_to_excel(self):
        """Save the returns and alpha sheets to an Excel file."""
        save_to_excel(self.return_df, self.alpha_df, self.output_file)
        
    def save_final_features(self):
        save_final_features(self.features_df, self.features_out_file)

    def run(self, features_df):        
        start_time = time.time()
        print("\n### START ### Stock Scraper")
        
        if features_df is None:
            self.features_df = pd.read_excel(self.features_file)
        else:
            self.features_df = features_df
        self.prepare_data()
        self.create_stock_data_sheet()
        self.save_to_excel()
        self.save_final_features()
        
        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"### END ### Stock Scraper - time elapsed: {elapsed_time}")
        
        return self.features_df, self.return_df, self.alpha_df
