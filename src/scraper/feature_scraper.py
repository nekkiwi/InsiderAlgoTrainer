import os
import time
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np

from .utils.feature_scraper_helpers import *
from .utils.technical_indicators_helpers import *
from .utils.financial_ratios_helpers import *

class FeatureScraper:
    def __init__(self):
        self.base_url = "http://openinsider.com/screener?"
        self.data = pd.DataFrame()
        self.train = False
        
    def process_web_page(self, date_range):
        start_date, end_date = date_range
        url = f"{self.base_url}pl=1&ph=&ll=&lh=&fd=-1&fdr={start_date.month}%2F{start_date.day}%2F{start_date.year}+-+{end_date.month}%2F{end_date.day}%2F{end_date.year}&td=0&tdr=&fdlyl=&fdlyh=&daysago=&xp=1&vl=10&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=1000&page=1"
        return fetch_and_parse(url)

    def fetch_data_from_pages(self, num_weeks):
        start_days_ago = 0
        if self.train:
            start_days_ago = 30
        end_date = datetime.datetime.now() - datetime.timedelta(days=start_days_ago)  # Start 1 month ago
        date_ranges = []

        # Prepare the date ranges
        for _ in range(num_weeks):
            start_date = end_date - datetime.timedelta(days=7)  # Each range is 1 week
            date_ranges.append((start_date, end_date))
            end_date = start_date  # Move back another week

        # Use multiprocessing to fetch and parse data in parallel
        with Pool(cpu_count()) as pool:
            data_frames = list(tqdm(pool.imap(self.process_web_page, date_ranges), total=len(date_ranges), desc="- Scraping entries from openinsider.com for each week"))

        # Filter out None values (pages where no valid table was found)
        data_frames = [df for df in data_frames if df is not None]

        if data_frames:
            self.data = pd.concat(data_frames, ignore_index=True)
            print(f"- {len(self.data)} total entries extracted!")
        else:
            print("- No data could be extracted.")
    
    def clean_table(self, drop_threshold=0.05):
        columns_of_interest = ["Filing Date", "Trade Date", "Ticker", "Title", "Price", "Qty", "Owned", "ΔOwn", "Value"]
        self.data = self.data[columns_of_interest]
        # This function correctly converts the column to datetime objects initially
        self.data = process_dates(self.data)
        
        # Filter out entries where Filing Date is less than 20 business days in the past
        if self.train:
            cutoff_date = pd.to_datetime('today') - pd.tseries.offsets.BDay(25)
        else:
            cutoff_date = pd.to_datetime('today')
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
        print(f"- Number of unique Ticker - Filing Date combinations before aggregation: {unique_combinations}")
        
        # Group by Ticker and Filing Date, then aggregate
        self.data = aggregate_group(self.data)
        
        # --- THIS IS THE KEY FIX ---
        # Ensure 'Filing Date' remains a datetime object and is not converted to a string.
        # We remove any string formatting like .dt.strftime().
        self.data['Filing Date'] = pd.to_datetime(self.data['Filing Date'])
        
        # Clean the data by dropping columns and rows with missing values
        self.data = clean_data(self.data, drop_threshold)


        
    def add_technical_indicators(self, drop_threshold=0.05):
        rows = self.data.to_dict('records')
        
        # Apply technical indicators
        with Pool(cpu_count()) as pool:
            processed_rows = list(tqdm(pool.imap(process_ticker_technical_indicators, rows), total=len(rows), desc="- Scraping technical indicators"))
        
        self.data = pd.DataFrame(filter(None, processed_rows))
        
        # Replace infinite values and drop rows with missing values
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Clean the data by dropping columns with more than 5% missing values and then dropping rows with missing values
        self.data = clean_data(self.data, drop_threshold)


    def add_financial_ratios(self, drop_threshold=0.2):
        """
        Fetches financial ratios and merges them using a robust strategy
        to ensure the merge keys align correctly.
        """
        if self.data.empty:
            print("- Data is empty, skipping financial ratio processing.")
            return

        print(f"[INFO] Fetching financial ratios for {len(self.data)} entries...")
        ratios_df = batch_fetch_financial_data(self.data[['Ticker', 'Filing Date']])

        if ratios_df.empty:
            print("- Could not fetch any financial ratios. Continuing without new data.")
            return

        # --- Robust Merge Preparation ---
        # 1. Ensure the merge key ('Filing Date') is a consistent datetime type in both DataFrames.
        #    This is the most critical step to prevent key misalignment and merge failures.
        self.data['Filing Date'] = pd.to_datetime(self.data['Filing Date'])
        ratios_df['Filing Date'] = pd.to_datetime(ratios_df['Filing Date'])

        # --- End of Preparation ---

        # 2. Perform the merge. It will now align correctly because the keys are of the same type.
        print(f"[INFO] Merging {len(ratios_df)} new financial ratios into the dataset.")
        self.data = pd.merge(self.data, ratios_df, on=['Ticker', 'Filing Date'], how='left')
        
        # 3. Process and clean the newly merged data.
        if 'Sector' in self.data.columns:
            sector_dummies = pd.get_dummies(self.data['Sector'], prefix='Sector', dtype=int)
            self.data = pd.concat([self.data, sector_dummies], axis=1)
            self.data.drop(columns=['Sector'], inplace=True)
        
        self.data = clean_data(self.data, drop_threshold)
        print(f"[INFO] Financial ratio processing complete. DataFrame now has {len(self.data.columns)} columns.")



    def add_insider_transactions(self, drop_threshold=0.05):
        rows = self.data.to_dict('records')
        
        tasks = [
            (row['Ticker'], row['Filing Date']) 
            for row in rows
        ]
        
        with Pool(2) as pool: # Using a reduced number of workers
            # We call pool.imap with our new helper function, get_recent_trades_star.
            # tqdm can now correctly iterate as each task is completed.
            processed_rows = list(tqdm(pool.imap(get_recent_trades_star, tasks, chunksize=1), total=len(tasks), desc="- Scraping recent insider trades"))
                
        for row, trade_data in zip(rows, processed_rows):
            if trade_data:
                row.update(trade_data)
        
        self.data = pd.DataFrame(rows)
        
        # Clean the data by dropping columns with more than 5% missing values and then dropping rows with missing values
        self.data = clean_data(self.data, drop_threshold)
    
        
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
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        summary_df.to_excel(output_file, sheet_name='Feature Distribution')
        print(f"- Feature distribution summary saved to {output_file}.")
    
    def save_to_excel(self, file_path='output.xlsx'):
        """Save the self.data DataFrame to an Excel file."""
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        file_path = os.path.join(data_dir, file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Create the data directory if it doesn't exist
        if not self.data.empty:
            try:
                self.data.to_excel(file_path, index=False)
                print(f"- Data successfully saved to {file_path}.\n")
            except Exception as e:
                print(f"- Failed to save data to Excel: {e}")
        else:
            print("- No data to save.")
            
    def load_sheet(self, file_path='output.xlsx'):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        file_path = os.path.join(data_dir, file_path)

        if os.path.exists(file_path):
            try:
                self.data = pd.read_excel(file_path)
                time.sleep(1)
                print(f"- Sheet successfully loaded from {file_path}.")
            except Exception as e:
                print(f"- Failed to load sheet from {file_path}: {e}")
        else:
            print(f"- File '{file_path}' does not exist.")
        
    def run(self, num_weeks, train):
        start_time = time.time()
        print("\n### START ### Feature Scraper")
        self.train = train
        self.fetch_data_from_pages(num_weeks)
        if train: self.save_to_excel(f'interim/train/0_features_raw.xlsx')
        
        self.clean_table(drop_threshold=0.05)
        if train: self.save_to_excel(f'interim/train/1_features_formatted.xlsx')
        
        self.add_technical_indicators(drop_threshold=0.05)
        if train: self.save_to_excel(f'interim/train/2_features_TI.xlsx')
        
        self.load_sheet(f'interim/train/2_features_TI.xlsx')
        
        self.add_financial_ratios(drop_threshold=0.5)
        if train: self.save_to_excel(f'interim/train/3_features_TI_FR.xlsx')
        
        # self.load_sheet(f'interim/train/3_features_TI_FR.xlsx')
        
        # self.add_insider_transactions(drop_threshold=0.05)
        if train: 
            # self.save_to_excel(f'interim/train/4_features_TI_FR_IT.xlsx')
            self.save_feature_distribution('analysis/feature_distribution.xlsx')
        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"### END ### Feature Scraper - time elapsed: {elapsed_time}")
        return self.data
        
if __name__ == "__main__":
    feature_scraper = FeatureScraper()
    feature_scraper.run(num_weeks=12*4)
    
