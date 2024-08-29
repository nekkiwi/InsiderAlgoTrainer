import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from matplotlib import pyplot as plt
from openpyxl import load_workbook
import sys

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.stock_analysis_helpers import filter_jumps, save_summary_statistics

class StockAnalysis:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.stock_returns_file = os.path.join(data_dir, 'final/stock_data_final.xlsx')
        self.simulations_dir = os.path.join(data_dir, 'training/simulation')
        self.base_output_dir = os.path.join(data_dir, 'output/stock_analysis')
        self.output_dir = ""
        self.backtest_dir = os.path.join(data_dir, 'backtest')
        self.model_name = None
        self.return_df = None
        self.all_summary_stats = None

    def load_data(self):
        """Load the stock returns data."""
        self.return_df = pd.read_excel(self.stock_returns_file, sheet_name='Returns')

    def filter_by_tickers(self, tickers_df):
        """Filter the data based on a provided DataFrame with tickers, filing dates, and optionally selling days."""
        if tickers_df is not None and self.return_df is not None:
            self.return_df = self.return_df.merge(tickers_df[['Ticker', 'Filing Date', 'Selling Day']], on=['Ticker', 'Filing Date'], how='inner')
            self.apply_selling_day()

    def apply_selling_day(self):
        """Adjust return data after the selling day to be equal to the return on the selling day."""
        if self.return_df is not None:
            selling_days = self.return_df['Selling Day'].dropna().astype(int)
            for day in range(20, max(selling_days) - 1, -1):
                self.return_df[f'Day {day}'] = np.where(self.return_df['Selling Day'] == day, self.return_df[f'Day {day}'], self.return_df[f'Day {day-1}'])
            self.return_df.drop(columns=['Selling Day'], inplace=True)

    def plot_combined(self):
        """Plot combined figures with 5th and 95th percentiles, mean, median, and all stock returns."""
        if self.return_df is None:
            print("No data available to plot.")
            return

        returns_data = self.return_df.drop(columns=['Ticker', 'Filing Date'])
        
        # Calculate summary statistics
        mean_returns = returns_data.mean()
        median_returns = returns_data.median()
        lower_bound = returns_data.quantile(0.05)
        upper_bound = returns_data.quantile(0.95)
        
        # Set up the plot with three subplots
        _, axs = plt.subplots(1, 3, figsize=(36, 8))  # Adjust the figsize to accommodate three plots
        x_values = np.arange(1, len(mean_returns) + 1)
        
        # Plot all stock returns overlayed (Plot 1)
        axs[0].plot(x_values, returns_data.T, color='gray', alpha=0.5)
        axs[0].axhline(y=0, color='red', linestyle='--', label='0% Return')
        axs[0].set_title('Overlay of All Stock Returns')
        axs[0].set_xlabel('Day')
        
        # Plot the 5th to 95th percentile range as a shaded area (Plot 2)
        axs[1].fill_between(x_values, lower_bound, upper_bound, color='gray', alpha=0.3, label='5th-95th Percentile')
        axs[1].plot(x_values, mean_returns, color='blue', label='Mean Return')
        axs[1].plot(x_values, median_returns, color='orange', label='Median Return')
        axs[1].axhline(y=0, color='red', linestyle='--', label='0% Return')
        axs[1].set_title('Stock Returns Percentiles, Mean & Median')
        axs[1].set_xlabel('Day')
        axs[1].set_ylabel('Return')
        axs[1].legend(loc='best')
        axs[1].set_ylim(-0.3, 0.3)

        # Plot the mean and median returns only (Plot 3)
        axs[2].plot(x_values, mean_returns, color='blue', label='Mean Return')
        axs[2].plot(x_values, median_returns, color='orange', label='Median Return')
        axs[2].axhline(y=0, color='red', linestyle='--', label='0% Return')
        axs[2].set_title('Mean & Median Returns')
        axs[2].set_xlabel('Day')
        axs[2].set_ylim(-0.1, 0.1)

        # Save the plot
        output_path = os.path.join(self.output_dir, 'stock_returns_combined.png')
        plt.savefig(output_path, dpi=300)
        plt.close()

    def calculate_backtest_statistics(self, tickers_df, limit, stop, unfiltered_stats=None, all_stock_stats=None):
        """Calculate backtest statistics for the given tickers."""

        # Align indices of tickers_df and self.return_df
        aligned_index = tickers_df.set_index(['Ticker', 'Filing Date']).index
        self.return_df = self.return_df.set_index(['Ticker', 'Filing Date'])

        # Filtered stats: where Criterion Pass is 1
        pred_filtered_df = self.return_df.loc[aligned_index][tickers_df['Pred Criterion Pass'] == 1]
        pred_filtered_day_20_returns = pred_filtered_df['Day 20']
        
        # Filtered stats: where GT Criterion Pass is 1
        gt_filtered_df = self.return_df.loc[aligned_index][tickers_df['GT Criterion Pass'] == 1]
        gt_filtered_day_20_returns = gt_filtered_df['Day 20']

        # Unfiltered stats: all tickers in the simulation
        unfiltered_day_20_returns = self.return_df.loc[aligned_index]['Day 20']

        pred_filtered_tickers_df = tickers_df[tickers_df['Pred Criterion Pass'] == 1]
        gt_filtered_tickers_df = tickers_df[tickers_df['GT Criterion Pass'] == 1]
        
        # Confusion matrix components
        tp = ((tickers_df['Pred Criterion Pass'] == 1) & (tickers_df['GT Criterion Pass'] == 1)).sum()
        tn = ((tickers_df['Pred Criterion Pass'] == 0) & (tickers_df['GT Criterion Pass'] == 0)).sum()
        fp = ((tickers_df['Pred Criterion Pass'] == 1) & (tickers_df['GT Criterion Pass'] == 0)).sum()
        fn = ((tickers_df['Pred Criterion Pass'] == 0) & (tickers_df['GT Criterion Pass'] == 1)).sum()

        statistics = {
            "Limit": limit,
            "Stop": stop,
            "Criterion TP": tp,
            "Criterion FP": fp,
            "Criterion FN": fn,
            "Criterion TN": tn,
            "Pred Median Return on Day 20": pred_filtered_day_20_returns.median(),
            "Pred Mean Return on Day 20": pred_filtered_day_20_returns.mean(),
            "Pred # Limit Filtered": (pred_filtered_tickers_df['Selling Day'] == pred_filtered_tickers_df['Limit Day']).sum(),
            "Pred # Stop Filtered": (pred_filtered_tickers_df['Selling Day'] == pred_filtered_tickers_df['Stop Day']).sum(),
            "Pred # Timeout Filtered": ((pred_filtered_tickers_df['Selling Day'] == 20) & (pred_filtered_tickers_df['Limit Day'] != 20)).sum(),
            "Pred Average Limit Price": pred_filtered_tickers_df['Limit Price'].mean(),
            "Pred Average Limit Day": pred_filtered_tickers_df['Limit Day'].mean(),
            "Pred Average Stop Price": pred_filtered_tickers_df['Stop Price'].mean(),
            "Pred Average Stop Day": pred_filtered_tickers_df['Stop Day'].mean(),
            "GT Median Return on Day 20": gt_filtered_day_20_returns.median(),
            "GT Mean Return on Day 20": gt_filtered_day_20_returns.mean(),
            "GT # Limit": (gt_filtered_tickers_df['Selling Day'] == gt_filtered_tickers_df['Limit Day']).sum(),
            "GT # Stop": (gt_filtered_tickers_df['Selling Day'] == gt_filtered_tickers_df['Stop Day']).sum(),
            "GT # Timeout": ((gt_filtered_tickers_df['Selling Day'] == 20) & (gt_filtered_tickers_df['Limit Day'] != 20)).sum(),
            "GT Average Limit Price": gt_filtered_tickers_df['Limit Price'].mean(),
            "GT Average Limit Day": gt_filtered_tickers_df['Limit Day'].mean(),
            "GT Average Stop Price": gt_filtered_tickers_df['Stop Price'].mean(),
            "GT Average Stop Day": gt_filtered_tickers_df['Stop Day'].mean(),
            "All-Limit-Stop Median Return on Day 20": unfiltered_day_20_returns.median(),
            "All-Limit-Stop Mean Return on Day 20": unfiltered_day_20_returns.mean(),
            "All-Limit-Stop # Limit": (tickers_df['Selling Day'] == tickers_df['Limit Day']).sum(),
            "All-Limit-Stop # Stop": (tickers_df['Selling Day'] == tickers_df['Stop Day']).sum(),
            "All-Limit-Stop # Timeout": ((tickers_df['Selling Day'] == 20) & (tickers_df['Limit Day'] != 20)).sum(),
            "All-Limit-Stop Average Limit Price": tickers_df['Limit Price'].mean(),
            "All-Limit-Stop Average Limit Day": tickers_df['Limit Day'].mean(),
            "All-Limit-Stop Average Stop Price": tickers_df['Stop Price'].mean(),
            "All-Limit-Stop Average Stop Day": tickers_df['Stop Day'].mean(),
            "All-Raw Median Return on Day 20": all_stock_stats['median'] if all_stock_stats else np.nan,
            "All-Raw Mean Return on Day 20": all_stock_stats['mean'] if all_stock_stats else np.nan,
        }
        return statistics
    
    def calculate_unfiltered_stats(self):
        """Calculate mean and median returns at day 20 for all tickers without any filtering."""
        return {
            "mean": self.return_df['Day 20'].mean(),
            "median": self.return_df['Day 20'].median()
        }

    def calculate_all_stock_stats(self):
        """Calculate mean and median returns at day 20 for all stock history."""
        mean_return = self.all_summary_stats.loc['Day 20', 'mean'] / 100
        median_return = self.all_summary_stats.loc['Day 20', '50%'] / 100
        return {
            "mean": mean_return,
            "median": median_return
        }

    def save_summary_statistics(self, df, output_dir, sheet_name):
        """Save summary statistics (min, 25%, median, mean, 75%, max) for each timestep to an Excel sheet."""
        returns_data = df.drop(columns=['Ticker', 'Filing Date'])
        summary_stats = returns_data.describe(percentiles=[0.25, 0.5, 0.75]).T
        summary_stats = summary_stats[['min', '25%', '50%', 'mean', '75%', 'max']] * 100

        output_path = os.path.join(output_dir, 'stock_returns_summary_stats.xlsx')
        os.makedirs(output_dir, exist_ok=True)
        with pd.ExcelWriter(output_path, engine='openpyxl', mode='a' if os.path.exists(output_path) else 'w') as writer:
            if sheet_name in writer.book.sheetnames:
                del writer.book[sheet_name]
            summary_stats.to_excel(writer, sheet_name=sheet_name, index_label='Day')

    def run(self, tickers_file=None):
        """Run the stock return analysis and plot generation."""

        self.load_data()

        if tickers_file is not None:
            self.all_summary_stats = pd.read_excel(os.path.join(self.base_output_dir, 'all', 'stock_returns_summary_stats.xlsx'), index_col=0)
            tickers_df = pd.read_excel(tickers_file)
            sheet_name = pd.ExcelFile(tickers_file).sheet_names[0]
            self.output_dir = os.path.join(self.base_output_dir, sheet_name)
            os.makedirs(self.output_dir, exist_ok=True)
            self.filter_by_tickers(tickers_df)
            
            # Filtered by Criterion Pass
            pred_filtered_return_df = self.return_df[tickers_df['Pred Criterion Pass'] == 1]
            self.save_summary_statistics(pred_filtered_return_df, self.output_dir, sheet_name='Filtered_by_Criterion_Pass')

            # Filtered by GT columns
            gt_filtered_return_df = self.return_df[tickers_df['GT Criterion Pass'] == 1]
            self.save_summary_statistics(gt_filtered_return_df, self.output_dir, sheet_name='Filtered_by_GT')
        else:
            self.output_dir = os.path.join(self.base_output_dir, 'all')

        # Unfiltered statistics
        self.save_summary_statistics(self.return_df, self.output_dir, sheet_name='Unfiltered')

        filter_jumps(self)
        self.plot_combined()

    def process_simulation_file(self, args):
        file_name, unfiltered_stats, all_stock_stats = args
        tickers_file = os.path.join(self.simulations_dir, self.model_name, file_name)
        self.run(tickers_file)
        limit, stop = self.extract_limit_stop_from_filename(file_name)
        tickers_df = pd.read_excel(tickers_file)
        return self.calculate_backtest_statistics(tickers_df, limit, stop, unfiltered_stats, all_stock_stats)

    def run_all_simulations(self, model_name, criterion):
        """Run stock return analysis for all simulations of a specific model and generate a backtest report."""
        self.model_name = model_name.replace(' ','-').lower()
        self.load_data()
        if self.return_df is None:
            print("No data loaded. Exiting.")
            return

        model_dir = os.path.join(self.simulations_dir, self.model_name)
        if not os.path.exists(model_dir):
            print(f"No simulations found for model: {self.model_name}")
            return

        unfiltered_stats = self.calculate_unfiltered_stats()
        all_stock_stats = self.calculate_all_stock_stats()

        simulation_files = [file for file in os.listdir(model_dir) if file.endswith('.xlsx')]

        # Prepare arguments for the pool
        pool_args = [(file_name, unfiltered_stats, all_stock_stats) for file_name in simulation_files]

        # Use multiprocessing Pool to parallelize the simulation process
        with Pool(cpu_count()) as pool:
            backtest_results = list(tqdm(pool.imap(self.process_simulation_file, pool_args),
                                         total=len(simulation_files),
                                         desc=f"Processing simulations for {self.model_name}"))

        # Save backtest results
        backtest_df = pd.DataFrame(backtest_results)
        os.makedirs(self.backtest_dir, exist_ok=True)
        backtest_file = os.path.join(self.backtest_dir, f'{self.model_name}_{criterion}_backtest.xlsx')
        backtest_df.to_excel(backtest_file, index=False)
        print(f"Backtest results saved to {backtest_file}")

    def extract_limit_stop_from_filename(self, filename):
        """Extract limit and stop values from the filename."""
        parts = filename.split('_')
        limit = float(parts[2].replace('l', ''))
        stop = float(parts[3].replace('.xlsx', '').replace('s', ''))
        return limit, stop

if __name__ == "__main__":
    analysis = StockAnalysis()
    model_name = "randomforest"
    criterion = 'limit-stop'
    analysis.run_all_simulations(model_name, criterion)
