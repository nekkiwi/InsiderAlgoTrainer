import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm import tqdm

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.stock_analysis_helpers import filter_jumps, save_summary_statistics

class StockAnalysis:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.stock_returns_file = os.path.join(data_dir, 'final/stock_data_final.xlsx')
        self.simulations_dir = os.path.join(data_dir, 'training/simulation')
        self.output_dir = os.path.join(data_dir, 'output/stock_analysis')
        self.backtest_dir = os.path.join(data_dir, 'backtest')
        self.model_name = None
        self.return_df = None
        self.all_summary_stats = None

    def load_data(self):
        """Load the stock returns data."""
        self.return_df = pd.read_excel(self.stock_returns_file, sheet_name='Returns')
        self.all_summary_stats = pd.read_excel(os.path.join(self.output_dir, 'all', 'stock_returns_summary_stats.xlsx'), index_col=0)

    def filter_by_tickers(self, tickers_df):
        """Filter the data based on a provided DataFrame with tickers, filing dates, and optionally selling days."""
        if tickers_df is not None:
            self.return_df = self.return_df.merge(tickers_df[['Ticker', 'Filing Date', 'Selling Day']], on=['Ticker', 'Filing Date'], how='inner')
            self.apply_selling_day()

    def apply_selling_day(self):
        """Adjust return data after the selling day to be equal to the return on the selling day."""
        for index, row in self.return_df.iterrows():
            selling_day = row['Selling Day']
            if pd.notna(selling_day):  # Only apply if Selling Day is not NaN
                selling_day = int(selling_day)  # Convert to integer
                # Adjust returns after the selling day
                for day in range(selling_day, 21):  # Assuming 20 days of data
                    self.return_df.loc[index, f'Day {day}'] = row[f'Day {selling_day}']

        # Drop the 'Selling Day' column as it's no longer needed
        self.return_df.drop(columns=['Selling Day'], inplace=True)
        
    def plot_combined(self):
        """Plot combined figures with 5th and 95th percentiles, mean, median, and all stock returns."""
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
        for _, row in returns_data.iterrows():
            axs[0].plot(x_values, row, color='gray', alpha=0.5)
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
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, 'stock_returns_combined.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        # print(f"Combined stock return figure saved at {output_path}")

    def calculate_backtest_statistics(self, tickers_df, limit, stop):
        """Calculate backtest statistics for the given tickers."""
        day_20_returns = self.return_df['Day 20']
        
        statistics = {
            "Limit": limit,
            "Stop": stop,
            "Number of Tickers passed condition (predicted)": len(tickers_df),
            "Number of Tickers passed condition (actual)": len(self.return_df),
            "Min Return on Day 20": day_20_returns.min(),
            "1st Quartile Return on Day 20": day_20_returns.quantile(0.25),
            "Median Return on Day 20": day_20_returns.median(),
            "Mean Return on Day 20": day_20_returns.mean(),
            "3rd Quartile Return on Day 20": day_20_returns.quantile(0.75),
            "Max Return on Day 20": day_20_returns.max(),
            "Day 20 Mean difference": day_20_returns.mean() - self.all_summary_stats.loc['Day 20', 'mean']/100,
            "Day 20 Median difference": day_20_returns.median() - self.all_summary_stats.loc['Day 20', '50%']/100,
            # "Number of Limit occurrences": (tickers_df['Selling Day'] == tickers_df['Limit Day']).sum(),
            # "Number of Stop occurrences": (tickers_df['Selling Day'] == tickers_df['Stop Day']).sum(),
            # "Number of Timeout occurrences": (tickers_df['Selling Day'].isna()).sum(),
            # "Average Limit Price": tickers_df['Limit Price'].mean(),
            # "Average Limit Day": tickers_df['Limit Day'].mean(),
            # "Average Stop Price": tickers_df['Stop Price'].mean(),
            # "Average Stop Day": tickers_df['Stop Day'].mean()
        }
        return statistics

    def run(self, tickers_file=None):
        """Run the stock return analysis and plot generation."""
        self.load_data()
        if tickers_file is not None:
            tickers_df = pd.read_excel(tickers_file)
            sheet_name = pd.ExcelFile(tickers_file).sheet_names[0]  # Get the first sheet name
            self.output_dir = os.path.join(self.output_dir, sheet_name)
            self.filter_by_tickers(tickers_df)
        else:
            self.output_dir = os.path.join(self.output_dir, 'all')

        filter_jumps(self)
        self.plot_combined()
        save_summary_statistics(self.return_df, self.output_dir, filtered=False)

    def run_all_simulations(self, model_name, criterion):
        """Run stock return analysis for all simulations of a specific model and generate a backtest report."""
        self.model_name = model_name.replace(' ','-').lower()
        model_dir = os.path.join(self.simulations_dir, self.model_name)
        if not os.path.exists(model_dir):
            print(f"No simulations found for model: {self.model_name}")
            return

        simulation_files = [file for file in os.listdir(model_dir) if file.endswith('.xlsx')]
        backtest_results = []

        # Use tqdm to add a progress bar
        for file_name in tqdm(simulation_files, desc=f"Processing simulations for {self.model_name}"):
            self.__init__()
            tickers_file = os.path.join(model_dir, file_name)
            self.run(tickers_file)
            limit, stop = self.extract_limit_stop_from_filename(file_name)
            tickers_df = pd.read_excel(tickers_file)
            statistics = self.calculate_backtest_statistics(tickers_df, limit, stop)
            backtest_results.append(statistics)

        # Save backtest results
        backtest_df = pd.DataFrame(backtest_results)
        os.makedirs(self.backtest_dir, exist_ok=True)
        backtest_file = os.path.join(self.backtest_dir, f'{model_name}_{criterion}_backtest.xlsx')
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

    # Example: Run analysis for all simulations of a specific model
    model_name = "randomforest"  # Change to the desired model name
    criterion = 'limit-stop'
    analysis.run_all_simulations(model_name, criterion)


    # TODO: parallelize this
    # TODO: Implement commented out sections in the files that use them