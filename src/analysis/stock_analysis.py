import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

class StockAnalysis:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.stock_returns_file = os.path.join(data_dir, 'final/stock_data_final.xlsx')
        self.output_dir = os.path.join(data_dir, 'output/stock_analysis')
        self.return_df = None
        self.filtered_return_df = None

    def load_data(self):
        """Load the stock returns data."""
        self.return_df = pd.read_excel(self.stock_returns_file, sheet_name='Returns')
        
    def filter_jumps(self, max_jump=1):
        """Remove rows with jumps (consecutive differences) higher than a specified threshold."""
        # Drop Ticker and Filing Date columns to focus on return data
        returns_data = self.return_df.drop(columns=['Ticker', 'Filing Date'])
        
        # Calculate consecutive differences and identify rows with jumps > specified threshold
        jumps = returns_data.diff(axis=1).abs().dropna(axis=1)
        mask = (jumps <= max_jump).all(axis=1)
        
        # Filter the return_df to remove those rows
        self.filtered_return_df = self.return_df[mask]
        print(f"Removed {len(mask) - mask.sum()} rows with jumps greater than {int(max_jump*100)}%.")

    def plot_stock_returns(self, filtered=False):
        """Plot stock returns with 5th and 95th percentiles, mean, and median."""
        # Select the appropriate dataset
        if filtered:
            returns_data = self.filtered_return_df.drop(columns=['Ticker', 'Filing Date'])
        else:
            returns_data = self.return_df.drop(columns=['Ticker', 'Filing Date'])
        
        # Calculate mean, median, 5th and 95th percentiles
        mean_returns = returns_data.mean()
        median_returns = returns_data.median()
        lower_bound = returns_data.quantile(0.05)
        upper_bound = returns_data.quantile(0.95)

        # Get mean and median at day 20
        mean_day_20 = mean_returns.iloc[-1]
        median_day_20 = median_returns.iloc[-1]

        # Set up the plot
        plt.figure(figsize=(12, 8))
        x_values = np.arange(1, len(mean_returns) + 1)

        # Plot the 5th to 95th percentile range as a shaded area
        plt.fill_between(x_values, lower_bound, upper_bound, color='gray', alpha=0.3, label='5th-95th Percentile')

        # Plot the mean and median returns
        plt.plot(x_values, mean_returns, color='blue', label='Mean Return')
        plt.plot(x_values, median_returns, color='orange', label='Median Return')

        # Add a red dotted line at 0% return
        plt.axhline(y=0, color='red', linestyle='--', label='0% Return')

        # Formatting the plot
        plt.xticks(ticks=x_values)
        plt.xlabel('Day')
        plt.ylabel('Return')

        # Adapt y-limits to the data
        plt.ylim(-0.3, 0.3)

        plt.title(f'Stock Returns Over 20 Days | Mean: {mean_day_20*100:.2f}%, Median: {median_day_20*100:.2f}% on Day 20')
        plt.legend(loc='best')

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Save and show the plot
        output_path = os.path.join(self.output_dir, 'stock_returns_analysis.png')
        if filtered:
            output_path = os.path.join(self.output_dir, 'stock_returns_analysis_filtered.png')
        plt.savefig(output_path, dpi=300)
        print(f"Stock return summary figure saved at {output_path}")

    def plot_all_stock_returns(self, filtered=False):
        """Plot all stock returns overlayed on the same plot."""
        # Select the appropriate dataset
        if filtered:
            returns_data = self.filtered_return_df.drop(columns=['Ticker', 'Filing Date'])
        else:
            returns_data = self.return_df.drop(columns=['Ticker', 'Filing Date'])

        # Set up the plot
        plt.figure(figsize=(12, 8))
        x_values = np.arange(1, len(returns_data.columns) + 1)

        # Plot all stock returns
        for _, row in returns_data.iterrows():
            plt.plot(x_values, row, color='gray', alpha=0.5)

        # Formatting the plot
        plt.xticks(ticks=x_values)
        plt.xlabel('Day')
        plt.ylabel('Return')
        plt.title('Overlay of All Stock Returns Over 20 Days')
        
        # Add a red dotted line at 0% return
        plt.axhline(y=0, color='red', linestyle='--', label='0% Return')

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Save and show the plot
        output_path = os.path.join(self.output_dir, 'all_stock_returns_overlay.png')
        if filtered:
            output_path = os.path.join(self.output_dir, 'all_stock_returns_overlay_filtered.png')
        plt.savefig(output_path, dpi=300)
        print(f"All stock returns figure saved at {output_path}")

    def save_summary_statistics(self):
        """Save summary statistics (min, 25%, median, mean, 75%, max) for each timestep to an Excel sheet."""
        returns_data = self.return_df.drop(columns=['Ticker', 'Filing Date'])
        filtered_returns_data = self.filtered_return_df.drop(columns=['Ticker', 'Filing Date'])

        summary_stats = returns_data.describe(percentiles=[0.25, 0.5, 0.75]).T
        summary_stats = summary_stats[['min', '25%', '50%', 'mean', '75%', 'max']]*100

        filtered_summary_stats = filtered_returns_data.describe(percentiles=[0.25, 0.5, 0.75]).T
        filtered_summary_stats = filtered_summary_stats[['min', '25%', '50%', 'mean', '75%', 'max']]*100

        output_path = os.path.join(self.output_dir, 'stock_returns_summary_stats.xlsx')
        with pd.ExcelWriter(output_path) as writer:
            summary_stats.to_excel(writer, sheet_name='Original', index_label='Day')
            filtered_summary_stats.to_excel(writer, sheet_name='Filtered', index_label='Day')
        print(f"Summary statistics saved to {output_path}")

    def run(self):
        """Run the stock return analysis and plot generation."""
        self.load_data()
        self.plot_all_stock_returns(filtered=False)
        self.plot_stock_returns(filtered=False)
        self.filter_jumps(max_jump=0.5) 
        self.plot_all_stock_returns(filtered=True)
        self.plot_stock_returns(filtered=True)
        self.save_summary_statistics()

if __name__ == "__main__":
    analysis = StockAnalysis()
    analysis.run()
