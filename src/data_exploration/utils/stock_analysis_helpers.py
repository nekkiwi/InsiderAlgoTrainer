import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def load_stock_data(stock_returns_file, sheet_name='Returns'):
    """Load the stock data for the specified sheet name."""
    return pd.read_excel(stock_returns_file, sheet_name=sheet_name)

def filter_jumps(df, max_jump=1):
    """Remove rows with jumps (consecutive differences) higher than a specified threshold."""
    returns_data = df.drop(columns=['Ticker', 'Filing Date'])
    jumps = returns_data.diff(axis=1).abs().dropna(axis=1)
    mask = (jumps <= max_jump).all(axis=1)
    return df[mask]

def save_summary_statistics(df, output_dir, output_filename):
    """Save summary statistics (min, 25%, median, mean, 75%, max) for each timestep to an Excel sheet."""
    returns_data = df.drop(columns=['Ticker', 'Filing Date'])
    summary_stats = returns_data.describe(percentiles=[0.25, 0.5, 0.75]).T
    summary_stats = summary_stats[['min', '25%', '50%', 'mean', '75%', 'max']] * 100

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
        summary_stats.to_excel(writer, sheet_name='Unfiltered', index_label='Day')
    print("- Statistics summary saved at", output_path)

def plot_combined(df, output_dir, suffix):
    """Plot combined figures with 5th and 95th percentiles, mean, median, and all stock returns."""
    returns_data = df.drop(columns=['Ticker', 'Filing Date'])
    mean_returns = returns_data.mean()
    median_returns = returns_data.median()
    lower_bound = returns_data.quantile(0.05)
    upper_bound = returns_data.quantile(0.95)

    _, axs = plt.subplots(1, 3, figsize=(36, 8))
    x_values = np.arange(1, len(mean_returns) + 1)

    # Plot all stock returns overlayed
    axs[0].plot(x_values, returns_data.T, color='gray', alpha=0.5)
    axs[0].axhline(y=0, color='red', linestyle='--')
    axs[0].set_title('Overlay of All Stock Returns')
    axs[0].set_xlabel('Day')

    # Plot percentiles, mean, median
    axs[1].fill_between(x_values, lower_bound, upper_bound, color='gray', alpha=0.3)
    axs[1].plot(x_values, mean_returns, color='blue', label='Mean Return')
    axs[1].plot(x_values, median_returns, color='orange', label='Median Return')
    axs[1].axhline(y=0, color='red', linestyle='--')
    axs[1].set_title('Percentiles, Mean, & Median')
    axs[1].set_xlabel('Day')
    axs[1].set_ylabel('Return')
    axs[1].legend(loc='best')
    axs[1].set_ylim(-0.3, 0.3)

    # Plot mean & median only
    axs[2].plot(x_values, mean_returns, color='blue', label='Mean Return')
    axs[2].plot(x_values, median_returns, color='orange', label='Median Return')
    axs[2].axhline(y=0, color='red', linestyle='--')
    axs[2].set_title('Mean & Median Returns')
    axs[2].set_xlabel('Day')
    axs[2].set_ylim(-0.1, 0.1)

    output_path = os.path.join(output_dir, f'stock_returns_combined{suffix}.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print("- Stock plots saved at", output_path)
