import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')    # use the non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import re

def load_stock_data(stock_returns_file, sheet_name='Returns'):
    """Load the stock data for the specified sheet name."""
    # Add a check to ensure the file exists
    if not os.path.exists(stock_returns_file):
        raise FileNotFoundError(f"Data file not found at: {stock_returns_file}")
    return pd.read_excel(stock_returns_file, sheet_name=sheet_name)

# --- REFACTORED AND FIXED ---
def filter_and_align_data(primary_df, other_dfs, max_jump: float = 0.1):
    """
    Filters rows based on a jump condition in the primary DataFrame and applies
    the same filter to other DataFrames to keep them aligned.
    """
    # 1. Select only the numeric columns from the primary DataFrame to check for jumps.
    numeric_cols = primary_df.select_dtypes(include=[np.number]).columns
    
    # 2. Create a boolean mask based on the jump condition.
    #    The condition is True for rows where ALL numeric values are within the limit.
    mask = (primary_df[numeric_cols].abs() <= max_jump).all(axis=1)

    # 3. Apply this same mask to the primary DataFrame and all other DataFrames.
    filtered_primary = primary_df.loc[mask].reset_index(drop=True)
    filtered_others = [df.loc[mask].reset_index(drop=True) for df in other_dfs]
    
    dropped_count = len(primary_df) - len(filtered_primary)
    print(f"- Dropped {dropped_count} rows from all dataframes where any absolute return > {max_jump}")

    return [filtered_primary] + filtered_others

def save_summary_statistics(df, output_dir, output_filename):
    """
    Saves summary statistics for each column to an Excel sheet.
    """
    # 1. Select only numeric columns for statistics
    returns_data = df.select_dtypes(include=[np.number])
    
    # 2. Calculate summary statistics
    # The .describe() method is a more direct way to get these stats
    summary_stats = returns_data.describe(percentiles=[.25, .5, .75]).transpose()
    
    # Select and rename for desired output format
    summary_stats = summary_stats[['min', '25%', '50%', 'mean', '75%', 'max']]
    
    # 3. Scale to percentages for display
    summary_stats *= 100

    # 4. Save to Excel
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    summary_stats.to_excel(output_path, sheet_name='Summary Statistics', index_label='Day')
    print(f"- Statistics summary saved at {output_path}")

def plot_combined(df, output_dir, suffix):
    """Plots and saves graphs for the given returns or alpha data."""
    # 1. Drop non-numeric metadata columns
    returns_data = df.drop(columns=['Ticker', 'Filing Date'], errors='ignore')

    # 2. Compute statistics for plotting
    mean_values = returns_data.mean(axis=0)
    median_values = returns_data.median(axis=0)
    lower_bound = returns_data.quantile(0.05, axis=0)
    upper_bound = returns_data.quantile(0.95, axis=0)

    # --- FIX: Make x-axis parsing more robust ---
    # Use a regular expression to find numbers in column names (e.g., 'Day 1', 'Return_1D')
    x_values = [int(re.search(r'\d+', col).group()) for col in returns_data.columns if re.search(r'\d+', col)]
    if not x_values: # Fallback if no numbers are found
        x_values = np.arange(1, len(returns_data.columns) + 1)
        
    # 3. Create the plots
    fig, axs = plt.subplots(1, 3, figsize=(36, 8), tight_layout=True)
    fig.suptitle(f'Stock Performance Analysis ({suffix.replace("_", "").title()})', fontsize=16)

    # Plot 1: Overlay of all individual trajectories
    axs[0].plot(x_values, returns_data.T, color='gray', alpha=0.3, linewidth=0.5)
    axs[0].axhline(0, color='black', linestyle='--')
    axs[0].set_title('All Individual Trajectories')
    axs[0].set_xlabel('Day')
    axs[0].set_ylabel('Value')

    # Plot 2: Percentile band with mean and median
    axs[1].fill_between(x_values, lower_bound, upper_bound, color='skyblue', alpha=0.4, label='5th-95th Percentile')
    axs[1].plot(x_values, mean_values, 'b-', label='Mean', linewidth=2)
    axs[1].plot(x_values, median_values, 'g--', label='Median', linewidth=2)
    axs[1].axhline(0, color='black', linestyle='--')
    axs[1].set_title('Mean, Median, and Percentile Band')
    axs[1].set_xlabel('Day')
    axs[1].legend()

    # Plot 3: Mean and median only for clarity
    axs[2].plot(x_values, mean_values, 'b-', label='Mean', linewidth=2)
    axs[2].plot(x_values, median_values, 'g--', label='Median', linewidth=2)
    axs[2].axhline(0, color='black', linestyle='--')
    axs[2].set_title('Aggregate Mean and Median')
    axs[2].set_xlabel('Day')
    axs[2].legend()

    # 4. Save the figure
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'stock_performance_chart{suffix}.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"- Performance chart saved at {out_path}")
