import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')    # use the non-GUI backend
import matplotlib.pyplot as plt
import numpy as np

def load_stock_data(stock_returns_file, sheet_name='Returns'):
    """Load the stock data for the specified sheet name."""
    return pd.read_excel(stock_returns_file, sheet_name=sheet_name)

def filter_jumps(df, max_jump: float = 0.1):
    """
    Remove any rows where any numeric return exceeds max_jump in absolute value.
    Returns a new DataFrame (does not modify df in place).
    """
    # 1) find the numeric return columns (exclude Ticker, Filing Date, etc)
    #    here we assume all non-index cols that are numeric are returns/alphas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 2) build the Boolean mask: True if *all* abs(value) <= max_jump
    mask = (df[numeric_cols].abs() <= max_jump).all(axis=1)

    # 3) use it to filter
    filtered = df.loc[mask].reset_index(drop=True)

    dropped = len(df) - len(filtered)
    print(f"- Dropped {dropped} rows with any absolute return > {max_jump}")

    return filtered

def save_summary_statistics(df, output_dir, output_filename):
    """
    Save summary statistics (min, 25%, 50%, mean, 75%, max) for each
    timestep to an Excel sheet, computing each stat via numpy per column.
    """
    # 1) Drop the non‐numeric identifier cols
    returns_data = df.select_dtypes(include=[np.number])
    
    # 2) Build a dict of per‐column stats
    stats = {}
    for col in returns_data.columns:
        arr = returns_data[col].dropna().to_numpy(dtype=float)
        if arr.size == 0:
            stats[col] = {'min': np.nan, '25%': np.nan, '50%': np.nan,
                          'mean': np.nan, '75%': np.nan, 'max': np.nan}
            continue
        
        stats[col] = {
            'min':  np.nanmin(arr),
            '25%':  np.nanpercentile(arr, 25),
            '50%':  np.nanpercentile(arr, 50),
            'mean': np.nanmean(arr),
            '75%':  np.nanpercentile(arr, 75),
            'max':  np.nanmax(arr),
        }
    
    # 3) Create DataFrame and scale to percentages
    summary_stats = pd.DataFrame.from_dict(stats, orient='index') * 100

    # 4) Write to Excel
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='w') as writer:
        summary_stats.to_excel(writer, sheet_name='Unfiltered', index_label='Day')

    print(f"- Statistics summary saved at {output_path}")

def plot_combined(df, output_dir, suffix):
    """Plot overlay, percentiles, mean & median of your day-by-day returns."""
    # 1) Drop the metadata columns
    returns_data = df.drop(columns=['Ticker', 'Filing Date'], errors='ignore')

    # 2) Coerce everything else to float (so mixed-type columns become NaN)
    returns_data = returns_data.apply(pd.to_numeric, errors='coerce')

    # 3) Now compute stats
    mean_returns   = returns_data.mean(axis=0, skipna=True)
    median_returns = returns_data.median(axis=0, skipna=True)
    lower_bound    = returns_data.quantile(0.05, axis=0)
    upper_bound    = returns_data.quantile(0.95, axis=0)

    # 4) Set up the x-axis as “Day 1”, “Day 2”, ...
    days = returns_data.columns.tolist()
    # if your columns are literally ["Day 1 Stock", "Day 2 Stock", …], you can extract the day number:
    try:
        x_values = [int(col.split()[1]) for col in days]
    except:
        # fallback to 1..n
        x_values = np.arange(1, len(days)+1)

    # 5) Plot
    fig, axs = plt.subplots(1, 3, figsize=(36, 8))

    # Overlay all trajectories (each ticker is a line)
    axs[0].plot(x_values, returns_data.T, color='gray', alpha=0.5)
    axs[0].axhline(0, color='red', linestyle='--')
    axs[0].set_title('Overlay of All Stock Returns')
    axs[0].set_xlabel('Day')

    # Percentiles band + mean & median
    axs[1].fill_between(x_values, lower_bound, upper_bound, alpha=0.3)
    axs[1].plot(x_values, mean_returns, label='Mean', linewidth=2)
    axs[1].plot(x_values, median_returns, label='Median', linewidth=2)
    axs[1].axhline(0, color='red', linestyle='--')
    axs[1].set_title('5th-95th Percentile Band with Mean & Median')
    axs[1].set_xlabel('Day')
    axs[1].set_ylabel('Return')
    axs[1].legend()

    # Mean & median only
    axs[2].plot(x_values, mean_returns, label='Mean', linewidth=2)
    axs[2].plot(x_values, median_returns, label='Median', linewidth=2)
    axs[2].axhline(0, color='red', linestyle='--')
    axs[2].set_title('Mean & Median Returns')
    axs[2].set_xlabel('Day')
    axs[2].legend()

    # 6) Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'stock_returns_combined{suffix}.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"- Stock plots saved at {out_path}")