import pandas as pd
import numpy as np

def load_stock_data(stock_returns_file):
    """Load the stock returns data."""
    return pd.read_excel(stock_returns_file, sheet_name='Returns')

def align_dfs(tickers_df, stock_returns_df):
    """Align the tickers and stock returns DataFrames based on their index (Ticker, Filing Date)."""
    tickers_df = tickers_df.set_index(['Ticker', 'Filing Date'])
    stock_returns_df = stock_returns_df.set_index(['Ticker', 'Filing Date'])

    # Align the DataFrames to ensure matching indices
    aligned_return_df = stock_returns_df.loc[tickers_df.index.intersection(stock_returns_df.index)]
    
    return aligned_return_df

def process_simulation(args):
    """Process a single simulation for the backtest."""
    tickers_file, limit, stop, all_stock_stats = args
    tickers_df = pd.read_excel(tickers_file)
    
    aligned_return_df = align_dfs(tickers_df, load_stock_data(tickers_file))
    pred_filtered_day_20_returns = aligned_return_df['Day 20']
    gt_filtered_day_20_returns = aligned_return_df['Day 20']

    statistics = calculate_statistics(pred_filtered_day_20_returns, gt_filtered_day_20_returns, all_stock_stats)
    return statistics

def calculate_statistics(pred_filtered_day_20_returns, gt_filtered_day_20_returns, all_stock_stats):
    """Calculate the statistics for backtesting."""
    tp, fp, tn, fn = calculate_confusion_matrix(pred_filtered_day_20_returns, gt_filtered_day_20_returns)
    statistics = {
        "PPV": tp/(tp+fp),
        "NPV": tn/(tn+fn),
        "Accuracy": (tp+tn)/(tp+tn+fp+fn),
        "Pred Median Return on Day 20": pred_filtered_day_20_returns.median(),
        "GT Median Return on Day 20": gt_filtered_day_20_returns.median(),
        "All-Raw Median Return on Day 20": all_stock_stats['median'],
        "All-Raw Mean Return on Day 20": all_stock_stats['mean'],
    }
    return statistics

def calculate_confusion_matrix(pred_filtered, gt_filtered):
    """Calculate TP, FP, TN, FN for backtesting."""
    tp = (pred_filtered & gt_filtered).sum()
    fp = (pred_filtered & ~gt_filtered).sum()
    tn = (~pred_filtered & ~gt_filtered).sum()
    fn = (~pred_filtered & gt_filtered).sum()
    return tp, fp, tn, fn

def calculate_all_stock_stats(summary_stats):
    """Calculate mean and median returns at day 20 for all stock history."""
    return {
        "mean": summary_stats.loc['Day 20', 'mean'] / 100,
        "median": summary_stats.loc['Day 20', '50%'] / 100
    }

def extract_limit_stop_from_filename(filename):
    """Extract limit and stop values from the filename."""
    parts = filename.split('_')
    limit = float(parts[2].replace('l', ''))
    stop = float(parts[3].replace('.xlsx', '').replace('s', ''))
    return limit, stop
