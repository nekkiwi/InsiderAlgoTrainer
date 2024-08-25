import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def download_daily_stock_data(ticker, start_date, end_date):
    """Download daily stock data for a given ticker between start_date and end_date."""
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1d', progress=False)
        if stock_data.empty:
            return None
        return stock_data
    except Exception as e:
        print(f"Failed to download data for {ticker}: {e}")
        return None

def calculate_alpha(stock_returns, spy_returns):
    """Calculate the alpha (difference in returns) between the stock and SPY."""
    return stock_returns - spy_returns

def plot_stock_histories(stock_data_dict, output_dir):
    """Plot all stock histories and save the plot to the output directory."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    for ticker, data in stock_data_dict.items():
        plt.plot(data.index, data['Close'], label=ticker)

    plt.title('Stock Histories')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend(loc='best')

    try:
        plt.tight_layout()
    except UserWarning:
        print("Warning: tight_layout could not be applied.")

    plt.savefig(os.path.join(output_dir, 'stock_histories.png'))
    plt.close()

def plot_median_mean_alpha(alpha_data, output_dir):
    """Plot the median and mean alpha over time."""
    alpha_df = pd.DataFrame(alpha_data)
    plt.figure(figsize=(14, 8))
    plt.plot(alpha_df.median(axis=1), label='Median Alpha')
    plt.plot(alpha_df.mean(axis=1), label='Mean Alpha')
    plt.title('Median and Mean Alpha Over Time')
    plt.xlabel('Date')
    plt.ylabel('Alpha')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'median_mean_alpha.png'))
    plt.close()

def plot_median_mean_return(return_data, output_dir):
    """Plot the median and mean return over time."""
    return_df = pd.DataFrame(return_data)
    plt.figure(figsize=(14, 8))
    plt.plot(return_df.median(axis=1), label='Median Return')
    plt.plot(return_df.mean(axis=1), label='Mean Return')
    plt.title('Median and Mean Return Over Time')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'median_mean_return.png'))
    plt.close()