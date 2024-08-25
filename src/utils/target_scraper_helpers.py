import yfinance as yf
import pandas as pd
import contextlib
import os
from datetime import timedelta

def download_daily_stock_data(ticker, filing_date, end_date):
    """Download daily stock data for a given ticker between start_date and end_date."""
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            try:
                stock_data = yf.download(ticker, start=filing_date, end=end_date, interval='1d', progress=False)
                spy_data = yf.download('SPY', start=filing_date, end=end_date, interval='1d', progress=False)
                if stock_data.empty or spy_data.empty:
                    return None, None
                # Ensure the datetime index is timezone-naive
                stock_data.index = stock_data.index.tz_localize(None)
                spy_data.index = spy_data.index.tz_localize(None)
                return stock_data, spy_data
            except Exception as e:
                print(f"Failed to download data for {ticker}: {e}")
                return None, None

def process_targets(ticker, stock_data, spy_data, limit_array, stop_array):
    """Calculate the targets for the given ticker based on the stock data and SPY data."""
    targets = {f'lim_{limit}_stop_{stop}': {} for limit in limit_array for stop in stop_array}

    for limit in limit_array:
        for stop in stop_array:
            target_name = f'lim_{limit}_stop_{stop}'
            targets[target_name] = {
                'limit_occurs_first': 0,
                'stop_occurs_first': 0,
                'alpha_at_cashout': 0.0,
                'days_at_cashout': 0,
                'return_at_cashout': 0.0,
                'spike_up_anytime': 0,
                'spike_down_anytime': 0,
            }

            for i in range(1, len(stock_data)):
                price_change = (stock_data['Close'].iloc[i] - stock_data['Close'].iloc[0]) / stock_data['Close'].iloc[0]
                spy_change = (spy_data['Close'].iloc[i] - spy_data['Close'].iloc[0]) / spy_data['Close'].iloc[0]
                current_close_price = stock_data['Close'].iloc[i]

                # Spike detection
                if (stock_data['Close'].iloc[i] - stock_data['Close'].iloc[i-1]) / stock_data['Close'].iloc[i-1] > 0.1:
                    targets[target_name]['spike_up_anytime'] = 1
                if (stock_data['Close'].iloc[i-1] - stock_data['Close'].iloc[i]) / stock_data['Close'].iloc[i-1] > 0.1:
                    targets[target_name]['spike_down_anytime'] = 1

                # Check for limit/stop
                if price_change >= limit:
                    targets[target_name]['limit_occurs_first'] = 1
                    targets[target_name]['alpha_at_cashout'] = (current_close_price - stock_data['Close'].iloc[0]) / stock_data['Close'].iloc[0] - spy_change
                    targets[target_name]['days_at_cashout'] = i
                    targets[target_name]['return_at_cashout'] = price_change
                    break

                if price_change <= stop:
                    targets[target_name]['stop_occurs_first'] = 1
                    targets[target_name]['alpha_at_cashout'] = (current_close_price - stock_data['Close'].iloc[0]) / stock_data['Close'].iloc[0] - spy_change
                    targets[target_name]['days_at_cashout'] = i
                    targets[target_name]['return_at_cashout'] = price_change
                    break

            # If no limit or stop was reached
            if targets[target_name]['days_at_cashout'] == 0:
                targets[target_name]['alpha_at_cashout'] = price_change - spy_change
                targets[target_name]['days_at_cashout'] = len(stock_data) - 1
                targets[target_name]['return_at_cashout'] = price_change

    return targets

def calculate_target_distribution(results):
    """Calculate and return the distribution of each target for each limit-stop combination."""
    distribution_data = []

    for target_name in results[next(iter(results))].keys():
        for ticker_filing_date, target_values in results.items():
            if target_values is not None:
                # Extract the limit and stop values from the target name
                parts = target_name.split('_')
                limit_value = parts[1]
                stop_value = parts[3]

                # Calculate the distribution metrics
                target_series = pd.Series(target_values[target_name])
                distribution_metrics = target_series.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

                distribution_data.append({
                    'Limit': limit_value,
                    'Stop': stop_value,
                    'Target': target_name,
                    'min': distribution_metrics['min'],
                    '1%': distribution_metrics['1%'],
                    '5%': distribution_metrics['5%'],
                    '10%': distribution_metrics['10%'],
                    '25%': distribution_metrics['25%'],
                    '50%': distribution_metrics['50%'],
                    '75%': distribution_metrics['75%'],
                    '90%': distribution_metrics['90%'],
                    '95%': distribution_metrics['95%'],
                    '99%': distribution_metrics['99%'],
                    'max': distribution_metrics['max'],
                    'mean': distribution_metrics['mean']
                })

    return pd.DataFrame(distribution_data)