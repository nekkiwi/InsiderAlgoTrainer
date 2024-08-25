import pandas as pd

def process_targets(stock_data, limit_array, stop_array):
    """Calculate the targets for the given ticker based on the stock data."""
    targets = {}

    for limit in limit_array:
        for stop in stop_array:
            target_key = (limit, stop)
            targets[target_key] = {
                'Spike up': 0,
                'Spike down': 0,
                'Limit occurred first': 0,
                'Stop occurred first': 0,
                'Return at cashout': 0.0,
                'Days at cashout': 0,
            }

            for i in range(1, len(stock_data)):
                price_change = (stock_data.iloc[i] - stock_data.iloc[0]) / stock_data.iloc[0]

                # Spike detection
                if (stock_data.iloc[i] - stock_data.iloc[i-1]) / stock_data.iloc[i-1] > 0.1:
                    targets[target_key]['Spike up'] = 1
                if (stock_data.iloc[i-1] - stock_data.iloc[i]) / stock_data.iloc[i-1] > 0.1:
                    targets[target_key]['Spike down'] = 1

                # Check for limit/stop
                if price_change >= limit:
                    targets[target_key]['Limit occurred first'] = 1
                    targets[target_key]['Days at cashout'] = i
                    targets[target_key]['Return at cashout'] = price_change
                    break

                if price_change <= stop:
                    targets[target_key]['Stop occurred first'] = 1
                    targets[target_key]['Days at cashout'] = i
                    targets[target_key]['Return at cashout'] = price_change
                    break

            # If no limit or stop was reached
            if targets[target_key]['Days at cashout'] == 0:
                targets[target_key]['Days at cashout'] = len(stock_data) - 1
                targets[target_key]['Return at cashout'] = price_change

    return targets

def calculate_target_distribution(results):
    """Calculate and return the distribution of each target for each limit-stop combination."""
    distribution_data = []

    # Iterate over the first entry to get all the metric names
    first_key = next(iter(results))
    first_data = results[first_key]
    
    for limit_stop_key in first_data.keys():
        for metric in first_data[limit_stop_key].keys():
            metric_values = []
            for target_data in results.values():
                metric_value = target_data[limit_stop_key].get(metric, None)
                if metric_value is not None:
                    metric_values.append(metric_value)
            
            if metric_values:
                metric_series = pd.Series(metric_values)
                distribution_metrics = metric_series.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

                distribution_data.append({
                    'Limit-Stop': limit_stop_key,
                    'Target': metric,
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

