import pandas as pd

def process_ticker_targets(ticker_info, return_df, alpha_df, limit_array, stop_array):
    """Process return and alpha data for a specific ticker and filing date."""
    ticker, filing_date = ticker_info

    # Get stock data for the given ticker and filing date
    stock_return_data = return_df[
        (return_df['Ticker'] == ticker) & 
        (return_df['Filing Date'] == filing_date)
    ].iloc[:, 2:].squeeze()

    stock_alpha_data = alpha_df[
        (alpha_df['Ticker'] == ticker) & 
        (alpha_df['Filing Date'] == filing_date)
    ].iloc[:, 2:].squeeze()

    if not stock_return_data.empty and not stock_alpha_data.empty:
        return (ticker, filing_date), process_targets(stock_return_data, stock_alpha_data, limit_array, stop_array)
    
    return (ticker, filing_date), None

def process_targets(stock_return_data, stock_alpha_data, limit_array, stop_array):
    """Calculate targets for both return and alpha data using specified limit and stop arrays."""
    targets = {}
    for limit in limit_array:
        for stop in stop_array:
            target_key = (limit, stop)
            targets[target_key] = {
                'return_limit_sell'         : 0,
                'return_stop_sell'          : 0,
                'alpha_limit_sell'          : 0,
                'alpha_stop_sell'           : 0,
                'pos_return_1w_limstop'     : 0,
                'pos_return_1m_limstop'     : 0,
                'final_return_1w_limstop'   : 0.0,
                'final_return_1m_limstop'   : 0.0,
                'pos_alpha_1w_limstop'      : 0,
                'pos_alpha_1m_limstop'      : 0,
                'final_alpha_1w_limstop'    : 0.0,
                'final_alpha_1m_limstop'    : 0.0,
                'pos_return_1w_raw'         : 0,
                'pos_alpha_1w_raw'          : 0,
                'final_return_1w_raw'       : 0.0,
                'final_alpha_1w_raw'        : 0.0,
                'pos_return_1m_raw'         : 0,
                'pos_alpha_1m_raw'          : 0,
                'final_return_1m_raw'       : 0.0,
                'final_alpha_1m_raw'        : 0.0,
                'final_return_2w_raw'       : 0.0,
                'final_return_3w_raw'       : 0.0,
                'final_alpha_2w_raw'        : 0.0,
                'final_alpha_3w_raw'        : 0.0,
                'final_return_6w_raw'       : 0.0, 
                'final_return_2m_raw'       : 0.0, 
                'final_alpha_6w_raw'        : 0.0, 
                'final_alpha_2m_raw'        : 0.0, 
            }

            # Process return limit/stop for 1 week (index 5) and 1 month (index 20)
            return_limit_value_1w = None
            return_limit_value_2w = None
            return_limit_value_3w = None
            return_limit_value_1m = None

            for i in range(1, len(stock_return_data)):
                return_price_change = stock_return_data.iloc[i]
                # Check for return limit/stop
                if return_price_change >= limit:
                    targets[target_key]['return_limit_sell'] = 1
                    return_limit_value_1w = return_price_change if i <= 5 else stock_return_data.iloc[5]
                    return_limit_value_2w = return_price_change if i <= 10 else stock_return_data.iloc[10]
                    return_limit_value_3w = return_price_change if i <= 15 else stock_return_data.iloc[15]
                    return_limit_value_1m = return_price_change if i <= 20 else stock_return_data.iloc[20]
                    break
                if return_price_change <= stop:
                    targets[target_key]['return_stop_sell'] = 1
                    return_limit_value_1w = return_price_change if i <= 5 else stock_return_data.iloc[5]
                    return_limit_value_2w = return_price_change if i <= 10 else stock_return_data.iloc[10]
                    return_limit_value_3w = return_price_change if i <= 15 else stock_return_data.iloc[15]
                    return_limit_value_1m = return_price_change if i <= 20 else stock_return_data.iloc[20]
                    break

            # If no limit/stop occurred, take the values at index 5 (1 week) and index 20 (1 month)
            if return_limit_value_1w is None:
                return_limit_value_1w = stock_return_data.iloc[5] if len(stock_return_data) > 5 else stock_return_data.iloc[-1]
            if return_limit_value_2w is None:
                return_limit_value_2w = stock_return_data.iloc[10] if len(stock_return_data) > 10 else stock_return_data.iloc[-1]
            if return_limit_value_3w is None:
                return_limit_value_3w = stock_return_data.iloc[15] if len(stock_return_data) > 15 else stock_return_data.iloc[-1]
            if return_limit_value_1m is None:
                return_limit_value_1m = stock_return_data.iloc[20] if len(stock_return_data) > 20 else stock_return_data.iloc[-1]

            # Store final return values
            targets[target_key]['final_return_1w_limstop'] = return_limit_value_1w
            targets[target_key]['final_return_1m_limstop'] = return_limit_value_1m
            targets[target_key]['pos_return_1w_limstop'] = int(return_limit_value_1w > 0)
            targets[target_key]['pos_return_1m_limstop'] = int(return_limit_value_1m > 0)

            # Process alpha limit/stop for 1 week (index 5) and 1 month (index 20)
            alpha_limit_value_1w = None
            alpha_limit_value_1m = None

            for i in range(1, len(stock_alpha_data)):
                alpha_price_change = stock_alpha_data.iloc[i]

                # If limit or stop is reached, store the alpha value and break
                if alpha_price_change >= limit:
                    targets[target_key]['alpha_limit_sell'] = 1
                    alpha_limit_value_1w = alpha_price_change if i <= 5 else stock_alpha_data.iloc[5]
                    alpha_limit_value_1m = alpha_price_change if i <= 20 else stock_alpha_data.iloc[20]
                    break
                if alpha_price_change <= stop:
                    targets[target_key]['alpha_stop_sell'] = 1
                    alpha_limit_value_1w = alpha_price_change if i <= 5 else stock_alpha_data.iloc[5]
                    alpha_limit_value_1m = alpha_price_change if i <= 20 else stock_alpha_data.iloc[20]
                    break

            # If no limit/stop occurred, take the values at index 5 (1 week) and index 20 (1 month)
            if alpha_limit_value_1w is None:
                alpha_limit_value_1w = stock_alpha_data.iloc[5] if len(stock_alpha_data) > 5 else stock_alpha_data.iloc[-1]
            if alpha_limit_value_1m is None:
                alpha_limit_value_1m = stock_alpha_data.iloc[20] if len(stock_alpha_data) > 20 else stock_alpha_data.iloc[-1]

            # Store final alpha values
            targets[target_key]['final_alpha_1w_limstop'] = alpha_limit_value_1w
            targets[target_key]['final_alpha_1m_limstop'] = alpha_limit_value_1m
            targets[target_key]['pos_alpha_1w_limstop'] = int(return_limit_value_1w > 0)
            targets[target_key]['pos_alpha_1m_limstop'] = int(return_limit_value_1m > 0)

            # Calculate the final targets (after going through all the days)
            targets[target_key]['pos_return_1w_raw'] = int(stock_return_data.iloc[5] > 0)
            targets[target_key]['pos_alpha_1w_raw'] = int(stock_alpha_data.iloc[5] > 0)
            targets[target_key]['final_return_1w_raw'] = stock_return_data.iloc[5]
            targets[target_key]['final_alpha_1w_raw'] = stock_alpha_data.iloc[5]
            
            targets[target_key]['final_return_2w_raw'] = stock_return_data.iloc[10]
            targets[target_key]['final_alpha_2w_raw'] = stock_alpha_data.iloc[10]
            
            targets[target_key]['final_return_3w_raw'] = stock_return_data.iloc[15]
            targets[target_key]['final_alpha_3w_raw'] = stock_alpha_data.iloc[15]
            
            targets[target_key]['pos_return_1m_raw'] = int(stock_return_data.iloc[20] > 0)
            targets[target_key]['pos_alpha_1m_raw'] = int(stock_alpha_data.iloc[20] > 0)
            targets[target_key]['final_return_1m_raw'] = stock_return_data.iloc[20]
            targets[target_key]['final_alpha_1m_raw'] = stock_alpha_data.iloc[20]
            
            # Raw final values for 6-week and 2-month horizons
            targets[target_key]['final_return_6w_raw'] = stock_return_data.iloc[30]
            targets[target_key]['final_return_2m_raw'] = stock_return_data.iloc[39]

            targets[target_key]['final_alpha_6w_raw']  = stock_alpha_data.iloc[30]
            targets[target_key]['final_alpha_2m_raw']  = stock_alpha_data.iloc[39]

    return targets


def calculate_target_distribution(results, dist_out_file):
    """
    Calculate and return the distribution of each target for each
    limit-stop combination across all tickers, and save to Excel.
    """
    distribution_data = []

    # 1) Gather the full set of limit/stop combos
    all_limit_stop = set()
    for ticker_data in results.values():
        all_limit_stop.update(ticker_data.keys())

    # 2) For each combo, gather every metric name that any ticker has for it
    for limit, stop in sorted(all_limit_stop):
        # Collect all metrics seen under this (limit, stop)
        all_metrics = set()
        for ticker_data in results.values():
            if (limit, stop) in ticker_data:
                all_metrics.update(ticker_data[(limit, stop)].keys())

        # 3) Now for each metric, gather values across all tickers
        for metric in sorted(all_metrics):
            vals = []
            for ticker_data in results.values():
                if (limit, stop) in ticker_data:
                    v = ticker_data[(limit, stop)].get(metric)
                    if v is not None:
                        vals.append(v)

            if not vals:
                continue

            ser = pd.Series(vals)
            desc = ser.describe(
                percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            )

            distribution_data.append({
                'Limit': limit,
                'Stop': stop,
                'Target': metric,           # or metric.replace(" ", "-").lower()
                'min':   desc['min'],
                '1%':    desc['1%'],
                '5%':    desc['5%'],
                '10%':   desc['10%'],
                '25%':   desc['25%'],
                '50%':   desc['50%'],
                '75%':   desc['75%'],
                '90%':   desc['90%'],
                '95%':   desc['95%'],
                '99%':   desc['99%'],
                'max':   desc['max'],
                'mean':  desc['mean'],
            })

    # 4) Build DataFrame and save
    distribution_df = pd.DataFrame(distribution_data)
    distribution_df.to_excel(dist_out_file, index=False)
    print(f"- Target distribution successfully saved to {dist_out_file}.")

def save_targets_to_excel(results, limit_array, stop_array, output_file):
    """Save targets to an Excel file with each target in its own sheet."""
    try:
        # Extract the first set of results to get the target keys
        first_ticker = next(iter(results))
        target_keys = list(results[first_ticker][(limit_array[0], stop_array[0])].keys())

        #TODO FIX THE SINGLE COLUMN TARGETS IN THE TARGETS DISTRIBUTION
        # Identify the targets that are independent of limit/stop
        single_column_targets = ['pos_return_1w_raw', 'pos_alpha_1w_raw', 'final_return_1w_raw', 'final_alpha_1w_raw',
                                 'final_return_2w_raw','final_alpha_2w_raw','final_return_3w_raw','final_alpha_3w_raw',
                                 'pos_return_1m_raw', 'pos_alpha_1m_raw', 'final_return_1m_raw', 'final_alpha_1m_raw']

        # Prepare a DataFrame to store these independent targets
        static_target_df = pd.DataFrame({
            'Ticker': [ticker_filing_date[0] for ticker_filing_date in results.keys()],
            'Filing Date': [ticker_filing_date[1] for ticker_filing_date in results.keys()],
        })

        # Add columns for the independent targets
        for target in single_column_targets:
            static_target_df[target] = [
                results[ticker_filing_date][(limit_array[0], stop_array[0])].get(target, None)
                for ticker_filing_date in results.keys()
            ]

        # Save static targets to individual sheets
        with pd.ExcelWriter(output_file) as writer:
            for target in single_column_targets:
                # Create a DataFrame for the target
                target_df = static_target_df[['Ticker', 'Filing Date', target]].copy()
                # Save each target as its own sheet
                target_df.to_excel(writer, sheet_name=target, index=False)

            # Loop over each limit/stop dependent target and save as a separate sheet
            dependent_target_keys = [key for key in target_keys if key not in single_column_targets]
            for target_key in dependent_target_keys:
                # Prepare a DataFrame for this target
                target_df = pd.DataFrame({
                    'Ticker': [ticker_filing_date[0] for ticker_filing_date in results.keys()],
                    'Filing Date': [ticker_filing_date[1] for ticker_filing_date in results.keys()]
                })

                # Add limit/stop columns for this target
                for limit in limit_array:
                    for stop in stop_array:
                        column_name = f'Limit {limit}, Stop {stop}'
                        target_df[column_name] = [
                            results[ticker_filing_date][(limit, stop)].get(target_key, None)
                            for ticker_filing_date in results.keys()
                        ]

                # Save each target as its own sheet
                target_df.to_excel(writer, sheet_name=target_key, index=False)

        print(f"- Target data successfully saved to {output_file} with individual sheets for each target.")
        return target_df

    except Exception as e:
        print(f"- Failed to save target data to Excel: {e}")
        return None

