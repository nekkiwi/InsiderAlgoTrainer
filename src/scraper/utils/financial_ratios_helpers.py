import yfinance as yf
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# This is a provided helper function, unchanged.
# def get_days_since_ipo(ticker, filing_date):
#     """
#     Calculate the number of days since IPO based on historical stock data.
#     This function makes a single API call per ticker.
#     """
#     with open(os.devnull, 'w') as fnull:
#         with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
#             stock = yf.Ticker(ticker)
#             # This .history call fetches the entire price history to find the start date.
#             historical_data = stock.history(period='max')
    
#             if historical_data.empty:
#                 return None
            
#             ipo_date = historical_data.index.min()

#             # Ensure timezone-naive for comparison
#             ipo_date = ipo_date.tz_localize(None)
#             if filing_date.tzinfo is not None:
#                 filing_date = filing_date.tz_localize(None)

#             return (filing_date - ipo_date).days

# This is a provided helper function, unchanged.
def calculate_financial_ratios(data):
    """Calculate and normalize financial ratios from the provided data payload."""
    ratios = {}
    if not data:
        return None

    balance_sheet = data.get('balance_sheet')
    cash_flow = data.get('cash_flow')
    income_statement = data.get('income_statement')

    if balance_sheet is None or cash_flow is None or income_statement is None:
        return None

    # Profitability Ratios
    net_income = income_statement.get('Net Income')
    revenue = income_statement.get('Total Revenue')
    total_assets = balance_sheet.get('Total Assets')
    stockholders_equity = balance_sheet.get('Stockholders Equity')

    ratios['Net_Profit_Margin'] = net_income / revenue if net_income and revenue else None
    ratios['ROA'] = net_income / total_assets if net_income and total_assets else None
    ratios['ROE'] = net_income / stockholders_equity if net_income and stockholders_equity else None

    # Leverage Ratios
    total_liabilities = balance_sheet.get('Total Liabilities Net Minority Interest')
    ratios['Debt_to_Equity'] = total_liabilities / stockholders_equity if total_liabilities and stockholders_equity else None
    
    # Cash Flow Ratios
    operating_cash_flow = cash_flow.get('Operating Cash Flow')
    investing_cash_flow = cash_flow.get('Investing Cash Flow')
    financing_cash_flow = cash_flow.get('Financing Cash Flow')
    capital_expenditure = cash_flow.get('Capital Expenditure')
    ratios['Operating_Cash_Flow'] = operating_cash_flow
    ratios['Investing_Cash_Flow'] = investing_cash_flow
    ratios['Financing_Cash_Flow'] = financing_cash_flow
    ratios['Free_Cash_Flow'] = (operating_cash_flow - capital_expenditure) if operating_cash_flow and capital_expenditure else None

    # Valuation & Other Ratios
    market_cap = data.get('market_cap')
    ratios['Market_Cap'] = market_cap
    ratios['Price_to_Earnings_Ratio'] = market_cap / net_income if market_cap and net_income else None
    ratios['Price_to_Book_Ratio'] = market_cap / stockholders_equity if market_cap and stockholders_equity else None
    ratios['Price_to_Sales_Ratio'] = market_cap / revenue if market_cap and revenue else None
    ratios['Operating_Cash_Flow_to_Market_Cap'] = operating_cash_flow / market_cap if operating_cash_flow and market_cap else None
    ratios['Net_Income_to_Market_Cap'] = net_income / market_cap if net_income and market_cap else None
    
    # Additional Information
    ratios['Sector'] = data.get('sector')
    ratios['EPS'] = data.get('eps')
    ratios['Beta'] = data.get('beta')
    
    current_price = data.get('current_price')
    high_52_week = data.get('high_52_week')
    low_52_week = data.get('low_52_week')
    if current_price and high_52_week:
        ratios['52_Week_High_Normalized'] = high_52_week / current_price
    if current_price and low_52_week:
        ratios['52_Week_Low_Normalized'] = low_52_week / current_price

    return ratios
import numpy as np
def calculate_point_in_time_beta(stock_prices: pd.Series, market_prices: pd.Series, lookback_days: int = 252) -> float:
    """
    Calculates the beta of a stock relative to the market for a given lookback period.

    Args:
        stock_prices (pd.Series): The historical closing prices of the stock.
        market_prices (pd.Series): The historical closing prices of the market index (e.g., SPY).
        lookback_days (int): The number of trading days to use for the calculation (default is 252, approx. 1 year).

    Returns:
        float: The calculated beta value, or np.nan if calculation is not possible.
    """
    if len(stock_prices) < lookback_days or len(market_prices) < lookback_days:
        return np.nan

    # Get the data for the lookback window
    stock_window = stock_prices.tail(lookback_days)
    market_window = market_prices.tail(lookback_days)

    # Calculate daily returns
    stock_returns = stock_window.pct_change(fill_method=None).dropna()
    market_returns = market_window.pct_change(fill_method=None).dropna()

    if len(stock_returns) < 2 or len(market_returns) < 2:
        return np.nan

    # Align returns by index to handle any missing dates
    aligned_returns = pd.DataFrame({'stock': stock_returns, 'market': market_returns}).dropna()
    if len(aligned_returns) < 2:
        return np.nan

    # Calculate covariance and market variance
    covariance_matrix = np.cov(aligned_returns['stock'], aligned_returns['market'])
    covariance = covariance_matrix[0, 1]
    market_variance = np.var(aligned_returns['market'])

    if market_variance == 0:
        return np.nan

    beta = covariance / market_variance
    return beta

import random

def process_single_ticker(row, tk_objects, hist_data, market_data_spy):
    """
    Worker function with a retry mechanism to handle API rate limiting.
    """
    ticker = row['Ticker']
    filing_date = row['Filing Date']

    # --- RETRY LOGIC PARAMETERS ---
    max_retries = 3
    base_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            # --- START of original logic ---
            t_obj = tk_objects.tickers.get(ticker)
            # Add a small, random delay to space out requests
            time.sleep(random.uniform(0.1, 0.5))

            t_info = t_obj.info if hasattr(t_obj, 'info') else {}
            if t_info:

                # Helper to get the latest financial statement before the filing date.
                def pick_latest(stmt_df):
                    if stmt_df is None or stmt_df.empty: return None
                    stmt_df.columns = pd.to_datetime(stmt_df.columns).tz_localize(None)
                    valid_cols = stmt_df.columns[stmt_df.columns <= filing_date]
                    return stmt_df.loc[:, valid_cols.max()] if not valid_cols.empty else None

                # Get the point-in-time financial statements (your existing logic is correct)
                balance_sheet = pick_latest(t_obj.balance_sheet)
                cash_flow = pick_latest(t_obj.cashflow)
                income_statement = pick_latest(t_obj.financials)
                if balance_sheet is None or income_statement is None: return None

                stock_market_data = hist_data.get(ticker) # Renamed for clarity
                if stock_market_data is None: return None
                stock_market_data.index = stock_market_data.index.tz_localize(None)
                market_data_filtered = stock_market_data.loc[stock_market_data.index <= filing_date]
                if market_data_filtered.empty: return None

                latest_market_data = market_data_filtered.iloc[-1]
                current_price = latest_market_data.get('Close')
                if pd.isna(current_price): return None

                one_year_prior = filing_date - pd.Timedelta(days=365)
                historical_window = market_data_filtered.loc[market_data_filtered.index >= one_year_prior]
                high_52_week = historical_window['High'].max()
                low_52_week = historical_window['Low'].min()
                
                shares_outstanding = balance_sheet.get('Share Issued')
                if pd.isna(shares_outstanding): return None
                market_cap = current_price * shares_outstanding
                eps = income_statement.get('Diluted EPS')

                # --- THE BETA CALCULATION IS NOW CORRECT ---
                
                # 1. Get the historical prices for the stock up to the filing date
                stock_prices_historical = stock_market_data['Close'].loc[:filing_date]
                
                # 2. Get historical prices for the market index (SPY) up to the filing date
                market_prices_historical = market_data_spy['Close'].loc[:filing_date]
                
                # 3. Calculate beta using both series
                beta = calculate_point_in_time_beta(stock_prices_historical.squeeze(), market_prices_historical.squeeze())
                
                # The data payload is now fully point-in-time correct
                data_payload = {
                    'balance_sheet': balance_sheet,
                    'cash_flow': cash_flow,
                    'income_statement': income_statement,
                    'market_cap': market_cap,
                    'high_52_week': high_52_week,
                    'low_52_week': low_52_week,
                    'current_price': current_price,
                    'eps': eps,
                    'beta': beta, # This is no longer a placeholder
                    'sector': t_info.get('sector')
                }
                
                ratios = calculate_financial_ratios(data_payload)

                if ratios:
                    ratios['Ticker'] = ticker
                    ratios['Filing Date'] = filing_date
                    
                    # Your existing IPO date logic is robust and can remain
                    ipo_timestamp_epoch = t_info.get('firstTradeDateEpochUtc')
                    ipo_date = pd.to_datetime(ipo_timestamp_epoch, unit='s') if ipo_timestamp_epoch else stock_market_data['Close'].first_valid_index()
                    if pd.isna(ipo_date) or ipo_date.year < 1990: return None
                    
                    filing_date_naive = filing_date.normalize()
                    ipo_date_naive = ipo_date.normalize()
                    ratios['Days_Since_IPO'] = (filing_date_naive - ipo_date_naive).days
                
                    return ratios

        except Exception as exc:
            error_message = str(exc)
            # Check for specific rate-limiting text in the error
            if "Too Many Requests" in error_message or "401" in error_message or "rate limited" in error_message:
                if attempt < max_retries - 1:
                    # Calculate wait time with exponential backoff and jitter
                    wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"--> [RATE LIMIT] for {ticker}. Retrying in {wait_time:.2f} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue # Go to the next attempt in the loop
                else:
                    # Final attempt failed
                    print(f"--> [FINAL ERROR] Failed to process {ticker} after {max_retries} attempts: {exc}")
                    return None
            else:
                # Handle other, non-rate-limit errors
                print(f"--> [ERROR] Failed to process {ticker} with a non-recoverable error: {exc}")
                return None # Exit loop for other errors

    return None


# def batch_fetch_financial_data(df, max_workers=4):
#     """
#     Processes each ticker, now including a point-in-time beta calculation
#     by fetching and distributing SPY market data.
#     """
#     df_copy = df.copy()
#     df_copy['Filing Date'] = pd.to_datetime(df_copy['Filing Date'], dayfirst=True)
#     tickers = df_copy['Ticker'].unique().tolist()
    
#     # --- Step 1: Bulk Data Downloads ---
#     print(f"Fetching fundamental data for {len(tickers)} tickers...")
#     tk_objects = yf.Tickers(" ".join(tickers))
    
#     start_date = '1970-01-01'
#     end_date = df_copy['Filing Date'].max() + pd.Timedelta(days=1)
    
#     print(f"Fetching historical prices for tickers from {start_date} to {end_date.date()}...")
#     hist_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', progress=True, auto_adjust=False, threads=True)

#     # --- Step 2: Fetch Market Index Data (SPY) ---
#     print("Fetching market index data (SPY) for beta calculation...")
#     market_data_spy = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True, threads=True)
#     # Ensure the index is a timezone-naive datetime object for safe comparisons
#     market_data_spy.index = market_data_spy.index.tz_localize(None)

#     # --- Step 3: Process Tickers in Parallel with SPY data ---
#     results = []
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         # Pass the market_data_spy object to each worker
#         future_to_row = {
#             executor.submit(process_single_ticker, row, tk_objects, hist_data, market_data_spy): row 
#             for row in df_copy.to_dict('records')
#         }
        
#         for future in tqdm(as_completed(future_to_row), total=len(future_to_row), desc="Processing Tickers in Parallel"):
#             try:
#                 result = future.result(timeout=30)
#                 if result is not None:
#                     results.append(result)
#             except Exception as exc:
#                 ticker = future_to_row[future].get('Ticker', 'Unknown')
#                 # print(f"\n[DEBUG] ERROR: Task for ticker '{ticker}' generated an exception: {exc}")

#     print(f"[INFO] Parallel processing finished. Successfully processed {len(results)} tickers.")
#     return pd.DataFrame(results)

def batch_fetch_financial_data(df, max_workers=4):
    """
    Processes each ticker to fetch company-specific data and then enriches the
    final output with pre-calculated, point-in-time market regime indicators.
    """
    df_copy = df.copy()
    df_copy['Filing Date'] = pd.to_datetime(df_copy['Filing Date'], dayfirst=True)
    tickers = df_copy['Ticker'].unique().tolist()

    # --- Step 1: Bulk Data Downloads (with new market indices) ---
    print(f"Fetching fundamental data for {len(tickers)} tickers...")
    tk_objects = yf.Tickers(" ".join(tickers))
    
    start_date = '1970-01-01'
    end_date = df_copy['Filing Date'].max() + pd.Timedelta(days=1)
    
    print(f"Fetching historical prices for tickers from {start_date} to {end_date.date()}...")
    hist_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', progress=True, auto_adjust=False, threads=True)
    
    print("Fetching market index data (SPY, VIX, GSPC) for regime indicators...")
    market_indices = yf.download('SPY ^VIX ^GSPC', start=start_date, end=end_date, auto_adjust=True, threads=True)
    market_data_spy = market_indices['Close']['SPY'].to_frame().rename(columns={'SPY': 'Close'})
    market_data_gspc = market_indices['Close']['^GSPC'].to_frame().rename(columns={'^GSPC': 'Close'})
    market_data_vix = market_indices['Close']['^VIX'].to_frame().rename(columns={'^VIX': 'Close'})

    # --- Step 2: Pre-calculate Market Regime Indicators ---
    print("[REGIME] Pre-calculating market regime indicators...")
    regime_df = pd.DataFrame(index=market_indices.index)
    
    # Volatility Indicators
    regime_df['VIX_Close'] = market_data_vix['Close']
    regime_df['VIX_SMA50'] = market_data_vix['Close'].rolling(window=50).mean()
    
    # Market Trend Indicators
    gspc_sma50 = market_data_gspc['Close'].rolling(window=50).mean()
    gspc_sma200 = market_data_gspc['Close'].rolling(window=200).mean()
    regime_df['SP500_Above_SMA50'] = (market_data_gspc['Close'] > gspc_sma50).astype(int)
    regime_df['SP500_Above_SMA200'] = (market_data_gspc['Close'] > gspc_sma200).astype(int)
    
    # Drop any initial rows with NaNs from rolling calculations
    regime_df.dropna(inplace=True)
    
    # --- Step 3: Process Company-Specific Tickers in Parallel (Unchanged) ---
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {
            executor.submit(process_single_ticker, row, tk_objects, hist_data, market_data_spy): row
            for row in df_copy.to_dict('records')
        }
        for future in tqdm(as_completed(future_to_row), total=len(future_to_row), desc="Processing Tickers in Parallel"):
            try:
                result = future.result(timeout=30)
                if result is not None:
                    results.append(result)
            except Exception as exc:
                pass # Error handling remains the same

    if not results:
        print("[INFO] No company-specific data could be processed.")
        return pd.DataFrame()

    company_ratios_df = pd.DataFrame(results)

    # --- Step 4: Point-in-Time Merge of Regime Indicators ---
    print("[REGIME] Merging market regime features into the final dataset...")
    
    # Ensure date columns are correctly formatted for merging
    company_ratios_df['Filing Date'] = pd.to_datetime(company_ratios_df['Filing Date'])
    company_ratios_df.sort_values('Filing Date', inplace=True)
    
    # Use merge_asof for a robust, point-in-time join.
    # This finds the latest regime indicator data available for each filing date.
    enriched_df = pd.merge_asof(
        company_ratios_df,
        regime_df,
        left_on='Filing Date',
        right_index=True,
        direction='backward'
    )
    
    print(f"[INFO] Final processing complete. Successfully processed {len(enriched_df)} entries.")
    return enriched_df