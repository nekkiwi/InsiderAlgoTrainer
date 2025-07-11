import yfinance as yf
import pandas as pd
import time
import os
import contextlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
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

def process_single_ticker(row, tk_objects, hist_data):
    """
    Worker function to process a single ticker-date combination using only
    pre-fetched data. This version includes a check for valid historical data.
    """
    ticker = row['Ticker']
    filing_date = row['Filing Date']

    try:
        t_obj = tk_objects.tickers.get(ticker)
        t_info = t_obj.info if hasattr(t_obj, 'info') else {}
        if not t_info:
            return None

        def pick_latest(stmt_df):
            if stmt_df is None or stmt_df.empty: return None
            stmt_df.columns = pd.to_datetime(stmt_df.columns).tz_localize(None)
            valid_cols = stmt_df.columns[stmt_df.columns <= filing_date]
            return stmt_df.loc[:, valid_cols.max()] if not valid_cols.empty else None

        market_data = hist_data.get(ticker)
        if market_data is None or market_data['Close'].isnull().all():
            return None
        
        market_data.index = market_data.index.tz_localize(None)
        
        market_data_filtered = market_data.loc[market_data.index <= filing_date]
        if market_data_filtered.empty: return None
        latest_market_data = market_data_filtered.iloc[-1]

        data_payload = {
            'balance_sheet': pick_latest(t_obj.balance_sheet),
            'cash_flow': pick_latest(t_obj.cashflow),
            'income_statement': pick_latest(t_obj.financials),
            'market_cap': t_info.get('marketCap'),
            'sector': t_info.get('sector'),
            'eps': t_info.get('trailingEps'),
            'beta': t_info.get('beta'),
            'high_52_week': t_info.get('fiftyTwoWeekHigh'),
            'low_52_week': t_info.get('fiftyTwoWeekLow'),
            'current_price': latest_market_data.get('Close')
        }
        
        ratios = calculate_financial_ratios(data_payload)

        if ratios:
            ratios['Ticker'] = ticker
            ratios['Filing Date'] = filing_date
            
            # --- Definitive IPO Date Logic ---
            ipo_date = None

            # 1. Primary Method: Use the 'info' object for the most reliable date.
            #    'firstTradeDateEpochUtc' is a Unix timestamp.
            ipo_timestamp_epoch = t_info.get('firstTradeDateEpochUtc')
            if ipo_timestamp_epoch:
                ipo_date = pd.to_datetime(ipo_timestamp_epoch, unit='s')

            # 2. Fallback Method: If metadata fails, use the first valid trade date.
            if pd.isna(ipo_date):
                ipo_date = market_data['Close'].first_valid_index()

            # 3. Final Validation: Ensure the date is plausible (e.g., after 1990).
            #    This check will eliminate the anomalous old dates.
            if pd.isna(ipo_date) or ipo_date.year < 1990:
                return None

            # Normalize dates to midnight to ensure an accurate day count
            filing_date_naive = filing_date.normalize()
            ipo_date_naive = ipo_date.normalize()
            ratios['Days_Since_IPO'] = (filing_date_naive - ipo_date_naive).days
        
            return ratios

    except Exception:
        return None
    return None

def batch_fetch_financial_data(df, max_workers=8):
    """
    Processes each ticker one by one to isolate crashes.
    """
    df_copy = df.copy()
    df_copy['Filing Date'] = pd.to_datetime(df_copy['Filing Date'], dayfirst=True)
    tickers = df_copy['Ticker'].unique().tolist()
    
    # Step 1: Bulk Data Downloads (unchanged)
    print(f"Fetching fundamental data for {len(tickers)} tickers...")
    tk_objects = yf.Tickers(" ".join(tickers))
    
    # start_date = df_copy['Filing Date'].min() - pd.Timedelta(days=5)
    
    # Fetch the full history to ensure IPO date calculation is correct.
    start_date = '1970-01-01'
    
    end_date = df_copy['Filing Date'].max() + pd.Timedelta(days=1)
    print(f"Fetching historical prices from {start_date} to {end_date.date()}...")
    hist_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', progress=True, auto_adjust=False, threads=True)

    # Step 2: Process tickers sequentially in the main thread
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {executor.submit(process_single_ticker, row, tk_objects, hist_data): row for row in df_copy.to_dict('records')}
        
        for future in tqdm(as_completed(future_to_row), total=len(future_to_row), desc="Processing Tickers in Parallel"):
            # The timeout can be shorter as no network calls are made here
            try:
                result = future.result(timeout=30)
                if result is not None:
                    results.append(result)
            except Exception as exc:
                ticker = future_to_row[future].get('Ticker', 'Unknown')
                # print(f"\n[DEBUG] ERROR: Task for ticker '{ticker}' generated an exception: {exc}")

    print(f"[INFO] Parallel processing finished. Successfully processed {len(results)} tickers.")
    return pd.DataFrame(results)