import yfinance as yf
import contextlib
import os
from datetime import timedelta
import pandas as pd
import json

def fetch_financial_data(ticker, filing_date):
    """Fetch historical financial data and metadata using yfinance."""
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            stock = yf.Ticker(ticker)

            # Ensure filing_date is a string in the format 'YYYY-MM-DD'
            filing_date_str = filing_date.strftime('%Y-%m-%d')
            
            stock.balance_sheet.columns = pd.to_datetime(stock.balance_sheet.columns)
            stock.cashflow.columns = pd.to_datetime(stock.cashflow.columns)
            stock.financials.columns = pd.to_datetime(stock.financials.columns)
            
            # Fetch historical balance sheet, cash flow, income statement
            try:
                balance_sheet = stock.balance_sheet.loc[:, stock.balance_sheet.columns[stock.balance_sheet.columns <= filing_date_str]].iloc[:, 0]
                cash_flow = stock.cashflow.loc[:, stock.cashflow.columns[stock.cashflow.columns <= filing_date_str]].iloc[:, 0]
                income_statement = stock.financials.loc[:, stock.financials.columns[stock.financials.columns <= filing_date_str]].iloc[:, 0]
            except IndexError:
                # If there is no data before the filing date, return None
                balance_sheet, cash_flow, income_statement = None, None, None
            except Exception as e:
                print(f"Failed to fetch financial data for {ticker}: {e}")
                return None

            # Fetch historical market data (stock price and others)
            try:
                historical_data = stock.history(start=filing_date - pd.tseries.offsets.BDay(3), end=filing_date)
            except Exception as e:
                print(f"Failed to fetch historical data for {ticker}: {e}")
                historical_data = pd.DataFrame()

            if not historical_data.empty:
                latest_market_data = historical_data.iloc[-1]
                try:
                    market_cap = stock.info.get('marketCap', None)
                    sector = stock.info.get('sector', None)
                    industry = stock.info.get('industry', None)
                    eps = stock.info.get('trailingEps', None)
                    beta = stock.info.get('beta', None)
                    high_52_week = stock.info.get('fiftyTwoWeekHigh', None)
                    low_52_week = stock.info.get('fiftyTwoWeekLow', None)
                    average_volume = stock.info.get('averageVolume', None)
                except (json.decoder.JSONDecodeError, KeyError) as e:
                    print(f"Failed to fetch metadata for {ticker}: {e}")
                    return None

                data = {
                    'balance_sheet': balance_sheet,
                    'cash_flow': cash_flow,
                    'income_statement': income_statement,
                    'market_cap': market_cap,
                    'sector': sector,
                    'industry': industry,
                    'eps': eps,
                    'beta': beta,
                    'high_52_week': high_52_week,
                    'low_52_week': low_52_week,
                    'average_volume': average_volume,
                    'current_price': latest_market_data.get('Close', None)
                }
            else:
                return None

    return data


def calculate_financial_ratios(data):
    """Calculate and normalize financial ratios from balance sheet, cash flow, income statement, and metadata."""
    ratios = {}

    # Check if data is available
    if not data or data.get('balance_sheet') is None or data.get('cash_flow') is None or data.get('income_statement') is None:
        return None

    # Assuming the balance_sheet, cash_flow, and income_statement are Series
    # (if they are DataFrames, you need to adjust accordingly)
    balance_sheet = data['balance_sheet']
    cash_flow = data['cash_flow']
    income_statement = data['income_statement']

    # Profitability Ratios
    net_income = income_statement.get('Net Income', None)
    revenue = income_statement.get('Total Revenue', None)
    total_assets = balance_sheet.get('Total Assets', None)
    stockholders_equity = balance_sheet.get('Stockholders Equity', None)

    ratios['Net_Profit_Margin'] = net_income / revenue if net_income and revenue else None
    ratios['ROA'] = net_income / total_assets if net_income and total_assets else None
    ratios['ROE'] = net_income / stockholders_equity if net_income and stockholders_equity else None

    # Leverage Ratios
    total_liabilities = balance_sheet.get('Total Liabilities Net Minority Interest', None)
    total_equity = balance_sheet.get('Total Equity Gross Minority Interest', None)
    ratios['Debt_to_Equity'] = total_liabilities / stockholders_equity if total_liabilities and stockholders_equity else None
    
    # Cash Flow Ratios
    operating_cash_flow = cash_flow.get('Operating Cash Flow', None)
    investing_cash_flow = cash_flow.get('Investing Cash Flow', None)
    financing_cash_flow = cash_flow.get('Financing Cash Flow', None)
    capital_expenditure = cash_flow.get('Capital Expenditure', None)
    ratios['Operating_Cash_Flow'] = operating_cash_flow if operating_cash_flow else None
    ratios['Investing_Cash_Flow'] = investing_cash_flow if investing_cash_flow else None
    ratios['Financing_Cash_Flow'] = financing_cash_flow if financing_cash_flow else None
    ratios['Free_Cash_Flow'] = operating_cash_flow - capital_expenditure if operating_cash_flow and capital_expenditure else None

    # Valuation Ratios (Normalize by Market Cap where it makes sense)
    market_cap = data.get('market_cap')
    ratios['Market_Cap'] = market_cap  # Add Market Cap
    ratios['Price_to_Earnings_Ratio'] = market_cap / net_income if market_cap and net_income else None
    ratios['Price_to_Book_Ratio'] = market_cap / stockholders_equity if market_cap and stockholders_equity else None
    ratios['Price_to_Sales_Ratio'] = market_cap / revenue if market_cap and revenue else None
    ratios['Operating_Cash_Flow_to_Market_Cap'] = operating_cash_flow / market_cap if operating_cash_flow and market_cap else None
    ratios['Investing_Cash_Flow_to_Market_Cap'] = investing_cash_flow / market_cap  if investing_cash_flow and market_cap else None
    ratios['Financing_Cash_Flow_to_Market_Cap'] = financing_cash_flow / market_cap  if financing_cash_flow and market_cap else None
    ratios['Net_Income_to_Market_Cap'] = net_income / market_cap if net_income and market_cap else None
    ratios['Total_Assets_to_Market_Cap'] = total_assets / market_cap if total_assets and market_cap else None
    ratios['Total_Liabilities_to_Market_Cap'] = total_liabilities / market_cap if total_liabilities and market_cap else None
    ratios['Total_Equity_to_Market_Cap'] = total_equity / market_cap if total_equity and market_cap else None
    ratios['Average_Volume_to_Market_Cap'] = data.get('average_volume') / market_cap if market_cap else None
    if ratios.get('Free_Cash_Flow'):
        ratios['Free_Cash_Flow_to_Market_Cap'] = ratios['Free_Cash_Flow'] / market_cap if market_cap else None

    # Additional Information
    ratios['Sector'] = data.get('sector')
    ratios['Industry'] = data.get('industry')
    ratios['EPS'] = data.get('eps')
    ratios['Beta'] = data.get('beta')

    current_price = data.get('current_price')
    if current_price:
        high_52_week = data.get('high_52_week')
        low_52_week = data.get('low_52_week')
        ratios['52_Week_High_Normalized'] = high_52_week / current_price if high_52_week else None
        ratios['52_Week_Low_Normalized'] = low_52_week / current_price if low_52_week else None

    return ratios


def process_ticker_financial_ratios(row):
    """Process each ticker by downloading the stock data, benchmark data, and calculating indicators."""
    ticker = row['Ticker']
    filing_date = pd.to_datetime(row['Filing Date'], dayfirst=True)

    # Fetch financial data up to the filing date
    data = fetch_financial_data(ticker, filing_date)
    
    # Calculate financial ratios
    financial_ratios = calculate_financial_ratios(data)
    
    if financial_ratios:
        row.update(financial_ratios)
        return row
    
    return None

