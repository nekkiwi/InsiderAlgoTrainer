import os
import pandas as pd
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST
from src.alpaca.utils.alpaca_trader_helpers import load_api_keys
import time
from datetime import timezone
from src.alpaca.utils.alpaca_trader_helpers import load_api_keys
from src.alpaca.utils.alpaca_trader_helpers import (
    get_latest_buy_order,
    make_timezone_aware,
    get_price_and_quantity,
    submit_sell_order,
    submit_buy_order,
)

class AlpacaTrader:
    """
    Connects to Alpaca API, reads inference signals, places buy orders,
    and schedules sell orders one month later.
    """
    def __init__(self):
        # Credentials and client
        keys = load_api_keys(os.path.join(os.getcwd(), "src/alpaca/api_keys.json"))
        print(os.path.join(os.getcwd(), "api_keys.json"))
        self.api_key = keys['api_key']
        self.api_secret = keys['api_secret_key']
    
        self.base_url = 'https://paper-api.alpaca.markets'
        self.client = REST(self.api_key, self.api_secret, self.base_url)

        # Trading settings
        self.holding_period_days = 0

        # Inference settings
        self.inference_file_dir = 'data/inference/'
        self.inference_file = ""
        self.threshold = 0

    def sell_matured_positions(self):
        print("Checking for positions to sell...")
        try:
            positions = self.client.list_positions()
            if not positions:
                print("No open positions to check.")
                return

            for position in positions:
                try:
                    orders = get_latest_buy_order(self.client, position.symbol)
                    if not orders:
                        print(f"No buy order found for {position.symbol}.")
                        continue

                    purchase_date = make_timezone_aware(orders[0].filled_at)
                    holding_duration = datetime.now(timezone.utc) - purchase_date

                    if holding_duration.days > self.holding_period_days:
                        print(f"Selling {position.symbol}, held {holding_duration.days} days.")
                        submit_sell_order(self.client, position.symbol, position.qty)
                        print(f"SELL order placed for {position.qty} shares of {position.symbol}.")
                    else:
                        print(f"{position.symbol} held {holding_duration.days} days — within holding period.")

                except Exception as e:
                    print(f"Error processing position {position.symbol}: {e}")

        except Exception as e:
            print(f"Error fetching open positions: {e}")

    def read_signals(self, df) -> pd.DataFrame:
        """
        Load inference output, filter tickers with score >= threshold.
        Expects first column 'Ticker', third column is score.
        """
        df.columns = [c.strip() for c in df.columns]
        ticker_col = df.columns[0]
        score_col = df.columns[2]
        signals = df[df[score_col] >= self.threshold][[ticker_col, score_col]].copy()
        signals.columns = ['symbol', 'score']
        return signals

    def place_orders(self, symbol, amount: float):
        print("\nChecking for stocks to buy...")

        # Get currently held positions
        held_symbols = {p.symbol for p in self.client.list_positions()}

        # Get open (unfilled) buy orders
        open_orders = self.client.list_orders(status='open')
        open_buy_symbols = {o.symbol for o in open_orders if o.side == 'buy'}

        if symbol in held_symbols:
            print(f"Skipping {symbol}: already held.")
            return

        if symbol in open_buy_symbols:
            print(f"Skipping {symbol}: buy order already open.")
            return

        try:
            price, qty_to_buy = get_price_and_quantity(self.client, symbol, amount)

            if qty_to_buy <= 0:
                print(f"Skipping {symbol}: ${amount:.2f} < ${price:.2f}")
                return

            print(f"Placing BUY for {qty_to_buy} shares of {symbol} at ~${price:.2f} for a total of: ${qty_to_buy * price:.2f}.")
            submit_buy_order(self.client, symbol, qty_to_buy)

        except Exception as e:
            print(f"Error buying {symbol}: {e}")


    def run(self, config, amount, holding_period, results_df=None):
        """
        High-level orchestration: sell matured positions, then read signals and place new buy orders.

        Config dict determines mode:
            - Score-based: config contains only 'symbol'
            - Threshold-based: config contains 'targets' and 'threshold'
        """
        self.holding_period_days = holding_period
        self.sell_matured_positions()

        if "symbol" in config and len(config) == 1:
            # --- SCORE-BASED MODE ---
            self.place_orders(config["symbol"], amount)
        elif "targets" in config and "threshold" in config:
            # --- THRESHOLD-BASED MODE ---
            self.threshold = config["threshold"]
            for target in config["targets"]:
                if results_df is None: 
                    self.inference_file = os.path.join(self.inference_file_dir, f"{target}_inference_output.xlsx")
                    df = pd.read_excel(self.inference_file)
                else:
                    df = results_df
                signals = self.read_signals(df)
                if signals.empty:
                    print(f"\nNo new signals above threshold for {target}. Nothing to buy.")
                else:
                    for _, row in signals.iterrows():
                        self.place_orders(row["symbol"], amount)
        else:
            raise ValueError("❌ Invalid config: must contain either only 'symbol', or both 'targets' and 'threshold'")


        
        
if __name__ == '__main__':
    trader = AlpacaTrader()
    trader.run()