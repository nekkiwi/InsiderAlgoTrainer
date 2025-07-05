import json
from datetime import timezone

def load_api_keys(filepath):
    with open(filepath, "r") as f:
        return json.load(f)
    
from datetime import timezone

def get_latest_buy_order(client, symbol):
    return client.list_orders(
        status='closed',
        symbols=[symbol],
        side='buy',
        limit=1,
        direction='desc'
    )

def make_timezone_aware(dt):
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def get_price_and_quantity(client, symbol, amount):
    """
    Returns latest price and quantity of shares to buy for a given amount.
    """
    latest_trade = client.get_latest_trade(symbol)
    price = latest_trade.p
    qty_to_buy = int(amount // price)
    return price, qty_to_buy

def submit_sell_order(client, symbol, qty):
    return client.submit_order(
        symbol=symbol,
        qty=qty,
        side='sell',
        type='market',
        time_in_force='day'
    )

def submit_buy_order(client, symbol, qty):
    return client.submit_order(
        symbol=symbol,
        qty=qty,
        side='buy',
        type='market',
        time_in_force='day'
    )
