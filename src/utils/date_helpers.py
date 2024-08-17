import pandas as pd
import datetime

def get_next_market_open(dt):
    # Assume market hours are 9:00 AM to 5:00 PM
    market_open = datetime.time(9, 0)
    market_close = datetime.time(17, 0)
    
    # If it's before 9:00 AM, move to 9:00 AM today
    if dt.time() < market_open:
        return dt.replace(hour=9, minute=0, second=0, microsecond=0)
    
    # If it's after 5:00 PM, move to 9:00 AM the next business day
    if dt.time() >= market_close:
        dt = dt + pd.tseries.offsets.BDay()
        return dt.replace(hour=9, minute=0, second=0, microsecond=0)
    
    # Round up to the next full or half hour
    minutes = dt.minute
    if minutes > 0 and minutes <= 30:
        return dt.replace(minute=30, second=0, microsecond=0)
    else:
        dt += pd.tseries.offsets.Hour()
        return dt.replace(minute=0, second=0, microsecond=0)