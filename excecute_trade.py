from src.alpaca.alpaca_trader import AlpacaTrader

def main():
    ###################
    # Initializations #
    ###################
    
    alpaca_trader = AlpacaTrader()
    
    #############
    # Run Trade #
    #############

    amount = 100 # €
    holding_period = 30 # days
    
    # If just following the model, specify model and threshold
    config = {
        "targets": ["final_return_1m_raw"],
        "threshold": 0.06
    }
    
    # If using the find_good_invesment, specify ticker
    # config = {"symbol": "AAPL"}

    alpaca_trader.run(config, amount, holding_period)
    
if __name__ == "__main__":
    main()
