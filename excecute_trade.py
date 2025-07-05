from src.alpaca.alpaca_trader import AlpacaTrader

def main():
    ###################
    # Initializations #
    ###################
    
    alpaca_trader = AlpacaTrader()
    
    #############
    # Run Trade #
    #############

    targets = ['pos_alpha_1m_raw']
    amount = 100 # €
    holding_period = 30 # days
    threshold = 0.7 # prediction threshold after which I buy

    alpaca_trader.run(targets, threshold, amount, holding_period)
    
if __name__ == "__main__":
    main()
