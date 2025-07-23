import matplotlib
matplotlib.use('Agg')
from src.training.train import ModelTrainer
    
def main(): 
    
    ###################
    # Initializations #
    ###################
    
    model_trainer = ModelTrainer()
        
    #########################
    # Walk Forward Training #
    #########################
    
    timepoints      = ['1w', '1m']#, '3m', '6m'] # ['1w', '1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m']
    thresholds      = [0, 2] # [0, 2, 4, 5, 6, 8, 10, 12, 14]
    category        = "alpha"
    top_n           = 10
    model           = "LightGBM"
    seeds           = [42, 123, 2024, 99, 7] #, 11, 23, 43, 100, 44, 76, 1283, 999]
    
    tp_sl_configs = [(0.1, 0.15),
                     (0.1, 0.2),
                     (0.1, 0.25)]
    
    tune_hyperparameters = False

    # Pass the TP/SL parameters to the run method
    model_trainer.run(
        category, 
        timepoints, 
        thresholds, 
        model, 
        top_n, 
        seeds, 
        tp_sl_configs,
        tune_hyperparameters
    )
    
if __name__ == "__main__":
    main()
