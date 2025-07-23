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
    
    timepoints = ["1w", "1m", "3m", "6m"]  # ["1w", "1m", "3m", "6m", "8m"]
    thresholds = [0, 2, 4, 5, 6, 8, 10]    # [0, 2, 5, 10, 15] # used for binary signals
    category   = "alpha"                   # alpha or return
    top_n      = 10                        # top 20 relevant features chosen, 10 was worse
    model      = "LightGBM"                # "RandomForest" or "LightGBM"
    seeds      = [42, 123, 2024, 99, 7, 234, 345]    # A list of 5 different seeds
    
    tune_hyperparameters = False    # Set to True to run the RandomizedSearch, turned out to be worse than default models

    model_trainer.run(category, timepoints, thresholds, model, top_n, seeds, tune_hyperparameters)
    
if __name__ == "__main__":
    main()
