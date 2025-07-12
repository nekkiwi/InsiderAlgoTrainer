import matplotlib
matplotlib.use('Agg')
from src.training.feature_selector import FeatureSelector
from src.training.train import ModelTrainer
    
def main():   
    ###################
    # Initializations #
    ###################
    
    feature_selector    = FeatureSelector()
    model_trainer       = ModelTrainer()
        
    #####################
    # Feature Selection #
    #####################
    
    timepoints  = ["1w", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m"]
    thresholds  = [0, 2, 4, 6, 8, 10, 12, 14] # used for binary signals
    category    = "alpha"       # alpha or return
    top_n       = 20            # top 20 relevant features chosen

    # feature_selector.run(category, timepoints, thresholds, top_n)
    
    ############
    # Training #
    ############
    
    model                   = "LightGBM" # "RandomForest" or "LightGBM"
    tune_hyperparameters    = True    # Set to True to run the RandomizedSearch
    seeds                   = [42, 123, 2024, 99, 7]  # A list of 5 different seeds

    model_trainer.run(category, timepoints, thresholds, model, top_n, seeds, tune_hyperparameters)
    
if __name__ == "__main__":
    main()
