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
    top_n       = 10            # top 20 relevant features chosen

    # feature_selector.run(category, timepoints, thresholds, top_n)
    
    ############
    # Training #
    ############
    
    model        = "RandomForest"
    optimize_for = "adjusted_sharpe" # sharpe, adjusted_sharpe or information_ratio
    seeds        = [42, 123, 2024, 99, 7]  # A list of 5 different seeds

    model_trainer.run(category, timepoints, thresholds, model, optimize_for, top_n, seeds)
    
if __name__ == "__main__":
    main()
