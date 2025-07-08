from src.training.feature_selector import FeatureSelector
from src.training.train import ModelTrainer
    
def main():   
    ###################
    # Initializations #
    ###################
    
    feature_selector    = FeatureSelector()
    model_trainer       = ModelTrainer()
    
    selected_features       = None
    
    #####################
    # Feature Selection #
    #####################

    # selected_features = feature_selector.run(p_threshold=0.05)
    
    ############
    # Training #
    ############
    
    models      = ["RandomForestOverSample"]
    timepoints  = ["1w", "2w", "3w", "6w", "2m"]
    metrics     = ["return"]

    for model in models: 
        model_trainer.run(model, timepoints, metrics)

if __name__ == "__main__":
    main()
