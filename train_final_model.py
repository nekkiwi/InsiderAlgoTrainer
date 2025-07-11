import matplotlib
matplotlib.use('Agg')
from src.training.train_final import FinalModelTrainer
    
def main():   
    ###################
    # Initializations #
    ###################
    
    final_model_trainer = FinalModelTrainer()
        
    #####################
    # Feature Selection #
    #####################
    
    timepoint   = "6m"
    threshold   = 8
    category    = "alpha"
    top_n       = 10 
    model       = "RandomForest"
    seeds       = [42, 123, 2024, 99, 7]  # A list of 5 different seeds
    
    ############
    # Training #
    ############
    
    final_model_trainer.run(category, timepoint, threshold, model, top_n, seeds)
    
if __name__ == "__main__":
    main()
