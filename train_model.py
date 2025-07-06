from src.training.feature_selector import FeatureSelector
from src.training.train import ModelTrainer
    
def main():   
    ###################
    # Initializations #
    ###################
    
    feature_selector    = FeatureSelector()
    
    features_df_filtered    = None
    targets_df              = None
    
    ############
    # Training #
    ############

    selected_features = feature_selector.run(features_df_filtered, targets_df, p_threshold=0.05)
    
    models  = ['RandomForestOverSample']
    targets = ['final_return_1m_raw']

    for model in models: 
        for target in targets:
            ModelTrainer().run(target, model, selected_features, features_df_filtered, targets_df)

if __name__ == "__main__":
    main()
