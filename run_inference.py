from src.scraper.feature_scraper import FeatureScraper
from src.scraper.feature_preprocess import FeaturePreprocessor
from src.inference.model_inference import ModelInference

def main():
    ###################
    # Initializations #
    ###################
    
    feature_scraper         = FeatureScraper()
    feature_preprocessor    = FeaturePreprocessor()
    model_inference         = ModelInference()
    
    current_features_df_preprocessed = None
    
    ####################
    # Get Current Data #
    ####################

    current_features_df = feature_scraper.run(num_weeks=1, train=False)
    current_features_df_preprocessed = feature_preprocessor.run(current_features_df, train=False)
    
    #######################
    # Run Model Inference #
    #######################
    
    models  = ['RandomForestOverSample']
    targets = ['final_return_1m_raw']
    model_inference.run(current_features_df_preprocessed, models, targets)
    
if __name__ == "__main__":
    main()
