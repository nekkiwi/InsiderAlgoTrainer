from src.scraper.target_scraper import TargetScraper
from src.scraper.feature_scraper import FeatureScraper
from src.scraper.stock_scraper import StockDataScraper
from src.training.feature_selector import FeatureSelector
from src.data_exploration.feature_analysis import FeatureAnalyzer
from src.data_exploration.stock_analysis import StockAnalysis
from src.training.train import ModelTrainer

def main():    
    ################################
    # Initializations (dont touch) #
    ################################
    
    feature_scraper     = FeatureScraper()
    feature_analyzer    = FeatureAnalyzer()
    stock_scraper       = StockDataScraper()
    stock_analyzer      = StockAnalysis()
    target_scraper      = TargetScraper()
    feature_selector    = FeatureSelector()
    
    features_df                 = None
    return_df                   = None
    alpha_df                    = None
    targets_df                  = None
    features_df_preprocessed    = None
    features_df_filtered        = None
    selected_features           = None
    
    ###################
    # Feature Scraper #
    ###################
    
    # num_months = 60
    # features_df = feature_scraper.run(num_months)
    # features_df_preprocessed = feature_analyzer.run(features_df)
    
    #################
    # Stock Scraper #
    #################
    
    # features_df_filtered, return_df, alpha_df = stock_scraper.run(features_df_preprocessed)
    # stock_analyzer.run(return_df, alpha_df)
    
    ##################
    # Target Scraper #
    ##################
    
    # limit_array = [ 0.02,  0.04,  0.06, 0.08,   0.1]
    # stop_array  = [-0.14, -0.12, -0.1, -0.08, -0.06, -0.04, -0.02]
    
    # targets_df = target_scraper.run(return_df, alpha_df, limit_array, stop_array)

    ############
    # Training #
    ############

    # selected_features = feature_selector.run(features_df_filtered, targets_df, p_threshold=0.05)
    
    models  = ['NeuralNetUnderSample', 'NeuralNetOverSample',    'RandomForestUnderSample', 'RandomForestOverSample']
    targets = ['pos_return_1m_raw',    'pos_alpha_1m_raw',       'final_return_1m_raw',     'final_alpha_1m_raw',
               'pos_return_1m_limstop','final_return_1m_limstop','pos_alpha_1m_limstop',    'final_alpha_1m_limstop']
               #'return_limit_sell',    'return_stop_sell',       'alpha_limit_sell',        'alpha_stop_sell']

    for model in models: 
        for target in targets:
            ModelTrainer().run(target, model, selected_features, features_df_filtered, targets_df)
        
    ###############
    # Backtesting #
    ###############
    
if __name__ == "__main__":
    main()
