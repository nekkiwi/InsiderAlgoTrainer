from src.scraper.target_scraper import TargetScraper
from src.scraper.feature_scraper import FeatureScraper
from src.scraper.stock_scraper import StockDataScraper
from src.scraper.feature_preprocess import FeaturePreprocessor
from src.scraper.stock_analysis import StockAnalysis

def main():    
    ###################
    # Initializations #
    ###################
    
    feature_scraper     = FeatureScraper()
    feature_preprocessor    = FeaturePreprocessor()
    stock_scraper       = StockDataScraper()
    stock_analyzer      = StockAnalysis()
    target_scraper      = TargetScraper()
    
    features_df                         = None
    return_df                           = None
    alpha_df                            = None
    features_df_preprocessed            = None
    
    ###################
    # Feature Scraper #
    ###################
    
    num_weeks = 4 # short test run
    # num_weeks = 5 * 12 * 4
    
    # features_df = feature_scraper.run(num_weeks, train=True)
    # features_df_preprocessed = feature_preprocessor.run(features_df, train=True)
    
    #################
    # Stock Scraper #
    #################
    
    # features_df_filtered, return_df, alpha_df = stock_scraper.run(features_df_preprocessed)
    # stock_analyzer.run(return_df, alpha_df)
    
    ##################
    # Target Scraper #
    ##################
    
    limit_array = [ 0.03]
    stop_array =  [-0.02]
    
    targets_df = target_scraper.run(return_df, alpha_df, limit_array, stop_array)

if __name__ == "__main__":
    main()
