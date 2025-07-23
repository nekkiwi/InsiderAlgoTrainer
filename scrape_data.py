from src.scraper.feature_scraper import FeatureScraper
from src.scraper.stock_scraper import StockDataScraper
from src.scraper.feature_preprocess import FeaturePreprocessor
from src.scraper.stock_analysis import StockAnalysis

def main():    
    ###################
    # Initializations #
    ###################
    
    feature_scraper         = FeatureScraper()
    feature_preprocessor    = FeaturePreprocessor()
    stock_scraper           = StockDataScraper()
    stock_analyzer          = StockAnalysis()
    
    features_df                         = None
    features_df_preprocessed            = None
    
    ###################
    # Feature Scraper #
    ###################
    
    num_weeks = 5 * 12 * 4
    
    features_df = feature_scraper.run(num_weeks)
    features_df_preprocessed = feature_preprocessor.run(features_df)
    
    #################
    # Stock Scraper #
    #################
    
    timepoints = ['1w', '1m', '2m', '3m', '4m', '5m', '6m']
    
    stock_scraper.run(features_df_preprocessed, timepoints)
    stock_analyzer.run()
    
if __name__ == "__main__":
    main()
