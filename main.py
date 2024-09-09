import sys
import os
import pandas as pd
from src.scraper.target_scraper import TargetScraper
from src.scraper.feature_scraper import FeatureScraper
from src.scraper.stock_scraper import StockDataScraper
from src.training.feature_selector import FeatureSelector
from src.data_exploration.feature_analysis import FeatureAnalyzer
from src.data_exploration.stock_analysis import StockAnalysis
from src.training.train import ModelTrainer
from src.analysis.evaluate import StockEvaluator
    
    
def evaluate_model(criterion, model_type):
    evaluator = StockEvaluator(model_type, criterion)
    evaluator.run_evaluation()
    sys.stdout.flush()
    
    analysis = StockAnalysis()
    analysis.run_all_simulations(model_type, criterion)
    sys.stdout.flush()
    
def clear_output(model_type):
    def remove_directory_content(directory, type):
        # Create a subdirectory for the model under the predictions directory
        data_dir = os.path.join(os.path.dirname(__file__), '../data')
        predictions_dir = os.path.join(data_dir, directory)
        
        if type == 'file':
            # Clear the directory before repopulating it
            files = os.listdir(predictions_dir)
            for file in files:
                file_path = os.path.join(predictions_dir, file)
                os.remove(file_path)
                print(f'Removed file at {file_path}.')
                
        if type == 'dir':
            for root, dirs, files in os.walk(predictions_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
                    
    remove_directory_content('training/predictions/'+model_type.replace(" ", "-").lower(), 'file')
    remove_directory_content('training/simulation/'+model_type.replace(" ", "-").lower(), 'file')
    
    # Clear the stock analysis directory but run the 'all' again
    remove_directory_content('output/stock_analysis', 'dir')
    
    analyzer = StockAnalysis()
    analyzer.run()
    sys.stdout.flush()


def main():    
    feature_scraper = FeatureScraper()
    feature_analyzer = FeatureAnalyzer()
    stock_scraper = StockDataScraper()
    stock_analyzer = StockAnalysis()
    target_scraper = TargetScraper()
    feature_selector = FeatureSelector()
    trainer = ModelTrainer()
    features_df = None
    return_df = None
    alpha_df = None
    targets_df = None
    features_df_preprocessed = None
    features_df_filtered = None
    selected_features = None
    
    # num_months=1
    # features_df = feature_scraper.run(num_months)
    # features_df_preprocessed = feature_analyzer.run(features_df)
    # features_df_filtered, return_df, alpha_df = stock_scraper.run(features_df_preprocessed)
    # stock_analyzer.run(return_df, alpha_df)
    
    # limit_array = [ 0.02, 0.08, 0.12]
    # stop_array  = [-0.16,-0.08,-0.02]
    # high_threshold = 0.04
    
    # targets_df = target_scraper.run(return_df, alpha_df, limit_array, stop_array, high_threshold)
    
    # p_threshold=0.05
    # selected_features = feature_selector.run(features_df_filtered, targets_df, p_threshold)
    
    models = ["RandomForest"]
    targets = ["final_alpha"] 
    
    for target, model in zip(targets, models):
        trainer.run(target, model, selected_features, features_df, targets_df)
        
if __name__ == "__main__":
    main()
