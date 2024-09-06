import sys
import os
from src.scraper.target_scraper import TargetScraper
from src.scraper.feature_scraper import FeatureScraper
from src.scraper.stock_scraper import StockDataScraper
from src.analysis.feature_selector import FeatureSelector
from src.analysis.feature_analysis import FeatureAnalyzer
from src.analysis.stock_analysis import StockAnalysis
from src.training.train import ModelTrainer
from src.training.evaluate import StockEvaluator

def scrape_features(num_months):    
    scraper = FeatureScraper()
    scraper.run(num_months)
    sys.stdout.flush()
    
def run_feature_analysis():
    analyzer = FeatureAnalyzer()
    analyzer.run_analysis()
    sys.stdout.flush()
    
def scrape_stockdata():
    scraper = StockDataScraper()
    scraper.run()
    sys.stdout.flush()
    
def run_analysis():    
    analyzer = StockAnalysis()
    analyzer.run()
    sys.stdout.flush()
    
def scrape_targets(limit_array, stop_array):
    scraper = TargetScraper()
    scraper.run(limit_array, stop_array)
    sys.stdout.flush()
    
def select_features():
    selector = FeatureSelector()
    selector.run()
    sys.stdout.flush()
    
def train_model(target_name, model_type):
    trainer = ModelTrainer()
    trainer.run(target_name, model_type)
    sys.stdout.flush()
    
    
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
    num_months=1
    
    scrape_features(num_months)
    run_feature_analysis()
    scrape_stockdata()
    # run_analysis()
    
    # limit_array = [ 0.02, 0.04,  0.05, 0.06,  0.07, 0.08,  0.09, 0.1,  0.12]
    # stop_array  = [-0.16,-0.14,-0.12, -0.1, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04, -0.02]
    
    # scrape_targets(limit_array, stop_array)
    # select_features()
    
    # model_types = ["RandomForest"]
    
    # # Limit-Stop Criterion
    # for model_type in model_types:
    #     clear_output(model_type)
        
    # for model_type in model_types:
    #     train_model('pos-return', model_type)
    #     evaluate_model('pos-return', model_type)
        
if __name__ == "__main__":
    main()
