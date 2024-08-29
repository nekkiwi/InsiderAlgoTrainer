import sys
import os
from scraper.target_scraper import TargetScraper
from scraper.feature_scraper import FeatureScraper
from scraper.stock_scraper import StockDataScraper
from analysis.feature_selector import FeatureSelector
from analysis.feature_analysis import FeatureAnalyzer
from analysis.stock_analysis import StockAnalysis
from training.train import ModelTrainer
from training.evaluate import StockEvaluator

def scrape_data(num_months):    
    # # Initialize and run the Feature Scraper
    print("Starting Feature Scraper...")
    scraper = FeatureScraper()
    scraper.run(num_months)
    print("Feature Scraper completed.\n")
    
    # # Ensure the scraper has completed before moving on
    # sys.stdout.flush()
    
    # TODO: wait until feature_final is saved and then move on 
    # Initialize and run the Stock Scraper
    print("Starting Stock Scraper...")
    scraper = StockDataScraper()
    scraper.run()
    print("Stock Data Scraper completed.\n")
    
    # Ensure the scraper has completed before moving on
    sys.stdout.flush()
    
def run_analysis():
    # Initialize and run the Feature Analyzer
    print("Starting Feature Analyzer...")
    analyzer = FeatureAnalyzer()
    analyzer.run_analysis()
    print("Feature Analyzer completed.\n")
    
    # Ensure the scraper has completed before moving on
    sys.stdout.flush()
    
    # Initialize and run the Stock Analyzer
    print("Starting Stock Analyzer...")
    analyzer = StockAnalysis()
    analyzer.run()
    print("Stock Analyzer completed.\n")
    
    # Ensure the scraper has completed before moving on
    sys.stdout.flush()
    
def scrape_targets(limit_array, stop_array):
    # Initialize and run the Target Scraper
    print("Starting Target Scraper...")
    scraper = TargetScraper()
    scraper.run(limit_array, stop_array)
    print("Target Scraper completed.\n")
    
    # Ensure the scraper has completed before moving on
    sys.stdout.flush()
    
def select_features():
    # Initialize and run the Feature Selector
    print("Starting Feature Selector...")
    selector = FeatureSelector()
    selector.run()
    print("Feature Selector completed.\n")

    # Ensure the feature selector has completed before moving on
    sys.stdout.flush()
    
def train_model(target_name, model_type):
    # Initialize and run the Model Trainer
    print("Starting Model Trainer...")
    trainer = ModelTrainer()
    trainer.run(target_name, model_type)
    print("Model Trainer completed.\n")
    
    # Ensure the scraper has completed before moving on
    sys.stdout.flush()
    
    
def evaluate_model(criterion, model_type):
    print("Starting Simulator...")
    evaluator = StockEvaluator(model_type, criterion)
    evaluator.run_evaluation()
    print("Simulator completed.\n")
    
    # Ensure the scraper has completed before moving on
    sys.stdout.flush()
    
    # Initialize the StockAnalysis instance
    print("Starting Backtesting...")
    analysis = StockAnalysis()
    analysis.run_all_simulations(model_type, criterion)
    print("Backtesting completed.\n")
    
    # Ensure the scraper has completed before moving on
    sys.stdout.flush()
    
def clear_output(model_type):
    def remove_directory_content(model_type, directory):
        # Create a subdirectory for the model under the predictions directory
        data_dir = os.path.join(os.path.dirname(__file__), '../data')
        predictions_dir = os.path.join(data_dir, directory)
                
        # Clear the directory before repopulating it
        dirs = os.listdir(predictions_dir)
        for dir in dirs:
            dir_path = os.path.join(predictions_dir, dir)
            os.rmdir(dir_path)
            print(f'Removed dir at {dir_path}.')
                
    remove_directory_content(model_type, 'training/predictions/'+model_type.replace(" ", "-").lower())
    remove_directory_content(model_type, 'training/simulation/'+model_type.replace(" ", "-").lower())
    
    # Clear the stock analysis directory but run the 'all' again
    remove_directory_content(model_type, 'output/stock_analysis')
    
    # Initialize and run the Stock Analyzer
    print("Starting Stock Analyzer...")
    analyzer = StockAnalysis()
    analyzer.run()
    print("Stock Analyzer completed.\n")
    
    # Ensure the scraper has completed before moving on
    sys.stdout.flush()

def main():    
    ########################################################################
    # Scrape Features and Stock Data for num_months
    num_months=36
    
    # scrape_data(num_months)
    # run_analysis()
    
    ########################################################################
    # Scrape Targets and select features for each target
    limit_array = [ 0.04, 0.05, 0.06, 0.08, 0.1, 0.12]
    stop_array  = [-0.06, -0.05, -0.04, -0.03, -0.02]
    
    # scrape_targets(limit_array, stop_array)
    # select_features()
    
    ########################################################################
    # Train models to predict given targets using given models and evaluate
    # Models: [RandomForest, NaivesBayes, RBF SVM, Gaussian Process, Neural Net]
    # Targets: [spike-up, spike-down, limit-occurred-first, stop-occurred-first, return-at-cashout, days-at-cashout]
    
    clear_output("RandomForest")
    for model_type in ["RandomForest"]:
        train_model('limit-occurred-first', model_type)
        train_model('stop-occurred-first', model_type)
        evaluate_model('limit-stop', model_type)

if __name__ == "__main__":
    main()
