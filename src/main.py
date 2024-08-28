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
    # Initialize and run the Feature Scraper
    print("Starting Feature Scraper...")
    scraper = FeatureScraper()
    scraper.run(num_months)
    print("Feature Scraper completed.\n")
    
    # Ensure the scraper has completed before moving on
    sys.stdout.flush()
    
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
    analyzer.run()
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
    evaluator = StockEvaluator(model_type, criterion)
    evaluator.run_evaluation()
    
    # Ensure the scraper has completed before moving on
    sys.stdout.flush()
    
    # Initialize the StockAnalysis instance
    analysis = StockAnalysis()
    analysis.run_all_simulations(model_type, criterion)
    
    # Ensure the scraper has completed before moving on
    sys.stdout.flush()

def main():    
    ########################################################################
    # Scrape Features and Stock Data for num_months
    num_months=36
    
    scrape_data(num_months)
    run_analysis()
    
    ########################################################################
    # Scrape Targets and select features for each target
    limit_array = [ 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    stop_array  = [-0.11, -0.1, -0.09, -0.08, -0.07, -0.06, -0.05]
    
    scrape_targets(limit_array, stop_array)
    select_features()
    
    ########################################################################
    # Train models to predict given targets using given models and evaluate
    # Models: [RandomForest, NaivesBayes, RBF SVM, Gaussian Process, Neural Net]
    # Targets: [spike-up, spike-down, limit-occurred-first, stop-occurred-first, return-at-cashout, days-at-cashout]
    for model_type in ["RandomForest", "NaivesBayes", "RBF SVM", "Gaussian Process", "Neural Net"]:
        train_model('limit-occurred-first', model_type)
        train_model('stop-occurred-first', model_type)
        evaluate_model('limit-stop', model_type)

if __name__ == "__main__":
    main()
