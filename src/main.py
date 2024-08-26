import sys
import os
from scraper.target_scraper import TargetScraper
from scraper.feature_scraper import FeatureScraper
from scraper.stock_scraper import StockDataScraper
from analysis.feature_selector import FeatureSelector
from analysis.feature_analysis import FeatureAnalyzer
from analysis.stock_analysis import StockAnalysis
from training.train import ModelTrainer

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
    
def train_model(target_name, model_type):
    # Initialize and run the Feature Selector
    print("Starting Feature Selector...")
    selector = FeatureSelector()
    selector.run()
    print("Feature Selector completed.\n")

    # Ensure the feature selector has completed before moving on
    sys.stdout.flush()

    # Initialize and run the Model Trainer
    print("Starting Model Trainer...")
    trainer = ModelTrainer()
    trainer.run(target_name, model_type)
    print("Model Trainer completed.\n")
    
    # Ensure the scraper has completed before moving on
    sys.stdout.flush()

def main():    
    num_months=36
    
    limit_array=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    stop_array=[-0.1, -0.09, -0.08, -0.07, -0.06, -0.05, -0.04]
    
    target_name="Limit occurred first"  # [Spike up,        Spike down,     Limit occurred first,   Stop occurred first,    Return at cashout,      Days at cashout]
    model_type="Neural Net"             # [RandomForest,    NaivesBayes,    RBF SVM,                Gaussian Process,       Neural Net]
    
    scrape_data(num_months)
    run_analysis()
    scrape_targets(limit_array, stop_array)
    train_model(target_name, model_type)

if __name__ == "__main__":
    main()
