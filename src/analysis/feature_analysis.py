import os
import pandas as pd
import sys

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.feature_analysis_helpers import *

class FeatureAnalyzer:
    def __init__(self):
        self.data = pd.DataFrame()
        self.continuous_features = None
        self.categorical_features = None
        self.corr_matrix = None
        self.ticker_filing_dates = None

    def load_feature_data(self, file_path):
        self.data, self.ticker_filing_dates = load_feature_data(file_path)
        if self.data is not None and self.ticker_filing_dates is not None:
            # Ensure Ticker and Filing Date columns are removed from the main data
            self.data.drop(columns=['Ticker', 'Filing Date'], inplace=True)
        else:
            raise ValueError("Failed to load feature data. Please check the file and its contents.")

    def save_feature_data(self, file_path):
        save_feature_data(self.data, self.ticker_filing_dates, file_path)

    def identify_feature_types(self):
        self.categorical_features, self.continuous_features = identify_feature_types(self.data)

    def filter_low_variance_features(self):
        self.data = filter_low_variance_features(self.data, self.continuous_features, self.categorical_features)

    def clip_and_normalize_features(self):
        self.data = clip_continuous_features(self.data, self.continuous_features)
        self.data = normalize_continuous_features(self.data, self.continuous_features)

    def calculate_and_plot_correlations(self, output_dir):
        self.corr_matrix = calculate_correlations(self.data, self.continuous_features)
        plot_correlation_heatmap(self.corr_matrix, output_dir)
        plot_sorted_correlations(self.corr_matrix, output_dir)

    def drop_highly_correlated_features(self, threshold=0.9):
        self.data, self.corr_matrix = drop_highly_correlated_features(
            self.data, self.corr_matrix, threshold
        )

    def run_analysis(self):
        """Run the full analysis pipeline."""
        self.load_feature_data('interim/4_features_TI_FR_IT.xlsx')
        self.identify_feature_types()
        self.filter_low_variance_features()
        self.clip_and_normalize_features()
        
        output_dir = os.path.join(os.path.dirname(__file__), '../../data/output/feature_analysis')
        os.makedirs(output_dir, exist_ok=True)

        self.calculate_and_plot_correlations(output_dir)
        self.drop_highly_correlated_features()
        self.save_feature_data('final/features_final.xlsx')

if __name__ == "__main__":
    analyzer = FeatureAnalyzer()
    analyzer.run_analysis()
