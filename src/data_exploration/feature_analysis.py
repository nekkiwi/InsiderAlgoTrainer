import os
import pandas as pd
import sys
from datetime import timedelta
import time

from .utils.feature_analysis_helpers import *

class FeatureAnalyzer:
    def __init__(self):
        self.data = pd.DataFrame()
        self.continuous_features = None
        self.categorical_features = None
        self.corr_matrix = None
        self.ticker_filing_dates = None
        self.normalization_params = {}

    def prepare_data(self):
        self.data['Filing Date'] = pd.to_datetime(self.data['Filing Date'], dayfirst=True, errors='coerce')
        self.ticker_filing_dates = get_ticker_filing_dates(self.data)
        if self.data is not None and self.ticker_filing_dates is not None:
            self.data.drop(columns=['Ticker', 'Filing Date'], inplace=True)
        else:
            raise ValueError("- Failed to load feature data. Please check the file and its contents.")

    def save_feature_data(self, file_path):
        features_cleaned = save_feature_data(self.data, self.ticker_filing_dates, file_path)
        return features_cleaned

    def identify_feature_types(self):
        self.categorical_features, self.continuous_features = identify_feature_types(self.data)

    def filter_low_variance_features(self, variance_threshold=0.02, categorical_threshold=0.02):
        self.data, self.continuous_features, self.categorical_features = filter_low_variance_features(
            self.data, self.continuous_features, self.categorical_features, variance_threshold=0.02, categorical_threshold=0.02)

    def clip_and_normalize_features(self, out_path, lower=0.01, upper=0.99):
        # Clip features
        self.data = clip_continuous_features(self.data, self.continuous_features, lower=0.01, upper=0.99)

        # Normalize and save parameters
        self.data, self.normalization_params = normalize_continuous_features(self.data, self.continuous_features)

        # Save normalization parameters for future inference
        save_normalization_params(self.normalization_params, out_path)

    def calculate_and_plot_correlations(self, output_dir):
        """Calculate a hybrid correlation matrix and plot the heatmap."""
        self.corr_matrix = hybrid_correlation_matrix(self.data, self.continuous_features, self.categorical_features)
        plot_correlation_heatmap(self.corr_matrix, output_dir)
        plot_sorted_correlations(self.corr_matrix, output_dir)

    def drop_highly_correlated_features(self, threshold=0.9):
        self.data, self.corr_matrix = drop_highly_correlated_features(
            self.data, self.corr_matrix, threshold
        )

    def run(self, features_df):
        start_time = time.time()
        print("\n### START ### Feature Analysis")

        """Run the full analysis pipeline."""
        if features_df is None:
            self.data = load_feature_data('interim/4_features_TI_FR_IT.xlsx')
        else:
            self.data = features_df
            
        self.prepare_data()
        self.identify_feature_types()
        self.filter_low_variance_features(variance_threshold=0.02, categorical_threshold=0.02)
        self.clip_and_normalize_features('output/feature_analysis/normalization_params.xlsx', lower=0.01, upper=0.99)

        self.calculate_and_plot_correlations('output/feature_analysis')
        self.drop_highly_correlated_features(threshold=0.9)
        features_cleaned = self.save_feature_data('interim/5_features_full_cleaned.xlsx')

        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"### END ### Feature Analysis - time elapsed: {elapsed_time}")

        return features_cleaned

if __name__ == "__main__":
    analyzer = FeatureAnalyzer()
    analyzer.run()
