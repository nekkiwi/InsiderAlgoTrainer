import os
import pandas as pd
from datetime import timedelta
import time

from .utils.feature_analysis_helpers import *

class FeatureAnalyzer:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')

        self.data = pd.DataFrame()
        self.continuous_features = None
        self.categorical_features = None
        self.corr_matrix = None
        self.ticker_filing_dates = None
        self.normalization_params = {}
        self.normalization_path = os.path.join(data_dir,'analysis/feature_analysis/normalization_params.xlsx')

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
        # 1) slice out the current continuous & categorical DataFrames
        cont_df = self.data[self.continuous_features]
        cat_df  = self.data[self.categorical_features]

        # 2) run the helper on DataFrames, not lists
        self.data, cont_df, cat_df = filter_low_variance_features(
            self.data,
            cont_df,
            cat_df,
            variance_threshold=variance_threshold,
            categorical_threshold=categorical_threshold
        )

        # 3) update the feature‐name lists to match the filtered DataFrames
        self.continuous_features = cont_df.columns.tolist()
        self.categorical_features = cat_df.columns.tolist()

    def clip_and_normalize_features(self, lower=0.01, upper=0.99):
        # Clip features
        self.data = clip_continuous_features(self.data, self.continuous_features, lower, upper)

        # Normalize and save parameters
        self.data, self.normalization_params = normalize_continuous_features(self.data, self.continuous_features)

        # Save normalization parameters for future inference
        save_normalization_params(self.normalization_params, self.normalization_path)
        
    def normalize_before_inference(self):
        # Read normalization parameters from Excel
        norm_params = pd.read_excel(self.normalization_path)

        # Ensure proper column naming
        norm_params.columns = ['key', 'min', 'max']
        
        # Clean numerical formatting (handle commas)
        norm_params['min'] = norm_params['min'].replace(",", "")
        norm_params['max'] = norm_params['max'].replace(",", "")

        # Create a dictionary for fast lookup
        normalization_dict = norm_params.set_index('key')[['min', 'max']].to_dict('index')

        # Normalize each column in df based on the params
        for col in self.data.columns:
            if col in normalization_dict:
                min_val = normalization_dict[col]['min']
                max_val = normalization_dict[col]['max']
                # Min-max normalization
                self.data[col] = (self.data[col] - min_val) / (max_val - min_val)

    def calculate_and_plot_correlations(self, output_dir):
        """Calculate a hybrid correlation matrix and plot the heatmap."""
        cont_df = self.data[self.continuous_features]
        cat_df  = self.data[self.categorical_features]
        self.corr_matrix = hybrid_correlation_matrix(self.data, cont_df, cat_df)
        plot_correlation_heatmap(self.corr_matrix, output_dir)
        plot_sorted_correlations(self.corr_matrix, output_dir)

    def drop_highly_correlated_features(self, threshold=0.9):
        self.data, self.corr_matrix = drop_highly_correlated_features(
            self.data, self.corr_matrix, threshold
        )

    def run(self, features_df, train):
        start_time = time.time()
        print("\n### START ### Feature Analysis")

        """Run the full analysis pipeline."""
        if train:
            stage = "train"
        else:
            stage = "infer"
            
        if features_df is None:
            self.data = load_feature_data(f'interim/{stage}/4_features_TI_FR_IT.xlsx')
        else:
            self.data = features_df
            
        self.prepare_data()
        self.identify_feature_types()
        
        if train:
            self.filter_low_variance_features(variance_threshold=0.02, categorical_threshold=0.02)
            self.clip_and_normalize_features(lower=0.01, upper=0.99)
            self.calculate_and_plot_correlations('analysis/feature_analysis')
            self.drop_highly_correlated_features(threshold=0.9)
            features_cleaned = self.save_feature_data('interim/train/5_features_full_cleaned.xlsx')
        else:
            self.normalize_before_inference()
            features_cleaned = self.save_feature_data('final/current_features_final.xlsx')

        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"### END ### Feature Analysis - time elapsed: {elapsed_time}")

        return features_cleaned

if __name__ == "__main__":
    analyzer = FeatureAnalyzer()
    analyzer.run()
