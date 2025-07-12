# In src/features/FeaturePreprocessor.py

import os
import pandas as pd
from datetime import timedelta
import time

# Import helper functions (now without normalization)
from .utils.feature_preprocess_helpers import *

class FeaturePreprocessor:
    def __init__(self):
        # Removed normalization_params and normalization_path
        self.data = pd.DataFrame()
        self.continuous_features = None
        self.categorical_features = None
        self.corr_matrix = None
        self.ticker_filing_dates = None

    def prepare_data(self):
        # This method remains the same
        self.data['Filing Date'] = pd.to_datetime(self.data['Filing Date'], dayfirst=True, errors='coerce')
        self.ticker_filing_dates = get_ticker_filing_dates(self.data)
        cols_to_drop = [col for col in ['Ticker', 'Filing Date'] if col in self.data.columns]
        if cols_to_drop:
            self.data.drop(columns=cols_to_drop, inplace=True)
        else:
            print("- Warning: 'Ticker' or 'Filing Date' not found for dropping.")

    def save_feature_data(self, file_path, train):
        # This method remains the same
        features_cleaned = save_feature_data(self.data, self.ticker_filing_dates, file_path, train)
        return features_cleaned

    def identify_feature_types(self):
        # This method remains the same
        self.categorical_features, self.continuous_features = identify_feature_types(self.data)

    def filter_low_variance_features(self, variance_threshold=0.02, categorical_threshold=0.02):
        # This method remains the same
        cont_df = self.data[self.continuous_features]
        cat_df  = self.data[self.categorical_features]
        self.data, cont_df_filtered, cat_df_filtered = filter_low_variance_features(
            self.data, cont_df, cat_df, variance_threshold, categorical_threshold
        )
        self.continuous_features = cont_df_filtered.columns.tolist()
        self.categorical_features = cat_df_filtered.columns.tolist()

    def calculate_and_plot_correlations(self, output_dir):
        # This method remains the same
        cont_df = self.data[self.continuous_features]
        cat_df  = self.data[self.categorical_features]
        self.corr_matrix = hybrid_correlation_matrix(self.data, cont_df, cat_df)
        plot_correlation_heatmap(self.corr_matrix, output_dir)
        plot_sorted_correlations(self.corr_matrix, output_dir)

    def drop_highly_correlated_features(self, threshold=0.9):
        # This method remains the same
        self.data, self.corr_matrix = drop_highly_correlated_features(
            self.data, self.corr_matrix, threshold
        )

    def run(self, features_df, train):
        start_time = time.time()
        print("\n### START ### Feature Preprocessing")

        if features_df is None and train:
            self.data = load_feature_data(f'interim/train/3_features_TI_FR.xlsx')
        else:
            self.data = features_df
            
        self.data = engineer_new_features(self.data)
        self.prepare_data()
        self.identify_feature_types()

        # The 'run' method is now simplified for both train and inference,
        # as normalization is no longer handled here.
        if train:
            self.filter_low_variance_features()
            plot_output_dir = os.path.join(os.path.dirname(__file__), '../../data', 'analysis/feature_preprocess')
            os.makedirs(plot_output_dir, exist_ok=True)
            self.calculate_and_plot_correlations(plot_output_dir)
            self.drop_highly_correlated_features(threshold=0.9)
            features_cleaned = self.save_feature_data('interim/train/5_features_full_cleaned.xlsx', train)
        else:
            # For inference, we simply use the data as is, since normalization
            # will be handled by the loaded training pipeline.
            features_cleaned = self.save_feature_data('', train)

        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"### END ### Feature Preprocess - time elapsed: {elapsed_time}")

        return features_cleaned
