import os
import pandas as pd
from datetime import timedelta
import time

from .utils.feature_preprocess_helpers import *

class FeaturePreprocessor:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.data = pd.DataFrame()
        self.continuous_features = None
        self.categorical_features = None
        self.corr_matrix = None
        self.ticker_filing_dates = None
        self.normalization_params = {}
        self.normalization_path = os.path.join(data_dir,'analysis/feature_preprocess/normalization_params.xlsx')

    def prepare_data(self):
        # Ensure 'Filing Date' is datetime before extracting ticker_filing_dates
        self.data['Filing Date'] = pd.to_datetime(self.data['Filing Date'], dayfirst=True, errors='coerce')
        self.ticker_filing_dates = get_ticker_filing_dates(self.data)
        
        # Drop 'Ticker' and 'Filing Date' from the main data DataFrame as they are handled separately
        # Check if columns exist before dropping to avoid KeyError
        cols_to_drop = [col for col in ['Ticker', 'Filing Date'] if col in self.data.columns]
        if cols_to_drop:
            self.data.drop(columns=cols_to_drop, inplace=True)
        else:
            # This else block might indicate an issue if these columns are truly expected but missing
            # Consider more specific error handling or a warning if this state is unexpected.
            print("- Warning: 'Ticker' or 'Filing Date' not found in DataFrame for dropping.")


    def save_feature_data(self, file_path, train):
        features_cleaned = save_feature_data(self.data, self.ticker_filing_dates, file_path, train)
        return features_cleaned

    def identify_feature_types(self):
        # This function identifies features based on the current state of self.data
        self.categorical_features, self.continuous_features = identify_feature_types(self.data)

    def filter_low_variance_features(self, variance_threshold=0.02, categorical_threshold=0.02):
        # 1) slice out the current continuous & categorical DataFrames
        cont_df = self.data[self.continuous_features]
        cat_df  = self.data[self.categorical_features]

        # 2) run the helper on DataFrames, not lists
        self.data, cont_df_filtered, cat_df_filtered = filter_low_variance_features( # Renamed to avoid confusion
            self.data,
            cont_df,
            cat_df,
            variance_threshold=variance_threshold,
            categorical_threshold=categorical_threshold
        )

        # 3) update the feature‐name lists to match the filtered DataFrames
        self.continuous_features = cont_df_filtered.columns.tolist() # Use filtered DataFrame columns
        self.categorical_features = cat_df_filtered.columns.tolist() # Use filtered DataFrame columns

    def clip_and_normalize_features(self, lower=0.01, upper=0.99):
        # Clip features
        self.data = clip_continuous_features(self.data, self.continuous_features, lower, upper)

        # Normalize and save parameters
        # Ensure only current continuous features are passed for normalization
        current_continuous_data = self.data[self.continuous_features]
        self.data, self.normalization_params = normalize_continuous_features(self.data, current_continuous_data.columns.tolist())

        # Save normalization parameters for future inference
        save_normalization_params(self.normalization_params, self.normalization_path)
        
    def normalize_before_inference(self):
        # Read normalization parameters from Excel
        norm_params = pd.read_excel(self.normalization_path)

        # Ensure proper column naming
        norm_params.columns = ['key', 'min', 'max']
        
        # Clean numerical formatting (handle commas) and convert to numeric
        # Use .loc to avoid SettingWithCopyWarning
        norm_params.loc[:, 'min'] = norm_params['min'].astype(str).str.replace(",", "").astype(float)
        norm_params.loc[:, 'max'] = norm_params['max'].astype(str).str.replace(",", "").astype(float)

        # Create a dictionary for fast lookup
        normalization_dict = norm_params.set_index('key')[['min', 'max']].to_dict('index')

        # Normalize each column in df based on the params
        for col in self.data.columns:
            if col in normalization_dict:
                min_val = normalization_dict[col]['min']
                max_val = normalization_dict[col]['max']
                # Check for zero range to prevent division by zero
                if (max_val - min_val) != 0:
                    self.data[col] = (self.data[col] - min_val) / (max_val - min_val)
                else:
                    # Handle cases where min_val == max_val (feature has no variance)
                    self.data[col] = 0.0 # Or some other appropriate value
                

    def calculate_and_plot_correlations(self, output_dir):
        """Calculate a hybrid correlation matrix and plot the heatmap."""
        # Ensure these are slices of the current self.data after feature engineering
        cont_df = self.data[self.continuous_features]
        cat_df  = self.data[self.categorical_features]
        self.corr_matrix = hybrid_correlation_matrix(self.data, cont_df, cat_df)
        plot_correlation_heatmap(self.corr_matrix, output_dir)
        plot_sorted_correlations(self.corr_matrix, output_dir)

    def drop_highly_correlated_features(self, threshold=0.9):
        # This function updates self.data and self.corr_matrix in place
        self.data, self.corr_matrix = drop_highly_correlated_features(
            self.data, self.corr_matrix, threshold
        )

    def run(self, features_df, train):
        start_time = time.time()
        print("\n### START ### Feature Preprocessing")

        """Run the full analysis pipeline."""
        if features_df is None and train:
            # Load initial data if not provided and in training mode
            self.data = load_feature_data(f'interim/train/4_features_TI_FR_IT.xlsx')
        else:
            # Use the provided features_df
            self.data = features_df
            
        # --- NEW: Feature Engineering Step ---
        self.data = engineer_new_features(self.data)
        
        self.prepare_data()
        self.identify_feature_types() # Re-identify types after engineering

        if train:
            self.filter_low_variance_features(variance_threshold=0.02, categorical_threshold=0.02)
            self.clip_and_normalize_features(lower=0.01, upper=0.99)
            # Ensure output_dir for plots exists
            plot_output_dir = os.path.join(os.path.dirname(__file__), '../../data', 'analysis/feature_preprocess')
            os.makedirs(plot_output_dir, exist_ok=True)
            self.calculate_and_plot_correlations(plot_output_dir) # Pass the full path
            self.drop_highly_correlated_features(threshold=0.9)
            features_cleaned = self.save_feature_data('interim/train/5_features_full_cleaned.xlsx', train)
        else:
            self.normalize_before_inference()
            features_cleaned = self.save_feature_data('', train)

        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"### END ### Feature Preprocess - time elapsed: {elapsed_time}")

        return features_cleaned