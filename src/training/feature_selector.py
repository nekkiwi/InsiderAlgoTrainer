# In src/analysis/feature_selector.py

import os
import time
from datetime import timedelta
import pandas as pd
from tqdm import tqdm

# --- Updated imports to use new helper functions ---
from .utils.feature_selector_helpers import (
    select_features_with_model, 
    save_feature_selection_results,
    create_feature_heatmap
)

class FeatureSelector:
    """
    Automates model-based feature selection for both return and alpha targets,
    saving results into separated, sorted sheets and visualizations.
    """
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.features_file = os.path.join(data_dir, "final/features_final.xlsx")
        self.targets_file = os.path.join(data_dir, "final/targets_final.xlsx")
        self.output_dir = os.path.join(data_dir, "analysis/feature_selection")
        
        self.features_df = None
        self.targets_df_dict = None

    def load_data(self):
        """Loads the engineered features and all raw target sheets."""
        print("- Loading features and raw targets...")
        self.features_df = pd.read_excel(self.features_file)
        self.targets_df_dict = pd.read_excel(self.targets_file, sheet_name=None)

    def prepare_data(self):
        """Ensures 'Filing Date' is a consistent datetime format across all data."""
        print("- Preparing data by standardizing 'Filing Date' format...")
        # Assumes dates are stored in a standard format that pandas can parse.
        self.features_df['Filing Date'] = pd.to_datetime(self.features_df['Filing Date'], dayfirst=True)
        for sheet_name in self.targets_df_dict:
            self.targets_df_dict[sheet_name]['Filing Date'] = pd.to_datetime(
                self.targets_df_dict[sheet_name]['Filing Date'], dayfirst=True
            )

    def run(self, category, timepoints: list, thresholds: list, top_n=20):
        """
        Executes a feature selection workflow for both return and alpha targets.

        Args:
            timepoints (list): A list of time horizons (e.g., ['1w', '1m']).
            thresholds (list): A list of performance thresholds in percent (e.g., [0, 5]).
            top_n (int): The number of top features to select.
        """
        start_time = time.time()
        print("\n### START ### Dynamic Threshold Feature Selector")
        
        self.load_data()
        self.prepare_data()
        
        all_feature_scores = {}

        # Loop through both categories, timepoints, and thresholds
        
        for tp in tqdm(timepoints, desc=f"Processing {category.title()} Timepoints"):
            for thresh_pct in thresholds:
                raw_target_name = f"{category}_{tp}_raw"
                strategy_name = f"{category}_{tp}_binary_{thresh_pct}pct"
                
                _, feature_scores = select_features_with_model(
                    features_df=self.features_df,
                    targets_df_dict=self.targets_df_dict,
                    raw_target_name=raw_target_name,
                    threshold_pct=thresh_pct,
                    top_n=top_n
                )
                
                if feature_scores is not None and not feature_scores.empty:
                    all_feature_scores[strategy_name] = feature_scores

        if all_feature_scores:
            # --- Save results to separate sheets in one Excel file ---
            save_feature_selection_results(all_feature_scores, self.output_dir)

            # --- Create separate heatmaps for return and alpha ---
            if category == "return": 
                return_scores = {k: v for k, v in all_feature_scores.items() if k.startswith('return')}
                create_feature_heatmap(return_scores, 'return', self.output_dir)
            if category == "alpha": 
                alpha_scores = {k: v for k, v in all_feature_scores.items() if k.startswith('alpha')}
                create_feature_heatmap(alpha_scores, 'alpha', self.output_dir)

        else:
            print("[WARN] No features with non-zero importance were selected.")

        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"### END ### Feature Selector - time elapsed: {elapsed_time}")
        
        return all_feature_scores
