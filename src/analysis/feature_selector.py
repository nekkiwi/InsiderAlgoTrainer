import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

import sys
import os 

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.feature_selection_helpers import (
    calculate_anova,
    calculate_chi_square,
    calculate_pearson,
    calculate_spearman,
    rfecv_selection,
    plot_selected_features,
    compare_feature_lists
)

class FeatureSelector:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.features_file = os.path.join(data_dir, 'final/features_final.xlsx')
        self.targets_file = os.path.join(data_dir, 'final/targets_final.xlsx')
        self.output_dir = os.path.join(data_dir, 'output/feature_selection')
        self.features_df = None
        self.targets_df = None

    def load_data(self):
        """Load the features and targets data."""
        self.features_df = pd.read_excel(self.features_file)
        self.targets_df = pd.read_excel(self.targets_file)

    def select_features(self, target_column, selection_method):
        """Apply the specified feature selection method to the target column."""
        target = self.targets_df[target_column]
        if selection_method == 'ANOVA':
            selector = calculate_anova(self.features_df, target)
        elif selection_method == 'Chi-Square':
            selector = calculate_chi_square(self.features_df, target)
        elif selection_method == 'Pearson':
            selector = calculate_pearson(self.features_df, target)
        elif selection_method == 'Spearman':
            selector = calculate_spearman(self.features_df, target)
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
        return selector

    def apply_rfecv(self, target_column):
        """Apply RFECV to select features for a given target."""
        target = self.targets_df[target_column]
        estimator = RandomForestClassifier(random_state=42)
        selector = rfecv_selection(estimator, self.features_df, target)
        return selector

    def run_selection(self):
        """Run feature selection for each target and each selection method."""
        os.makedirs(self.output_dir, exist_ok=True)
        selection_methods = ['ANOVA', 'Chi-Square', 'Pearson', 'Spearman']
        results = {}

        for target_column in self.targets_df.columns:
            for method in selection_methods:
                selector = self.select_features(target_column, method)
                feature_importances = selector.scores_ if hasattr(selector, 'scores_') else selector.ranking_
                plot_selected_features(
                    feature_importances,
                    self.features_df.columns,
                    os.path.join(self.output_dir, f'{target_column}_{method}_selection.png'),
                    title=f'{target_column} - {method} Feature Selection'
                )
                selected_features = self.features_df.columns[selector.get_support()]
                results[(target_column, method)] = selected_features

            # Apply RFECV
            rfecv_selector = self.apply_rfecv(target_column)
            plot_selected_features(
                rfecv_selector.ranking_,
                self.features_df.columns,
                os.path.join(self.output_dir, f'{target_column}_RFECV_selection.png'),
                title=f'{target_column} - RFECV Feature Selection'
            )
            selected_features_rfecv = self.features_df.columns[rfecv_selector.support_]
            results[(target_column, 'RFECV')] = selected_features_rfecv

        # Compare feature lists across methods
        compare_feature_lists(
            selection_results=results,
            method_names=selection_methods + ['RFECV'],
            output_dir=os.path.join(self.output_dir, 'feature_selection_comparison.xlsx')
        )

    def run(self):
        """Run the full feature selection process."""
        self.load_data()
        self.run_selection()
        print("Feature selection completed. Check the output directory for results.")

if __name__ == "__main__":
    selector = FeatureSelector()
    selector.run()
