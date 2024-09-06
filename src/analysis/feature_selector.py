import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from .utils.feature_selector_helpers import select_features

class FeatureSelector:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.features_file = os.path.join(data_dir, "final/features_final.xlsx")
        self.targets_file = os.path.join(data_dir, "final/targets_final.xlsx")
        self.output_dir = os.path.join(data_dir, "output/feature_selection")
        self.features_df = None
        self.targets_df = None
        self.p_threshold = 0
        self.selected_features_dict = []

    def load_data(self):
        """Load the features and targets data."""
        self.features_df = pd.read_excel(self.features_file)
        self.features_df.drop(columns=['Ticker', 'Filing Date'], inplace=True)
        self.targets_df = pd.read_excel(self.targets_file, sheet_name=None)  # Load all sheets as a dictionary

    def feature_selection_task(self, args):
        """Perform feature selection for a specific target."""
        limit_value, stop_value, target_value, y, X_encoded = args
        selected_features, scores, p_values = select_features(X_encoded, y, self.p_threshold)
        
        sorted_features = sorted(zip(selected_features, scores[selected_features]), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features]

        result = {
            "Limit": limit_value,
            "Stop": stop_value,
            "Target": target_value,
            "Selected Features": selected_features,
            "Scores": scores[selected_features].tolist(),
            "p_values": p_values[selected_features].tolist()
        }
        
        return result

    def perform_feature_selection(self, p_threshold=0.1):
        """Perform feature selection for each target and save results."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.p_threshold = p_threshold
        X_encoded = self.features_df.copy()
        
        tasks = []
        
        for sheet_name, sheet_data in self.targets_df.items():
            limit_stop = sheet_name.split(' ')
            if len(limit_stop) == 4:  # Expecting "lim <value> stop <value>"
                limit_value = limit_stop[1]
                stop_value = limit_stop[3]

                for target_name in sheet_data.columns[2:]:
                    y = sheet_data[target_name]
                    tasks.append((limit_value, stop_value, target_name, y, X_encoded))
        
        # Use ProcessPoolExecutor to parallelize the feature selection process
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.feature_selection_task, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Feature Selection Progress"):
                self.selected_features_dict.append(future.result())

        # Sort the results by Stop and then by Limit
        self.selected_features_dict.sort(key=lambda x: (x['Stop'], x['Limit']))
        
        self.save_selected_features()
        
        # self.create_heatmap('Scores', "score_heatmap.png")
        # self.create_heatmap('p_values', "p_value_heatmap.png")

    def save_selected_features(self):
        """Save selected features to an Excel file."""
        excel_path = os.path.join(self.output_dir, "selected_features.xlsx")
        with pd.ExcelWriter(excel_path) as writer:
            for target_name, selections in pd.DataFrame(self.selected_features_dict).groupby('Target'):
                sheet_data = selections.drop(columns=['Scores', 'p_values', 'Target'], errors='ignore')
                sheet_data.to_excel(writer, sheet_name=target_name.replace(" ", "-").lower(), index=False)
        print(f"Selected features saved to {excel_path}")

    def create_heatmap(self, key, filename):
        """Create and save a heatmap of either scores or p-values."""
        data = []

        for selection in self.selected_features_dict:
            values = selection[key]
            if isinstance(values, float):
                values = [values]

            series = pd.Series(values, index=selection['Selected Features'], name=f'{selection["Limit"]}_{selection["Stop"]}_{selection["Target"]}')
            data.append(series)

        if data:
            df = pd.concat(data, axis=1).T

            # Calculate figure height dynamically based on the number of rows (lim/stop combinations)
            n_combinations = len(df)
            cell_height = 0.5  # Height per row
            height = n_combinations * cell_height

            plt.figure(figsize=(60, height))
            if key == 'Scores':
                sns.heatmap(df, annot=True, fmt=".1f", cmap="coolwarm", cbar=True, linewidths=0.5, vmin=df.quantile(0.05).min(), vmax=df.quantile(0.95).max())
            elif key == 'p_values':
                sns.heatmap(df, annot=True, fmt=".3f", cmap="coolwarm", cbar=True, linewidths=0.5, vmin=0, vmax=self.p_threshold)
            plt.xticks(rotation=90, ha='center')
            plt.yticks(rotation=0)
            heatmap_path = os.path.join(self.output_dir, filename)
            plt.savefig(heatmap_path, dpi=300)
            plt.close()
            print(f"Heatmap saved to {heatmap_path}")
        else:
            print(f"No valid data found for {key}. Skipping heatmap creation.")

    def run(self):
        """Run the full feature selection process."""
        self.load_data()
        self.perform_feature_selection(p_threshold=0.05)

# Example Usage
if __name__ == "__main__":
    selector = FeatureSelector()
    selector.run()
