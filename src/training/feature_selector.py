import os
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from datetime import timedelta
from .utils.feature_selector_helpers import select_features, save_selected_features, create_heatmap


class FeatureSelector:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.features_file = os.path.join(data_dir, "final/features_final.xlsx")
        self.targets_file = os.path.join(data_dir, "final/targets_final.xlsx")
        self.output_dir = os.path.join(data_dir, "output/feature_selection")
        self.features_df = None
        self.targets_df = None
        self.p_threshold = 0.05
        self.selected_features_dict = {}

    def load_data(self):
        """Load the features and targets data from the files."""
        self.features_df = pd.read_excel(self.features_file)
        self.targets_df = pd.read_excel(self.targets_file, sheet_name=None)  # Load all sheets as a dictionary

        # Convert Filing Date to datetime format in both features and targets
        self.features_df['Filing Date'] = pd.to_datetime(self.features_df['Filing Date'], format='%d/%m/%Y %H:%M', errors='coerce')
        for sheet_name, sheet_data in self.targets_df.items():
            self.targets_df[sheet_name]['Filing Date'] = pd.to_datetime(self.targets_df[sheet_name]['Filing Date'], format='%d/%m/%Y %H:%M', errors='coerce')

    def feature_selection_task(self, args):
        """Perform feature selection for a specific target."""
        limit_value, stop_value, target_value, y, X_encoded = args
        selected_features, scores, p_values = select_features(X_encoded, y, self.p_threshold)

        # Sort and rank features by their scores
        sorted_features = sorted(zip(selected_features, scores[selected_features]), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features]

        result = {
            "Limit": limit_value,
            "Stop": stop_value,
            "Selected Features": selected_features,
            "p_values": p_values[selected_features].tolist()
        }
        return result

    def perform_feature_selection(self, p_threshold=0.05, features_df=None, targets_df=None):
        """Perform feature selection for each target and collect results."""
        os.makedirs(self.output_dir, exist_ok=True)

        self.p_threshold = p_threshold
        if features_df is None:
            features_df = self.features_df
        if targets_df is None:
            targets_df = self.targets_df

        # Drop 'Ticker' and 'Filing Date' from the features DataFrame
        X_encoded = features_df.drop(columns=['Ticker', 'Filing Date'], errors='ignore')

        for sheet_name, sheet_data in tqdm(targets_df.items(),desc="- Selecting features for targets"):
            tasks = self.create_tasks(sheet_data, X_encoded)

            # Parallelize the feature selection
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(self.feature_selection_task, task) for task in tasks]
                for future in as_completed(futures):
                    if sheet_name not in self.selected_features_dict:
                        self.selected_features_dict[sheet_name] = []
                    self.selected_features_dict[sheet_name].append(future.result())

    def create_tasks(self, target_data, X_encoded):
        """Create tasks for each target column."""
        # Drop 'Ticker' and 'Filing Date' from the target data as well
        target_data = target_data.drop(columns=['Ticker', 'Filing Date'], errors='ignore')

        tasks = []
        for column in target_data.columns:
            try:
                limit_stop = column.split(',')
                limit_value = limit_stop[0].replace('Limit ', '')
                stop_value = limit_stop[1].replace('Stop ', '')
            except IndexError:
                limit_value, stop_value = 'all', 'all'
            y = target_data[column]
            tasks.append((limit_value, stop_value, column, y, X_encoded))
        return tasks

    def run(self, features_df=None, targets_df=None, p_threshold=0.05):
        """Run the feature selection process."""
        if features_df is None or targets_df is None:
            self.load_data()
        
        start_time = time.time()
        print("\n### START ### Feature Selector")

        # Perform feature selection and collect results
        self.perform_feature_selection(p_threshold, features_df, targets_df)

        # Save the results for all targets into one Excel file, each in a separate sheet
        save_selected_features(self.selected_features_dict, self.output_dir)

        # Optionally create heatmaps
        print("- Creating heatmap for p-values")
        create_heatmap(self.selected_features_dict, 'p_values', self.output_dir, self.p_threshold)

        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"### END ### Target Scraper - time elapsed: {elapsed_time}")
        
        return self.selected_features_dict


# Example Usage
if __name__ == "__main__":
    selector = FeatureSelector()
    selector.run()
