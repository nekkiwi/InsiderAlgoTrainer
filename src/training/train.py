import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
from datetime import timedelta
from .utils.train_helpers import *

class ModelTrainer:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.features_file = os.path.join(data_dir, "final/features_final.xlsx")
        self.targets_file = os.path.join(data_dir, "final/targets_final.xlsx")
        self.selected_features_file = os.path.join(data_dir, "analysis/feature_selection/selected_features.xlsx")
        self.output_dir = os.path.join(data_dir, "training/stats")
        self.predictions_dir = os.path.join(data_dir, "training/predictions")
        self.model_save_dir = os.path.join(data_dir, "models")
        self.target_name = ""
        self.model_type = "RandomForest"  # Default model

    def load_data_if_needed(self, features_df, targets_df, selected_features):
        """Load the data if any of the inputs are None."""
        if features_df is None or targets_df is None or selected_features is None:
            return load_data(self.features_file, self.targets_file, self.selected_features_file, self.target_name)
        return features_df, targets_df, selected_features

    def train_model(self, selected_features, features_df=None, targets_df=None):
        """Train the model for each limit/stop combination in parallel."""
        if features_df is None:
            features_df = self.features_df
        if targets_df is None:
            targets_df = self.targets_df

        rows = selected_features.to_dict(orient='records')
        is_continuous = "final" in self.target_name
        model = get_model(self.model_type, is_continuous)
        all_predictions = []

        # Check if the target contains "sell", and adjust the model save directory
        self.model_save_dir = os.path.join(self.model_save_dir, f'{self.model_type}_{self.target_name}')

        # Prepare arguments for the multiprocessing pool
        args_list = [(row, features_df, targets_df, self.target_name, model, self.model_save_dir) for row in rows]

        with Pool(cpu_count()) as pool:
            results = list(tqdm(
                pool.imap(train_model_task_wrapper, args_list),
                total=len(rows),
                desc="- Training Models"
            ))

        filtered_results = []
        for res, pred_data in results:
            if res is not None:
                filtered_results.append(res)
            if pred_data is not None:
                all_predictions.append(pred_data)

        return filtered_results, all_predictions

    def save_training_results(self, results, all_predictions, features_df):
        """Save models, results, and predictions."""
        save_training_data(results, all_predictions, self.output_dir, self.predictions_dir, self.model_type, self.target_name, features_df)

    def run(self, target_name, model_type, selected_features=None, features_df=None, targets_df=None):
        """Run the full training process."""
        start_time = time.time()
        self.target_name = target_name.replace(" ", "-").lower()
        self.model_type = model_type
        print(f"\n### START ### Training for {self.target_name} using {self.model_type}")

        # Load data if needed
        features_df, targets_df, selected_features = self.load_data_if_needed(features_df, targets_df, selected_features)

        # Train the model and collect results and predictions
        results, all_predictions = self.train_model(selected_features, features_df, targets_df)

        # Save models, predictions, and results in a single line
        self.save_training_results(results, all_predictions, features_df)

        elapsed_time = timedelta(seconds=int(time.time() - start_time))
        print(f"### END ### Training - time elapsed: {elapsed_time}")
