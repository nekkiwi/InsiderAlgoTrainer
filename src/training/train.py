import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from .utils.train_helpers import *

class ModelTrainer:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.features_file = os.path.join(data_dir, "final/features_final.xlsx")
        self.targets_file = os.path.join(data_dir, "final/targets_final.xlsx")
        self.selected_features_file = os.path.join(data_dir, "output/feature_selection/selected_features.xlsx")
        self.output_dir = os.path.join(data_dir, "training/stats")
        self.predictions_dir = os.path.join(data_dir, "training/predictions")
        self.model_save_dir = os.path.join(data_dir, "models")
        self.target_name = ""
        self.model_type = "RandomForest"  # Default model

    def train_model(self):
        """Train the model for each limit/stop combination in parallel."""
        rows = self.selected_features_df.to_dict(orient='records')
        model = get_model(self.model_type)
        all_predictions = []

        with Pool(cpu_count()) as pool:
            results = list(tqdm(pool.imap(lambda row: train_model_task(row, self.features_df, self.targets_df, self.target_name, model), rows), total=len(rows), desc="Training Models"))

        filtered_results = []
        for res, pred_data in results:
            if res is not None:
                filtered_results.append(res)
            if pred_data is not None:
                all_predictions.append(pred_data)

        if filtered_results:
            save_results(filtered_results, self.output_dir, self.model_type, self.target_name)
        
        save_predictions(all_predictions, self.predictions_dir, self.model_type.replace(" ", "-").lower())
        
        # Save model
        save_model(model, self.model_type, self.target_name, "trained", self.model_save_dir)

    def run(self, target_name, model_type="RandomForest"):
        """Run the full training process."""
        self.binary_target = target_name.replace(" ", "-").lower() not in ['return-at-cashout', 'days-at-cashout']
        self.model_type = model_type
        self.features_df, self.targets_df, self.selected_features_df = load_data(self.features_file, self.targets_file, self.selected_features_file, target_name)
        self.target_name = target_name.replace(" ", "-").lower()
        self.train_model()

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run(target_name="Limit occurred first", model_type="RandomForest")
