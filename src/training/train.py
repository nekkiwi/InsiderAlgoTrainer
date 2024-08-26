import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
import sys
from multiprocessing import Process, Queue

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class ModelTrainer:
    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        self.features_file = os.path.join(data_dir, "final/features_final.xlsx")
        self.targets_file = os.path.join(data_dir, "final/targets_final.xlsx")
        self.selected_features_file = os.path.join(data_dir, "output/feature_selection/selected_features.xlsx")
        self.output_dir = os.path.join(data_dir, "training/raw")
        self.target_name = ""
        self.model_type = "RandomForest"  # Default model

    def load_data(self, target_name):
        """Load the data from the Excel files."""
        self.target_name = target_name
        self.features_df = pd.read_excel(self.features_file)
        self.targets_df = pd.read_excel(self.targets_file, sheet_name=None)
        self.selected_features_df = pd.read_excel(self.selected_features_file, sheet_name=self.target_name)

    def get_model(self):
        """Return the model instance based on the specified model type."""
        if self.model_type == "RandomForest":
            return RandomForestClassifier(random_state=42)
        elif self.model_type == "NaivesBayes":
            return GaussianNB()
        elif self.model_type == "RBF SVM":
            return SVC(kernel='rbf', probability=True, random_state=42)
        elif self.model_type == "Gaussian Process":
            return GaussianProcessClassifier(random_state=42)
        elif self.model_type == "Neural Net":
            return MLPClassifier(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train_model_task(self, limit, stop, selected_features_str, q):
        """Train the model for a specific limit/stop combination and send the results via a queue."""
        selected_features = eval(selected_features_str)
        selected_features = [feature.strip("'") for feature in selected_features]

        # Select the features and target
        X = self.features_df[selected_features]
        y = self.targets_df[f'lim {limit} stop {stop}'][self.target_name]

        # Initialize the model
        model = self.get_model()

        # Perform 5-fold cross-validation
        skf = StratifiedKFold(n_splits=5)
        predictions = cross_val_predict(model, X, y, cv=skf)

        # Calculate metrics
        mcc = matthews_corrcoef(y, predictions)
        f1_positive = f1_score(y, predictions, pos_label=1)
        f1_negative = f1_score(y, predictions, pos_label=0)
        cm = confusion_matrix(y, predictions)
        
        balanced = abs(f1_positive - f1_negative) <= 0.1 # Assume balanced if F1 scores are close

        result = None
        if balanced and (mcc > 0.6 or (f1_positive > 0.8 and f1_negative > 0.8)):
            result = {
                "Limit": limit,
                "Stop": stop,
                "TP": cm[1, 1],
                "TN": cm[0, 0],
                "FP": cm[0, 1],
                "FN": cm[1, 0],
                "MCC": mcc,
                "F1": (f1_positive + f1_negative) / 2
            }

        # Print live results
        print(f"Limit: {limit:<4} | Stop: {stop:<6} | MCC: {mcc:.2f} | F1 (positive): {f1_positive:.2f} | F1 (negative): {f1_negative:.2f} | Balanced: {balanced}")

        # Send the result through the queue
        q.put(result)

    def train_model(self):
        """Train the model for each limit/stop combination in parallel."""
        results = []
        processes = []
        q = Queue()

        for _, row in self.selected_features_df.iterrows():
            limit = row['Limit']
            stop = row['Stop']
            selected_features_str = row['Selected Features']
            p = Process(target=self.train_model_task, args=(limit, stop, selected_features_str, q))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Collect results from the queue
        while not q.empty():
            result = q.get()
            if result:
                results.append(result)

        # Save the results if any
        if results:
            # Sort results first by Stop then by Limit
            results.sort(key=lambda x: (x['Limit'], x['Stop']))
            self.save_results(results)

    def save_results(self, results):
        """Save the results to an Excel file."""
        output_path = os.path.join(self.output_dir, f'{self.target_name.replace(" ", "-").lower()}_{self.model_type.replace(" ", "-").lower()}.xlsx')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.to_excel(output_path, index=False)
        print(f"Results saved to {output_path}")

    def run(self, target_name, model_type="RandomForest"):
        """Run the full training process."""
        if target_name not in ['Return at cashout', 'Days at cashout']:
            binary_target = True
        self.model_type = model_type  # Set the model type
        self.load_data(target_name)
        self.train_model()

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run(target_name="Limit occurred first", model_type="RandomForest")
