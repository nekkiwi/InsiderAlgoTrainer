import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold, cross_validate
from tqdm import tqdm
import joblib  # To save models

def load_data(features_file, targets_file, selected_features_file, target_name):
    """Load the data from Excel files."""
    target_name_clean = target_name.replace(" ", "-").lower()
    features_df = pd.read_excel(features_file)
    targets_df = pd.read_excel(targets_file, sheet_name=None)
    selected_features_df = pd.read_excel(selected_features_file, sheet_name=target_name_clean)
    
    return features_df, targets_df, selected_features_df


def train_model_task_wrapper(args):
    """Wrapper function to unpack arguments and call train_model_task."""
    row, features_df, targets_df, target_name, model, model_save_dir = args
    return train_model_task(row, features_df, targets_df, target_name, model, model_save_dir)

def get_model(model_type, is_continuous):
    """Return the model instance based on the specified model type and target type."""
    if "RandomForest" in model_type:
        if is_continuous:
            return RandomForestRegressor(random_state=42)
        else:
            if model_type == "RandomForestUnderSample":
                return Pipeline([('resample', RandomUnderSampler(random_state=42)),
                                ('classifier', RandomForestClassifier(random_state=42))])
            elif model_type == "RandomForestOverSample":
                return Pipeline([('resample', SMOTE(random_state=42)),
                                ('classifier', RandomForestClassifier(random_state=42))])
    elif "NeuralNet" in model_type:
        if is_continuous:
            return MLPRegressor(hidden_layer_sizes=(128, 64, 32),  # Three hidden layers with 128, 64, and 32 neurons
                                alpha=0.001,                        # L2 regularization
                                max_iter=1000,                      # Maximum number of iterations
                                early_stopping=True,                # Use early stopping to prevent overfitting
                                random_state=42) 
        else:
            if model_type == "NeuralNetUnderSample":
                return Pipeline([('resample', RandomUnderSampler(random_state=42)),
                                ('classifier', MLPClassifier(hidden_layer_sizes=(128, 64, 32),  # Three hidden layers with 128, 64, and 32 neurons
                                                            alpha=0.001,                        # L2 regularization
                                                            max_iter=1000,                      # Maximum number of iterations
                                                            early_stopping=True,                # Use early stopping to prevent overfitting
                                                            random_state=42))])
            if model_type == "NeuralNetOverSample":
                return Pipeline([('resample', SMOTE(random_state=42)),
                                ('classifier', MLPClassifier(hidden_layer_sizes=(128, 64, 32),  # Three hidden layers with 128, 64, and 32 neurons
                                                            alpha=0.001,                        # L2 regularization
                                                            max_iter=1000,                      # Maximum number of iterations
                                                            early_stopping=True,                # Use early stopping to prevent overfitting
                                                            random_state=42))])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_model_task(row, features_df, targets_df, target_name, model, model_save_dir):
    """Train the model for a specific limit/stop combination and return the results."""
    limit, stop, selected_features_str = row['Limit'], row['Stop'], row['Selected Features']
    selected_features = eval(selected_features_str)
    selected_features = [feature.strip("'") for feature in selected_features]

    # Select the features and target
    X = features_df[selected_features]
    target_column = f'Limit {limit}, Stop {stop}'

    if target_column not in targets_df[target_name].columns:
        target_column = target_name
        if target_column not in targets_df[target_name].columns:
            return None, None

    y = targets_df[target_name][target_column]

    # Ensure X and y have consistent lengths by aligning their indices
    common_index = X.index.intersection(y.dropna().index)  # Drop rows with NaN values in y
    X_aligned = X.loc[common_index].dropna()  # Drop rows with NaN values in X
    y_aligned = y.loc[common_index]

    # Check if we are in a "sell" target and adjust model saving accordingly
    if "sell" in target_name or "limstop" in target_name:
        limit_stop_dir = os.path.join(model_save_dir, f'lim_{limit}_stop_{stop}')
        os.makedirs(limit_stop_dir, exist_ok=True)
    else:
        limit_stop_dir = model_save_dir

    # Use StratifiedKFold for classification and KFold for regression
    if "final" in target_name:
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(model, X_aligned, y_aligned, cv=kfold, return_estimator=True)
    else:  # Classification task (binary or multiclass)
        skf = StratifiedKFold(n_splits=5)
        cv_results = cross_validate(model, X_aligned, y_aligned, cv=skf, return_estimator=True)

    # Save each model from the 5-fold cross-validation
    for i, estimator in enumerate(cv_results['estimator'], start=1):
        model_path = os.path.join(limit_stop_dir, f'fold_{i}.joblib')
        save_model(estimator, model_path)

    # Handle the predictions as before
    predictions = cross_val_predict(model, X_aligned, y_aligned, cv=kfold if 'final' in target_name else skf)

    # Determine if the target is continuous (final targets)
    if "final" in target_name:  # Continuous target
        mse = mean_squared_error(y_aligned, predictions)
        y_sign = np.sign(y_aligned) > 0
        pred_sign = np.sign(predictions) > 0

        mcc = matthews_corrcoef(y_sign, pred_sign)
        f1_positive = f1_score(y_sign, pred_sign, pos_label=1)
        f1_negative = f1_score(y_sign, pred_sign, pos_label=0)
        acc = accuracy_score(y_sign, pred_sign)
        cm = confusion_matrix(y_sign, pred_sign)

        result = {
            "Limit": limit,
            "Stop": stop,
            "Accuracy": acc,
            "MSE": mse,
            "MCC": mcc,
            "PPV": cm[1, 1]/(cm[1, 1]+cm[0, 1]),
            "NPV": cm[0, 0]/(cm[0, 0]+cm[1, 0]),
            "F1_pos": f1_positive,
            "F1_neg": f1_negative,
            "TN": cm[0, 0],
            "FP": cm[0, 1],
            "FN": cm[1, 0],
            "TP": cm[1, 1]
        }

    else:  # Binary target
        mcc = matthews_corrcoef(y_aligned, predictions)
        f1_positive = f1_score(y_aligned, predictions, pos_label=1)
        f1_negative = f1_score(y_aligned, predictions, pos_label=0)
        acc = accuracy_score(y_aligned, predictions)
        cm = confusion_matrix(y_aligned, predictions)

        result = {
            "Limit": limit,
            "Stop": stop,
            "Accuracy": acc,
            "MCC": mcc,
            "PPV": cm[1, 1]/(cm[1, 1]+cm[0, 1]),
            "NPV": cm[0, 0]/(cm[0, 0]+cm[1, 0]),
            "F1_pos": f1_positive,
            "F1_neg": f1_negative,
            "TN": cm[0, 0],
            "FP": cm[0, 1],
            "FN": cm[1, 0],
            "TP": cm[1, 1],
        }

    # Construct the prediction data with limit/stop in the column names
    pred_column_name = f'Pred_{target_name}_l{limit}_s{stop}'
    gt_column_name = f'GT_{target_name}_l{limit}_s{stop}'

    prediction_data = pd.DataFrame({
        pred_column_name: predictions,
        gt_column_name: y_aligned
    })

    return result, {"Predictions": prediction_data, "Limit": limit, "Stop": stop}

def save_training_data(results, all_predictions, output_dir, predictions_dir, model_type, target_name, features_df):
    """Save all models, results, and predictions in one call."""
    # Save the results
    save_results(results, output_dir, model_type, target_name)

    # Combine predictions and save them to a single Excel file
    save_predictions(all_predictions, predictions_dir, model_type, target_name, features_df)

def save_results(results, output_dir, model_type, target_name):
    """Save the results to an Excel file, creating a new subsheet if necessary."""
    stats_file = os.path.join(output_dir, f'stats_{model_type.replace(" ", "-").lower()}.xlsx')
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)

    results_df = pd.DataFrame(results)

    with pd.ExcelWriter(stats_file, engine='openpyxl', mode='a' if os.path.exists(stats_file) else 'w') as writer:
        if target_name in writer.book.sheetnames:
            del writer.book[target_name]
        results_df.to_excel(writer, sheet_name=target_name, index=False)
        print(f"- Training statistics saved to {stats_file}")

def save_predictions(all_predictions, predictions_dir, model_short, target_name, features_df):
    """Save predictions to a single Excel sheet, combining all limit/stop results."""
    model_subdir = os.path.join(predictions_dir, model_short.lower())
    os.makedirs(model_subdir, exist_ok=True)

    combined_predictions = None
    ticker_date_columns = ["Ticker", "Filing Date"]

    for pred_data in all_predictions:
        df = pred_data["Predictions"]

        # Concatenate with existing predictions
        if combined_predictions is None:
            # Add Ticker and Filing Date once at the beginning
            combined_predictions = pd.concat([features_df[ticker_date_columns], df], axis=1)
        else:
            combined_predictions = pd.concat([combined_predictions, df], axis=1)

    prediction_file = os.path.join(model_subdir, f'pred_{model_short.lower()}_{target_name}.xlsx')

    # Save all the combined predictions into a single file
    with pd.ExcelWriter(prediction_file, engine='openpyxl') as writer:
        combined_predictions.to_excel(writer, sheet_name=target_name, index=False)

    print(f"- Predictions saved to {prediction_file}")


def save_model(model, model_path):
    """Save the model to a specific file path."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"- Model saved to {model_path}")

