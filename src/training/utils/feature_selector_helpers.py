import pandas as pd
from sklearn.feature_selection import f_classif, chi2, f_regression
import os
import seaborn as sns
from matplotlib import pyplot as plt
from src.scraper.utils.feature_preprocess_helpers import identify_feature_types


def select_features(X, y, p_threshold=0.1):
    """Select features based on the type of the target and feature types."""
    categorical_features, _ = identify_feature_types(X)

    scores = pd.Series(index=X.columns, dtype=float)
    p_values = pd.Series(index=X.columns, dtype=float)
    
    for feature in X.columns:
        if feature in categorical_features:
            if y.nunique() <= 2:  # Binary target (categorical)
                score, p_value = chi2(X[[feature]], y)
            else:  # Continuous target
                score, p_value = f_regression(X[[feature]], y)
        else:
            if y.nunique() <= 2:  # Binary target (categorical)
                score, p_value = f_classif(X[[feature]], y)
            else:  # Continuous target
                score, p_value = f_regression(X[[feature]], y)
        
        scores[feature] = score[0]  # Extract the first element since score is a 1D array
        p_values[feature] = p_value[0]  # Extract the first element since p_value is a 1D array

    # Filter features based on p-value threshold
    selected_features = p_values[p_values < p_threshold].index.tolist()

    return selected_features, scores, p_values


def save_selected_features(selected_features_dict, output_dir):
    """Save selected features from all sheets into a single Excel file with multiple sheets."""
    import pandas as pd  # Ensure pandas is imported
    from openpyxl import Workbook  # Ensure openpyxl is available

    excel_path = os.path.join(output_dir, "selected_features.xlsx")
    
    if not selected_features_dict:
        print("- No features were selected. Skipping save.")
        return
    
    # Initialize Excel writer
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for sheet_name, selections in selected_features_dict.items():
            if not selections:
                print(f"- No selections for sheet {sheet_name}. Skipping.")
                continue
            
            # Convert list of dicts to DataFrame
            df = pd.DataFrame(selections)
            
            # Check if 'Limit' and 'Stop' columns exist
            if 'Limit' in df.columns and 'Stop' in df.columns:
                
                # Reorder columns to have 'Limit' and 'Stop' first
                columns_order = ['Limit', 'Stop'] + [col for col in df.columns if col not in ['Limit', 'Stop']]
                df = df[columns_order]
            else:
                print(f"- 'Limit' and 'Stop' columns not found in sheet {sheet_name}. Skipping.")
                continue
            
            # Replace spaces and convert to lowercase for sheet naming
            safe_sheet_name = sheet_name.replace(" ", "-").lower()
            
            # Write to the respective sheet
            try:
                df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
            except Exception as e:
                print(f"- Failed to write sheet '{safe_sheet_name}': {e}")
    
    print(f"- Selected features saved to {excel_path}")




def create_heatmap(selected_features_dict, key, output_dir, p_threshold):
    """Create and save a heatmap of either scores or p-values."""
    data = []

    for sheet_name, selections in selected_features_dict.items():
        for selection in selections:
            values = selection[key]
            if isinstance(values, float):
                values = [values]

            series = pd.Series(values, index=selection['Selected Features'], name=f'{sheet_name}_{selection["Limit"]}_{selection["Stop"]}')
            data.append(series)

    if data:
        df = pd.concat(data, axis=1).T

        # Calculate figure height dynamically based on the number of rows (lim/stop combinations)
        n_combinations = len(df)
        cell_height = 0.5  # Height per row
        height = n_combinations * cell_height

        plt.figure(figsize=(60, height))
        if key == 'p_values':
            sns.heatmap(df, annot=True, fmt=".3f", cmap="coolwarm", cbar=True, linewidths=0.5, vmin=0, vmax=p_threshold)
        plt.xticks(rotation=90, ha='center')
        plt.yticks(rotation=0)
        heatmap_path = os.path.join(output_dir, f"{key}_heatmap.png")
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        print(f"- Heatmap saved to {heatmap_path}")
    else:
        print(f"- No valid data found for {key}. Skipping heatmap creation.")


def process_sheet_data(targets_df, X_encoded):
    """Process the targets dataframe to extract tasks for feature selection."""
    tasks = []

    # Ensure 'Ticker' and 'Filing Date' columns are available for merging
    if 'Ticker' not in X_encoded.columns or 'Filing Date' not in X_encoded.columns:
        raise KeyError("'Ticker' and 'Filing Date' must be present in features DataFrame to align with targets")

    for sheet_name, sheet_data in targets_df.items():
        # Ensure 'Filing Date' columns in both DataFrames are in datetime format
        X_encoded['Filing Date'] = pd.to_datetime(X_encoded['Filing Date'], format='%d/%m/%Y %H:%M', errors='coerce')
        sheet_data['Filing Date'] = pd.to_datetime(sheet_data['Filing Date'], format='%d/%m/%Y %H:%M', errors='coerce')

        # Remove any rows where the Filing Date conversion failed (NaT values)
        X_encoded.dropna(subset=['Filing Date'], inplace=True)
        sheet_data.dropna(subset=['Filing Date'], inplace=True)

        # Merge on Ticker and Filing Date to ensure consistency
        merged_data = pd.merge(X_encoded, sheet_data, on=['Ticker', 'Filing Date'], how='inner')

        if merged_data.empty:
            continue  # Skip if there are no rows after merging

        # Drop 'Ticker' and 'Filing Date' from features after merging
        X_encoded_aligned = merged_data[X_encoded.columns.drop(['Ticker', 'Filing Date'])]

        # Process each target column
        for target_name in sheet_data.columns[2:]:  # Starting from the 3rd column (skip 'Ticker' and 'Filing Date')
            try:
                # Example column name: 'return_limit_sell (Limit 0.02 / Stop -0.16)'
                limit_stop_info = target_name.split('(')[1].split(')')[0]  # Extract the 'Limit 0.02 / Stop -0.16' part
                limit_value, stop_value = limit_stop_info.replace("Limit ", "").replace("Stop ", "").split(" / ")
            except (IndexError, ValueError):
                # If parsing fails, skip this target (e.g., static targets like 'pos_return', 'high_return')
                limit_value, stop_value = None, None  # No dynamic target

            y = merged_data[target_name]
            non_empty_count = y.dropna().shape[0]
            if non_empty_count > 0:  # Ensure the target column has valid data
                tasks.append((limit_value, stop_value, target_name, y, X_encoded_aligned))

    return tasks
