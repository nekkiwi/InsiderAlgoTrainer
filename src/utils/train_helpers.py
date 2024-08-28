import os
import pandas as pd

def save_predictions_to_temp_file(features_df, predictions_dir, limit, stop, target_name, y, predictions):
    """Save the predictions to a temporary CSV file."""
    temp_file = os.path.join(predictions_dir, f'temp_predictions_{limit}_{stop}.csv')
    os.makedirs(predictions_dir, exist_ok=True)

    # Prepare the DataFrame
    df = pd.DataFrame({
        "Ticker": features_df['Ticker'],
        "Filing Date": features_df['Filing Date'],
        f'GT_{target_name}': y,
        f'Pred_{target_name}': predictions
    })

    df.to_csv(temp_file, index=False)
    # print(f"Temporary predictions saved to {temp_file}")

def combine_predictions(predictions_dir, model_type):
    """Combine temporary prediction files into a single Excel file."""
    prediction_file = os.path.join(predictions_dir, f'predictions_{model_type.replace(" ", "-").lower()}.xlsx')

    # Initialize a Pandas Excel writer
    with pd.ExcelWriter(prediction_file, engine='openpyxl') as writer:
        for temp_file in os.listdir(predictions_dir):
            if temp_file.startswith('temp_predictions_') and temp_file.endswith('.csv'):
                df = pd.read_csv(os.path.join(predictions_dir, temp_file))
                sheet_name = temp_file.replace('temp_predictions_', '').replace('.csv', '')
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    # print(f"Predictions combined and saved to {prediction_file}")

    # Clean up temporary files
    for temp_file in os.listdir(predictions_dir):
        if temp_file.startswith('temp_predictions_') and temp_file.endswith('.csv'):
            os.remove(os.path.join(predictions_dir, temp_file))
