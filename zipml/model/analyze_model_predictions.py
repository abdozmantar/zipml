import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Tuple, Any


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_model_predictions(
    model,                                # Loaded model
    val_data: List[Any],                  # Validation data (general type)
    val_labels: List[int],                # True labels
    identify_column: Optional[str] = None # Optional identify column name
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyzes model predictions by creating a DataFrame of validation data, 
    true labels, predicted labels, and prediction probabilities.
    It identifies and logs false positives and false negatives.

    Args:
        model: A trained model to make predictions.
        val_data (List[Any]): Validation data for model predictions.
        val_labels (List[int]): True labels corresponding to validation data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - DataFrame with all predictions and probabilities
            - DataFrame with the most incorrect predictions
    """
    # Make predictions with the loaded model
    model_pred_probs = model.predict(val_data)  # Get prediction probabilities from the model
    model_preds = np.round(model_pred_probs)  # Convert probabilities to label format

      # Determine which column to use for identification
    if identify_column is not None and identify_column in val_data.columns:
        identify_data = val_data[identify_column].values.reshape(-1, 1)
    else:
        identify_data = val_data.iloc[:, 0].values.reshape(-1, 1)  # Use the first column by default

    # Create DataFrame with validation data, validation labels, and predictions
    val_df = pd.DataFrame({
        "data": identify_data.flatten(),                  # Validation data (could be any type)
        "target": val_labels,              # True labels
        "pred": model_preds,               # Model predictions
        "pred_prob": model_pred_probs      # Prediction probabilities
    })

    # Identify the incorrect predictions and sort by prediction probabilities
    most_wrong = val_df[val_df["target"] != val_df["pred"]].sort_values("pred_prob", ascending=False)

    # Log the top 10 false positives
    logging.info("Top 10 False Positives (Model predicted 1 when it should have predicted 0):")
    logging.info(most_wrong[most_wrong['pred'] == 1].head(10).to_string(index=False))

    # Log the top 10 false negatives
    logging.info("Top 10 False Negatives (Model predicted 0 when it should have predicted 1):")
    logging.info(most_wrong[most_wrong['pred'] == 0].tail(10).to_string(index=False))

    # Check the false positives
    logging.info("Detailed False Positives:")
    for row in most_wrong[most_wrong['pred'] == 1].head(10).itertuples(index=False):
        logging.info(f"Target: {row.target}, Pred: {row.pred}, Prob: {row.pred_prob:.4f}")
        logging.info(f"Data:\n {row.data}\n")
        logging.info("----\n")

    # Check the false negatives
    logging.info("Detailed False Negatives:")
    for row in most_wrong[most_wrong['pred'] == 0].tail(10).itertuples(index=False):
        logging.info(f"Target: {row.target}, Pred: {row.pred}, Prob: {row.pred_prob:.4f}")
        logging.info(f"Data:\n {row.data}\n")
        logging.info("----\n")

    return val_df, most_wrong  # Return the DataFrames
