from typing import List
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_model_results(y_true: List[int], y_pred: List[int]) -> pd.DataFrame:
    """
    Calculates the accuracy, precision, recall, and F1 score of a binary classification model.

    Args:
        y_true (List[int]): True labels in the form of a 1D array.
        y_pred (List[int]): Predicted labels in the form of a 1D array.

    Returns:
        pd.DataFrame: A DataFrame containing accuracy, precision, recall, and F1 score.
    """
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100

    # Calculate model precision, recall, and F1 score using "weighted" average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    # Create a DataFrame to store the results
    model_results_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [model_accuracy, model_precision, model_recall, model_f1]
    })

    return model_results_df  # Return the DataFrame containing the calculated metrics