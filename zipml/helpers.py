# zipml/helpers.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import logging
from typing import Tuple, Any, Union

logger = logging.getLogger("Helpers")

def split_data(X: Union[pd.DataFrame, Any], y: Any, test_size: float = 0.2) -> Tuple[Union[pd.DataFrame, Any], Union[pd.DataFrame, Any], Any, Any]:
    """
    Splits data into training and testing sets.
    
    Parameters:
    X (DataFrame or array-like): Features.
    y (array-like): Target labels.
    test_size (float): Proportion of the dataset to include in the test split.
    
    Returns:
    tuple: Split datasets (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Splitting data with test size of {test_size}.")
    return train_test_split(X, y, test_size=test_size, random_state=42)

def save_confusion_matrix(y_true: Any, y_pred: Any, filename: str = "confusion_matrix.png") -> None:
    """
    Generates and saves a confusion matrix as a PNG file.
    
    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    filename (str): The filename where the confusion matrix will be saved.
    """
    logger.info("Generating confusion matrix.")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)
    logger.info(f"Confusion matrix saved as {filename}.")

# Additional Helper Functions

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads a dataset from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    DataFrame: Loaded data as a pandas DataFrame.
    """
    logger.info(f"Loading dataset from {file_path}.")
    return pd.read_csv(file_path)

def get_class_distribution(y: Any) -> pd.Series:
    """
    Calculates the distribution of target classes in the dataset.
    
    Parameters:
    y (array-like): Target labels.
    
    Returns:
    Series: The class distribution as a pandas Series.
    """
    logger.info("Calculating class distribution.")
    return pd.Series(y).value_counts(normalize=True)

def plot_class_distribution(y: Any) -> None:
    """
    Plots the distribution of target classes in the dataset.
    
    Parameters:
    y (array-like): Target labels.
    """
    logger.info("Plotting class distribution.")
    distribution = get_class_distribution(y)
    distribution.plot(kind='bar', figsize=(6, 4))
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Proportion')
    plt.show()

# Function to visualize scores
def plot_results(results: dict) -> None:
    """
    Plots the model comparison results as a bar chart.

    Parameters:
    results (dict): Dictionary containing the performance metrics of the models.
    """
    logger.info("Plotting model comparison results.")
    df = pd.DataFrame(results).T
    df.plot(kind='bar', figsize=(10, 6))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.show()
