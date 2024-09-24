# zipml/helpers.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import logging

logger = logging.getLogger()

def split_data(X, y, test_size=0.2):
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

def save_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
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

def load_data(file_path):
    """
    Loads a dataset from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    DataFrame: Loaded data as a pandas DataFrame.
    """
    logger.info(f"Loading dataset from {file_path}.")
    return pd.read_csv(file_path)

def get_class_distribution(y):
    """
    Calculates the distribution of target classes in the dataset.
    
    Parameters:
    y (array-like): Target labels.
    
    Returns:
    Series: The class distribution as a pandas Series.
    """
    logger.info("Calculating class distribution.")
    return pd.Series(y).value_counts(normalize=True)

def plot_class_distribution(y):
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
