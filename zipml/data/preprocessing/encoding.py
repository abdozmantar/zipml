import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def one_hot_encode_labels(df: pd.DataFrame, column_name: str) -> np.ndarray:
    """
    One-hot encodes the target labels from the specified column in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the target column.
        column_name (str): Name of the column to be one-hot encoded.

    Returns:
        np.ndarray: One-hot encoded labels.
    """
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    labels = df[column_name].to_numpy().reshape(-1, 1)
    return one_hot_encoder.fit_transform(labels)

def label_encode_labels(df: pd.DataFrame, column_name: str) -> tuple:
    """
    Label encodes the target labels and returns both the encoded labels and the class names.

    Args:
        df (pd.DataFrame): DataFrame containing the target column.
        column_name (str): Name of the column to be label encoded.

    Returns:
        tuple: (Encoded labels, Class names).
    """
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(df[column_name].to_numpy())
    class_names = label_encoder.classes_
    return labels_encoded, class_names