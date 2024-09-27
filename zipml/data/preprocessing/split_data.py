
import logging
import pandas as pd
from typing import Any, Tuple, Union
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    logging.info(f"Splitting data with test size of {test_size}.")
    return train_test_split(X, y, test_size=test_size, random_state=42)