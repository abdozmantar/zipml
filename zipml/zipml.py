# zipml/zipml.py
import argparse
import logging
from .zipml.helpers import split_data, save_confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd

# Setting up logger
logger = logging.getLogger()

# Other functions and the main function remain the same

# Function to evaluate model performance
def evaluate_model(y_true, y_pred):
    logger.info("Evaluating model performance metrics.")
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }

# Function for hyperparameter optimization (same as before)
# Function to train and compare models (same as before)
# Function to visualize scores (same as before)

def main():
    """
    Main CLI function to train and compare models, and save the confusion matrix.
    """
    parser = argparse.ArgumentParser(description='ZipML: A simple AutoML for small datasets')
    parser.add_argument('file_path', type=str, help='Path to the dataset CSV file')
    parser.add_argument('target_column', type=str, help='Name of the target column in the dataset')
    args = parser.parse_args()
    
    data = pd.read_csv(args.file_path)
    X = data.drop(args.target_column, axis=1)
    y = data[args.target_column]
    
    best_model, results = train_models(X, y)
    
    # Test the best model and generate confusion matrix
    X_train, X_test, y_train, y_test = split_data(X, y)
    y_pred = best_model.predict(X_test)
    
    save_confusion_matrix(y_test, y_pred)
    plot_results(results)
    
    return best_model

if __name__ == '__main__':
    main()
