# zipml/zipml.py
import argparse
import logging
from .helpers import plot_results, split_data, save_confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd

# Setting up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Other functions and the main function remain the same

# Function to evaluate model performance
def evaluate_model(y_true, y_pred):
    """
    Evaluates performance metrics for the model's predictions.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    dict: A dictionary containing accuracy, precision, recall, and F1 score.
    """
    logger.info("Evaluating model performance metrics.")
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted')
    }

# Function for hyperparameter optimization
def optimize_hyperparameters(model, X_train, y_train, param_grid):
    """
    Performs hyperparameter optimization using GridSearchCV.

    Parameters:
    model: The machine learning model to be optimized.
    X_train: Training data features.
    y_train: Training data labels.
    param_grid (dict): Hyperparameters to be optimized.

    Returns:
    The best estimator after optimization.
    """
    logger.info(f"Optimizing hyperparameters for {model.__class__.__name__}.")
    grid_search = GridSearchCV(model, param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Function to train and compare models
def train_models(X, y):
    """
    Trains multiple models, compares their performance, and returns the best model.

    Parameters:
    X (DataFrame or array-like): Features.
    y (array-like): Target labels.

    Returns:
    best_model: The best performing model.
    results: Performance metrics of all models.
    """
    logger.info("Starting model training and comparison.")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    models = {
        'RandomForest': RandomForestClassifier(),
        'SVC': SVC(),
        'KNeighbors': KNeighborsClassifier(),
        'GradientBoosting': GradientBoostingClassifier()
    }
    
    results = {}

    for model_name, model in models.items():
        logger.info(f"Training {model_name} model.")
        if model_name == 'RandomForest':
            param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20]}
        elif model_name == 'SVC':
            param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        elif model_name == 'KNeighbors':
            param_grid = {'n_neighbors': [3, 5, 10]}
        elif model_name == 'GradientBoosting':
            param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
        
        best_model = optimize_hyperparameters(model, X_train, y_train, param_grid)
        predictions = best_model.predict(X_test)
        metrics = evaluate_model(y_test, predictions)
        results[model_name] = metrics

    for model_name, metrics in results.items():
        logger.info(f"\nModel: {model_name}")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")

    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    best_metrics = results[best_model_name]
    
    logger.info(f"\nBest model: {best_model_name} with F1 Score: {best_metrics['f1_score']:.4f}")
    
    return models[best_model_name], results

# Function to compare specific models
def compare_models(models, X_train, X_test, y_train, y_test):
    """
    Compares the performance of multiple machine learning models.
    
    Parameters:
    models (list): A list of machine learning models to compare.
    X_train (array-like): Feature set for training the models.
    X_test (array-like): Feature set for testing the models.
    y_train (array-like): Target values for training the models.
    y_test (array-like): Target values for testing the models.
    
    Returns:
    best_model: The model with the highest accuracy score.
    performance (dict): A dictionary containing accuracy scores for each model.
    """
    performance = {}
    
    for model in models:
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        performance[model.__class__.__name__] = accuracy

    # Determine the best model based on accuracy
    best_model_name = max(performance, key=performance.get)
    best_model = next(model for model in models if model.__class__.__name__ == best_model_name)

    return best_model, performance

# Main function to run the CLI
def main():
    """
    Main CLI function to train and compare models, and save the confusion matrix.
    """
    parser = argparse.ArgumentParser(description='ZipML: A simple AutoML for small datasets')
    parser.add_argument('file_path', type=str, help='Path to the dataset CSV file')
    parser.add_argument('target_column', type=str, help='Name of the target column in the dataset')
    args = parser.parse_args()
    
    logger.info(f"Loading dataset from {args.file_path}.")
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
