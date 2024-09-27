import argparse
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from typing import Tuple, Any, Dict, Union
import json
import pickle
from .model import calculate_model_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ZipML")

# Global model options
model_options = {
    'randomforest': RandomForestClassifier(),
    'svc': SVC(),
    'knn': KNeighborsClassifier(),
    'gradientboosting': GradientBoostingClassifier()
}

# Function for hyperparameter optimization
def optimize_hyperparameters(model: Any, X_train: Union[pd.DataFrame, Any], y_train: Any, param_grid: Dict[str, list]) -> Any:
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

# Function to train a single model
def train_model(model_name: str, X: Union[pd.DataFrame, Any], y: Any) -> Any:
    """
    Trains a specific model and returns the trained model.

    Parameters:
        model_name (str): The name of the model to be trained.
        X (DataFrame or array-like): Features.
        y (array-like): Target labels.

    Returns:
        The trained model.
    """
    logger.info(f"Training model: {model_name}.")
    model = model_options[model_name]
    model.fit(X, y)  # Fit the model with provided data
    return model

# Function to compare specific models
def compare_models(models: list, X_train: Union[pd.DataFrame, Any], X_test: Union[pd.DataFrame, Any], y_train: Any, y_test: Any) -> Tuple[Any, Dict[str, float]]:
    """
    Compares the performance of multiple machine learning models.
    
    Parameters:
        models (list): A list of machine learning models to compare.
        X_train (array-like): Feature set for training the models.
        X_test (array-like): Feature set for testing the models.
        y_train (array-like): Target values for training the models.
        y_test (array-like): Target values for testing the models.
    
    Returns:
        Tuple: The model with the highest accuracy score and a dictionary containing accuracy scores for each model.
    """
    performance = {}
    
    for model in models:
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate accuracy
        results_df = calculate_model_results(y_test, predictions)
        f1_score = results_df.loc[results_df["Metric"] == "F1 Score", "Value"].values[0]
        performance[model.__class__.__name__] = f1_score

    # Determine the best model based on accuracy
    best_model_name = max(performance, key=performance.get)
    best_model = next(model for model in models if model.__class__.__name__ == best_model_name)

    return best_model, performance

# Function to save the trained model
def save_model(model: Any, file_name: str = "model.pkl") -> None:
    """
    Save pre-trained model on the file system.
    
    Args:
        model (Any): The trained model to be saved.
        file_name (str): Target file name as exact path.
    """
    if not file_name.endswith('.pkl'):
        file_name += '.pkl'
    
    logger.info(f"Saving model to {file_name}.")
    with open(file_name, "wb") as f:
        pickle.dump(model, f)

# Function to load a pre-trained model
def load_model(file_name: str = "model.pkl") -> Any:
    """
    Load the saved pre-trained model from file system.
    
    Args:
        file_name (str): Exact path of the target saved model.
    
    Returns:
        Any: The loaded model.
    """
    logger.info(f"Loading model from {file_name}.")
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model

# Function to predict using a trained model
def predict(model: Any, X: Union[pd.DataFrame, Any]) -> Any:
    """
    Make predictions using the trained model.

    Parameters:
        model (Any): The trained model to make predictions with.
        X (DataFrame or array-like): Features to predict.

    Returns:
        array-like: Predicted labels for the provided features.
    """
    logger.info("Making predictions.")
    return model.predict(X)

# Function to display a welcome message with ASCII art
def display_welcome_message():
    """
        Displays an ASCII art welcome message for ZipML.
    """
    
    author = r"""
 ______     __     ______   __    __     __       
/\___  \   /\ \   /\  == \ /\ "-./  \   /\ \      
\/_/  /__  \ \ \  \ \  _-/ \ \ \-./\ \  \ \ \____ 
  /\_____\  \ \_\  \ \_\    \ \_\ \ \_\  \ \_____\
  \/_____/   \/_/   \/_/     \/_/  \/_/   \/_____/
    
                 Z I P M L
        Developed by Abdullah Ozmantar
    """
    
    colored_description = f"""
        \033[34mWelcome to ZipML!\033[0m
            
        \033[37mThis tool helps you quickly train, compare, and evaluate machine learning models.
        You can either train a single model or compare multiple models based on performance metrics.\033[0m
            
        \033[33mUsage examples:\033[0m

        \033[90m1. Train a specific model:\033[0m
        \033[32m   zipml --train train.csv --test test.csv --model randomforest --result results.json\033[0m       

        \033[90m2. Compare multiple models:\033[0m
        \033[32m   zipml --train train.csv --test test.csv --compare --compare_models randomforest svc knn --result results.json\033[0m

        \033[90m3. Load a pre-trained model:\033[0m
        \033[32m   zipml --load_model model.pkl --test test.csv --result results.json\033[0m
    """
    
    colored_author = f"\033[36m{author}\033[0m"
    
    print(colored_author)
    print(colored_description)

# Main function to run the CLI
def main() -> None:
    """
        Main CLI function to train and compare models, and save the confusion matrix.
    """
    
    display_welcome_message()

    parser = argparse.ArgumentParser(description='ZipML: A simple AutoML for small datasets')

    # CLI Arguments
    parser.add_argument('--train', type=str, help='Path to the training dataset CSV file (required unless using --load_model)')
    parser.add_argument('--test', type=str, help='Path to the testing dataset CSV file')
    parser.add_argument('--model', type=str, help='Model to be trained (optional if using compare_models)')
    parser.add_argument('--compare', action='store_true', help='Flag to compare multiple models')
    parser.add_argument('--compare_models', type=str, nargs='+', help='Models to be compared. E.g., "randomforest svc knn gradientboosting"')
    parser.add_argument('--result', type=str, help='Path to save the prediction results as a JSON file')
    parser.add_argument('--load_model', type=str, help='Path to load a pre-trained model')
    parser.add_argument('--save_model', type=str, help='Path to save the trained model')

    args = parser.parse_args()

    print("\nAvailable Models for Comparison or Training:")
    for model_name in model_options.keys():
        print(f"- {model_name}")
        
    # Load the data
    if args.train:
        logger.info(f"Loading training dataset from {args.train}.")
        train_data = pd.read_csv(args.train)

        print("\nAvailable Columns in Training Data:")
        print(train_data.head())

        # Ask the user to select the target column
        target_column = input("\nPlease select the target column for prediction: ")

        X_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column]

        test_data = pd.read_csv(args.test)
        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]

    elif args.test:
        logger.info(f"Loading testing dataset from {args.test}.")
        test_data = pd.read_csv(args.test)
        
        # Show columns and the first few rows of data
        print("\nAvailable Columns in Test Data:")
        print(test_data.head())

        target_column = input("\nPlease specify the target column for prediction: ")
        
        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]
    


    # Load a pre-trained model if specified
    if args.load_model:
        trained_model = load_model(args.load_model)
        logger.info(f"Loaded model: {args.load_model}.")
        
        # Make predictions and evaluate the model
        predictions = trained_model.predict(X_test)
        metrics = calculate_model_results(y_test, predictions)
        print("\nModel Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.2f}")

        # Save prediction results if specified
        if args.result:
            with open(args.result, 'w') as f:
                json.dump(predictions.tolist(), f)
            logger.info(f"Predictions saved to {args.result}.")

    # Model comparison
    elif args.compare:
        if args.compare_models:
            selected_models = [model_options[model_name] for model_name in args.compare_models if model_name in model_options]
            best_model, performance = compare_models(selected_models, X_train, X_test, y_train, y_test)

            print("\nModel Comparison Results:")
            for model_name, accuracy in performance.items():
                print(f"{model_name}: Accuracy = {accuracy:.2f}")
            print(f"\nBest Model: {best_model.__class__.__name__}")

            # Save results to output file if specified
            if args.result:
                with open(args.result, 'w') as f:
                    json.dump(performance, f)
                logger.info(f"Results saved to {args.result}.")
            
            # Save the best model if specified
            if args.save_model:
                if args.save_model.strip():
                    save_model(best_model, args.save_model)
                    logger.info(f"Best model saved as {args.save_model}.")
                else:
                    save_model(best_model)  # Default save without filename

    # Train a single model if specified
    elif args.model:
        if not args.train:
            print("Please specify --train to train the model.")
            return

        logger.info(f"Training specified model: {args.model}.")
        trained_model = train_model(args.model, X_train, y_train)

        # Make predictions and evaluate the model
        predictions = trained_model.predict(X_test)
        metrics = calculate_model_results(y_test, predictions)
        print("\nModel Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.2f}")

        # Save prediction results if specified
        if args.result:
            with open(args.result, 'w') as f:
                json.dump(predictions.tolist(), f)
            logger.info(f"Predictions saved to {args.result}.")

        # Save the trained model if specified
        if args.save_model:
            if args.save_model.strip():
                save_model(trained_model, args.save_model)
                logger.info(f"Trained model saved as {args.save_model}.")
            else:
                save_model(trained_model)  # Default save without filename
    else:
        print("Please specify either --model to train a model or --compare to compare models.")

if __name__ == "__main__":
    main()
