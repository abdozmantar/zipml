import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

def combine_and_plot_model_results(
    model_results: Dict[str, List[float]]
) -> None:
    """
    Combines model results into a DataFrame, normalizes accuracy, and plots the results.

    Args:
        model_results (Dict[str, List[float]]): A dictionary where keys are model names 
                                                  and values are lists of model metrics.
    """
    # Combine model results into a DataFrame
    all_model_results = pd.DataFrame(model_results)
    all_model_results = all_model_results.transpose()

    # Normalize the accuracy to the same scale as other metrics
    if 'accuracy' in all_model_results.columns:
        all_model_results['accuracy'] = all_model_results['accuracy'] / 100

    # Plot and compare all model results
    all_model_results.plot(kind="bar", figsize=(10, 7), legend=True)
    plt.title("Comparison of Model Results")
    plt.ylabel("Metric Value")
    plt.xlabel("Models")
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.show()  # Show the plot

    # Sort models results by f1-score and plot
    if 'f1' in all_model_results.columns:
        all_model_results.sort_values("f1", ascending=True)["f1"].plot(kind="bar", figsize=(10, 7))
        plt.title("F1 Score of Models")
        plt.ylabel("F1 Score")
        plt.xlabel("Models")
        plt.show()  # Show the plot