import numpy as np
import matplotlib.pyplot as plt
import itertools
import logging
from sklearn.metrics import confusion_matrix
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_and_plot_confusion_matrix(
    y_true: np.ndarray,                    # Array of true labels
    y_pred: np.ndarray,                    # Array of predicted labels
    classes: list = None,                  # List of class names for labeling the matrix
    figsize: tuple = (10, 10),             # Size of the plot
    text_size: int = 15,                   # Font size for the text labels
    save_path: Optional[str] = None        # Optional path to save the plot
) -> None:
    """
    Plots a confusion matrix with actual and predicted labels.

    Args:
        y_true (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted labels.
        classes (list, optional): List of class names for labeling the matrix. Defaults to None.
        figsize (tuple, optional): Size of the plot. Defaults to (10, 10).
        text_size (int, optional): Font size for the text labels in the plot. Defaults to 15.
        save_path (Optional[str]): Path to save the confusion matrix plot as an image. Defaults to None.

    Returns:
        None
    """
    # Create the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # Normalize the confusion matrix
    num_classes = cm.shape[0]

    # Create a plot for the confusion matrix
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Set labels and axes
    if classes:
        labels = classes
    else:
        labels = np.arange(num_classes)

    ax.set(title="Confusion Matrix",
           xlabel="Predicted Label",
           ylabel="True Label",
           xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=labels,
           yticklabels=labels)

    # Move x-axis labels to the bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Adjust label size
    ax.xaxis.label.set_size(text_size)
    ax.yaxis.label.set_size(text_size)
    ax.title.set_size(text_size)

    # Plot text on each cell
    threshold = (cm.max() + cm.min()) / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 size=text_size)

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  # Save the figure
        logging.info(f"Confusion matrix saved as '{save_path}'.")  # Log the save message
    
    plt.show()  # Display the plot
