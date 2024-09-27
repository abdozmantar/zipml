import random
import matplotlib.pyplot as plt
import numpy as np  # NumPy kütüphanesini ekleyelim
from typing import List, Any

def plot_random_image(
    model: Any,  # TensorFlow model used for making predictions on the images
    images: List[np.ndarray],  # A list of images where each image is represented as a NumPy array
    true_labels: List[int],  # A list containing the true label indices for each image in 'images'
    classes: List[str]  # A list of class names corresponding to each label index
) -> None:
    """
    Picks a random image, plots it, and labels it with a prediction and truth label.

    Parameters:
        model (Any): The trained model used for making predictions.
        images (List[np.ndarray]): A list of images to choose from, where each image is represented as a NumPy array.
        true_labels (List[int]): A list of true labels corresponding to the images, represented as integer indices.
        classes (List[str]): A list of class names corresponding to label indices.
    """
    
    # Generate a random index to select an image from the provided list
    random_index = random.randint(0, len(images) - 1)

    # Retrieve the randomly selected image and its corresponding true label
    target_image = images[random_index]
    true_label_index = true_labels[random_index]
    true_label = classes[true_label_index]

    # Use the model to predict the class probabilities for the target image after reshaping it for the model's input
    pred_probs = model.predict(target_image.reshape(1, 28, 28))
    pred_label_index = pred_probs.argmax()  # Determine the index of the predicted class with the highest probability
    pred_label = classes[pred_label_index]  # Retrieve the predicted class name from the classes list

    # Display the target image using a binary colormap for better contrast
    plt.imshow(target_image, cmap=plt.cm.binary)

    # Determine the color for the label based on whether the prediction matches the true label
    color = "green" if pred_label == true_label else "red"

    # Use NumPy to find the maximum prediction probability
    max_prob = np.max(pred_probs)  # Get the maximum probability using NumPy

    # Set the x-axis label to show the predicted label, prediction confidence, and true label,
    # with the label color indicating the correctness of the prediction
    plt.xlabel(
        "Pred: {} {:2.0f}% (True: {})".format(
            pred_label, 
            100 * max_prob,  # Convert prediction probabilities to percentage
            true_label
        ), 
        color=color  # Change label color based on prediction accuracy
    )
    
    # Display the plotted image
    plt.show()
