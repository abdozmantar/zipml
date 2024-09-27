import time
from typing import Tuple, Any

def measure_prediction_time(
    model: Any,                  # Trained model to make predictions
    samples: Any                 # Samples to make predictions on
) -> Tuple[float, float]:
    """
    Measures the time taken by a model to make predictions on the given samples.

    Args:
        model (Any): A trained model that will make predictions.
        samples (Any): Input data for which predictions will be made.

    Returns:
        Tuple[float, float]: A tuple containing:
            - total_time (float): Total time taken for predictions in seconds.
            - time_per_pred (float): Average time taken per prediction in seconds.
    """
    start_time = time.perf_counter()  # Get the start time
    model.predict(samples)              # Make predictions
    end_time = time.perf_counter()      # Get the finish time

    total_time = end_time - start_time  # Calculate total prediction time
    time_per_pred = total_time / len(samples)  # Calculate average time per prediction

    return total_time, time_per_pred  # Return total time and average time per prediction
