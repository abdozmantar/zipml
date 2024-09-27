from datetime import datetime
import matplotlib.pyplot as plt
from typing import List, Optional

def plot_time_series(
    timesteps: List[datetime],        # List of time values (datetime objects)
    values: List[float],               # List of corresponding values (e.g., BTC prices)
    format: str = ".",                 # Style of the plot (default is a dot)
    start: int = 0,                    # Index to start plotting from
    end: Optional[int] = None,         # Index to end plotting at (None means till the end)
    label: Optional[str] = None        # Label for the plot legend (default is None)
) -> None:
    """
    Plots timesteps against values to visualize time series data.

    Args:
        timesteps (List[datetime]): A list of datetime objects representing the time points.
        values (List[float]): A list of corresponding float values (e.g., prices).
        format (str): The style of the plot (default is ".").
        start (int): The starting index for the plot (default is 0).
        end (Optional[int]): The ending index for the plot (default is None, meaning to the end).
        label (Optional[str]): An optional label for the plot legend (default is None).

    Returns:
        None: The function will display the plot.
    """
    # Plot the series with the specified formatting
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    
    if label:
        plt.legend(fontsize=14)  # Increase legend font size if a label is provided
    plt.grid(True)  # Add a grid for better readability
    plt.title("Time Series Data")  # Optional title for the plot
    plt.show()  # Display the plot
