import csv
from datetime import datetime
from typing import List, Tuple

def read_time_series_data(
    filename: str,               # Path to the CSV file
    date_column_index: int,      # Index of the column containing date values
    value_column_index: int      # Index of the column containing the values
) -> Tuple[List[datetime], List[float]]:
    """
    Reads time series data from a CSV file and returns dates and corresponding values.

    Args:
        filename (str): The path to the CSV file containing the time series data.
        date_column_index (int): The index of the column containing date values.
        value_column_index (int): The index of the column containing values.

    Returns:
        Tuple[List[datetime], List[float]]: A tuple containing:
            - A list of datetime objects for the dates.
            - A list of floats for the corresponding values.
    """
    # Initialize lists to store dates and values
    dates = []    # To store the dates as datetime objects
    values = []   # To store the corresponding values as floats

    # Open and read the CSV file
    with open(filename, "r") as f:
        csv_reader = csv.reader(f, delimiter=",")
        next(csv_reader)  # Skip the header row
        for line in csv_reader:
            # Parse the date and value from the specified columns
            dates.append(datetime.strptime(line[date_column_index], "%Y-%m-%d"))  # Convert to datetime
            values.append(float(line[value_column_index]))  # Convert to float

    return dates, values  # Return the lists of dates and values
